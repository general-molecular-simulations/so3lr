"""SO3LR energy and forces for millions of atoms on one GPU (Pallas-Triton kernels).

    from so3lr.flash.loader import load_params
    from so3lr.flash.engine import FlashSO3LR
    params, statics = load_params("so3lr-tauri-m", precision="float32", lr_cutoff=12.0)
    eng = FlashSO3LR(params, statics, z, box)          # box: orthorhombic, [3, 3]
    energy, forces = eng.energy_forces(positions)      # jitted

A cell grid and an int32 short-range neighbour table built on the GPU, attention
kernels that take positions and return forces, short-range pair and streamed
long-range kernels, and the per-atom parts (projections, MLPs, heads) in atom tiles
with ``jax.vjp``. Nothing is stored per edge or per pair, so memory grows by a few KiB
per atom; the forward and backward are scheduled by hand, layer by layer.

Three of these ideas come from Symmetrix-XL (Fu et al., arXiv:2610.01036): tabulated
radial functions (each layer's radial filters are cubic Hermite tables of r), streaming
the edges (each kernel program walks its atom's neighbours in chunks, so no edge array
exists), and tiled execution (the per-atom work runs over fixed-size atom tiles).

* Per-atom sums (messages, projection gradients, forces) are loop carries written once
  per atom, so the kernels need no read-modify-write of their own rows.
* The electrostatic pair function depends only on r and Pallas-Triton has no erf, so
  it is a cubic Hermite table of r (16384 intervals, exact values and derivatives at
  the nodes; |error| < 1e-12 of the function).
* The tile loops are ``lax.fori_loop`` over fixed-size tiles; the last tile is shifted
  back to end at N and only its new atoms are written.
* Memory modes (``MODES``) trade speed for memory: what is kept between the forward and
  the backward, and what is computed again.
* Long range in real space, or with particle-mesh Ewald for electrostatics and/or
  dispersion (``pme_electrostatics=(smearing, spacing)``, ``pme_dispersion=...``: the
  mesh as jaxpme builds it for ``--kspace-electrostatics pme`` /
  ``kspace_dispersion: pme``).
* ``energy_forces_strain`` also returns dE/d(strain) (the isotropic virial, box and
  positions scaled together, as jax-md's NPT takes it); the box is an argument of the
  compiled programs (``set_box``), so a box that moves compiles nothing.
* Several GPUs of one node (``devices``): every kernel and per-atom stage writes only its
  own atom's row and reads the others', so each device computes an equal share of the
  rows from full copies of the per-atom arrays, and the devices exchange their rows
  (all-gather) wherever the next stage reads other atoms: each layer's output, its
  gradients in the backward, the charges and dispersion ratios. Each device spreads
  every atom on the PME mesh; the forces are the same as on one.

Periodic orthorhombic boxes, at least 2 x the long-range cutoff across and wide enough
for the cell grid (``check.box_problem``); models with one head per degree entry,
degrees 1-4 (so3lr-tauri-s, -m, -l). ``so3lr.flash.md`` runs MD with it;
``python -m so3lr.flash.verify`` compares it with the neighbour-list potential.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

import jaxpme

from so3lr.flash import blocks
from so3lr.flash.check import box_problem, cell_grid, is_rocm
from so3lr.mlff.nn.embed.embed_sparse import PAIR_FLOOR_WIDTH, nlh_reference_distance
from so3lr.mlff.nn.observable.observable_sparse import c6_raw_potential
from so3lr.mlff.utils.kspace_utils import kspace_grid_shape

BE = 16              # edges per chunk in the pair kernels
NUM_WARPS = 2
# The attention kernels' launch, (edges per chunk, warps), by the width of the spherical
# filters' hidden layer (so3lr-tauri-s/m 32, so3lr-tauri-l 64): the fastest on an H200 for 12,288
# waters. so3lr-tauri-l takes 4 warps: a float32 dot lowered to FMAs keeps its whole reduction
# in each thread's registers, and at 2 warps its kernels spill.
ATT_LAUNCH = {
    32: {"fwd": (16, 2), "bwd_heads": (16, 2), "bwd_degrees": (8, 2)},
    64: {"fwd": (16, 4), "bwd_heads": (32, 4), "bwd_degrees": (16, 4)},
}
BL = 64              # candidates per chunk in the long-range kernel
AXIS = "atoms"       # the device axis of the several-GPU mesh


def _warps(n):
    """Warps for a launch tuned with n on NVIDIA: half on ROCm (64-lane wavefronts), as measured
    on the MI355X."""
    return max(1, n // 2) if is_rocm() else n


def _att_launch(width):
    """ATT_LAUNCH for this filter width (the nearest wider tabulated), warps for this platform."""
    launch = ATT_LAUNCH[min((w for w in ATT_LAUNCH if w >= width), default=max(ATT_LAUNCH))]
    return {k: (be, _warps(w)) for k, (be, w) in launch.items()}


def _cp(num_warps):
    return plgpu.CompilerParams(num_warps=num_warps, num_stages=1)


def _mm(a, b, x3=False):
    """A float32 dot: FMAs (the forward), or x3: on tensor cores as three TF32 products (the
    backward), whose float32 energies and forces are as close to float64."""
    # no TF32 on ROCm
    precision = lax.DotAlgorithmPreset.TF32_TF32_F32_X3 if x3 and not is_rocm() else lax.Precision.HIGHEST
    return jnp.dot(a, b, precision=precision, preferred_element_type=a.dtype)


def _filter_mm(hz, wb, c0, G, x3=False):
    """hz [BE, SH] @ wb[:, c0:c0+G]. Pallas-Triton has no float64 dot, so float64 sums
    over the SH hidden units (weight rows read from the ref: values cannot be sliced)."""
    if hz.dtype != jnp.float64:
        return _mm(hz, wb[:, pl.ds(c0, G)], x3)
    sh = jnp.arange(hz.shape[1])
    out = jnp.zeros((hz.shape[0], G), hz.dtype)
    for k in range(hz.shape[1]):
        col = jnp.sum(jnp.where(sh[None, :] == k, hz, 0.0), axis=1)
        out = out + col[:, None] * wb[k, pl.ds(c0, G)][None, :]
    return out


def _filter_mm_t(gw, wb, c0, G):
    """gw [BE, G] @ wb[:, c0:c0+G].T -> [BE, SH] (float64 likewise; the backward's: TF32x3)."""
    if gw.dtype != jnp.float64:
        return _mm(gw, wb[:, pl.ds(c0, G)].T, x3=True)
    nsh = wb.shape[0]
    sh = jnp.arange(nsh)
    out = jnp.zeros((gw.shape[0], nsh), gw.dtype)
    for k in range(nsh):
        out = jnp.where(sh[None, :] == k, jnp.sum(gw * wb[k, pl.ds(c0, G)][None, :], axis=1)[:, None], out)
    return out


# =========================================================================== #
# kernel pieces (plain functions, traced into the kernels)
# =========================================================================== #
def _i32(x):
    return jnp.asarray(x).astype(jnp.int32)


def _rem(a, b):
    """a mod b for a >= 0 (lax.rem: jnp's % lowers to a floor-mod select that Pallas-Triton
    miscompiles for integers)."""
    return lax.rem(_i32(a), _i32(b))


def _div(a, b):
    """a // b for a >= 0."""
    return lax.div(_i32(a), _i32(b))


def _cdiv(a, b):
    return _div(_i32(a) + (b - 1), b)


def _min_image(d, L):
    return d - L * jnp.floor(d / L + 0.5)


def _switch(x, on, off):
    """switching_fn(x, on, off) and its derivative."""
    c = (x - on) / (off - on)
    p = 1.0 - c
    sa = jnp.where(p > 0.0, jnp.exp(-1.0 / jnp.where(p > 0.0, p, 1.0)), 0.0)
    sb = jnp.where(c > 0.0, jnp.exp(-1.0 / jnp.where(c > 0.0, c, 1.0)), 0.0)
    da = jnp.where(p > 0.0, -sa / jnp.where(p > 0.0, p * p, 1.0), 0.0) / (off - on)
    db = jnp.where(c > 0.0, sb / jnp.where(c > 0.0, c * c, 1.0), 0.0) / (off - on)
    den = sa + sb
    ok = den > 0.0
    ds = jnp.where(ok, den, 1.0)
    return jnp.where(ok, sa / ds, 0.0), jnp.where(ok, (da * sb - sa * db) / (ds * ds), 0.0)


def _geometry(px, py, pz, jj, emask, ax, ay, az, Lx, Ly, Lz, rc):
    """Unit vector, distance, cutoff and its derivative for edges a <- jj (masked lanes:
    r = 2 rc, cut = 0)."""
    # the displacement in the positions' precision (float64 positions, float32 model:
    # no rounding from the size of the coordinates), the rest in the model's (rc's)
    dx = _min_image(plgpu.load(px.at[jj], mask=emask, other=0.0) - ax, Lx).astype(rc.dtype)
    dy = _min_image(plgpu.load(py.at[jj], mask=emask, other=0.0) - ay, Ly).astype(rc.dtype)
    dz = _min_image(plgpu.load(pz.at[jj], mask=emask, other=0.0) - az, Lz).astype(rc.dtype)
    r2 = dx * dx + dy * dy + dz * dz
    ok = emask & (r2 > 0.0)
    r = jnp.where(ok, jnp.sqrt(jnp.where(ok, r2, 1.0)), 2.0 * rc)
    inv = 1.0 / r
    ux = jnp.where(ok, dx * inv, 1.0)
    uy = jnp.where(ok, dy * inv, 0.0)
    uz = jnp.where(ok, dz * inv, 0.0)
    x = r / rc
    inside = r < rc
    cut = jnp.where(inside, 1.0 - 6.0 * x ** 5 + 15.0 * x ** 4 - 10.0 * x ** 3, 0.0)
    dcut = jnp.where(inside, (-30.0 * x ** 4 + 60.0 * x ** 3 - 30.0 * x ** 2) / rc, 0.0)
    return ux, uy, uz, r, cut, dcut


def _floor(r, za, jj, emask, z, rmin, ZMAX):
    """The descriptor floor (so3lr_dev): the distance the radial filters see, and its
    derivative. Identity above rmin(Z_a, Z_b) + width."""
    zb = plgpu.load(z.at[jj], mask=emask, other=1)
    lo = jnp.minimum(za, zb) - 1
    hi = jnp.maximum(za, zb) - 1
    d_off = plgpu.load(rmin.at[lo * ZMAX + hi], mask=emask, other=0.0) + PAIR_FLOOR_WIDTH
    t = jnp.exp(-2.0 * jnp.maximum((d_off - r) / PAIR_FLOOR_WIDTH, 0.0))
    th = (1.0 - t) / (1.0 + t)                      # tanh(u), u >= 0
    above = r >= d_off
    return (jnp.where(above, r, d_off - PAIR_FLOOR_WIDTH * th),
            jnp.where(above, 1.0, 1.0 - th * th))


def _table_coords(r, spacing, n_intervals, column=True):
    """Interval index and offset in it; the offset as a column for [BE, G] tables
    (Pallas cannot slice a value, so the 1-D form is a separate request)."""
    t = r / spacing
    i0 = jnp.minimum(jnp.maximum(jnp.floor(t), 0.0), n_intervals - 1.0)
    return i0.astype(jnp.int32), ((t - i0)[:, None] if column else t - i0)


def _rows(ref, rows, co, mask, dtype):
    return plgpu.load(ref.at[rows[:, None], co[None, :]], mask=mask[:, None], other=0.0).astype(dtype)


def _radial(tv, td, i0, u, spacing, co, emask, with_derivative=False):
    f0 = _rows(tv, i0, co, emask, u.dtype)
    f1 = _rows(tv, i0 + 1, co, emask, u.dtype)
    d0 = _rows(td, i0, co, emask, u.dtype)
    d1 = _rows(td, i0 + 1, co, emask, u.dtype)
    u2 = u * u
    u3 = u2 * u
    w = ((2.0 * u3 - 3.0 * u2 + 1.0) * f0 + (3.0 * u2 - 2.0 * u3) * f1
         + spacing * ((u3 - 2.0 * u2 + u) * d0 + (u3 - u2) * d1))
    if not with_derivative:
        return w
    dw = ((6.0 * u2 - 6.0 * u) * (f0 - f1) / spacing
          + (3.0 * u2 - 4.0 * u + 1.0) * d0 + (3.0 * u2 - 2.0 * u) * d1)
    return w, dw


def _ev_diff(ev, ev_a, jj, mo, valid, emask):
    m2 = emask[:, None] & valid[None, :]
    g = plgpu.load(ev.at[jj[:, None], mo[None, :]], mask=m2, other=0.0)
    return jnp.where(m2, g - ev_a[None, :], 0.0)


def _spherical_z(diff, md, wa, ba, cdeg, ND):
    """One spherical filter's hidden pre-activation from |ev_j - ev_a|^2 per degree."""
    d2 = diff * diff
    z = jnp.zeros((diff.shape[0], wa.shape[1]), diff.dtype) + ba[...][None, :]
    for d in range(ND):
        c = jnp.sum(jnp.where(md[None, :] == d, d2, 0.0), axis=1) * cdeg[d]
        z = z + c[:, None] * wa[d, :][None, :]
    return z


def _spherical_grad(diff, gz, md, wa, cdeg, ND):
    gdiff = jnp.zeros_like(diff)
    for d in range(ND):
        gc = jnp.sum(gz * wa[d, :][None, :], axis=1) * cdeg[d]
        gdiff = gdiff + jnp.where(md[None, :] == d, 2.0 * gc[:, None] * diff, 0.0)
    return gdiff


def _silu_and_derivative(z):
    sg = jax.nn.sigmoid(z)
    return z * sg, sg * (1.0 + z * (1.0 - sg))


# spherical harmonics, degrees 1-4, as a [BE, MP] tile
_YC = dict(
    c1=math.sqrt(3.0 / (4.0 * math.pi)), a2=0.5 * math.sqrt(15.0 / math.pi),
    b2=0.25 * math.sqrt(5.0 / math.pi), c2=0.25 * math.sqrt(15.0 / math.pi),
    d0=0.25 * math.sqrt(35.0 / (2.0 * math.pi)), d1=0.5 * math.sqrt(105.0 / math.pi),
    d2=0.25 * math.sqrt(21.0 / (2.0 * math.pi)), d3=0.25 * math.sqrt(7.0 / math.pi),
    d5=0.25 * math.sqrt(105.0 / math.pi), e0=0.75 * math.sqrt(35.0 / math.pi),
    e1=0.75 * math.sqrt(35.0 / (2.0 * math.pi)), e2=0.75 * math.sqrt(5.0 / math.pi),
    e3=0.75 * math.sqrt(5.0 / (2.0 * math.pi)), e4=3.0 / 16.0 * math.sqrt(1.0 / math.pi),
    e6=0.375 * math.sqrt(5.0 / math.pi), e8=3.0 / 16.0 * math.sqrt(35.0 / math.pi))


def _ylm_components(x, y, z, k):
    return [
        k["c1"] * y, k["c1"] * z, k["c1"] * x,
        k["a2"] * x * y, k["a2"] * y * z, k["b2"] * (3.0 * z * z - 1.0), k["a2"] * x * z,
        k["c2"] * (x * x - y * y),
        k["d0"] * y * (3.0 * x * x - y * y), k["d1"] * x * y * z, k["d2"] * y * (5.0 * z * z - 1.0),
        k["d3"] * (5.0 * z ** 3 - 3.0 * z), k["d2"] * x * (5.0 * z * z - 1.0),
        k["d5"] * (x * x - y * y) * z, k["d0"] * x * (x * x - 3.0 * y * y),
        k["e0"] * x * y * (x * x - y * y), k["e1"] * y * (3.0 * x * x - y * y) * z,
        k["e2"] * x * y * (7.0 * z * z - 1.0), k["e3"] * y * (7.0 * z ** 3 - 3.0 * z),
        k["e4"] * (35.0 * z ** 4 - 30.0 * z * z + 3.0), k["e3"] * x * (7.0 * z ** 3 - 3.0 * z),
        k["e6"] * (x * x - y * y) * (7.0 * z * z - 1.0), k["e1"] * x * (x * x - 3.0 * y * y) * z,
        k["e8"] * (x * x * (x * x - 3.0 * y * y) - y * y * (3.0 * x * x - y * y)),
    ]


def _consts(dtype):
    return {k: jnp.asarray(v, dtype) for k, v in _YC.items()}


def _ycols(md, mo, YS):
    """Each column's spherical-harmonic component (its index in _ylm_components, -1 for
    padding): the columns of degree entry e start at YS[e][0] and hold its degree's
    components from YS[e][1] on. A degree listed twice (so3lr-tauri-l) reads the same ones."""
    yc = mo * 0 - 1
    for e, (start, base) in enumerate(YS):
        yc = jnp.where(md == e, mo - start + base, yc)
    return yc


def _ylm(x, y, z, yc, MP):
    t = jnp.zeros((x.shape[0], MP), x.dtype)
    for m, v in enumerate(_ylm_components(x, y, z, _consts(x.dtype))):
        t = jnp.where(yc[None, :] == m, v[:, None], t)
    return t


def _ylm_vjp(x, y, z, g, yc):
    """sum over columns c of g[:, c] dY_yc[c]/d(x, y, z)."""
    k = _consts(x.dtype)
    col = [jnp.sum(jnp.where(yc[None, :] == m, g, 0.0), axis=1) for m in range(24)]
    q5 = 5.0 * z * z - 1.0
    q7 = 7.0 * z * z - 1.0
    c3 = 7.0 * z ** 3 - 3.0 * z
    dc3 = 21.0 * z * z - 3.0
    gx = (k["c1"] * col[2] + k["a2"] * (col[3] * y + col[6] * z) + 2.0 * k["c2"] * col[7] * x
          + 6.0 * k["d0"] * col[8] * x * y + k["d1"] * col[9] * y * z + k["d2"] * col[12] * q5
          + 2.0 * k["d5"] * col[13] * x * z + k["d0"] * col[14] * (3.0 * x * x - 3.0 * y * y)
          + k["e0"] * col[15] * (3.0 * x * x * y - y ** 3) + 6.0 * k["e1"] * col[16] * x * y * z
          + k["e2"] * col[17] * y * q7 + k["e3"] * col[20] * c3 + 2.0 * k["e6"] * col[21] * x * q7
          + k["e1"] * col[22] * (3.0 * x * x * z - 3.0 * y * y * z)
          + k["e8"] * col[23] * (4.0 * x ** 3 - 12.0 * x * y * y))
    gy = (k["c1"] * col[0] + k["a2"] * (col[3] * x + col[4] * z) - 2.0 * k["c2"] * col[7] * y
          + k["d0"] * col[8] * (3.0 * x * x - 3.0 * y * y) + k["d1"] * col[9] * x * z
          + k["d2"] * col[10] * q5 - 2.0 * k["d5"] * col[13] * y * z - 6.0 * k["d0"] * col[14] * x * y
          + k["e0"] * col[15] * (x ** 3 - 3.0 * x * y * y)
          + k["e1"] * col[16] * (3.0 * x * x * z - 3.0 * y * y * z) + k["e2"] * col[17] * x * q7
          + k["e3"] * col[18] * c3 - 2.0 * k["e6"] * col[21] * y * q7 - 6.0 * k["e1"] * col[22] * x * y * z
          + k["e8"] * col[23] * (4.0 * y ** 3 - 12.0 * x * x * y))
    gz = (k["c1"] * col[1] + k["a2"] * (col[4] * y + col[6] * x) + 6.0 * k["b2"] * col[5] * z
          + k["d1"] * col[9] * x * y + 10.0 * k["d2"] * col[10] * y * z
          + k["d3"] * col[11] * (15.0 * z * z - 3.0) + 10.0 * k["d2"] * col[12] * x * z
          + k["d5"] * col[13] * (x * x - y * y)
          + k["e1"] * col[16] * (3.0 * x * x * y - y ** 3) + 14.0 * k["e2"] * col[17] * x * y * z
          + k["e3"] * col[18] * y * dc3 + k["e4"] * col[19] * (140.0 * z ** 3 - 60.0 * z)
          + k["e3"] * col[20] * x * dc3 + 14.0 * k["e6"] * col[21] * z * (x * x - y * y)
          + k["e1"] * col[22] * (x ** 3 - 3.0 * x * y * y))
    return gx, gy, gz


def _put_row(acc, h, part):
    return jnp.where(jnp.arange(acc.shape[0])[:, None] == h, acc + part[None, :], acc)


def _row(acc, h):
    return jnp.sum(jnp.where(jnp.arange(acc.shape[0])[:, None] == h, acc, 0.0), axis=0)


# =========================================================================== #
# neighbour table
# =========================================================================== #
def _each_atom(rows, N, APP, atom):
    """atom(a) for this program's atom, or its APP atoms (guarded) where N programs would pass
    HIP's 2^32 threads a launch."""
    if APP == 1:
        atom(rows[0] + _i32(pl.program_id(0)))
        return
    assert N > 0, "APP > 1 needs the number of rows N"
    for i in range(APP):
        a = rows[0] + _i32(pl.program_id(0)) * APP + i
        pl.when(a < rows[0] + N)(partial(atom, a))


def _ell_kernel(px, py, pz, order, cell_start, cell_of, geo, boxl, rows, nbr, count, *, NX, NY, NZ, K, BLE, N, APP):
    def atom(a):
        Lx, Ly, Lz, rc2 = boxl[0], boxl[1], boxl[2], geo[0]
        ax, ay, az = px[a], py[a], pz[a]
        c = cell_of[a]
        cx = _rem(c, NX)
        cy = _rem(_div(c, NX), NY)
        cz = _div(c, NX * NY)

        def cell(k, n):
            dz = _div(k, 9) - 1
            dy = _rem(_div(k, 3), 3) - 1
            dx = _rem(k, 3) - 1
            cc = (_rem(cz + dz + NZ, NZ) * NY + _rem(cy + dy + NY, NY)) * NX + _rem(cx + dx + NX, NX)
            s = cell_start[cc]
            e = cell_start[cc + 1]

            def chunk(t, n):
                ks = s + t * BLE + jnp.arange(BLE)
                m = ks < e
                j = plgpu.load(order.at[ks], mask=m, other=0)
                ddx = _min_image(plgpu.load(px.at[j], mask=m, other=0.0) - ax, Lx)
                ddy = _min_image(plgpu.load(py.at[j], mask=m, other=0.0) - ay, Ly)
                ddz = _min_image(plgpu.load(pz.at[j], mask=m, other=0.0) - az, Lz)
                hit = m & (ddx * ddx + ddy * ddy + ddz * ddz < rc2) & (j != a)
                slot = n + jnp.cumsum(hit.astype(jnp.int32)) - 1
                plgpu.store(nbr.at[a, slot], j, mask=hit & (slot < K))
                return (n + jnp.sum(hit.astype(jnp.int32))).astype(jnp.int32)
            return lax.fori_loop(0, _cdiv(e - s, BLE), chunk, n)

        count[a] = lax.fori_loop(0, 27, cell, jnp.int32(0))

    _each_atom(rows, N, APP, atom)


# =========================================================================== #
# attention forward: x_out = x_in + messages, ev_out = ev_in + messages
# =========================================================================== #
def _att_fwd_kernel(q1, k1, v1, q2, k2, ev, px, py, pz, nbr, cnt, z, rmin,
                    t1v, t1d, t2v, t2d, wa1, ba1, wb1, bb1, wa2, ba2, wb2, bb2, mdeg, cdeg, scal,
                    x_in, ev_in, boxl, rows, x_out, ev_out, *, G, ND, MP, ZMAX, YS, CH=0, BE=BE, H0=0, NH=4, N,
                    APP):
    # CH 0: both channels; 1: heads only (x_out); 2: degrees only (ev_out). The outputs
    # alias the inputs, so a channel not run leaves its output as it was.
    def atom(a):
        n_a = jnp.minimum(cnt[a], nbr.shape[1])         # an overflowed row holds K entries
        inv_nc, spacing, n_int, rc = scal[0], scal[1], scal[2], scal[3]
        Lx, Ly, Lz = boxl[0], boxl[1], boxl[2]
        ax, ay, az = px[a], py[a], pz[a]
        dt = rc.dtype
        mo = jnp.arange(MP)
        md = mdeg[...]
        valid = md >= 0
        ev_a = plgpu.load(ev.at[a, mo], mask=valid, other=0.0)
        za = z[a]
        yc = _ycols(md, mo, YS)

        def chunk(c, carry):
            acc, dev_a = carry
            oe = c * BE + jnp.arange(BE)
            emask = oe < n_a
            jj = plgpu.load(nbr.at[a, oe], mask=emask, other=0)
            ux, uy, uz, r, ce, _ = _geometry(px, py, pz, jj, emask, ax, ay, az, Lx, Ly, Lz, rc)
            rt = _floor(r, za, jj, emask, z, rmin, ZMAX)[0]
            nz = ce != 0.0
            i0, u = _table_coords(rt, spacing, n_int)
            diff = _ev_diff(ev, ev_a, jj, mo, valid, emask)

            def head(j, acc):
                h = H0 + j
                co = h * G + jnp.arange(G)
                cp = j * G + jnp.arange(G)
                w = (_filter_mm(hz, wb1, h * G, G) + bb1[pl.ds(h * G, G)][None, :]
                     + _radial(t1v, t1d, i0, u, spacing, co, emask))
                s = jnp.sum(q1[a, pl.ds(j * G, G)].astype(dt)[None, :] * w * _rows(k1, jj, cp, emask, dt), axis=1)
                al = jnp.where(nz, ce * s * inv_nc, 0.0)
                return _put_row(acc, j, jnp.sum(al[:, None] * _rows(v1, jj, cp, emask, dt), axis=0))
            if CH != 2:
                hz = jax.nn.silu(_spherical_z(diff, md, wa1, ba1, cdeg, ND))
                acc = lax.fori_loop(0, NH, head, acc)
            if CH == 1:
                return acc, dev_a

            def degree(j, a_m):
                d = H0 + j
                co = d * G + jnp.arange(G)
                cp = j * G + jnp.arange(G)
                w = (_filter_mm(hz, wb2, d * G, G) + bb2[pl.ds(d * G, G)][None, :]
                     + _radial(t2v, t2d, i0, u, spacing, co, emask))
                s = jnp.sum(q2[a, pl.ds(j * G, G)].astype(dt)[None, :] * w * _rows(k2, jj, cp, emask, dt), axis=1)
                return a_m + jnp.where(md[None, :] == d, jnp.where(nz, ce * s * inv_nc, 0.0)[:, None], 0.0)
            hz = jax.nn.silu(_spherical_z(diff, md, wa2, ba2, cdeg, ND))
            a_m = lax.fori_loop(0, NH, degree, jnp.zeros((BE, MP), dt))
            return acc, dev_a + jnp.sum(a_m * _ylm(ux, uy, uz, yc, MP), axis=0)

        acc, dev_a = lax.fori_loop(0, _cdiv(n_a, BE), chunk,
                                   (jnp.zeros((NH, G), dt), jnp.zeros((MP,), dt)))
        if CH != 2:
            for j in range(NH):
                h = H0 + j
                x_out[a, pl.ds(h * G, G)] = x_in[a, pl.ds(h * G, G)] + _row(acc, j)
        if CH != 1:
            plgpu.store(ev_out.at[a, mo], plgpu.load(ev_in.at[a, mo], mask=valid, other=0.0) + dev_a,
                        mask=valid)

    _each_atom(rows, N, APP, atom)


# =========================================================================== #
# attention backward (one tile of atoms): projection gradients for the tile, dE/dev and
# forces added into the graph-wide arrays. Both ends of every edge.
# =========================================================================== #
def _att_bwd_heads_kernel(q1, k1, v1, ev, px, py, pz, nbr, cnt, z, rmin, tv, td, wa, ba, wb, bb,
                          mdeg, cdeg, gdx, scal, a0, gev_in, fx_in, fy_in, fz_in, fw_in, boxl,
                          gq, gk, gv, gev, fx_o, fy_o, fz_o, fw_o, *, G, ND, MP, ZMAX, BE=BE, H0=0, NH=4):
    t = _i32(pl.program_id(0))
    a = a0[0] + t

    @pl.when(a >= a0[1])                          # the last tile overlaps: new atoms only
    def _():
        n_a = jnp.minimum(cnt[a], nbr.shape[1])
        inv_nc, spacing, n_int, rc = scal[0], scal[1], scal[2], scal[3]
        Lx, Ly, Lz = boxl[0], boxl[1], boxl[2]
        ax, ay, az = px[a], py[a], pz[a]
        dt = rc.dtype
        mo = jnp.arange(MP)
        md = mdeg[...]
        valid = md >= 0
        ev_a = plgpu.load(ev.at[a, mo], mask=valid, other=0.0)
        za = z[a]

        def chunk(c, carry):
            agq, agk, agv, gev_a, fx, fy, fz, fw = carry
            oe = c * BE + jnp.arange(BE)
            emask = oe < n_a
            jj = plgpu.load(nbr.at[a, oe], mask=emask, other=0)
            ux, uy, uz, r, ce, dce = _geometry(px, py, pz, jj, emask, ax, ay, az, Lx, Ly, Lz, rc)
            rt, drt = _floor(r, za, jj, emask, z, rmin, ZMAX)
            nz = ce != 0.0
            i0, u = _table_coords(rt, spacing, n_int)
            diff = _ev_diff(ev, ev_a, jj, mo, valid, emask)
            hz, dz = _silu_and_derivative(_spherical_z(diff, md, wa, ba, cdeg, ND))

            def head(j, c2):
                h = H0 + j
                agq, agk, agv, gz, gc, grr = c2
                co = h * G + jnp.arange(G)
                cp = j * G + jnp.arange(G)
                wr, dwr = _radial(tv, td, i0, u, spacing, co, emask, True)
                w = _filter_mm(hz, wb, h * G, G, x3=True) + bb[pl.ds(h * G, G)][None, :] + wr
                qa = q1[a, pl.ds(j * G, G)].astype(dt)[None, :]
                ka = k1[a, pl.ds(j * G, G)].astype(dt)[None, :]
                va = v1[a, pl.ds(j * G, G)].astype(dt)[None, :]
                qb = _rows(q1, jj, cp, emask, dt)
                kb = _rows(k1, jj, cp, emask, dt)
                gxb = _rows(gdx, jj, co, emask, dt)
                s = jnp.sum(qa * w * kb, axis=1)
                sr = jnp.sum(qb * w * ka, axis=1)
                ga = jnp.sum(gdx[a, pl.ds(h * G, G)][None, :] * _rows(v1, jj, cp, emask, dt), axis=1)
                gar = jnp.sum(gxb * va, axis=1)
                gs = jnp.where(nz, ga * ce * inv_nc, 0.0)[:, None]
                gsr = jnp.where(nz, gar * ce * inv_nc, 0.0)[:, None]
                alr = jnp.where(nz, ce * sr * inv_nc, 0.0)[:, None]
                agq = _put_row(agq, j, jnp.sum(gs * w * kb, axis=0))
                agk = _put_row(agk, j, jnp.sum(gsr * w * qb, axis=0))
                agv = _put_row(agv, j, jnp.sum(alr * gxb, axis=0))
                gw = gs * qa * kb + gsr * qb * ka
                gc = gc + jnp.where(emask, (ga * s + gar * sr) * inv_nc, 0.0)
                grr = grr + jnp.sum(gw * dwr, axis=1)
                gz = gz + _filter_mm_t(gw, wb, h * G, G)
                return agq, agk, agv, gz, gc, grr
            z0 = jnp.zeros((BE,), dt)
            agq, agk, agv, gz, gc, grr = lax.fori_loop(
                0, NH, head, (agq, agk, agv, jnp.zeros((BE, wa.shape[1]), dt), z0, z0))
            gev_a = gev_a - jnp.sum(_spherical_grad(diff, gz * dz, md, wa, cdeg, ND), axis=0)
            fr = jnp.where(emask, grr * drt + gc * dce, 0.0)
            return (agq, agk, agv, gev_a, fx + jnp.sum(fr * ux), fy + jnp.sum(fr * uy),
                    fz + jnp.sum(fr * uz), fw + jnp.sum(fr * r))

        zr = jnp.zeros((NH, G), dt)
        s0 = jnp.zeros((), dt)
        agq, agk, agv, gev_a, fx, fy, fz, fw = lax.fori_loop(
            0, _cdiv(n_a, BE), chunk, (zr, zr, zr, jnp.zeros((MP,), dt), s0, s0, s0, s0))
        for j in range(NH):
            gq[t, pl.ds(j * G, G)] = _row(agq, j)
            gk[t, pl.ds(j * G, G)] = _row(agk, j)
            gv[t, pl.ds(j * G, G)] = _row(agv, j)
        plgpu.store(gev.at[a, mo], plgpu.load(gev_in.at[a, mo], mask=valid, other=0.0) + gev_a,
                    mask=valid)
        fx_o[a] = fx_in[a] + fx
        fy_o[a] = fy_in[a] + fy
        fz_o[a] = fz_in[a] + fz
        fw_o[a] = fw_in[a] + fw


def _att_bwd_degrees_kernel(q2, k2, ev, px, py, pz, nbr, cnt, z, rmin, tv, td, wa, ba, wb, bb,
                            mdeg, cdeg, par, gdev, scal, a0, gev_in, fx_in, fy_in, fz_in, fw_in, boxl,
                            gq, gk, gev, fx_o, fy_o, fz_o, fw_o, *, G, ND, MP, ZMAX, YS, BE=BE, H0=0,
                            NH=4):
    t = _i32(pl.program_id(0))
    a = a0[0] + t

    @pl.when(a >= a0[1])
    def _():
        n_a = jnp.minimum(cnt[a], nbr.shape[1])
        inv_nc, spacing, n_int, rc = scal[0], scal[1], scal[2], scal[3]
        Lx, Ly, Lz = boxl[0], boxl[1], boxl[2]
        ax, ay, az = px[a], py[a], pz[a]
        dt = rc.dtype
        mo = jnp.arange(MP)
        md = mdeg[...]
        valid = md >= 0
        ev_a = plgpu.load(ev.at[a, mo], mask=valid, other=0.0)
        gdev_a = plgpu.load(gdev.at[a, mo], mask=valid, other=0.0)[None, :]
        par_m = plgpu.load(par.at[jnp.maximum(md, 0)], mask=valid, other=0.0)[None, :]
        za = z[a]
        yc = _ycols(md, mo, YS)

        def chunk(c, carry):
            agq, agk, gev_a, fx, fy, fz, fw = carry
            oe = c * BE + jnp.arange(BE)
            emask = oe < n_a
            jj = plgpu.load(nbr.at[a, oe], mask=emask, other=0)
            ux, uy, uz, r, ce, dce = _geometry(px, py, pz, jj, emask, ax, ay, az, Lx, Ly, Lz, rc)
            rt, drt = _floor(r, za, jj, emask, z, rmin, ZMAX)
            nz = ce != 0.0
            i0, u = _table_coords(rt, spacing, n_int)
            diff = _ev_diff(ev, ev_a, jj, mo, valid, emask)
            hz, dz = _silu_and_derivative(_spherical_z(diff, md, wa, ba, cdeg, ND))
            m2 = emask[:, None] & valid[None, :]
            y = _ylm(ux, uy, uz, yc, MP)
            gdev_b = plgpu.load(gdev.at[jj[:, None], mo[None, :]], mask=m2, other=0.0)
            gy = gdev_a * y
            gyr = gdev_b * y * par_m

            def degree(j, c2):
                d = H0 + j
                agq, agk, gz, a_m, a_mr, gc, grr = c2
                co = d * G + jnp.arange(G)
                cp = j * G + jnp.arange(G)
                wr, dwr = _radial(tv, td, i0, u, spacing, co, emask, True)
                w = _filter_mm(hz, wb, d * G, G, x3=True) + bb[pl.ds(d * G, G)][None, :] + wr
                qa = q2[a, pl.ds(j * G, G)].astype(dt)[None, :]
                ka = k2[a, pl.ds(j * G, G)].astype(dt)[None, :]
                qb = _rows(q2, jj, cp, emask, dt)
                kb = _rows(k2, jj, cp, emask, dt)
                s = jnp.sum(qa * w * kb, axis=1)
                sr = jnp.sum(qb * w * ka, axis=1)
                on_d = md[None, :] == d
                a_m = a_m + jnp.where(on_d, jnp.where(nz, ce * s * inv_nc, 0.0)[:, None], 0.0)
                a_mr = a_mr + jnp.where(on_d, jnp.where(nz, ce * sr * inv_nc, 0.0)[:, None], 0.0)
                ga = jnp.sum(jnp.where(on_d, gy, 0.0), axis=1)
                gar = jnp.sum(jnp.where(on_d, gyr, 0.0), axis=1)
                gs = jnp.where(nz, ga * ce * inv_nc, 0.0)[:, None]
                gsr = jnp.where(nz, gar * ce * inv_nc, 0.0)[:, None]
                agq = _put_row(agq, j, jnp.sum(gs * w * kb, axis=0))
                agk = _put_row(agk, j, jnp.sum(gsr * w * qb, axis=0))
                gw = gs * qa * kb + gsr * qb * ka
                gc = gc + jnp.where(emask, (ga * s + gar * sr) * inv_nc, 0.0)
                grr = grr + jnp.sum(gw * dwr, axis=1)
                gz = gz + _filter_mm_t(gw, wb, d * G, G)
                return agq, agk, gz, a_m, a_mr, gc, grr
            z0 = jnp.zeros((BE,), dt)
            zt = jnp.zeros((BE, MP), dt)
            agq, agk, gz, a_m, a_mr, gc, grr = lax.fori_loop(
                0, NH, degree, (agq, agk, jnp.zeros((BE, wa.shape[1]), dt), zt, zt, z0, z0))
            gev_a = gev_a - jnp.sum(_spherical_grad(diff, gz * dz, md, wa, cdeg, ND), axis=0)
            gx, gy_, gz_ = _ylm_vjp(ux, uy, uz, jnp.where(m2, a_m * gdev_a + par_m * a_mr * gdev_b, 0.0), yc)
            inv_r = jnp.where(emask, 1.0 / r, 0.0)
            proj = gx * ux + gy_ * uy + gz_ * uz
            fr = jnp.where(emask, grr * drt + gc * dce, 0.0)
            # the virial takes the radial part only: the angular part is across the edge
            return (agq, agk, gev_a,
                    fx + jnp.sum(fr * ux + (gx - proj * ux) * inv_r),
                    fy + jnp.sum(fr * uy + (gy_ - proj * uy) * inv_r),
                    fz + jnp.sum(fr * uz + (gz_ - proj * uz) * inv_r),
                    fw + jnp.sum(fr * r))

        zr = jnp.zeros((NH, G), dt)
        s0 = jnp.zeros((), dt)
        agq, agk, gev_a, fx, fy, fz, fw = lax.fori_loop(
            0, _cdiv(n_a, BE), chunk, (zr, zr, jnp.zeros((MP,), dt), s0, s0, s0, s0))
        for j in range(NH):
            gq[t, pl.ds(j * G, G)] = _row(agq, j)
            gk[t, pl.ds(j * G, G)] = _row(agk, j)
        plgpu.store(gev.at[a, mo], plgpu.load(gev_in.at[a, mo], mask=valid, other=0.0) + gev_a,
                    mask=valid)
        fx_o[a] = fx_in[a] + fx
        fy_o[a] = fy_in[a] + fy
        fz_o[a] = fz_in[a] + fz
        fw_o[a] = fw_in[a] + fw


# =========================================================================== #
# short-range pairs: the NLH repulsion (per-atom energy, forces)
# =========================================================================== #
def _sr_kernel(px, py, pz, z, nbr, cnt, nlh_a, nlh_b, scal, fx_in, fy_in, fz_in, fw_in, boxl, rows,
               energy, fx_o, fy_o, fz_o, fw_o, *, ZMAX, NK, NKP, N, APP):
    """Per-atom NLH energy (half of each pair) and its forces."""
    def atom(a):
        n_a = jnp.minimum(cnt[a], nbr.shape[1])         # an overflowed row holds K entries
        ke, rc = scal[0], scal[3]
        Lx, Ly, Lz = boxl[0], boxl[1], boxl[2]
        ax, ay, az = px[a], py[a], pz[a]
        dt = rc.dtype
        za = z[a]
        kk = jnp.arange(NKP)

        def chunk(c, carry):
            e_a, fx, fy, fz, fw = carry
            oe = c * BE + jnp.arange(BE)
            emask = oe < n_a
            jj = plgpu.load(nbr.at[a, oe], mask=emask, other=0)
            ux, uy, uz, r, ce, dce = _geometry(px, py, pz, jj, emask, ax, ay, az, Lx, Ly, Lz, rc)
            zb = plgpu.load(z.at[jj], mask=emask, other=1)
            lo = jnp.minimum(za, zb) - 1
            hi = jnp.maximum(za, zb) - 1
            tab = (lo * ZMAX + hi)[:, None] * NK + kk[None, :]
            tm = emask[:, None] & (kk < NK)[None, :]
            ca = plgpu.load(nlh_a.at[tab], mask=tm, other=0.0)
            cb = plgpu.load(nlh_b.at[tab], mask=tm, other=0.0)
            ex = ca * jnp.exp(-cb * r[:, None])
            yv = jnp.sum(ex, axis=1)
            dyv = -jnp.sum(cb * ex, axis=1)
            w, dw = _switch(r, 0.0, 1.5)
            x = ke * (za * zb).astype(dt) / r
            e_pair = w * ce * x * yv
            de = dw * ce * x * yv + w * dce * x * yv - w * ce * x / r * yv + w * ce * x * dyv
            e_a = e_a + 0.5 * jnp.sum(jnp.where(emask, e_pair, 0.0))
            fr = jnp.where(emask, de, 0.0)
            return (e_a, fx + jnp.sum(fr * ux), fy + jnp.sum(fr * uy), fz + jnp.sum(fr * uz),
                    fw + jnp.sum(fr * r))

        s0 = jnp.zeros((), dt)
        e_a, fx, fy, fz, fw = lax.fori_loop(0, _cdiv(n_a, BE), chunk, (s0, s0, s0, s0, s0))
        energy[a] = e_a
        fx_o[a] = fx_in[a] + fx
        fy_o[a] = fy_in[a] + fy
        fz_o[a] = fz_in[a] + fz
        fw_o[a] = fw_in[a] + fw

    _each_atom(rows, N, APP, atom)


# =========================================================================== #
# long range: electrostatics (tabulated phi) and dispersion, streamed from the grid
# =========================================================================== #
def _lr_kernel(px, py, pz, z, q, a0r, c6r, alphas, c6tab, order, cell_start, cell_of,
               phi_v, phi_d, lrs, e_in, fx_in, fy_in, fz_in, fw_in, scratch_in, boxl, c6lr_v, c6lr_d, rows,
               energy, gq, ga0, gc6, fx_o, fy_o, fz_o, fw_o, scratch, most,
               *, N, NX, NY, NZ, KC, MAXH, ELEC, DISP, DPME):
    pid = _i32(pl.program_id(0))
    nprog = _i32(pl.num_programs(0))
    # [rc, phi spacing, phi intervals, ke, damping start, gamma_scale, fs^(-4/21), hartree,
    #  bohr]; the box comes in boxl
    rc, phs, phn, ke = lrs[0], lrs[1], lrs[2], lrs[3]
    dstart, gsc, fs = lrs[4], lrs[5], lrs[6]
    Lx, Ly, Lz = boxl[0], boxl[1], boxl[2]
    hartree, bohr = lrs[7], lrs[8]
    dt = rc.dtype
    rc2 = rc * rc
    sxd, syd, szd = Lx / NX, Ly / NY, Lz / NZ                   # positions' dtype
    sx, sy, sz = sxd.astype(dt), syd.astype(dt), szd.astype(dt)
    b0, b1, b2, b3 = (jnp.asarray(v, dt) for v in (-0.00433008, 0.24428889, 0.04125273, -0.00078893))
    p0 = jnp.asarray(5.08, dt)

    def atom(it, most_seen):
        # N rows from rows[0] on (a device's own rows)
        a = rows[0] + _i32(pid + it * nprog)
        ax, ay, az = px[a], py[a], pz[a]
        c = cell_of[a]
        cx = _rem(c, NX)
        cy = _rem(_div(c, NX), NY)
        cz = _div(c, NX * NY)
        ox = jnp.clip(ax - Lx * jnp.floor(ax / Lx) - cx.astype(ax.dtype) * sxd, 0.0, sxd).astype(dt)
        oy = jnp.clip(ay - Ly * jnp.floor(ay / Ly) - cy.astype(ay.dtype) * syd, 0.0, syd).astype(dt)
        oz = jnp.clip(az - Lz * jnp.floor(az / Lz) - cz.astype(az.dtype) * szd, 0.0, szd).astype(dt)

        # ---- pass 1: partners within the cutoff into this program's scratch row ----
        def row_fn(k, nh):
            dzc = _div(k, 2 * KC + 1) - KC
            dyc = _rem(k, 2 * KC + 1) - KC
            gz_ = jnp.where(dzc > 0, dzc.astype(dt) * sz - oz,
                            jnp.where(dzc < 0, oz + (-dzc - 1).astype(dt) * sz, 0.0))
            gy_ = jnp.where(dyc > 0, dyc.astype(dt) * sy - oy,
                            jnp.where(dyc < 0, oy + (-dyc - 1).astype(dt) * sy, 0.0))
            g2 = gy_ * gy_ + gz_ * gz_
            xr = jnp.sqrt(jnp.maximum(rc2 - g2, 0.0))
            lo = jnp.maximum(cx + jnp.floor((ox - xr) / sx).astype(jnp.int32), cx - KC)
            hi = jnp.minimum(cx + jnp.floor((ox + xr) / sx).astype(jnp.int32), cx + KC)
            hi = jnp.where(g2 < rc2, hi, lo - 1)
            zc = _rem(cz + dzc + NZ, NZ)
            yc = _rem(cy + dyc + NY, NY)
            row = (zc * NY + yc) * NX

            def part_fn(part, nh):
                # branches as traced ints: where(c, 0, 1) with two literals is rewritten
                # into boolean arithmetic that Pallas-Triton lowers as an i1 - i32 subtract
                zero = lo * 0
                c0 = jnp.where(part == 0, jnp.maximum(lo, zero),
                               jnp.where(lo < 0, lo + NX, jnp.where(hi >= NX, zero, zero + 1)))
                c1 = jnp.where(part == 0, jnp.minimum(hi, zero + NX - 1),
                               jnp.where(lo < 0, zero + NX - 1, jnp.where(hi >= NX, hi - NX, zero)))
                c1 = jnp.where(hi >= lo, c1, c0 - 1)
                s = cell_start[row + c0]
                e = jnp.where(c1 >= c0, cell_start[row + jnp.maximum(c1, c0 - 1) + 1], s)

                def chunk(tc, nh):
                    ks = s + tc * BL + jnp.arange(BL)
                    m = ks < e
                    j = plgpu.load(order.at[ks], mask=m, other=0)
                    dx = _min_image(plgpu.load(px.at[j], mask=m, other=0.0) - ax, Lx).astype(dt)
                    dy = _min_image(plgpu.load(py.at[j], mask=m, other=0.0) - ay, Ly).astype(dt)
                    dz = _min_image(plgpu.load(pz.at[j], mask=m, other=0.0) - az, Lz).astype(dt)
                    r2 = dx * dx + dy * dy + dz * dz
                    hit = m & (r2 < rc2) & (j != a) & (r2 > 0.0)
                    slot = nh + jnp.cumsum(hit.astype(jnp.int32)) - 1
                    plgpu.store(scratch.at[pid, slot], j, mask=hit & (slot < MAXH))
                    return (nh + jnp.sum(hit.astype(jnp.int32))).astype(jnp.int32)
                return lax.fori_loop(0, _cdiv(e - s, BL), chunk, nh)
            return lax.fori_loop(0, 2, part_fn, nh)
        nh = lax.fori_loop(0, (2 * KC + 1) * (2 * KC + 1), row_fn, jnp.int32(0))
        plgpu.debug_barrier()

        # ---- pass 2: the pair terms on the partners only ----
        za = z[a] - 1
        qa = q[a]
        al_ta = alphas[za]
        c6_ta = c6tab[za]
        al_a = al_ta * a0r[a]
        c6_a = c6_ta * c6r[a]
        n_use = jnp.minimum(nh, MAXH)

        def pairs(tc, acc):
            e_a, gq_a, gal_a, gc6_a, fx, fy, fz, fw = acc
            hidx = tc * BL + jnp.arange(BL)
            hit = hidx < n_use
            j = plgpu.load(scratch.at[pid, hidx], mask=hit, other=0)
            dx = _min_image(plgpu.load(px.at[j], mask=hit, other=0.0) - ax, Lx).astype(dt)
            dy = _min_image(plgpu.load(py.at[j], mask=hit, other=0.0) - ay, Ly).astype(dt)
            dz = _min_image(plgpu.load(pz.at[j], mask=hit, other=0.0) - az, Lz).astype(dt)
            r = jnp.sqrt(jnp.where(hit, dx * dx + dy * dy + dz * dz, rc2 * 0.25))
            de = jnp.zeros_like(r)
            ti, tu = _table_coords(r, phs, phn, column=False)

            def hermite(tv, td):
                """A cubic Hermite table of r (values, d/dr) at r, and its d/dr."""
                f0 = plgpu.load(tv.at[ti], mask=hit, other=0.0)
                f1 = plgpu.load(tv.at[ti + 1], mask=hit, other=0.0)
                d0 = plgpu.load(td.at[ti], mask=hit, other=0.0)
                d1 = plgpu.load(td.at[ti + 1], mask=hit, other=0.0)
                u2 = tu * tu
                u3 = u2 * tu
                return (((2.0 * u3 - 3.0 * u2 + 1.0) * f0 + (3.0 * u2 - 2.0 * u3) * f1
                         + phs * ((u3 - 2.0 * u2 + tu) * d0 + (u3 - u2) * d1)),
                        ((6.0 * u2 - 6.0 * tu) * (f0 - f1) / phs
                         + (3.0 * u2 - 4.0 * tu + 1.0) * d0 + (3.0 * u2 - 2.0 * tu) * d1))
            if ELEC:
                qb = plgpu.load(q.at[j], mask=hit, other=0.0)
                phi, dphi = hermite(phi_v, phi_d)
                kq = ke * qb
                e_a = e_a + 0.5 * jnp.sum(jnp.where(hit, kq * qa * phi, 0.0))
                gq_a = gq_a + jnp.sum(jnp.where(hit, kq * phi, 0.0))
                de = de + kq * qa * dphi
            if DISP:
                zb = plgpu.load(z.at[j], mask=hit, other=1) - 1
                al_b = (plgpu.load(alphas.at[zb], mask=hit, other=1.0)
                        * plgpu.load(a0r.at[j], mask=hit, other=1.0))
                c6_b = (plgpu.load(c6tab.at[zb], mask=hit, other=1.0)
                        * plgpu.load(c6r.at[j], mask=hit, other=1.0))
                den = al_a * al_a * c6_b + al_b * al_b * c6_a
                c6p = 2.0 * c6_a * c6_b * al_a * al_b / den
                dc6_dal = (2.0 * c6_a * c6_b * al_b - c6p * 2.0 * al_a * c6_b) / den
                dc6_dc6 = (2.0 * c6_b * al_a * al_b - c6p * al_b * al_b) / den
                alp = 0.5 * (al_a + al_b)
                a17 = jnp.exp(jnp.log(alp) / 7.0)
                rv = fs * a17
                sg = ((b3 * rv + b2) * rv + b1) * rv + b0
                gam = 0.5 / (sg * sg)
                dgam = -(3.0 * b3 * rv * rv + 2.0 * b2 * rv + b1) * rv / (7.0 * alp) / (sg * sg * sg)
                p = gsc * p0 * a17
                dp = p / (7.0 * alp)
                R = r / bohr
                R2 = R * R
                R6 = R2 * R2 * R2
                p2 = p * p
                p6 = p2 * p2 * p2
                i6 = 1.0 / (R6 + p6)
                i8 = 1.0 / (R6 * R2 + p6 * p2)
                i10 = 1.0 / (R6 * R2 * R2 + p6 * p2 * p2)
                ig = 1.0 / gam
                c8 = 5.0 * ig * c6p
                c10 = 245.0 / 8.0 * ig * ig * c6p
                v = -c6p * i6 - c8 * i8 - c10 * i10
                dvr = (6.0 * R2 * R2 * R * c6p * i6 * i6 + 8.0 * R6 * R * c8 * i8 * i8
                       + 10.0 * R6 * R2 * R * c10 * i10 * i10) / bohr
                dv_c6 = -i6 - 5.0 * ig * i8 - 245.0 / 8.0 * ig * ig * i10
                dv_gam = 5.0 * c6p * ig * ig * i8 + 2.0 * 245.0 / 8.0 * c6p * ig * ig * ig * i10
                dv_p = (6.0 * p2 * p2 * p * c6p * i6 * i6 + 8.0 * p6 * p * c8 * i8 * i8
                        + 10.0 * p6 * p2 * p * c10 * i10 * i10)
                w, dw = _switch(r, dstart, rc)
                hw = hartree * w
                g6 = hw * dv_c6 * dc6_dc6
                if DPME:
                    # with dispersion PME the mesh adds sqrt(C6_a C6_b) lr(R) for every
                    # pair, so the real-space part is the residual (so3lr_dev's split);
                    # lr is a table of r like phi
                    c6g = jnp.sqrt(c6_a * c6_b)
                    lr6, dlr6 = hermite(c6lr_v, c6lr_d)
                    v = v + c6g * lr6
                    dvr = dvr + c6g * dlr6
                    g6 = g6 + hw * lr6 * c6g / (2.0 * c6_a)
                e_a = e_a + 0.5 * jnp.sum(jnp.where(hit, hw * v, 0.0))
                gal_a = gal_a + jnp.sum(jnp.where(hit, hw * (dv_c6 * dc6_dal + 0.5 * (dv_gam * dgam + dv_p * dp)), 0.0))
                gc6_a = gc6_a + jnp.sum(jnp.where(hit, g6, 0.0))
                de = de + hartree * (w * dvr + dw * v)
            fr = jnp.where(hit, de / r, 0.0)
            return (e_a, gq_a, gal_a, gc6_a, fx + jnp.sum(fr * dx), fy + jnp.sum(fr * dy),
                    fz + jnp.sum(fr * dz), fw + jnp.sum(jnp.where(hit, de * r, 0.0)))
        s0 = jnp.zeros((), dt)
        e_a, gq_a, gal_a, gc6_a, fx, fy, fz, fw = lax.fori_loop(
            0, _cdiv(n_use, BL), pairs, (s0, s0, s0, s0, s0, s0, s0, s0))
        plgpu.debug_barrier()                      # the scratch row is reused next atom
        energy[a] = e_in[a] + e_a
        gq[a] = gq_a
        ga0[a] = gal_a * al_ta
        gc6[a] = gc6_a * c6_ta
        fx_o[a] = fx_in[a] + fx
        fy_o[a] = fy_in[a] + fy
        fz_o[a] = fz_in[a] + fz
        fw_o[a] = fw_in[a] + fw
        return jnp.maximum(most_seen, nh).astype(jnp.int32)

    n_mine = _div(N - pid + nprog - 1, nprog)
    most[pid] = lax.fori_loop(0, n_mine, atom, jnp.int32(0))


# =========================================================================== #
# tables (numpy, float64)
# =========================================================================== #
def _bernstein64(r, n_rbf, gamma=0.9448630629184640):
    """The model's Bernstein radial basis in float64 numpy, with d/dr."""
    log_fact = float(np.sum(np.log(np.arange(1, n_rbf))))
    b = np.array([log_fact - float(np.sum(np.log(np.arange(1, k + 1))))
                  - float(np.sum(np.log(np.arange(1, n_rbf - k)))) for k in range(n_rbf)])
    k = np.arange(n_rbf, dtype=np.float64)
    krev = k[::-1].copy()
    x = np.exp(-gamma * r)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lx = np.where(k != 0, k * np.log(x), 0.0)
        l1x = np.where(krev != 0, krev * np.log1p(-x), 0.0)
        basis = np.exp(b + lx + l1x)
        ok = (x != 0.0) & (x != 1.0)
        basis = np.where(ok, basis, 0.0)
        dlog = np.where(k != 0, k / x, 0.0) - np.where(krev != 0, krev / (1.0 - x), 0.0)
        dbasis = np.where(ok, basis * dlog * (-gamma * x), 0.0)
    return basis, dbasis


def _radial_table(p, names, n_rbf, grid):
    """Values and d/dr of mlp(bernstein(r)) on grid, float64 (silu between the layers)."""
    basis, dbasis = _bernstein64(grid, n_rbf)
    w1 = np.asarray(p[names[0]]["kernel"], np.float64)
    c1 = np.asarray(p[names[0]]["bias"], np.float64)
    w2 = np.asarray(p[names[1]]["kernel"], np.float64)
    c2 = np.asarray(p[names[1]]["bias"], np.float64)
    h = basis @ w1 + c1
    sg = 1.0 / (1.0 + np.exp(-h))
    act = h * sg
    dact = sg * (1.0 + h * (1.0 - sg))
    return act @ w2 + c2, ((dbasis @ w1) * dact) @ w2


def _phi_table(rc, sigma, n, smearing=None):
    """The electrostatic pair function phi(r) = V(r) - V(rc) - (1 - f) V'(rc) (r - rc),
    V = erf(r/sigma)/r, f = switching_fn(r, 0.45 rc, rc), and phi', on [0, rc]. With PME
    (Ewald ``smearing``) V = erf(r/sigma)/r - erf(r/(sqrt(2) smearing))/r, the mesh's
    part taken out (so3lr_dev's coulomb_erf_shifted_force_smooth_pme)."""
    from scipy.special import erf
    r = np.linspace(0.0, rc, n + 1)
    rs = np.where(r > 0, r, 1.0)
    widths = [(sigma, 1.0)] + ([] if smearing is None else [(math.sqrt(2.0) * smearing, -1.0)])

    def V(x):
        xs = np.where(x > 0, x, 1.0)
        return sum(s * np.where(x > 0, erf(xs / w) / xs, 2.0 / (math.sqrt(math.pi) * w)) for w, s in widths)

    def dV(x):
        xs = np.where(x > 0, x, 1.0)
        return sum(s * np.where(x > 0, (2.0 * xs * np.exp(-(xs / w) ** 2) / (math.sqrt(math.pi) * w)
                                        - erf(xs / w)) / xs ** 2, 0.0) for w, s in widths)

    def sw(x, on, off):
        c = (x - on) / (off - on)
        sa = np.where(1 - c > 0, np.exp(-1.0 / np.where(1 - c > 0, 1 - c, 1.0)), 0.0)
        sb = np.where(c > 0, np.exp(-1.0 / np.where(c > 0, c, 1.0)), 0.0)
        da = np.where(1 - c > 0, -sa / np.where(1 - c > 0, (1 - c) ** 2, 1.0), 0.0) / (off - on)
        db = np.where(c > 0, sb / np.where(c > 0, c * c, 1.0), 0.0) / (off - on)
        den = sa + sb
        return sa / den, (da * sb - sa * db) / den ** 2

    f, df = sw(r, 0.45 * rc, rc)
    vc, dvc = float(V(np.array([rc]))[0]), float(dV(np.array([rc]))[0])
    phi = V(rs) * (r > 0) + (r == 0) * V(np.array([0.0]))[0] - vc - (1.0 - f) * dvc * (r - rc)
    dphi = dV(r) - (1.0 - f) * dvc + df * dvc * (r - rc)
    return phi, dphi, rc / n


def _c6_lr_table(rc, smearing_bohr, n, bohr):
    """lr(R) = (1 - exp(-x) (1 + x + x^2/2)) / R^6, x = R^2 / (2 s^2), R and s in Bohr:
    the part of 1/R^6 the dispersion mesh carries (jaxpme lr_r of so3lr_dev's
    c6_raw_potential), and d lr/dr per A, on the grid of _phi_table. Below x = 0.5, where
    the closed form cancels, the series lr = e^-x sum_k>=3 x^(k-3)/k! / (2 s^2)^3."""
    from math import factorial
    r = np.linspace(0.0, rc, n + 1)
    R = r / bohr
    t = 2.0 * smearing_bohr ** 2
    x = R * R / t
    ex = np.exp(-x)
    s3 = sum(x ** (k - 3) / factorial(k) for k in range(3, 24))
    s4 = sum(x ** (k - 4) / factorial(k) for k in range(4, 25))
    Rs = np.where(R > 0, R, 1.0)
    g = 1.0 - ex * (1.0 + x + 0.5 * x * x)
    small = x < 0.5
    lr = np.where(small, ex * s3 / t ** 3, g / Rs ** 6)
    dlr = np.where(small, -6.0 * ex * s4 * R / t ** 4, (ex * x ** 3 - 6.0 * g) / Rs ** 7)
    return lr, dlr / bohr


# =========================================================================== #
# particle-mesh Ewald: the mesh part (jaxpme's pme solver, as so3lr_dev's
# --kspace-electrostatics pme and --kspace-dispersion pme use it)
# =========================================================================== #
def pme_mesh_shape(L, spacing):
    """The PME mesh so3lr_dev builds for an orthorhombic box of sides L."""
    return kspace_grid_shape("pme", spacing, np.diag(np.asarray(L, np.float64)))


def _pme_potential(kind):
    """jaxpme's potential (lr, with k = 0 masked) and raw potential (the self and
    background corrections) for Coulomb or so3lr_dev's C6 dispersion."""
    raw = jaxpme.potentials.coulomb() if kind == "coulomb" else c6_raw_potential()
    return jaxpme.potentials.potential(custom_potential=raw), raw


def _lagrange4(u, n, dt):
    """jaxpme's 4-point Lagrange interpolation along one axis at mesh coordinates u:
    mesh indices [4, m], weights and their d/du [4, m]."""
    i0 = jnp.floor(u)
    x = (u - (i0 + 0.5)).astype(dt)
    x2 = x * x
    x3 = x2 * x
    w = jnp.stack([-3.0 + 2.0 * x + 12.0 * x2 - 8.0 * x3, 27.0 - 54.0 * x - 12.0 * x2 + 24.0 * x3,
                   27.0 + 54.0 * x - 12.0 * x2 - 24.0 * x3, -3.0 - 2.0 * x + 12.0 * x2 + 8.0 * x3]) / 48.0
    dw = jnp.stack([2.0 + 24.0 * x - 24.0 * x2, -54.0 - 24.0 * x + 72.0 * x2,
                    54.0 - 24.0 * x - 72.0 * x2, -2.0 + 24.0 * x + 24.0 * x2]) / 48.0
    idx = jnp.remainder(i0.astype(jnp.int32)[None, :] + jnp.arange(-1, 3, dtype=jnp.int32)[:, None], n)
    return idx, w, dw


def _pme_mesh(pos3, w, L, shape, Lk, kernel, dt, chunk, at=None):
    """At every atom i, phi_i = sum_m W_i(m) (K * rho)(m) / V and d phi_i / d r_i (three
    arrays; W_j alone moves with r_j): rho = sum_j w_j W_j on the mesh, K = kernel(k2)
    applied by FFT -- jaxpme's pme kspace without the corrections and the 1/2. Positions
    and L in one unit, Lk (k-vectors, V) in the kernel's; L and Lk may be traced (a box
    that changes). Atoms a chunk at a time, so what is held per atom and mesh point is
    one chunk's. Also d/de of sum_i w_i phi_i with box and positions scaled by 1 + e
    (the mesh coordinates stay, k shrinks, V grows): sum_k |rho_k|^2 K'(k) / V,
    K' = -2 k^2 dK/dk^2 - 3 K. ``at``: the positions where phi is wanted (a device's own
    atoms), if not at every atom spread."""
    n = pos3[0].shape[0]
    n0, n1, n2 = shape
    C = min(int(chunk), n)
    nch = -(-n // C)
    scale = [shape[a] / L[a] for a in range(3)]

    def coords(start, p3=pos3, C=C):
        out = [_lagrange4(lax.dynamic_slice_in_dim(p3[a], start, C) * scale[a], shape[a], dt)
               for a in range(3)]
        (ix, wx, dx), (iy, wy, dy), (iz, wz, dz) = out
        flat = (ix[:, None, None] * n1 + iy[None, :, None]) * n2 + iz[None, None, :]
        return flat, (wx, wy, wz), (dx, dy, dz)

    def spread(c, rho):
        # the last chunk is shifted back to end at n; its rows already spread add nothing
        start = jnp.minimum(c * C, n - C)
        flat, (wx, wy, wz), _ = coords(start)
        wc = lax.dynamic_slice_in_dim(w, start, C)
        wc = jnp.where(start + jnp.arange(C) >= c * C, wc, 0.0)
        vals = wc * wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        return rho.at[flat.reshape(-1)].add(vals.reshape(-1))
    rho = lax.fori_loop(0, nch, spread, jnp.zeros((n0 * n1 * n2,), dt))

    def freq(nn, lk, half=False):
        m = (np.fft.rfftfreq(nn) if half else np.fft.fftfreq(nn)) * nn
        return jnp.asarray(2.0 * np.pi * m, dt) / jnp.asarray(lk, dt)
    kx, ky, kz = freq(n0, Lk[0]), freq(n1, Lk[1]), freq(n2, Lk[2], half=True)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    rho_hat = jnp.fft.rfftn(rho.reshape(shape))
    del rho
    K, dK = jax.jvp(kernel, (k2,), (jnp.ones_like(k2),))
    inv_v = (1.0 / (jnp.asarray(Lk[0]) * Lk[1] * Lk[2])).astype(dt)
    # the half spectrum: the planes kz = 0 and n2/2 once, the others twice
    wz = jnp.asarray(np.where((np.arange(n2 // 2 + 1) == 0) | (np.arange(n2 // 2 + 1) * 2 == n2), 1.0, 2.0), dt)
    vir = (wz * (rho_hat.real ** 2 + rho_hat.imag ** 2) * (-2.0 * k2 * dK - 3.0 * K)).sum() * inv_v
    pot = jnp.fft.irfftn(rho_hat * K, s=shape, norm="forward").reshape(-1)
    at = pos3 if at is None else at
    m = at[0].shape[0]
    Cm = min(int(chunk), m)

    def gather(c, out):
        start = jnp.minimum(c * Cm, m - Cm)
        flat, (wx, wy, wz), (dx, dy, dz) = coords(start, at, Cm)
        g = pot[flat]

        def contract(a, b, c, f):
            # elementwise (an einsum would be a dot, TF32 under default precision)
            return (a[:, None, None] * b[None, :, None] * c[None, None, :] * g).sum((0, 1, 2)) * f
        f = [jnp.asarray(scale[a] * inv_v).astype(dt) for a in range(3)]
        cols = [contract(wx, wy, wz, inv_v), contract(dx, wy, wz, f[0]),
                contract(wx, dy, wz, f[1]), contract(wx, wy, dz, f[2])]
        return tuple(lax.dynamic_update_slice_in_dim(o, v, start, 0) for o, v in zip(out, cols))
    zero = jnp.zeros((m,), dt)
    return lax.fori_loop(0, -(-m // Cm), gather, (zero, zero + 0.0, zero + 0.0, zero + 0.0)) + (vir,)


def _pme_part(pos3, w, L, mesh, Lk, smearing, kind, sign, dt, chunk, at=None, w_at=None):
    """The mesh part of PME as so3lr_dev adds it (jaxpme's pme kspace with its self and
    background corrections) for weights w (charges, or sqrt(C6)), smearing and Lk in the
    potential's unit: E = sign sum_i w_i pot_i, pot = (phi + c) / 2,
    c = -w self - 2 bg sum(w) / V. The per-atom energy, dE/dw, forces (three arrays) and
    dE/d(strain), to which the background energy -sign bg W^2 / V adds as 1/V. The
    per-atom terms at positions ``at`` with weights ``w_at`` (a device's own atoms) if
    given: every device spreads all atoms."""
    pot, raw = _pme_potential(kind)
    phi, gx, gy, gz, vir = _pme_mesh(pos3, w, L, mesh, Lk, partial(pot.lr, smearing), dt, chunk, at)
    bg = raw.correction_background(smearing)
    inv_v = (1.0 / (jnp.asarray(Lk[0]) * Lk[1] * Lk[2])).astype(dt)
    W = w.sum()
    w = w if w_at is None else w_at
    pot2 = phi - raw.correction_self(smearing) * w - 2.0 * bg * W * inv_v
    dstrain = 0.5 * sign * vir + 3.0 * sign * bg * W * W * inv_v
    return 0.5 * sign * w * pot2, sign * pot2, tuple(-sign * w * g for g in (gx, gy, gz)), dstrain


def pme_electrostatics(pos3, q, L, mesh, smearing, ke, dt, chunk=1 << 20, **kw):
    """The mesh part of Coulomb PME (positions, L, smearing in A): per-atom energy, dE/dq,
    forces, dE/d(strain). kw: a device's share (``_pme_part``)."""
    return _pme_part(pos3, q, L, mesh, L, smearing, "coulomb", ke, dt, chunk, **kw)


def pme_dispersion(pos3, qd, L, mesh, smearing, bohr, hartree, dt, chunk=1 << 20, **kw):
    """The mesh part of C6 dispersion PME, in atomic units (qd = sqrt(C6); positions, L,
    smearing in A): per-atom energy, dE/dqd, forces, dE/d(strain)."""
    return _pme_part(pos3, qd, L, mesh, [v / bohr for v in L], smearing / bohr, "c6", -hartree, dt, chunk,
                     **kw)


# =========================================================================== #
# the engine
# =========================================================================== #
# the constants so3lr_dev uses (ase.units)
ASE_UNITS = dict(ke=14.399645351950548, hartree=27.211386024367243, bohr=0.5291772105638411)
TABLE_INTERVALS = 4096       # radial filter tables on [0, 1.05 cutoff]
PAIR_INTERVALS = 16384       # long-range pair tables on [0, lr cutoff]
PME_CHUNK = 1 << 20          # atoms spread or gathered at a time in the PME mesh


# memory modes, fastest first: (engine options, staged, peak KiB per atom of so3lr-tauri-m on an
# H100 in water at 10-20M atoms; us/atom per force call). fast keeps each layer's input and
# attention output; default keeps the same but runs as separately jitted stages from
# Python (energy_forces_staged: a jitted MD step cannot call them, but they keep XLA
# from holding work across stages); lean recomputes the attention forward in the
# backward; leanest also recomputes earlier layers and runs the attention backward a
# channel at a time; minimal runs the attention one head's block of features at a time.
MODES = {
    "fast": (dict(keep_attention=True), False, 9.0),          # 1.23 us/atom
    "default": (dict(keep_attention=True), True, 6.6),        # 1.24
    "lean": (dict(), True, 5.3),                              # 1.42
    "leanest": (dict(keep_layer_inputs=False, split_channels=True, tile=65536), True, 4.1),  # 1.66
    # the attention a block at a time (head b and degree entry b read only block b)
    "minimal": (dict(keep_layer_inputs=False, split_blocks=True, tile=65536), True, 3.0),    # 3.06
}


def mode_kib(mode, statics):
    """MODES' peak KiB per atom for this model: what a mode holds per atom scales with the
    features and equivariant channels (so3lr-tauri-m: 128 + 24)."""
    width = statics["num_features"] + sum(2 * int(d) + 1 for d in statics["degrees"])
    return MODES[mode][2] * width / 152.0


def pick_mode(n, free_bytes, statics):
    """The fastest of MODES whose peak for n atoms fits in free_bytes, else the leanest."""
    for name in MODES:
        if n * mode_kib(name, statics) * 1024 * 1.03 + (1 << 30) <= free_bytes:
            return name
    return name


def pme_bytes(L, pme, itemsize):
    """Memory the mesh part of PME holds at its peak, for (smearing, spacing) pairs pme:
    the charges, their transform, the filtered transform, the potential and the FFT's
    work space, about 5 meshes' worth, for the larger mesh (one is gone before the
    other starts)."""
    return max((5 * math.prod(pme_mesh_shape(L, p[1])) * itemsize for p in pme if p), default=0)


class FlashSO3LR:
    """Energy and forces of SO3LR (params, statics from ``loader.load_params``) for atoms
    ``z`` in an orthorhombic periodic ``box``, with so3lr_dev's descriptor floor at
    contacts. The memory options are those of ``MODES``."""

    def __init__(self, params, statics, z, box, *, total_charge=0.0, total_spin=0.0,
                 equilibration_charge=None, tile=1 << 17, max_neighbours=None, lr_scratch=1024,
                 keep_layer_inputs=True, keep_attention=False, split_channels=False,
                 split_blocks=False, pme_electrostatics=None, pme_dispersion=None, box_margin=0.0,
                 devices=1, atoms_per_program=None):
        degrees = [int(d) for d in statics["degrees"]]
        self.nh = int(statics["num_heads"])
        # one head per degree entry, each on its own block of G features (so3lr-tauri-s/m: 4 x 32,
        # so3lr-tauri-l: 8 x 32, degrees 1-4 each twice); harmonics up to degree 4
        if self.nh != len(degrees) or not set(degrees) <= {1, 2, 3, 4}:
            raise ValueError(f"written for one head per degree entry, degrees 1-4 (got {self.nh} heads, "
                             f"degrees {degrees})")
        self.params, self.statics = params, statics
        self.dtype = statics.get("dtype", jnp.float32)
        z = np.asarray(z).astype(np.int32)
        self.n = n = int(z.shape[0])
        box = np.asarray(box, np.float64).reshape(3, 3)
        if np.abs(box - np.diag(np.diag(box))).max() > 0:
            raise ValueError("needs an orthorhombic box")
        self.rc = float(statics["cutoff"])
        self.rc_lr = float(statics["long_range_cutoff"])
        # the box: a traced argument of every jitted program (so an NPT step compiles
        # nothing); box_margin: the fraction by which it may shrink or grow before the
        # cell grid is laid out again (set_box)
        self.box_margin = float(box_margin)
        self._Lt = None
        # the devices: each computes nl rows from r0 on, the last device's shifted back to
        # end at n (its first rows are the previous device's too, counted there)
        self.P = int(devices)
        devs = jax.local_devices()[:max(self.P, 0)]
        if not 1 <= self.P <= len(devs):
            raise ValueError(f"devices={devices}: 1 to {len(jax.local_devices())} here")
        self.nl = -(-n // self.P)
        if self.P > 1 and (self.P - 1) * self.nl >= n:
            raise ValueError(f"{n} atoms are too few for {self.P} devices")
        self.device_mesh = jax.sharding.Mesh(np.array(devs), (AXIS,)) if self.P > 1 else None
        self.tile = min(int(tile), self.nl)
        # neighbour rows sized for water at the model's cutoff (widened on overflow)
        self.K = int(max_neighbours or math.ceil(64 * (self.rc / 4.5) ** 3 / 16) * 16)
        self.keep = bool(keep_layer_inputs)
        self.keep_attention = bool(keep_attention) and self.keep
        self.split = bool(split_channels)
        self.blocks = bool(split_blocks)
        # the charge embedding sees total_charge, the charge equilibration conserves
        # equilibration_charge (so3lr_dev's two overrides; both follow total_charge by default)
        self.total_charge = float(total_charge if equilibration_charge is None else equilibration_charge)
        dt = self.dtype
        self.F = int(params["feature_embeddings_0"]["Embed_0"]["embedding"].shape[1])
        self.G = self.F // self.nh
        if self.G * self.nh != self.F or self.G & (self.G - 1):
            raise ValueError(f"{self.F} features over {self.nh} heads: needs a power-of-two block per head")
        mdeg = [i for i, d in enumerate(degrees) for _ in range(2 * d + 1)]
        self.M = len(mdeg)
        self.MP = max(32, 1 << (self.M - 1).bit_length())     # the kernels' power-of-two width
        self.mdeg = jnp.asarray(mdeg + [-1] * (self.MP - self.M), jnp.int32)
        # where each degree entry's columns start, and its degree's first harmonic component
        starts = np.cumsum([0] + [2 * d + 1 for d in degrees])[:-1]
        self.ys = tuple((int(c), d * d - 1) for c, d in zip(starts, degrees))
        self.cdeg = jnp.asarray([(2 * d + 1) ** -0.5 for d in degrees], dt)
        # the attention kernels' launch for this width of the spherical filters' hidden
        # layer: the nearest wider one tabulated (the widest if none is)
        sh = int(params["layers_0"]["attention_block"]["spherical_filter1_layer_1"]["kernel"].shape[1])
        self.launch = _att_launch(sh)
        self.par = jnp.asarray([(-1.0) ** d for d in degrees], dt)
        self.z = jnp.asarray(z)
        # x0 depends on the element only: one row per element present
        zu, inv = np.unique(z, return_inverse=True)
        counts = jnp.asarray(np.bincount(inv), dt)
        self.elem = jnp.asarray(inv.astype(np.int32))
        self.x0_table = self._embed_table(jnp.asarray(zu), counts, total_charge, total_spin)
        # radial tables per layer, both filters
        r_max = 1.05 * self.rc
        grid = np.linspace(0.0, r_max, TABLE_INTERVALS + 1)
        self.tables = []
        for li in range(statics["num_layers"]):
            p = params[f"layers_{li}"]["attention_block"]
            t = []
            for f in ("radial_filter1", "radial_filter2"):
                v, d = _radial_table(p, (f + "_layer_1", f + "_layer_2"), statics["num_radial_basis_fn"], grid)
                t += [jnp.asarray(v, dt), jnp.asarray(d, dt)]
            self.tables.append(t)
        self.scal = jnp.asarray([1.0 / statics["avg_num_neighbors"], r_max / TABLE_INTERVALS,
                                 TABLE_INTERVALS, self.rc], dt)
        self.sr_scal = self.scal.at[0].set(ASE_UNITS["ke"])
        self.geo = jnp.asarray([self.rc ** 2], dt)
        self.nlh_a = jnp.asarray(params["nlh_aa"], dt).reshape(-1)
        self.nlh_b = jnp.asarray(params["nlh_bb"], dt).reshape(-1)
        self.zmax = int(params["nlh_aa"].shape[0])
        self.nk = int(params["nlh_aa"].shape[-1])
        self.rmin = jnp.asarray(nlh_reference_distance(), dt).reshape(-1)     # so3lr_dev's floor
        # long range
        sig = float(statics["electrostatic_energy_scale"])
        self.elec = sig != 0.0
        self.disp = float(statics["dispersion_energy_scale"]) != 0.0
        # particle-mesh Ewald, (smearing, spacing) in A each: the real-space pair terms
        # lose what the mesh adds back (_pme_terms)
        self.pme_el = tuple(float(v) for v in pme_electrostatics) if pme_electrostatics and self.elec else None
        self.pme_disp = tuple(float(v) for v in pme_dispersion) if pme_dispersion and self.disp else None
        self._layout([float(v) for v in np.diag(box)])
        phi, dphi, phs = _phi_table(self.rc_lr, sig if self.elec else 1.0, PAIR_INTERVALS,
                                    smearing=self.pme_el[0] if self.pme_el else None)
        self.phi_v, self.phi_d = jnp.asarray(phi, dt), jnp.asarray(dphi, dt)
        if self.pme_disp:
            lr6, dlr6 = _c6_lr_table(self.rc_lr, self.pme_disp[0] / ASE_UNITS["bohr"], PAIR_INTERVALS,
                                     ASE_UNITS["bohr"])
            self.c6lr_v, self.c6lr_d = jnp.asarray(lr6, dt), jnp.asarray(dlr6, dt)
        else:
            self.c6lr_v = self.c6lr_d = jnp.zeros((2,), dt)
        self.lrs = jnp.asarray([self.rc_lr, phs, PAIR_INTERVALS, ASE_UNITS["ke"],
                                self.rc_lr - statics["dispersion_energy_cutoff_lr_damping"],
                                statics["dispersion_energy_scale"], float(statics["fine_structure"]) ** (-4.0 / 21.0),
                                ASE_UNITS["hartree"], ASE_UNITS["bohr"]], dt)
        self.alphas = jnp.asarray(statics["dispersion_alphas"], dt)
        self.c6tab = jnp.asarray(statics["dispersion_c6"], dt)
        # the long-range kernel's resident programs: 16 per streaming multiprocessor
        self.lr_programs = 16 * (getattr(jax.devices()[0], "core_count", None) or 132)
        self.maxh = int(lr_scratch)
        self.atoms_per_program = atoms_per_program
        self._build_jit()

    def _layout(self, L):
        """What depends on the box beyond its value: the cell grid's cells per side and
        long-range reach, laid out for the box shrunk by box_margin (so it holds for
        any box within the margin), and the PME meshes (fixed between layouts, as
        so3lr_dev keeps its k-grid)."""
        L = [float(v) for v in L]
        small = [v * (1.0 - self.box_margin) for v in L]
        if box_problem(small, self.rc, self.rc_lr):
            small = L                     # near the smallest box: no room to shrink in this layout
        problem = box_problem(small, self.rc, self.rc_lr)
        if problem:
            raise ValueError(problem)
        self.L, self.L_lo = L, small
        self.box = jnp.asarray(L, jnp.float64 if jax.config.jax_enable_x64 else self.dtype)
        self.nc, self.kc = cell_grid(small, self.rc, self.rc_lr)
        self.mesh_el = pme_mesh_shape(L, self.pme_el[1]) if self.pme_el else None
        self.mesh_disp = pme_mesh_shape(L, self.pme_disp[1]) if self.pme_disp else None
        for m in (self.mesh_el, self.mesh_disp):
            if m and math.prod(m) >= 2 ** 31:
                raise ValueError(f"PME mesh {m} has 2^31 points or more: a larger spacing")

    # ---- the embedding, per element -----------------------------------------
    def _embed_table(self, zu, counts, total_charge, total_spin):
        p = self.params
        dt = self.dtype

        def cs(pp, psi):
            psi = jnp.asarray(psi, dt)
            qv = pp["Embed_0"]["embedding"][zu]
            bucket = (psi < 0.0).astype(jnp.int32)
            k, v = pp["Embed_1"]["embedding"][bucket], pp["Embed_2"]["embedding"][bucket]
            y = jax.nn.softplus((qv * k).sum(-1) / jnp.sqrt(float(qv.shape[-1])))
            a = psi * y / jnp.sum(counts * y)
            x = a[:, None] * v
            return x + blocks.mlp(pp["Residual_0"], ("layers_0", "layers_1"), jax.nn.silu(x),
                              use_bias=False, final_use_bias=False)
        e = jnp.stack([p["feature_embeddings_0"]["Embed_0"]["embedding"][zu],
                       cs(p["feature_embeddings_1"]["ChargeSpinEmbedSparse_0"], total_charge),
                       cs(p["feature_embeddings_2"]["ChargeSpinEmbedSparse_0"], total_spin)], -1)
        return e.sum(-1) / jnp.sqrt(3.0)

    # ---- per-atom functions ------------------------------------------------------
    def _x_in(self, li, x_tile, elem_tile):
        x0 = self.x0_table[elem_tile]
        return self.params["resid_lambdas"][li] * (x0 if x_tile is None else x_tile) + \
            self.params["x0_lambdas"][li] * x0

    def _project(self, li, x_in):
        p = self.params[f"layers_{li}"]["attention_block"]
        n = x_in.shape[0]
        nh = self.statics["num_heads"]
        nd = len(self.statics["degrees"])
        x1 = x_in.reshape(n, nh, -1)
        x2 = x_in.reshape(n, nd, -1)
        out = [jnp.einsum("hij,nhj->nhi", p["Wq1"], x1), jnp.einsum("hij,nhj->nhi", p["Wk1"], x1),
               jnp.einsum("hij,nhj->nhi", p["Wv1"], x1), jnp.einsum("hij,nhj->nhi", p["Wq2"], x2),
               jnp.einsum("hij,nhj->nhi", p["Wk2"], x2)]
        if self.statics["qk_norm"]:
            for i in (0, 1, 3, 4):
                out[i] = blocks.functional_rms_norm(out[i])
        return [o.reshape(n, -1) for o in out]

    def _project_block(self, li, xb, b):
        """Block b of q1, k1, v1, q2, k2 from block b of the layer input: the projections
        are per head and per degree entry, and head b and degree entry b see the same G
        features."""
        p = self.params[f"layers_{li}"]["attention_block"]
        out = [xb @ p[w][b].T for w in ("Wq1", "Wk1", "Wv1", "Wq2", "Wk2")]
        if self.statics["qk_norm"]:
            for i in (0, 1, 3, 4):
                out[i] = blocks.functional_rms_norm(out[i])
        return out

    def _post(self, li, x1, ev1):
        p = self.params[f"layers_{li}"]
        x = blocks.functional_rms_norm(x1) if self.statics["ln1"] else x1
        x = x + blocks.mlp(p, ("res_mlp_1_layer_1", "res_mlp_1_layer_2"), jax.nn.silu(x))
        su, eu = blocks.scalar_equivariant_exchange(p["exchange_block"], x, ev1, degrees=self.statics["degrees"])
        x = x + su
        ev = ev1 + eu
        if self.statics["ln2"]:
            x = blocks.functional_rms_norm(x)
        return x, ev

    def _heads(self, x, z):
        p = self.params
        ep = p["observables_0"]
        e_loc = blocks.local_node_energy(ep, x, z, head_index=self.statics["head_index"])
        pc = ep["electrostatic_energy"]["partial_charges"]
        raw_q = (blocks.mlp(pc, ("charge_dense_regression_vec", "charge_dense_final_vec"), x).squeeze(-1)
                 + pc["Embed_0"]["embedding"][z].squeeze(-1))
        a0 = blocks.ratio_head(p["observables_2"], ("hirshfeld_ratios_dense_regression",
                                                "hirshfeld_ratios_dense_final"), x, z)
        c6 = blocks.ratio_head(p["observables_3"], ("c6_ratios_dense_regression",
                                                "c6_ratios_dense_final"), x, z)
        return e_loc, raw_q, a0, c6

    # ---- tiles --------------------------------------------------------------------
    def _tile_loop(self, body, carry):
        """body(start, first_new, carry) over tiles of this device's rows; the last tile is
        shifted back to end at its last row, and first_new marks where its new atoms begin."""
        T, n, r0 = self.tile, self.nl, self._r0
        nt = -(-n // T)

        def step(t, c):
            start = jnp.minimum(t * T, n - T)
            return body(r0 + start, r0 + t * T, c)
        return lax.fori_loop(0, nt, step, carry)

    # ---- several devices: each one's rows, and the exchange between stages ---------------
    @property
    def _r0(self):
        """The first row this device computes (traced inside the per-device program)."""
        if self.P == 1:
            return 0
        return jnp.minimum(lax.axis_index(AXIS) * self.nl, self.n - self.nl)

    def _row_arg(self):
        """The first row this device computes, as the kernels take it."""
        return jnp.reshape(jnp.asarray(self._r0, jnp.int32), (1,))

    def _row_grid(self, warps):
        """(programs, atoms per program) of a row kernel: one atom each, or on ROCm as many as
        keep it under 2^32 threads (atoms_per_program: forced, for tests)."""
        app = self.atoms_per_program or (
            max(1, -(-self.nl // ((2 ** 32 - 1) // (warps * 64) - 1))) if is_rocm() else 1)
        return -(-self.nl // app), app

    def _mine(self, a):
        """This device's rows of a (all of a on one device)."""
        return a if self.P == 1 else lax.dynamic_slice_in_dim(a, self._r0, self.nl)

    def _add_mine(self, a, v):
        """a with v added to this device's rows."""
        if self.P == 1:
            return a + v
        return lax.dynamic_update_slice_in_dim(a, (self._mine(a) + v).astype(a.dtype), self._r0, 0)

    def _counted(self, v):
        """v (this device's rows) with the rows the previous device also computes zeroed,
        so that summed over the devices each atom counts once."""
        if self.P == 1:
            return v
        i = self._r0 + jnp.arange(self.nl)
        return jnp.where((i >= lax.axis_index(AXIS) * self.nl).reshape((-1,) + (1,) * (v.ndim - 1)), v, 0)

    def _gather_rows(self, *rows):
        """Full arrays from each device's rows (all-gather)."""
        if self.P == 1:
            return rows
        extra = self.P * self.nl - self.n            # rows the last device shares
        cut = (self.P - 1) * self.nl
        out = []
        for v in rows:
            g = lax.all_gather(v, AXIS, axis=0, tiled=True)
            if extra:
                g = jnp.concatenate([g[:cut], g[cut + extra:]], 0)
            out.append(g)
        return tuple(out)

    def _gather(self, *arrays):
        """The arrays with every device's rows, everywhere (each device holds valid rows only
        where it computed them)."""
        return self._gather_rows(*map(self._mine, arrays))

    def _sum(self, v):
        return v if self.P == 1 else lax.psum(v, AXIS)

    def _rows_mask(self, start, first_new):
        return (start + jnp.arange(self.tile) >= first_new)[:, None]

    def _map_rows(self, fn, arrays, outs):
        """outs[i][rows] = fn(*arrays[rows])[i] for every tile (new rows only). An input
        that is also an output is read from the loop carry, so the update is in place."""
        T = self.tile
        alias = [next((k for k, o in enumerate(outs) if o is a), None) for a in arrays]

        def body(start, first_new, outs):
            src = [outs[k] if k is not None else a for a, k in zip(arrays, alias)]
            tiles = [lax.dynamic_slice_in_dim(a, start, T) if a is not None else None for a in src]
            res = fn(start, *tiles)
            keep = self._rows_mask(start, first_new)
            new = []
            for o, r in zip(outs, res):
                old = lax.dynamic_slice_in_dim(o, start, T)
                k = keep if o.ndim == 2 else keep[:, 0]
                new.append(lax.dynamic_update_slice_in_dim(o, jnp.where(k, r.astype(o.dtype), old), start, 0))
            return tuple(new)
        return self._tile_loop(body, tuple(outs))

    # ---- kernels -------------------------------------------------------------------
    def _passes(self):
        """(channel, projections, first block, blocks) per attention pass: all at once;
        (split) one channel at a time; (blocks) one G-feature block at a time, head b
        and degree entry b together: both read only block b of the layer input."""
        nh = self.nh
        if self.blocks:
            return [(0, (0, 1, 2, 3, 4), b, 1) for b in range(nh)]
        if self.split:
            return [(1, (0, 1, 2), 0, nh), (2, (3, 4), 0, nh)]
        return [(0, (0, 1, 2, 3, 4), 0, nh)]

    def _projections(self, li, x_l, which, block=None):
        """Layer li's projections in ``which`` (None elsewhere), on whole arrays: without an
        autograd tape XLA fuses them with bounded temporaries. (Tile loops writing into
        zero-initialised outputs were worse: XLA hoists every zeros() to the start.) With
        ``block``, that G-feature block of each only."""
        xi = self._x_in(li, x_l, self.elem)
        if block is None:
            out = self._project(li, xi)
        else:
            G = self.G
            out = self._project_block(li, xi[:, block * G:(block + 1) * G], block)
        return [out[k] if k in which else None for k in range(5)]

    def _attend(self, li, x_l, ev_l, pos3, nbr, cnt):
        """(x_in + dx, ev + dev) for layer li."""
        n, F, dt = self.n, self.F, self.dtype
        p = self.params[f"layers_{li}"]["attention_block"]
        t1v, t1d, t2v, t2d = self.tables[li]
        sf = lambda name, k: p[f"{name}_layer_{k}"]
        px, py, pz = pos3
        x1, ev1 = None, ev_l + 0.0
        for ch, which, h0, nh in self._passes():
            blk = h0 if self.blocks else None
            if x1 is None:
                qkv = self._projections(li, x_l, which, block=blk)
                # x_in again, as its own buffer for the kernel to add into: aliasing the one
                # the projections read makes XLA insert a copy
                x1 = self._x_in(li, x_l, self.elem)
            else:
                # the next pass's projections only after the previous pass (XLA would make
                # them all up front)
                x_l, x1, ev1 = lax.optimization_barrier((x_l, x1, ev1))
                qkv = self._projections(li, x_l, which, block=blk)
            ref = qkv[which[0]]
            be, warps = self.launch["fwd"]
            progs, app = self._row_grid(warps)
            kern = partial(_att_fwd_kernel, G=self.G, ND=self.nh, MP=self.MP, ZMAX=self.zmax, YS=self.ys,
                           CH=ch, BE=be, H0=h0, NH=nh, N=self.nl, APP=app)
            x1, ev1 = pl.pallas_call(
                kern, grid=(progs,), compiler_params=_cp(warps),
                out_shape=(jax.ShapeDtypeStruct((n, F), dt), jax.ShapeDtypeStruct((n, self.M), dt)),
                input_output_aliases={28: 0, 29: 1},
            )(*[q if q is not None else ref for q in qkv], ev_l, px, py, pz, nbr, cnt, self.z,
              self.rmin, t1v, t1d, t2v, t2d,
              sf("spherical_filter1", 1)["kernel"], sf("spherical_filter1", 1)["bias"],
              sf("spherical_filter1", 2)["kernel"], sf("spherical_filter1", 2)["bias"],
              sf("spherical_filter2", 1)["kernel"], sf("spherical_filter2", 1)["bias"],
              sf("spherical_filter2", 2)["kernel"], sf("spherical_filter2", 2)["bias"],
              self.mdeg, self.cdeg, self.scal, x1, ev1, self._box(px), self._row_arg())
        return x1, ev1

    def _attention_backward(self, li, x_l, ev, pos3, nbr, cnt, gdx, gdev, g_ev, f3):
        """(dE/dx_in, dE/dev, forces) after layer li's attention, a tile at a time; the
        projections are made here, one channel at a time if split."""
        # dE/dx_in = gdx + the projections' part. Unsplit it overwrites x_l in place (each
        # tile reads its rows of x_l before writing them; nothing reads x_l after); split,
        # the second pass still projects from x_l, so it gets its own buffer. Layer 0's
        # input is the constant embedding: no projection gradient at all.
        carry = (self._g_xin_start(li, x_l, gdx), g_ev, *f3)
        for pi in range(len(self._passes())):
            # not before the previous pass is done (XLA would make both passes' arrays early)
            x_l, carry = lax.optimization_barrier((x_l, carry))
            carry = self._attention_backward_pass(li, pi, None if self._inplace(li) else x_l,
                                                  ev, pos3, nbr, cnt, gdx, gdev, carry)
        return carry

    def _inplace(self, li):
        """dE/dx_in overwrites x_l: unsplit (x_l is dead once projected) or a block at a time
        (each pass projects from its own block only)."""
        return li > 0 and (self.blocks or not self.split)

    def _g_xin_start(self, li, x_l, gdx):
        return x_l if self._inplace(li) else (gdx + 0.0 if li > 0 else jnp.zeros((), self.dtype))

    def _attention_backward_pass(self, li, pi, x_l, ev, pos3, nbr, cnt, gdx, gdev, carry):
        """One pass (channel) of _attention_backward. carry = (dE/dx_in, dE/dev, fx, fy,
        fz, fw: the per-atom virial); in place (x_l None), carry[0] holds x_l until each
        tile overwrites it."""
        n, F, dt, T = self.n, self.F, self.dtype, self.tile
        p = self.params[f"layers_{li}"]["attention_block"]
        t1v, t1d, t2v, t2d = self.tables[li]
        sf = lambda name, k: p[f"{name}_layer_{k}"]
        px, py, pz = pos3
        meta = dict(G=self.G, ND=self.nh, MP=self.MP, ZMAX=self.zmax)
        shp = jax.ShapeDtypeStruct
        inplace = self._inplace(li)
        ch, which, h0, nh = self._passes()[pi]
        blk = h0 if self.blocks else None
        FP = self.G * nh                               # width of the projection arrays
        cols = slice(0, F) if blk is None else slice(blk * FP, (blk + 1) * FP)
        q1, k1, v1, q2, k2 = self._projections(li, carry[0] if inplace else x_l, which, block=blk)

        def body(start, first_new, carry):
            g_xin, g_ev, fx, fy, fz, fw = carry
            a0 = jnp.stack([start, first_new]).astype(jnp.int32)
            g = {}
            if ch != 2:
                be, warps = self.launch["bwd_heads"]
                g[0], g[1], g[2], g_ev, fx, fy, fz, fw = pl.pallas_call(
                    partial(_att_bwd_heads_kernel, BE=be, H0=h0, NH=nh, **meta),
                    grid=(T,), compiler_params=_cp(warps),
                    out_shape=(shp((T, FP), dt), shp((T, FP), dt), shp((T, FP), dt), shp((n, self.M), dt),
                               shp((n,), dt), shp((n,), dt), shp((n,), dt), shp((n,), dt)),
                    input_output_aliases={22: 3, 23: 4, 24: 5, 25: 6, 26: 7},
                )(q1, k1, v1, ev, px, py, pz, nbr, cnt, self.z, self.rmin, t1v, t1d,
                  sf("spherical_filter1", 1)["kernel"], sf("spherical_filter1", 1)["bias"],
                  sf("spherical_filter1", 2)["kernel"], sf("spherical_filter1", 2)["bias"],
                  self.mdeg, self.cdeg, gdx, self.scal, a0, g_ev, fx, fy, fz, fw, self._box(px))
            if ch != 1:
                be, warps = self.launch["bwd_degrees"]
                g[3], g[4], g_ev, fx, fy, fz, fw = pl.pallas_call(
                    partial(_att_bwd_degrees_kernel, BE=be, H0=h0, NH=nh, YS=self.ys, **meta),
                    grid=(T,), compiler_params=_cp(warps),
                    out_shape=(shp((T, FP), dt), shp((T, FP), dt), shp((n, self.M), dt),
                               shp((n,), dt), shp((n,), dt), shp((n,), dt), shp((n,), dt)),
                    input_output_aliases={22: 2, 23: 3, 24: 4, 25: 5, 26: 6},
                )(q2, k2, ev, px, py, pz, nbr, cnt, self.z, self.rmin, t2v, t2d,
                  sf("spherical_filter2", 1)["kernel"], sf("spherical_filter2", 1)["bias"],
                  sf("spherical_filter2", 2)["kernel"], sf("spherical_filter2", 2)["bias"],
                  self.mdeg, self.cdeg, self.par, gdev, self.scal, a0, g_ev, fx, fy, fz, fw, self._box(px))
            if li > 0:
                rows = lax.dynamic_slice_in_dim(g_xin if inplace else x_l, start, T)
                xt = self._x_in(li, rows, lax.dynamic_slice_in_dim(self.elem, start, T))[:, cols]
                if blk is None:
                    _, vjp = jax.vjp(lambda xx: [self._project(li, xx)[k] for k in which], xt)
                else:
                    _, vjp = jax.vjp(lambda xx: [self._project_block(li, xx, blk)[k] for k in which], xt)
                (gx,) = vjp([g[k] for k in which])
                keep = self._rows_mask(start, first_new)
                old = lax.dynamic_slice_in_dim(g_xin, start, T)
                gdx_t = lax.dynamic_slice_in_dim(gdx, start, T)
                if blk is None:
                    new = (gdx_t if inplace else old) + gx
                else:
                    # this block's columns only: the others are other passes'
                    new = old.at[:, cols].set(gdx_t[:, cols] + gx)
                g_xin = lax.dynamic_update_slice_in_dim(g_xin, jnp.where(keep, new, old), start, 0)
            return g_xin, g_ev, fx, fy, fz, fw
        return self._tile_loop(body, carry)

    def build_grid(self, pos3):
        """The cell grid: atoms in cell order, where each cell starts, each atom's cell."""
        nx, ny, nz = self.nc
        px = pos3[0]
        L = self._box(px)
        frac = jnp.stack(pos3, -1) / L
        frac = frac - jnp.floor(frac)
        cell = jnp.clip((frac * jnp.asarray(self.nc, self.dtype)).astype(jnp.int32), 0,
                        jnp.asarray(self.nc, jnp.int32) - 1)
        cell_of = (cell[:, 2] * ny + cell[:, 1]) * nx + cell[:, 0]
        order = jnp.argsort(cell_of).astype(jnp.int32)
        cell_start = jnp.searchsorted(cell_of[order], jnp.arange(nx * ny * nz + 1, dtype=jnp.int32)).astype(jnp.int32)
        return order, cell_start, cell_of

    def build(self, pos3):
        n, (nx, ny, nz) = self.n, self.nc
        px, py, pz = pos3
        order, cell_start, cell_of = self.build_grid(pos3)
        # rows of other devices' atoms are left unwritten (never read here)
        progs, app = self._row_grid(1)
        # one warp, one wavefront on ROCm (as measured)
        nbr, cnt = pl.pallas_call(
            partial(_ell_kernel, NX=nx, NY=ny, NZ=nz, K=self.K, BLE=16, N=self.nl, APP=app), grid=(progs,),
            compiler_params=_cp(1),
            out_shape=(jax.ShapeDtypeStruct((n, self.K), jnp.int32), jax.ShapeDtypeStruct((n,), jnp.int32)),
        )(px, py, pz, order, cell_start, cell_of, self.geo, self._box(px), self._row_arg())
        return order, cell_start, cell_of, nbr, cnt

    # ---- energy and forces ---------------------------------------------------------
    # ---- the pieces of one force call ---------------------------------------------
    def _pos3(self, R):
        """x, y, z columns: float64 positions stay float64 (with a float32 model: mixed
        precision, the displacements formed in float64), others become the model dtype."""
        R = R if R.dtype == jnp.float64 else R.astype(self.dtype)
        return (R[:, 0], R[:, 1], R[:, 2])

    def _box(self, px):
        """The box: traced inside the jitted programs (an argument of each), so a new box
        (set_box) needs no compile."""
        return (self.box if self._Lt is None else self._Lt).astype(px.dtype)

    def _layer_forward(self, li, x, ev, pos3, nbr, cnt):
        """Layer li: (x', ev'), plus its attention output with keep_attention."""
        x1, ev1 = self._attend(li, x, ev, pos3, nbr, cnt)
        kept = (x1, ev1) if self.keep_attention else ()

        # per atom, in tiles, in place (same shapes in and out, so no fresh buffers):
        # whole-array, its norms and MLP hidden layers are live together
        def post(start, x_t, e_t):
            return self._post(li, x_t, e_t)
        x2, ev2 = self._map_rows(post, [x1, ev1], [x1, ev1])
        if li < self.statics["num_layers"] - 1:
            x2, ev2 = self._gather(x2, ev2)          # the next layer reads the neighbours' rows
        return (x2, ev2) + kept

    def _head(self, x, pos3, nb):
        """Repulsion, heads, long range and the heads' backward: (energy, dE/dx, forces so
        far, most long-range partners of any atom)."""
        e_atom, raw_q, a0, c6, f3 = self._head_local(x, pos3, nb)
        q = raw_q + (self.total_charge - raw_q.sum()) / self.n
        e_atom, gq, ga0, gc6, f3, most = self._long_range(pos3, nb[:3], q, a0, c6, e_atom, f3)
        if self.pme_el or self.pme_disp:
            e_atom, gq, gc6, f3 = self._pme_terms(pos3, q, c6, e_atom, gq, gc6, f3)
        px = pos3[0]
        # summed in float64 with float64 positions (a float32 total of 10^7 atoms rounds to eV)
        e_atom = self._counted(self._mine(e_atom))
        energy = self._sum(e_atom.astype(jnp.promote_types(e_atom.dtype, px.dtype)).sum())
        g_mean = self._sum(self._counted(self._mine(gq)).sum()) / self.n
        g_x = self._head_backward(x, gq - g_mean, ga0, gc6)
        return energy, g_x, f3, most

    def _head_local(self, x, pos3, nb):
        """Repulsion and heads: per-atom energy without the long range, raw charge, a0 and
        c6 ratios (every atom's: the long range reads the neighbours'), and the repulsion's
        forces."""
        n, dt = self.n, self.dtype
        nbr, cnt = nb[3], nb[4]
        px, py, pz = pos3
        zero = jnp.zeros((n,), dt)
        shp = jax.ShapeDtypeStruct
        nkp = int(2 ** math.ceil(math.log2(self.nk)))
        warps = _warps(NUM_WARPS)
        progs, app = self._row_grid(warps)
        e_atom, fx, fy, fz, fw = pl.pallas_call(
            partial(_sr_kernel, ZMAX=self.zmax, NK=self.nk, NKP=nkp, N=self.nl, APP=app),
            grid=(progs,), compiler_params=_cp(warps),
            out_shape=(shp((n,), dt),) * 5,
            input_output_aliases={9: 1, 10: 2, 11: 3, 12: 4},
        )(px, py, pz, self.z, nbr, cnt, self.nlh_a, self.nlh_b, self.sr_scal, zero, zero + 0.0, zero + 0.0,
          zero + 0.0, self._box(px), self._row_arg())
        e_loc, raw_q, a0, c6 = self._heads(self._mine(x), self._mine(self.z))
        raw_q, a0, c6 = self._gather_rows(raw_q, a0, c6)
        return self._add_mine(e_atom, e_loc), raw_q, a0, c6, (fx, fy, fz, fw)

    def _long_range(self, pos3, grid, q, a0, c6, e_atom, f3):
        """Electrostatics and dispersion over the cell grid: e_atom and forces added to,
        dE/d(q, a0, c6) per atom, the most partners of any atom."""
        n, dt = self.n, self.dtype
        order, cell_start, cell_of = grid
        px, py, pz = pos3
        shp = jax.ShapeDtypeStruct
        # two warps, two wavefronts on ROCm (as measured)
        nprog = min(self.lr_programs, self.nl)
        scratch = jnp.zeros((nprog, self.maxh), jnp.int32)
        e_atom, gq, ga0, gc6, fx, fy, fz, fw, _, most = pl.pallas_call(
            partial(_lr_kernel, N=self.nl, NX=self.nc[0], NY=self.nc[1], NZ=self.nc[2], KC=self.kc,
                    MAXH=self.maxh, ELEC=self.elec, DISP=self.disp, DPME=self.pme_disp is not None),
            grid=(nprog,), compiler_params=_cp(2),
            out_shape=(shp((n,), dt),) * 8 + (shp((nprog, self.maxh), jnp.int32), shp((nprog,), jnp.int32)),
            input_output_aliases={15: 0, 16: 4, 17: 5, 18: 6, 19: 7, 20: 8},
        )(px, py, pz, self.z, q, a0, c6, self.alphas, self.c6tab, order, cell_start, cell_of,
          self.phi_v, self.phi_d, self.lrs, e_atom, *f3, scratch, self._box(px), self.c6lr_v, self.c6lr_d,
          self._row_arg())
        return e_atom, gq, ga0, gc6, (fx, fy, fz, fw), jnp.max(most)

    def _pme_terms(self, pos3, q, c6, e_atom, gq, gc6, f3):
        """The mesh parts of PME: energies added to e_atom, dE/dq and dE/dc6 (c6 ratios)
        to gq and gc6, forces to f3."""
        dt = self.dtype
        f3 = list(f3)
        L = self._box(pos3[0])
        mine, add_mine = self._mine, self._add_mine
        # on several devices each spreads every atom (the charges are everywhere: no mesh to
        # sum) and takes the potential at its own
        rows = [mine(p) for p in pos3]
        share = (lambda w: dict(at=rows, w_at=mine(w))) if self.P > 1 else (lambda w: {})

        def add(f3, f, w):
            # the mesh's dU/d(strain) is one number: on atom 0's row of the per-atom virial,
            # which counts half (the pair terms count every pair from both ends); atom 0 is
            # the first device's
            return [add_mine(a, b) for a, b in zip(f3[:3], f)] + [f3[3].at[0].add(2.0 * w.astype(dt))]
        if self.pme_el:
            e, g, f, w = pme_electrostatics(pos3, q, L, self.mesh_el, self.pme_el[0], ASE_UNITS["ke"], dt,
                                            PME_CHUNK, **share(q))
            e_atom, gq = add_mine(e_atom, e), add_mine(gq, g)
            f3 = add(f3, f, w)
            if self.pme_disp:
                # the dispersion mesh after the electrostatic one is gone (XLA would run
                # the two side by side, both meshes alive)
                pos3, rows, c6, e_atom, gq, f3 = lax.optimization_barrier((pos3, rows, c6, e_atom, gq, f3))
        if self.pme_disp:
            # dispersion charges sqrt(C6_i), C6_i = C6_free(Z_i) c6_ratio_i (atomic units)
            c6f = self.c6tab[self.z - 1]
            qd = jnp.sqrt(jnp.maximum(c6f * c6, 0.0))
            e, g, f, w = pme_dispersion(pos3, qd, L, self.mesh_disp, self.pme_disp[0], ASE_UNITS["bohr"],
                                        ASE_UNITS["hartree"], dt, PME_CHUNK, **share(qd))
            e_atom = add_mine(e_atom, e)
            dq = jnp.where(qd > 0.0, c6f / (2.0 * jnp.where(qd > 0.0, qd, 1.0)), 0.0)
            gc6 = add_mine(gc6, g * mine(dq))
            f3 = add(f3, f, w)
        return e_atom, gq, gc6, tuple(f3)

    def _head_backward(self, x, g_raw, ga0, gc6):
        """dE/dx of sum_i e_local[i] + g_raw[i] raw_q[i] + ga0[i] a0[i] + gc6[i] c6[i], written
        over x (dead after)."""
        def heads_bwd(start, x_t, gr_t, ga_t, gc_t):
            zt = lax.dynamic_slice_in_dim(self.z, start, self.tile)

            def f(xx):
                e_l, rq, a, c = self._heads(xx, zt)
                s = e_l.sum() + (gr_t * rq).sum()
                if self.disp:
                    s = s + (ga_t * a).sum() + (gc_t * c).sum()
                return s
            return (jax.grad(f)(x_t),)
        return self._map_rows(heads_bwd, [x, g_raw, ga0, gc6], [x])[0]

    def _post_backward(self, li, x_l, ev_l, kept, g_x, g_ev, pos3, nbr, cnt):
        """dE/d(layer li's attention output), from dE/d(its output); the attention output
        is kept or made again."""
        x1, ev1 = kept if kept else self._attend(li, x_l, ev_l, pos3, nbr, cnt)

        def post_bwd(start, x_t, e_t, gx_t, ge_t):
            _, vjp = jax.vjp(lambda a, b: self._post(li, a, b), x_t, e_t)
            return vjp((gx_t, ge_t))
        return self._map_rows(post_bwd, [x1, ev1, g_x, g_ev], [g_x, g_ev])

    def _layer_backward(self, li, x_l, ev_l, kept, g_x, g_ev, f3, pos3, nbr, cnt):
        """Layer li's backward: dE/d(its input x, ev) and the forces added into f3."""
        g_x, g_ev = self._post_backward(li, x_l, ev_l, kept, g_x, g_ev, pos3, nbr, cnt)
        g_x, g_ev = self._gather(g_x, g_ev)          # the attention backward reads the neighbours'
        # the projections are made again inside, once x1, ev1 are dead
        x_l, ev_l, g_x, g_ev = lax.optimization_barrier((x_l, ev_l, g_x, g_ev))
        g_xin, g_ev, *f3 = self._attention_backward(
            li, x_l, ev_l, pos3, nbr, cnt, g_x, g_ev, g_ev + 0.0, f3)
        return self._g_x_below(li, g_xin), g_ev, tuple(f3)

    def _g_x_below(self, li, g_xin):
        return g_xin * self.params["resid_lambdas"][li] if li > 0 else None

    # ---- one jitted program (usable inside jax-md's jitted steps) --------------------
    def _energy_forces(self, R):
        """(energy, forces, overflow, dE/d(strain)); on several devices one program per
        device (shard_map), each computing its own rows, R and the results the same on all."""
        if self.P == 1:
            return self._energy_forces_rows(R)
        from jax.sharding import PartitionSpec
        rep = PartitionSpec()

        def per_device(R, L):
            keep, self._Lt = self._Lt, L
            try:
                return self._energy_forces_rows(R)
            finally:
                self._Lt = keep
        return jax.shard_map(per_device, mesh=self.device_mesh, in_specs=(rep, rep), out_specs=(rep,) * 4,
                             check_vma=False)(R, self.box if self._Lt is None else self._Lt)

    def _energy_forces_rows(self, R):
        n, dt, M = self.n, self.dtype, self.M
        pos3 = self._pos3(R)
        nb = self.build(pos3)
        nbr, cnt = nb[3], nb[4]
        bar = lax.optimization_barrier

        def run_layers(upto, saved, pos3=pos3):
            x, ev = None, jnp.zeros((n, M), dt)
            for li in range(upto):
                out = self._layer_forward(li, x, ev, pos3, nbr, cnt)
                if saved is not None:
                    saved.append((x, ev) + tuple(out[2:]))
                x, ev = out[:2]
            return x, ev
        saved = [] if self.keep else None
        x, _ = run_layers(self.statics["num_layers"], saved)
        energy, g_x, f3, most = self._head(x, pos3, nb)

        # Everything a layer's backward computes from saved or recomputed inputs (the
        # recompute, the attention forward, the projections) goes through an
        # optimization_barrier with the gradients it follows. Without that XLA runs it as
        # soon as its inputs exist -- every layer's projections at the start of the
        # backward -- and they are all live at once.
        g_ev = jnp.zeros((n, M), dt)
        for li in reversed(range(self.statics["num_layers"])):
            if self.keep:
                (x_l, ev_l, *kept), (g_x, g_ev) = bar((saved[li], (g_x, g_ev)))
            else:
                pos3_l, (g_x, g_ev) = bar((pos3, (g_x, g_ev)))
                (x_l, ev_l), kept = run_layers(li, None, pos3_l), None
            g_x, g_ev, f3 = self._layer_backward(li, x_l, ev_l, tuple(kept or ()), g_x, g_ev, f3,
                                                 pos3, nbr, cnt)
        overflow = (jnp.max(self._mine(cnt)) > self.K) | (most > self.maxh)
        overflow = self._sum(overflow.astype(jnp.int32)) > 0
        f3 = self._gather(*f3)
        return energy, jnp.stack(f3[:3], -1), overflow, self._strain(f3[3], R)

    def _strain(self, fw, R):
        """dE/d(strain), box and positions scaled together (the isotropic virial): half
        the per-atom sums, which count every pair from both ends."""
        return 0.5 * fw.astype(jnp.promote_types(fw.dtype, R.dtype)).sum()

    # ---- separately jitted stages (the most atoms: nothing can be hoisted across) ---
    def _stage(self, fn, donate=()):
        """fn jitted with the atoms (z, elem) and the box as arguments rather than
        constants: one compiled program serves any box within the cell grid's margin
        (NPT)."""
        def g(z, elem, L, *args):
            keep = self.z, self.elem, self._Lt
            self.z, self.elem, self._Lt = z, elem, L
            try:
                return fn(*args)
            finally:
                self.z, self.elem, self._Lt = keep
        j = jax.jit(g, donate_argnums=tuple(d + 3 for d in donate))
        return lambda *args: j(self.z, self.elem, self.box, *args)

    def _stages(self):
        if self.P > 1:
            raise NotImplementedError("the staged memory modes run on one device; several run 'fast'")
        if getattr(self, "_stage_cache", None) is None:
            jit = lambda fn, donate_argnums=(): self._stage(fn, donate_argnums)  # noqa: E731
            c = {"build": jit(lambda R: self.build(self._pos3(R)))}
            for li in range(self.statics["num_layers"]):
                def fwd(x, ev, R, nbr, cnt, li=li):
                    return self._layer_forward(li, x, ev, self._pos3(R), nbr, cnt)
                # kept inputs must survive; otherwise the input is dead once read
                c["fwd", li] = jit(fwd, donate_argnums=() if self.keep else ((0, 1) if li else (1,)))

                def post(x_l, ev_l, kept, g_x, g_ev, R, nbr, cnt, li=li):
                    return self._post_backward(li, x_l, ev_l, kept, g_x, g_ev, self._pos3(R), nbr, cnt)
                c["post", li] = jit(post, donate_argnums=(2, 3, 4))
                for pi in range(len(self._passes())):
                    def att(x_l, ev_l, gdx, gdev, carry, R, nbr, cnt, li=li, pi=pi):
                        return self._attention_backward_pass(li, pi, x_l, ev_l, self._pos3(R), nbr, cnt,
                                                             gdx, gdev, carry)
                    c["att", li, pi] = jit(att, donate_argnums=(4,))
            c["head"] = jit(lambda x, R, nb: self._head(x, self._pos3(R), nb), donate_argnums=(0,))
            self._stage_cache = c
        return self._stage_cache

    def energy_forces_staged(self, R):
        """(energy, forces) as a sequence of jitted stages driven from Python: between
        stages only the arrays held here are alive. Not traceable inside a jitted jax-md
        step; for the largest systems."""
        return self.energy_forces_strain_staged(R)[:2]

    def _forward_staged(self, R):
        """The neighbour table and the layers' forward: {"nb", "x", "saved"}."""
        c = self._stages()
        nb = c["build"](R)
        nbr, cnt = nb[3], nb[4]
        x, ev = None, jnp.zeros((self.n, self.M), self.dtype)
        saved = []
        for li in range(self.statics["num_layers"]):
            if self.keep:
                saved.append([x, ev])
            out = c["fwd", li](x, ev, R, nbr, cnt)
            x, ev = out[:2]
            if self.keep:
                saved[-1] += list(out[2:])
            del out
        return {"nb": nb, "x": x, "saved": saved}

    def _backward_staged(self, R, st):
        """The heads and the layers' backward from a forward's state (consumed): energy,
        forces and the strain derivative's per-atom terms, the most long-range partners
        of any atom."""
        c = self._stages()
        nb, x, saved = st.pop("nb"), st.pop("x"), st.pop("saved")
        nbr, cnt = nb[3], nb[4]
        energy, g_x, f3, most = c["head"](x, R, nb)
        del x
        g_ev = jnp.zeros((self.n, self.M), self.dtype)
        for li in reversed(range(self.statics["num_layers"])):
            if self.keep:
                x_l, ev_l, *kept = saved[li]
                saved[li] = None
            else:
                x_l, ev_l, kept = None, jnp.zeros((self.n, self.M), self.dtype), []
                for k in range(li):
                    x_l, ev_l = c["fwd", k](x_l, ev_l, R, nbr, cnt)[:2]
            # dE/d(attention output); then the attention's backward a pass at a time,
            # each its own program, so only one pass's projections are ever alive
            g_x, g_ev = c["post", li](x_l, ev_l, tuple(kept), g_x, g_ev, R, nbr, cnt)
            del kept
            carry = (self._g_xin_start(li, x_l, g_x), g_ev + 0.0, *f3)
            if self._inplace(li):
                x_l = None                         # lives on as carry[0]
            for pi in range(len(self._passes())):
                carry = c["att", li, pi](x_l, ev_l, g_x, g_ev, carry, R, nbr, cnt)
            del x_l, ev_l
            g_xin, g_ev, *f3 = carry
            del carry
            g_x = self._g_x_below(li, g_xin)
            del g_xin
        return energy, f3, most

    def energy_forces_strain_staged(self, R):
        """energy_forces_staged and dE/d(strain) (box and positions scaled together)."""
        R = jnp.asarray(R)
        R = R if R.dtype == jnp.float64 else R.astype(self.dtype)
        while True:
            st = self._forward_staged(R)
            cnt = st["nb"][4]
            energy, f3, most = self._backward_staged(R, st)
            if int(jnp.max(cnt)) <= self.K and int(most) <= self.maxh:
                return energy, jnp.stack(f3[:3], -1), self._strain(f3[3], R)
            self.widen()

    def widen(self):
        """Wider neighbour rows and long-range scratch after an overflow; recompiles."""
        self.K = int(math.ceil(self.K * 1.25 / 16) * 16)
        self.maxh = int(self.maxh * 1.5)
        self._build_jit()

    def _build_jit(self):
        def ef(R, L):
            keep, self._Lt = self._Lt, L
            try:
                return self._energy_forces(R)
            finally:
                self._Lt = keep
        self._jit = jax.jit(ef)
        self._stage_cache = None
        # programs others build on this engine's force (md's jitted step loops): they
        # hold the old tables and grid, so a rebuild drops them
        self.loops = {}

    def energy_forces_strain(self, R):
        """(energy, forces, dE/d(strain)) in the current box (set_box). Rebuilds with
        wider tables if a row or scratch overflowed."""
        while True:
            e, f, over, w = self._jit(R, self.box)
            if not bool(over):
                return e, f, w
            self.widen()

    def energy_forces(self, R):
        """(energy, forces). Rebuilds with wider tables if a row or scratch overflowed."""
        return self.energy_forces_strain(R)[:2]

    def set_box(self, L):
        """A new orthorhombic box (sides L) for the next calls, e.g. an NPT step. Within
        the margin the cell grid was laid out for, nothing compiles; outside it the
        grid (and the PME mesh, as so3lr_dev's re-tuning would) is laid out again for L."""
        L = [float(v) for v in np.asarray(L, np.float64).reshape(-1)[:3]]
        hi = 1.0 + self.box_margin
        if all(lo <= v <= hi * b for v, lo, b in zip(L, self.L_lo, self.L)):
            self.box = jnp.asarray(L, jnp.float64 if jax.config.jax_enable_x64 else self.dtype)
            return
        self._layout(L)
        self._build_jit()
