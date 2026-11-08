"""SO3LR-Flash on CUDA kernels: the attention of so3lr-tauri-s/m/l and the legacy so3lr (32
features per head, 4 or 8 heads of degrees 1-4) per unordered pair on tensor cores (TF32 x3:
float32-level accuracy), gather-only message and gradient kernels, and the long range per
atom, in place of the Pallas kernels of the fast memory mode. Deterministic: every sum is a
gather in a fixed order. The kernels are built per model variant (``build.kernel_flags``); a
model of 8 heads runs every head of a channel in one CTA.

``attach(eng)`` replaces the engine's methods on the instance; the kernels are built with nvcc
on first use (see ``build``). Per force call the neighbour rows are sorted, and each unordered
pair's record (its atoms, its two directed slots, its geometry) is made once; every layer's
forward keeps the filter per pair and the scores per slot for its backward (``kib_per_atom``:
more than the Pallas fast mode holds, so the kernels decline where that does not fit).

Numerics: radial filter tables are the engine's at every 8th node (512 intervals, cubic
Hermite: the same accuracy, an 8x smaller table), the long-range pair tables at every 64th
(256 intervals), where the float32 Hermite derivative is 70x more accurate than at 16384
(the finite difference (f0 - f1) / h loses digits as h shrinks)."""
import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np

from .build import BuildError, bwd_copies, load_kernels, problem

logger = logging.getLogger("SO3LR")
I32MAX = np.iinfo(np.int32).max
TABLE_SUB = 8            # radial filter tables: every TABLE_SUB-th node of the engine's
PAIR_SUB = 64            # long-range pair tables: every PAIR_SUB-th node


def _f64(pos3):
    """The position columns in float64 (a float32 run passes float32 ones; the kernels take
    float64 and form every distance in it, as the mixed-precision runs do)."""
    return tuple(c if c.dtype == jnp.float64 else c.astype(jnp.float64) for c in pos3)


def variant(eng):
    """The kernels' build variant of this engine's model: (features, heads, equivariant
    components, the spherical filters' hidden units, the entries' degrees as nibbles, the
    radial tables' rank), or None where the kernels are not written for it."""
    degrees = [int(d) for d in eng.statics["degrees"]]
    if not (eng.G == 32 and eng.nh in (4, 8) and set(degrees) <= {1, 2, 3, 4} and eng.sh % 32 == 0):
        return None
    degs = sum(d << (4 * e) for e, d in enumerate(degrees))
    return (eng.F, eng.nh, eng.M, eng.sh, degs, 48 if eng.nh == 4 else 64)


def kib_per_atom(eng):
    """What the engine holds per atom at its peak on the kernels, in KiB (an estimate; the
    margin is unsupported's): the fast mode's arrays, per layer the filter per pair (K F
    floats), the scores per slot, the projections with their VJP's inputs and the post
    block's, the heads' VJP, and one layer's backward temporaries per slot (its slot scalars,
    and per channel copy the force and per-axis virial record and the ev gradient)."""
    from ..engine import mode_kib
    per_layer = eng.K * eng.F + 3 * eng.K * eng.nh + 16 * eng.F + 3 * eng.M     # floats
    bwd = eng.K * (5 * eng.nh + bwd_copies(variant(eng)) * (8 + eng.M))
    return mode_kib("fast", eng.statics) + (eng.statics["num_layers"] * per_layer + 8 * eng.F + bwd) * 4 / 1024


def unsupported(eng):
    """Why the CUDA kernels cannot serve this engine (None: they can)."""
    why = problem()
    if why:
        return why
    if eng.dtype != jnp.float32:
        return "float64"
    if not jax.config.jax_enable_x64:
        return "float32 positions (the kernels take float64 ones: jax_enable_x64)"
    if variant(eng) is None:
        return "a model other than so3lr-tauri-s/m/l or so3lr (32 features per head, 4 or 8 heads of degrees 1-4)"
    if not eng.keep_attention or eng.split or eng.blocks:
        return "a memory mode other than fast"
    from ..md import _free_bytes
    need, free = eng.n * kib_per_atom(eng) * 1024 * 1.1 + (1 << 30), _free_bytes()
    if need > free:
        return f"{eng.n} atoms' pair filters and scores need ~{need / 2 ** 30:.0f} GiB of the {free / 2 ** 30:.0f} free"
    return None


def low_rank(T, D, rank, spacing=None):
    """T, D [nodes, F] -> coefficients of T and D on the top ``rank`` right singular vectors
    of T (spacing None) or of [T; spacing D] (the derivative scaled to the values, where the
    wider tables need it), and those vectors [rank, F]."""
    T, D = np.asarray(T, np.float64), np.asarray(D, np.float64)
    basis = T if spacing is None else np.concatenate([T, spacing * D], axis=0)
    V = np.linalg.svd(basis, full_matrices=False)[2][:rank].T
    return jnp.asarray(T @ V, jnp.float32), jnp.asarray(D @ V, jnp.float32), jnp.asarray(V.T, jnp.float32)


@functools.cache
def ffi_prefix(var):
    """The FFI target names' prefix of a variant's kernels, registered once per process. The
    names depend on the variant alone, so the jitted programs hit the persistent compilation
    cache from one process to the next."""
    libs = load_kernels(var)
    prefix = "so3lr_flash_cuda_" + "_".join(f"{v:x}" if i == 4 else str(v) for i, v in enumerate(var))
    for name, (lib, sym) in (("prep", (libs["fwd"], "PairPrep")), ("fwd", (libs["fwd"], "AttPair")),
                             ("msg", (libs["fwd"], "AttMsg")), ("bwd", (libs["bwd"], "BwdPair7")),
                             ("gather", (libs["bwd"], "BwdGather")), ("lr", (libs["lr"], "LongRange"))):
        jax.ffi.register_ffi_target(f"{prefix}_{name}", jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA")
    return prefix


class CudaKernels:
    """The engine's CUDA methods: build (sorted rows + pair records), the layers' forward
    and backward, the long range, and the heads with their VJP reused."""

    def __init__(self, eng):
        self.eng = eng
        self.variant = variant(eng)
        NH, rank = self.variant[1], self.variant[5]
        self.copies = bwd_copies(self.variant)      # of the per-slot forces and ev gradients
        self.prefix = ffi_prefix(self.variant)
        self.layers = []
        for li in range(eng.statics["num_layers"]):
            p = eng.params[f"layers_{li}"]["attention_block"]
            sf = lambda name, k: p[f"{name}_layer_{k}"]
            W = [sf("spherical_filter1", 1)["kernel"], sf("spherical_filter1", 1)["bias"],
                 sf("spherical_filter1", 2)["kernel"], sf("spherical_filter1", 2)["bias"],
                 sf("spherical_filter2", 1)["kernel"], sf("spherical_filter2", 1)["bias"],
                 sf("spherical_filter2", 2)["kernel"], sf("spherical_filter2", 2)["bias"]]
            t = [x[::TABLE_SUB] for x in eng.tables[li]]
            joint = None if NH == 4 else float(eng.scal[1]) * TABLE_SUB
            u1v, u1d, v1 = low_rank(t[0], t[1], rank, joint)
            u2v, u2d, v2 = low_rank(t[2], t[3], rank, joint)
            self.layers.append(dict(W=W, tabs=[u1v, u1d, u2v, u2d], V=[v1, v2]))
        self.scal = eng.scal.at[1].multiply(TABLE_SUB).at[2].divide(TABLE_SUB)
        self.pair_tables = (eng.phi_v[::PAIR_SUB], eng.phi_d[::PAIR_SUB], eng.c6lr_v[::PAIR_SUB], eng.c6lr_d[::PAIR_SUB])
        self.kib = kib_per_atom(eng)
        self._orig_build, self._orig_heads, self._orig_head_backward = eng.build, eng._heads, eng._head_backward
        self._calls = None                # the FFI calls for the rows' current layout (_sized)
        self._pp = None                   # the current trace's pair records (PI, PG, PIDX, off)
        self._post_vjp, self._proj, self._heads_vjp = {}, {}, None

    def _sized(self):
        """The FFI calls with their output shapes, and the pair capacity, for the rows as laid
        out now: made at trace time, so wider rows (widen) take effect at the next program."""
        eng = self.eng
        n, K, P = eng.n, eng.K, eng.P
        if self._calls is None or self._calls["key"] != (n, K, P):
            F, NH, M = eng.F, eng.nh, eng.M
            f32, i32, S = jnp.float32, jnp.int32, jax.ShapeDtypeStruct
            pmax = n * K // 2
            # several devices: each its rows' pairs; a partner on another device has no slot in
            # its row, its reverse score goes to the pair's shadow slot (n K + pair). The per-row
            # kernels then write this device's rows into the input (aliased), the other rows
            # keep the input's; one device writes every row, so its outputs need not alias
            al = lambda d: d if P > 1 else None
            call = lambda name, out, **kw: jax.ffi.ffi_call(f"{self.prefix}_{name}", out, **kw)
            self._calls = dict(
                key=(n, K, P), pmax=pmax, tiles16=-(-pmax // 16),
                # the pair records; PIDX from a zeroed input (a slot without a pair, while a row
                # overflows, reads pair 0 rather than anything)
                prep=call("prep", (S((pmax * 4,), i32), S((pmax * 8,), f32), S((n * K,), i32)), input_output_aliases={14: 2}),
                fwd=call("fwd", (S(((n * K + pmax) * 2 * NH,), f32), S((pmax * 2 * F,), f32))),
                msg=call("msg", (S((n, F), f32), S((n, M), f32)), input_output_aliases=al({8: 0, 9: 1})),
                # the backward: slot scalars, and the forces and ev gradients per slot (a copy per
                # channel where each channel's CTA writes its own)
                bwd=call("bwd", (S((n * K * 5 * NH,), f32), S((self.copies * n * K * 8,), f32),   # 8 floats per slot (FVS)
                                 S((self.copies * n * K * M,), f32))),
                gather=call("gather", tuple(S((n, F), f32) for _ in range(5)) + (S((n, M), f32), S((6, n), f32)),
                            input_output_aliases=al({9: 5, 10: 6})),
                lr=call("lr", tuple(S((n,), f32) for _ in range(4)) + (S((6, n), f32),),
                        input_output_aliases=al({18: 0, 19: 4})))
        return self._calls

    # ---- once per force call: sorted rows and the pair records ---------------------------
    def build(self, pos3):
        """The engine's build, with the neighbour rows sorted and, appended to its tuple,
        the pair records (PI, PG, PIDX, off) every layer reads."""
        eng, c = self.eng, self._sized()
        pos3 = _f64(pos3)                 # the kernels form every distance in float64
        order, cell_start, cell_of, nbr, cnt = self._orig_build(pos3)
        n, Kc = nbr.shape
        cm = jnp.minimum(cnt, Kc)
        ns = jnp.sort(jnp.where(jnp.arange(Kc)[None, :] < cm[:, None], nbr, I32MAX), axis=1)
        # this device's rows [r0, r0 + nl): a row keeps its partners on other devices and the
        # larger local ones (sorted: columns [0, lo) and [hi, cnt))
        row0 = eng._row_arg()
        ar = jnp.arange(n, dtype=jnp.int32)
        local = (ar >= row0[0]) & (ar < row0[0] + eng.nl)
        lo = jnp.where(local, jnp.sum(ns < row0[0], axis=1), 0).astype(jnp.int32)
        hi = jnp.where(local, jnp.sum(ns < ar[:, None], axis=1), 0).astype(jnp.int32)
        npairs = jnp.where(local, cm - (hi - lo), 0)
        off = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(npairs).astype(jnp.int32)])
        tile_atom = jnp.clip(jnp.searchsorted(off, 16 * jnp.arange(c["tiles16"], dtype=jnp.int32), side="right") - 1,
                             0, n - 1).astype(jnp.int32)
        PI, PG, PIDX = c["prep"](*pos3, eng._box(pos3[0]), ns, cnt, eng.z, off, lo, hi, tile_atom, row0, eng.rmin,
                                 self.scal, jnp.zeros((n * Kc,), jnp.int32),
                                 zmax=np.int64(eng.zmax), P=np.int64(c["pmax"]), NL=np.int64(eng.nl))
        self._pp = (PI, PG, PIDX, off)
        return order, cell_start, cell_of, ns, cnt, PI, PG, PIDX, off

    def use(self, nb):
        """The pair records of a build tuple, for a program traced apart from the build."""
        if len(nb) > 5:
            self._pp = tuple(nb[5:9])

    # ---- layers ------------------------------------------------------------------------
    def layer_forward(self, li, x, ev, pos3, nbr, cnt):
        eng, L, c = self.eng, self.layers[li], self._sized()
        PI, PG, PIDX, off = self._pp
        pos3 = _f64(pos3)
        xt = eng._x_in(li, x, eng.elem)
        (q1, k1, v1, q2, k2), pvjp = jax.vjp(lambda xx: tuple(eng._project(li, xx)), xt)
        S, Wst = c["fwd"](q1, k1, q2, k2, ev, PI, PG, off, *L["tabs"], *L["W"], *L["V"], eng.cdeg, self.scal,
                          P=np.int64(c["pmax"]))
        x1, ev1 = c["msg"](v1, S, *pos3, eng._box(pos3[0]), nbr, cnt, xt, ev, eng.scal, eng._row_arg(),
                           NL=np.int64(eng.nl))
        # the post block on this device's rows, gathered for the next layer's neighbours (the
        # heads take the last layer's rows as they are)
        (x2l, ev2l), post_vjp = jax.vjp(lambda a, b: eng._post(li, a, b), eng._mine(x1), eng._mine(ev1))
        x2, ev2 = eng._gather_rows(x2l, ev2l) if li < eng.statics["num_layers"] - 1 else (x2l, ev2l)
        # kept for the backward of the same program (the jitted force call runs both in one
        # trace; the staged programs ask for fresh VJPs instead, see forget)
        self._proj[li] = (q1, k1, v1, q2, k2, pvjp)
        self._post_vjp[li] = post_vjp
        return (x2, ev2, x1, ev1, Wst, S)

    def forget(self, li):
        """Drop what layer_forward kept for a backward in the same program (a staged
        program's forward has none)."""
        self._proj.pop(li, None)
        self._post_vjp.pop(li, None)

    def post_backward(self, li, kept, g_x, g_ev, fresh=False):
        """dE/d(attention output) from dE/d(layer output): the forward's VJP in the same
        program, else (fresh: a program of its own) the post block's VJP again."""
        eng = self.eng
        x1, ev1 = kept[0], kept[1]
        vjp = None if fresh else self._post_vjp.pop(li, None)
        if vjp is None:
            _, vjp = jax.vjp(lambda a, b: eng._post(li, a, b), eng._mine(x1), eng._mine(ev1))
        gdxl, gdevl = vjp((eng._mine(g_x).astype(x1.dtype), eng._mine(g_ev).astype(ev1.dtype)))
        return eng._gather_rows(gdxl, gdevl)       # the attention backward reads the neighbours' rows

    def attention_backward(self, li, x_l, ev_l, Wst, Sf, gdx, gdev, f3, nbr, cnt, fresh=False):
        """(dE/dx_in, dE/dev, forces) of layer li from dE/d(its attention output), with the
        forward's filters per pair (Wst) and scores per slot (Sf)."""
        eng, L, c = self.eng, self.layers[li], self._sized()
        PI, PG, PIDX, off = self._pp
        cached = None if fresh else self._proj.pop(li, None)
        if cached is not None:
            q1, k1, v1, q2, k2, vjp = cached
        else:
            xt = eng._x_in(li, x_l, eng.elem)
            (q1, k1, v1, q2, k2), vjp = jax.vjp(lambda xx: tuple(eng._project(li, xx)), xt)
        W = L["W"]
        SC, FV, GEV = c["bwd"](q1, k1, v1, q2, k2, ev_l, gdx, gdev, PI, PG, off, *L["tabs"], *W[:3], *W[4:7], *L["V"],
                               eng.cdeg, self.scal, eng.par, Wst, Sf, P=np.int64(c["pmax"]))
        gq1, gk1, gv1, gq2, gk2, gev, f = c["gather"](q1, k1, q2, k2, gdx, Wst, SC, FV, GEV, gdev,
                                                      jnp.stack(f3).astype(jnp.float32), nbr, cnt, PIDX,
                                                      eng._row_arg(), NL=np.int64(eng.nl))
        f = tuple(f)                          # forces and the per-axis virial
        if li == 0:                           # no layer below takes dE/dx_in (_g_x_below)
            return gdx, gev, f
        (gx,) = vjp((gq1, gk1, gv1, gq2, gk2))
        return gdx + gx, gev, f

    def layer_backward(self, li, x_l, ev_l, kept, g_x, g_ev, f3, pos3, nbr, cnt):
        eng = self.eng
        gdx, gdev = self.post_backward(li, kept, g_x, g_ev)
        x_l, ev_l, gdx, gdev = jax.lax.optimization_barrier((x_l, ev_l, gdx, gdev))
        g_xin, gev, f3 = self.attention_backward(li, x_l, ev_l, kept[2], kept[3], gdx, gdev, f3, nbr, cnt)
        return eng._g_x_below(li, g_xin), gev, f3

    # ---- the staged programs (domains, the staged memory path) --------------------------
    def stage_fns(self, li):
        """(post, att) for the staged backward: post(x_l, ev_l, kept, g_x, g_ev, R, nb) ->
        (gdx, gdev); att(x_l, ev_l, gdx, gdev, carry, extra, R, nb) -> carry', extra being
        what the forward kept beyond (x1, ev1): the filters per pair and the scores."""
        eng = self.eng

        def post(x_l, ev_l, kept, g_x, g_ev, R, nb):
            self.use(nb)
            return self.post_backward(li, kept, g_x, g_ev, fresh=True)

        def att(x_l, ev_l, gdx, gdev, carry, extra, R, nb):
            self.use(nb)
            if x_l is None and eng._inplace(li):
                x_l = carry[0]                     # in place: the layer input rides in the carry (layer 0: None)
            g_xin, gev, f3 = self.attention_backward(li, x_l, ev_l, extra[0], extra[1], gdx, gdev, carry[2:],
                                                     nb[3], nb[4], fresh=True)
            return (g_xin, gev) + tuple(f3)
        return post, att

    # ---- the long range ---------------------------------------------------------------
    def _lr_flags(self):
        """The long-range kernel's switches, from the engine as laid out now (its reach
        changes with free space's box, _fit)."""
        eng = self.eng
        return (int(eng.elec) | int(eng.disp) << 1 | int(eng.pme_disp is not None) << 2
                | int(eng.r_tab < eng.reach) << 3 | int(eng.reach > 0.45 * eng.rc_lr) << 4
                | int(eng.reach > eng.rc_lr - eng.statics["dispersion_energy_cutoff_lr_damping"]) << 5)

    def long_range(self, pos3, grid, q, a0, c6, e_atom, f3):
        eng, c = self.eng, self._sized()
        order, cell_start, cell_of = grid
        phi_v, phi_d, c6lr_v, c6lr_d = self.pair_tables
        pos3 = _f64(pos3)
        lrs = eng.lrs.at[1].multiply(PAIR_SUB).at[2].divide(PAIR_SUB)      # the tables at every PAIR_SUB-th node
        e, gq, ga0, gc6, f = c["lr"](
            *(x[order] for x in pos3), eng._box(pos3[0]), eng.z[order], order, cell_start, cell_of,
            q.astype(jnp.float32)[order], a0[order], c6[order], eng.alphas, eng.c6tab, phi_v, phi_d, lrs,
            c6lr_v, c6lr_d, e_atom, jnp.stack(f3), eng._row_arg(),
            NX=np.int64(eng.nc[0]), NY=np.int64(eng.nc[1]), NZ=np.int64(eng.nc[2]), KC=np.int64(eng.kc),
            flags=np.int64(self._lr_flags()), NL=np.int64(eng.nl))
        return e, gq, ga0, gc6, tuple(f), jnp.zeros((), jnp.int32)

    # ---- the heads, their VJP from the forward when the backward follows in the same program
    def heads(self, x, z):
        if x.shape[0] != self.eng.n:
            return self._orig_heads(x, z)
        out, vjp = jax.vjp(lambda xx: self._orig_heads(xx, z), x)
        self._heads_vjp = (x, vjp)
        return out

    def head_backward(self, x, g_raw, ga0, gc6, w_e=None):
        cached, self._heads_vjp = self._heads_vjp, None
        if cached is None or cached[0] is not x:
            return self._orig_head_backward(x, g_raw, ga0, gc6, w_e)
        e_l = jnp.ones_like(g_raw) if w_e is None else w_e
        if not self.eng.disp:
            ga0, gc6 = jnp.zeros_like(ga0), jnp.zeros_like(gc6)
        return cached[1]((e_l.astype(g_raw.dtype), g_raw, ga0.astype(g_raw.dtype), gc6.astype(g_raw.dtype)))[0]


def attach(eng, strict=False):
    """The engine's methods on the CUDA kernels (the instance only). Returns the kernels
    object, or None (with a log line) where they do not apply or cannot be built; strict
    (kernels="cuda" asked for explicitly) raises instead of falling back."""
    why = unsupported(eng)
    if why:
        if strict:
            raise RuntimeError(f"Flash: the CUDA kernels do not serve this engine ({why})")
        logger.info("Flash: CUDA kernels not used (%s); Pallas kernels", why)
        return None
    try:
        ck = CudaKernels(eng)
    except (BuildError, OSError) as e:       # nvcc failed or unusable, a library does not load, no lock
        if strict:
            raise
        logger.warning("Flash: CUDA kernels not built (%s); Pallas kernels", str(e).splitlines()[0][:160])
        return None
    eng.build = ck.build
    eng._layer_forward = ck.layer_forward
    eng._layer_backward = ck.layer_backward
    eng._long_range = ck.long_range
    eng._heads = ck.heads
    eng._head_backward = ck.head_backward
    eng.cuda_kernels = ck
    logger.info("Flash: CUDA kernels (TF32 x3 attention, gather-only messages and gradients; ~%.0f KiB/atom)", ck.kib)
    return ck
