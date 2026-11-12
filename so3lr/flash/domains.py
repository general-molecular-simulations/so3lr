"""Flash on systems larger than one GPU holds: a domain at a time, exactly.

The box is cut into ``splits`` domains along its axes and every force call makes three
sweeps over them on one GPU, each domain with a halo (the atoms within h of it):

1. per-atom values. Each domain with a halo of R = layers x cutoff gives its own atoms'
   local energy (with their halves of the repulsion), raw charges and a0, c6 ratios:
   exact, as an atom's features depend only on atoms within R of it.
2. the long range. With the charges equilibrated over the whole system, each domain with
   a halo of the long-range cutoff gives its own atoms' long-range energy, dE/d(q, a0,
   c6) and long-range forces.
3. forces. Each domain with a halo of R backpropagates its own atoms' terms (local
   energy; charges and ratios times dE/d(them)); the gradients with respect to every
   position in the domain and its halo are partial forces, summed over the domains.
   The repulsion's forces are taken for the own atoms, for whom they are complete.

Every domain is padded to one atom count with dummy atoms on a grid in the gap between
its halo and its periodic image, where they are farther than the cutoff from anything,
and the cut, gathers and scatters work on index arrays of that length (n where there is
no atom): one engine per sweep compiles once, for every domain and every MD step. The
atoms are sorted by domain once per call, and a domain's atoms with its halo are cut
from its own and its neighbours' (a domain is at least a halo wide), so the cuts cost
O(atoms) per sweep, not O(atoms x domains).

The whole system's per-atom arrays stay on the GPU (positions, domain index and the
per-atom values of the sweeps: about 60 bytes per atom with float64 positions and a
float32 model) and the domains take what is left. The cost per atom does not grow with
the system. The long range is real space only: a PME mesh needs the whole box. Atom
indices are int32, so a system has fewer than 2^31 - 1 atoms.

On several GPUs (``devices``) each holds the whole system's per-atom arrays and takes
every P-th domain, one thread per GPU; between the sweeps the per-atom values are summed
on the first GPU and copied back to every GPU.
"""
import gc
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .engine import MAX_REBUILDS, FlashSO3LR, mode_kib


@partial(jax.jit, donate_argnums=(0,))
def _scatter_set(arr, idx, val):
    return arr.at[idx].set(val, mode="drop")


@partial(jax.jit, donate_argnums=(0,))
def _scatter_add(arr, idx, val):
    return arr.at[idx].add(val, mode="drop")


def _halo(statics):
    """The local halo: an atom's features depend on atoms within layers x cutoff."""
    return int(statics["num_layers"]) * float(statics["cutoff"]) + 0.1


def _pad(m):
    """A row count with 2% to spare, rounded up to 1024: sizes only grow, so MD steps
    reuse the compiled programs."""
    return int(math.ceil(m * 1.02 / 1024) * 1024)


def _staged(eng, name, make):
    """eng's staged program ``name``, made by make() (again after a widen, which drops
    the engine's programs)."""
    c = eng._stages()
    if name not in c:
        c[name] = eng._stage(make())
    return c[name]


def pick_splits(n, box, statics, free_bytes, mode="fast", devices=1, kib_per_row=None):
    """Domains per side (the same along every axis) for n atoms spread evenly over the
    box: the fewest whose largest domain with its halo fits ``mode`` in free_bytes, or on
    several devices a few more if the atoms (halos included) of the device with the most
    domains are then fewer; (1, 1, 1) if the whole system fits (then call FlashSO3LR).
    kib_per_row: what a domain's engine holds per row, if measured (default: the
    whole-system engine's estimate for ``mode`` with a quarter more, which is about twice
    what the domain engines hold)."""
    halo = _halo(statics)
    hmax = max(halo, float(statics["long_range_cutoff"]) + 0.1)
    L = [float(v) for v in np.diag(np.asarray(box, np.float64).reshape(3, 3))]
    density = n / math.prod(L)
    per_atom = (kib_per_row or mode_kib(mode, statics) * 1.25) * 1024

    def atoms(k):                      # atoms of one domain with its halo
        return density * math.prod(v / k + 2.0 * halo if k > 1 else v for v in L)

    def allowed(k):                    # FlashDomains' rules for the domains' width
        return all(v / k >= hmax and v / k + 2.0 * hmax + 0.1 <= v for v in L)
    if atoms(1) * per_atom + (2 << 30) <= free_bytes:
        return (1, 1, 1)               # no need for domains: one FlashSO3LR call
    k0 = next((k for k in range(2, 64) if allowed(k) and atoms(k) * per_atom + (2 << 30) <= free_bytes), None)
    if k0 is None:
        raise ValueError("no domain split fits")
    ks = [k for k in range(k0, k0 + 4) if allowed(k)]
    k = min(ks, key=lambda k: -(-k ** 3 // devices) * atoms(k)) if devices > 1 else k0
    return (k, k, k)


class FlashDomains:
    """Energy and forces of FlashSO3LR for atoms ``z`` in the orthorhombic ``box``, a
    domain at a time (splits per axis; see the module docstring), on ``devices`` GPUs (a
    count or a list). ``engine_kw`` go to the domains' engines (memory options; no PME).
    The force calls run under ``matmul_precision`` (as the flash calculator and MD do):
    with TF32 the float32 energy depends on how the work is split."""

    def __init__(self, params, statics, z, box, *, splits, total_charge=0.0, total_spin=0.0,
                 equilibration_charge=None, devices=1, matmul_precision="highest", **engine_kw):
        if engine_kw.get("pme_electrostatics") or engine_kw.get("pme_dispersion"):
            raise NotImplementedError("PME with domains: the mesh needs the whole box")
        if engine_kw.get("devices", 1) != 1:
            raise ValueError("FlashDomains takes devices itself; its domains' engines run on one each")
        self.params, self.statics = params, statics
        self.devices = jax.local_devices()[:devices] if isinstance(devices, int) else list(devices)
        if not self.devices or isinstance(devices, int) and len(self.devices) < devices:
            raise ValueError(f"devices={devices}: 1 to {len(jax.local_devices())} here")
        self.splits = tuple(int(k) for k in splits)
        if len(self.splits) != 3 or min(self.splits) < 1 or max(self.splits) == 1:
            raise ValueError(f"splits {splits}: three positive counts, at least one above 1 "
                             "(one domain is the whole system: FlashSO3LR)")
        self.z = np.asarray(z).astype(np.int32)
        self.n = int(self.z.shape[0])
        if self.n >= 2 ** 31 - 1:
            raise ValueError(f"{self.n} atoms: domains index atoms with int32, fewer than 2^31 - 1")
        self.matmul_precision = matmul_precision
        box = np.asarray(box, np.float64).reshape(3, 3)
        if np.abs(box - np.diag(np.diag(box))).max() > 0:
            raise ValueError("needs an orthorhombic box")
        self.L = [float(v) for v in np.diag(box)]
        self.rc = float(statics["cutoff"])
        self.rc_lr = float(statics["long_range_cutoff"])
        self.halo = _halo(statics)
        self.halo_lr = self.rc_lr + 0.1
        hmax = max(self.halo, self.halo_lr)
        for L, k in zip(self.L, self.splits):
            if k > 1 and L / k + 2.0 * hmax + 0.1 > L:
                raise ValueError(f"a domain of {L / k:.1f} A with its halo overlaps itself in a box of "
                                 f"{L:.1f} A: fewer splits")
            if k > 1 and L / k < hmax:
                raise ValueError(f"domains of {L / k:.1f} A are narrower than the halo "
                                 f"({hmax:.1f} A): fewer splits")
        self.engine_kw = engine_kw
        self.dtype = statics.get("dtype", jnp.float32)
        self.total_charge, self.total_spin = float(total_charge), float(total_spin)
        self.q_total = float(total_charge if equilibration_charge is None else equilibration_charge)
        # the embedding depends on the element counts of the whole system: one table of
        # every element present, shared by the domains' engines
        self.zu, inv = np.unique(self.z, return_inverse=True)
        self.counts = np.bincount(inv)
        self._lut = np.zeros(int(self.zu.max()) + 1, np.int32)
        self._lut[self.zu] = np.arange(len(self.zu), dtype=np.int32)
        # per device: the parameters, embedding table, atomic numbers and element lookup
        self._dev = [dict() for _ in self.devices]
        # dummies on a grid wider than the cutoff: no two within it
        self.grid = max(5.0, self.rc + 0.5)
        self._engines, self._shape, self._cuts, self._dummy = {}, {}, {}, {}
        # each domain and its neighbours (flat ids, padded with nd: an empty domain)
        self.nd = nd = math.prod(self.splits)
        offsets = [(i - 1, j - 1, k - 1) for i, j, k in np.ndindex(3, 3, 3)]
        self._nbrs = {}
        for d in self._domains():
            near = sorted({self._flat(tuple((d[a] + o[a]) % self.splits[a] for a in range(3))) for o in offsets})
            self._nbrs[d] = np.asarray(near + [nd] * (27 - len(near)), np.int32)
        self._m_pad = 0
        self.timings = {}

    # ---- devices ---------------------------------------------------------------------
    def _each(self, fn):
        """[fn(p) for each device p], one thread per device, each with its device as the
        default."""
        def run(p):
            # both contexts are thread-local: set in each worker
            with jax.default_device(self.devices[p]), jax.default_matmul_precision(self.matmul_precision):
                return fn(p)
        if len(self.devices) == 1:
            return [run(0)]
        with ThreadPoolExecutor(len(self.devices)) as ex:
            return list(ex.map(run, range(len(self.devices))))

    def _on(self, p, name, make):
        """Device p's copy of something made once (make() on that device)."""
        if name not in self._dev[p]:
            self._dev[p][name] = jax.device_put(make(), self.devices[p])
        return self._dev[p][name]

    def _summed(self, parts, everywhere=True):
        """The sum of per-device arrays (parts[p] on device p), on every device (or the
        first): summed on the first one at a time (one extra copy there), then copied."""
        if len(parts) == 1:
            return parts
        d0 = self.devices[0]
        tot = parts[0]
        for p in range(1, len(parts)):
            tot = jax.block_until_ready(tot + jax.device_put(parts[p], d0))
            parts[p] = None
        return [tot] + ([jax.device_put(tot, d) for d in self.devices[1:]] if everywhere else [])

    # ---- domains and their atoms ------------------------------------------------
    def _domains(self):
        return list(np.ndindex(*self.splits))

    def _flat(self, d):
        return int(np.ravel_multi_index(d, self.splits))

    def _domain_of(self, R):
        dom = jnp.zeros((self.n,), jnp.int32)
        for ax in range(3):
            k, L = self.splits[ax], self.L[ax]
            if k > 1:
                d = jnp.clip(jnp.floor(jnp.remainder(R[:, ax], L) / (L / k)), 0, k - 1).astype(jnp.int32)
                dom = dom * k + d
        return dom

    def _inside(self, R, idx3, h):
        keep = jnp.ones((R.shape[0],), bool)
        for ax in range(3):
            k = self.splits[ax]
            if k > 1:
                L, D = self.L[ax], self.L[ax] / k
                keep = keep & (jnp.remainder(R[:, ax] - idx3[ax] * D + h, L) < D + 2.0 * h)
        return keep

    def _index(self, dom):
        """The atoms sorted by domain: (order, where each domain starts, how many each
        has; one more, empty, domain at the end)."""
        counts = jnp.bincount(dom, length=self.nd + 1).astype(jnp.int32)
        starts = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(counts)[:-1].astype(jnp.int32)])
        m = int(jnp.max(counts))
        if m > self._m_pad:                 # only grows: MD steps compile nothing
            self._m_pad = _pad(m)
        return jnp.argsort(dom).astype(jnp.int32), starts, counts

    def _candidates(self, index, nbrs):
        """The atoms of a domain's neighbourhood (itself and its neighbours), n where none."""
        key = ("cand", self._m_pad)
        if key not in self._cuts:
            n, M = self.n, self._m_pad

            def cand(order, starts, counts, nbrs):
                slot = jnp.arange(M, dtype=jnp.int32)
                at = starts[nbrs][:, None] + slot[None, :]
                ok = slot[None, :] < counts[nbrs][:, None]
                return jnp.where(ok, order[jnp.minimum(at, n - 1)], n).reshape(-1)
            self._cuts[key] = jax.jit(cand)
        return self._cuts[key](*index, nbrs)

    def _keep(self, R, cand, idx3, h):
        """Which of the neighbourhood's atoms (cand, n where none) are in domain idx3 or its
        halo h."""
        return self._inside(R.at[cand].get(mode="fill", fill_value=0.0), idx3, h) & (cand < self.n)

    def _count(self, R, cand, idx3, h):
        """Atoms of domain idx3 with a halo h (of its neighbourhood cand)."""
        if "count" not in self._cuts:
            def count(R, cand, idx3, h):
                return jnp.sum(self._keep(R, cand, idx3, h), dtype=jnp.int32)
            self._cuts["count"] = jax.jit(count)
        return int(self._cuts["count"](R, cand, idx3, h))

    def _cut(self, n_pad):
        """Jitted: domain idx3 with a halo h (of its neighbourhood cand) in n_pad rows --
        atom ids (n where there is none), positions in the domain's own box (dummies in
        the spare rows), atomic numbers, which rows are its own atoms, which are atoms at
        all, and how many."""
        if n_pad not in self._cuts:
            n, splits, Ls = self.n, self.splits, self.L

            def cut(R, dom, z, cand, idx3, flat, h, dummy, z_dummy):
                keep = self._keep(R, cand, idx3, h)
                m = jnp.sum(keep, dtype=jnp.int32)
                rows = jnp.nonzero(keep, size=n_pad, fill_value=cand.shape[0])[0]
                ids = jnp.concatenate([cand, jnp.full((1,), n, cand.dtype)])[rows].astype(jnp.int32)
                p = R.at[ids].get(mode="fill", fill_value=0.0)
                cols = []
                for ax in range(3):
                    k, L = splits[ax], Ls[ax]
                    # from the domain's lower face less the halo, along split axes
                    shift = idx3[ax] * (L / k) - h if k > 1 else 0.0
                    cols.append(jnp.remainder(p[:, ax] - shift, L))
                slot = jnp.arange(n_pad, dtype=jnp.int32)
                real = slot < m
                spare = dummy[jnp.clip(slot - m, 0, n_pad - 1)].astype(p.dtype)
                pos = jnp.where(real[:, None], jnp.stack(cols, -1), spare)
                zz = jnp.where(real, z.at[ids].get(mode="fill", fill_value=0), z_dummy)
                own = real & (dom.at[ids].get(mode="fill", fill_value=-1) == flat)
                return ids, pos, zz, own, real, m
            self._cuts[n_pad] = jax.jit(cut)
        return self._cuts[n_pad]

    def _plane(self, sub_L):
        """The gap's axis (the first split one), the other two, and the dummy sites along
        each of them per plane."""
        ax = next(a for a in range(3) if self.splits[a] > 1)
        o = [a for a in range(3) if a != ax]
        return ax, o, int(sub_L[o[0]] // self.grid), int(sub_L[o[1]] // self.grid)

    def _dummies(self, sub_L, h, count):
        """count positions in planes ``grid`` apart in the gap of the first split axis,
        the first and last plane rc + 0.5 A from the real atoms on either side."""
        ax, o, g1, g2 = self._plane(sub_L)
        D = self.L[ax] / self.splits[ax]
        k = np.arange(count)
        p = np.zeros((count, 3))
        p[:, ax] = D + 2.0 * h + self.rc + 0.5 + self.grid * (k // (g1 * g2))
        p[:, o[0]] = (k % g1) * self.grid + 1.0
        p[:, o[1]] = ((k // g1) % g2) * self.grid + 1.0
        return p

    def _engine(self, key, n_pad, sub_L, zz, p=0):
        """Device p's engine of a sweep (sweeps 1 and 3 share one), made again only when
        the padded size or the domain box grows."""
        old = self._engines.get((p, key))
        if old is None or old[0] != (n_pad, tuple(sub_L)):
            # a new engine starts from the old one's widened rows and scratch
            kw = dict(self.engine_kw, **(dict(max_neighbours=old[1].K, lr_scratch=old[1].maxh) if old else {}))
            # the old engine is freed before the new one is built (it holds itself in
            # cycles, through its jitted programs: collect them)
            del old
            self._engines.pop((p, key), None)
            gc.collect()
            params = self._on(p, "params", lambda: self.params)
            eng = FlashSO3LR(params, self.statics, np.asarray(zz), np.diag(sub_L),     # run staged, a domain at a time
                             total_charge=self.total_charge, total_spin=self.total_spin, **dict(kw, staged=True))
            eng.x0_table = self._on(p, "x0", lambda: eng._embed_table(
                jnp.asarray(self.zu), jnp.asarray(self.counts, eng.dtype), self.total_charge,
                self.total_spin))                # before anything is traced
            self._engines[(p, key)] = ((n_pad, tuple(sub_L)), eng)
        return self._engines[(p, key)][1]

    def _sweep(self, Rs, doms, indexes, h, sweep, fn, prep):
        """fn(p, engine, positions, extra) for every domain with a halo h, each padded to
        the largest's atom count, device p taking every P-th domain; extra = prep(p, ids,
        own, real). Rs, doms, indexes: per device."""
        domains = self._domains()
        P = len(self.devices)
        idx3 = {d: np.asarray(d, Rs[0].dtype) for d in domains}
        counts = sum(self._each(lambda p: [
            self._count(Rs[p], self._candidates(indexes[p], self._nbrs[d]), idx3[d], h)
            for d in domains[p::P]]), [])
        key = 1 if sweep == 3 else sweep
        # the padded size and the gap only grow, so MD steps reuse one compiled engine
        n_pad, gap0 = self._shape.get(key, (0, 0.0))
        if max(counts) > n_pad:
            n_pad = _pad(max(counts))

        def domain_box(gap):
            return [self.L[ax] / k + 2.0 * h + gap if k > 1 else self.L[ax] for ax, k in enumerate(self.splits)]
        # enough planes of dummies for the emptiest domain; the gap widens to hold them
        _, _, g1, g2 = self._plane(domain_box(gap0))
        layers = max(1, -(-(n_pad - min(counts)) // (g1 * g2)))
        gap = max(gap0, 2.0 * self.rc + 1.0 + self.grid * (layers - 1))
        self._shape[key] = (n_pad, gap)
        sub_L = domain_box(gap)
        if (key, n_pad, gap) not in self._dummy:
            self._dummy = {(key, n_pad, gap): self._dummies(sub_L, h, n_pad).astype(Rs[0].dtype),
                           **{k: v for k, v in self._dummy.items() if k[0] != key}}
        cut = self._cut(n_pad)

        def work(p):
            dummy = jax.device_put(self._dummy[(key, n_pad, gap)], self.devices[p])
            z = self._on(p, "z", lambda: self.z)
            lut = self._on(p, "lut", lambda: self._lut)
            for d in domains[p::P]:
                ids, pos, zz, own, real, m = cut(Rs[p], doms[p], z, self._candidates(indexes[p], self._nbrs[d]),
                                                 idx3[d], self._flat(d), h, dummy, int(self.zu[0]))
                eng = self._engine(key, n_pad, sub_L, zz, p)
                eng.z, eng.elem = zz, lut[zz]
                fn(p, eng, pos, prep(p, ids, own, real))
        self._each(work)

    # ---- the three passes on one domain -------------------------------------------
    @staticmethod
    def _local_forward(eng, pos):
        """Per atom: energy without the long range, raw charge, a0 and c6 ratios."""
        for _ in range(MAX_REBUILDS + 1):
            st = eng._forward_staged(pos)
            out = _staged(eng, "head_local", lambda: lambda x, R, nb: eng._head_local(x, eng._pos3(R), nb)[:4])(
                st["x"], pos, st["nb"])
            if int(jnp.max(st["nb"][4])) <= eng.K:
                return out
            del st, out
            eng.widen()
        raise RuntimeError(f"Flash domains: overflowed again after {MAX_REBUILDS} rebuilds")

    @staticmethod
    def _long_range(eng, pos, q, a0, c6):
        """Long-range energy per atom, dE/d(q, a0, c6) and forces [n, 3]."""
        def make():
            def lr(R, q, a0, c6):
                pos3 = eng._pos3(R)
                return eng._long_range(pos3, eng.build_grid(pos3), q, a0, c6, jnp.zeros((eng.n,), eng.dtype),
                                       eng._fw_zeros())
            return lr
        for _ in range(MAX_REBUILDS + 1):
            e, gq, ga0, gc6, f3, most = _staged(eng, "long_range", make)(pos, q, a0, c6)
            if int(most) <= eng.maxh:
                return e, gq, ga0, gc6, jnp.stack(f3[:3], -1)
            # over the periodic images an atom can have more partners than n - 1
            eng.maxh = int(int(most) * 1.25) if eng.images else min(int(int(most) * 1.25), eng.n)
            eng._build_jit()
        raise RuntimeError(f"Flash domains: overflowed again after {MAX_REBUILDS} rebuilds")

    @staticmethod
    def _local_forces(eng, pos, w, g_raw, ga0, gc6):
        """Forces of sum_i w[i] e_local[i] + g_raw[i] raw_q[i] + ga0[i] a0[i] + gc6[i] c6[i],
        with the repulsion's forces of the atoms with w = 1 only."""
        def head(x, R, nb):
            f3 = _staged(eng, "head_local_f3", lambda: lambda x, R, nb: eng._head_local(x, eng._pos3(R), nb)[4])(
                x, R, nb)
            f3 = tuple(v * w for v in f3)
            g_x = _staged(eng, "head_bwd_w", lambda: lambda x, w, gr, ga, gc: eng._head_backward(
                x, gr, ga, gc, w_e=w))(x, w, g_raw, ga0, gc6)
            return None, g_x, f3, jnp.int32(0)
        for _ in range(MAX_REBUILDS + 1):
            st = eng._forward_staged(pos)
            cnt = st["nb"][4]
            _, f3, _ = eng._backward_staged(pos, st, head)
            if int(jnp.max(cnt)) <= eng.K:
                return jnp.stack(f3[:3], -1)
            eng.widen()
        raise RuntimeError(f"Flash domains: overflowed again after {MAX_REBUILDS} rebuilds")

    # ---- the force call --------------------------------------------------------------
    def energy_forces(self, R):
        """(energy, forces [n, 3]) at positions R [n, 3] (float64 keeps the displacements
        in float64, as FlashSO3LR), on the first device."""
        with jax.default_matmul_precision(self.matmul_precision):
            return self._energy_forces(R)

    def _energy_forces(self, R):
        R = jnp.asarray(R)
        R = R if R.dtype == jnp.float64 else R.astype(self.dtype)
        n, dt = self.n, self.dtype
        acc = jnp.promote_types(dt, R.dtype)
        self.timings = {}
        clock = [time.perf_counter()]
        each = self._each

        def lap(phase, *arrays):
            jax.block_until_ready(arrays)
            now = time.perf_counter()
            self.timings[phase] = now - clock[0]
            clock[0] = now

        # the positions, each atom's domain and the atoms sorted by domain on every device
        Rs = [jax.device_put(R, d) for d in self.devices]
        doms = each(lambda p: self._domain_of(Rs[p]))
        indexes = each(lambda p: self._index(doms[p]))

        def zeros(*shape):
            return each(lambda p: jnp.zeros((n,) + shape, dt))

        # 1. each domain's own atoms' local energies, charges and ratios (row n: dropped)
        out = {k: zeros() for k in "eqac"}

        def per_atom(p, eng, pos, extra):
            for k, v in zip("eqac", self._local_forward(eng, pos)):
                out[k][p] = _scatter_set(out[k][p], extra[0], v)
        self._sweep(Rs, doms, indexes, self.halo, 1, per_atom, lambda p, ids, own, real: [jnp.where(own, ids, n)])
        e_local = self._summed(out.pop("e"), everywhere=False)
        raw_q, a0, c6 = (self._summed(out.pop(k)) for k in "qac")
        lap("1. per-atom values", e_local[0], raw_q, a0, c6)

        # 2. the long range of each domain's own atoms, from the equilibrated charges
        shift = (self.q_total - raw_q[0].astype(acc).sum().astype(dt)) / n
        q = each(lambda p: raw_q[p] + jax.device_put(shift, self.devices[p]))
        del raw_q
        lrs = {k: zeros() for k in "eqac"}
        f_parts = zeros(3)

        def lr_prep(p, ids, own, real):
            # the spare rows (dummies, far from everything real): no charge
            return [jnp.where(own, ids, n), q[p].at[ids].get(mode="fill", fill_value=0.0),
                    a0[p].at[ids].get(mode="fill", fill_value=1.0), c6[p].at[ids].get(mode="fill", fill_value=1.0)]

        def long_range(p, eng, pos, extra):
            mine, qd, ad, cd = extra
            e, gq, ga, gc, f = self._long_range(eng, pos, qd, ad, cd)
            for k, v in zip("eqac", (e, gq, ga, gc)):
                lrs[k][p] = _scatter_set(lrs[k][p], mine, v)
            f_parts[p] = _scatter_set(f_parts[p], mine, f)
        self._sweep(Rs, doms, indexes, self.halo_lr, 2, long_range, lr_prep)
        del q, a0, c6
        e_lr = self._summed(lrs.pop("e"), everywhere=False)
        gq, ga0, gc6 = (self._summed(lrs.pop(k)) for k in "qac")
        energy = e_local[0].astype(acc).sum() + e_lr[0].astype(acc).sum()
        # dE/d(raw charge): through the shift that equilibrates the charges
        g_mean = (gq[0].astype(acc).sum() / n).astype(dt)
        g_raw = each(lambda p: gq[p] - jax.device_put(g_mean, self.devices[p]))
        del e_local, e_lr, gq
        lap("2. long range", f_parts, g_raw)

        # 3. each domain backpropagates its own atoms' terms; the gradients with respect to
        # its halo too are partial forces, summed over the domains (and the devices, with the
        # long range's forces)
        def forces_prep(p, ids, own, real):
            w = own.astype(dt)
            g = [v[p].at[ids].get(mode="fill", fill_value=0.0) * w for v in (g_raw, ga0, gc6)]
            return [w, *g, jnp.where(real, ids, n)]

        def forces(p, eng, pos, extra):
            w, gr, ga, gc, rows = extra
            f_parts[p] = _scatter_add(f_parts[p], rows, self._local_forces(eng, pos, w, gr, ga, gc))
        self._sweep(Rs, doms, indexes, self.halo, 3, forces, forces_prep)
        del g_raw, ga0, gc6, Rs, doms, indexes
        f = self._summed(f_parts, everywhere=False)[0]
        lap("3. forces", f)
        return energy, f

    energy_forces_staged = energy_forces
