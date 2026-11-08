"""SO3LR-Flash on CUDA kernels: the attention of so3lr-tauri-s/m/l and the legacy so3lr (32
features per head, 4 or 8 heads of degrees 1-4) per unordered pair on tensor cores (TF32 x3:
float32-level accuracy), gather-only message and gradient kernels, and the long range per
atom, in place of the Pallas kernels of the fast memory mode. Deterministic: every sum is a
gather in a fixed order. The kernels are built per model variant (``build.kernel_flags``); a
model of 8 heads runs every head of a channel in one CTA.

``attach(eng)`` replaces the engine's methods on the instance; the kernels are built with nvcc
on first use (see ``build``). Per force call the neighbour rows are sorted, and each unordered
pair's record (its atoms, its two directed slots, its geometry) is made once; every layer's
forward keeps the filter per pair and the scores per slot for its backward (``kib_per_atom``).
Where that does not fit, the kernels serve the engine's attention alone, a chunk of rows at a
time: each chunk's per-pair data is made in the forward for the scores and again in the
backward, at about 1.3x the time of keeping it, and the engine's memory mode decides what is
kept per atom (``chunked_kib_per_atom``; the modes picked: ``MODES``). Where none fits, the
engine keeps the Pallas kernels.

Numerics: radial filter tables are the engine's at every 8th node (512 intervals, cubic
Hermite: the same accuracy, an 8x smaller table), the long-range pair tables at every 64th
(256 intervals), where the float32 Hermite derivative is 70x more accurate than at 16384
(the finite difference (f0 - f1) / h loses digits as h shrinks)."""
import functools
import logging
import types

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


def _row_arg(start, first_new):
    """The kernels' row argument of a range of rows: its first row, and the first it writes
    (the rows before are another range's: a chunk shifted back to end at the last row)."""
    return jnp.stack([jnp.asarray(start, jnp.int32), jnp.asarray(first_new, jnp.int32)])


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
    """What the engine holds per atom at its peak on the kernels, every layer's per-pair data
    kept for its backward, in KiB (an estimate, MARGIN aside; on water so3lr-tauri-s/m/l hold
    95/136/406 KiB): the fast mode's arrays, per layer the filter per pair (K F floats), the
    scores per slot, the projections and the post block's VJP inputs, the heads' VJP, and one
    layer's backward temporaries per slot (its slot scalars, and per channel copy the force and
    per-axis virial record and the ev gradient)."""
    from ..engine import mode_kib
    per_layer = eng.K * eng.F + 3 * eng.K * eng.nh + 3 * eng.F + 3 * eng.M     # floats
    floats = eng.statics["num_layers"] * per_layer + 8 * eng.F + _slot_floats(eng)
    return mode_kib("fast", eng.statics) + floats * 4 / 1024


def _slot_floats(eng):
    """One layer's backward temporaries per row, in floats: per slot its scalars, and per
    channel copy the force and per-axis virial record and the ev gradient."""
    return eng.K * (5 * eng.nh + bwd_copies(variant(eng)) * (8 + eng.M))


def chunked_kib_per_atom(eng):
    """What the engine holds per atom at its peak with the per-pair data made a chunk of rows
    at a time (chunk_kib_per_row), in KiB (MARGIN aside; so3lr-tauri-s/m/l on water in MD:
    the measured peaks within 2%). The peak is the top layer's backward: its projections and
    gradients, the neighbour rows, the MD state, and what the memory mode keeps of the layers
    below, their inputs and attention outputs; one program (jitted) holds more of the
    backward's temporaries at once than its own stage does. PME's meshes are not alive then."""
    F, M, K, L = eng.F, eng.M, eng.K, eng.statics["num_layers"]
    w = F + M
    floats = 6.5 * w + K + 145
    if eng.keep:
        floats += (L - 1) * w - F
    if eng.keep_attention:
        floats += (L - 1) * w
    if not eng.staged:
        floats += 4.8 * w
    return floats * 4 / 1024


def chunk_kib_per_row(eng):
    """A chunk's per-pair and per-slot buffers per row, in KiB: its pairs' filters (at most K a
    row: a pair with a partner outside the chunk is the chunk's too), its slots' and shadow
    slots' scores, the pair records and slot index, the backward's slot records, and its
    rows' projections' gradients."""
    K, F, NH = eng.K, eng.F, eng.nh
    return (K * 2 * F + 2 * K * 2 * NH + K * 12 + 2 * K + _slot_floats(eng) + 5 * F) * 4 / 1024


GIB = 1 << 30
MARGIN = 1.05            # on the per-atom estimates
MIN_CHUNK = 2 * GIB      # bytes for a chunk's buffers: fewer rows would spend more on duplicated pairs
CHUNK_BUDGET = (24 * GIB, 64 * GIB)  # bytes for a chunk's buffers: at least (room allowing), at most
# staged: a stage's temporaries are one allocation, which the GPU's memory, fragmented by what
# stays between the stages and steps, may not hold in one piece; the largest (every atom's
# projections, the attention output where it is made again, a chunk's buffers) at most this
# share of the free memory (larger ones failed after the first MD step)
STAGE_SHARE = 0.4

# The engine's memory modes (engine.MODES) picked for the kernels, fastest first; "default"
# is "fast" staged (each stage a program of its own: fewer temporaries alive at once, the
# same speed). The kernels serve the leaner modes too (the attention outputs, then the layer
# inputs, made again in the backward; flash_mode), but they are not picked: on the systems
# they are for, one stage holds every atom's projections in a single allocation of most of
# the GPU, which after the first step the allocator does not always find in one piece.
MODES = ("fast", "default")


def unsupported(eng):
    """Why the CUDA kernels cannot serve this engine (None: they can, memory aside)."""
    why = problem()
    if why:
        return why
    if eng.dtype != jnp.float32:
        return "float64"
    if not jax.config.jax_enable_x64:
        return "float32 positions (the kernels take float64 ones: jax_enable_x64)"
    if variant(eng) is None:
        return "a model other than so3lr-tauri-s/m/l or so3lr (32 features per head, 4 or 8 heads of degrees 1-4)"
    if eng.split or eng.blocks:
        return "a memory mode of the Pallas kernels' own (the attention in passes)"
    return None


def plan(eng, free=None):
    """(bytes for a chunk's buffers, None) for the kernels on this engine: None where every
    layer's per-pair data fits for its backward (in a mode keeping the attention outputs),
    else a chunk's (CHUNK_BUDGET) next to the per-atom arrays of the engine's mode; or (None,
    why) where neither fits."""
    if free is None:
        from ..md import _free_bytes
        free = _free_bytes()
    if eng.keep_attention and eng.n * kib_per_atom(eng) * 1024 * MARGIN + GIB <= free:
        return None, None
    room = free - eng.n * chunked_kib_per_atom(eng) * 1024 * MARGIN - GIB
    # half the room, between the bounds: larger chunks make fewer pairs twice (at their edges)
    budget = min(room, max(CHUNK_BUDGET[0], min(room / 2, CHUNK_BUDGET[1])))
    if eng.staged:
        stage = 4 * eng.n * (5 * eng.F + (0 if eng.keep_attention else eng.F + eng.M))
        budget = min(budget, STAGE_SHARE * free - stage)
    if budget < min(MIN_CHUNK, eng.nl * chunk_kib_per_row(eng) * 1024):
        need = eng.n * chunked_kib_per_atom(eng) * 1024 * MARGIN + GIB + MIN_CHUNK
        return None, (f"{eng.n} atoms need ~{need / GIB:.0f} GiB of the {free / GIB:.0f} free, a chunk of rows "
                      f"at a time, and a stage's temporaries in one piece")
    return budget, None


def pick_mode(n, free, params, statics):
    """The fastest of MODES in which the kernels fit n atoms in free bytes (their per-pair data
    kept, or a chunk of rows at a time), or None: they do not serve this model here, or it
    fits none (the Pallas kernels' modes go further)."""
    from .. import engine as E
    # what the estimates read of an engine for these atoms, as the engine would make it
    F, nh, G, M, sh = E.model_widths(params, statics)
    eng = types.SimpleNamespace(n=n, nl=n, statics=statics, dtype=statics.get("dtype", jnp.float32), F=F, nh=nh,
                                G=G, M=M, sh=sh, K=E.neighbour_rows(float(statics["cutoff"])), split=False,
                                blocks=False, keep=True, keep_attention=True, staged=False)
    if unsupported(eng):
        return None
    for name in MODES:
        kw, eng.staged = E.MODES[name][:2]
        eng.keep = kw.get("keep_layer_inputs", True)
        eng.keep_attention = kw.get("keep_attention", False) and eng.keep
        if plan(eng, free)[1] is None:
            return name
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
    and backward, the long range, and the heads with their VJP reused. In chunks (``chunk``
    rows, or as many as ``budget`` bytes hold) the per-pair data is made a chunk of rows at a
    time, in the forward for the scores and again in the backward, instead of kept for every
    layer (for systems whose per-pair data does not fit), and the kernels serve the engine's
    attention alone: a chunk's pairs are those of its rows, a pair with a partner outside the
    chunk made by both chunks, as the devices do."""

    def __init__(self, eng, chunk=None, budget=None):
        self.eng = eng
        self._chunk, self._budget = chunk, budget
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
        self.kib = (kib_per_atom(eng) if self.chunk is None
                    else chunked_kib_per_atom(eng) + chunk_kib_per_row(eng) * self.chunk / eng.n)
        self._orig_build, self._orig_heads, self._orig_head_backward = eng.build, eng._heads, eng._head_backward
        self._calls = {}                  # the FFI calls per layout of the rows and range (_sized)
        self._pp = None                   # the current trace's pair records (PI, PG, PIDX, off)
        self._post_vjp, self._proj, self._heads_vjp = {}, {}, None

    @property
    def chunk(self):
        """Rows per chunk (None: every layer's per-pair data kept): as asked, or as many as the
        budget holds with the rows as laid out now (wider after an overflow: fewer), in chunks
        of equal size (the last one, shifted back to end at the range's last row, then makes
        fewer than one row a chunk twice)."""
        if self._chunk is None and self._budget is None:
            return None
        nl = self.eng.nl
        rows = self._chunk or self._budget // (chunk_kib_per_row(self.eng) * 1024)
        rows = max(1, min(int(rows), nl))
        return -(-nl // -(-nl // rows))

    def _sized(self, rows=None):
        """The FFI calls with their output shapes and fixed attributes, and the pair capacity,
        for ranges of ``rows`` rows (default: the device's) and the rows as laid out now: made
        at trace time, so wider rows (widen) take effect at the next program."""
        eng = self.eng
        n, K = eng.n, eng.K
        nl = eng.nl if rows is None else rows
        if (K, nl) not in self._calls:
            F, NH, M = eng.F, eng.nh, eng.M
            f32, i32, S = jnp.float32, jnp.int32, jax.ShapeDtypeStruct
            # the per-slot and per-pair buffers hold this device's range of rows: its nl K
            # directed slots, and its pairs (each once where the range is every row: n K / 2;
            # else at most K a row, a pair with a partner outside the range being the range's
            # too). A partner outside the range has no slot in its row: its reverse score goes
            # to the pair's shadow slot (nl K + pair). The per-row kernels then write the
            # range's rows into the input (aliased), the other rows keep the input's; a range
            # of every row writes them all, so its outputs need not alias
            pmax, slots = min(n * K // 2, nl * K), nl * K
            al = lambda d: d if nl < n else None

            def call(name, out, aliases=None, **attrs):
                """A kernel's FFI call, its attributes fixed."""
                return functools.partial(jax.ffi.ffi_call(f"{self.prefix}_{name}", out, input_output_aliases=aliases),
                                         **{k: np.int64(v) for k, v in attrs.items()})
            scores = S(((slots + pmax) * 2 * NH,), f32)
            # in chunks the gather writes the projections' gradients of the chunk's rows only
            # (local), into a carry of the dE/dev and force sums (aliased)
            local = self.chunk is not None
            self._calls[K, nl] = dict(
                pmax=pmax, tiles16=-(-pmax // 16),
                # the pair records; PIDX from a zeroed input (a slot without a pair, while a row
                # overflows, reads pair 0 rather than anything)
                prep=call("prep", (S((pmax * 4,), i32), S((pmax * 8,), f32), S((slots,), i32)), {14: 2},
                          zmax=eng.zmax, P=pmax, NL=nl),
                # the scores per slot, with the filters per pair for the backward (fwd) or not
                fwd=call("fwd", (scores, S((pmax * 2 * F,), f32)), P=pmax, keep_w=1),
                scores=call("fwd", (scores, S((1,), f32)), P=pmax, keep_w=0),
                msg=call("msg", (S((n, F), f32), S((n, M), f32)), al({8: 0, 9: 1}), NL=nl),
                # the backward: slot scalars, and the forces and ev gradients per slot (a copy per
                # channel where each channel's CTA writes its own)
                bwd=call("bwd", (S((slots * 5 * NH,), f32), S((self.copies * slots * 8,), f32),   # 8 floats per slot (FVS)
                                 S((self.copies * slots * M,), f32)), P=pmax),
                gather=call("gather", tuple(S((nl if local else n, F), f32) for _ in range(5))
                            + (S((n, M), f32), S((6, n), f32)), {9: 5, 10: 6} if local else al({9: 5, 10: 6}),
                            NL=nl, local=int(local)),
                lr=call("lr", tuple(S((n,), f32) for _ in range(4)) + (S((6, n), f32),), al({18: 0, 19: 4})))
        return self._calls[K, nl]

    # ---- once per force call: sorted rows and the pair records ---------------------------
    def build(self, pos3):
        """The engine's build, with the neighbour rows sorted and, appended to its tuple,
        the pair records (PI, PG, PIDX, off) every layer reads (in chunks: none, each chunk
        makes its own)."""
        eng = self.eng
        pos3 = _f64(pos3)                 # the kernels form every distance in float64
        order, cell_start, cell_of, nbr, cnt = self._orig_build(pos3)
        n, Kc = nbr.shape
        cm = jnp.minimum(cnt, Kc)
        ns = jnp.sort(jnp.where(jnp.arange(Kc)[None, :] < cm[:, None], nbr, I32MAX), axis=1)
        if self.chunk is not None:
            return order, cell_start, cell_of, ns, cnt
        self._pp = self._records(pos3, ns, cnt, _row_arg(eng._r0, eng._r0), eng.nl)
        return (order, cell_start, cell_of, ns, cnt) + self._pp

    def _records(self, pos3, ns, cnt, row0, nl):
        """The pair records (PI, PG, PIDX, off) of the rows [r0, r0 + nl) (r0 = row0[0]), with
        their per-row arrays: a row keeps its partners outside the range and the larger ones
        inside it (sorted: columns [0, lo) and [hi, cnt))."""
        eng, c = self.eng, self._sized(nl)
        r0 = row0[0]
        rows = jax.lax.dynamic_slice_in_dim(ns, r0, nl)
        cm = jnp.minimum(jax.lax.dynamic_slice_in_dim(cnt, r0, nl), ns.shape[1])
        lo = jnp.sum(rows < r0, axis=1).astype(jnp.int32)
        hi = jnp.sum(rows < (r0 + jnp.arange(nl, dtype=jnp.int32))[:, None], axis=1).astype(jnp.int32)
        off = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum(cm - (hi - lo)).astype(jnp.int32)])
        tile_atom = jnp.clip(jnp.searchsorted(off, 16 * jnp.arange(c["tiles16"], dtype=jnp.int32), side="right") - 1,
                             0, nl - 1).astype(jnp.int32)
        PI, PG, PIDX = c["prep"](*pos3, eng._box(pos3[0]), ns, cnt, eng.z, off, lo, hi, tile_atom, row0, eng.rmin,
                                 self.scal, jnp.zeros((nl * ns.shape[1],), jnp.int32))
        return PI, PG, PIDX, off

    def _pairs(self, c, li, proj, ev, rec, keep_w=True):
        """Layer li's scores per slot and (keep_w) filters per pair, for the pair records rec."""
        L = self.layers[li]
        q1, k1, _, q2, k2 = proj
        return c["fwd" if keep_w else "scores"](q1, k1, q2, k2, ev, rec[0], rec[1], rec[3], *L["tabs"], *L["W"],
                                                *L["V"], self.eng.cdeg, self.scal)

    def _pair_backward(self, c, li, proj, ev, gdx, gdev, rec, Wst, S):
        """Layer li's backward per pair: (slot scalars, forces and virial records, ev gradients)."""
        L, W = self.layers[li], self.layers[li]["W"]
        return c["bwd"](*proj, ev, gdx, gdev, rec[0], rec[1], rec[3], *L["tabs"], *W[:3], *W[4:7], *L["V"],
                        self.eng.cdeg, self.scal, self.eng.par, Wst, S)

    def use(self, nb):
        """The pair records of a build tuple, for a program traced apart from the build."""
        if len(nb) > 5:
            self._pp = tuple(nb[5:9])

    # ---- layers, every layer's per-pair data kept ----------------------------------------
    def layer_forward(self, li, x, ev, pos3, nbr, cnt):
        eng, c = self.eng, self._sized()
        pos3 = _f64(pos3)
        xt = eng._x_in(li, x, eng.elem)
        proj, pvjp = jax.vjp(lambda xx: tuple(eng._project(li, xx)), xt)
        S, Wst = self._pairs(c, li, proj, ev, self._pp)
        x1, ev1 = c["msg"](proj[2], S, *pos3, eng._box(pos3[0]), nbr, cnt, xt, ev, eng.scal,
                           _row_arg(eng._r0, eng._r0))
        # the post block on this device's rows, gathered for the next layer's neighbours (the
        # heads take the last layer's rows as they are)
        (x2l, ev2l), post_vjp = jax.vjp(lambda a, b: eng._post(li, a, b), eng._mine(x1), eng._mine(ev1))
        x2, ev2 = eng._gather_rows(x2l, ev2l) if li < eng.statics["num_layers"] - 1 else (x2l, ev2l)
        # kept for the backward of the same program (the jitted force call runs both in one
        # trace; the staged programs ask for fresh VJPs instead, see forget)
        self._proj[li] = (proj, pvjp)
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
        eng, c = self.eng, self._sized()
        cached = None if fresh else self._proj.pop(li, None)
        if cached is not None:
            proj, vjp = cached
        else:
            xt = eng._x_in(li, x_l, eng.elem)
            proj, vjp = jax.vjp(lambda xx: tuple(eng._project(li, xx)), xt)
        SC, FV, GEV = self._pair_backward(c, li, proj, ev_l, gdx, gdev, self._pp, Wst, Sf)
        q1, k1, _, q2, k2 = proj
        *g, gev, f = c["gather"](q1, k1, q2, k2, gdx, Wst, SC, FV, GEV, gdev, jnp.stack(f3).astype(jnp.float32),
                                 nbr, cnt, self._pp[2], _row_arg(eng._r0, eng._r0))
        f = tuple(f)                          # forces and the per-axis virial
        if li == 0:                           # no layer below takes dE/dx_in (_g_x_below)
            return gdx, gev, f
        (gx,) = vjp(tuple(g))
        return gdx + gx, gev, f

    def layer_backward(self, li, x_l, ev_l, kept, g_x, g_ev, f3, pos3, nbr, cnt):
        eng = self.eng
        gdx, gdev = self.post_backward(li, kept, g_x, g_ev)
        x_l, ev_l, gdx, gdev = jax.lax.optimization_barrier((x_l, ev_l, gdx, gdev))
        g_xin, gev, f3 = self.attention_backward(li, x_l, ev_l, kept[2], kept[3], gdx, gdev, f3, nbr, cnt)
        return eng._g_x_below(li, g_xin), gev, f3

    # ---- in chunks: the engine's attention, a chunk of rows at a time ---------------------
    def attend(self, li, x_l, ev_l, pos3, nbr, cnt):
        """The engine's _attend: (x_in + dx, ev + dev) of layer li, each chunk's pair records
        and scores made, its rows' messages added, and dropped."""
        eng, nc = self.eng, self.chunk
        c = self._sized(nc)
        pos3 = _f64(pos3)
        box = eng._box(pos3[0])
        proj = eng._projections(li, x_l, range(5))

        def chunk(start, first_new, carry):
            row = _row_arg(start, first_new)
            S, _ = self._pairs(c, li, proj, ev_l, self._records(pos3, nbr, cnt, row, nc), keep_w=False)
            return tuple(c["msg"](proj[2], S, *pos3, box, nbr, cnt, *carry, eng.scal, row))
        # the messages added into x_in (its own buffer) and a copy of ev, each row once
        return eng._tile_loop(chunk, (eng._x_in(li, x_l, eng.elem), ev_l + 0.0), nc)

    def attention_backward_pass(self, li, pi, x_l, ev, pos3, nbr, cnt, gdx, gdev, carry):
        """The engine's _attention_backward_pass (one pass: the kernels take both channels):
        each chunk's pair records, filters and scores made again, its pairs' backward, its
        rows' gathers, and its rows' dE/dx_in from their projections' gradients. carry = (dE/dx_in,
        dE/dev, the per-atom force and virial sums); in place (x_l None), carry[0] holds the
        layer input until a chunk overwrites its rows."""
        eng, nc = self.eng, self.chunk
        c = self._sized(nc)
        pos3 = _f64(pos3)
        inplace = eng._inplace(li)
        proj = eng._projections(li, carry[0] if inplace else x_l, range(5))
        q1, k1, _, q2, k2 = proj

        def chunk(start, first_new, carry):
            g_xin, g_ev, f = carry
            row = _row_arg(start, first_new)
            rec = self._records(pos3, nbr, cnt, row, nc)
            S, Wst = self._pairs(c, li, proj, ev, rec)
            SC, FV, GEV = self._pair_backward(c, li, proj, ev, gdx, gdev, rec, Wst, S)
            *g, g_ev, f = c["gather"](q1, k1, q2, k2, gdx, Wst, SC, FV, GEV, g_ev, f, nbr, cnt, rec[2], row)
            if li > 0:                        # layer 0's input is the constant embedding
                rows = jax.lax.dynamic_slice_in_dim(g_xin if inplace else x_l, start, nc)
                xt = eng._x_in(li, rows, jax.lax.dynamic_slice_in_dim(eng.elem, start, nc))
                _, vjp = jax.vjp(lambda xx: tuple(eng._project(li, xx)), xt)
                (gx,) = vjp(tuple(g))
                old = jax.lax.dynamic_slice_in_dim(g_xin, start, nc)
                new = (jax.lax.dynamic_slice_in_dim(gdx, start, nc) if inplace else old) + gx
                # the rows the previous chunk wrote (the last chunk shifted back) keep theirs
                g_xin = jax.lax.dynamic_update_slice_in_dim(
                    g_xin, jnp.where(eng._rows_mask(start, first_new, nc), new, old), start, 0)
            return g_xin, g_ev, f
        g_xin, g_ev, f = eng._tile_loop(chunk, (carry[0], carry[1], jnp.stack(carry[2:])), nc)
        return (g_xin, g_ev, *f)

    # ---- the staged programs (domains, the staged memory path) --------------------------
    def stage_fns(self, li):
        """(post, att) for the staged backward: post(x_l, ev_l, kept, g_x, g_ev, R, nb) ->
        (gdx, gdev); att(x_l, ev_l, gdx, gdev, carry, extra, R, nb) -> carry', extra being
        what the forward kept beyond (x1, ev1): the filters per pair and the scores. None in
        chunks: the engine's own stages, its attention on the kernels."""
        eng = self.eng
        if self.chunk is not None:
            return None

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


def attach(eng, strict=False, chunk=None):
    """The engine's methods on the CUDA kernels (the instance only). Returns the kernels
    object, or None (with a log line) where they do not apply or cannot be built; strict
    (kernels="cuda" asked for explicitly) raises instead of falling back. chunk (rows): the
    per-pair data a chunk at a time whatever the memory (default: as plan finds room)."""
    why, budget = unsupported(eng), None
    if why is None and chunk is None:
        budget, why = plan(eng)
    if why:
        if strict:
            raise RuntimeError(f"Flash: the CUDA kernels do not serve this engine ({why})")
        logger.info("Flash: CUDA kernels not used (%s); Pallas kernels", why)
        return None
    try:
        ck = CudaKernels(eng, chunk, budget)
    except (BuildError, OSError) as e:       # nvcc failed or unusable, a library does not load, no lock
        if strict:
            raise
        logger.warning("Flash: CUDA kernels not built (%s); Pallas kernels", str(e).splitlines()[0][:160])
        return None
    eng.build = ck.build
    eng._long_range = ck.long_range
    if ck.chunk is None:                     # the layers, with what the forward keeps for the backward
        eng._layer_forward = ck.layer_forward
        eng._layer_backward = ck.layer_backward
        eng._heads = ck.heads
        eng._head_backward = ck.head_backward
    else:                                    # the attention alone, in the engine's layers and modes
        eng._attend = ck.attend
        eng._attention_backward_pass = ck.attention_backward_pass
    eng.cuda_kernels = ck
    logger.info("Flash: CUDA kernels (TF32 x3 attention, gather-only messages and gradients; %s; ~%.0f KiB/atom)",
                f"chunks of {ck.chunk} rows" if ck.chunk else "per-pair data kept", ck.kib)
    return ck
