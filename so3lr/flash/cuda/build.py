"""The CUDA kernels built with nvcc on first use, each library cached by its source (and the
headers), flags, the compiler's version, the GPU architecture and the JAX version
(``$SO3LR_CUDA_CACHE``, default ``~/.cache/so3lr/flash_cuda``). The attention kernels' tiles
are the largest whose shared memory fits the GPU's (``tiles``). ``$SO3LR_CUDA_FLAGS`` adds
nvcc options: ``-lineinfo``, or one of the kernels' ablation switches for a timing-only
build (wrong results): ``-DNOMMA`` ``-DNOPROD`` ``-DNOQK`` ``-DNOWST`` (the forward),
``-DB_NOMMA`` ``-DB_NOPROD`` ``-DB_NOEPI`` ``-DB_NOQK`` (the backward), ``-DNOPAIR`` (the long
range)."""
import ctypes
import functools
import glob
import hashlib
import itertools
import logging
import os
import shutil
import subprocess
from typing import NamedTuple

import jax

from ..check import is_rocm

logger = logging.getLogger("SO3LR")
HERE = os.path.dirname(os.path.abspath(__file__))


class BuildError(RuntimeError):
    """nvcc failed, or a built library does not load."""


# library: (source, flags) for a model variant. The attention kernels: producer warps fill a
# stage per tile of pairs, a consumer warp per channel and head takes it (double buffered:
# two stages). Four heads (so3lr-tauri-s/m, so3lr): the forward one channel per CTA (two CTAs
# an SM where they fit), the backward both channels in one CTA. Eight heads (so3lr-tauri-l):
# a channel per CTA in both, the backward's two channel CTAs writing two copies of the
# per-slot forces and ev gradients (``bwd_copies``). Where the shared memory is smaller, the
# tiles are (``tiles``). The gather: two warps per atom (four with eight heads). The long
# range: fast math.
def kernel_flags(variant):
    """{"fwd", "bwd", "lr"}: (source, flags) for ``variant`` = (features, heads, components,
    hidden units, degrees as nibbles, rank) on this GPU."""
    F, NH, M, SH, degs, rank = variant
    t = tiles(variant)
    if t is None:
        raise BuildError(f"Flash CUDA kernels: {variant} needs more shared memory per block than this GPU's")
    v = [f"-DVF={F}", f"-DVNH={NH}", f"-DVM={M}", f"-DVSH={SH}", f"-DVDEGS={degs:#x}", f"-DRANK={rank}"]
    extra = os.environ.get("SO3LR_CUDA_FLAGS", "").split()
    return {
        "fwd": ("attn_pair7.cu", v + t["fwd"].flags() + ["-DLDM"] + extra),
        "bwd": ("attn_bwd7.cu", v + t["bwd"].flags() + ["-DGSPLIT=2", "-DLDM"] + extra),
        "lr": ("lr.cu", ["--use_fast_math"] + extra),
    }


def bwd_copies(variant):
    """The copies of the per-slot forces and ev gradients the backward of ``variant`` writes:
    one per channel where each channel runs in its own CTA."""
    return 2 if tiles(variant)["bwd"].split else 1


class Tile(NamedTuple):
    """An attention kernel's tiles: pairs per tile, pipeline stages, a CTA per channel, CTAs an
    SM holds (the registers' bound), and its shared memory per CTA in bytes."""
    pairs: int
    stages: int
    split: bool
    ctas: int
    smem: int

    def flags(self):
        return (["-DCHSPLIT"] if self.split else []) + [f"-DTP={self.pairs}", f"-DNSTAGES={self.stages}",
                                                       f"-DCTAS={self.ctas}", f"-DSMEM={self.smem}"]


def fwd_smem(variant, pairs, stages, split=True):
    """The forward's shared memory per CTA in bytes (attn_pair7.cu asserts the same): a channel's
    weights [FC][KD + 4], the stages of a tile of pairs (their [hz | coefficients] rows,
    records, cutoffs and scores), the filters' first layer. It runs a CTA per channel (split)."""
    _, NH, _, SH, _, R = variant
    FC, KD = 32 * NH, SH + R
    return 4 * (FC * (KD + 4) + 2 * (NH + 1) * SH) + stages * pairs * (4 * (KD + 4) + 20 + 8 * NH)


def bwd_smem(variant, pairs, stages, split):
    """The backward's shared memory per CTA in bytes (attn_bwd7.cu asserts the same): its
    channels' weights (V transposed, wb), the stages of a tile of pairs, the epilogue's partials,
    the filters' first layer. split: a CTA per channel."""
    _, NH, _, SH, _, R = variant
    FC, nch = 32 * NH, 1 if split else 2
    ncw = nch * NH                                       # consumer warps
    stage = pairs * (4 * nch * (SH + R + 12) + 32 * NH + 48)
    epi = 4 * pairs * (ncw * NH + 2 * ncw + 10 * NH)
    return 4 * (nch * FC * (R + 4) + nch * SH * (FC + 4) + 2 * (NH + 1) * SH) + stages * stage + epi


def tiles(variant):
    """{"fwd", "bwd"}: the attention kernels' Tile on this GPU, or None where none fits its
    shared memory. Of the tiles that fit a block (the largest first, two stages before one, the
    backward's channels in one CTA before a CTA each), the first of which an SM holds as many
    CTAs as the kernel is written for (two where a CTA has four heads), else the first."""
    return _tiles(variant, smem())


@functools.cache
def _tiles(variant, limits):
    per_block, per_sm, reserved = limits
    NH = variant[1]

    def pick(size, pairs, splits):
        first = None
        for split, stages, p in itertools.product(splits, (2, 1), pairs):
            nbytes = size(variant, p, stages, split)
            if nbytes <= min(per_block, per_sm - reserved):
                most = 2 if split and NH == 4 else 1
                t = Tile(p, stages, split, min(most, per_sm // (nbytes + reserved)), nbytes)
                if t.ctas == most:
                    return t
                first = first or t
        return first
    fwd = pick(fwd_smem, (64, 32, 16), (True,))
    bwd = pick(bwd_smem, (32, 16), (False, True) if NH == 4 else (True,))
    return None if fwd is None or bwd is None else {"fwd": fwd, "bwd": bwd}


@functools.cache
def smem():
    """The first GPU's shared memory in bytes: per block (opt-in), per SM, and the reserve per
    block (the driver's attributes; the process's GPUs taken alike, as for the architecture)."""
    cu, dev = ctypes.CDLL("libcuda.so.1"), ctypes.c_int()
    if cu.cuInit(0) or cu.cuDeviceGet(ctypes.byref(dev), jax.local_devices()[0].local_hardware_id):
        raise BuildError("Flash CUDA kernels: no GPU for the CUDA driver")

    def attribute(a):
        v = ctypes.c_int()
        if cu.cuDeviceGetAttribute(ctypes.byref(v), a, dev):
            raise BuildError(f"Flash CUDA kernels: the GPU's attribute {a} not found (CUDA driver)")
        return v.value
    # CU_DEVICE_ATTRIBUTE_ MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, _PER_MULTIPROCESSOR, RESERVED_SHARED_MEMORY_PER_BLOCK
    return tuple(attribute(a) for a in (97, 81, 111))


def nvcc_path():
    for cand in (os.environ.get("SO3LR_NVCC"), os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"),
                 shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if cand and os.path.isfile(cand):
            return cand
    raise RuntimeError("nvcc not found (set SO3LR_NVCC or CUDA_HOME)")


@functools.cache
def nvcc_version(nvcc):
    """nvcc's version banner: part of the cache key, so a toolkit upgraded in place rebuilds."""
    return subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout


def arch():
    """The sm_XX of the first GPU (sm_90a for compute capability 9)."""
    major, minor = jax.devices()[0].compute_capability.split(".")           # e.g. "9.0"
    return f"sm_{major}{minor}{'a' if major == '9' else ''}"


@functools.cache
def problem():
    """Why the kernels cannot be built here (None: they can): no GPU, an AMD GPU, one of compute
    capability below 8.0 (the kernels' TF32 tensor-core products), no nvcc. Decided once per
    process."""
    try:
        dev = jax.devices()[0]
    except Exception as exc:                 # no backend at all
        return f"no JAX device ({exc})"
    if dev.platform != "gpu":
        return f"no GPU (this process has {dev.platform})"
    if is_rocm():
        return "an AMD GPU"
    if int(dev.compute_capability.split(".")[0]) < 8:
        return f"a GPU of compute capability {dev.compute_capability} (the kernels need 8.0 or newer)"
    try:
        smem()
    except (OSError, BuildError) as exc:      # the driver library, or its answer
        return f"the GPU's shared memory unknown ({exc})"
    try:
        nvcc_path()
    except RuntimeError as exc:
        return str(exc)
    return None


def load_kernels(variant):
    """{"fwd", "bwd", "lr"}: the loaded libraries of a model variant on this GPU, built where
    the cache has none: the three at once, under a lock (processes starting together on an
    empty cache build once), each keyed on its own source and flags (the long range's library
    is shared by the variants)."""
    why = problem()
    if why:
        raise RuntimeError(f"Flash CUDA kernels: {why}")
    nvcc, sm = nvcc_path(), arch()
    root = os.environ.get("SO3LR_CUDA_CACHE", os.path.expanduser("~/.cache/so3lr/flash_cuda"))
    os.makedirs(root, exist_ok=True)
    headers = sorted(glob.glob(os.path.join(HERE, "*.cuh")))
    outs = {}
    for name, (src, flags) in kernel_flags(variant).items():
        h = hashlib.sha256()
        for path in [os.path.join(HERE, src)] + headers:
            with open(path, "rb") as f:
                h.update(f.read())
        h.update((" ".join(flags) + nvcc_version(nvcc) + sm + jax.__version__).encode())
        outs[name] = (os.path.join(root, f"lib_{name}_{h.hexdigest()[:16]}.so"), src, flags)
    if not all(os.path.isfile(out) for out, _, _ in outs.values()):
        import fcntl
        with open(os.path.join(root, ".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _build({name: v for name, v in outs.items() if not os.path.isfile(v[0])}, nvcc, sm)
    libs = {}
    for name, (out, _, _) in outs.items():
        try:
            libs[name] = ctypes.cdll.LoadLibrary(out)
        except OSError as exc:
            raise BuildError(f"Flash CUDA kernels: cannot load {out} ({exc})") from exc
    return libs


def _build(todo, nvcc, sm):
    """nvcc on the libraries in todo {name: (path, source, flags)}, at once; each to its
    path when done."""
    inc = jax.ffi.include_dir()
    procs = {}
    for name, (out, src, flags) in todo.items():
        tmp = f"{out}.{os.getpid()}.tmp"
        cmd = [nvcc, "-O3", f"-arch={sm}", "-std=c++17", "--shared", "-Xcompiler", "-fPIC",
               "-diag-suppress", "940,2473,177,550", f"-I{inc}", f"-I{HERE}", *flags, os.path.join(HERE, src), "-o", tmp]
        logger.info("Flash: building CUDA kernel library %s (%s, once)", name, sm)
        procs[name] = (subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True), tmp, out, src)
    try:
        for proc, tmp, out, src in procs.values():
            _, err = proc.communicate()
            if proc.returncode != 0:
                raise BuildError(f"Flash CUDA kernels: nvcc failed for {src}:\n{err[-4000:]}")
            os.replace(tmp, out)
    finally:
        for proc, tmp, _, _ in procs.values():
            if proc.poll() is None:
                proc.kill()
            if os.path.exists(tmp):
                os.remove(tmp)
