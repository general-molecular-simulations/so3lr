"""The CUDA kernels built with nvcc on first use, each library cached by its source (and the
headers), flags, the compiler's version, the GPU architecture and the JAX version
(``$SO3LR_CUDA_CACHE``, default ``~/.cache/so3lr/flash_cuda``). ``$SO3LR_CUDA_FLAGS`` adds
nvcc options: ``-lineinfo``, or one of the kernels' ablation switches for a timing-only
build (wrong results): ``-DNOMMA`` ``-DNOPROD`` ``-DNOQK`` ``-DNOWST`` (the forward),
``-DB_NOMMA`` ``-DB_NOPROD`` ``-DB_NOEPI`` ``-DB_NOQK`` (the backward), ``-DNOPAIR`` (the long
range)."""
import ctypes
import functools
import glob
import hashlib
import logging
import os
import shutil
import subprocess

import jax

from ..check import is_rocm

logger = logging.getLogger("SO3LR")
HERE = os.path.dirname(os.path.abspath(__file__))


class BuildError(RuntimeError):
    """nvcc failed, or a built library does not load."""


# library: (source, flags) for a model variant. Four heads (so3lr-tauri-s/m, so3lr): the
# forward one channel per CTA (two per SM), 64-pair tiles, the backward both channels in one
# CTA, 32-pair tiles. Eight heads (so3lr-tauri-l): every head of a channel in one CTA (one per
# SM) in both, the backward's two channel CTAs writing two copies of the per-slot forces and ev
# gradients (``bwd_copies``). The gather: two warps per atom (four with eight heads). The long
# range: fast math.
def kernel_flags(variant):
    """{"fwd", "bwd", "lr"}: (source, flags) for ``variant`` = (features, heads, components,
    hidden units, degrees as nibbles, rank)."""
    F, NH, M, SH, degs, rank = variant
    v = [f"-DVF={F}", f"-DVNH={NH}", f"-DVM={M}", f"-DVSH={SH}", f"-DVDEGS={degs:#x}", f"-DRANK={rank}"]
    extra = os.environ.get("SO3LR_CUDA_FLAGS", "").split()
    return {
        "fwd": ("attn_pair7.cu", v + ["-DCHSPLIT", "-DTP=64", "-DLDM"] + extra),
        "bwd": ("attn_bwd7.cu", v + (["-DCHSPLIT"] if NH == 8 else []) + ["-DTP=32", "-DGSPLIT=2", "-DLDM"] + extra),
        "lr": ("lr.cu", ["--use_fast_math"] + extra),
    }


def bwd_copies(variant):
    """The copies of the per-slot forces and ev gradients the backward of ``variant`` writes:
    one per channel where each channel runs in its own CTA."""
    return 2 if "-DCHSPLIT" in kernel_flags(variant)["bwd"][1] else 1


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
    """Why the kernels cannot be built here (None: they can): no GPU, an AMD GPU, a GPU of
    another architecture than sm_90, no nvcc. Decided once per process."""
    try:
        dev = jax.devices()[0]
    except Exception as exc:                 # no backend at all
        return f"no JAX device ({exc})"
    if dev.platform != "gpu":
        return f"no GPU (this process has {dev.platform})"
    if is_rocm():
        return "an AMD GPU"
    sm = arch()
    if not sm.startswith("sm_9"):
        return f"a {sm} GPU (the kernels are written for sm_90)"
    try:
        nvcc_path()
    except RuntimeError as exc:
        return str(exc)
    return None


@functools.cache
def load_kernels(variant):
    """{"fwd", "bwd", "lr"}: the loaded libraries of a model variant, built where the cache
    has none: the three at once, under a lock (processes starting together on an empty cache
    build once), each keyed on its own source and flags (the long range's library is shared
    by the variants)."""
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
