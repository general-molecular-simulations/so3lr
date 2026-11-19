"""Choose a local JAX device for the requested SO3LR precision."""

from functools import lru_cache
import logging
import sys

import jax
import numpy as np


def precision_context(precision):
    if precision not in ("mixed", "float32"):
        raise ValueError("Choose mixed or float32 precision.")
    return jax.enable_x64(precision == "mixed")


def device_name(device):
    if device.platform == "cpu":
        return "CPU"
    return device.device_kind


def _probe_jax(device, precision):
    dtype = np.float64 if precision == "mixed" else np.float32
    with precision_context(precision), jax.default_device(device):
        x = jax.device_put(np.array([2.0], dtype=dtype), device)
        y = jax.jit(lambda a: a * a)(x)
        return y.devices() == {device} and y.dtype == dtype and float(y.item()) == 4.0


def _water_prediction(device, precision):
    from ase import Atoms
    from so3lr import So3lrCalculator

    with precision_context(precision), jax.default_device(device):
        calc = So3lrCalculator(model="so3lr-tauri-s", lr_cutoff=100.0,
                               precision=precision, has_aux=True,
                               observables=["partial_charges", "dipole_vec"])
        calculate = calc.calculate_fn

        def check_placement(*args):
            output = calculate(*args)
            if any(x.devices() != {device} for x in jax.tree.leaves(output)):
                raise RuntimeError("SO3LR model outputs were placed on another device.")
            return output

        calc.calculate_fn = check_placement
        atoms = Atoms("OH2", positions=[[0, 0, 0], [0.9572, 0, 0], [-0.24, 0.9266, 0]])
        atoms.calc = calc
        return (atoms.get_potential_energy(), atoms.get_forces(),
                calc.results["aux"]["partial_charges"])


@lru_cache(maxsize=1)
def _devices():
    cpu = jax.local_devices(backend="cpu")[0]
    # The first traced SO3LR inference affects cached constants used by later
    # mixed-precision jobs. Initialize them at full state precision on CPU;
    # otherwise a float32 first job perturbs subsequent mixed CPU results.
    _water_prediction(cpu, "mixed")
    devices = {"cpu": (cpu, ("mixed", "float32"))}
    if sys.platform == "darwin":
        return devices

    try:
        gpu_devices = jax.local_devices(backend="gpu")
    except RuntimeError:
        return devices
    for target in gpu_devices:
        supported = []
        for precision in ("mixed", "float32"):
            try:
                if _probe_jax(target, precision):
                    supported.append(precision)
            except RuntimeError as error:
                logging.warning("%s could not use %s precision: %s", target, precision, error)
        if supported:
            devices["gpu"] = (target, tuple(supported))
            break
    return devices


def choose_device(request, precision="mixed"):
    if precision not in ("mixed", "float32"):
        raise ValueError("Choose mixed or float32 precision.")
    devices = _devices()
    if request == "auto":
        request = "gpu" if "gpu" in devices and precision in devices["gpu"][1] else "cpu"
    if request not in ("gpu", "cpu"):
        raise ValueError("Choose Auto, CPU, or GPU.")
    if request not in devices or precision not in devices[request][1]:
        raise ValueError(f"{request.upper()} is unavailable in {precision} precision on this computer.")
    return request, devices[request][0]


def device_status():
    return {
        "default_device": choose_device("auto", "mixed")[0],
        "devices": [
            {"id": key, "name": device_name(target), "backend": target.platform,
             "precisions": list(precisions)}
            for key, (target, precisions) in _devices().items()
        ],
    }
