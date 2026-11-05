__version__ = "0.2.0"

# Lazy public API (PEP 562): importing `so3lr` (or `from so3lr import __version__`)
# must stay cheap and NOT pull in jax/flax/jax-md/glp — otherwise `so3lr --help`
# pays full JAX initialization. The heavy submodules are imported only when one
# of these names is actually accessed.
__all__ = ["So3lrCalculator", "FlashCalculator", "Graph", "to_jax_md", "So3lrPotential", "So3lr"]


def __getattr__(name):
    if name == "So3lrCalculator":
        from .ase_utils import make_ase_calculator
        return make_ase_calculator
    if name == "FlashCalculator":
        from .flash.calculator import FlashCalculator
        return FlashCalculator
    if name == "Graph":
        from .graph import Graph
        return Graph
    if name == "to_jax_md":
        from .jaxmd_utils import to_jax_md
        return to_jax_md
    if name == "So3lrPotential":
        from .potential import make_potential_fn
        return make_potential_fn
    if name == "So3lr":
        from .base_calculator import make_so3lr
        return make_so3lr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
