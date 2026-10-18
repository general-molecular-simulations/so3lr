"""Central registry resolving SO3LR model names to bundled checkpoint directories.

A "model" can be referred to either by a registered name or by an arbitrary
filesystem path to a custom/finetuned model directory.

Names follow ``so3lr-tauri-<size>[-<release>]`` and come in two tiers:

* ``so3lr-tauri-m-2606`` -- one exact release; immutable.
* ``so3lr-tauri-m`` -- current recommended M; moves when a new release lands.

``<size>`` is one of ``s``/``m``/``l``; ``<release>`` is the ``YYMM`` of the
release. The original SO3LR model is plain ``so3lr``. See
https://so3lr.github.io/models/ for the user-facing version of this.

Only canonical names correspond 1:1 to a directory under ``models/``; the bare
sizes are aliases. Cite the canonical name in papers and benchmark tables --
floating aliases are resolved at load time and log which target they picked.

This is the single source of truth used by the Python API (``make_so3lr``,
``make_ase_calculator``, ``make_potential_fn``), the ASE calculator, and the CLI.
"""

import dataclasses
import logging
import pathlib

# The package-wide logger the CLI attaches its console/file handlers to (see
# ``setup_logger`` in so3lr/cli/so3lr_md.py). Using ``__name__`` here would make
# this a sibling of "SO3LR" rather than a child, so the alias-redirect line below
# would be dropped on every CLI run.
logger = logging.getLogger("SO3LR")

# .../so3lr/
_PKG = pathlib.Path(__file__).parent.resolve()
_MODELS = _PKG / "models"

# Theory-level head used at inference by every model shipped so far.
#
# ``ModelSpec.head_index`` carries this per-model so a future lineage can differ,
# but only the ASE path (so3lr/ase_utils.py) currently threads it through:
# make_so3lr / make_potential_fn / the jax-md path still assume head 5 (see the
# hardcoded ``jnp.eye(16)[5]`` in so3lr/jaxmd_utils.py and the head_index=5
# defaults in so3lr/mlff/calculators/ase_calculator.py). A family that needs a
# different head must plumb spec.head_index through those entry points first --
# they would otherwise read the wrong head and return plausible wrong energies.
DEFAULT_HEAD_INDEX = 5


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Everything the loaders need in order to load a model.

    Attributes:
        dir: Directory holding ``params.pkl`` and/or a ``checkpoints/`` folder.
        family: Architecture lineage -- ``"v2"`` for the current bundled models,
            ``"vanilla"`` for the original v1 SO3LR, ``"custom"`` for
            user-supplied paths. Not part of the model name.
        head_index: Theory-level head index to read the prediction from.
    """

    dir: pathlib.Path
    family: str
    head_index: int = DEFAULT_HEAD_INDEX

    @property
    def from_file(self):
        """True when a canonical ``params.pkl`` export is present.

        Directories that ship only an orbax ``checkpoints/`` folder resolve to
        ``False`` and are loaded through the checkpoint manager instead.
        """
        return (self.dir / "params.pkl").exists()

    def _as_legacy_tuple(self):
        return (self.dir, self.from_file, self.head_index)

    # Backwards compatibility with the ``(dir, from_file, head_index)`` triple
    # this function used to return, so both ``a, b, c = resolve_model(...)`` and
    # ``resolve_model(...)[0]`` keep working. Prefer attribute access.
    def __iter__(self):
        return iter(self._as_legacy_tuple())

    def __getitem__(self, index):
        return self._as_legacy_tuple()[index]


# Canonical names -- exactly one per directory under models/.
_CANONICAL = {
    "so3lr-tauri-s-2602": ModelSpec(_MODELS / "so3lr-tauri-s-2602", family="v2"),
    "so3lr-tauri-m-2602": ModelSpec(_MODELS / "so3lr-tauri-m-2602", family="v2"),
    "so3lr-tauri-l-2602": ModelSpec(_MODELS / "so3lr-tauri-l-2602", family="v2"),
    "so3lr-tauri-s-2606": ModelSpec(_MODELS / "so3lr-tauri-s-2606", family="v2"),
    "so3lr-tauri-m-2606": ModelSpec(_MODELS / "so3lr-tauri-m-2606", family="v2"),
    "so3lr-tauri-l-2606": ModelSpec(_MODELS / "so3lr-tauri-l-2606", family="v2"),
    # Original vanilla SO3LR (v1). Kept reachable by name for reproducibility.
    "so3lr": ModelSpec(_MODELS / "so3lr", family="vanilla"),
}

# Moving pointers. What these resolve to changes when a new release lands, so
# resolving one emits a log line naming the canonical target it picked.
_FLOATING_ALIASES = {
    "so3lr-tauri-s": "so3lr-tauri-s-2606",
    "so3lr-tauri-m": "so3lr-tauri-m-2606",
    "so3lr-tauri-l": "so3lr-tauri-l-2606",
}

# Model used when nothing is specified (bare calculator / CLI without --model /
# YAML without a model key). Pinned to a canonical name so the default can never
# drift silently when a new release lands.
DEFAULT_MODEL = "so3lr-tauri-m-2606"

# Aliases already logged, so resolving inside a loop does not spam the log.
_LOGGED_ALIASES = set()


def canonical_models():
    """Return the tuple of canonical model names (one per bundled directory)."""
    return tuple(_CANONICAL)


def model_aliases():
    """Return a mapping of alias name -> canonical name."""
    return dict(_FLOATING_ALIASES)


def available_models():
    """Return every accepted model name, canonical names first."""
    return canonical_models() + tuple(_FLOATING_ALIASES)


def resolve_model(name_or_path=None):
    """Resolve a model name or filesystem path to a :class:`ModelSpec`.

    Args:
        name_or_path: A registered model name (see :func:`available_models`), a
            path to a custom model directory, or ``None`` (uses
            :data:`DEFAULT_MODEL`).

    Returns:
        A :class:`ModelSpec`. It also unpacks as the legacy
        ``(dir, from_file, head_index)`` triple.

    Raises:
        FileNotFoundError: If ``name_or_path`` is neither a registered name nor
            an existing path.
    """
    key = str(name_or_path) if name_or_path is not None else DEFAULT_MODEL

    if key in _FLOATING_ALIASES:
        target = _FLOATING_ALIASES[key]
        if key not in _LOGGED_ALIASES:
            _LOGGED_ALIASES.add(key)
            logger.info(
                "Model '%s' resolves to '%s'. This alias moves between releases "
                "- use the canonical name to pin results.", key, target
            )
        key = target

    if key in _CANONICAL:
        return _CANONICAL[key]

    workdir = pathlib.Path(key).expanduser().resolve()
    if not workdir.exists():
        raise FileNotFoundError(
            f"Model '{key}' is not a known name {available_models()} "
            f"and not an existing path."
        )
    return ModelSpec(workdir, family="custom")
