"""Central registry resolving SO3LR model names to bundled checkpoint directories.

A "model" can be referred to either by a registered name or by an arbitrary
filesystem path to a custom/finetuned model directory.

Names follow ``so3lr-<family>-<size>[-<release>]`` and come in three tiers:

=======================  =======================================  ==============
Name                     Means                                    Stability
=======================  =======================================  ==============
``so3lr-m``              current recommended M, whichever family  moves *across*
                                                                  families
``so3lr-protos-m``       current M within the Protos family       moves *within*
                                                                  the family
``so3lr-protos-m-2606``  one exact release                        immutable
=======================  =======================================  ==============

``<family>`` names an architecture lineage (all Protos models share the same
base network; a future family may not). ``<size>`` is family-relative: sizes are
comparable within a family but not necessarily across families. ``<release>`` is
the ``YYMM`` of the release.

Only canonical names correspond 1:1 to a directory under ``models/``; the other
tiers are aliases. Cite the canonical name in papers and benchmark tables --
floating aliases are resolved at load time and log which target they picked.

This is the single source of truth used by the Python API (``make_so3lr``,
``make_ase_calculator``, ``make_potential_fn``), the ASE calculator, and the CLI.
"""

import dataclasses
import logging
import pathlib

logger = logging.getLogger(__name__)

# .../so3lr/
_PKG = pathlib.Path(__file__).parent.resolve()
_MODELS = _PKG / "models"

# Theory-level head used at inference by every model shipped so far. Kept as a
# module constant for backwards compatibility, but the authoritative value is
# per-model (``ModelSpec.head_index``) so a future family can differ.
DEFAULT_HEAD_INDEX = 5


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    """Everything the loaders need in order to load a model.

    Attributes:
        dir: Directory holding ``params.pkl`` and/or a ``checkpoints/`` folder.
        family: Architecture lineage (``"protos"``, ``"vanilla"``, or
            ``"custom"`` for user-supplied paths).
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
    "so3lr-protos-s-2602": ModelSpec(_MODELS / "so3lr-protos-s-2602", family="protos"),
    "so3lr-protos-m-2602": ModelSpec(_MODELS / "so3lr-protos-m-2602", family="protos"),
    "so3lr-protos-l-2602": ModelSpec(_MODELS / "so3lr-protos-l-2602", family="protos"),
    "so3lr-protos-s-2606": ModelSpec(_MODELS / "so3lr-protos-s-2606", family="protos"),
    "so3lr-protos-m-2606": ModelSpec(_MODELS / "so3lr-protos-m-2606", family="protos"),
    "so3lr-protos-l-2606": ModelSpec(_MODELS / "so3lr-protos-l-2606", family="protos"),
    # Original vanilla SO3LR, predating the family scheme. Kept reachable by
    # name for reproducibility.
    "so3lr": ModelSpec(_MODELS / "so3lr", family="vanilla"),
}

# Moving pointers. What these resolve to changes when a new release lands, so
# resolving one emits a log line naming the canonical target it picked.
_FLOATING_ALIASES = {
    "so3lr-s": "so3lr-protos-s-2606",
    "so3lr-m": "so3lr-protos-m-2606",
    "so3lr-l": "so3lr-protos-l-2606",
    "so3lr-protos-s": "so3lr-protos-s-2606",
    "so3lr-protos-m": "so3lr-protos-m-2606",
    "so3lr-protos-l": "so3lr-protos-l-2606",
}

# Fixed synonyms: another spelling of one canonical name, never repointed.
_SYNONYMS = {
    "so3lr-o": "so3lr",
}

# Model used when nothing is specified (bare calculator / CLI without --model /
# YAML without a model key). Pinned to a canonical name so the default can never
# drift silently when a new release lands.
DEFAULT_MODEL = "so3lr-protos-m-2606"

# Aliases already logged, so resolving inside a loop does not spam the log.
_LOGGED_ALIASES = set()


def canonical_models():
    """Return the tuple of canonical model names (one per bundled directory)."""
    return tuple(_CANONICAL)


def model_aliases():
    """Return a mapping of alias name -> canonical name."""
    return {**_FLOATING_ALIASES, **_SYNONYMS}


def available_models():
    """Return every accepted model name, canonical names first."""
    return canonical_models() + tuple(_FLOATING_ALIASES) + tuple(_SYNONYMS)


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

    if key in _SYNONYMS:
        key = _SYNONYMS[key]

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
