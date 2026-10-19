#!/usr/bin/env python3
from importlib.resources import files

import numpy as np

species = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
    'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
    'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No'
]

# Legacy SO3LR (v1) free-atom reference polarizabilities and C6 coefficients.
# The original v1 model was calibrated against these values; the new so3lr-tauri-s/m/l
# models use the refitted ``alphas``/``C6_coef`` defined below. ``mixing_rules``
# selects between the two via its ``legacy`` flag.
alphas_legacy = np.array(
    [4.5, 1.38, 164.2, 38., 21., 12., 7.4, 5.4, 3.8,
     2.67, 162.7, 71., 60., 37., 25., 19.6, 15., 11.1,
     292.9, 160., 120., 98., 84., 78., 63., 56., 50.,
     48., 42., 40., 60., 41., 29., 25., 20., 16.8,
     319.2, 199., 126.74, 119.97, 101.6, 88.42, 80.08, 65.89, 56.1,
     23.68, 50.6, 39.7, 70.22, 55.95, 43.67, 37.65, 35., 27.3,
     399.9, 275., 213.7, 204.7, 215.8, 208.4, 200.2, 192.1, 184.2,
     158.3, 169.5, 164.64, 156.3, 150.2, 144.3, 138.9, 137.2, 99.52,
     82.53, 71.04, 63.04, 55.06, 42.51, 39.68, 36.5, 33.9, 69.92,
     61.8, 49.02, 45.01, 38.93, 33.54, 317.8, 246.2, 203.3, 217.,
     154.4, 127.8, 150.5, 132.2, 131.2, 143.6, 125.3, 121.5, 117.5,
     113.4, 109.4, 105.4
     ]
)

C6_coef_legacy = np.array(
    [6.50000e+00, 1.46000e+00, 1.38700e+03, 2.14000e+02, 9.95000e+01,
     4.66000e+01, 2.42000e+01, 1.56000e+01, 9.52000e+00, 6.38000e+00,
     1.55600e+03, 6.27000e+02, 5.28000e+02, 3.05000e+02, 1.85000e+02,
     1.34000e+02, 9.46000e+01, 6.43000e+01, 3.89700e+03, 2.22100e+03,
     1.38300e+03, 1.04400e+03, 8.32000e+02, 6.02000e+02, 5.52000e+02,
     4.82000e+02, 4.08000e+02, 3.73000e+02, 2.53000e+02, 2.84000e+02,
     4.98000e+02, 3.54000e+02, 2.46000e+02, 2.10000e+02, 1.62000e+02,
     1.29600e+02, 4.69100e+03, 3.17000e+03, 1.96858e+03, 1.67791e+03,
     1.26361e+03, 1.02873e+03, 1.39087e+03, 6.09750e+02, 4.69000e+02,
     1.57500e+02, 3.39000e+02, 4.52000e+02, 7.07050e+02, 5.87420e+02,
     4.59320e+02, 3.96000e+02, 3.85000e+02, 2.85900e+02, 6.84600e+03,
     5.72700e+03, 3.88450e+03, 3.70833e+03, 3.91184e+03, 3.90875e+03,
     3.84768e+03, 3.70869e+03, 3.51171e+03, 2.78153e+03, 3.12441e+03,
     2.98429e+03, 2.83995e+03, 2.72412e+03, 2.57678e+03, 2.38753e+03,
     2.37180e+03, 1.27480e+03, 1.01992e+03, 8.47930e+02, 7.10200e+02,
     5.96670e+02, 3.59100e+02, 3.47100e+02, 2.98000e+02, 3.92000e+02,
     7.17440e+02, 6.97000e+02, 5.71000e+02, 5.30920e+02, 4.57530e+02,
     3.90630e+02, 4.22444e+03, 4.85132e+03, 3.60441e+03, 4.04754e+03,
     2.36742e+03, 1.87710e+03, 2.50788e+03, 2.11727e+03, 2.11098e+03,
     2.40322e+03, 1.98582e+03, 1.89192e+03, 1.85110e+03, 1.78707e+03,
     1.70100e+03, 1.57818e+03
     ]
)

alphas = np.array(
        [4.6013, 1.307572, 79.382988, 31.560228, 18.93866, 11.205754, 7.116014, 5.269107,
        3.83477, 2.853014, 85.572388, 52.277222, 48.47131, 33.84922, 23.790392, 18.445013,
        14.10233, 10.928427, 146.909647, 99.958761, 72.61107, 60.064834, 50.850333, 43.815038,
        38.270106, 33.788789, 30.093825, 26.997299, 20.710185, 22.10799, 34.942462, 33.681677,
        28.399077, 23.321302, 19.19348, 15.950471, 154.780898, 123.618505, 85.401207, 70.019557,
        52.56491, 43.648334, 37.018289, 31.977683, 28.064213, 24.968207, 22.47767, 25.935647,
        41.770813, 43.535418, 39.325673, 34.110658, 29.340869, 25.297729, 180.259764, 160.89729,
        106.0452, 114.647741, 107.761198, 103.832022, 100.299301, 97.037549, 93.97242, 83.586623,
        88.262218, 85.566074, 82.954507, 80.41711, 77.946292, 75.536481, 69.940144, 59.684519,
        50.597627, 43.303031, 37.509873, 32.865022, 29.087295, 22.993127, 20.405648, 21.151891,
        38.270557, 43.929399, 42.06478, 38.019983, 33.771328, 29.904324]
)

C6_coef = np.array(
        [6.916624, 1.397397, 575.475708, 179.386322, 98.951836, 50.583488, 27.934183, 18.739491, 
         12.341105, 8.315834, 650.263489, 393.328064, 405.658264, 276.749054, 184.279343, 135.418503, 
         96.937271, 70.413261, 1401.339569, 1129.488823, 756.106511, 589.053867, 468.548409, 379.729743, 
         312.602904, 260.72911, 219.876294, 187.181687, 116.577838, 138.872935, 265.020574, 280.410762, 
         242.502866, 198.300121, 160.355477, 130.094387, 1609.76002, 1618.321683, 1067.496571, 847.654158, 
         548.890813, 432.495863, 347.767869, 284.883393, 237.219306, 200.366207, 171.353258, 213.266544, 
         390.734521, 451.276909, 427.633152, 377.50486, 325.229538, 278.357742, 2208.91417, 2540.699298, 
         1603.955128, 1737.696688, 1484.208888, 1394.80338, 1315.600646, 1244.086358, 1178.621008, 1083.884791, 
         1061.69603, 1008.86115, 959.141391, 912.192997, 867.743255, 825.572174, 809.380719, 694.262769, 
         580.226857, 485.665646, 409.916812, 349.28229, 300.318606, 216.950506, 187.244403, 199.769249, 
         373.259116, 476.679798, 488.446995, 457.82145, 413.916128, 368.90949]
)

# 0-based NLH coefficient tables (float64): shape = (92, 92, 3)
# Use with z_i = atomic_numbers[idx_i] - 1, z_j = atomic_numbers[idx_j] - 1
with (files('so3lr.mlff.nn.observable') / 'nlh_ref.npz').open('rb') as _f, np.load(_f) as _nlh:
    NLH_AA = _nlh['NLH_AA']
    NLH_BB = _nlh['NLH_BB']


# ---------------------------------------------------------------------------
# Which elements each model may be asked about
# ---------------------------------------------------------------------------
# Two independent limits, and the stricter one wins.
#
# 1. Reference-table coverage. Every table above is indexed with ``Z - 1``, so a
#    table with ``n`` rows covers Z = 1..n. The tables are NOT all the same
#    length and the smallest binds:
#      NLH_AA / NLH_BB   (92)  -- NLH repulsion, used by so3lr-tauri-s/m/l
#      alphas / C6_coef  (86)  -- refitted alpha_0 / C6 free-atom references
#      *_legacy          (102) -- v1 SO3LR references
#
# 2. What the network was actually TRAINED on, which is stricter than any table
#    and is what really matters: a table can hold parameters for an element the
#    network never saw, and the network will still return a confident number.
#    v1 SO3LR covers only the eight biomolecular elements; the v2 models were
#    trained across Z = 1..85.
#
# Out-of-range Z is not a clean failure without a check: the gathers in
# ``mixing_rules`` / ``NLHRepulsionSparse`` are ``jnp.take`` under jit, so an
# index past the end is resolved by JAX's out-of-bounds policy instead of
# raising -- a plausible-looking number belonging to some other element.
MAX_Z_NLH = int(NLH_AA.shape[0])                     # 92 (H..U)
MAX_Z_DISPERSION = int(np.asarray(alphas).shape[0])   # 86 (H..Rn)
MAX_Z_DISPERSION_LEGACY = int(np.asarray(alphas_legacy).shape[0])  # 102 (H..Fm)

# v1 SO3LR was trained on biomolecular chemistry only. This is a SET, not a
# range: Z <= 17 would wrongly admit Ne, Na, Mg, Al and Si.
SO3LR_V1_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17})  # H C N O F P S Cl
MAX_Z_TRAINED = 85                                          # v2 models: H..At


def supported_atomic_numbers(legacy: bool = False, dispersion: bool = True):
    """The set of atomic numbers a model may be asked about.

    Args:
        legacy: ``True`` for v1 SO3LR (``legacy_so3lr_bool``), which was trained
            on the eight biomolecular elements only. Its training set is well
            inside every table, so ``dispersion`` does not affect it.
        dispersion: Whether the dispersion term is active. When it is off, only
            the repulsion table binds.

    Returns:
        ``frozenset`` of permitted Z, the intersection of what the reference
        tables cover with what the model was trained on.
    """
    if legacy:
        return SO3LR_V1_ELEMENTS
    table_limit = min(MAX_Z_NLH, MAX_Z_DISPERSION) if dispersion else MAX_Z_NLH
    return frozenset(range(1, min(MAX_Z_TRAINED, table_limit) + 1))


def supported_atomic_numbers_for_config(model_cfg):
    """Element coverage for a model's ``hyperparameters['model']`` section."""
    return supported_atomic_numbers(
        legacy=model_cfg.get('legacy_so3lr_bool', False) is True,
        dispersion=model_cfg.get('dispersion_energy_bool', True) is True,
    )


def supported_atomic_numbers_for_model(model=None):
    """Element coverage of a bundled model name or custom model directory."""
    import json
    from so3lr.model_registry import resolve_model
    with open(resolve_model(model).dir / "hyperparameters.json") as fp:
        return supported_atomic_numbers_for_config(json.load(fp)["model"])


def check_atomic_numbers_supported(atomic_numbers, supported, *, context=None):
    """Raise ``ValueError`` if any atomic number is outside ``supported``.

    Call this at eager (non-jit) boundaries -- reading a structure, building a
    graph -- where a useful message can still be produced. Padding atoms (Z=0)
    must be excluded by the caller; here they are reported as invalid, since at
    these boundaries the input is real atoms only.

    Args:
        atomic_numbers: Array-like of integer atomic numbers.
        supported: Container of permitted Z, from :func:`supported_atomic_numbers`.
            ``None`` means the caller could not determine the model's coverage
            (e.g. a potential built by an older code path); the check is then
            skipped rather than failing on every structure.
        context: Optional string (e.g. a filename) prefixed to the message.

    Raises:
        ValueError: If any atomic number is not in ``supported``.
    """
    if supported is None:
        return
    z = np.asarray(atomic_numbers).reshape(-1)
    if z.size == 0:
        return
    allowed = sorted(int(x) for x in supported)
    bad = np.unique(z[~np.isin(z, allowed)])
    if bad.size == 0:
        return
    named = ', '.join(
        f'{species[int(b) - 1]} (Z={int(b)})' if 1 <= int(b) <= len(species) else f'Z={int(b)}'
        for b in bad
    )
    if allowed == list(range(1, allowed[-1] + 1)):
        allowed_txt = f'Z = 1..{allowed[-1]}'
    else:
        allowed_txt = 'only ' + ', '.join(
            f'{species[z0 - 1]}' for z0 in allowed if 1 <= z0 <= len(species))
    where = f'{context}: ' if context else ''
    raise ValueError(
        f'{where}unsupported element(s) for this model: {named}. This model '
        f'supports {allowed_txt}; outside that the network was never trained '
        f'and would silently return values gathered from the wrong element '
        f'rather than fail. Remove those atoms or use a model that covers them.'
    )
