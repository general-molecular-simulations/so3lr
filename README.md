![workflow-test-ci](https://github.com/general-molecular-simulations/so3lr/actions/workflows/CI.yml/badge.svg)
[![docs](https://img.shields.io/badge/docs-so3lr.github.io-4a9d4a)](https://so3lr.github.io/)
[![examples-link](https://img.shields.io/badge/example-notebooks-F37726)](./examples)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb)
[![cite-link](https://img.shields.io/badge/how_to-cite-000000)](#citation)
[![preprint-link](https://img.shields.io/badge/paper-JACS-FFCE34)](https://doi.org/10.1021/jacs.5c09558)
[![data](https://zenodo.org/badge/DOI/10.5281/zenodo.14779793.svg)](https://doi.org/10.5281/zenodo.14779793)

![Logo](./logo.png)

**SO3LR** — pronounced *Solar* — is a pretrained machine-learned force field for
(bio)molecular simulations. It combines the fast and stable SO3krates neural
network for semi-local interactions with universal pairwise force fields for
short-range repulsion, long-range electrostatics and dispersion.

📖 **[Documentation: so3lr.github.io](https://so3lr.github.io/)**

## Installation

Try SO3LR with no installation in the
[Colab notebook](https://colab.research.google.com/github/general-molecular-simulations/so3lr/blob/main/examples/so3lr_colab_example.ipynb).

SO3LR supports Python 3.12 and 3.13. Install it from a checkout with
[uv](https://docs.astral.sh/uv/) and the committed lockfile:

```shell
git clone https://github.com/general-molecular-simulations/so3lr.git
cd so3lr
uv sync --locked                  # CPU
uv sync --locked --extra cuda12   # or: CUDA 12 on Linux x86-64
uv run so3lr --help
```

Already have JAX on a cluster (CUDA or ROCm)? Run `uv pip install .` in that
environment. Optional extras and other options:
[installation guide](https://so3lr.github.io/installation/).

## Quick start

Units are eV and Å. The default long-range cutoff, 12 Å, targets periodic
condensed-phase systems; for gas-phase molecules use 100 Å (`--lr-cutoff 100`).

```shell
so3lr opt --input geometry.xyz --force-conv 0.05
so3lr nvt --input geometry.xyz --temperature 300 --dt 0.5 --md-cycles 100 --md-steps 100
so3lr npt --input geometry.xyz --temperature 300 --pressure 1.0 --dt 0.5 --md-cycles 100 --md-steps 100
so3lr eval --datafile dataset.extxyz --lr-cutoff 100 --save-to predictions.extxyz
```

For longer runs, use a settings file: `so3lr --settings settings.yaml`; see
[molecular dynamics](https://so3lr.github.io/usage/md/).

In Python, use SO3LR as an ASE calculator:

```python
from ase.io import read
from so3lr import So3lrCalculator

atoms = read("geometry.xyz")
atoms.info["charge"] = 0          # total charge; 0 if unset
atoms.calc = So3lrCalculator(model="so3lr-tauri-m", lr_cutoff=100)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Models

The models ship inside the package; select one by name, no download needed.
The SO3LR Tauri models cover H to At (Z = 1–85).

- **`so3lr-tauri-m`** *(default)*: start here; the model we use for biomolecular MD.
- **`so3lr-tauri-l`**: the most accurate, recommended for reactions; 3.7× slower
  than `-m`.
- **`so3lr-tauri-s`**: 1.4× faster than `-m`.
- **`so3lr`**: the original model from the
  [JACS paper](https://doi.org/10.1021/jacs.5c09558), for H, C, N, O, F, P, S and
  Cl, kept to reproduce published results.

Speeds are for MD of 12,288-atom water with PME on one NVIDIA A100.

Use the same name everywhere (`So3lrCalculator(model=...)`, `--model`, `model:`
in a settings file), or a path for a fine-tuned model. To pin a release for
papers and benchmarks, add its date: `so3lr-tauri-m-2606`.

## Documentation

| | |
|---|---|
| [Quick start](https://so3lr.github.io/quickstart/) | First simulation, step by step |
| [Models](https://so3lr.github.io/models/) | Sizes, naming, checksums, custom models |
| [Command line](https://so3lr.github.io/usage/cli/) · [Settings file](https://so3lr.github.io/usage/settings/) | Every command, flag and YAML key |
| [Molecular dynamics](https://so3lr.github.io/usage/md/) | Ensembles, engines, restarts, frozen atoms, output |
| [ASE calculator](https://so3lr.github.io/usage/ase/) · [Dataset evaluation](https://so3lr.github.io/usage/evaluation/) | Python interface, metrics and predictions |
| [Long-range electrostatics](https://so3lr.github.io/advanced/electrostatics/) | Ewald / PME / P3M and `so3lr tune-ewald` |
| [Enhanced sampling](https://so3lr.github.io/advanced/enhanced-sampling/) | PySAGES methods and collective variables |
| [Observables](https://so3lr.github.io/advanced/observables/) · [Gradients and Hessians](https://so3lr.github.io/advanced/derivatives/) | Energy components, charges, embeddings, derivatives |
| [Binding energies](https://so3lr.github.io/advanced/binding-energies/) | Dimer interaction energies |
| [Fine-tuning](https://so3lr.github.io/training/finetuning/) · [Training from scratch](https://so3lr.github.io/training/from-scratch/) | Adapting the model to your data |
| [API reference](https://so3lr.github.io/reference/api/) · [FAQ](https://so3lr.github.io/reference/faq/) | The Python surface, common questions |

The [`examples`](./examples) folder has notebooks for evaluation, NVT/NPT with
JAX-MD and a production run.

## Datasets

The quantum-mechanical datasets used to train and test SO3LR are on Zenodo:
[10.5281/zenodo.14779793](https://doi.org/10.5281/zenodo.14779793).

## Contributing

The code and models are still evolving and feedback is invaluable: please
[open an issue](https://github.com/general-molecular-simulations/so3lr/issues)
for errors, inconsistencies or feature ideas. See [`DEVELOPERS.md`](./DEVELOPERS.md)
for development environments, tests and the lockfile policy.

## Citation

If you use SO3LR, please cite the paper below and give the model you used, e.g.
`so3lr-tauri-m-2606`:

```bibtex
@article{kabylda2025molecular,
  title={Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},
  author={Kabylda, Adil and Frank, J Thorben and Su{\'a}rez-Dou, Sergio and Khabibrakhmanov, Almaz and Medrano Sandonas, Leonardo and Unke, Oliver T and Chmiela, Stefan and M{\"u}ller, Klaus-Robert and Tkatchenko, Alexandre},
  journal={Journal of the American Chemical Society},
  volume={147},
  number={37},
  pages={33723},
  year={2025},
  doi={10.1021/jacs.5c09558}
}

@article{frank2024euclidean,
  title={A Euclidean transformer for fast and stable machine learned force fields},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert and Chmiela, Stefan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6539},
  year={2024},
  doi={10.1038/s41467-024-50620-6}
}
```

## License

SO3LR is released under the [MIT License](./LICENSE), except the robust loss in
[`so3lr/mlff/utils/robust_loss_jax/`](./so3lr/mlff/utils/robust_loss_jax/),
derived from Google Research, which remains under Apache-2.0 (see the `LICENSE`
there). These licenses cover the source code; they do not relicense the
checkpoints, datasets or scientific reference data.
