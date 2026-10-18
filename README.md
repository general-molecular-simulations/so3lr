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

On a cluster with a working JAX (CUDA or ROCm) install, `pip install .` into
that environment instead. JAX is only capped (`jax<0.11`), so an existing build
is kept; jax-md, glp, jax-pme and jraph come from pinned Git commits. Optional extras: `pysages` (enhanced sampling), `tfds` (TFDS training
loader), `wandb` (opt-in W&B logging), `matscipy` (faster neighbour search for
small periodic cells in the ASE calculator). For reproducing published results,
check out a release tag first. Details:
[installation guide](https://so3lr.github.io/installation/).

## Quick start

Units are eV and Å. The default long-range cutoff (`12` Å) targets periodic
condensed-phase systems; for gas-phase molecules use `100`.

```shell
so3lr opt --input geometry.xyz --force-conv 0.05
so3lr nvt --input geometry.xyz --temperature 300 --dt 0.5 --md-cycles 100 --md-steps 100
so3lr npt --input geometry.xyz --temperature 300 --pressure 1.0 --dt 0.5 --md-cycles 100 --md-steps 100
so3lr eval --datafile dataset.extxyz --lr-cutoff 100 --save-to predictions.extxyz
```

Longer runs are best driven by a settings file, which also covers features with
no CLI flag (frozen atoms, k-space dispersion, enhanced sampling):
`so3lr --settings settings.yaml`. Periodic MD on an NVIDIA GPU (compute
capability 8.0+) runs on the **flash** engine when the box is orthorhombic and
NPT is isotropic: fused kernels that keep a few KiB per atom, 2.5x faster than
the neighbour-list engine on 12,288 water atoms, up to 33.7 million atoms on one
H200, and on several GPUs of a node with `--flash-devices N`. The log names the
engine used and why; see [molecular dynamics](https://so3lr.github.io/usage/md/).

From Python, SO3LR is an ASE calculator:

```python
from ase.io import read
from so3lr import So3lrCalculator

atoms = read("geometry.xyz")
atoms.info["charge"] = 0          # total charge; 0 if unset
atoms.calc = So3lrCalculator(model="so3lr-tauri-m-2606", lr_cutoff=100)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Models

The **SO3LR Tauri** models ship in three sizes inside the package; select one by
name, no download needed.

| Model | Layers | Features | Heads | Cutoff (Å) |
|-------|--------|----------|-------|-----------|
| `so3lr-tauri-s-2606` | 2 | 128 | 4 | 4.5 |
| `so3lr-tauri-m-2606` *(default)* | 3 | 128 | 4 | 4.5 |
| `so3lr-tauri-l-2606` | 3 | 256 | 8 | 5.0 |

The same names work everywhere: `So3lrCalculator(model=...)`, `--model` on the
CLI, `model:` in a settings file; a path loads a fine-tuned or custom model. The
previous release is bundled as `so3lr-tauri-{s,m,l}-2602`, and the original
model from the [JACS paper](https://doi.org/10.1021/jacs.5c09558) as `so3lr`; it
was not trained on energies, so only its relative energies are meaningful.

> [!WARNING]
> `so3lr-tauri-s`, `so3lr-tauri-m` and `so3lr-tauri-l` are aliases for the
> current release and are repointed when a new one lands. Cite the dated name
> (e.g. `so3lr-tauri-m-2606`) in papers and benchmark tables.

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

If you use SO3LR, cite the software version (`so3lr.__version__`; metadata in
[`CITATION.cff`](./CITATION.cff)) and the dated model name, e.g. `so3lr-tauri-m-2606`,
together with:

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

SO3LR is released under the [MIT License](./LICENSE). Portions derived from
Google Research and JAX MD remain under [Apache-2.0](./LICENSE-APACHE-2.0), and
the Ewald tuning code derived from torch-pme under
[BSD-3-Clause](./LICENSE-BSD-3-CLAUSE); see
[`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md). These licenses cover the
source code; they do not relicense the checkpoints, datasets or scientific
reference data.
