# Reproducible environments

SO3LR uses [uv](https://docs.astral.sh/uv/) and the committed `uv.lock` as the
authoritative development and paper-reproduction environment. The lock records
exact package versions, artifact hashes, and Git commit IDs for every supported
platform. The `.python-version` file pins the default interpreter used to create
and validate release artifacts.

## Authoritative reproduction artifact

Run the paper workflows from a versioned GitHub source archive or an exact
checkout of its recorded 40-character commit. That source artifact is the
reproduction bundle: it contains `uv.lock`, `.python-version`, the notebooks, and
their repository input data. Record the source-archive SHA-256 beside the paper's
dataset and model manifests.

The Python wheel and sdist are installation artifacts, not self-contained paper
reproduction bundles. They deliberately omit the repository notebooks and input
data; consumers of either Python artifact must obtain the matching source
archive before following the commands below. A wheel/sdist smoke test establishes
installability, while `uv sync --locked` from the source archive establishes the
frozen environment.

## CPU environment

```sh
uv sync --locked --group dev
uv run pytest tests/
```

`--locked` is important: it fails rather than silently resolving a newer
environment. Use the same option in batch scripts and release automation.

## Optional environments

```sh
# TFDS training/data-loader support
uv sync --locked --extra tfds --group dev

# Enhanced-sampling support
uv sync --locked --extra pysages --group dev

# Explicit opt-in telemetry support
uv sync --locked --extra wandb --group dev

# CUDA 12 on Linux x86-64
uv sync --locked --extra cuda12 --group dev
```

The TFDS extra selects only TensorFlow distributions present in the lock:
`tensorflow` on Apple-silicon macOS, and `tensorflow-cpu` on Linux x86-64 or
Windows AMD64. On other platforms, the TFDS utility package can still be
installed, but the extra deliberately does not claim that a compatible
TensorFlow runtime is available. W&B is absent from the base environment and is
loaded only after `training.use_wandb: true` explicitly opts in. CUDA packages
are locked but selected only by the `cuda12` extra on supported Linux hosts.

## Release artifacts

```sh
uv sync --locked --no-default-groups --group release --no-install-project
uv build --no-sources --build-constraints build-constraints.txt --require-hashes
uv run --no-sync python tools/check_package_contents.py dist
uv run --no-sync twine check --strict dist/*
```

The build constraint pins `poetry-core==2.2.1` to the SHA-256 hashes published
by PyPI. `--require-hashes` makes a missing hash an error, while `--no-sources`
checks the same build-system declaration that third-party build frontends see.

Release CI must install and smoke-test the wheel produced by `uv build`, rather
than importing the source checkout. The release workflow checks the wheel on
both Python 3.12 and 3.13; `.python-version` records the exact interpreter used
for the authoritative source environment; it is not shipped inside the Python
installation artifacts. Model checksums are maintained in their publication
manifest separately from the Python dependency lock. For paper reproduction,
also archive the datasets, split identifiers/checksums, and effective settings.
The package-content check compares runtime files and
licenses in both archives with the checkout. Twine checks package metadata;
installed-wheel inference exercises the packaged model. These checks do not
establish dataset licensing or scientific reproduction by themselves.

## Notebook environment

Notebook execution tools and visualization-only packages are kept out of the
runtime dependencies. Create their locked environment separately:

```sh
uv sync --python 3.12 --locked --no-default-groups --group notebooks
```

Run the four repository-native notebooks from a clean checkout so their
relative `data/` and `../tests/test_data/` paths resolve correctly. They create
trajectory or HDF5 outputs, so use a fresh checkout or remove outputs between
runs:

```sh
uv run --no-sync jupyter execute --kernel_name=python3 --timeout=-1 \
  examples/evaluate_so3lr_on_dataset.ipynb \
  examples/npt_jaxmd.ipynb \
  examples/nvt_jaxmd.ipynb \
  examples/production_run.ipynb
```

`examples/so3lr_colab_example.ipynb` is not part of that batch. Its first cell
modifies a hosted Colab runtime, installs packages outside the uv environment,
and clones into `/content`; execute it in a fresh Colab session instead. The
locked notebook group includes `py3Dmol` for its visualization cell, but does
not make the Colab setup cell a cluster-reproducible installation procedure.

## Updating dependencies

Do not edit `uv.lock` manually. Make dependency changes in `pyproject.toml`, then
run either a targeted update or a full deliberate refresh:

```sh
uv lock --upgrade-package PACKAGE
uv lock --upgrade
```

The resulting lockfile diff belongs in the same pull request as compatibility
tests and a short explanation of the update. Direct Git dependencies must use
immutable 40-character commit revisions in `pyproject.toml`; branch names are
not accepted for release environments. GLP currently declares its own `comms`
dependency from `main`, so `[tool.uv].override-dependencies` pins that transitive
dependency for locked environments without emitting two incompatible URL
identities in wheel metadata. Plain pip resolution is tested in CI, but the
fully immutable installation remains `uv sync --locked` until GLP pins `comms`
upstream.
