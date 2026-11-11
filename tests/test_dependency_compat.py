"""Exercise dependency APIs that a successful lock resolution cannot validate."""

import importlib.util
import io
import os
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import pytest


def _require_optional(module):
    # Ordinary core-only test runs may skip extras. The dedicated CI lane must
    # fail if its environment omitted one, rather than pass without testing it.
    if importlib.util.find_spec(module) is None:
        message = f"optional dependency {module!r} is not installed"
        if os.environ.get("SO3LR_REQUIRE_OPTIONAL_DEPS") == "1":
            pytest.fail(message)
        pytest.skip(message)


def test_matplotlib_pillow_png_round_trip():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    figure = Figure(figsize=(2, 2), dpi=100)
    FigureCanvasAgg(figure)
    figure.subplots().plot([0, 1, 2], [0, 1, 0])
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.load()
        assert image.format == "PNG"
        assert image.size == (200, 200)
        assert np.ptp(np.asarray(image.convert("RGB"))) > 0


def test_flax_msgpack_parameter_round_trip(tmp_path):
    from flax import serialization

    params = {
        "dense": {
            "kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
            "bias": np.array([0.25, -0.5, 1.0], dtype=np.float64),
        },
        "step": np.array(17, dtype=np.int32),
    }
    path = tmp_path / "params.msgpack"
    path.write_bytes(serialization.to_bytes(params))
    restored = serialization.from_bytes(params, path.read_bytes())
    for name in ("kernel", "bias"):
        np.testing.assert_array_equal(restored["dense"][name], params["dense"][name])
        assert restored["dense"][name].dtype == params["dense"][name].dtype
    np.testing.assert_array_equal(restored["step"], params["step"])


def test_tensorboard_startup(tmp_path):
    _require_optional("tensorboard")
    # TensorBoard 2.20 resolved alongside new setuptools, but failed here after
    # setuptools removed pkg_resources. TFDS data loading alone did not catch it.
    imported = subprocess.run(
        [sys.executable, "-c", "import tensorboard.default; import tensorboard.program"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
    result = subprocess.run(
        [sys.executable, "-m", "tensorboard.main", "--version"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert version("tensorboard") in result.stdout.splitlines()


def test_notebook_html_export():
    _require_optional("nbconvert")
    _require_optional("nbformat")
    import nbformat
    from nbconvert import HTMLExporter

    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Dependency compatibility"),
        nbformat.v4.new_code_cell(
            "print('dependency smoke output')",
            execution_count=1,
            metadata={"trusted": True},
            outputs=[nbformat.v4.new_output(
                "stream", name="stdout", text="dependency smoke output\n",
            )],
        ),
    ])
    body, _ = HTMLExporter().from_notebook_node(notebook)
    assert "Dependency compatibility" in body
    assert "dependency smoke output" in body
    assert "jp-OutputArea-output" in body


def test_wandb_import_without_starting_a_run(tmp_path):
    _require_optional("wandb")
    environment = dict(os.environ, WANDB_MODE="disabled", WANDB_SILENT="true")
    result = subprocess.run(
        [sys.executable, "-c", "import wandb; assert wandb.run is None"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
