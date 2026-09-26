import ast
import json
from pathlib import Path
import tomllib

import pytest
import yaml

import so3lr
from so3lr.verify_model_artifacts import VerificationError, verify_manifest, write_manifest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_NATIVE_NOTEBOOKS = (
    "evaluate_so3lr_on_dataset.ipynb",
    "npt_jaxmd.ipynb",
    "nvt_jaxmd.ipynb",
    "production_run.ipynb",
)
PINNED_PUBLICATION_MODEL = "so3lr-m-2606"
PUBLICATION_MODEL_CONSTRUCTORS = frozenset({"So3lr", "So3lrPotential"})
UNRESOLVED_MODEL = "<unresolved>"


def _source_text(cell):
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _notebook_code_source(path):
    data = json.loads(path.read_text())
    return "\n".join(
        _source_text(cell)
        for cell in data["cells"]
        if cell.get("cell_type") == "code"
    )


def _binding_event_names(tree):
    """Yield names bound, deleted, or made scope-dependent in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            yield node.id
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.alias):
            yield node.asname or node.name.split(".", maxsplit=1)[0]
        elif isinstance(node, ast.ExceptHandler) and node.name:
            yield node.name
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            yield node.name
        elif isinstance(node, ast.MatchMapping) and node.rest:
            yield node.rest
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            yield from node.names


def _notebook_constructor_model_pins(path):
    """Resolve unambiguous static model pins on SO3LR constructors."""
    data = json.loads(path.read_text())
    binding_counts = {}
    constant_bindings = {}
    constructor_calls = []

    for cell_index, cell in enumerate(data["cells"]):
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse(_source_text(cell), filename=str(path))

        for name in _binding_event_names(tree):
            binding_counts[name] = binding_counts.get(name, 0) + 1

        for statement in tree.body:
            target = None
            value = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target = statement.targets[0]
                value = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(
                statement.target, ast.Name
            ):
                target = statement.target
                value = statement.value
            if (
                target is not None
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                position = (cell_index, statement.lineno, statement.col_offset)
                constant_bindings.setdefault(target.id, []).append(
                    (position, value.value)
                )

        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id not in PUBLICATION_MODEL_CONSTRUCTORS
            ):
                continue
            model_keywords = [
                keyword.value for keyword in node.keywords if keyword.arg == "model"
            ]
            position = (cell_index, node.lineno, node.col_offset)
            constructor_calls.append((position, node.func.id, model_keywords))

    resolved_pins = []
    for position, constructor, model_keywords in constructor_calls:
        resolved_model = UNRESOLVED_MODEL
        if len(model_keywords) == 1:
            value = model_keywords[0]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                resolved_model = value.value
            elif isinstance(value, ast.Name) and binding_counts.get(value.id) == 1:
                bindings = constant_bindings.get(value.id, [])
                if len(bindings) == 1 and bindings[0][0] < position:
                    resolved_model = bindings[0][1]
        resolved_pins.append((constructor, resolved_model))

    return resolved_pins


def test_model_artifact_manifest_is_complete_and_valid():
    assert verify_manifest() > 0


def test_model_artifact_verification_rejects_an_empty_release(tmp_path):
    (tmp_path / "SHA256SUMS").write_text("\n")
    with pytest.raises(VerificationError, match="manifest is empty"):
        verify_manifest(tmp_path)


def test_model_manifest_regeneration_keeps_existing_manifest_if_payloads_missing(tmp_path):
    manifest = tmp_path / "SHA256SUMS"
    original = "an existing manifest must not be erased\n"
    manifest.write_text(original)
    with pytest.raises(VerificationError, match="No model artifacts"):
        write_manifest(tmp_path)
    assert manifest.read_text() == original


def test_example_notebooks_are_clean_and_portable():
    notebooks = sorted((ROOT / "examples").glob("*.ipynb"))
    assert notebooks

    for notebook in notebooks:
        data = json.loads(notebook.read_text())
        source = "\n".join(_source_text(cell) for cell in data["cells"])
        assert "from mlff" not in source, notebook
        assert "import mlff" not in source, notebook
        for private_prefix in (
            "/Users/",
            "/home/",
            "/project/",
            "/mnt/",
            "/scratch/",
            "/gpfs/",
            "/lustre/",
        ):
            assert private_prefix not in source, notebook

        for cell in data["cells"]:
            if cell.get("cell_type") == "code":
                assert cell.get("execution_count") is None, notebook
                assert cell.get("outputs") == [], notebook


def test_repository_native_notebooks_pin_the_dated_medium_model():
    for name in REPOSITORY_NATIVE_NOTEBOOKS:
        notebook = ROOT / "examples" / name
        constructor_pins = _notebook_constructor_model_pins(notebook)
        assert len(constructor_pins) == 1, (notebook, constructor_pins)
        assert constructor_pins[0][1] == PINNED_PUBLICATION_MODEL, (
            notebook,
            constructor_pins,
        )


def test_evaluation_notebook_uses_the_v2_atomic_volume_key():
    notebook = ROOT / "examples" / "evaluate_so3lr_on_dataset.ipynb"
    source = _notebook_code_source(notebook)
    assert "output['a0_ratios']" in source
    assert "output['hirshfeld_ratios']" not in source
    assert "jax.block_until_ready(so3lr_calc(inputs))" in source
    assert "processed_structures == len(data) == 50" in source
    assert source.count("jraph_utils.dynamically_batch_with_lr(") == 3
    assert "jraph.dynamically_batch(" not in source
    assert (
        source.count("n_pairs=stats['max_num_of_pairs'] * batch_size + 1")
        == 3
    )
    assert source.count("for graph_batch, lr_batch in batched_graphs:") == 2
    assert source.count("graph_batch,\n        lr_batch\n    )") == 2
    assert "example_batch, example_lr_batch = next(batched_graphs)" in source
    assert "batch_info_fn(example_batch)" in source


def test_md_notebooks_use_explicit_thermodynamic_units_and_3d_boxes():
    nvt_source = _notebook_code_source(ROOT / "examples" / "nvt_jaxmd.ipynb")
    assert "T_nvt = T_nvt_kelvin * unit['temperature']" in nvt_source
    assert "kT=temp_i" in nvt_source
    assert nvt_source.count("Neighbor-list capacity overflowed") >= 2

    for name in ("npt_jaxmd.ipynb", "production_run.ipynb"):
        source = _notebook_code_source(ROOT / "examples" / name)
        assert "box.shape == (3,)" in source, name
        assert "pressure = " in source and " * unit['pressure']" in source, name
        assert source.count("Neighbor-list capacity overflowed") >= 2, name


def test_production_notebook_flushes_and_validates_every_frame():
    source = _notebook_code_source(ROOT / "examples" / "production_run.ipynb")
    assert "saved_frames += flush_hdf5_buffer" in source
    assert "assert saved_frames == npt_cycles" in source
    assert "trajectory_positions.shape == expected_vector_shape" in source
    assert "num_box_entries=3" in source
    assert "write_to_hdf5(store, momenta, positions, boxes)" in source
    assert "trajectory_positions.dtype == np.asarray(state.position).dtype" in source


def test_colab_notebook_fails_loudly_and_rejects_partial_periodicity():
    source = _notebook_code_source(
        ROOT / "examples" / "so3lr_colab_example.ipynb"
    )
    assert "uv==0.11.26" in source
    assert "input_pbc.all()" in source
    assert "raise subprocess.CalledProcessError(code, cmd)" in source
    assert "raise RuntimeError(" in source


def test_citation_metadata_matches_package_version():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["cff-version"] == "1.2.0"
    assert citation["version"] == so3lr.__version__
    assert citation["preferred-citation"]["doi"] == "10.1021/jacs.5c09558"
    assert "doi" not in citation  # no software/SO3LR-2 DOI has been issued yet


def test_third_party_license_files_and_notices_are_present():
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    for filename in (
        "LICENSE",
        "LICENSE-APACHE-2.0",
        "LICENSE-BSD-3-CLAUSE",
    ):
        assert (ROOT / filename).is_file(), filename
    assert "Google Research robust loss" in notice
    assert "JAX MD safe masking" in notice
    assert "torch-pme Ewald tuning" in notice
    assert "1f612664e8e627143b19e8622bac9d2d81e85022" in notice


def test_root_project_documents_are_source_archive_only():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    included = {
        entry["path"]: entry["format"]
        for entry in pyproject["tool"]["poetry"]["include"]
    }

    for filename in (
        "CITATION.cff",
        "CHANGELOG.md",
    ):
        assert included[filename] == "sdist"
