"""Focused checks for missing or changed package resources."""

import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "check_package_contents",
    Path(__file__).resolve().parents[1] / "tools" / "check_package_contents.py",
)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


def _artifacts(tmp_path, *, omit=None, changed=None):
    files = {
        "pyproject.toml": b'''[project]
name = "so3lr"
license-files = ["LICENSE"]
[tool.poetry]
include = []
''',
        "README.md": b"example",
        "LICENSE": b"license text",
        "so3lr/__init__.py": b"",
        "so3lr/models/params.pkl": b"model bytes, not executed",
    }
    for name, contents in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "so3lr.whl", "w") as wheel:
        wheel.writestr("so3lr-0.2.0.dist-info/METADATA", b"Name: so3lr\n")
        for name, contents in files.items():
            target = "so3lr-0.2.0.dist-info/licenses/LICENSE" if name == "LICENSE" else name
            if name.startswith("so3lr/") or name == "LICENSE":
                if name != omit:
                    wheel.writestr(target, b"modified" if name == changed else contents)
    with tarfile.open(dist / "so3lr.tar.gz", "w:gz") as sdist:
        for name, contents in files.items():
            info = tarfile.TarInfo("so3lr-0.2.0/" + name)
            info.size = len(contents)
            sdist.addfile(info, io.BytesIO(contents))
    return dist


def test_complete_package_contents(tmp_path):
    checker.check_package_contents(tmp_path, _artifacts(tmp_path))


@pytest.mark.parametrize("name", ["so3lr/models/params.pkl", "LICENSE"])
def test_missing_package_resource_is_rejected(tmp_path, name):
    dist = _artifacts(tmp_path, omit=name)
    with pytest.raises(ValueError, match="inventory|Missing archive file"):
        checker.check_package_contents(tmp_path, dist)


def test_changed_model_is_rejected(tmp_path):
    dist = _artifacts(tmp_path, changed="so3lr/models/params.pkl")
    with pytest.raises(ValueError, match="differ from source"):
        checker.check_package_contents(tmp_path, dist)
