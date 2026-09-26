#!/usr/bin/env python3
"""Check that release archives contain unchanged runtime files and licenses.

Twine validates package metadata; this check covers SO3LR's non-Python payloads
and source documents. Archives are read in place, never extracted.
"""

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
import tomllib
import zipfile


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _names(names):
    names = list(names)
    _check(len(names) == len(set(names)), "Duplicate archive entries")
    for name in names:
        path = PurePosixPath(name)
        _check(not path.is_absolute() and ".." not in path.parts,
               f"Unsafe archive path: {name}")
    return set(names)


def _same_bytes(source, archive_name, names, open_member):
    _check(archive_name in names, f"Missing archive file: {archive_name}")
    with source.open("rb") as expected, open_member(archive_name) as actual:
        _check(hashlib.file_digest(expected, "sha256").digest()
               == hashlib.file_digest(actual, "sha256").digest(),
               f"Archive contents differ from source: {archive_name}")


def check_package_contents(root, dist):
    root, dist = Path(root), Path(dist)
    config = tomllib.loads((root / "pyproject.toml").read_text())
    project = config["project"]
    wheels, sdists = list(dist.glob("*.whl")), list(dist.glob("*.tar.gz"))
    _check(len(wheels) == len(sdists) == 1, "Expected one wheel and one sdist")
    excluded = {"__pycache__", ".pytest_cache", ".DS_Store"}
    package = {
        path.relative_to(root).as_posix()
        for path in (root / "so3lr").rglob("*")
        if path.is_file() and not excluded.intersection(path.parts)
        and path.suffix not in {".pyc", ".pyo"}
    }
    _check(any(name.startswith("so3lr/models/") for name in package),
           "No bundled model files found")
    licenses = set()
    for pattern in project["license-files"]:
        matches = {p.relative_to(root).as_posix() for p in root.glob(pattern)
                   if p.is_file()}
        _check(matches, f"No source license matches {pattern}")
        licenses.update(matches)

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = _names(i.filename for i in wheel.infolist() if not i.is_dir())
        metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
        _check(len(metadata) == 1, "Expected one wheel metadata directory")
        info = str(PurePosixPath(metadata[0]).parent)
        _check(all(n.startswith(("so3lr/", info + "/")) for n in names),
               "Repository-only files found at the wheel root")
        _check({n for n in names if n.startswith("so3lr/")} == package,
               "Wheel package inventory differs from source")
        for name in sorted(package):
            _same_bytes(root / name, name, names, wheel.open)
        for name in sorted(licenses):
            _same_bytes(root / name, f"{info}/licenses/{name}", names, wheel.open)

    with tarfile.open(sdists[0], "r:gz") as sdist:
        members = sdist.getmembers()
        _check(all(m.isfile() or m.isdir() for m in members),
               "Source archive contains links or special files")
        names = _names(m.name for m in members if m.isfile())
        roots = {PurePosixPath(n).parts[0] for n in names}
        _check(len(roots) == 1, "Expected one source archive root")
        prefix = next(iter(roots))
        archived_package = {n[len(prefix) + 1:] for n in names
                            if n.startswith(prefix + "/so3lr/")}
        _check(archived_package == package,
               "Source archive package inventory differs from source")
        sources = package | licenses | {"pyproject.toml", "README.md"}
        for entry in config["tool"]["poetry"]["include"]:
            if "sdist" in entry["format"]:
                sources.update(p.relative_to(root).as_posix()
                               for p in root.glob(entry["path"]) if p.is_file())
        for name in sorted(sources):
            _same_bytes(root / name, f"{prefix}/{name}", names, sdist.extractfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()
    check_package_contents(Path(__file__).resolve().parents[1], args.dist)
    print("Verified wheel and sdist runtime files, source documents, and licenses")
