"""Verify the SHA-256 manifest for the model artifacts bundled with SO3LR."""

from __future__ import annotations

import argparse
import hashlib
import hmac
from pathlib import Path
from typing import Iterable


MODELS_DIR = Path(__file__).resolve().parent / "models"
MANIFEST_NAME = "SHA256SUMS"


class VerificationError(RuntimeError):
    """Raised when the model-artifact manifest is invalid or does not match."""


def _artifact_paths(models_dir: Path) -> Iterable[Path]:
    for path in sorted(models_dir.rglob("*")):
        if path.is_file() and path.name != MANIFEST_NAME:
            yield path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(models_dir: Path = MODELS_DIR) -> int:
    """Regenerate ``SHA256SUMS`` for maintainers after an intentional update."""
    models_dir = Path(models_dir).resolve()
    artifacts = list(_artifact_paths(models_dir))
    if not artifacts:
        raise VerificationError(f"No model artifacts found in {models_dir}")
    lines = [
        f"{_sha256(path)}  {path.relative_to(models_dir).as_posix()}"
        for path in artifacts
    ]
    (models_dir / MANIFEST_NAME).write_text("\n".join(lines) + "\n")
    return len(artifacts)


def _read_manifest(manifest_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(manifest_path.read_text().splitlines(), 1):
        if not raw_line.strip():
            continue
        digest, separator, relative_name = raw_line.partition("  ")
        if not separator or len(digest) != 64:
            raise VerificationError(
                f"Malformed {manifest_path.name} line {line_number}: {raw_line!r}"
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise VerificationError(
                f"Non-hex digest on {manifest_path.name} line {line_number}"
            ) from exc

        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise VerificationError(
                f"Unsafe path on {manifest_path.name} line {line_number}: "
                f"{relative_name!r}"
            )
        normalized = relative_path.as_posix()
        if normalized in entries:
            raise VerificationError(f"Duplicate manifest entry: {normalized}")
        entries[normalized] = digest.lower()
    return entries


def verify_manifest(models_dir: Path = MODELS_DIR) -> int:
    """Verify content hashes and complete manifest coverage; return file count."""
    models_dir = Path(models_dir).resolve()
    manifest_path = models_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise VerificationError(f"Missing model manifest: {manifest_path}")

    entries = _read_manifest(manifest_path)
    if not entries:
        raise VerificationError(f"Model manifest is empty: {manifest_path}")
    actual_names = {
        path.relative_to(models_dir).as_posix()
        for path in _artifact_paths(models_dir)
    }
    listed_names = set(entries)
    problems = []

    for missing in sorted(listed_names - actual_names):
        problems.append(f"missing file: {missing}")
    for unlisted in sorted(actual_names - listed_names):
        problems.append(f"unlisted file: {unlisted}")
    for relative_name in sorted(actual_names & listed_names):
        actual_digest = _sha256(models_dir / relative_name)
        if not hmac.compare_digest(actual_digest, entries[relative_name]):
            problems.append(f"checksum mismatch: {relative_name}")

    if problems:
        raise VerificationError("Model-artifact verification failed:\n" + "\n".join(problems))
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the model artifacts bundled with SO3LR."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="regenerate the manifest after an intentional artifact update",
    )
    args = parser.parse_args(argv)

    try:
        if args.write:
            count = write_manifest()
            print(f"Wrote {MANIFEST_NAME} for {count} model artifact(s).")
        else:
            count = verify_manifest()
            print(f"Verified {count} model artifact(s).")
    except (OSError, VerificationError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
