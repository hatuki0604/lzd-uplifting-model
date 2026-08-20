"""Offline evidence that DS artifacts handed to the DE repo are byte-identical.

This check proves the repository handoff, not the contents of a running Docker
image. Runtime identity still comes from the API-reported model version unless
the serving API later exposes an image/artifact digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ARTIFACT_FILES = (
    "model_booster.txt",
    "feature_contract.json",
    "metadata.json",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_DE_DIR = (
    PROJECT_ROOT.parent / "nhung-lala" / "LZD" / "models" / "uplift_voucher"
)


def file_hashes(path: Path) -> dict[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()  # noqa: S324 - compatibility evidence, not security use
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            md5.update(chunk)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def compare_artifacts(
    ds_dir: Path = DEFAULT_DS_DIR,
    de_dir: Path = DEFAULT_DE_DIR,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ARTIFACT_FILES:
        ds_path = ds_dir / name
        de_path = de_dir / name
        row: dict[str, Any] = {
            "file": name,
            "ds_path": str(ds_path),
            "de_path": str(de_path),
            "ds_exists": ds_path.is_file(),
            "de_exists": de_path.is_file(),
            "byte_identical": False,
        }
        if row["ds_exists"]:
            row["ds_hashes"] = file_hashes(ds_path)
        if row["de_exists"]:
            row["de_hashes"] = file_hashes(de_path)
        if row["ds_exists"] and row["de_exists"]:
            row["byte_identical"] = (
                row["ds_hashes"]["sha256"] == row["de_hashes"]["sha256"]
            )
        rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds-dir", type=Path, default=DEFAULT_DS_DIR)
    parser.add_argument("--de-dir", type=Path, default=DEFAULT_DE_DIR)
    args = parser.parse_args(argv)
    rows = compare_artifacts(args.ds_dir, args.de_dir)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0 if rows and all(row["byte_identical"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

