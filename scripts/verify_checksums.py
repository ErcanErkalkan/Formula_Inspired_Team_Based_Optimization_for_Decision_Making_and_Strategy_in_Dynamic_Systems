#!/usr/bin/env python3
"""Verify SHA-256 checksums for the reproducibility snapshot."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print("Missing SHA256SUMS.txt; run python scripts/generate_checksums.py first.", file=sys.stderr)
        return 2
    failures: list[str] = []
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        path = ROOT / rel
        if not path.exists():
            failures.append(f"MISSING  {rel}")
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append(f"FAILED   {rel}")
    if failures:
        print("Checksum verification failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"Checksum verification passed for {checked} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
