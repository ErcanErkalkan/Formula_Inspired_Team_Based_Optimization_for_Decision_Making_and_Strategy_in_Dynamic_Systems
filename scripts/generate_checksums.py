#!/usr/bin/env python3
"""Generate SHA-256 checksums for the reproducibility snapshot."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SHA256SUMS.txt"
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
EXCLUDE_FILES = {"SHA256SUMS.txt"}


def include(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    if any(part in EXCLUDE_DIRS for part in rel_parts):
        return False
    if path.name in EXCLUDE_FILES:
        return False
    if path.suffix in {".aux", ".log", ".out", ".toc", ".fls", ".fdb_latexmk", ".synctex.gz"}:
        return False
    return path.is_file()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    files = sorted(p for p in ROOT.rglob("*") if include(p))
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# SHA-256 manifest for the FITO reproducibility snapshot\n")
        f.write("# Generated with: python scripts/generate_checksums.py\n")
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            f.write(f"{sha256(path)}  {rel}\n")
    print(f"Wrote {OUT.relative_to(ROOT)} with {len(files)} entries.")


if __name__ == "__main__":
    main()
