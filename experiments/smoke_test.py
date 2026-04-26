#!/usr/bin/env python3
"""Lightweight smoke test for the FITO reproducibility snapshot.

The smoke test is intentionally quick. It does not rerun the full experiments.
It verifies imports, key scripts, existing result artifacts, primary fixed-budget
rank files, portfolio coverage, and checksum tooling.
"""
from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

REQUIRED_DISTRIBUTIONS = [
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "pymoo",
]

REQUIRED_FILES = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "environment.yml",
    ROOT / "reproduce_all.ps1",
    ROOT / "reproduce_all.sh",
    ROOT / "Makefile",
    ROOT / "scripts" / "generate_checksums.py",
    ROOT / "scripts" / "verify_checksums.py",
    RESULTS / "asoc_dynamic_summary.txt",
    RESULTS / "asoc_dynamic_budget_ranks.csv",
    RESULTS / "asoc_dynamic_budget_stats.csv",
    RESULTS / "asoc_dynamic_budget_summary.csv",
    RESULTS / "asoc_dynamic_fixed_budget_table.tex",
    RESULTS / "asoc_dynamic_eval_budget.csv",
    RESULTS / "asoc_dynamic_eval_budget_table.tex",
    RESULTS / "asoc_portfolio_summary.txt",
    RESULTS / "asoc_portfolio_algorithm_coverage.csv",
    ROOT / "manuscript" / "asoc_fito.tex",
    ROOT / "manuscript" / "asoc_fito.pdf",
]


def check_imports() -> None:
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as exc:
            raise AssertionError(
                f"Missing dependency {name!r}. Install the pinned environment first with \"python -m pip install -r requirements.txt\"."
            ) from exc
        print(f"[ok] dependency {name} ({version})")


def check_required_files() -> None:
    missing = [p.relative_to(ROOT).as_posix() for p in REQUIRED_FILES if not p.exists()]
    if missing:
        raise AssertionError("Missing required files:\n" + "\n".join(missing))
    print(f"[ok] required files present ({len(REQUIRED_FILES)} files)")


def check_dynamic_budget_ranks() -> None:
    path = RESULTS / "asoc_dynamic_budget_ranks.csv"
    df = pd.read_csv(path)
    required_cols = {"protocol", "problem", "metric", "algorithm", "average_rank"}
    if not required_cols.issubset(df.columns):
        raise AssertionError(f"Unexpected columns in {path}: {df.columns.tolist()}")
    fito = df[df["algorithm"].eq("FITO")]
    if fito.empty:
        raise AssertionError("FITO rows are missing from fixed-budget rank table")
    overall = df.groupby("algorithm")["average_rank"].mean().sort_values()
    if overall.index[0] != "FITO":
        raise AssertionError(f"FITO is not first in fixed-budget average ranks: {overall.head().to_dict()}")
    print(f"[ok] primary fixed-budget rank table; FITO overall mean rank={overall.loc['FITO']:.3f}")


def check_budget_summary() -> None:
    path = RESULTS / "asoc_dynamic_budget_summary.csv"
    df = pd.read_csv(path)
    if "algorithm" not in df.columns:
        raise AssertionError(f"algorithm column missing from {path}")
    numeric_cols = [c for c in df.columns if "eval" in c.lower()]
    if not numeric_cols:
        raise AssertionError(f"No evaluation-budget columns found in {path}")
    print(f"[ok] realized budget summary contains evaluation columns: {numeric_cols}")


def check_portfolio_coverage() -> None:
    path = RESULTS / "asoc_portfolio_algorithm_coverage.csv"
    df = pd.read_csv(path)
    if "coverage_ok" not in df.columns:
        raise AssertionError("coverage_ok column missing from portfolio coverage file")
    if not df["coverage_ok"].astype(bool).all():
        bad = df.loc[~df["coverage_ok"].astype(bool)]
        raise AssertionError(f"Portfolio coverage failed:\n{bad}")
    print(f"[ok] portfolio coverage table passes ({len(df)} rows)")


def check_manifests() -> None:
    manifest_path = RESULTS / "asoc_portfolio_algorithm_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        print(f"[ok] portfolio manifest loaded ({len(manifest) if hasattr(manifest, '__len__') else 'object'} entries)")
    else:
        print("[warn] portfolio manifest not found; continuing because coverage CSV is present")


def main() -> None:
    print("FITO reproducibility smoke test")
    print(f"Repository root: {ROOT}")
    check_imports()
    check_required_files()
    check_dynamic_budget_ranks()
    check_budget_summary()
    check_portfolio_coverage()
    check_manifests()
    print("[ok] smoke test completed successfully")


if __name__ == "__main__":
    main()
