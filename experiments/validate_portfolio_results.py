"""Validate ASOC portfolio result-file consistency without importing yfinance.

This script checks only generated CSV files, so it can run on machines that do
not need to rerun the portfolio optimization itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EXPECTED_ALGORITHMS = (
    "FITO",
    "DNSGA-II-A",
    "DNSGA-II-B",
    "KGB-DMOEA",
    "MDDM-DMOEA",
    "NSGA-II",
    "PPS-DMOEA",
)
EXPECTED_FAMILIES = ("main", "budget")
EXPECTED_UNIVERSES = ("tech14", "market20")
EXPECTED_SEEDS = tuple(range(20))


def validate_portfolio_algorithm_coverage(raw_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"family", "universe", "algorithm", "seed"}
    missing_cols = required_cols.difference(raw_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in asoc_portfolio_raw_metrics.csv: {sorted(missing_cols)}")

    unexpected_algorithms = sorted(set(raw_df["algorithm"].dropna()) - set(EXPECTED_ALGORITHMS))
    if unexpected_algorithms:
        raise ValueError(f"Unexpected portfolio algorithms: {unexpected_algorithms}")

    rows = []
    missing_entries = []
    for family in EXPECTED_FAMILIES:
        for universe in EXPECTED_UNIVERSES:
            for algorithm in EXPECTED_ALGORITHMS:
                subset = raw_df[(raw_df["family"] == family) & (raw_df["universe"] == universe) & (raw_df["algorithm"] == algorithm)]
                observed_seeds = sorted(int(seed) for seed in subset["seed"].dropna().unique())
                missing_seeds = [seed for seed in EXPECTED_SEEDS if seed not in observed_seeds]
                coverage_ok = not missing_seeds
                if not coverage_ok:
                    missing_entries.append({
                        "family": family,
                        "universe": universe,
                        "algorithm": algorithm,
                        "missing_seeds": missing_seeds,
                    })
                rows.append({
                    "family": family,
                    "universe": universe,
                    "algorithm": algorithm,
                    "expected_seed_count": len(EXPECTED_SEEDS),
                    "observed_seed_count": len(observed_seeds),
                    "row_count": int(len(subset)),
                    "missing_seeds": ";".join(str(seed) for seed in missing_seeds),
                    "coverage_ok": bool(coverage_ok),
                })

    coverage = pd.DataFrame(rows)
    manifest = {
        "active_result_family": "asoc_portfolio",
        "expected_algorithms": list(EXPECTED_ALGORITHMS),
        "expected_families": list(EXPECTED_FAMILIES),
        "expected_universes": list(EXPECTED_UNIVERSES),
        "expected_seeds": list(EXPECTED_SEEDS),
        "observed_algorithms": sorted(raw_df["algorithm"].dropna().unique().tolist()),
        "unexpected_algorithms": unexpected_algorithms,
        "missing_entries": missing_entries,
        "coverage_ok": bool(coverage["coverage_ok"].all() and not unexpected_algorithms),
        "legacy_note": "portfolio_case_* files are archived five-algorithm legacy results and are not valid for ASOC tables.",
    }
    (RESULTS_DIR / "asoc_portfolio_algorithm_coverage.csv").write_text(coverage.to_csv(index=False), encoding="utf-8")
    (RESULTS_DIR / "asoc_portfolio_algorithm_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["coverage_ok"]:
        raise ValueError(f"Incomplete ASOC portfolio coverage: {json.dumps(missing_entries, indent=2)}")
    return coverage


def main() -> None:
    active_path = RESULTS_DIR / "asoc_portfolio_raw_metrics.csv"
    legacy_path = RESULTS_DIR / "portfolio_case_raw_metrics.csv"

    if not active_path.exists():
        msg = "Missing active ASOC result file: experiments/results/asoc_portfolio_raw_metrics.csv"
        if legacy_path.exists():
            msg += (
                "\nA legacy five-algorithm portfolio_case_raw_metrics.csv exists, "
                "but it is intentionally rejected for ASOC manuscript tables."
            )
        raise FileNotFoundError(msg)

    raw_df = pd.read_csv(active_path)
    coverage = validate_portfolio_algorithm_coverage(raw_df)
    print("ASOC portfolio result coverage is complete.")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
