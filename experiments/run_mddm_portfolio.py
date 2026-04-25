"""Complete active ASOC portfolio results to match MAIN_ALGORITHMS.

This script supersedes the earlier two-algorithm append helper.  It treats
`asoc_portfolio_raw_metrics.csv` as the only active manuscript-level portfolio
raw result file and fills any missing (family, universe, algorithm, seed)
combinations required by `run_portfolio_asoc_suite.MAIN_ALGORITHMS`.

The older `portfolio_case_*` files are a five-algorithm legacy case study and
are not used as input here.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from run_portfolio_asoc_suite import (
    MAIN_ALGORITHMS,
    SEEDS,
    UNIVERSES,
    POP_SIZE,
    RESULTS_DIR,
    build_tasks,
    calibrate_budget_pop_sizes,
    run_task,
    validate_portfolio_algorithm_coverage,
)


def _existing_keys(df: pd.DataFrame) -> set[tuple[str, str, str, int]]:
    if df.empty:
        return set()
    required = {"family", "universe", "algorithm", "seed"}
    if not required.issubset(df.columns):
        return set()
    primary = df.drop_duplicates(subset=["family", "universe", "algorithm", "seed"])
    return {
        (str(row.family), str(row.universe), str(row.algorithm), int(row.seed))
        for row in primary.itertuples(index=False)
    }


def _load_existing(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    legacy_path = RESULTS_DIR / "portfolio_case_raw_metrics.csv"
    if legacy_path.exists():
        print(
            "Legacy portfolio_case_raw_metrics.csv found but ignored. "
            "It contains the old five-algorithm case study, not the active ASOC suite."
        )
    return pd.DataFrame()


def run_missing_portfolio_tasks() -> None:
    csv_path = RESULTS_DIR / "asoc_portfolio_raw_metrics.csv"
    existing_df = _load_existing(csv_path)
    existing = _existing_keys(existing_df)

    fixed_budget_pop_sizes = calibrate_budget_pop_sizes()
    all_tasks = build_tasks(fixed_budget_pop_sizes)
    missing_tasks = [
        task
        for task in all_tasks
        if (str(task["family"]), str(task["universe"]), str(task["algorithm"]), int(task["seed"])) not in existing
    ]

    if not missing_tasks:
        validate_portfolio_algorithm_coverage(existing_df, RESULTS_DIR / "asoc_portfolio_algorithm")
        print("Active ASOC portfolio raw metrics already cover all algorithms, universes, families, and seeds.")
        return

    print(f"Running {len(missing_tasks)} missing ASOC portfolio tasks.")
    print(f"Expected algorithms: {', '.join(MAIN_ALGORITHMS)}")
    print(f"Expected universes: {', '.join(UNIVERSES.keys())}")
    print(f"Expected seeds: {SEEDS[0]}..{SEEDS[-1]}")
    print(f"Default generation-matched pop size: {POP_SIZE}")

    new_rows: list[dict[str, object]] = []
    max_workers = min(4, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in missing_tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                rows = future.result()
            except Exception as exc:
                raise RuntimeError(f"Missing portfolio task failed: {task}") from exc
            new_rows.extend(rows)

    new_df = pd.DataFrame(new_rows)
    if existing_df.empty:
        combined = new_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["family", "universe", "algorithm", "seed", "decision_rule", "cost_rate"],
        keep="last",
    )
    validate_portfolio_algorithm_coverage(combined, RESULTS_DIR / "asoc_portfolio_algorithm")
    combined.to_csv(csv_path, index=False)
    print(f"Saved complete active ASOC portfolio raw metrics to {csv_path}")


if __name__ == "__main__":
    run_missing_portfolio_tasks()
