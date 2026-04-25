from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from run_dynamic_asoc_suite import (
        ABLATION_VARIANTS,
        PROBLEMS,
        PROTOCOLS,
        RESULTS_DIR,
        SEEDS,
        average_ranks,
        calibrate_budget_pop_sizes,
        run_dynamic_fito,
    )
except ModuleNotFoundError:
    from experiments.run_dynamic_asoc_suite import (
        ABLATION_VARIANTS,
        PROBLEMS,
        PROTOCOLS,
        RESULTS_DIR,
        SEEDS,
        average_ranks,
        calibrate_budget_pop_sizes,
        run_dynamic_fito,
    )


def run_task(task: dict[str, object]) -> dict[str, object]:
    protocol_name = str(task["protocol"])
    problem_name = str(task["problem"])
    algorithm_name = str(task["algorithm"])
    seed = int(task["seed"])
    pop_size = int(task["pop_size"])
    cfg = {k: v for k, v in ABLATION_VARIANTS[algorithm_name].items() if k != "description"}
    result = run_dynamic_fito(protocol_name, problem_name, seed, pop_size=pop_size, **cfg)
    result.update(
        {
            "protocol": protocol_name,
            "problem": problem_name,
            "algorithm": algorithm_name,
            "seed": seed,
            "pop_size": pop_size,
        }
    )
    return result


def main() -> None:
    calibration_path = RESULTS_DIR / "asoc_dynamic_budget_calibration.json"
    if calibration_path.exists():
        fixed_budget_pop_sizes = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        fixed_budget_pop_sizes = calibrate_budget_pop_sizes()

    tasks = []
    for protocol_name in PROTOCOLS:
        pop_size = int(fixed_budget_pop_sizes[protocol_name]["FITO"])
        for problem_name in PROBLEMS:
            for algorithm_name in ("FITO", "FITO-noPS"):
                for seed in SEEDS:
                    tasks.append(
                        {
                            "protocol": protocol_name,
                            "problem": problem_name,
                            "algorithm": algorithm_name,
                            "seed": seed,
                            "pop_size": pop_size,
                        }
                    )

    rows = []
    max_workers = min(4, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            rows.append(future.result())

    raw_df = pd.DataFrame(rows).sort_values(["protocol", "problem", "algorithm", "seed"])
    raw_df.to_csv(RESULTS_DIR / "asoc_dynamic_pitstop_budget_probe.csv", index=False)

    rank_df = average_ranks(raw_df, "migd")
    rank_df.to_csv(RESULTS_DIR / "asoc_dynamic_pitstop_budget_probe_ranks.csv", index=False)

    summary = raw_df.groupby(["protocol", "problem", "algorithm"])["migd"].agg(["mean", "std"]).reset_index()
    summary.to_csv(RESULTS_DIR / "asoc_dynamic_pitstop_budget_probe_summary.csv", index=False)

    lines = [
        "Dynamic Fixed-Budget Pit-Stop Probe",
        "=================================",
        "Calibrated pop sizes reused from asoc_dynamic_budget_calibration.json.",
        "",
        "Overall average ranks:",
    ]
    for algorithm, value in rank_df.groupby("algorithm")["average_rank"].mean().sort_values().items():
        lines.append(f"  - {algorithm}: {value:.3f}")
    (RESULTS_DIR / "asoc_dynamic_pitstop_budget_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
