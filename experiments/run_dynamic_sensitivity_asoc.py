from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import run_dynamic_asoc_suite as dyn
except ModuleNotFoundError:
    from experiments import run_dynamic_asoc_suite as dyn

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Multi-problem sensitivity audit. DF1 alone is not enough evidence; these
# cases are intentionally spread across the DF suite to expose different
# response surfaces while keeping the auxiliary study reproducible.
SENSITIVITY_PROTOCOL = "moderate_t10_n10"
SENSITIVITY_PROBLEMS = ("df1", "df4", "df7", "df9")
SENSITIVITY_SEEDS = tuple(range(10))

# Default values used in the reported FITO configuration.
DEFAULT_LAMBDA_1 = 0.10
DEFAULT_STAGNATION_LIMIT = 8

LAMBDA_1_VALUES = (0.05, 0.08, 0.10, 0.12, 0.15)
STAGNATION_LIMIT_VALUES = (4, 6, 8, 10, 12)


def _set_runtime_parameters(lambda_1: float, stagnation_limit: int) -> None:
    """Patch FITO runtime globals inside the imported dynamic suite module."""
    dyn.LEADER_PULL = float(lambda_1)
    dyn.STAGNATION_LIMIT = int(stagnation_limit)


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    parameter = str(task["parameter"])
    value = task["value"]
    problem = str(task["problem"])
    seed = int(task["seed"])

    lambda_1 = DEFAULT_LAMBDA_1
    stagnation_limit = DEFAULT_STAGNATION_LIMIT
    if parameter == "lambda_1":
        lambda_1 = float(value)
    elif parameter == "stagnation_limit":
        stagnation_limit = int(value)
    else:
        raise ValueError(f"Unknown sensitivity parameter: {parameter}")

    _set_runtime_parameters(lambda_1=lambda_1, stagnation_limit=stagnation_limit)
    cfg = dyn.PROTOCOLS[SENSITIVITY_PROTOCOL]
    start = time.perf_counter()
    result = dyn.run_dynamic_fito(
        SENSITIVITY_PROTOCOL,
        problem,
        seed,
        pop_size=int(cfg["pop_size"]),
        **dyn.FITO_DEFAULT_CONFIG,
    )
    runtime_sec = time.perf_counter() - start

    return {
        "protocol": SENSITIVITY_PROTOCOL,
        "problem": problem,
        "parameter": parameter,
        "value": value,
        "lambda_1": lambda_1,
        "stagnation_limit": stagnation_limit,
        "seed": seed,
        "migd": result["migd"],
        "tail_igd": result["tail_igd"],
        "n_evals": result["n_evals"],
        "pitstop_count": result.get("pitstop_count", 0),
        "redeployment_count": result.get("redeployment_count", 0),
        "environment_change_count": result.get("environment_change_count", 0),
        "change_response_count": result.get("change_response_count", 0),
        "replaced_count": result.get("replaced_count", 0),
        "runtime_sec": runtime_sec,
    }


def build_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for problem in SENSITIVITY_PROBLEMS:
        for value in LAMBDA_1_VALUES:
            for seed in SENSITIVITY_SEEDS:
                tasks.append({"parameter": "lambda_1", "value": float(value), "problem": problem, "seed": int(seed)})
        for value in STAGNATION_LIMIT_VALUES:
            for seed in SENSITIVITY_SEEDS:
                tasks.append({"parameter": "stagnation_limit", "value": int(value), "problem": problem, "seed": int(seed)})
    return tasks


def mean_ci95(values: pd.Series) -> tuple[float, float]:
    arr = values.dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean
    half_width = 1.96 * float(np.std(arr, ddof=1)) / float(np.sqrt(len(arr)))
    return mean - half_width, mean + half_width


def summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby(["protocol", "problem", "parameter", "value"], dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        protocol, problem, parameter, value = keys
        ci_low, ci_high = mean_ci95(group["migd"])
        rows.append(
            {
                "protocol": protocol,
                "problem": problem,
                "parameter": parameter,
                "value": value,
                "mean_migd": float(group["migd"].mean()),
                "std_migd": float(group["migd"].std(ddof=1)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_tail_igd": float(group["tail_igd"].mean()),
                "mean_pitstop_count": float(group["pitstop_count"].mean()),
                "mean_redeployment_count": float(group["redeployment_count"].mean()),
                "mean_runtime_sec": float(group["runtime_sec"].mean()),
                "n_runs": int(len(group)),
            }
        )

    summary = pd.DataFrame(rows)
    overall_rows: list[dict[str, Any]] = []
    for (protocol, parameter, value), group in raw_df.groupby(["protocol", "parameter", "value"], dropna=False):
        ci_low, ci_high = mean_ci95(group["migd"])
        overall_rows.append(
            {
                "protocol": protocol,
                "problem": "OVERALL",
                "parameter": parameter,
                "value": value,
                "mean_migd": float(group["migd"].mean()),
                "std_migd": float(group["migd"].std(ddof=1)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "mean_tail_igd": float(group["tail_igd"].mean()),
                "mean_pitstop_count": float(group["pitstop_count"].mean()),
                "mean_redeployment_count": float(group["redeployment_count"].mean()),
                "mean_runtime_sec": float(group["runtime_sec"].mean()),
                "n_runs": int(len(group)),
            }
        )
    return pd.concat([summary, pd.DataFrame(overall_rows)], ignore_index=True)


def sensitivity_table_latex(summary: pd.DataFrame) -> str:
    problems = list(SENSITIVITY_PROBLEMS) + ["OVERALL"]
    pretty_param = {"lambda_1": r"$\lambda_1$", "stagnation_limit": r"$s$"}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Multi-problem FITO parameter sensitivity on the moderate dynamic protocol. Entries report mean MIGD over 10 independent seeds; lower is better.}",
        r"\label{tab:dynamic-sensitivity}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * len(problems) + r"}",
        r"\toprule",
        "Parameter & Value & " + " & ".join(p.upper() if p != "OVERALL" else "Overall" for p in problems) + r" \\",
        r"\midrule",
    ]
    for parameter in ("lambda_1", "stagnation_limit"):
        param_rows = summary[summary["parameter"] == parameter]
        for value in sorted(param_rows["value"].unique(), key=lambda x: float(x)):
            cells = []
            for problem in problems:
                row = param_rows[(param_rows["problem"] == problem) & (param_rows["value"] == value)]
                if row.empty:
                    cells.append("--")
                else:
                    mean = float(row.iloc[0]["mean_migd"])
                    std = float(row.iloc[0]["std_migd"])
                    cells.append(f"{mean:.4f} $\\pm$ {std:.4f}")
            display_value = f"{float(value):.2f}" if parameter == "lambda_1" else f"{int(float(value))}"
            lines.append(f"{pretty_param[parameter]} & {display_value} & " + " & ".join(cells) + r" \\")
        if parameter == "lambda_1":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_manifest(summary: pd.DataFrame) -> None:
    best_by_parameter = {}
    overall = summary[summary["problem"] == "OVERALL"]
    for parameter in ("lambda_1", "stagnation_limit"):
        rows = overall[overall["parameter"] == parameter].sort_values("mean_migd")
        best_by_parameter[parameter] = rows.iloc[0].to_dict() if not rows.empty else None
    manifest = {
        "protocol": SENSITIVITY_PROTOCOL,
        "problems": list(SENSITIVITY_PROBLEMS),
        "seeds": list(SENSITIVITY_SEEDS),
        "lambda_1_values": list(LAMBDA_1_VALUES),
        "stagnation_limit_values": list(STAGNATION_LIMIT_VALUES),
        "default_lambda_1": DEFAULT_LAMBDA_1,
        "default_stagnation_limit": DEFAULT_STAGNATION_LIMIT,
        "best_by_parameter": best_by_parameter,
        "outputs": {
            "raw": "experiments/results/asoc_dynamic_sensitivity_raw.csv",
            "summary": "experiments/results/asoc_dynamic_sensitivity_summary.csv",
            "latex_table": "experiments/results/asoc_dynamic_sensitivity_table.tex",
            "figure": "experiments/results/sensitivity_analysis.pdf",
        },
    }
    (RESULTS_DIR / "asoc_dynamic_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    tasks = build_tasks()
    rows: list[dict[str, Any]] = []
    max_workers = min(4, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_one, task): task for task in tasks}
        for future in as_completed(future_map):
            rows.append(future.result())

    raw_df = pd.DataFrame(rows).sort_values(["parameter", "value", "problem", "seed"])
    summary = summarize(raw_df).sort_values(["parameter", "value", "problem"])

    raw_df.to_csv(RESULTS_DIR / "asoc_dynamic_sensitivity_raw.csv", index=False)
    summary.to_csv(RESULTS_DIR / "asoc_dynamic_sensitivity_summary.csv", index=False)
    (RESULTS_DIR / "asoc_dynamic_sensitivity_table.tex").write_text(sensitivity_table_latex(summary), encoding="utf-8")
    write_manifest(summary)

    try:
        import plot_sensitivity
        plot_sensitivity.main()
    except Exception as exc:  # pragma: no cover - plotting failure should not discard numeric results
        print(f"Sensitivity numeric results were written, but plotting failed: {exc}")

    print("Wrote multi-problem sensitivity outputs under experiments/results/.")


if __name__ == "__main__":
    main()
