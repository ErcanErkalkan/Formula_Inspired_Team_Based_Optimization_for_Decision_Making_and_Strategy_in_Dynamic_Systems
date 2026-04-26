from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from run_dynamic_asoc_suite import (
        ABLATION_VARIANTS,
        MAIN_ALGORITHMS,
        PROTOCOLS,
        RESULTS_DIR,
        average_ranks,
        eval_table_latex,
        pairwise_tests,
        problem_table_latex,
        rank_table_latex,
        summarize_eval_budget,
        summarize_metric,
        write_summary,
    )
except ModuleNotFoundError:
    from experiments.run_dynamic_asoc_suite import (
        ABLATION_VARIANTS,
        MAIN_ALGORITHMS,
        PROTOCOLS,
        RESULTS_DIR,
        average_ranks,
        eval_table_latex,
        pairwise_tests,
        problem_table_latex,
        rank_table_latex,
        summarize_eval_budget,
        summarize_metric,
        write_summary,
    )


def normalize_ablation_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    if "FITO+PS" in set(renamed["algorithm"]):
        renamed["algorithm"] = renamed["algorithm"].replace({"FITO+PS": "FITO", "FITO": "FITO-noPS"})
    return renamed


def normalize_probe_names(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    if "FITO+PS" in set(renamed["algorithm"]):
        renamed["algorithm"] = renamed["algorithm"].replace({"FITO+PS": "FITO", "FITO": "FITO-noPS"})
    return renamed


def main() -> None:
    raw_path = RESULTS_DIR / "asoc_dynamic_raw_metrics.csv"
    probe_path = RESULTS_DIR / "asoc_dynamic_pitstop_budget_probe.csv"

    raw_df = pd.read_csv(raw_path)
    probe_df = normalize_probe_names(pd.read_csv(probe_path))

    main_rows = raw_df[raw_df["family"] == "main"].copy()
    ablation_rows = normalize_ablation_names(raw_df[raw_df["family"] == "ablation"].copy())
    budget_rows = raw_df[raw_df["family"] == "budget"].copy()

    main_baselines = main_rows[main_rows["algorithm"] != "FITO"].copy()
    promoted_main_fito = ablation_rows[ablation_rows["algorithm"] == "FITO"].copy()
    promoted_main_fito["family"] = "main"
    main_final = pd.concat((main_baselines, promoted_main_fito), ignore_index=True).sort_values(["protocol", "problem", "algorithm", "seed"])

    budget_baselines = budget_rows[budget_rows["algorithm"] != "FITO"].copy()
    promoted_budget_fito = probe_df[probe_df["algorithm"] == "FITO"].copy()
    promoted_budget_fito["family"] = "budget"
    budget_final = pd.concat((budget_baselines, promoted_budget_fito), ignore_index=True).sort_values(["protocol", "problem", "algorithm", "seed"])

    ablation_final = ablation_rows.copy().sort_values(["protocol", "problem", "algorithm", "seed"])
    raw_final = pd.concat((main_final, ablation_final, budget_final), ignore_index=True).sort_values(["family", "protocol", "problem", "algorithm", "seed"])

    main_summary = summarize_metric(main_final, "migd")
    ablation_summary = summarize_metric(ablation_final, "migd")
    budget_summary = summarize_metric(budget_final, "migd")

    main_ranks = average_ranks(main_final, "migd")
    ablation_ranks = average_ranks(ablation_final, "migd")
    budget_ranks = average_ranks(budget_final, "migd")

    main_stats = pairwise_tests(main_final, family_name="dynamic_main")
    budget_stats = pairwise_tests(budget_final, family_name="dynamic_budget")
    eval_budget = summarize_eval_budget(raw_final)

    raw_final.to_csv(raw_path, index=False)
    main_summary.to_csv(RESULTS_DIR / "asoc_dynamic_main_summary.csv", index=False)
    ablation_summary.to_csv(RESULTS_DIR / "asoc_dynamic_ablation_summary.csv", index=False)
    budget_summary.to_csv(RESULTS_DIR / "asoc_dynamic_budget_summary.csv", index=False)
    main_ranks.to_csv(RESULTS_DIR / "asoc_dynamic_main_ranks.csv", index=False)
    ablation_ranks.to_csv(RESULTS_DIR / "asoc_dynamic_ablation_ranks.csv", index=False)
    budget_ranks.to_csv(RESULTS_DIR / "asoc_dynamic_budget_ranks.csv", index=False)
    main_stats.to_csv(RESULTS_DIR / "asoc_dynamic_main_stats.csv", index=False)
    budget_stats.to_csv(RESULTS_DIR / "asoc_dynamic_budget_stats.csv", index=False)
    eval_budget.to_csv(RESULTS_DIR / "asoc_dynamic_eval_budget.csv", index=False)

    for protocol_name in PROTOCOLS:
        table = problem_table_latex(
            main_summary,
            protocol_name,
            MAIN_ALGORITHMS,
            caption=f"MIGD results on the {protocol_name.replace('_', ' ')} dynamic protocol over 20 independent runs.",
            label=f"tab:dynamic-{protocol_name}",
        )
        (RESULTS_DIR / f"asoc_dynamic_{protocol_name}_table.tex").write_text(table, encoding="utf-8")

    (RESULTS_DIR / "asoc_dynamic_protocol_ranks.tex").write_text(
        rank_table_latex(
            main_ranks,
            MAIN_ALGORITHMS,
            caption="Average MIGD ranks across dynamic protocols. Lower is better.",
            label="tab:dynamic-protocol-ranks",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "asoc_dynamic_ablation_ranks.tex").write_text(
        rank_table_latex(
            ablation_ranks,
            tuple(ABLATION_VARIANTS.keys()),
            caption="Average MIGD ranks for FITO component ablations across dynamic protocols. Lower is better.",
            label="tab:dynamic-ablation-ranks",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "asoc_dynamic_budget_ranks.tex").write_text(
        rank_table_latex(
            budget_ranks,
            MAIN_ALGORITHMS,
            caption="Average MIGD ranks in the fixed-budget dynamic auxiliary study. Lower is better.",
            label="tab:dynamic-budget-ranks",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "asoc_dynamic_eval_budget_table.tex").write_text(
        eval_table_latex(
            eval_budget,
            family_name="main",
            caption="Auxiliary generation-matched dynamic evaluation audit. Nominal optimizer evaluations, separate response-refresh evaluations, total objective calls, and population-size summaries are reported as means over final runs.",
            label="tab:dynamic-eval-budget",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "asoc_dynamic_fixed_budget_table.tex").write_text(
        eval_table_latex(
            eval_budget,
            family_name="budget",
            caption="Primary fixed-budget dynamic audit. The calibration equalizes the nominal optimizer budget; response-refresh evaluations for predictive baselines are reported separately, and total objective calls are shown explicitly.",
            label="tab:dynamic-fixed-budget",
        ),
        encoding="utf-8",
    )

    calibration_json = RESULTS_DIR / "asoc_dynamic_budget_calibration.json"
    fixed_budget_pop_sizes = json.loads(calibration_json.read_text(encoding="utf-8")) if calibration_json.exists() else {}
    write_summary(
        main_ranks,
        budget_ranks,
        main_stats,
        budget_stats,
        eval_budget,
        fixed_budget_pop_sizes,
        RESULTS_DIR / "asoc_dynamic_summary.txt",
    )


if __name__ == "__main__":
    main()
