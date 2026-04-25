import json
from pathlib import Path
import pandas as pd
import numpy as np

import sys
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from run_portfolio_asoc_suite import (
    RESULTS_DIR, UNIVERSES, DECISION_RULES, PRIMARY_RULE, PRIMARY_COST,
    MAIN_ALGORITHMS, SEEDS,
    summarize_primary, summarize_sensitivity, evaluate_simple_benchmarks,
    pairwise_stats, primary_table_latex, deployment_table_latex, write_summary,
    activation_event_frame, summarize_activation_audit,
    validate_portfolio_algorithm_coverage,
)


def load_budget_calibration() -> dict[str, dict[str, object]]:
    """Load recorded fixed-budget pop sizes without recalibrating experiments."""
    path = RESULTS_DIR / "asoc_portfolio_budget_calibration.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        universe_name: {algorithm_name: "not_recorded" for algorithm_name in MAIN_ALGORITHMS}
        for universe_name in UNIVERSES
    }


def main():
    csv_path = RESULTS_DIR / "asoc_portfolio_raw_metrics.csv"
    if not csv_path.exists():
        legacy_path = RESULTS_DIR / "portfolio_case_raw_metrics.csv"
        legacy_msg = ""
        if legacy_path.exists():
            legacy_msg = (
                " A legacy five-algorithm portfolio_case_raw_metrics.csv was found, "
                "but it is intentionally not accepted for ASOC tables."
            )
        raise FileNotFoundError(
            "Missing experiments/results/asoc_portfolio_raw_metrics.csv. "
            "Run experiments/run_portfolio_asoc_suite.py or experiments/run_mddm_portfolio.py first."
            + legacy_msg
        )

    raw_df = pd.read_csv(csv_path)
    raw_df["universe"] = pd.Categorical(raw_df["universe"], categories=list(UNIVERSES.keys()), ordered=True)
    raw_df["decision_rule"] = pd.Categorical(raw_df["decision_rule"], categories=list(DECISION_RULES), ordered=True)

    coverage_df = validate_portfolio_algorithm_coverage(raw_df, RESULTS_DIR / "asoc_portfolio_algorithm")

    primary_df = raw_df[(raw_df["decision_rule"] == PRIMARY_RULE) & np.isclose(raw_df["cost_rate"], PRIMARY_COST)].copy()
    activation_events = activation_event_frame(primary_df.to_dict("records"))
    activation_summary = summarize_activation_audit(primary_df)

    mhv_summary, migd_summary, deployment_summary = summarize_primary(raw_df)
    sensitivity_full, sensitivity_counts = summarize_sensitivity(raw_df)
    benchmark_df = evaluate_simple_benchmarks()
    stats_df = pairwise_stats(primary_df, family_name="main")
    budget_stats_df = pairwise_stats(primary_df, family_name="budget")

    mhv_summary.to_csv(RESULTS_DIR / "asoc_portfolio_mhv_summary.csv", index=False)
    migd_summary.to_csv(RESULTS_DIR / "asoc_portfolio_migd_summary.csv", index=False)
    deployment_summary.to_csv(RESULTS_DIR / "asoc_portfolio_deployment_summary.csv", index=False)
    sensitivity_full.to_csv(RESULTS_DIR / "asoc_portfolio_sensitivity_full.csv", index=False)
    sensitivity_counts.to_csv(RESULTS_DIR / "asoc_portfolio_sensitivity_counts.csv", index=False)
    benchmark_df.to_csv(RESULTS_DIR / "asoc_portfolio_benchmarks.csv", index=False)
    stats_df.to_csv(RESULTS_DIR / "asoc_portfolio_stats.csv", index=False)
    budget_stats_df.to_csv(RESULTS_DIR / "asoc_portfolio_budget_stats.csv", index=False)
    coverage_df.to_csv(RESULTS_DIR / "asoc_portfolio_algorithm_coverage.csv", index=False)
    activation_summary.to_csv(RESULTS_DIR / "asoc_portfolio_activation_audit_summary.csv", index=False)
    activation_events.to_csv(RESULTS_DIR / "asoc_portfolio_activation_audit_events.csv", index=False)

    fixed_budget_pop_sizes = load_budget_calibration()

    (RESULTS_DIR / "asoc_portfolio_mhv_table.tex").write_text(
        primary_table_latex(
            mhv_summary,
            metric="mhv",
            higher_is_better=True,
            caption="Mean out-of-sample normalized HV on the primary walk-forward portfolio protocol.",
            label="tab:portfolio-mhv",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "asoc_portfolio_migd_table.tex").write_text(
        primary_table_latex(
            migd_summary,
            metric="migd",
            higher_is_better=False,
            caption="Mean out-of-sample IGD on the primary walk-forward portfolio protocol.",
            label="tab:portfolio-migd",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "asoc_portfolio_deployment_table.tex").write_text(
        deployment_table_latex(
            deployment_summary,
            caption="Primary walk-forward deployment metrics under closest-to-utopia selection at 10 bps.",
            label="tab:portfolio-deployment",
        ),
        encoding="utf-8",
    )

    write_summary(
        primary_df,
        sensitivity_counts,
        benchmark_df,
        stats_df,
        budget_stats_df,
        fixed_budget_pop_sizes,
        RESULTS_DIR / "asoc_portfolio_summary.txt",
    )


if __name__ == "__main__":
    main()
