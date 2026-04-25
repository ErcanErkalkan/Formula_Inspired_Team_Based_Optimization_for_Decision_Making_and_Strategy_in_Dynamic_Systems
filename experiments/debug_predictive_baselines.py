from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from run_dynamic_asoc_suite import PROTOCOLS, run_dynamic_baseline
except ModuleNotFoundError:
    from experiments.run_dynamic_asoc_suite import PROTOCOLS, run_dynamic_baseline


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


BASELINE_DEBUG_ALGORITHMS = ["NSGA-II", "MDDM-DMOEA", "PPS-DMOEA"]


def _validate_debug_frame(df: pd.DataFrame) -> None:
    real = df[df["algorithm"].isin(BASELINE_DEBUG_ALGORITHMS)].copy()
    failures: list[str] = []

    for key, group in real.groupby(["protocol", "problem", "seed"], sort=True):
        by_alg = {row["algorithm"]: row for _, row in group.iterrows()}
        missing = [alg for alg in BASELINE_DEBUG_ALGORITHMS if alg not in by_alg]
        if missing:
            failures.append(f"{key}: missing algorithms {missing}")
            continue

        nsga = by_alg["NSGA-II"]
        if int(nsga["environment_change_count"]) <= 0:
            failures.append(f"{key}: NSGA-II did not observe environment changes")
        if int(nsga["change_response_count"]) != 0:
            failures.append(f"{key}: NSGA-II should remain zero-response")

        nsga_evals = int(nsga["n_evals"])
        for alg in ["MDDM-DMOEA", "PPS-DMOEA"]:
            row = by_alg[alg]
            if int(row["environment_change_count"]) <= 0:
                failures.append(f"{key}: {alg} did not observe environment changes")
            if int(row["change_response_count"]) <= 0:
                failures.append(f"{key}: {alg} response was not activated")
            if float(row["response_activation_rate"]) <= 0.0:
                failures.append(f"{key}: {alg} activation rate is zero")
            if int(row["replaced_count"]) <= 0:
                failures.append(f"{key}: {alg} did not replace any individuals")
            if int(row["response_evaluation_count"]) <= 0:
                failures.append(f"{key}: {alg} response refresh evaluations were not audited")
            if int(row["n_evals"]) != nsga_evals:
                failures.append(
                    f"{key}: {alg} nominal n_evals={int(row['n_evals'])} differs from NSGA-II n_evals={nsga_evals}; "
                    "response refreshes should be recorded in response_evaluation_count, not added to nominal n_evals"
                )

        if abs(float(by_alg["MDDM-DMOEA"]["migd"]) - float(nsga["migd"])) == 0.0:
            failures.append(f"{key}: MDDM-DMOEA MIGD is still identical to NSGA-II")
        if abs(float(by_alg["PPS-DMOEA"]["migd"]) - float(nsga["migd"])) == 0.0:
            failures.append(f"{key}: PPS-DMOEA MIGD is still identical to NSGA-II")

    if failures:
        message = "Predictive baseline debug validation failed:\n" + "\n".join(f"- {item}" for item in failures[:30])
        if len(failures) > 30:
            message += f"\n- ... {len(failures) - 30} additional failures"
        raise SystemExit(message)


def main() -> None:
    """Small audit run for MDDM-DMOEA/PPS-DMOEA change-response behavior.

    The script verifies four properties before the full ASOC suite is launched:
    (i) NSGA-II observes environmental changes but remains a zero-response baseline;
    (ii) MDDM-DMOEA and PPS-DMOEA activate at changes;
    (iii) MDDM-DMOEA/PPS-DMOEA no longer duplicate NSGA-II metrics;
    (iv) response-refresh evaluations are audited separately and do not inflate
    nominal generation-matched ``n_evals``.
    """
    protocols = ["fast_t5_n10", "moderate_t10_n10"]
    problems = ["df1", "df2"]
    algorithms = BASELINE_DEBUG_ALGORITHMS
    seeds = [0, 1, 2]
    rows = []
    for protocol in protocols:
        pop_size = int(PROTOCOLS[protocol]["pop_size"])
        for problem in problems:
            for seed in seeds:
                seed_results = {}
                for algorithm in algorithms:
                    result = run_dynamic_baseline(protocol, problem, algorithm, seed, pop_size)
                    seed_results[algorithm] = result
                    rows.append(
                        {
                            "protocol": protocol,
                            "problem": problem,
                            "seed": seed,
                            "algorithm": algorithm,
                            "migd": result["migd"],
                            "tail_igd": result["tail_igd"],
                            "n_evals": result["n_evals"],
                            "environment_change_count": result.get("environment_change_count", 0),
                            "change_response_count": result.get("change_response_count", 0),
                            "response_activation_rate": result.get("response_activation_rate", 0.0),
                            "prediction_count": result.get("prediction_count", 0),
                            "replaced_count": result.get("replaced_count", 0),
                            "kde_success_count": result.get("kde_success_count", 0),
                            "kde_fallback_count": result.get("kde_fallback_count", 0),
                            "response_evaluation_count": result.get("response_evaluation_count", 0),
                        }
                    )
                nsga_migd = seed_results["NSGA-II"]["migd"]
                for algorithm in ["MDDM-DMOEA", "PPS-DMOEA"]:
                    diff = abs(seed_results[algorithm]["migd"] - nsga_migd)
                    rows.append(
                        {
                            "protocol": protocol,
                            "problem": problem,
                            "seed": seed,
                            "algorithm": f"{algorithm}_vs_NSGAII_DIFF",
                            "migd": diff,
                            "tail_igd": abs(seed_results[algorithm]["tail_igd"] - seed_results["NSGA-II"]["tail_igd"]),
                            "n_evals": 0,
                            "environment_change_count": seed_results[algorithm].get("environment_change_count", 0),
                            "change_response_count": seed_results[algorithm].get("change_response_count", 0),
                            "response_activation_rate": seed_results[algorithm].get("response_activation_rate", 0.0),
                            "prediction_count": seed_results[algorithm].get("prediction_count", 0),
                            "replaced_count": seed_results[algorithm].get("replaced_count", 0),
                            "kde_success_count": seed_results[algorithm].get("kde_success_count", 0),
                            "kde_fallback_count": seed_results[algorithm].get("kde_fallback_count", 0),
                            "response_evaluation_count": seed_results[algorithm].get("response_evaluation_count", 0),
                        }
                    )
    out = RESULTS_DIR / "predictive_baseline_debug_smoke.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    _validate_debug_frame(df)
    print(f"Wrote {out}")
    print(
        df.groupby("algorithm")[
            [
                "environment_change_count",
                "change_response_count",
                "response_activation_rate",
                "prediction_count",
                "replaced_count",
                "response_evaluation_count",
                "n_evals",
                "migd",
            ]
        ].mean()
    )
    print("Predictive baseline debug validation passed.")


if __name__ == "__main__":
    main()
