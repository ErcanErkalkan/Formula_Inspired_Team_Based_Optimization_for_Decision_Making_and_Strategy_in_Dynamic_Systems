from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RAW_PATH = RESULTS_DIR / "asoc_dynamic_raw_metrics.csv"
DEBUG_PATH = RESULTS_DIR / "predictive_baseline_debug_smoke.csv"
SUMMARY_PATH = RESULTS_DIR / "asoc_dynamic_activation_validation_summary.csv"


def _failures_for_debug(df: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    real = df[df["algorithm"].isin(["NSGA-II", "MDDM-DMOEA", "PPS-DMOEA"])]
    for key, group in real.groupby(["protocol", "problem", "seed"], sort=True):
        by_alg = {row["algorithm"]: row for _, row in group.iterrows()}
        if "NSGA-II" not in by_alg:
            failures.append(f"{key}: NSGA-II missing")
            continue
        nsga = by_alg["NSGA-II"]
        if int(nsga.get("change_response_count", 0)) != 0:
            failures.append(f"{key}: NSGA-II response count should be zero")
        for alg in ["MDDM-DMOEA", "PPS-DMOEA"]:
            if alg not in by_alg:
                failures.append(f"{key}: {alg} missing")
                continue
            row = by_alg[alg]
            if int(row.get("environment_change_count", 0)) <= 0:
                failures.append(f"{key}: {alg} did not observe environment changes")
            if int(row.get("change_response_count", 0)) <= 0:
                failures.append(f"{key}: {alg} did not activate response")
            if int(row.get("replaced_count", 0)) <= 0:
                failures.append(f"{key}: {alg} did not replace individuals")
            if int(row.get("response_evaluation_count", 0)) <= 0:
                failures.append(f"{key}: {alg} lacks response_evaluation_count audit")
            if int(row.get("n_evals", -1)) != int(nsga.get("n_evals", -2)):
                failures.append(f"{key}: {alg} nominal n_evals differs from NSGA-II in debug smoke test")
    return failures


def _failures_for_full_dynamic(df: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    required_cols = {
        "family",
        "protocol",
        "problem",
        "seed",
        "algorithm",
        "environment_change_count",
        "change_response_count",
        "response_activation_rate",
        "replaced_count",
        "response_evaluation_count",
        "n_evals",
    }
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        return [f"asoc_dynamic_raw_metrics.csv missing columns: {missing_cols}"]

    main = df[df["family"].astype(str).isin(["main", "budget"])]
    if main.empty:
        return ["No main/budget dynamic rows found"]

    summary = (
        main.groupby(["family", "algorithm"])[
            [
                "environment_change_count",
                "change_response_count",
                "response_activation_rate",
                "replaced_count",
                "response_evaluation_count",
                "n_evals",
            ]
        ]
        .mean()
        .reset_index()
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    for family in ["main", "budget"]:
        fam = main[main["family"] == family]
        if fam.empty:
            continue
        nsga = fam[fam["algorithm"] == "NSGA-II"]
        if nsga.empty:
            failures.append(f"{family}: NSGA-II rows missing")
        elif nsga["change_response_count"].fillna(0).max() != 0:
            failures.append(f"{family}: NSGA-II should remain zero-response")

        for alg in ["MDDM-DMOEA", "PPS-DMOEA"]:
            rows = fam[fam["algorithm"] == alg]
            if rows.empty:
                failures.append(f"{family}: {alg} rows missing")
                continue
            if rows["environment_change_count"].fillna(0).mean() <= 0:
                failures.append(f"{family}: {alg} did not observe environmental changes on average")
            if rows["change_response_count"].fillna(0).mean() <= 0:
                failures.append(f"{family}: {alg} did not activate responses on average")
            if rows["replaced_count"].fillna(0).mean() <= 0:
                failures.append(f"{family}: {alg} did not replace individuals on average")
            if rows["response_evaluation_count"].fillna(0).mean() <= 0:
                failures.append(f"{family}: {alg} response_evaluation_count audit is empty")

        # In the generation-matched family, MDDM/PPS nominal n_evals should not be
        # inflated by response-refresh evaluation.  Debug script performs exact
        # per-seed equality; here we use a coarse mean-level guard for the full suite.
        if family == "main" and not nsga.empty:
            nsga_mean = float(nsga["n_evals"].mean())
            for alg in ["MDDM-DMOEA", "PPS-DMOEA"]:
                rows = fam[fam["algorithm"] == alg]
                if not rows.empty and abs(float(rows["n_evals"].mean()) - nsga_mean) > 1e-9:
                    failures.append(f"{family}: {alg} mean nominal n_evals differs from NSGA-II after response-refresh correction")

    return failures


def main() -> None:
    failures: list[str] = []
    if DEBUG_PATH.exists():
        failures.extend(_failures_for_debug(pd.read_csv(DEBUG_PATH)))
    else:
        failures.append(f"Missing debug file: {DEBUG_PATH}")

    if RAW_PATH.exists():
        failures.extend(_failures_for_full_dynamic(pd.read_csv(RAW_PATH)))
    else:
        failures.append(f"Missing full dynamic raw metrics file: {RAW_PATH}")

    if failures:
        message = "Dynamic activation audit validation failed:\n" + "\n".join(f"- {item}" for item in failures[:40])
        if len(failures) > 40:
            message += f"\n- ... {len(failures) - 40} additional failures"
        raise SystemExit(message)

    print("Dynamic activation audit validation passed.")
    print(f"Wrote summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
