from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SUMMARY_PATH = RESULTS_DIR / "asoc_dynamic_sensitivity_summary.csv"


def _plot_parameter(ax, df: pd.DataFrame, parameter: str, xlabel: str, title: str) -> None:
    overall = df[(df["parameter"] == parameter) & (df["problem"] == "OVERALL")].copy()
    overall["value_numeric"] = overall["value"].astype(float)
    overall = overall.sort_values("value_numeric")
    yerr_low = overall["mean_migd"] - overall["ci95_low"]
    yerr_high = overall["ci95_high"] - overall["mean_migd"]
    ax.errorbar(
        overall["value_numeric"],
        overall["mean_migd"],
        yerr=[yerr_low, yerr_high],
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean MIGD across DF1/DF4/DF7/DF9")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(overall["value_numeric"].tolist())


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run `python experiments/run_dynamic_sensitivity_asoc.py` first."
        )
    df = pd.read_csv(SUMMARY_PATH)
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    _plot_parameter(
        axes[0],
        df,
        "lambda_1",
        r"Directed correction scaling ($\lambda_1$)",
        r"Sensitivity to $\lambda_1$",
    )
    _plot_parameter(
        axes[1],
        df,
        "stagnation_limit",
        r"Stagnation tolerance ($s$ generations)",
        r"Sensitivity to $s$",
    )
    fig.tight_layout()
    out = RESULTS_DIR / "sensitivity_analysis.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / "sensitivity_analysis.png", dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
