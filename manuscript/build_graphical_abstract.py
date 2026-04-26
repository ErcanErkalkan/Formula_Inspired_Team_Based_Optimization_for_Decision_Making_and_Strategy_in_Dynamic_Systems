from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "experiments" / "results"


def add_box(ax, xy, width, height, title, lines, facecolor):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.8,
        edgecolor="#1f2933",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + 0.02, xy[1] + height - 0.06, title, fontsize=16, fontweight="bold", color="#102a43")
    y = xy[1] + height - 0.12
    for line in lines:
        ax.text(xy[0] + 0.025, y, line, fontsize=11.5, color="#243b53")
        y -= 0.055


def main() -> None:
    dynamic_ranks = pd.read_csv(RESULTS / "asoc_dynamic_main_ranks.csv")
    dynamic_budget_ranks = pd.read_csv(RESULTS / "asoc_dynamic_budget_ranks.csv")
    portfolio_deploy = pd.read_csv(RESULTS / "asoc_portfolio_deployment_summary.csv", header=[0, 1])

    dynamic_rank = dynamic_ranks.groupby("algorithm")["average_rank"].mean().sort_values()
    budget_rank = dynamic_budget_ranks.groupby("algorithm")["average_rank"].mean().sort_values()

    tech14 = portfolio_deploy[
        (portfolio_deploy[("family", "Unnamed: 0_level_1")] == "main")
        & (portfolio_deploy[("universe", "Unnamed: 1_level_1")] == "tech14")
    ]
    market20 = portfolio_deploy[
        (portfolio_deploy[("family", "Unnamed: 0_level_1")] == "main")
        & (portfolio_deploy[("universe", "Unnamed: 1_level_1")] == "market20")
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.text(0.04, 0.95, "Graphical Abstract", fontsize=12, color="#486581")
    ax.text(
        0.04,
        0.90,
        "Formula-Inspired Team Optimization (FITO) for Dynamic Multi-Objective Tracking",
        fontsize=20,
        fontweight="bold",
        color="#102a43",
    )

    add_box(
        ax,
        (0.04, 0.18),
        0.26,
        0.60,
        "Method",
        [
            "Leader-support search",
            "Team-guided variation",
            "Pit-stop restart after stagnation",
            "Change-triggered redeployment",
            "Predictive leader anchors",
            "Rejected add-ons: boundary risk,",
            "anchor-leader blending",
        ],
        "#e0fbfc",
    )

    add_box(
        ax,
        (0.37, 0.18),
        0.26,
        0.60,
        "Dynamic Evidence",
        [
            "DF1-DF9, 3 protocols, 20 runs",
            f"Generation-matched avg. rank: {dynamic_rank['FITO']:.3f}",
            f"Fixed-budget avg. rank: {budget_rank['FITO']:.3f}",
            "Holm-significant wins: 134",
            "Holm-significant losses: 18",
            "Pit-stop ablation:",
            "FITO 1.386 vs FITO-noPS 1.614",
        ],
        "#fff3c4",
    )

    tech_fito = tech14[tech14[("algorithm", "Unnamed: 2_level_1")] == "FITO"].iloc[0]
    market_fito = market20[market20[("algorithm", "Unnamed: 2_level_1")] == "FITO"].iloc[0]
    market_nsga = market20[market20[("algorithm", "Unnamed: 2_level_1")] == "NSGA-II"].iloc[0]
    tech_dnsga = tech14[tech14[("algorithm", "Unnamed: 2_level_1")] == "DNSGA-II-A"].iloc[0]

    add_box(
        ax,
        (0.70, 0.18),
        0.26,
        0.60,
        "Walk-Forward Evidence",
        [
            "2 universes, 38 periods, 3 rules, 3 costs",
            f"Tech14 wealth: FITO {tech_fito[('final_wealth', 'mean')]:.2f}",
            f"Tech14 runner-up: DNSGA-II-A {tech_dnsga[('final_wealth', 'mean')]:.2f}",
            f"Market20 FITO wealth: {market_fito[('final_wealth', 'mean')]:.2f}",
            f"Market20 best wealth: NSGA-II {market_nsga[('final_wealth', 'mean')]:.2f}",
            "Interpretation:",
            "strong synthetic evidence, mixed",
            "application evidence",
        ],
        "#fde2e4",
    )

    for start_x, end_x in ((0.31, 0.37), (0.63, 0.70)):
        ax.add_patch(
            FancyArrowPatch(
                (start_x, 0.48),
                (end_x, 0.48),
                arrowstyle="simple",
                mutation_scale=20,
                linewidth=0,
                facecolor="#334e68",
                edgecolor="#334e68",
                alpha=0.9,
            )
        )

    ax.text(
        0.5,
        0.08,
        "Claim scope: competitive and budget-aware dynamic tracking, not universal dominance",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#102a43",
    )

    fig.tight_layout()
    fig.savefig(ROOT / "graphical_abstract.png", dpi=300, bbox_inches="tight")
    fig.savefig(ROOT / "graphical_abstract.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
