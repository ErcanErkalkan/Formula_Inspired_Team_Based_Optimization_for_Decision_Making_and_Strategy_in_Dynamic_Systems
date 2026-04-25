from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

try:
    from evaluation_counter import attach_evaluation_counter
except ModuleNotFoundError:
    from experiments.evaluation_counter import attach_evaluation_counter

try:
    from stats_utils import holm_adjust, mann_whitney_summary, mean_ci95
except ModuleNotFoundError:
    from experiments.stats_utils import holm_adjust, mann_whitney_summary, mean_ci95


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBLEM_SETTINGS = {
    "zdt1": {"pop_size": 100, "generations": 180},
    "zdt2": {"pop_size": 100, "generations": 180},
    "zdt3": {"pop_size": 100, "generations": 180},
    "zdt4": {"pop_size": 100, "generations": 250},
    "zdt6": {"pop_size": 100, "generations": 180},
}

MAIN_ALGORITHMS = ("FITO", "NSGA-II", "MOEA/D", "RVEA")
ABLATION_ALGORITHMS = ("FITO", "FITO+BR", "FITO-noSR", "FITO-noLS")
SEEDS = tuple(range(20))

LEADER_PAIRING_PROB = 0.80
LEADER_PULL = 0.10
DIFFERENTIAL_PUSH = 0.15
RESTART_RATE = 0.125
RESTART_SIGMA = 0.08
STAGNATION_LIMIT = 8
FITO_USE_BOUNDARY_RISK = False


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n_points, n_obj = F.shape
    if n_points == 0:
        return np.array([])
    if n_points <= 2:
        return np.full(n_points, np.inf)

    distance = np.zeros(n_points)
    for j in range(n_obj):
        order = np.argsort(F[:, j])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        f_min = F[order[0], j]
        f_max = F[order[-1], j]
        if np.isclose(f_max, f_min):
            continue
        distance[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / (f_max - f_min)
    return distance


def environmental_selection(X: np.ndarray, F: np.ndarray, pop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fronts = NonDominatedSorting().do(F)
    chosen = []
    crowd = np.zeros(len(F))
    ranks = np.full(len(F), np.inf)

    for rank, front in enumerate(fronts):
        ranks[front] = rank
        cd = crowding_distance(F[front])
        crowd[front] = cd
        if len(chosen) + len(front) <= pop_size:
            chosen.extend(front.tolist())
        else:
            remain = pop_size - len(chosen)
            order = np.argsort(-cd)
            chosen.extend(front[order[:remain]].tolist())
            break

    chosen = np.asarray(chosen, dtype=int)
    return X[chosen], F[chosen], ranks[chosen], crowd[chosen]


def non_dominated_unique(F: np.ndarray) -> np.ndarray:
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    points = F[front]
    unique_points, idx = np.unique(np.round(points, decimals=8), axis=0, return_index=True)
    return unique_points[np.argsort(idx)]


def tournament_indices(ranks: np.ndarray, crowd: np.ndarray, rng: np.random.Generator, n_select: int) -> np.ndarray:
    selected = []
    n = len(ranks)
    for _ in range(n_select):
        a, b = rng.integers(0, n, size=2)
        if ranks[a] < ranks[b]:
            selected.append(a)
        elif ranks[b] < ranks[a]:
            selected.append(b)
        else:
            selected.append(a if crowd[a] > crowd[b] else b)
    return np.asarray(selected, dtype=int)


def weakest_support_indices(
    order: np.ndarray,
    ranks: np.ndarray,
    crowd: np.ndarray,
    elite_n: int,
    n_select: int,
) -> np.ndarray:
    support = order[elite_n:]
    if len(support) == 0 or n_select <= 0:
        return np.array([], dtype=int)
    weakness = np.lexsort((crowd[support], -ranks[support]))
    return support[weakness[: min(n_select, len(support))]]


def sbx(
    p1: np.ndarray,
    p2: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta: float = 25.0,
    prob: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    c1 = p1.copy()
    c2 = p2.copy()

    if rng.random() > prob:
        return c1, c2

    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) <= 1e-14:
            continue

        y1, y2 = sorted((p1[i], p2[i]))
        yl = xl[i]
        yu = xu[i]
        rand = rng.random()

        beta = 1.0 + 2.0 * (y1 - yl) / (y2 - y1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        beta = 1.0 + 2.0 * (yu - y2) / (y2 - y1)
        alpha = 2.0 - beta ** (-(eta + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
        else:
            betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
        child2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        child1 = np.clip(child1, yl, yu)
        child2 = np.clip(child2, yl, yu)

        if rng.random() <= 0.5:
            c1[i], c2[i] = child2, child1
        else:
            c1[i], c2[i] = child1, child2

    return c1, c2


def polynomial_mutation(
    x: np.ndarray,
    xl: np.ndarray,
    xu: np.ndarray,
    eta: float = 20.0,
    prob: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    y = x.copy()
    prob = 1.0 / len(x) if prob is None else prob

    for i in range(len(x)):
        if rng.random() > prob:
            continue

        yl = xl[i]
        yu = xu[i]
        if yl == yu:
            continue

        delta1 = (y[i] - yl) / (yu - yl)
        delta2 = (yu - y[i]) / (yu - yl)
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)

        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
            deltaq = 1.0 - val ** mut_pow

        y[i] = np.clip(y[i] + deltaq * (yu - yl), yl, yu)

    return y


def boundary_risk_scale(x: np.ndarray, xl: np.ndarray, xu: np.ndarray) -> float:
    normalized = (x - xl) / (xu - xl + 1e-12)
    margin = np.minimum(normalized, 1.0 - normalized)
    return float(np.clip(0.35 + 1.3 * np.mean(margin), 0.35, 1.0))


def adaptive_hv_score(F: np.ndarray) -> float:
    nd_front = non_dominated_unique(F)
    if len(nd_front) == 0:
        return float("-inf")

    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    normalized = np.clip((nd_front - ideal) / scale, 0.0, 1.0)
    ref_point = np.full(F.shape[1], 1.1)
    return float(HV(ref_point=ref_point)(normalized))


def fito_optimize(
    problem_name: str,
    seed: int,
    use_stagnation_restart: bool = True,
    use_leader_support: bool = True,
    use_boundary_risk: bool = FITO_USE_BOUNDARY_RISK,
) -> tuple[np.ndarray, int]:
    cfg = PROBLEM_SETTINGS[problem_name]
    problem = get_problem(problem_name)
    eval_counter = attach_evaluation_counter(problem)
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    pop_size = cfg["pop_size"]
    generations = cfg["generations"]

    X = rng.uniform(xl, xu, size=(pop_size, problem.n_var))
    F = problem.evaluate(X)

    best_score = adaptive_hv_score(F)
    stagnant = 0

    for _ in range(generations):
        X, F, ranks, crowd = environmental_selection(X, F, pop_size)
        order = np.lexsort((-crowd, ranks))
        elite_n = max(6, pop_size // 10)
        leaders = X[order[:elite_n]]
        guides = X[order[: max(elite_n * 3, elite_n + 1)]]

        offspring = []
        while len(offspring) < pop_size:
            if use_leader_support and rng.random() < LEADER_PAIRING_PROB:
                p1 = leaders[rng.integers(len(leaders))]
                p2 = guides[rng.integers(len(guides))]
            else:
                idx = tournament_indices(ranks, crowd, rng, 2)
                p1 = X[idx[0]]
                p2 = X[idx[1]]

            c1, c2 = sbx(p1, p2, xl, xu, eta=25.0, prob=0.95, rng=rng)

            if use_leader_support:
                leader = leaders[rng.integers(len(leaders))]
                diff = X[rng.integers(pop_size)] - X[rng.integers(pop_size)]
                risk_c1 = boundary_risk_scale(c1, xl, xu) if use_boundary_risk else 1.0
                risk_c2 = boundary_risk_scale(c2, xl, xu) if use_boundary_risk else 1.0
                c1 = np.clip(c1 + risk_c1 * LEADER_PULL * (leader - c1), xl, xu)
                c2 = np.clip(c2 + risk_c2 * DIFFERENTIAL_PUSH * diff, xl, xu)

            c1 = polynomial_mutation(c1, xl, xu, eta=30.0, rng=rng)
            c2 = polynomial_mutation(c2, xl, xu, eta=20.0, rng=rng)
            offspring.extend((c1, c2))

        Xo = np.asarray(offspring[:pop_size])
        Fo = problem.evaluate(Xo)
        X = np.vstack((X, Xo))
        F = np.vstack((F, Fo))
        X, F, ranks, crowd = environmental_selection(X, F, pop_size)
        order = np.lexsort((-crowd, ranks))
        leaders = X[order[:elite_n]]

        current_score = adaptive_hv_score(F)
        if current_score > best_score + 1e-7:
            best_score = current_score
            stagnant = 0
        else:
            stagnant += 1

        if use_stagnation_restart and stagnant >= STAGNATION_LIMIT:
            restart_idx = weakest_support_indices(order, ranks, crowd, elite_n, max(4, int(pop_size * RESTART_RATE)))
            for idx in restart_idx:
                if rng.random() < 0.6:
                    base = leaders[rng.integers(len(leaders))]
                    X[idx] = np.clip(base + rng.normal(0.0, RESTART_SIGMA, size=problem.n_var) * (xu - xl), xl, xu)
                else:
                    X[idx] = np.clip(xl + xu - X[idx], xl, xu)
            F[restart_idx] = problem.evaluate(X[restart_idx])
            stagnant = 0

    return non_dominated_unique(F), int(eval_counter.count)


def run_baseline(problem_name: str, algorithm_name: str, seed: int) -> tuple[np.ndarray, int]:
    cfg = PROBLEM_SETTINGS[problem_name]
    problem = get_problem(problem_name)
    eval_counter = attach_evaluation_counter(problem)
    pop_size = cfg["pop_size"]
    generations = cfg["generations"]

    if algorithm_name == "NSGA-II":
        algorithm = NSGA2(pop_size=pop_size)
    elif algorithm_name == "MOEA/D":
        ref_dirs = get_reference_directions("uniform", problem.n_obj, n_points=pop_size)
        algorithm = MOEAD(ref_dirs=ref_dirs)
    elif algorithm_name == "RVEA":
        ref_dirs = get_reference_directions("das-dennis", problem.n_obj, n_partitions=99)
        algorithm = RVEA(ref_dirs=ref_dirs)
    else:
        raise ValueError(f"Unknown baseline: {algorithm_name}")

    result = minimize(problem, algorithm, ("n_gen", generations), seed=seed, verbose=False)
    return non_dominated_unique(result.F), int(eval_counter.count)


def evaluate_front(problem_name: str, front: np.ndarray) -> tuple[float, float]:
    problem = get_problem(problem_name)
    pareto_front = problem.pareto_front()
    hv = HV(ref_point=np.array([1.1] * problem.n_obj))(front)
    igd = IGD(pareto_front)(front)
    return float(hv), float(igd)


def run_task(task: dict[str, object]) -> dict[str, object]:
    problem_name = str(task["problem"])
    algorithm_name = str(task["algorithm"])
    seed = int(task["seed"])
    ablation = bool(task.get("ablation", False))

    start = time.perf_counter()
    if algorithm_name == "FITO":
        front, n_evals = fito_optimize(problem_name, seed, use_stagnation_restart=True, use_leader_support=True)
    elif algorithm_name == "FITO+BR":
        front, n_evals = fito_optimize(
            problem_name,
            seed,
            use_stagnation_restart=True,
            use_leader_support=True,
            use_boundary_risk=True,
        )
    elif algorithm_name == "FITO-noSR":
        front, n_evals = fito_optimize(problem_name, seed, use_stagnation_restart=False, use_leader_support=True)
    elif algorithm_name == "FITO-noLS":
        front, n_evals = fito_optimize(problem_name, seed, use_stagnation_restart=True, use_leader_support=False)
    else:
        front, n_evals = run_baseline(problem_name, algorithm_name, seed)

    runtime_sec = time.perf_counter() - start
    hv, igd = evaluate_front(problem_name, front)

    return {
        "problem": problem_name,
        "algorithm": algorithm_name,
        "seed": seed,
        "ablation": ablation,
        "hv": hv,
        "igd": igd,
        "n_evals": n_evals,
        "runtime_sec": runtime_sec,
        "front_size": int(len(front)),
        "front": front.tolist(),
    }


def build_tasks() -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for algorithm in MAIN_ALGORITHMS:
        for problem_name in PROBLEM_SETTINGS:
            for seed in SEEDS:
                tasks.append({"algorithm": algorithm, "problem": problem_name, "seed": seed, "ablation": False})

    for algorithm in ABLATION_ALGORITHMS:
        for problem_name in ("zdt4", "zdt6"):
            for seed in SEEDS:
                tasks.append({"algorithm": algorithm, "problem": problem_name, "seed": seed, "ablation": True})
    return tasks


def summarize_metrics(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_df = (
        raw_df.groupby(["problem", "algorithm"])[["hv", "igd", "runtime_sec", "front_size"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    ci_records = []
    for (problem_name, algorithm_name), group in raw_df.groupby(["problem", "algorithm"]):
        for metric in ("hv", "igd"):
            ci_low, ci_high = mean_ci95(group[metric].to_numpy())
            ci_records.append(
                {
                    "problem": problem_name,
                    "algorithm": algorithm_name,
                    "metric": metric,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )
    ci_df = pd.DataFrame(ci_records)
    mean_df = mean_df.merge(ci_df.pivot(index=["problem", "algorithm"], columns="metric"), on=["problem", "algorithm"])

    records = []
    for metric, higher_is_better in (("hv", True), ("igd", False)):
        for problem_name, group in raw_df.groupby("problem"):
            pivot = group.pivot(index="seed", columns="algorithm", values=metric)
            if higher_is_better:
                rank_matrix = pivot.rank(axis=1, ascending=False, method="average")
            else:
                rank_matrix = pivot.rank(axis=1, ascending=True, method="average")
            avg_ranks = rank_matrix.mean(axis=0)
            for algorithm_name, value in avg_ranks.items():
                records.append(
                    {
                        "problem": problem_name,
                        "metric": metric,
                        "algorithm": algorithm_name,
                        "average_rank": float(value),
                    }
                )

    rank_df = pd.DataFrame(records)
    return mean_df, rank_df


def summarize_eval_budget(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.groupby("algorithm")["n_evals"].agg(["mean", "std", "min", "max"]).reset_index()


def pairwise_significance_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    raw_p_values = []
    for metric, higher_is_better in (("hv", True), ("igd", False)):
        for problem_name in PROBLEM_SETTINGS:
            fito_values = (
                raw_df[(raw_df["problem"] == problem_name) & (raw_df["algorithm"] == "FITO")]
                .sort_values("seed")[metric]
                .to_numpy()
            )
            for baseline in ("NSGA-II", "MOEA/D", "RVEA"):
                base_values = (
                    raw_df[(raw_df["problem"] == problem_name) & (raw_df["algorithm"] == baseline)]
                    .sort_values("seed")[metric]
                    .to_numpy()
                )
                stats = mann_whitney_summary(fito_values, base_values, higher_is_better=higher_is_better)
                comparisons.append(
                    {
                        "problem": problem_name,
                        "metric": metric,
                        "baseline": baseline,
                        "fito_mean": stats["mean_a"],
                        "baseline_mean": stats["mean_b"],
                        "mann_whitney_u": stats["mann_whitney_u"],
                        "p_value": stats["p_value"],
                        "fito_better": stats["a_better"],
                        "cliffs_delta": stats["cliffs_delta"],
                        "fito_ci95_low": stats["a_ci95_low"],
                        "fito_ci95_high": stats["a_ci95_high"],
                        "baseline_ci95_low": stats["b_ci95_low"],
                        "baseline_ci95_high": stats["b_ci95_high"],
                    }
                )
                raw_p_values.append(float(stats["p_value"]))
    adjusted = holm_adjust(raw_p_values)
    family_size = len(raw_p_values)
    for row, corrected in zip(comparisons, adjusted):
        row["holm_p_value"] = corrected
        row["holm_family_size"] = family_size
        row["holm_scope"] = f"global_static_family_{family_size}_comparisons"
    return pd.DataFrame(comparisons)


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} $\\pm$ {std_value:.4f}"


def table_to_latex(
    summary_df: pd.DataFrame,
    metric: str,
    algorithms: tuple[str, ...],
    caption: str,
    label: str,
    problem_order: list[str] | tuple[str, ...] | None = None,
) -> str:
    metric_summary = summary_df.copy()
    problem_order = list(problem_order or PROBLEM_SETTINGS.keys())
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(algorithms) + "}",
        "\\toprule",
        "Problem & " + " & ".join(algorithms) + " \\\\",
        "\\midrule",
    ]

    for problem_name in problem_order:
        row = []
        problem_rows = metric_summary[metric_summary["problem"] == problem_name]
        means = []
        for algorithm in algorithms:
            algo_row = problem_rows[problem_rows["algorithm"] == algorithm].iloc[0]
            means.append(float(algo_row[(metric, "mean")]))

        best_value = max(means) if metric == "hv" else min(means)
        for algorithm in algorithms:
            algo_row = problem_rows[problem_rows["algorithm"] == algorithm].iloc[0]
            cell = format_mean_std(float(algo_row[(metric, "mean")]), float(algo_row[(metric, "std")]))
            if np.isclose(float(algo_row[(metric, "mean")]), best_value):
                cell = "\\textbf{" + cell + "}"
            row.append(cell)

        lines.append(problem_name.upper() + " & " + " & ".join(row) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def static_ablation_table_to_latex(
    summary_df: pd.DataFrame,
    algorithms: tuple[str, ...],
    caption: str,
    label: str,
    problem_order: list[str] | tuple[str, ...] = ("zdt4", "zdt6"),
) -> str:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "c" * len(algorithms) + "}",
        "\\toprule",
        "Problem & Metric & " + " & ".join(algorithms) + " \\\\",
        "\\midrule",
    ]

    for problem_name in problem_order:
        problem_rows = summary_df[summary_df["problem"] == problem_name]
        for metric in ("hv", "igd"):
            means = []
            for algorithm in algorithms:
                algo_row = problem_rows[problem_rows["algorithm"] == algorithm].iloc[0]
                means.append(float(algo_row[(metric, "mean")]))

            best_value = max(means) if metric == "hv" else min(means)
            cells = []
            for algorithm in algorithms:
                algo_row = problem_rows[problem_rows["algorithm"] == algorithm].iloc[0]
                mean_value = float(algo_row[(metric, "mean")])
                std_value = float(algo_row[(metric, "std")])
                cell = f"{mean_value:.5f} $\\pm$ {std_value:.5f}"
                if np.isclose(mean_value, best_value):
                    cell = "\\textbf{" + cell + "}"
                cells.append(cell)
            lines.append(problem_name.upper() + f" & {metric.upper()} & " + " & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def plot_metric_boxplots(raw_df: pd.DataFrame, metric: str, algorithms: tuple[str, ...], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    palette = {
        "FITO": "#c0392b",
        "NSGA-II": "#1f77b4",
        "MOEA/D": "#2ca02c",
        "RVEA": "#9467bd",
        "FITO+BR": "#bcbd22",
        "FITO-noSR": "#ff7f0e",
        "FITO-noLS": "#8c564b",
    }

    for idx, problem_name in enumerate(PROBLEM_SETTINGS):
        ax = axes[idx]
        group = raw_df[raw_df["problem"] == problem_name]
        values = [group[group["algorithm"] == algo][metric].to_numpy() for algo in algorithms]
        bp = ax.boxplot(values, patch_artist=True, tick_labels=algorithms)
        for patch, algo in zip(bp["boxes"], algorithms):
            patch.set_facecolor(palette.get(algo, "#7f7f7f"))
            patch.set_alpha(0.65)
        ax.set_title(problem_name.upper())
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.2, axis="y")

    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_fronts(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    problem = get_problem("zdt6")
    true_pf = problem.pareto_front()
    ax.plot(true_pf[:, 0], true_pf[:, 1], color="black", linewidth=2, label="True PF")

    colors = {
        "FITO": "#c0392b",
        "NSGA-II": "#1f77b4",
        "MOEA/D": "#2ca02c",
        "RVEA": "#9467bd",
    }

    for algorithm in MAIN_ALGORITHMS:
        if algorithm == "FITO":
            front, _ = fito_optimize("zdt6", seed=0, use_stagnation_restart=True, use_leader_support=True)
        else:
            front, _ = run_baseline("zdt6", algorithm, seed=0)
        ax.scatter(front[:, 0], front[:, 1], s=18, alpha=0.75, color=colors[algorithm], label=algorithm)

    ax.set_xlabel("$f_1$")
    ax.set_ylabel("$f_2$")
    ax.set_title("Representative Pareto fronts on ZDT6")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_high_level_summary(
    main_raw: pd.DataFrame,
    rank_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    eval_budget_df: pd.DataFrame,
) -> dict[str, object]:
    hv_best = {}
    igd_best = {}
    for metric in ("hv", "igd"):
        grouped = main_raw.groupby(["problem", "algorithm"])[metric].mean().reset_index()
        for problem_name in PROBLEM_SETTINGS:
            group = grouped[grouped["problem"] == problem_name]
            if metric == "hv":
                winner = group.sort_values(metric, ascending=False).iloc[0]["algorithm"]
                hv_best[winner] = hv_best.get(winner, 0) + 1
            else:
                winner = group.sort_values(metric, ascending=True).iloc[0]["algorithm"]
                igd_best[winner] = igd_best.get(winner, 0) + 1

    mean_ranks = (
        rank_df.groupby(["metric", "algorithm"])["average_rank"].mean().reset_index().sort_values(["metric", "average_rank"])
    )

    significant_wins = stats_df[(stats_df["holm_p_value"] < 0.05) & (stats_df["fito_better"])]

    return {
        "seeds": list(SEEDS),
        "problems": list(PROBLEM_SETTINGS.keys()),
        "hv_best_problem_counts": hv_best,
        "igd_best_problem_counts": igd_best,
        "mean_ranks": mean_ranks.to_dict(orient="records"),
        "evaluation_budget": eval_budget_df.to_dict(orient="records"),
        "significant_fito_wins": significant_wins.to_dict(orient="records"),
    }


def save_text_summary(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        "Benchmark Summary",
        "=================",
        f"Seeds: {summary['seeds']}",
        f"Problems: {summary['problems']}",
        f"HV best counts: {summary['hv_best_problem_counts']}",
        f"IGD best counts: {summary['igd_best_problem_counts']}",
        "",
        "Average ranks:",
    ]
    for row in summary["mean_ranks"]:
        lines.append(
            f"  - {row['metric'].upper()} | {row['algorithm']}: {row['average_rank']:.3f}"
        )

    lines.append("")
    lines.append("Mean objective evaluations per run:")
    for row in sorted(summary["evaluation_budget"], key=lambda item: item["mean"]):
        lines.append(
            f"  - {row['algorithm']}: {row['mean']:.1f} +- {row['std']:.1f} "
            f"(min={int(row['min'])}, max={int(row['max'])})"
        )

    lines.append("")
    lines.append("Significant FITO wins (p < 0.05):")
    if summary["significant_fito_wins"]:
        for row in summary["significant_fito_wins"]:
            lines.append(
                f"  - {row['metric'].upper()} | {row['problem'].upper()} vs {row['baseline']} "
                f"(raw p={row['p_value']:.4f}, Holm p={row['holm_p_value']:.4f}, delta={row['cliffs_delta']:.3f})"
            )
    else:
        lines.append("  - None under the selected evaluation budget.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tasks = build_tasks()
    results = []
    max_workers = min(4, max(1, os.cpu_count() or 1))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())

    raw_df = pd.DataFrame(results)
    raw_df["problem"] = pd.Categorical(raw_df["problem"], categories=list(PROBLEM_SETTINGS.keys()), ordered=True)

    main_raw = raw_df[~raw_df["ablation"]].copy().sort_values(["problem", "algorithm", "seed"])
    ablation_raw = raw_df[raw_df["ablation"]].copy().sort_values(["problem", "algorithm", "seed"])

    main_summary, rank_df = summarize_metrics(main_raw)
    ablation_summary, _ = summarize_metrics(ablation_raw)
    eval_budget_df = summarize_eval_budget(main_raw)
    stats_df = pairwise_significance_summary(main_raw)
    high_level = build_high_level_summary(main_raw, rank_df, stats_df, eval_budget_df)

    main_raw.drop(columns=["front"]).to_csv(RESULTS_DIR / "main_raw_metrics.csv", index=False)
    ablation_raw.drop(columns=["front"]).to_csv(RESULTS_DIR / "ablation_raw_metrics.csv", index=False)
    main_summary.to_csv(RESULTS_DIR / "main_summary.csv", index=False)
    ablation_summary.to_csv(RESULTS_DIR / "ablation_summary.csv", index=False)
    eval_budget_df.to_csv(RESULTS_DIR / "eval_budget_summary.csv", index=False)
    rank_df.to_csv(RESULTS_DIR / "average_ranks.csv", index=False)
    stats_df.to_csv(RESULTS_DIR / "significance_tests.csv", index=False)
    (RESULTS_DIR / "high_level_summary.json").write_text(json.dumps(high_level, indent=2), encoding="utf-8")
    save_text_summary(high_level, RESULTS_DIR / "summary.txt")

    hv_table = table_to_latex(
        main_summary,
        metric="hv",
        algorithms=MAIN_ALGORITHMS,
        caption=f"Hypervolume results over {len(SEEDS)} independent runs.",
        label="tab:hv",
    )
    igd_table = table_to_latex(
        main_summary,
        metric="igd",
        algorithms=MAIN_ALGORITHMS,
        caption=f"IGD results over {len(SEEDS)} independent runs.",
        label="tab:igd",
    )
    ablation_table = static_ablation_table_to_latex(
        ablation_summary,
        algorithms=ABLATION_ALGORITHMS,
        caption="Static sanity check for selected FITO variants. FITO is the final dynamic default; FITO-noSR disables the stagnation restart for sensitivity analysis.",
        label="tab:ablation",
        problem_order=["zdt4", "zdt6"],
    )

    (RESULTS_DIR / "hv_table.tex").write_text(hv_table, encoding="utf-8")
    (RESULTS_DIR / "igd_table.tex").write_text(igd_table, encoding="utf-8")
    (RESULTS_DIR / "ablation_table.tex").write_text(ablation_table, encoding="utf-8")

    plot_metric_boxplots(main_raw, metric="hv", algorithms=MAIN_ALGORITHMS, output_path=RESULTS_DIR / "hv_boxplots.png")
    plot_metric_boxplots(main_raw, metric="igd", algorithms=MAIN_ALGORITHMS, output_path=RESULTS_DIR / "igd_boxplots.png")
    plot_pareto_fronts(RESULTS_DIR / "pareto_zdt6.png")


if __name__ == "__main__":
    main()
