from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from scipy.stats import mannwhitneyu

from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

try:
    from evaluation_counter import attach_evaluation_counter
except ModuleNotFoundError:
    from experiments.evaluation_counter import attach_evaluation_counter

try:
    from predictive_baselines import AuditedDNSGA2, AuditedKGB, AuditedNSGA2, PPS, activation_audit_summary
except ModuleNotFoundError:
    from experiments.predictive_baselines import AuditedDNSGA2, AuditedKGB, AuditedNSGA2, PPS, activation_audit_summary


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DYNAMIC_PROBLEMS = {
    "df1": {"pop_size": 100, "generations": 60, "taut": 5},
    "df3": {"pop_size": 100, "generations": 60, "taut": 5},
    "df4": {"pop_size": 100, "generations": 60, "taut": 5},
    "df5": {"pop_size": 100, "generations": 60, "taut": 5},
    "df7": {"pop_size": 100, "generations": 60, "taut": 5},
    "df8": {"pop_size": 100, "generations": 60, "taut": 5},
    "df9": {"pop_size": 100, "generations": 60, "taut": 5},
}

MAIN_ALGORITHMS = ("FITO", "DNSGA-II-A", "DNSGA-II-B", "KGB-DMOEA", "NSGA-II", "PPS-DMOEA")
SEEDS = tuple(range(20))
DYNAMIC_BASELINES = tuple(algo for algo in MAIN_ALGORITHMS if algo != "FITO")

LEADER_PAIRING_PROB = 0.80
LEADER_PULL = 0.10
DIFFERENTIAL_PUSH = 0.15
RESTART_RATE = 0.125
RESTART_SIGMA = 0.08
STAGNATION_LIMIT = 8
POST_CHANGE_RATE = 0.08
CHANGE_MEMORY_BLEND = 0.65
CHANGE_MEMORY_SHARE = 0.5
CHANGE_MEMORY_SIGMA_MULT = 0.35

FITO_DEFAULT_CONFIG = {
    "use_pitstop_restart": True,
    "use_leader_support": True,
    "use_boundary_risk": False,
    "use_redeployment": True,
    "use_predictive_anchors": True,
    "use_change_memory_blend": False,
}

ABLATION_VARIANTS = {
    "FITO": {
        **FITO_DEFAULT_CONFIG,
        "description": "Pruned FITO default after removing unsupported boundary-risk scaling and anchor-leader blending.",
    },
    "FITO-noLS": {
        **FITO_DEFAULT_CONFIG,
        "use_leader_support": False,
        "description": "Disable leader-support pairing and directional correction.",
    },
    "FITO+BR": {
        **FITO_DEFAULT_CONFIG,
        "use_boundary_risk": True,
        "description": "Re-enable boundary-risk-aware step scaling on top of the pruned FITO default.",
    },
    "FITO-noPS": {
        **FITO_DEFAULT_CONFIG,
        "use_pitstop_restart": False,
        "description": "Disable only the optional stagnation-triggered pit-stop branch; keep post-change redeployment enabled.",
    },
    "FITO-noRD": {
        **FITO_DEFAULT_CONFIG,
        "use_redeployment": False,
        "description": "Disable the post-change redeployment stage only.",
    },
    "FITO-noPA": {
        **FITO_DEFAULT_CONFIG,
        "use_predictive_anchors": False,
        "description": "Disable predictive anchor extrapolation inside redeployment.",
    },
    "FITO+MB": {
        **FITO_DEFAULT_CONFIG,
        "use_change_memory_blend": True,
        "description": "Re-enable anchor-leader blending on top of the pruned FITO default.",
    },
}
ABLATION_ALGORITHMS = tuple(ABLATION_VARIANTS.keys())


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


def environmental_selection(
    X: np.ndarray,
    F: np.ndarray,
    pop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def tournament_indices(
    ranks: np.ndarray,
    crowd: np.ndarray,
    rng: np.random.Generator,
    n_select: int,
) -> np.ndarray:
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


def predictive_leader_anchors(
    previous_env_leaders: np.ndarray | None,
    current_env_leaders: np.ndarray | None,
    xl: np.ndarray,
    xu: np.ndarray,
) -> np.ndarray | None:
    if previous_env_leaders is None or current_env_leaders is None:
        return None
    if len(previous_env_leaders) == 0 or len(current_env_leaders) == 0:
        return None

    shift = current_env_leaders.mean(axis=0) - previous_env_leaders.mean(axis=0)
    if np.linalg.norm(shift) <= 1e-12:
        return None
    return np.clip(current_env_leaders + shift, xl, xu)


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


def dynamic_problem(name: str):
    cfg = DYNAMIC_PROBLEMS[name]
    return get_problem(name, taut=cfg["taut"])


def evaluate_igd(problem, F: np.ndarray) -> float:
    return float(IGD(problem.pareto_front())(non_dominated_unique(F)))


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


def run_dynamic_fito(
    problem_name: str,
    seed: int,
    use_pitstop_restart: bool = True,
    use_leader_support: bool = True,
    use_boundary_risk: bool = True,
    use_redeployment: bool = True,
    use_predictive_anchors: bool = True,
    use_change_memory_blend: bool = True,
) -> tuple[float, float, list[float], int]:
    cfg = DYNAMIC_PROBLEMS[problem_name]
    problem = dynamic_problem(problem_name)
    eval_counter = attach_evaluation_counter(problem)
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    pop_size = cfg["pop_size"]
    generations = cfg["generations"]

    X = rng.uniform(xl, xu, size=(pop_size, problem.n_var))
    F = problem.evaluate(X)
    igd_curve = []
    best_score = adaptive_hv_score(F)
    stagnant = 0
    previous_env_leaders: np.ndarray | None = None

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

        igd_curve.append(evaluate_igd(problem, F))
        current_score = adaptive_hv_score(F)
        if current_score > best_score + 1e-7:
            best_score = current_score
            stagnant = 0
        else:
            stagnant += 1

        if use_pitstop_restart and stagnant >= STAGNATION_LIMIT:
            restart_idx = weakest_support_indices(order, ranks, crowd, elite_n, max(4, int(pop_size * RESTART_RATE)))
            for idx in restart_idx:
                if rng.random() < 0.6:
                    base = leaders[rng.integers(len(leaders))]
                    X[idx] = np.clip(base + rng.normal(0.0, RESTART_SIGMA, size=problem.n_var) * (xu - xl), xl, xu)
                else:
                    X[idx] = np.clip(xl + xu - X[idx], xl, xu)
            F[restart_idx] = problem.evaluate(X[restart_idx])
            stagnant = 0

        pre_change_leaders = leaders.copy()
        previous_time = problem.time
        problem.tic()
        F = problem.evaluate(X)

        if problem.time != previous_time:
            X, F, ranks, crowd = environmental_selection(X, F, pop_size)
            order = np.lexsort((-crowd, ranks))
            leaders = X[order[:elite_n]]
            if use_redeployment:
                anchor_pool = (
                    predictive_leader_anchors(previous_env_leaders, pre_change_leaders, xl, xu)
                    if use_predictive_anchors
                    else None
                )
                change_idx = weakest_support_indices(order, ranks, crowd, elite_n, max(4, int(pop_size * POST_CHANGE_RATE)))
                memory_cutoff = int(np.ceil(len(change_idx) * CHANGE_MEMORY_SHARE))
                for pos, idx in enumerate(change_idx):
                    use_memory = anchor_pool is not None and pos < memory_cutoff
                    if use_memory:
                        anchor = anchor_pool[pos % len(anchor_pool)]
                        if use_change_memory_blend:
                            leader = leaders[rng.integers(len(leaders))]
                            base = np.clip(CHANGE_MEMORY_BLEND * anchor + (1.0 - CHANGE_MEMORY_BLEND) * leader, xl, xu)
                        else:
                            base = anchor
                        sigma = RESTART_SIGMA * CHANGE_MEMORY_SIGMA_MULT
                    else:
                        base = leaders[rng.integers(len(leaders))]
                        sigma = RESTART_SIGMA * 0.6
                    X[idx] = np.clip(base + rng.normal(0.0, sigma, size=problem.n_var) * (xu - xl), xl, xu)
                F[change_idx] = problem.evaluate(X[change_idx])
            best_score = adaptive_hv_score(F)
            stagnant = 0
            previous_env_leaders = pre_change_leaders

    return float(np.mean(igd_curve)), float(np.mean(igd_curve[-10:])), [float(v) for v in igd_curve], int(eval_counter.count)


class DynamicTrace(Callback):
    def __init__(self):
        super().__init__()
        self.igd_curve: list[float] = []

    def update(self, algorithm):
        F = algorithm.pop.get("F")
        self.igd_curve.append(evaluate_igd(algorithm.problem, F))
        algorithm.problem.tic()


def make_dynamic_baseline(algorithm_name: str, pop_size: int):
    if algorithm_name == "DNSGA-II-A":
        return AuditedDNSGA2(pop_size=pop_size, version="A", audit_response_mode="dnsga2_a_internal_diversity_response")
    if algorithm_name == "DNSGA-II-B":
        return AuditedDNSGA2(pop_size=pop_size, version="B", audit_response_mode="dnsga2_b_internal_diversity_response")
    if algorithm_name == "KGB-DMOEA":
        return AuditedKGB(pop_size=pop_size)
    if algorithm_name == "NSGA-II":
        return AuditedNSGA2(pop_size=pop_size)
    if algorithm_name == "PPS-DMOEA":
        return PPS(pop_size=pop_size)
    raise ValueError(f"Unknown dynamic baseline: {algorithm_name}")


def run_dynamic_baseline(problem_name: str, algorithm_name: str, seed: int) -> tuple[float, float, list[float], int, dict[str, object]]:
    cfg = DYNAMIC_PROBLEMS[problem_name]
    problem = dynamic_problem(problem_name)
    eval_counter = attach_evaluation_counter(problem)
    callback = DynamicTrace()
    algorithm = make_dynamic_baseline(algorithm_name, cfg["pop_size"])
    res = minimize(problem, algorithm, ("n_gen", cfg["generations"]), seed=seed, callback=callback, verbose=False)
    audited_algorithm = getattr(res, "algorithm", algorithm)
    curve = callback.igd_curve
    return float(np.mean(curve)), float(np.mean(curve[-10:])), curve, int(eval_counter.count), activation_audit_summary(audited_algorithm)


def run_task(task: dict[str, object]) -> dict[str, object]:
    problem_name = str(task["problem"])
    algorithm_name = str(task["algorithm"])
    seed = int(task["seed"])
    ablation = bool(task.get("ablation", False))

    start = time.perf_counter()
    if algorithm_name in ABLATION_VARIANTS:
        migd, tail_igd, curve, n_evals = run_dynamic_fito(
            problem_name,
            seed,
            **{k: v for k, v in ABLATION_VARIANTS[algorithm_name].items() if k != "description"},
        )
        audit = {
            "environment_change_count": 0,
            "change_response_count": 0,
            "prediction_count": 0,
            "replaced_count": 0,
            "kde_success_count": 0,
            "kde_fallback_count": 0,
            "response_activation_rate": 0.0,
            "response_audit_mode": "legacy_fito_tuple_not_expanded",
            "activation_events_json": "[]",
        }
    else:
        migd, tail_igd, curve, n_evals, audit = run_dynamic_baseline(problem_name, algorithm_name, seed)

    result = {
        "problem": problem_name,
        "algorithm": algorithm_name,
        "seed": seed,
        "ablation": ablation,
        "migd": migd,
        "tail_igd": tail_igd,
        "n_evals": n_evals,
        "runtime_sec": time.perf_counter() - start,
        "curve": curve,
    }
    result.update(audit)
    return result


def build_tasks() -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for algorithm in MAIN_ALGORITHMS:
        for problem_name in DYNAMIC_PROBLEMS:
            for seed in SEEDS:
                tasks.append({"algorithm": algorithm, "problem": problem_name, "seed": seed, "ablation": False})

    for algorithm in ABLATION_ALGORITHMS:
        for problem_name in DYNAMIC_PROBLEMS:
            for seed in SEEDS:
                tasks.append({"algorithm": algorithm, "problem": problem_name, "seed": seed, "ablation": True})
    return tasks


def summarize(raw_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return raw_df.groupby(["problem", "algorithm"])[metric].agg(["mean", "std"]).reset_index()


def summarize_eval_budget(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_cols = [
        "n_evals",
        "environment_change_count",
        "change_response_count",
        "prediction_count",
        "replaced_count",
        "kde_success_count",
        "kde_fallback_count",
        "response_evaluation_count",
    ]
    audit_cols = [col for col in audit_cols if col in raw_df.columns]
    by_problem = raw_df.groupby(["problem", "algorithm"])[audit_cols].agg(["mean", "std"]).reset_index()
    overall = raw_df.groupby("algorithm")[audit_cols].agg(["mean", "std", "min", "max"]).reset_index()
    return by_problem, overall


def average_ranks(raw_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    records = []
    for problem_name, group in raw_df.groupby("problem"):
        pivot = group.pivot(index="seed", columns="algorithm", values=metric)
        rank_matrix = pivot.rank(axis=1, ascending=True, method="average")
        avg_rank = rank_matrix.mean(axis=0)
        for algorithm_name, rank_value in avg_rank.items():
            records.append(
                {
                    "problem": problem_name,
                    "metric": metric,
                    "algorithm": algorithm_name,
                    "average_rank": float(rank_value),
                }
            )
    return pd.DataFrame(records)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted_value = (m - rank) * p_values[idx]
        running = max(running, adjusted_value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def dynamic_pairwise_tests(main_raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    raw_p_values = []
    for problem_name in DYNAMIC_PROBLEMS:
        fito_values = (
            main_raw[(main_raw["problem"] == problem_name) & (main_raw["algorithm"] == "FITO")]
            .sort_values("seed")["migd"]
            .to_numpy()
        )
        for baseline in DYNAMIC_BASELINES:
            baseline_values = (
                main_raw[(main_raw["problem"] == problem_name) & (main_raw["algorithm"] == baseline)]
                .sort_values("seed")["migd"]
                .to_numpy()
            )
            stat, p_value = mannwhitneyu(fito_values, baseline_values, alternative="two-sided", method="asymptotic")
            records.append(
                {
                    "problem": problem_name,
                    "baseline": baseline,
                    "fito_mean": float(np.mean(fito_values)),
                    "baseline_mean": float(np.mean(baseline_values)),
                    "mann_whitney_u": float(stat),
                    "p_value": float(p_value),
                    "fito_better": float(np.mean(fito_values)) < float(np.mean(baseline_values)),
                }
            )
            raw_p_values.append(float(p_value))

    adjusted = holm_adjust(raw_p_values)
    family_size = len(raw_p_values)
    holm_scope = f"global_dynamic_family_{family_size}_comparisons"
    for row, corrected in zip(records, adjusted):
        row["holm_p_value"] = corrected
        row["holm_family_size"] = family_size
        row["holm_scope"] = holm_scope

    return pd.DataFrame(records)


def dynamic_table_latex(summary_df: pd.DataFrame, algorithms: tuple[str, ...], caption: str, label: str) -> str:
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

    for problem_name in DYNAMIC_PROBLEMS:
        rows = summary_df[summary_df["problem"] == problem_name]
        best_value = rows["mean"].min()
        cells = []
        for algorithm in algorithms:
            row = rows[rows["algorithm"] == algorithm].iloc[0]
            cell = f"{row['mean']:.4f} $\\pm$ {row['std']:.4f}"
            if np.isclose(row["mean"], best_value):
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(problem_name.upper() + " & " + " & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def plot_dynamic_trace(main_raw: pd.DataFrame, problem_name: str, output_path: Path) -> None:
    colors = {
        "FITO": "#c0392b",
        "DNSGA-II-A": "#1f77b4",
        "DNSGA-II-B": "#17becf",
        "KGB-DMOEA": "#2ca02c",
        "NSGA-II": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    subset = main_raw[main_raw["problem"] == problem_name]
    for algorithm in MAIN_ALGORITHMS:
        curves = subset[subset["algorithm"] == algorithm]["curve"].tolist()
        curve_matrix = np.asarray(curves, dtype=float)
        mean_curve = curve_matrix.mean(axis=0)
        std_curve = curve_matrix.std(axis=0)
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, color=colors[algorithm], linewidth=2, label=algorithm)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=colors[algorithm], alpha=0.15)

    ax.set_xlabel("Generation")
    ax.set_ylabel("IGD")
    ax.set_title(f"Dynamic tracking on {problem_name.upper()}")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_dynamic_boxplots(main_raw: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "FITO": "#c0392b",
        "DNSGA-II-A": "#1f77b4",
        "DNSGA-II-B": "#17becf",
        "KGB-DMOEA": "#2ca02c",
        "NSGA-II": "#7f7f7f",
    }
    n_plots = len(DYNAMIC_PROBLEMS)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, problem_name in zip(axes, DYNAMIC_PROBLEMS):
        subset = main_raw[main_raw["problem"] == problem_name]
        values = [subset[subset["algorithm"] == algo]["migd"].to_numpy() for algo in MAIN_ALGORITHMS]
        bp = ax.boxplot(values, patch_artist=True, tick_labels=MAIN_ALGORITHMS)
        for patch, algo in zip(bp["boxes"], MAIN_ALGORITHMS):
            patch.set_facecolor(colors[algo])
            patch.set_alpha(0.65)
        ax.set_title(problem_name.upper())
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("MIGD")
        ax.grid(alpha=0.2, axis="y")

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_summary_text(
    migd_summary: pd.DataFrame,
    rank_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    eval_overall_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "Dynamic Benchmark Summary",
        "========================",
        f"Problems: {list(DYNAMIC_PROBLEMS.keys())}",
        f"Seeds: {list(SEEDS)}",
        "",
        "MIGD means:",
    ]
    for problem_name in DYNAMIC_PROBLEMS:
        group = migd_summary[migd_summary["problem"] == problem_name].sort_values("mean")
        lines.append("  - " + problem_name.upper() + ":")
        for _, row in group.iterrows():
            lines.append(f"    {row['algorithm']}: {row['mean']:.4f} +- {row['std']:.4f}")

    lines.append("")
    lines.append("Average ranks:")
    rank_mean = rank_df.groupby("algorithm")["average_rank"].mean().sort_values()
    for algorithm_name, rank_value in rank_mean.items():
        lines.append(f"  - {algorithm_name}: {rank_value:.3f}")

    lines.append("")
    lines.append("Pruned FITO default configuration:")
    lines.append(
        "  - use_boundary_risk="
        f"{FITO_DEFAULT_CONFIG['use_boundary_risk']}, "
        f"use_predictive_anchors={FITO_DEFAULT_CONFIG['use_predictive_anchors']}, "
        f"use_change_memory_blend={FITO_DEFAULT_CONFIG['use_change_memory_blend']}"
    )

    lines.append("")
    lines.append("Dynamic pit-stop reachability under current protocol:")
    reachability = {problem: STAGNATION_LIMIT <= DYNAMIC_PROBLEMS[problem]["taut"] for problem in DYNAMIC_PROBLEMS}
    for problem_name, reachable in reachability.items():
        status = "reachable before change reset" if reachable else "not reachable before change reset"
        lines.append(
            f"  - {problem_name.upper()}: {status} "
            f"(taut={DYNAMIC_PROBLEMS[problem_name]['taut']}, stagnation_limit={STAGNATION_LIMIT})"
        )
    if not any(reachability.values()):
        lines.append(
            "  - Under the current accelerated protocol, the stagnation-triggered pit-stop branch is analytically "
            "inactive in the dynamic benchmark."
        )

    lines.append("")
    lines.append("Mean objective evaluations per run:")
    for _, row in eval_overall_df.sort_values("mean").iterrows():
        lines.append(
            f"  - {row['algorithm']}: {row['mean']:.1f} +- {row['std']:.1f} "
            f"(min={int(row['min'])}, max={int(row['max'])})"
        )

    lines.append("")
    family_size = int(stats_df["holm_family_size"].iloc[0]) if not stats_df.empty else 0
    env_changes = {problem: DYNAMIC_PROBLEMS[problem]["generations"] // DYNAMIC_PROBLEMS[problem]["taut"] for problem in DYNAMIC_PROBLEMS}
    lines.append(f"Nominal environment changes per run: {env_changes}")
    lines.append("")
    lines.append(f"Significant FITO wins after Holm correction across all {family_size} dynamic pairwise comparisons (p < 0.05):")
    wins = stats_df[(stats_df["holm_p_value"] < 0.05) & (stats_df["fito_better"])]
    if wins.empty:
        lines.append("  - None")
    else:
        for _, row in wins.iterrows():
            lines.append(
                f"  - {row['problem'].upper()} vs {row['baseline']} "
                f"(raw p={row['p_value']:.4f}, Holm p={row['holm_p_value']:.4f})"
            )

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
    raw_df["problem"] = pd.Categorical(raw_df["problem"], categories=list(DYNAMIC_PROBLEMS.keys()), ordered=True)

    main_raw = raw_df[~raw_df["ablation"]].copy().sort_values(["problem", "algorithm", "seed"])
    ablation_raw = raw_df[raw_df["ablation"]].copy().sort_values(["problem", "algorithm", "seed"])

    migd_summary = summarize(main_raw, "migd")
    tail_summary = summarize(main_raw, "tail_igd")
    ablation_summary = summarize(ablation_raw, "migd")
    eval_by_problem_df, eval_overall_df = summarize_eval_budget(main_raw)
    rank_df = average_ranks(main_raw, "migd")
    stats_df = dynamic_pairwise_tests(main_raw)

    main_raw.to_json(RESULTS_DIR / "dynamic_raw_metrics.json", orient="records", indent=2)
    main_raw.drop(columns=["curve"]).to_csv(RESULTS_DIR / "dynamic_raw_metrics.csv", index=False)
    ablation_raw.drop(columns=["curve"]).to_csv(RESULTS_DIR / "dynamic_ablation_raw_metrics.csv", index=False)
    migd_summary.to_csv(RESULTS_DIR / "dynamic_migd_summary.csv", index=False)
    tail_summary.to_csv(RESULTS_DIR / "dynamic_tail_summary.csv", index=False)
    ablation_summary.to_csv(RESULTS_DIR / "dynamic_ablation_summary.csv", index=False)
    eval_by_problem_df.to_csv(RESULTS_DIR / "dynamic_eval_budget_summary.csv", index=False)
    eval_overall_df.to_csv(RESULTS_DIR / "dynamic_eval_budget_overall.csv", index=False)
    rank_df.to_csv(RESULTS_DIR / "dynamic_average_ranks.csv", index=False)
    stats_df.to_csv(RESULTS_DIR / "dynamic_stats.csv", index=False)

    (RESULTS_DIR / "dynamic_migd_table.tex").write_text(
        dynamic_table_latex(
            migd_summary,
            MAIN_ALGORITHMS,
            f"Dynamic benchmark results measured by MIGD over {len(SEEDS)} independent runs.",
            "tab:dynamic-migd",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "dynamic_tail_table.tex").write_text(
        dynamic_table_latex(
            tail_summary,
            MAIN_ALGORITHMS,
            "Last-10-generation IGD values on dynamic benchmarks.",
            "tab:dynamic-tail",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "dynamic_ablation_table.tex").write_text(
        dynamic_table_latex(
            ablation_summary,
            ABLATION_ALGORITHMS,
            "One-at-a-time dynamic component ablation results for FITO variants measured by MIGD.",
            "tab:dynamic-ablation",
        ),
        encoding="utf-8",
    )

    plot_dynamic_trace(main_raw, "df5", RESULTS_DIR / "dynamic_trace_df5.png")
    plot_dynamic_boxplots(main_raw, RESULTS_DIR / "dynamic_migd_boxplots.png")
    save_summary_text(migd_summary, rank_df, stats_df, eval_overall_df, RESULTS_DIR / "dynamic_summary.txt")
    (RESULTS_DIR / "dynamic_config.json").write_text(
        json.dumps(
            {
                "problems": DYNAMIC_PROBLEMS,
                "algorithms": MAIN_ALGORITHMS,
                "fito_default_config": FITO_DEFAULT_CONFIG,
                "ablation_algorithms": ABLATION_ALGORITHMS,
                "ablation_variants": ABLATION_VARIANTS,
                "seeds": list(SEEDS),
                "params": {
                    "leader_pairing_prob": LEADER_PAIRING_PROB,
                    "leader_pull": LEADER_PULL,
                    "differential_push": DIFFERENTIAL_PUSH,
                    "restart_rate": RESTART_RATE,
                    "restart_sigma": RESTART_SIGMA,
                    "stagnation_limit": STAGNATION_LIMIT,
                    "post_change_rate": POST_CHANGE_RATE,
                    "change_memory_blend": CHANGE_MEMORY_BLEND,
                    "change_memory_share": CHANGE_MEMORY_SHARE,
                    "change_memory_sigma_mult": CHANGE_MEMORY_SIGMA_MULT,
                },
                "statistics": {
                    "test": "Mann-Whitney U",
                    "test_design": "independent two-sided comparison across independent runs",
                    "metric": "migd",
                    "holm_family_size": len(DYNAMIC_PROBLEMS) * len(DYNAMIC_BASELINES),
                    "holm_scope": "global across all dynamic problem/baseline FITO comparisons",
                },
                "budgeting": {
                    "population_size_matched": True,
                    "nominal_generations_matched": True,
                    "objective_evaluations_matched": False,
                    "objective_evaluation_summary_file": "dynamic_eval_budget_overall.csv",
                },
                "dynamic_protocol": {
                    "taut_per_problem": {name: cfg["taut"] for name, cfg in DYNAMIC_PROBLEMS.items()},
                    "nominal_environment_changes_per_run": {
                        name: cfg["generations"] // cfg["taut"] for name, cfg in DYNAMIC_PROBLEMS.items()
                    },
                    "pitstop_activation_audit": {
                        "stagnation_limit": STAGNATION_LIMIT,
                        "reachable_before_change_reset": {
                            name: STAGNATION_LIMIT <= cfg["taut"] for name, cfg in DYNAMIC_PROBLEMS.items()
                        },
                        "note": (
                            "Under the current taut schedule, the dynamic stagnation-triggered pit-stop branch "
                            "cannot activate before each environment change resets the stagnation counter."
                        ),
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
