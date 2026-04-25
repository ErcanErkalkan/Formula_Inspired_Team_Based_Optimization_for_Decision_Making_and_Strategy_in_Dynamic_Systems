from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems import get_problem

try:
    from evaluation_counter import attach_evaluation_counter
except ModuleNotFoundError:
    from experiments.evaluation_counter import attach_evaluation_counter

try:
    from run_dynamic_benchmarks import (
        CHANGE_MEMORY_BLEND,
        CHANGE_MEMORY_SHARE,
        CHANGE_MEMORY_SIGMA_MULT,
        DIFFERENTIAL_PUSH,
        LEADER_PAIRING_PROB,
        LEADER_PULL,
        POST_CHANGE_RATE,
        RESTART_RATE,
        RESTART_SIGMA,
        boundary_risk_scale,
        environmental_selection,
        non_dominated_unique,
        predictive_leader_anchors,
        polynomial_mutation,
        sbx,
        tournament_indices,
        weakest_support_indices,
    )
except ModuleNotFoundError:
    from experiments.run_dynamic_benchmarks import (
        CHANGE_MEMORY_BLEND,
        CHANGE_MEMORY_SHARE,
        CHANGE_MEMORY_SIGMA_MULT,
        DIFFERENTIAL_PUSH,
        LEADER_PAIRING_PROB,
        LEADER_PULL,
        POST_CHANGE_RATE,
        RESTART_RATE,
        RESTART_SIGMA,
        boundary_risk_scale,
        environmental_selection,
        non_dominated_unique,
        predictive_leader_anchors,
        polynomial_mutation,
        sbx,
        tournament_indices,
        weakest_support_indices,
    )

try:
    from stats_utils import holm_adjust, mann_whitney_summary, mean_ci95
except ModuleNotFoundError:
    from experiments.stats_utils import holm_adjust, mann_whitney_summary, mean_ci95


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROBLEMS = tuple(f"df{i}" for i in range(1, 10))
PROTOCOLS = {
    "fast_t5_n10": {"taut": 5, "nt": 10, "generations": 60, "pop_size": 100},
    "moderate_t10_n10": {"taut": 10, "nt": 10, "generations": 60, "pop_size": 100},
    "severe_t10_n20": {"taut": 10, "nt": 20, "generations": 60, "pop_size": 100},
}
MAIN_ALGORITHMS = ("FITO", "DNSGA-II-A", "DNSGA-II-B", "KGB-DMOEA", "MDDM-DMOEA", "NSGA-II", "PPS-DMOEA")
BASELINES = tuple(algo for algo in MAIN_ALGORITHMS if algo != "FITO")
SEEDS = tuple(range(20))
STAGNATION_LIMIT = 8
FIXED_BUDGET_TARGET = 8000

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
        "description": "Final FITO default with pit-stop restart, leader-support search, redeployment, and predictive anchors.",
    },
    "FITO-noLS": {
        **FITO_DEFAULT_CONFIG,
        "use_leader_support": False,
        "description": "Disable leader-support search.",
    },
    "FITO-noRD": {
        **FITO_DEFAULT_CONFIG,
        "use_redeployment": False,
        "description": "Disable post-change redeployment.",
    },
    "FITO-noPA": {
        **FITO_DEFAULT_CONFIG,
        "use_predictive_anchors": False,
        "description": "Disable predictive anchor extrapolation.",
    },
    "FITO-noPS": {
        **FITO_DEFAULT_CONFIG,
        "use_pitstop_restart": False,
        "description": "Disable the stagnation-triggered pit-stop restart branch while keeping the remaining final FITO mechanisms active.",
    },
    "FITO+BR": {
        **FITO_DEFAULT_CONFIG,
        "use_boundary_risk": True,
        "description": "Re-enable rejected boundary-risk scaling.",
    },
    "FITO+MB": {
        **FITO_DEFAULT_CONFIG,
        "use_change_memory_blend": True,
        "description": "Re-enable rejected anchor-leader blending.",
    },
}


def dynamic_problem(name: str, protocol_name: str):
    cfg = PROTOCOLS[protocol_name]
    return get_problem(name, taut=cfg["taut"], nt=cfg["nt"])


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
    from pymoo.indicators.hv import HV

    return float(HV(ref_point=ref_point)(normalized))


def run_dynamic_fito(
    protocol_name: str,
    problem_name: str,
    seed: int,
    pop_size: int,
    use_pitstop_restart: bool = False,
    use_leader_support: bool = True,
    use_boundary_risk: bool = False,
    use_redeployment: bool = True,
    use_predictive_anchors: bool = True,
    use_change_memory_blend: bool = False,
) -> dict[str, object]:
    cfg = PROTOCOLS[protocol_name]
    problem = dynamic_problem(problem_name, protocol_name)
    eval_counter = attach_evaluation_counter(problem)
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    generations = int(cfg["generations"])

    X = rng.uniform(xl, xu, size=(pop_size, problem.n_var))
    F = problem.evaluate(X)
    igd_curve: list[float] = []
    best_score = adaptive_hv_score(F)
    stagnant = 0
    previous_env_leaders: np.ndarray | None = None
    pitstop_count = 0
    redeployment_count = 0
    environment_change_count = 0
    total_replaced_count = 0
    activation_events: list[dict[str, object]] = []

    for generation in range(generations):
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
            pitstop_count += 1

        pre_change_leaders = leaders.copy()
        previous_time = problem.time
        problem.tic()
        F = problem.evaluate(X)

        if problem.time != previous_time:
            environment_change_count += 1
            X, F, ranks, crowd = environmental_selection(X, F, pop_size)
            order = np.lexsort((-crowd, ranks))
            leaders = X[order[:elite_n]]
            change_idx = np.array([], dtype=int)
            if use_redeployment:
                anchor_pool = (
                    predictive_leader_anchors(previous_env_leaders, pre_change_leaders, xl, xu)
                    if use_predictive_anchors
                    else None
                )
                change_idx = weakest_support_indices(order, ranks, crowd, elite_n, max(4, int(pop_size * POST_CHANGE_RATE)))
                total_replaced_count += int(len(change_idx))
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
                redeployment_count += 1
            activation_events.append(
                {
                    "event_index": environment_change_count,
                    "generation": generation,
                    "previous_time": float(previous_time),
                    "current_time": float(problem.time),
                    "response_activated": int(bool(use_redeployment)),
                    "response_type": "fito_redeployment_predictive_anchor" if use_redeployment and use_predictive_anchors else ("fito_redeployment_leader_jitter" if use_redeployment else "none"),
                    "replaced_count": int(len(change_idx)),
                    "prediction_used": int(bool(use_redeployment and use_predictive_anchors)),
                    "kde_success": 0,
                    "kde_fallback": 0,
                    "note": "FITO post-change redeployment activation audit." if use_redeployment else "FITO redeployment branch disabled for this ablation.",
                }
            )
            best_score = adaptive_hv_score(F)
            stagnant = 0
            previous_env_leaders = pre_change_leaders

    return {
        "migd": float(np.mean(igd_curve)),
        "tail_igd": float(np.mean(igd_curve[-10:])),
        "curve": [float(v) for v in igd_curve],
        "n_evals": int(eval_counter.count),
        "pitstop_count": pitstop_count,
        "redeployment_count": redeployment_count,
        "environment_change_count": environment_change_count,
        "change_response_count": redeployment_count,
        "prediction_count": redeployment_count if use_predictive_anchors else 0,
        "replaced_count": total_replaced_count,
        "kde_success_count": 0,
        "kde_fallback_count": 0,
        "response_evaluation_count": 0,
        "response_activation_rate": float(redeployment_count / environment_change_count) if environment_change_count else 0.0,
        "response_audit_mode": "fito_manual_redeployment",
        "activation_events_json": json.dumps(activation_events, default=str, ensure_ascii=False),
    }


class DynamicTrace(Callback):
    def __init__(self):
        super().__init__()
        self.igd_curve: list[float] = []

    def update(self, algorithm):
        F = algorithm.pop.get("F")
        self.igd_curve.append(evaluate_igd(algorithm.problem, F))
        algorithm.problem.tic()


try:
    from predictive_baselines import AuditedDNSGA2, AuditedKGB, AuditedNSGA2, MDDM, PPS, activation_audit_summary
except ModuleNotFoundError:
    from experiments.predictive_baselines import AuditedDNSGA2, AuditedKGB, AuditedNSGA2, MDDM, PPS, activation_audit_summary

def make_dynamic_baseline(name: str, pop_size: int):
    if name == "DNSGA-II-A":
        return AuditedDNSGA2(pop_size=pop_size, version="A", audit_response_mode="dnsga2_a_internal_diversity_response")
    if name == "DNSGA-II-B":
        return AuditedDNSGA2(pop_size=pop_size, version="B", audit_response_mode="dnsga2_b_internal_diversity_response")
    if name == "KGB-DMOEA":
        return AuditedKGB(pop_size=pop_size)
    if name == "MDDM-DMOEA":
        return MDDM(pop_size=pop_size)
    if name == "NSGA-II":
        return AuditedNSGA2(pop_size=pop_size)
    if name == "PPS-DMOEA":
        return PPS(pop_size=pop_size)
    raise ValueError(f"Unknown dynamic baseline: {name}")


def run_dynamic_baseline(protocol_name: str, problem_name: str, algorithm_name: str, seed: int, pop_size: int) -> dict[str, object]:
    cfg = PROTOCOLS[protocol_name]
    problem = dynamic_problem(problem_name, protocol_name)
    eval_counter = attach_evaluation_counter(problem)
    callback = DynamicTrace()
    algorithm = make_dynamic_baseline(algorithm_name, pop_size)
    res = minimize(problem, algorithm, ("n_gen", cfg["generations"]), seed=seed, callback=callback, verbose=False)
    audited_algorithm = getattr(res, "algorithm", algorithm)
    curve = callback.igd_curve
    result = {
        "migd": float(np.mean(curve)),
        "tail_igd": float(np.mean(curve[-10:])),
        "curve": [float(v) for v in curve],
        "n_evals": int(eval_counter.count),
        "pitstop_count": 0,
        "redeployment_count": 0,
    }
    result.update(activation_audit_summary(audited_algorithm))
    return result


def calibrate_budget_pop_sizes() -> dict[str, dict[str, int]]:
    calibration: dict[str, dict[str, int]] = {}
    for protocol_name, cfg in PROTOCOLS.items():
        calibration[protocol_name] = {}
        for algorithm_name in MAIN_ALGORITHMS:
            if algorithm_name == "FITO":
                result = run_dynamic_fito(protocol_name, "df1", seed=0, pop_size=cfg["pop_size"], **FITO_DEFAULT_CONFIG)
            else:
                result = run_dynamic_baseline(protocol_name, "df1", algorithm_name, seed=0, pop_size=cfg["pop_size"])
            evals = max(1, int(result["n_evals"]))
            estimated = int(round(cfg["pop_size"] * FIXED_BUDGET_TARGET / evals))
            calibration[protocol_name][algorithm_name] = max(40, estimated)
    return calibration


def build_tasks(fixed_budget_pop_sizes: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []

    for protocol_name in PROTOCOLS:
        cfg = PROTOCOLS[protocol_name]
        for problem_name in PROBLEMS:
            for algorithm_name in MAIN_ALGORITHMS:
                for seed in SEEDS:
                    tasks.append(
                        {
                            "family": "main",
                            "protocol": protocol_name,
                            "problem": problem_name,
                            "algorithm": algorithm_name,
                            "seed": seed,
                            "pop_size": cfg["pop_size"],
                        }
                    )

    for protocol_name in PROTOCOLS:
        cfg = PROTOCOLS[protocol_name]
        for problem_name in PROBLEMS:
            for algorithm_name in ABLATION_VARIANTS:
                for seed in SEEDS:
                    tasks.append(
                        {
                            "family": "ablation",
                            "protocol": protocol_name,
                            "problem": problem_name,
                            "algorithm": algorithm_name,
                            "seed": seed,
                            "pop_size": cfg["pop_size"],
                        }
                    )

    for protocol_name in PROTOCOLS:
        for problem_name in PROBLEMS:
            for algorithm_name in MAIN_ALGORITHMS:
                for seed in SEEDS:
                    tasks.append(
                        {
                            "family": "budget",
                            "protocol": protocol_name,
                            "problem": problem_name,
                            "algorithm": algorithm_name,
                            "seed": seed,
                            "pop_size": fixed_budget_pop_sizes[protocol_name][algorithm_name],
                        }
                    )
    return tasks


def run_task(task: dict[str, object]) -> dict[str, object]:
    protocol_name = str(task["protocol"])
    problem_name = str(task["problem"])
    algorithm_name = str(task["algorithm"])
    seed = int(task["seed"])
    family = str(task["family"])
    pop_size = int(task["pop_size"])

    start = time.perf_counter()
    if algorithm_name in ABLATION_VARIANTS:
        result = run_dynamic_fito(
            protocol_name,
            problem_name,
            seed,
            pop_size=pop_size,
            **{k: v for k, v in ABLATION_VARIANTS[algorithm_name].items() if k != "description"},
        )
    elif algorithm_name == "FITO":
        result = run_dynamic_fito(protocol_name, problem_name, seed, pop_size=pop_size, **FITO_DEFAULT_CONFIG)
    else:
        result = run_dynamic_baseline(protocol_name, problem_name, algorithm_name, seed, pop_size=pop_size)

    result.update(
        {
            "family": family,
            "protocol": protocol_name,
            "problem": problem_name,
            "algorithm": algorithm_name,
            "seed": seed,
            "pop_size": pop_size,
            "runtime_sec": time.perf_counter() - start,
        }
    )
    return result


def summarize_metric(raw_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = raw_df.groupby(["protocol", "problem", "algorithm"])[metric].agg(["mean", "std"]).reset_index()
    ci_records = []
    for (protocol_name, problem_name, algorithm_name), group in raw_df.groupby(["protocol", "problem", "algorithm"]):
        ci_low, ci_high = mean_ci95(group[metric].to_numpy())
        ci_records.append(
            {
                "protocol": protocol_name,
                "problem": problem_name,
                "algorithm": algorithm_name,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return summary.merge(pd.DataFrame(ci_records), on=["protocol", "problem", "algorithm"])


def summarize_eval_budget(raw_df: pd.DataFrame) -> pd.DataFrame:
    audit_columns = [
        "n_evals",
        "pop_size",
        "runtime_sec",
        "pitstop_count",
        "redeployment_count",
        "change_response_count",
        "prediction_count",
        "replaced_count",
        "kde_success_count",
        "kde_fallback_count",
        "response_evaluation_count",
    ]
    audit_columns = [column for column in audit_columns if column in raw_df.columns]
    return (
        raw_df.groupby(["family", "protocol", "algorithm"])[audit_columns]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )


def average_ranks(raw_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (protocol_name, problem_name), group in raw_df.groupby(["protocol", "problem"]):
        pivot = group.pivot(index="seed", columns="algorithm", values=metric)
        rank_matrix = pivot.rank(axis=1, ascending=True, method="average")
        avg_rank = rank_matrix.mean(axis=0)
        for algorithm_name, value in avg_rank.items():
            rows.append(
                {
                    "protocol": protocol_name,
                    "problem": problem_name,
                    "metric": metric,
                    "algorithm": algorithm_name,
                    "average_rank": float(value),
                }
            )
    return pd.DataFrame(rows)


def pairwise_tests(raw_df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    records = []
    raw_p_values = []
    for (protocol_name, problem_name), group in raw_df.groupby(["protocol", "problem"]):
        fito_values = group[group["algorithm"] == "FITO"].sort_values("seed")["migd"].to_numpy()
        for baseline in BASELINES:
            baseline_values = group[group["algorithm"] == baseline].sort_values("seed")["migd"].to_numpy()
            stats = mann_whitney_summary(fito_values, baseline_values, higher_is_better=False)
            records.append(
                {
                    "family": family_name,
                    "protocol": protocol_name,
                    "problem": problem_name,
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
    for row, corrected in zip(records, adjusted):
        row["holm_p_value"] = corrected
        row["holm_family_size"] = family_size
        row["holm_scope"] = f"{family_name}_{family_size}_comparisons"
    return pd.DataFrame(records)


def problem_table_latex(summary_df: pd.DataFrame, protocol_name: str, algorithms: tuple[str, ...], caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(algorithms) + "}",
        "\\toprule",
        "Problem & " + " & ".join(algorithms) + " \\\\",
        "\\midrule",
    ]
    rows = summary_df[summary_df["protocol"] == protocol_name]
    for problem_name in PROBLEMS:
        problem_rows = rows[rows["problem"] == problem_name]
        best_value = problem_rows["mean"].min()
        cells = []
        for algorithm in algorithms:
            row = problem_rows[problem_rows["algorithm"] == algorithm].iloc[0]
            cell = f"{row['mean']:.4f} $\\pm$ {row['std']:.4f}"
            if np.isclose(float(row["mean"]), float(best_value)):
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(problem_name.upper() + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def rank_table_latex(rank_df: pd.DataFrame, algorithms: tuple[str, ...], caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(algorithms) + "}",
        "\\toprule",
        "Protocol & " + " & ".join(algorithms) + " \\\\",
        "\\midrule",
    ]
    for protocol_name in PROTOCOLS:
        rows = rank_df[rank_df["protocol"] == protocol_name]
        mean_ranks = rows.groupby("algorithm")["average_rank"].mean()
        best_rank = float(mean_ranks.min())
        cells = []
        for algorithm in algorithms:
            value = float(mean_ranks.loc[algorithm])
            cell = f"{value:.3f}"
            if np.isclose(value, best_rank):
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(protocol_name.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")

    overall = rank_df.groupby("algorithm")["average_rank"].mean()
    best_overall = float(overall.min())
    cells = []
    for algorithm in algorithms:
        value = float(overall.loc[algorithm])
        cell = f"{value:.3f}"
        if np.isclose(value, best_overall):
            cell = "\\textbf{" + cell + "}"
        cells.append(cell)
    lines.extend(["\\midrule", "Overall & " + " & ".join(cells) + " \\\\", "\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"

def eval_table_latex(eval_df: pd.DataFrame, family_name: str, caption: str, label: str) -> str:
    family_rows = eval_df[eval_df[("family", "")] == family_name]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Protocol & Algorithm & Mean evals & Mean pop & Mean runtime (s) \\\\",
        "\\midrule",
    ]
    for protocol_name in PROTOCOLS:
        rows = family_rows[family_rows[("protocol", "")] == protocol_name]
        for algorithm in MAIN_ALGORITHMS:
            row = rows[rows[("algorithm", "")] == algorithm].iloc[0]
            lines.append(
                f"{protocol_name.replace('_', '\\_')} & {algorithm} & "
                f"{row[('n_evals', 'mean')]:.1f} & {row[('pop_size', 'mean')]:.1f} & {row[('runtime_sec', 'mean')]:.3f} \\\\"
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.extend(["\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def write_summary(
    main_rank_df: pd.DataFrame,
    budget_rank_df: pd.DataFrame,
    main_stats_df: pd.DataFrame,
    budget_stats_df: pd.DataFrame,
    ablation_eval_df: pd.DataFrame,
    fixed_budget_pop_sizes: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    lines = [
        "ASOC Dynamic Benchmark Suite",
        "==========================",
        f"Problems: {list(PROBLEMS)}",
        f"Protocols: {list(PROTOCOLS.keys())}",
        f"Seeds: {list(SEEDS)}",
        "",
        "Final FITO default:",
        "  - leader-support enabled",
        "  - post-change redeployment enabled",
        "  - predictive anchors enabled",
        "  - stagnation-triggered pit-stop restart enabled",
        "  - boundary-risk scaling excluded",
        "  - anchor-leader blending excluded",
        "",
        "Overall average ranks (primary fixed-budget suite; approximately equal objective-evaluation budget):",
    ]
    for algorithm, value in budget_rank_df.groupby("algorithm")["average_rank"].mean().sort_values().items():
        lines.append(f"  - {algorithm}: {value:.3f}")

    lines.extend(["", f"Fixed-budget target: {FIXED_BUDGET_TARGET} objective evaluations (approximate, protocol-specific population-size calibration)."])
    for protocol_name, pop_map in fixed_budget_pop_sizes.items():
        lines.append(f"  - {protocol_name}: {pop_map}")

    lines.extend(["", "Overall average ranks (auxiliary generation-matched diagnostic suite; not the primary fairness claim):"])
    for algorithm, value in main_rank_df.groupby("algorithm")["average_rank"].mean().sort_values().items():
        lines.append(f"  - {algorithm}: {value:.3f}")

    lines.extend(["", "Holm-corrected significant FITO wins in the auxiliary generation-matched diagnostic suite:"])
    wins = main_stats_df[(main_stats_df["holm_p_value"] < 0.05) & (main_stats_df["fito_better"])]
    if wins.empty:
        lines.append("  - None.")
    else:
        for _, row in wins.sort_values(["protocol", "problem", "baseline"]).iterrows():
            lines.append(
                f"  - {row['protocol']} | {row['problem'].upper()} vs {row['baseline']} "
                f"(raw p={row['p_value']:.4g}, Holm p={row['holm_p_value']:.4g}, delta={row['cliffs_delta']:.3f})"
            )

    lines.extend(["", "Holm-corrected significant FITO wins in the primary fixed-budget suite:"])
    wins = budget_stats_df[(budget_stats_df["holm_p_value"] < 0.05) & (budget_stats_df["fito_better"])]
    if wins.empty:
        lines.append("  - None.")
    else:
        for _, row in wins.sort_values(["protocol", "problem", "baseline"]).iterrows():
            lines.append(
                f"  - {row['protocol']} | {row['problem'].upper()} vs {row['baseline']} "
                f"(raw p={row['p_value']:.4g}, Holm p={row['holm_p_value']:.4g}, delta={row['cliffs_delta']:.3f})"
            )

    lines.extend(["", "Pit-stop activation audit (ablation family):"])
    pit_rows = ablation_eval_df[ablation_eval_df[("algorithm", "")] == "FITO"]
    for protocol_name in PROTOCOLS:
        row = pit_rows[pit_rows[("protocol", "")] == protocol_name].iloc[0]
        lines.append(
            f"  - {protocol_name}: mean pit-stop activations={row[('pitstop_count', 'mean')]:.2f}, "
            f"mean redeployments={row[('redeployment_count', 'mean')]:.2f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def activation_event_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Expand per-run JSON activation logs into one auditable event-level CSV."""
    records: list[dict[str, object]] = []
    for row in rows:
        raw_events = row.get("activation_events_json", "[]")
        try:
            events = json.loads(raw_events) if isinstance(raw_events, str) else list(raw_events or [])
        except Exception:
            events = []
        for event in events:
            if not isinstance(event, dict):
                continue
            records.append(
                {
                    "family": row.get("family"),
                    "protocol": row.get("protocol"),
                    "problem": row.get("problem"),
                    "algorithm": row.get("algorithm"),
                    "seed": row.get("seed"),
                    "pop_size": row.get("pop_size"),
                    **event,
                }
            )
    return pd.DataFrame(records)


def summarize_activation_audit(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate change-response activation counters for all baselines and FITO variants.

    The returned frame intentionally uses flat column names so that the CSV is
    directly readable by spreadsheet software and validation scripts.
    """
    audit_cols = [
        "environment_change_count",
        "change_response_count",
        "prediction_count",
        "replaced_count",
        "kde_success_count",
        "kde_fallback_count",
        "response_evaluation_count",
        "response_activation_rate",
    ]
    available = [col for col in audit_cols if col in raw_df.columns]
    grouped = raw_df.groupby(["family", "protocol", "algorithm"], dropna=False)[available].agg(["mean", "std", "min", "max"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    return grouped

def main() -> None:
    fixed_budget_pop_sizes = calibrate_budget_pop_sizes()
    tasks = build_tasks(fixed_budget_pop_sizes)

    results = []
    max_workers = min(4, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())

    raw_df = pd.DataFrame(results)
    raw_df["protocol"] = pd.Categorical(raw_df["protocol"], categories=list(PROTOCOLS.keys()), ordered=True)
    raw_df["problem"] = pd.Categorical(raw_df["problem"], categories=list(PROBLEMS), ordered=True)

    activation_events = activation_event_frame(results)
    activation_summary = summarize_activation_audit(raw_df)

    main_raw = raw_df[raw_df["family"] == "main"].copy().sort_values(["protocol", "problem", "algorithm", "seed"])
    ablation_raw = raw_df[raw_df["family"] == "ablation"].copy().sort_values(["protocol", "problem", "algorithm", "seed"])
    budget_raw = raw_df[raw_df["family"] == "budget"].copy().sort_values(["protocol", "problem", "algorithm", "seed"])

    main_summary = summarize_metric(main_raw, "migd")
    ablation_summary = summarize_metric(ablation_raw, "migd")
    budget_summary = summarize_metric(budget_raw, "migd")

    main_ranks = average_ranks(main_raw, "migd")
    ablation_ranks = average_ranks(ablation_raw, "migd")
    budget_ranks = average_ranks(budget_raw, "migd")

    main_stats = pairwise_tests(main_raw, family_name="dynamic_main")
    budget_stats = pairwise_tests(budget_raw, family_name="dynamic_budget")
    eval_budget = summarize_eval_budget(raw_df)

    raw_df.drop(columns=["curve"]).to_csv(RESULTS_DIR / "asoc_dynamic_raw_metrics.csv", index=False)
    main_summary.to_csv(RESULTS_DIR / "asoc_dynamic_main_summary.csv", index=False)
    ablation_summary.to_csv(RESULTS_DIR / "asoc_dynamic_ablation_summary.csv", index=False)
    budget_summary.to_csv(RESULTS_DIR / "asoc_dynamic_budget_summary.csv", index=False)
    main_ranks.to_csv(RESULTS_DIR / "asoc_dynamic_main_ranks.csv", index=False)
    ablation_ranks.to_csv(RESULTS_DIR / "asoc_dynamic_ablation_ranks.csv", index=False)
    budget_ranks.to_csv(RESULTS_DIR / "asoc_dynamic_budget_ranks.csv", index=False)
    main_stats.to_csv(RESULTS_DIR / "asoc_dynamic_main_stats.csv", index=False)
    budget_stats.to_csv(RESULTS_DIR / "asoc_dynamic_budget_stats.csv", index=False)
    eval_budget.to_csv(RESULTS_DIR / "asoc_dynamic_eval_budget.csv", index=False)
    activation_summary.to_csv(RESULTS_DIR / "asoc_dynamic_activation_audit_summary.csv", index=False)
    activation_events.to_csv(RESULTS_DIR / "asoc_dynamic_activation_audit_events.csv", index=False)
    (RESULTS_DIR / "asoc_dynamic_budget_calibration.json").write_text(json.dumps(fixed_budget_pop_sizes, indent=2), encoding="utf-8")

    for protocol_name in PROTOCOLS:
        table = problem_table_latex(
            main_summary,
            protocol_name,
            MAIN_ALGORITHMS,
            caption=f"MIGD results on the {protocol_name.replace('_', ' ')} dynamic protocol over {len(SEEDS)} independent runs.",
            label=f"tab:dynamic-{protocol_name}",
        )
        (RESULTS_DIR / f"asoc_dynamic_{protocol_name}_table.tex").write_text(table, encoding="utf-8")

    rank_table = rank_table_latex(
        main_ranks,
        MAIN_ALGORITHMS,
        caption="Auxiliary generation-matched average MIGD ranks across dynamic protocols. Lower is better.",
        label="tab:dynamic-protocol-ranks",
    )
    (RESULTS_DIR / "asoc_dynamic_protocol_ranks.tex").write_text(rank_table, encoding="utf-8")

    ablation_rank_table = rank_table_latex(
        ablation_ranks,
        tuple(ABLATION_VARIANTS.keys()),
        caption="Average MIGD ranks for FITO component ablations across dynamic protocols. Lower is better.",
        label="tab:dynamic-ablation-ranks",
    )
    (RESULTS_DIR / "asoc_dynamic_ablation_ranks.tex").write_text(ablation_rank_table, encoding="utf-8")

    budget_rank_table = rank_table_latex(
        budget_ranks,
        MAIN_ALGORITHMS,
        caption="Primary fixed-budget average MIGD ranks across dynamic protocols. Lower is better.",
        label="tab:dynamic-budget-ranks",
    )
    (RESULTS_DIR / "asoc_dynamic_budget_ranks.tex").write_text(budget_rank_table, encoding="utf-8")

    main_eval_table = eval_table_latex(
        eval_budget,
        family_name="main",
        caption="Auxiliary generation-matched evaluation and runtime diagnostic. Different internal response mechanisms lead to unequal objective-evaluation counts; this table is therefore not used as the primary fairness claim.",
        label="tab:dynamic-eval-budget",
    )
    (RESULTS_DIR / "asoc_dynamic_eval_budget_table.tex").write_text(main_eval_table, encoding="utf-8")

    budget_eval_table = eval_table_latex(
        eval_budget,
        family_name="budget",
        caption="Primary fixed-budget calibration summary. Population sizes are protocol-specific and calibrated to approximately 8000 objective evaluations.",
        label="tab:dynamic-fixed-budget",
    )
    (RESULTS_DIR / "asoc_dynamic_fixed_budget_table.tex").write_text(budget_eval_table, encoding="utf-8")

    write_summary(main_ranks, budget_ranks, main_stats, budget_stats, eval_budget, fixed_budget_pop_sizes, RESULTS_DIR / "asoc_dynamic_summary.txt")


if __name__ == "__main__":
    main()
