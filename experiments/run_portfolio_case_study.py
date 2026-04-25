from __future__ import annotations

import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.problems.dyn import DynamicTestProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import mannwhitneyu

try:
    from run_dynamic_benchmarks import (
        CHANGE_MEMORY_BLEND,
        CHANGE_MEMORY_SHARE,
        CHANGE_MEMORY_SIGMA_MULT,
        DIFFERENTIAL_PUSH,
        FITO_DEFAULT_CONFIG,
        LEADER_PAIRING_PROB,
        LEADER_PULL,
        POST_CHANGE_RATE,
        RESTART_RATE,
        RESTART_SIGMA,
        STAGNATION_LIMIT,
        boundary_risk_scale,
        environmental_selection,
        holm_adjust,
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
        FITO_DEFAULT_CONFIG,
        LEADER_PAIRING_PROB,
        LEADER_PULL,
        POST_CHANGE_RATE,
        RESTART_RATE,
        RESTART_SIGMA,
        STAGNATION_LIMIT,
        boundary_risk_scale,
        environmental_selection,
        holm_adjust,
        non_dominated_unique,
        predictive_leader_anchors,
        polynomial_mutation,
        sbx,
        tournament_indices,
        weakest_support_indices,
    )

try:
    from evaluation_counter import attach_evaluation_counter
except ModuleNotFoundError:
    from experiments.evaluation_counter import attach_evaluation_counter


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "legacy_portfolio_case"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRICE_DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-02-07/big_tech_stock_prices.csv"
COMPANY_DATA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-02-07/big_tech_companies.csv"
PRICE_DATA_PATH = DATA_DIR / "big_tech_stock_prices.csv"
COMPANY_DATA_PATH = DATA_DIR / "big_tech_companies.csv"
ENV_CACHE_PATH = DATA_DIR / "big_tech_walkforward_envs.pkl"

MAIN_ALGORITHMS = ("FITO", "DNSGA-II-A", "DNSGA-II-B", "KGB-DMOEA", "NSGA-II")
SEEDS = tuple(range(20))
ASSET_UNIVERSE = (
    "AAPL",
    "ADBE",
    "AMZN",
    "CRM",
    "CSCO",
    "GOOGL",
    "IBM",
    "INTC",
    "META",
    "MSFT",
    "NFLX",
    "NVDA",
    "ORCL",
    "TSLA",
)
TRAIN_WINDOW_DAYS = 252
HOLD_WINDOW_DAYS = 63
STEP_SIZE = HOLD_WINDOW_DAYS
GENERATIONS_PER_ENV = 10
FRONTIER_POINTS = 60
POP_SIZE = 100
TRANSACTION_COST_RATE = 0.001


def ensure_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PRICE_DATA_PATH.exists():
        prices = pd.read_csv(PRICE_DATA_URL)
        prices.to_csv(PRICE_DATA_PATH, index=False)
    if not COMPANY_DATA_PATH.exists():
        companies = pd.read_csv(COMPANY_DATA_URL)
        companies.to_csv(COMPANY_DATA_PATH, index=False)
    return pd.read_csv(PRICE_DATA_PATH), pd.read_csv(COMPANY_DATA_PATH)


def project_weights(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    clipped = np.clip(arr, 0.0, None)
    sums = clipped.sum(axis=1, keepdims=True)
    sums = np.where(sums <= 1e-12, 1.0, sums)
    projected = clipped / sums
    return projected


def portfolio_objectives(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    weights = project_weights(weights)
    risk = np.einsum("ij,jk,ik->i", weights, cov, weights)
    neg_return = -(weights @ mu)
    return np.column_stack((risk, neg_return))


def frontier_with_weights(weights: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    front_weights = weights[front]
    front_points = F[front]
    rounded = np.round(front_points, decimals=10)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    idx = np.sort(idx)
    front_weights = front_weights[idx]
    front_points = front_points[idx]
    order = np.argsort(front_points[:, 0])
    return front_weights[order], front_points[order]


def solve_long_only_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = FRONTIER_POINTS) -> tuple[np.ndarray, np.ndarray]:
    n_assets = len(mu)
    bounds = [(0.0, 1.0)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)

    def variance_objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def solve(constraints: list[dict[str, object]], guess: np.ndarray) -> np.ndarray:
        result = scipy_minimize(
            variance_objective,
            guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500, "disp": False},
        )
        if not result.success:
            result = scipy_minimize(
                variance_objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500, "disp": False},
            )
        weights = result.x if result.success else guess
        return project_weights(weights)[0]

    sum_constraint = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    min_var = solve([sum_constraint], x0)
    target_min = float(min(mu.min(), np.dot(min_var, mu)))
    target_max = float(mu.max())
    targets = np.linspace(target_min, target_max, n_points)

    frontier_weights = [min_var, np.eye(n_assets)[int(np.argmax(mu))]]
    frontier_points = [portfolio_objectives(min_var, mu, cov)[0], portfolio_objectives(frontier_weights[-1], mu, cov)[0]]

    guess = min_var
    for target in targets:
        constraints = [
            sum_constraint,
            {"type": "ineq", "fun": lambda w, t=float(target): float(np.dot(w, mu) - t)},
        ]
        weights = solve(constraints, guess)
        frontier_weights.append(weights)
        frontier_points.append(portfolio_objectives(weights, mu, cov)[0])
        guess = weights

    weights = np.asarray(frontier_weights, dtype=float)
    points = np.asarray(frontier_points, dtype=float)
    return frontier_with_weights(weights, points)


def build_environments() -> list[dict[str, object]]:
    if ENV_CACHE_PATH.exists():
        with ENV_CACHE_PATH.open("rb") as handle:
            return pickle.load(handle)

    prices_raw, companies = ensure_data()
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    selected = prices_raw[prices_raw["stock_symbol"].isin(ASSET_UNIVERSE)].copy()
    wide_prices = (
        selected.pivot(index="date", columns="stock_symbol", values="adj_close")
        .sort_index()
        .dropna()
    )
    returns = wide_prices.pct_change().dropna()
    asset_names = [symbol for symbol in ASSET_UNIVERSE if symbol in wide_prices.columns]
    company_lookup = dict(zip(companies["stock_symbol"], companies["company"], strict=False))

    envs: list[dict[str, object]] = []
    for start_idx in range(TRAIN_WINDOW_DAYS, len(returns) - HOLD_WINDOW_DAYS + 1, STEP_SIZE):
        train = returns.iloc[start_idx - TRAIN_WINDOW_DAYS : start_idx]
        hold = returns.iloc[start_idx : start_idx + HOLD_WINDOW_DAYS]

        train_mu = train.mean().to_numpy(dtype=float)
        train_cov = train.cov().to_numpy(dtype=float) + np.eye(train.shape[1]) * 1e-10
        hold_mu = hold.mean().to_numpy(dtype=float)
        hold_cov = hold.cov().to_numpy(dtype=float) + np.eye(hold.shape[1]) * 1e-10

        _, train_pf = solve_long_only_frontier(train_mu, train_cov)
        _, hold_pf = solve_long_only_frontier(hold_mu, hold_cov)

        envs.append(
            {
                "asset_names": asset_names,
                "asset_labels": [company_lookup.get(symbol, symbol) for symbol in asset_names],
                "train_mu": train_mu,
                "train_cov": train_cov,
                "train_pf": train_pf,
                "hold_mu": hold_mu,
                "hold_cov": hold_cov,
                "hold_pf": hold_pf,
                "hold_returns": hold.to_numpy(dtype=float),
                "train_start": str(train.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "hold_start": str(hold.index[0].date()),
                "hold_end": str(hold.index[-1].date()),
            }
        )

    with ENV_CACHE_PATH.open("wb") as handle:
        pickle.dump(envs, handle)
    return envs


class WalkForwardPortfolioProblem(DynamicTestProblem):
    def __init__(self, environments: list[dict[str, object]], taut: int):
        self.environments = environments
        self.asset_names = tuple(environments[0]["asset_names"])
        super().__init__(
            nt=len(environments),
            taut=taut,
            tau=0,
            n_var=len(self.asset_names),
            n_obj=2,
            xl=np.zeros(len(self.asset_names)),
            xu=np.ones(len(self.asset_names)),
        )

    def env_index(self) -> int:
        return min(len(self.environments) - 1, int(self.time * self.nt))

    def current_env(self) -> dict[str, object]:
        return self.environments[self.env_index()]

    def project_weights(self, x: np.ndarray) -> np.ndarray:
        return project_weights(x)

    def _evaluate(self, x, out, *args, **kwargs):
        env = self.current_env()
        out["F"] = portfolio_objectives(
            self.project_weights(np.asarray(x, dtype=float)),
            np.asarray(env["train_mu"], dtype=float),
            np.asarray(env["train_cov"], dtype=float),
        )

    def _calc_pareto_front(self, *args, **kwargs):
        return np.asarray(self.current_env()["train_pf"], dtype=float)


def normalized_hv(pf: np.ndarray, approx: np.ndarray) -> float:
    nd_front = non_dominated_unique(approx)
    ideal = pf.min(axis=0)
    nadir = pf.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    normalized = np.clip((nd_front - ideal) / scale, 0.0, 1.0)
    return float(HV(ref_point=np.array([1.1, 1.1]))(normalized))


def adaptive_hv_score(F: np.ndarray) -> float:
    nd_front = non_dominated_unique(F)
    if len(nd_front) == 0:
        return float("-inf")
    ideal = F.min(axis=0)
    nadir = F.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    normalized = np.clip((nd_front - ideal) / scale, 0.0, 1.0)
    return float(HV(ref_point=np.array([1.1, 1.1]))(normalized))


def evaluate_igd(pf: np.ndarray, approx: np.ndarray) -> float:
    return float(IGD(pf)(non_dominated_unique(approx)))


def turnover(prev_weights: np.ndarray, current_weights: np.ndarray) -> float:
    return 0.5 * float(np.abs(current_weights - prev_weights).sum())


def select_closest_to_utopia_portfolio(
    X: np.ndarray, F: np.ndarray, prev_weights: np.ndarray
) -> np.ndarray:
    """Pick the non-dominated portfolio closest to the normalized ideal point."""
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    front_weights = project_weights(X[front])
    front_points = F[front]
    ideal = front_points.min(axis=0)
    nadir = front_points.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    normalized = np.clip((front_points - ideal) / scale, 0.0, 1.0)
    distance = np.linalg.norm(normalized, axis=1)
    turns = np.asarray([turnover(prev_weights, weights) for weights in front_weights], dtype=float)
    order = np.lexsort((turns, distance))
    return front_weights[order[0]]


def deployment_metrics(env: dict[str, object], weights: np.ndarray, prev_weights: np.ndarray) -> dict[str, object]:
    hold_returns = np.asarray(env["hold_returns"], dtype=float)
    daily_returns = hold_returns @ weights
    cost = TRANSACTION_COST_RATE * turnover(prev_weights, weights)
    net_daily_returns = daily_returns.copy()
    if len(net_daily_returns):
        net_daily_returns[0] = (1.0 + net_daily_returns[0]) * (1.0 - cost) - 1.0

    gross_growth = float(np.prod(1.0 + daily_returns))
    net_growth = float(np.prod(1.0 + net_daily_returns))
    realized = portfolio_objectives(weights, np.asarray(env["hold_mu"], dtype=float), np.asarray(env["hold_cov"], dtype=float))[0]
    return {
        "gross_return": gross_growth - 1.0,
        "net_return": net_growth - 1.0,
        "net_growth": net_growth,
        "turnover": turnover(prev_weights, weights),
        "transaction_cost": cost,
        "realized_risk": float(realized[0]),
        "realized_return": float(-realized[1]),
        "net_daily_returns": net_daily_returns.tolist(),
    }


def annualized_return(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return 0.0
    growth = float(np.prod(1.0 + daily_returns))
    return growth ** (252.0 / len(daily_returns)) - 1.0


def annualized_volatility(daily_returns: np.ndarray) -> float:
    if len(daily_returns) <= 1:
        return 0.0
    return float(np.std(daily_returns, ddof=1) * np.sqrt(252.0))


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    volatility = annualized_volatility(daily_returns)
    if volatility <= 1e-12:
        return 0.0
    return float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252.0))


def max_drawdown(wealth_path: np.ndarray) -> float:
    if len(wealth_path) == 0:
        return 0.0
    running_max = np.maximum.accumulate(wealth_path)
    drawdowns = 1.0 - wealth_path / np.maximum(running_max, 1e-12)
    return float(np.max(drawdowns))


def finalize_run(
    oos_hv_curve: list[float],
    oos_igd_curve: list[float],
    wealth_path: list[float],
    net_return_curve: list[float],
    turnover_curve: list[float],
    realized_risk_curve: list[float],
    realized_return_curve: list[float],
    daily_returns: list[float],
    selected_weights: list[list[float]],
    decision_dates: list[str],
    final_front: np.ndarray,
    final_pf: np.ndarray,
    asset_names: tuple[str, ...],
) -> dict[str, object]:
    daily_array = np.asarray(daily_returns, dtype=float)
    wealth_array = np.asarray(wealth_path, dtype=float)
    return {
        "migd": float(np.mean(oos_igd_curve)),
        "mhv": float(np.mean(oos_hv_curve)),
        "final_wealth": float(wealth_array[-1]),
        "mean_net_return": float(np.mean(net_return_curve)),
        "mean_turnover": float(np.mean(turnover_curve)),
        "mean_realized_risk": float(np.mean(realized_risk_curve)),
        "mean_realized_return": float(np.mean(realized_return_curve)),
        "annualized_return": annualized_return(daily_array),
        "annualized_volatility": annualized_volatility(daily_array),
        "annualized_sharpe": annualized_sharpe(daily_array),
        "max_drawdown": max_drawdown(wealth_array),
        "oos_hv_curve": oos_hv_curve,
        "oos_igd_curve": oos_igd_curve,
        "wealth_path": wealth_path,
        "net_return_curve": net_return_curve,
        "turnover_curve": turnover_curve,
        "selected_weights": selected_weights,
        "decision_dates": decision_dates,
        "final_front": non_dominated_unique(final_front).tolist(),
        "final_pf": np.asarray(final_pf, dtype=float).tolist(),
        "asset_names": list(asset_names),
    }


def run_fito(seed: int) -> dict[str, object]:
    environments = build_environments()
    problem = WalkForwardPortfolioProblem(environments, taut=GENERATIONS_PER_ENV)
    eval_counter = attach_evaluation_counter(problem)
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    pop_size = POP_SIZE
    total_generations = len(environments) * GENERATIONS_PER_ENV

    X = rng.uniform(xl, xu, size=(pop_size, problem.n_var))
    F = problem.evaluate(X)
    best_score = adaptive_hv_score(F)
    stagnant = 0

    prev_weights = np.full(problem.n_var, 1.0 / problem.n_var)
    wealth = 1.0
    oos_hv_curve: list[float] = []
    oos_igd_curve: list[float] = []
    wealth_path: list[float] = []
    net_return_curve: list[float] = []
    turnover_curve: list[float] = []
    realized_risk_curve: list[float] = []
    realized_return_curve: list[float] = []
    daily_returns: list[float] = []
    selected_weights: list[list[float]] = []
    decision_dates: list[str] = []
    final_front = np.empty((0, 2))
    final_pf = np.empty((0, 2))
    previous_env_leaders: np.ndarray | None = None
    use_boundary_risk = bool(FITO_DEFAULT_CONFIG["use_boundary_risk"])
    use_predictive_anchors = bool(FITO_DEFAULT_CONFIG["use_predictive_anchors"])
    use_change_memory_blend = bool(FITO_DEFAULT_CONFIG["use_change_memory_blend"])

    for generation in range(total_generations):
        X, F, ranks, crowd = environmental_selection(X, F, pop_size)
        order = np.lexsort((-crowd, ranks))
        elite_n = max(6, pop_size // 10)
        leaders = X[order[:elite_n]]
        guides = X[order[: max(elite_n * 3, elite_n + 1)]]

        offspring = []
        while len(offspring) < pop_size:
            if rng.random() < LEADER_PAIRING_PROB:
                p1 = leaders[rng.integers(len(leaders))]
                p2 = guides[rng.integers(len(guides))]
            else:
                idx = tournament_indices(ranks, crowd, rng, 2)
                p1 = X[idx[0]]
                p2 = X[idx[1]]

            c1, c2 = sbx(p1, p2, xl, xu, eta=25.0, prob=0.95, rng=rng)
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

        if stagnant >= STAGNATION_LIMIT:
            restart_idx = weakest_support_indices(order, ranks, crowd, elite_n, max(4, int(pop_size * RESTART_RATE)))
            for idx in restart_idx:
                if rng.random() < 0.6:
                    base = leaders[rng.integers(len(leaders))]
                    X[idx] = np.clip(base + rng.normal(0.0, RESTART_SIGMA, size=problem.n_var) * (xu - xl), xl, xu)
                else:
                    X[idx] = np.clip(xl + xu - X[idx], xl, xu)
            F[restart_idx] = problem.evaluate(X[restart_idx])
            stagnant = 0

        if (generation + 1) % GENERATIONS_PER_ENV == 0:
            env = problem.current_env()
            hold_F = portfolio_objectives(X, np.asarray(env["hold_mu"], dtype=float), np.asarray(env["hold_cov"], dtype=float))
            oos_hv_curve.append(normalized_hv(np.asarray(env["hold_pf"], dtype=float), hold_F))
            oos_igd_curve.append(evaluate_igd(np.asarray(env["hold_pf"], dtype=float), hold_F))

            weights = select_closest_to_utopia_portfolio(X, F, prev_weights)
            metrics = deployment_metrics(env, weights, prev_weights)
            wealth *= metrics["net_growth"]
            wealth_path.append(wealth)
            net_return_curve.append(metrics["net_return"])
            turnover_curve.append(metrics["turnover"])
            realized_risk_curve.append(metrics["realized_risk"])
            realized_return_curve.append(metrics["realized_return"])
            daily_returns.extend(metrics["net_daily_returns"])
            selected_weights.append(weights.tolist())
            decision_dates.append(str(env["hold_start"]))
            prev_weights = weights
            final_front = hold_F
            final_pf = np.asarray(env["hold_pf"], dtype=float)

        pre_change_leaders = leaders.copy()
        previous_time = problem.time
        problem.tic()
        F = problem.evaluate(X)

        if problem.time != previous_time:
            X, F, ranks, crowd = environmental_selection(X, F, pop_size)
            order = np.lexsort((-crowd, ranks))
            leaders = X[order[:elite_n]]
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

    result = finalize_run(
        oos_hv_curve,
        oos_igd_curve,
        wealth_path,
        net_return_curve,
        turnover_curve,
        realized_risk_curve,
        realized_return_curve,
        daily_returns,
        selected_weights,
        decision_dates,
        final_front,
        final_pf,
        problem.asset_names,
    )
    result["n_evals"] = int(eval_counter.count)
    return result


class WalkForwardCallback(Callback):
    def __init__(self):
        super().__init__()
        self.generation = 0
        self.prev_weights: np.ndarray | None = None
        self.wealth = 1.0
        self.oos_hv_curve: list[float] = []
        self.oos_igd_curve: list[float] = []
        self.wealth_path: list[float] = []
        self.net_return_curve: list[float] = []
        self.turnover_curve: list[float] = []
        self.realized_risk_curve: list[float] = []
        self.realized_return_curve: list[float] = []
        self.daily_returns: list[float] = []
        self.selected_weights: list[list[float]] = []
        self.decision_dates: list[str] = []
        self.final_front = np.empty((0, 2))
        self.final_pf = np.empty((0, 2))

    def update(self, algorithm):
        problem: WalkForwardPortfolioProblem = algorithm.problem
        X = project_weights(np.asarray(algorithm.pop.get("X"), dtype=float))
        F = np.asarray(algorithm.pop.get("F"), dtype=float)

        if self.prev_weights is None:
            self.prev_weights = np.full(problem.n_var, 1.0 / problem.n_var)

        if (self.generation + 1) % GENERATIONS_PER_ENV == 0:
            env = problem.current_env()
            hold_F = portfolio_objectives(X, np.asarray(env["hold_mu"], dtype=float), np.asarray(env["hold_cov"], dtype=float))
            self.oos_hv_curve.append(normalized_hv(np.asarray(env["hold_pf"], dtype=float), hold_F))
            self.oos_igd_curve.append(evaluate_igd(np.asarray(env["hold_pf"], dtype=float), hold_F))

            weights = select_closest_to_utopia_portfolio(X, F, self.prev_weights)
            metrics = deployment_metrics(env, weights, self.prev_weights)
            self.wealth *= metrics["net_growth"]
            self.wealth_path.append(self.wealth)
            self.net_return_curve.append(metrics["net_return"])
            self.turnover_curve.append(metrics["turnover"])
            self.realized_risk_curve.append(metrics["realized_risk"])
            self.realized_return_curve.append(metrics["realized_return"])
            self.daily_returns.extend(metrics["net_daily_returns"])
            self.selected_weights.append(weights.tolist())
            self.decision_dates.append(str(env["hold_start"]))
            self.prev_weights = weights
            self.final_front = hold_F
            self.final_pf = np.asarray(env["hold_pf"], dtype=float)

        self.generation += 1
        algorithm.problem.tic()


def make_baseline(name: str):
    if name == "DNSGA-II-A":
        return DNSGA2(pop_size=POP_SIZE, version="A")
    if name == "DNSGA-II-B":
        return DNSGA2(pop_size=POP_SIZE, version="B")
    if name == "KGB-DMOEA":
        return KGB(pop_size=POP_SIZE)
    if name == "NSGA-II":
        return NSGA2(pop_size=POP_SIZE)
    raise ValueError(f"Unknown baseline: {name}")


def run_baseline(name: str, seed: int) -> dict[str, object]:
    environments = build_environments()
    problem = WalkForwardPortfolioProblem(environments, taut=GENERATIONS_PER_ENV)
    eval_counter = attach_evaluation_counter(problem)
    callback = WalkForwardCallback()
    minimize(
        problem,
        make_baseline(name),
        ("n_gen", len(environments) * GENERATIONS_PER_ENV),
        seed=seed,
        callback=callback,
        verbose=False,
    )
    result = finalize_run(
        callback.oos_hv_curve,
        callback.oos_igd_curve,
        callback.wealth_path,
        callback.net_return_curve,
        callback.turnover_curve,
        callback.realized_risk_curve,
        callback.realized_return_curve,
        callback.daily_returns,
        callback.selected_weights,
        callback.decision_dates,
        callback.final_front,
        callback.final_pf,
        problem.asset_names,
    )
    result["n_evals"] = int(eval_counter.count)
    return result


def run_task(task: dict[str, object]) -> dict[str, object]:
    algorithm_name = str(task["algorithm"])
    seed = int(task["seed"])
    start = time.perf_counter()
    if algorithm_name == "FITO":
        result = run_fito(seed)
    else:
        result = run_baseline(algorithm_name, seed)
    result["algorithm"] = algorithm_name
    result["seed"] = seed
    result["runtime_sec"] = time.perf_counter() - start
    return result


def summarize(raw_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    higher_is_better = metric in {"mhv", "final_wealth", "mean_net_return", "annualized_return", "annualized_sharpe", "mean_realized_return"}
    return raw_df.groupby("algorithm")[metric].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=not higher_is_better)


def summarize_deployment(raw_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["final_wealth", "annualized_return", "annualized_volatility", "mean_turnover", "max_drawdown"]
    summary = raw_df.groupby("algorithm")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def summarize_eval_budget(raw_df: pd.DataFrame) -> pd.DataFrame:
    return raw_df.groupby("algorithm")["n_evals"].agg(["mean", "std", "min", "max"]).reset_index()


def independent_test_against_fito(raw_df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    fito_values = raw_df[raw_df["algorithm"] == "FITO"].sort_values("seed")[metric].to_numpy()
    rows = []
    p_values = []
    baselines = [algo for algo in MAIN_ALGORITHMS if algo != "FITO"]
    for baseline in baselines:
        baseline_values = raw_df[raw_df["algorithm"] == baseline].sort_values("seed")[metric].to_numpy()
        stat, p_value = mannwhitneyu(fito_values, baseline_values, alternative="two-sided", method="asymptotic")
        fito_mean = float(np.mean(fito_values))
        base_mean = float(np.mean(baseline_values))
        better = fito_mean > base_mean if higher_is_better else fito_mean < base_mean
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "fito_mean": fito_mean,
                "baseline_mean": base_mean,
                "mann_whitney_u": float(stat),
                "p_value": float(p_value),
                "fito_better": better,
            }
        )
        p_values.append(float(p_value))

    adjusted = holm_adjust(p_values)
    for row, corrected in zip(rows, adjusted):
        row["holm_p_value"] = corrected
    return pd.DataFrame(rows)


def case_table_latex(summary_df: pd.DataFrame, metric: str, caption: str, label: str) -> str:
    higher_is_better = metric == "mhv"
    ordered = summary_df.sort_values("mean", ascending=not higher_is_better)
    best_value = ordered.iloc[0]["mean"]
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Algorithm & Value \\\\",
        "\\midrule",
    ]
    for _, row in ordered.iterrows():
        cell = f"{row['mean']:.4f} $\\pm$ {row['std']:.4f}"
        if np.isclose(row["mean"], best_value):
            cell = "\\textbf{" + cell + "}"
        lines.append(f"{row['algorithm']} & {cell} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def deployment_table_latex(summary_df: pd.DataFrame) -> str:
    ordered = summary_df.sort_values("final_wealth_mean", ascending=False)
    best_wealth = ordered.iloc[0]["final_wealth_mean"]
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\small",
        f"\\caption{{Walk-forward deployment metrics over {len(SEEDS)} independent runs.}}",
        "\\label{tab:portfolio-deployment}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Algorithm & Final wealth & Ann. return & Ann. vol. & Mean turnover & Max drawdown \\\\",
        "\\midrule",
    ]
    for _, row in ordered.iterrows():
        wealth_cell = f"{row['final_wealth_mean']:.3f} $\\pm$ {row['final_wealth_std']:.3f}"
        if np.isclose(row["final_wealth_mean"], best_wealth):
            wealth_cell = "\\textbf{" + wealth_cell + "}"
        lines.append(
            f"{row['algorithm']} & {wealth_cell} & "
            f"{row['annualized_return_mean']:.3f} $\\pm$ {row['annualized_return_std']:.3f} & "
            f"{row['annualized_volatility_mean']:.3f} $\\pm$ {row['annualized_volatility_std']:.3f} & "
            f"{row['mean_turnover_mean']:.3f} $\\pm$ {row['mean_turnover_std']:.3f} & "
            f"{row['max_drawdown_mean']:.3f} $\\pm$ {row['max_drawdown_std']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table*}"])
    return "\n".join(lines) + "\n"


def plot_wealth_trace(raw_df: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "FITO": "#c0392b",
        "DNSGA-II-A": "#1f77b4",
        "DNSGA-II-B": "#17becf",
        "KGB-DMOEA": "#2ca02c",
        "NSGA-II": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    horizon = len(raw_df.iloc[0]["wealth_path"])
    x = np.arange(horizon + 1)

    for algorithm in MAIN_ALGORITHMS:
        paths = raw_df[raw_df["algorithm"] == algorithm]["wealth_path"].tolist()
        wealth_matrix = np.asarray([[1.0] + list(path) for path in paths], dtype=float)
        mean_path = wealth_matrix.mean(axis=0)
        std_path = wealth_matrix.std(axis=0)
        ax.plot(x, mean_path, color=colors[algorithm], linewidth=2, label=algorithm)
        ax.fill_between(x, mean_path - std_path, mean_path + std_path, color=colors[algorithm], alpha=0.14)

    ax.set_xlabel("Holdout period")
    ax.set_ylabel("Cumulative wealth")
    ax.set_title("Walk-forward out-of-sample portfolio deployment")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_case_frontier(raw_df: pd.DataFrame, output_path: Path) -> None:
    colors = {
        "FITO": "#c0392b",
        "DNSGA-II-A": "#1f77b4",
        "DNSGA-II-B": "#17becf",
        "KGB-DMOEA": "#2ca02c",
        "NSGA-II": "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    fito_row = raw_df[(raw_df["algorithm"] == "FITO") & (raw_df["seed"] == 0)].iloc[0]
    final_pf = np.asarray(fito_row["final_pf"], dtype=float)
    ax.plot(final_pf[:, 0], -final_pf[:, 1], color="black", linewidth=2, label="Approx. realized PF")
    for algorithm in MAIN_ALGORITHMS:
        row = raw_df[(raw_df["algorithm"] == algorithm) & (raw_df["seed"] == 0)].iloc[0]
        front = np.asarray(row["final_front"], dtype=float)
        ax.scatter(front[:, 0], -front[:, 1], s=16, alpha=0.75, color=colors[algorithm], label=algorithm)
    ax.set_xlabel("Realized portfolio variance")
    ax.set_ylabel("Realized mean return")
    ax.set_title("Final holdout frontier")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_summary_text(
    environments: list[dict[str, object]],
    migd_summary: pd.DataFrame,
    mhv_summary: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    eval_budget_summary: pd.DataFrame,
    runtime_summary: pd.Series,
    stats: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "Walk-Forward Big Tech Rebalancing Study",
        "======================================",
        f"Assets: {len(environments[0]['asset_names'])}",
        f"Seeds: {list(SEEDS)}",
        f"Training window (days): {TRAIN_WINDOW_DAYS}",
        f"Holdout window (days): {HOLD_WINDOW_DAYS}",
        f"Environments: {len(environments)}",
        f"Generations per environment: {GENERATIONS_PER_ENV}",
        f"Transaction cost rate: {TRANSACTION_COST_RATE * 1e4:.0f} bps",
        "",
        "Mean out-of-sample normalized HV:",
    ]
    for _, row in mhv_summary.sort_values("mean", ascending=False).iterrows():
        lines.append(f"  - {row['algorithm']}: {row['mean']:.4f} +- {row['std']:.4f}")
    lines.extend(["", "Mean out-of-sample IGD:"])
    for _, row in migd_summary.sort_values("mean", ascending=True).iterrows():
        lines.append(f"  - {row['algorithm']}: {row['mean']:.4f} +- {row['std']:.4f}")
    lines.extend(["", "Deployment metrics:"])
    for _, row in deployment_summary.sort_values("final_wealth_mean", ascending=False).iterrows():
        lines.append(
            f"  - {row['algorithm']}: final wealth={row['final_wealth_mean']:.3f}, "
            f"ann. return={row['annualized_return_mean']:.3f}, "
            f"ann. vol={row['annualized_volatility_mean']:.3f}, "
            f"turnover={row['mean_turnover_mean']:.3f}, "
            f"max drawdown={row['max_drawdown_mean']:.3f}"
        )
    lines.extend(["", "Mean training-objective evaluations per run:"])
    for _, row in eval_budget_summary.sort_values("mean").iterrows():
        lines.append(
            f"  - {row['algorithm']}: {row['mean']:.1f} +- {row['std']:.1f} "
            f"(min={int(row['min'])}, max={int(row['max'])})"
        )
    lines.extend(["", "Average runtime (sec):"])
    for algorithm_name, value in runtime_summary.sort_values().items():
        lines.append(f"  - {algorithm_name}: {value:.3f}")
    lines.extend(["", "Holm-corrected FITO wins on final wealth:"])
    wins = stats[(stats["metric"] == "final_wealth") & (stats["fito_better"]) & (stats["holm_p_value"] < 0.05)]
    if wins.empty:
        lines.append("  - None")
    else:
        for _, row in wins.iterrows():
            lines.append(f"  - vs {row['baseline']} (raw p={row['p_value']:.4f}, Holm p={row['holm_p_value']:.4f})")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    environments = build_environments()
    tasks = [{"algorithm": algorithm, "seed": seed} for algorithm in MAIN_ALGORITHMS for seed in SEEDS]
    results = []
    max_workers = min(4, max(1, os.cpu_count() or 1))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())

    raw_df = pd.DataFrame(results).sort_values(["algorithm", "seed"])
    migd_summary = summarize(raw_df, "migd")
    mhv_summary = summarize(raw_df, "mhv")
    deployment_summary = summarize_deployment(raw_df)
    eval_budget_summary = summarize_eval_budget(raw_df)
    runtime_summary = raw_df.groupby("algorithm")["runtime_sec"].mean()
    stats = pd.concat(
        [
            independent_test_against_fito(raw_df, "migd", higher_is_better=False),
            independent_test_against_fito(raw_df, "mhv", higher_is_better=True),
            independent_test_against_fito(raw_df, "final_wealth", higher_is_better=True),
            independent_test_against_fito(raw_df, "mean_net_return", higher_is_better=True),
            independent_test_against_fito(raw_df, "mean_turnover", higher_is_better=False),
        ],
        ignore_index=True,
    )

    raw_export = raw_df.copy()
    for column in ["oos_hv_curve", "oos_igd_curve", "wealth_path", "net_return_curve", "turnover_curve", "selected_weights", "decision_dates", "final_front", "final_pf", "asset_names"]:
        raw_export[column] = raw_export[column].apply(json.dumps)

    raw_export.to_csv(RESULTS_DIR / "portfolio_case_raw_metrics.csv", index=False)
    migd_summary.to_csv(RESULTS_DIR / "portfolio_case_migd_summary.csv", index=False)
    mhv_summary.to_csv(RESULTS_DIR / "portfolio_case_mhv_summary.csv", index=False)
    deployment_summary.to_csv(RESULTS_DIR / "portfolio_case_deployment_summary.csv", index=False)
    eval_budget_summary.to_csv(RESULTS_DIR / "portfolio_case_eval_budget_summary.csv", index=False)
    runtime_summary.to_csv(RESULTS_DIR / "portfolio_case_runtime_summary.csv")
    stats.to_csv(RESULTS_DIR / "portfolio_case_stats.csv", index=False)
    (
        pd.DataFrame(
            {
                "parameter": [
                    "assets",
                    "training_window_days",
                    "holdout_window_days",
                    "environments",
                    "generations_per_environment",
                    "population_size",
                    "seeds",
                    "transaction_cost_rate",
                    "fito_use_boundary_risk",
                    "fito_use_predictive_anchors",
                    "fito_use_change_memory_blend",
                ],
                "value": [
                    len(environments[0]["asset_names"]),
                    TRAIN_WINDOW_DAYS,
                    HOLD_WINDOW_DAYS,
                    len(environments),
                    GENERATIONS_PER_ENV,
                    POP_SIZE,
                    len(SEEDS),
                    TRANSACTION_COST_RATE,
                    FITO_DEFAULT_CONFIG["use_boundary_risk"],
                    FITO_DEFAULT_CONFIG["use_predictive_anchors"],
                    FITO_DEFAULT_CONFIG["use_change_memory_blend"],
                ],
            }
        ).to_json(RESULTS_DIR / "portfolio_case_config.json", orient="records", indent=2)
    )

    (RESULTS_DIR / "portfolio_case_mhv_table.tex").write_text(
        case_table_latex(
            mhv_summary,
            "mhv",
            f"Walk-forward big-tech rebalancing study measured by mean out-of-sample normalized HV over {len(SEEDS)} independent runs.",
            "tab:portfolio-mhv",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "portfolio_case_migd_table.tex").write_text(
        case_table_latex(
            migd_summary,
            "migd",
            f"Walk-forward big-tech rebalancing study measured by mean out-of-sample IGD over {len(SEEDS)} independent runs.",
            "tab:portfolio-migd",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "portfolio_case_deployment_table.tex").write_text(
        deployment_table_latex(deployment_summary),
        encoding="utf-8",
    )

    plot_wealth_trace(raw_df, RESULTS_DIR / "portfolio_case_wealth.png")
    plot_case_frontier(raw_df, RESULTS_DIR / "portfolio_case_frontier.png")
    save_summary_text(
        environments,
        migd_summary,
        mhv_summary,
        deployment_summary,
        eval_budget_summary,
        runtime_summary,
        stats,
        RESULTS_DIR / "portfolio_case_summary.txt",
    )


if __name__ == "__main__":
    main()
