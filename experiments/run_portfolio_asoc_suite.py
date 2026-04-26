from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

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
    from run_portfolio_case_study import (
        FRONTIER_POINTS,
        GENERATIONS_PER_ENV,
        HOLD_WINDOW_DAYS,
        POP_SIZE,
        STEP_SIZE,
        TRAIN_WINDOW_DAYS,
        WalkForwardPortfolioProblem,
        annualized_return,
        annualized_sharpe,
        annualized_volatility,
        evaluate_igd,
        frontier_with_weights,
        max_drawdown,
        normalized_hv,
        portfolio_objectives,
        project_weights,
        solve_long_only_frontier,
        turnover,
    )
except ModuleNotFoundError:
    from experiments.run_portfolio_case_study import (
        FRONTIER_POINTS,
        GENERATIONS_PER_ENV,
        HOLD_WINDOW_DAYS,
        POP_SIZE,
        STEP_SIZE,
        TRAIN_WINDOW_DAYS,
        WalkForwardPortfolioProblem,
        annualized_return,
        annualized_sharpe,
        annualized_volatility,
        evaluate_igd,
        frontier_with_weights,
        max_drawdown,
        normalized_hv,
        portfolio_objectives,
        project_weights,
        solve_long_only_frontier,
        turnover,
    )

try:
    from stats_utils import holm_adjust, mann_whitney_summary
except ModuleNotFoundError:
    from experiments.stats_utils import holm_adjust, mann_whitney_summary


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRICE_DATA_PATH = DATA_DIR / "big_tech_stock_prices.csv"
COMPANY_DATA_PATH = DATA_DIR / "big_tech_companies.csv"
BIG_TECH_ENV_CACHE = DATA_DIR / "big_tech_walkforward_envs.pkl"
MARKET20_PRICE_CACHE = DATA_DIR / "market20_adj_close.csv"
MARKET20_ENV_CACHE = DATA_DIR / "market20_walkforward_envs.pkl"

MAIN_ALGORITHMS = ("FITO", "DNSGA-II-A", "DNSGA-II-B", "KGB-DMOEA", "MDDM-DMOEA", "NSGA-II", "PPS-DMOEA")
UNIVERSE_DISPLAY_LABELS = {
    "tech14": "tech14 (14 assets)",
    "market20": "market20 (20 assets)",
}
SEEDS = tuple(range(20))
PRIMARY_RULE = "utopia"
PRIMARY_COST = 0.001
DECISION_RULES = ("utopia", "max_sharpe", "min_variance")
COST_RATES = (0.0005, 0.0010, 0.0025)
# Nominal target; realized values differ by algorithm/universe and are audited in raw metrics.
FIXED_BUDGET_TARGET = 60_000
STAGNATION_LIMIT = 8

FITO_DEFAULT_CONFIG = {
    "use_pitstop_restart": True,
    "use_boundary_risk": False,
    "use_predictive_anchors": True,
    "use_change_memory_blend": False,
}

UNIVERSES = {
    "tech14": {
        "source": "bundled_csv",
        "cache_path": BIG_TECH_ENV_CACHE,
        "tickers": (
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
        ),
    },
    "market20": {
        "source": "yfinance",
        "cache_path": MARKET20_ENV_CACHE,
        "price_cache": MARKET20_PRICE_CACHE,
        "tickers": (
            "AAPL",
            "AMZN",
            "BAC",
            "CAT",
            "CVX",
            "GOOGL",
            "HD",
            "HON",
            "IBM",
            "JNJ",
            "JPM",
            "KO",
            "MSFT",
            "NEE",
            "ORCL",
            "PFE",
            "PG",
            "UNH",
            "WMT",
            "XOM",
        ),
    },
}


def load_big_tech_prices() -> pd.DataFrame:
    prices_raw = pd.read_csv(PRICE_DATA_PATH)
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    selected = prices_raw[prices_raw["stock_symbol"].isin(UNIVERSES["tech14"]["tickers"])].copy()
    wide_prices = (
        selected.pivot(index="date", columns="stock_symbol", values="adj_close")
        .sort_index()
        .dropna()
    )
    return wide_prices


def download_market_prices(tickers: tuple[str, ...], cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return frame.sort_index().dropna()

    warnings.filterwarnings("ignore", category=FutureWarning)
    frames = []
    for ticker in tickers:
        series = None
        for _ in range(3):
            try:
                downloaded = yf.download(
                    ticker,
                    # Download/cache range is intentionally broader than the generated
                    # 38 holdout windows, which run from 2013-05-23 to 2022-11-22.
                    start="2012-05-18",
                    end="2023-01-01",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )["Adj Close"]
                series = downloaded.squeeze("columns")
                break
            except Exception:
                time.sleep(0.5)
        if series is None:
            raise RuntimeError(f"Failed to download price history for {ticker}")
        frames.append(series.rename(ticker))

    wide_prices = pd.concat(frames, axis=1).dropna().sort_index()
    wide_prices.to_csv(cache_path)
    return wide_prices


def build_environments(universe_name: str) -> list[dict[str, object]]:
    cfg = UNIVERSES[universe_name]
    cache_path = Path(cfg["cache_path"])
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            envs = pickle.load(handle)
        required = {"train_pf", "hold_pf", "hold_returns"}
        if universe_name == "market20":
            required.add("train_pf_weights")
        else:
            required.add("train_pf_weights")
        if envs and required.issubset(envs[0].keys()):
            return envs

    if cfg["source"] == "bundled_csv":
        wide_prices = load_big_tech_prices()
    else:
        wide_prices = download_market_prices(cfg["tickers"], Path(cfg["price_cache"]))

    returns = wide_prices.pct_change().dropna()
    asset_names = list(wide_prices.columns)

    envs: list[dict[str, object]] = []
    for start_idx in range(TRAIN_WINDOW_DAYS, len(returns) - HOLD_WINDOW_DAYS + 1, STEP_SIZE):
        train = returns.iloc[start_idx - TRAIN_WINDOW_DAYS : start_idx]
        hold = returns.iloc[start_idx : start_idx + HOLD_WINDOW_DAYS]

        train_mu = train.mean().to_numpy(dtype=float)
        train_cov = train.cov().to_numpy(dtype=float) + np.eye(train.shape[1]) * 1e-10
        hold_mu = hold.mean().to_numpy(dtype=float)
        hold_cov = hold.cov().to_numpy(dtype=float) + np.eye(hold.shape[1]) * 1e-10

        train_pf_weights, train_pf = solve_long_only_frontier(train_mu, train_cov, n_points=FRONTIER_POINTS)
        hold_pf_weights, hold_pf = solve_long_only_frontier(hold_mu, hold_cov, n_points=FRONTIER_POINTS)

        envs.append(
            {
                "asset_names": asset_names,
                "train_mu": train_mu,
                "train_cov": train_cov,
                "train_pf": train_pf,
                "train_pf_weights": train_pf_weights,
                "hold_mu": hold_mu,
                "hold_cov": hold_cov,
                "hold_pf": hold_pf,
                "hold_pf_weights": hold_pf_weights,
                "hold_returns": hold.to_numpy(dtype=float),
                "train_start": str(train.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "hold_start": str(hold.index[0].date()),
                "hold_end": str(hold.index[-1].date()),
            }
        )

    with cache_path.open("wb") as handle:
        pickle.dump(envs, handle)
    return envs


def select_portfolio(rule: str, front_weights: np.ndarray, front_points: np.ndarray, prev_weights: np.ndarray) -> np.ndarray:
    ideal = front_points.min(axis=0)
    nadir = front_points.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    normalized = np.clip((front_points - ideal) / scale, 0.0, 1.0)
    turns = np.asarray([turnover(prev_weights, weights) for weights in front_weights], dtype=float)

    if rule == "utopia":
        score = np.linalg.norm(normalized, axis=1)
        order = np.lexsort((turns, score))
        return front_weights[order[0]]

    if rule == "min_variance":
        score = front_points[:, 0]
        order = np.lexsort((turns, score))
        return front_weights[order[0]]

    if rule == "max_sharpe":
        variance = np.maximum(front_points[:, 0], 1e-12)
        expected_return = -front_points[:, 1]
        score = -(expected_return / np.sqrt(variance))
        order = np.lexsort((turns, score))
        return front_weights[order[0]]

    raise ValueError(f"Unknown decision rule: {rule}")


def net_daily_returns(gross_daily_returns: np.ndarray, turn: float, cost_rate: float) -> np.ndarray:
    net = np.asarray(gross_daily_returns, dtype=float).copy()
    if len(net):
        net[0] = (1.0 + net[0]) * (1.0 - cost_rate * turn) - 1.0
    return net


def init_rule_state(n_assets: int) -> dict[str, dict[str, object]]:
    state: dict[str, dict[str, object]] = {}
    for rule in DECISION_RULES:
        state[rule] = {
            "prev_weights": np.full(n_assets, 1.0 / n_assets),
            "selected_weights": [],
            "decision_dates": [],
            "costs": {
                cost: {
                    "wealth": 1.0,
                    "wealth_path": [],
                    "daily_returns": [],
                    "net_return_curve": [],
                    "turnover_curve": [],
                    "realized_risk_curve": [],
                    "realized_return_curve": [],
                }
                for cost in COST_RATES
            },
        }
    return state


def update_rule_state(
    state: dict[str, dict[str, object]],
    env: dict[str, object],
    front_weights: np.ndarray,
    front_points: np.ndarray,
) -> None:
    for rule in DECISION_RULES:
        prev_weights = np.asarray(state[rule]["prev_weights"], dtype=float)
        weights = select_portfolio(rule, front_weights, front_points, prev_weights)
        gross_daily = np.asarray(env["hold_returns"], dtype=float) @ weights
        realized = portfolio_objectives(
            weights,
            np.asarray(env["hold_mu"], dtype=float),
            np.asarray(env["hold_cov"], dtype=float),
        )[0]
        turn = turnover(prev_weights, weights)

        for cost_rate in COST_RATES:
            bundle = state[rule]["costs"][cost_rate]
            net_daily = net_daily_returns(gross_daily, turn, cost_rate)
            growth = float(np.prod(1.0 + net_daily))
            bundle["wealth"] *= growth
            bundle["wealth_path"].append(bundle["wealth"])
            bundle["daily_returns"].extend(net_daily.tolist())
            bundle["net_return_curve"].append(growth - 1.0)
            bundle["turnover_curve"].append(turn)
            bundle["realized_risk_curve"].append(float(realized[0]))
            bundle["realized_return_curve"].append(float(-realized[1]))

        state[rule]["selected_weights"].append(weights.tolist())
        state[rule]["decision_dates"].append(str(env["hold_start"]))
        state[rule]["prev_weights"] = weights


def finalize_rule_bundle(
    universe_name: str,
    algorithm_name: str,
    seed: int,
    oos_hv_curve: list[float],
    oos_igd_curve: list[float],
    state: dict[str, dict[str, object]],
    final_front: np.ndarray,
    final_pf: np.ndarray,
    n_evals: int,
    runtime_sec: float,
) -> list[dict[str, object]]:
    rows = []
    for rule in DECISION_RULES:
        for cost_rate in COST_RATES:
            bundle = state[rule]["costs"][cost_rate]
            daily = np.asarray(bundle["daily_returns"], dtype=float)
            wealth_path = np.asarray(bundle["wealth_path"], dtype=float)
            rows.append(
                {
                    "universe": universe_name,
                    "algorithm": algorithm_name,
                    "seed": seed,
                    "decision_rule": rule,
                    "cost_rate": cost_rate,
                    "mhv": float(np.mean(oos_hv_curve)),
                    "migd": float(np.mean(oos_igd_curve)),
                    "final_wealth": float(wealth_path[-1]),
                    "mean_net_return": float(np.mean(bundle["net_return_curve"])),
                    "mean_turnover": float(np.mean(bundle["turnover_curve"])),
                    "mean_realized_risk": float(np.mean(bundle["realized_risk_curve"])),
                    "mean_realized_return": float(np.mean(bundle["realized_return_curve"])),
                    "annualized_return": annualized_return(daily),
                    "annualized_volatility": annualized_volatility(daily),
                    "annualized_sharpe": annualized_sharpe(daily),
                    "max_drawdown": max_drawdown(wealth_path),
                    "n_evals": n_evals,
                    "runtime_sec": runtime_sec,
                    "decision_dates": state[rule]["decision_dates"],
                    "selected_weights": state[rule]["selected_weights"],
                    "final_front": non_dominated_unique(final_front).tolist(),
                    "final_pf": np.asarray(final_pf, dtype=float).tolist(),
                }
            )
    return rows


def run_fito(universe_name: str, seed: int, pop_size: int) -> list[dict[str, object]]:
    environments = build_environments(universe_name)
    problem = WalkForwardPortfolioProblem(environments, taut=GENERATIONS_PER_ENV)
    eval_counter = attach_evaluation_counter(problem)
    rng = np.random.default_rng(seed)
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    total_generations = len(environments) * GENERATIONS_PER_ENV

    X = rng.uniform(xl, xu, size=(pop_size, problem.n_var))
    F = problem.evaluate(X)
    best_score = normalized_hv(np.asarray(problem.current_env()["train_pf"], dtype=float), F)
    stagnant = 0
    previous_env_leaders: np.ndarray | None = None
    use_boundary_risk = bool(FITO_DEFAULT_CONFIG["use_boundary_risk"])
    use_predictive_anchors = bool(FITO_DEFAULT_CONFIG["use_predictive_anchors"])
    use_change_memory_blend = bool(FITO_DEFAULT_CONFIG["use_change_memory_blend"])

    oos_hv_curve: list[float] = []
    oos_igd_curve: list[float] = []
    state = init_rule_state(problem.n_var)
    final_front = np.empty((0, 2))
    final_pf = np.empty((0, 2))
    environment_change_count = 0
    change_response_count = 0
    total_replaced_count = 0
    activation_events: list[dict[str, object]] = []

    start = time.perf_counter()
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

        current_score = normalized_hv(np.asarray(problem.current_env()["train_pf"], dtype=float), F)
        if current_score > best_score + 1e-7:
            best_score = current_score
            stagnant = 0
        else:
            stagnant += 1

        if FITO_DEFAULT_CONFIG["use_pitstop_restart"] and stagnant >= STAGNATION_LIMIT:
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
            front_weights, front_points = frontier_with_weights(project_weights(X), F)
            update_rule_state(state, env, front_weights, front_points)
            final_front = hold_F
            final_pf = np.asarray(env["hold_pf"], dtype=float)

        pre_change_leaders = leaders.copy()
        previous_time = problem.time
        problem.tic()
        F = problem.evaluate(X)

        if problem.time != previous_time:
            environment_change_count += 1
            X, F, ranks, crowd = environmental_selection(X, F, pop_size)
            order = np.lexsort((-crowd, ranks))
            leaders = X[order[:elite_n]]
            anchor_pool = (
                predictive_leader_anchors(previous_env_leaders, pre_change_leaders, xl, xu)
                if use_predictive_anchors
                else None
            )
            change_idx = weakest_support_indices(order, ranks, crowd, elite_n, max(4, int(pop_size * POST_CHANGE_RATE)))
            total_replaced_count += int(len(change_idx))
            change_response_count += 1
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
            activation_events.append(
                {
                    "event_index": environment_change_count,
                    "generation": generation,
                    "previous_time": float(previous_time),
                    "current_time": float(problem.time),
                    "response_activated": 1,
                    "response_type": "fito_redeployment_predictive_anchor" if use_predictive_anchors else "fito_redeployment_leader_jitter",
                    "replaced_count": int(len(change_idx)),
                    "prediction_used": int(bool(use_predictive_anchors)),
                    "kde_success": 0,
                    "kde_fallback": 0,
                    "note": "FITO portfolio post-change redeployment activation audit.",
                }
            )
            best_score = normalized_hv(np.asarray(problem.current_env()["train_pf"], dtype=float), F)
            stagnant = 0
            previous_env_leaders = pre_change_leaders

    runtime_sec = time.perf_counter() - start
    records = finalize_rule_bundle(
        universe_name,
        "FITO",
        seed,
        oos_hv_curve,
        oos_igd_curve,
        state,
        final_front,
        final_pf,
        int(eval_counter.count),
        runtime_sec,
    )
    audit = {
        "environment_change_count": environment_change_count,
        "change_response_count": change_response_count,
        "prediction_count": change_response_count if use_predictive_anchors else 0,
        "replaced_count": total_replaced_count,
        "kde_success_count": 0,
        "kde_fallback_count": 0,
        "response_evaluation_count": 0,
        "response_activation_rate": float(change_response_count / environment_change_count) if environment_change_count else 0.0,
        "response_audit_mode": "fito_manual_redeployment",
        "activation_events_json": json.dumps(activation_events, default=str, ensure_ascii=False),
    }
    for record in records:
        record.update(audit)
    return records


class WalkForwardCallback(Callback):
    def __init__(self, universe_name: str):
        super().__init__()
        self.universe_name = universe_name
        self.generation = 0
        self.oos_hv_curve: list[float] = []
        self.oos_igd_curve: list[float] = []
        self.state: dict[str, dict[str, object]] | None = None
        self.final_front = np.empty((0, 2))
        self.final_pf = np.empty((0, 2))

    def update(self, algorithm):
        problem: WalkForwardPortfolioProblem = algorithm.problem
        X = project_weights(np.asarray(algorithm.pop.get("X"), dtype=float))
        F = np.asarray(algorithm.pop.get("F"), dtype=float)
        if self.state is None:
            self.state = init_rule_state(problem.n_var)

        if (self.generation + 1) % GENERATIONS_PER_ENV == 0:
            env = problem.current_env()
            hold_F = portfolio_objectives(X, np.asarray(env["hold_mu"], dtype=float), np.asarray(env["hold_cov"], dtype=float))
            self.oos_hv_curve.append(normalized_hv(np.asarray(env["hold_pf"], dtype=float), hold_F))
            self.oos_igd_curve.append(evaluate_igd(np.asarray(env["hold_pf"], dtype=float), hold_F))
            front_weights, front_points = frontier_with_weights(X, F)
            update_rule_state(self.state, env, front_weights, front_points)
            self.final_front = hold_F
            self.final_pf = np.asarray(env["hold_pf"], dtype=float)

        self.generation += 1
        algorithm.problem.tic()


try:
    from predictive_baselines import AuditedDNSGA2, AuditedKGB, AuditedNSGA2, MDDM, PPS, activation_audit_summary
except ModuleNotFoundError:
    from experiments.predictive_baselines import AuditedDNSGA2, AuditedKGB, AuditedNSGA2, MDDM, PPS, activation_audit_summary

def make_baseline(name: str, pop_size: int):
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
    raise ValueError(f"Unknown baseline: {name}")


def run_baseline(universe_name: str, algorithm_name: str, seed: int, pop_size: int) -> list[dict[str, object]]:
    environments = build_environments(universe_name)
    problem = WalkForwardPortfolioProblem(environments, taut=GENERATIONS_PER_ENV)
    eval_counter = attach_evaluation_counter(problem)
    callback = WalkForwardCallback(universe_name)
    start = time.perf_counter()
    algorithm = make_baseline(algorithm_name, pop_size)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", len(environments) * GENERATIONS_PER_ENV),
        seed=seed,
        callback=callback,
        verbose=False,
    )
    audited_algorithm = getattr(res, "algorithm", algorithm)
    runtime_sec = time.perf_counter() - start
    records = finalize_rule_bundle(
        universe_name,
        algorithm_name,
        seed,
        callback.oos_hv_curve,
        callback.oos_igd_curve,
        callback.state or init_rule_state(problem.n_var),
        callback.final_front,
        callback.final_pf,
        int(eval_counter.count),
        runtime_sec,
    )
    audit = activation_audit_summary(audited_algorithm)
    for record in records:
        record.update(audit)
    return records


def calibrate_budget_pop_sizes() -> dict[str, dict[str, int]]:
    calibration: dict[str, dict[str, int]] = {}
    for universe_name in UNIVERSES:
        calibration[universe_name] = {}
        for algorithm_name in MAIN_ALGORITHMS:
            if algorithm_name == "FITO":
                rows = run_fito(universe_name, seed=0, pop_size=POP_SIZE)
            else:
                rows = run_baseline(universe_name, algorithm_name, seed=0, pop_size=POP_SIZE)
            evals = max(1, int(rows[0]["n_evals"]))
            estimated = int(round(POP_SIZE * FIXED_BUDGET_TARGET / evals))
            calibration[universe_name][algorithm_name] = max(40, estimated)
    return calibration


def run_task(task: dict[str, object]) -> list[dict[str, object]]:
    universe_name = str(task["universe"])
    algorithm_name = str(task["algorithm"])
    seed = int(task["seed"])
    pop_size = int(task["pop_size"])
    family = str(task["family"])

    if algorithm_name == "FITO":
        rows = run_fito(universe_name, seed, pop_size)
    else:
        rows = run_baseline(universe_name, algorithm_name, seed, pop_size)

    for row in rows:
        row["family"] = family
        row["pop_size"] = pop_size
    return rows


def build_tasks(fixed_budget_pop_sizes: dict[str, dict[str, int]]) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for universe_name in UNIVERSES:
        for algorithm_name in MAIN_ALGORITHMS:
            for seed in SEEDS:
                tasks.append(
                    {
                        "family": "main",
                        "universe": universe_name,
                        "algorithm": algorithm_name,
                        "seed": seed,
                        "pop_size": POP_SIZE,
                    }
                )
                tasks.append(
                    {
                        "family": "budget",
                        "universe": universe_name,
                        "algorithm": algorithm_name,
                        "seed": seed,
                        "pop_size": fixed_budget_pop_sizes[universe_name][algorithm_name],
                    }
                )
    return tasks




def validate_portfolio_algorithm_coverage(raw_df: pd.DataFrame, output_prefix: Path | None = None) -> pd.DataFrame:
    """Validate that active ASOC portfolio raw metrics cover every required algorithm.

    The manuscript-level portfolio suite uses MAIN_ALGORITHMS. Older
    `portfolio_case_*` artifacts contained only the five-algorithm legacy
    case study. This validator prevents those legacy rows from being silently
    reused for the ASOC manuscript tables.
    """
    required_columns = {"family", "universe", "algorithm", "seed"}
    missing_columns = sorted(required_columns.difference(raw_df.columns))
    if missing_columns:
        raise ValueError(f"Portfolio raw metrics missing required columns: {missing_columns}")

    expected_algorithms = set(MAIN_ALGORITHMS)
    observed_algorithms = set(map(str, raw_df["algorithm"].dropna().unique()))
    unexpected_algorithms = sorted(observed_algorithms.difference(expected_algorithms))

    coverage_rows: list[dict[str, object]] = []
    missing_entries: list[dict[str, object]] = []
    expected_seeds = set(int(seed) for seed in SEEDS)
    required_families = ("main", "budget")

    for family_name in required_families:
        family_df = raw_df[raw_df["family"].astype(str) == family_name]
        for universe_name in UNIVERSES:
            universe_df = family_df[family_df["universe"].astype(str) == str(universe_name)]
            for algorithm_name in MAIN_ALGORITHMS:
                algo_df = universe_df[universe_df["algorithm"].astype(str) == algorithm_name]
                observed_seeds = {int(seed) for seed in algo_df["seed"].dropna().unique()}
                missing_seeds = sorted(expected_seeds.difference(observed_seeds))
                coverage_rows.append(
                    {
                        "family": family_name,
                        "universe": universe_name,
                        "algorithm": algorithm_name,
                        "expected_seed_count": len(expected_seeds),
                        "observed_seed_count": len(observed_seeds),
                        "row_count": int(len(algo_df)),
                        "missing_seeds": ";".join(map(str, missing_seeds)),
                        "coverage_ok": int(not missing_seeds),
                    }
                )
                if missing_seeds:
                    missing_entries.append(
                        {
                            "family": family_name,
                            "universe": universe_name,
                            "algorithm": algorithm_name,
                            "missing_seeds": missing_seeds,
                        }
                    )

    coverage_df = pd.DataFrame(coverage_rows)
    manifest = {
        "active_result_family": "asoc_portfolio",
        "expected_algorithms": list(MAIN_ALGORITHMS),
        "expected_families": list(required_families),
        "expected_universes": list(UNIVERSES.keys()),
        "expected_seeds": list(SEEDS),
        "observed_algorithms": sorted(observed_algorithms),
        "unexpected_algorithms": unexpected_algorithms,
        "missing_entries": missing_entries,
        "coverage_ok": bool(not missing_entries and not unexpected_algorithms),
        "legacy_note": "portfolio_case_* files are the archived five-algorithm legacy case study and must not be used for ASOC portfolio tables.",
    }
    if output_prefix is not None:
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        coverage_df.to_csv(output_prefix.with_name(output_prefix.name + "_coverage.csv"), index=False)
        output_prefix.with_name(output_prefix.name + "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if unexpected_algorithms:
        raise RuntimeError(f"Unexpected algorithms found in portfolio raw metrics: {unexpected_algorithms}")
    if missing_entries:
        details = "; ".join(
            f"{entry['family']}/{entry['universe']}/{entry['algorithm']} missing seeds {entry['missing_seeds']}"
            for entry in missing_entries[:10]
        )
        if len(missing_entries) > 10:
            details += f"; ... plus {len(missing_entries) - 10} more"
        raise RuntimeError(
            "ASOC portfolio raw metrics do not match MAIN_ALGORITHMS. "
            "Regenerate or complete experiments/results/asoc_portfolio_raw_metrics.csv. "
            + details
        )
    return coverage_df

def summarize_primary(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = raw_df[(raw_df["decision_rule"] == PRIMARY_RULE) & np.isclose(raw_df["cost_rate"], PRIMARY_COST)].copy()
    mhv_summary = primary.groupby(["family", "universe", "algorithm"])["mhv"].agg(["mean", "std"]).reset_index()
    migd_summary = primary.groupby(["family", "universe", "algorithm"])["migd"].agg(["mean", "std"]).reset_index()
    deployment_metrics = ["final_wealth", "annualized_return", "annualized_volatility", "annualized_sharpe", "mean_turnover", "max_drawdown"]
    deployment_summary = primary.groupby(["family", "universe", "algorithm"])[deployment_metrics].agg(["mean", "std"]).reset_index()
    deployment_summary.columns = [
        "_".join([str(part) for part in col if part]).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in deployment_summary.columns
    ]
    return mhv_summary, migd_summary, deployment_summary


def summarize_sensitivity(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    agg = raw_df.groupby(["family", "universe", "decision_rule", "cost_rate", "algorithm"])[["mhv", "migd", "final_wealth", "annualized_return"]].mean().reset_index()
    winners = []
    for (family_name, universe_name, rule, cost), group in agg.groupby(["family", "universe", "decision_rule", "cost_rate"]):
        winners.append(
            {
                "family": family_name,
                "universe": universe_name,
                "decision_rule": rule,
                "cost_rate": cost,
                "best_mhv": group.sort_values("mhv", ascending=False).iloc[0]["algorithm"],
                "best_migd": group.sort_values("migd", ascending=True).iloc[0]["algorithm"],
                "best_wealth": group.sort_values("final_wealth", ascending=False).iloc[0]["algorithm"],
                "best_ann_return": group.sort_values("annualized_return", ascending=False).iloc[0]["algorithm"],
            }
        )
    counts = []
    winner_df = pd.DataFrame(winners)
    for family_name, group in winner_df.groupby("family"):
        for metric in ("best_mhv", "best_migd", "best_wealth", "best_ann_return"):
            for algorithm_name, count in group[metric].value_counts().items():
                counts.append({"family": family_name, "metric": metric, "algorithm": algorithm_name, "count": int(count)})
    return agg, pd.DataFrame(counts)


def evaluate_simple_benchmarks() -> pd.DataFrame:
    rows = []
    for universe_name in UNIVERSES:
        environments = build_environments(universe_name)
        n_assets = len(environments[0]["asset_names"])
        prev_eq = np.full(n_assets, 1.0 / n_assets)
        prev_minvar = np.full(n_assets, 1.0 / n_assets)
        wealth_eq = 1.0
        wealth_mv = 1.0
        daily_eq: list[float] = []
        daily_mv: list[float] = []
        turns_eq = []
        turns_mv = []
        for env in environments:
            eq_weights = np.full(n_assets, 1.0 / n_assets)
            minvar_weights = np.asarray(env["train_pf_weights"], dtype=float)[0]
            turn_eq = turnover(prev_eq, eq_weights)
            turn_mv = turnover(prev_minvar, minvar_weights)
            net_eq = net_daily_returns(np.asarray(env["hold_returns"], dtype=float) @ eq_weights, turn_eq, PRIMARY_COST)
            net_mv = net_daily_returns(np.asarray(env["hold_returns"], dtype=float) @ minvar_weights, turn_mv, PRIMARY_COST)
            wealth_eq *= float(np.prod(1.0 + net_eq))
            wealth_mv *= float(np.prod(1.0 + net_mv))
            daily_eq.extend(net_eq.tolist())
            daily_mv.extend(net_mv.tolist())
            turns_eq.append(turn_eq)
            turns_mv.append(turn_mv)
            prev_eq = eq_weights
            prev_minvar = minvar_weights
        rows.extend(
            [
                {
                    "universe": universe_name,
                    "benchmark": "EqualWeight",
                    "final_wealth": wealth_eq,
                    "annualized_return": annualized_return(np.asarray(daily_eq, dtype=float)),
                    "annualized_volatility": annualized_volatility(np.asarray(daily_eq, dtype=float)),
                    "annualized_sharpe": annualized_sharpe(np.asarray(daily_eq, dtype=float)),
                    "mean_turnover": float(np.mean(turns_eq)),
                },
                {
                    "universe": universe_name,
                    "benchmark": "RollingMinVar",
                    "final_wealth": wealth_mv,
                    "annualized_return": annualized_return(np.asarray(daily_mv, dtype=float)),
                    "annualized_volatility": annualized_volatility(np.asarray(daily_mv, dtype=float)),
                    "annualized_sharpe": annualized_sharpe(np.asarray(daily_mv, dtype=float)),
                    "mean_turnover": float(np.mean(turns_mv)),
                },
            ]
        )
    return pd.DataFrame(rows)


def pairwise_stats(primary_df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    records = []
    families = []
    metrics = [
        ("mhv", True),
        ("migd", False),
        ("final_wealth", True),
        ("annualized_return", True),
    ]
    for metric_name, higher_is_better in metrics:
        raw_p_values = []
        metric_rows = []
        for universe_name in UNIVERSES:
            fito_values = primary_df[(primary_df["family"] == family_name) & (primary_df["universe"] == universe_name) & (primary_df["algorithm"] == "FITO")].sort_values("seed")[metric_name].to_numpy()
            for baseline in [algo for algo in MAIN_ALGORITHMS if algo != "FITO"]:
                baseline_values = primary_df[(primary_df["family"] == family_name) & (primary_df["universe"] == universe_name) & (primary_df["algorithm"] == baseline)].sort_values("seed")[metric_name].to_numpy()
                stats = mann_whitney_summary(fito_values, baseline_values, higher_is_better=higher_is_better)
                metric_rows.append(
                    {
                        "family": family_name,
                        "metric": metric_name,
                        "universe": universe_name,
                        "baseline": baseline,
                        "fito_mean": stats["mean_a"],
                        "baseline_mean": stats["mean_b"],
                        "mann_whitney_u": stats["mann_whitney_u"],
                        "p_value": stats["p_value"],
                        "fito_better": stats["a_better"],
                        "cliffs_delta": stats["cliffs_delta"],
                    }
                )
                raw_p_values.append(float(stats["p_value"]))
        adjusted = holm_adjust(raw_p_values)
        for row, corrected in zip(metric_rows, adjusted):
            row["holm_p_value"] = corrected
            row["holm_scope"] = f"{family_name}_{metric_name}_{len(metric_rows)}_comparisons"
            records.append(row)
        families.append((metric_name, len(metric_rows)))
    return pd.DataFrame(records)


def primary_table_latex(summary_df: pd.DataFrame, metric: str, higher_is_better: bool, caption: str, label: str) -> str:
    """Render two-universe, seven-algorithm main-family portfolio metric tables."""
    row_end = r" \\"
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * len(MAIN_ALGORITHMS) + "}",
        "\\toprule",
        "Universe & " + " & ".join(MAIN_ALGORITHMS) + row_end,
        "\\midrule",
    ]
    for universe_name in UNIVERSES:
        rows = summary_df[(summary_df["family"] == "main") & (summary_df["universe"] == universe_name)]
        best = rows["mean"].max() if higher_is_better else rows["mean"].min()
        cells = []
        for algorithm_name in MAIN_ALGORITHMS:
            row = rows[rows["algorithm"] == algorithm_name].iloc[0]
            cell = f"{row['mean']:.4f} $\\pm$ {row['std']:.4f}"
            if np.isclose(float(row["mean"]), float(best)):
                cell = "\\textbf{" + cell + "}"
            cells.append(cell)
        label_text = UNIVERSE_DISPLAY_LABELS.get(str(universe_name), str(universe_name))
        lines.append(label_text + " & " + " & ".join(cells) + row_end)
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def deployment_table_latex(summary_df: pd.DataFrame, caption: str, label: str) -> str:
    """Render the main-family two-universe, seven-algorithm deployment table.

    The input summary is a flattened CSV-friendly frame with columns such as
    final_wealth_mean and final_wealth_std; this avoids duplicated two-row CSV
    headers and keeps the table generator aligned with the shipped artifact.
    """
    row_end = r" \\"
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Universe & Algorithm & Final wealth & Ann. return & Ann. vol. & Mean turnover" + row_end,
        "\\midrule",
    ]
    universe_items = list(UNIVERSES)
    for ui, universe_name in enumerate(universe_items):
        rows = summary_df[(summary_df["family"] == "main") & (summary_df["universe"] == universe_name)]
        best_wealth = rows["final_wealth_mean"].max()
        for algorithm_name in MAIN_ALGORITHMS:
            row = rows[rows["algorithm"] == algorithm_name].iloc[0]
            wealth = f"{row['final_wealth_mean']:.3f} $\\pm$ {row['final_wealth_std']:.3f}"
            if np.isclose(float(row["final_wealth_mean"]), float(best_wealth)):
                wealth = "\\textbf{" + wealth + "}"
            label_text = UNIVERSE_DISPLAY_LABELS.get(str(universe_name), str(universe_name))
            lines.append(
                f"{label_text} & {algorithm_name} & "
                f"{wealth} & "
                f"{row['annualized_return_mean']:.3f} & "
                f"{row['annualized_volatility_mean']:.3f} & "
                f"{row['mean_turnover_mean']:.3f}" + row_end
            )
        if ui != len(universe_items) - 1:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def benchmark_table_latex(benchmark_df: pd.DataFrame, caption: str, label: str) -> str:
    """Render deterministic EqualWeight/RollingMinVar deployment references."""
    row_end = r" \\"
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Universe & Reference & Final wealth & Ann. return & Ann. vol. & Mean turnover" + row_end,
        "\\midrule",
    ]
    universe_items = list(UNIVERSES)
    for ui, universe_name in enumerate(universe_items):
        rows = benchmark_df[benchmark_df["universe"] == universe_name]
        for _, row in rows.iterrows():
            label_text = UNIVERSE_DISPLAY_LABELS.get(str(universe_name), str(universe_name))
            lines.append(
                f"{label_text} & {row['benchmark']} & "
                f"{row['final_wealth']:.3f} & {row['annualized_return']:.3f} & "
                f"{row['annualized_volatility']:.3f} & {row['mean_turnover']:.3f}" + row_end
            )
        if ui != len(universe_items) - 1:
            lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"])
    return "\n".join(lines) + "\n"

def write_summary(
    primary_df: pd.DataFrame,
    sensitivity_counts: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    budget_stats_df: pd.DataFrame,
    fixed_budget_pop_sizes: dict[str, dict[str, int]],
    output_path: Path,
) -> None:
    lines = [
        "ASOC Portfolio Walk-Forward Suite",
        "===============================",
        f"Universes: {list(UNIVERSES.keys())}",
        f"Decision rules: {list(DECISION_RULES)}",
        f"Cost rates: {list(COST_RATES)}",
        f"Primary selection: {PRIMARY_RULE} at {int(PRIMARY_COST * 10000)} bps",
        "",
        "Fixed-budget pop calibration:",
    ]
    for universe_name, mapping in fixed_budget_pop_sizes.items():
        lines.append(f"  - {universe_name}: {mapping}")

    lines.extend(["", "Primary final wealth means (main family):"])
    wealth = primary_df[(primary_df["family"] == "main")].groupby(["universe", "algorithm"])["final_wealth"].mean().reset_index()
    for universe_name in UNIVERSES:
        lines.append(f"  - {universe_name}:")
        subset = wealth[wealth["universe"] == universe_name].sort_values("final_wealth", ascending=False)
        for _, row in subset.iterrows():
            lines.append(f"    {row['algorithm']}: {row['final_wealth']:.3f}")

    lines.extend(["", "Sensitivity winner counts across rule/cost combinations:"])
    for family_name in ("main", "budget"):
        lines.append(f"  - {family_name}:")
        group = sensitivity_counts[sensitivity_counts["family"] == family_name]
        for metric in ("best_mhv", "best_migd", "best_wealth", "best_ann_return"):
            lines.append(f"    {metric}:")
            for _, row in group[group["metric"] == metric].sort_values(["count", "algorithm"], ascending=[False, True]).iterrows():
                lines.append(f"      {row['algorithm']}: {int(row['count'])}")

    lines.extend(["", "Simple benchmark strategies (primary 10 bps):"])
    for universe_name in UNIVERSES:
        subset = benchmark_df[benchmark_df["universe"] == universe_name].sort_values("final_wealth", ascending=False)
        lines.append(f"  - {universe_name}:")
        for _, row in subset.iterrows():
            lines.append(
                f"    {row['benchmark']}: wealth={row['final_wealth']:.3f}, ann_return={row['annualized_return']:.3f}, "
                f"ann_vol={row['annualized_volatility']:.3f}, turnover={row['mean_turnover']:.3f}"
            )

    lines.extend(["", "Holm-corrected FITO wins in the main family:"])
    wins = stats_df[(stats_df["holm_p_value"] < 0.05) & (stats_df["fito_better"])]
    if wins.empty:
        lines.append("  - None.")
    else:
        for _, row in wins.sort_values(["metric", "universe", "baseline"]).iterrows():
            lines.append(
                f"  - {row['metric']} | {row['universe']} vs {row['baseline']} "
                f"(raw p={row['p_value']:.4g}, Holm p={row['holm_p_value']:.4g}, delta={row['cliffs_delta']:.3f})"
            )

    lines.extend(["", "Holm-corrected FITO wins in the fixed-budget family:"])
    wins = budget_stats_df[(budget_stats_df["holm_p_value"] < 0.05) & (budget_stats_df["fito_better"])]
    if wins.empty:
        lines.append("  - None.")
    else:
        for _, row in wins.sort_values(["metric", "universe", "baseline"]).iterrows():
            lines.append(
                f"  - {row['metric']} | {row['universe']} vs {row['baseline']} "
                f"(raw p={row['p_value']:.4g}, Holm p={row['holm_p_value']:.4g}, delta={row['cliffs_delta']:.3f})"
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
                    "universe": row.get("universe"),
                    "algorithm": row.get("algorithm"),
                    "seed": row.get("seed"),
                    "pop_size": row.get("pop_size"),
                    **event,
                }
            )
    return pd.DataFrame(records)


def summarize_activation_audit(primary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate portfolio activation audit counters with flat CSV columns."""
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
    available = [col for col in audit_cols if col in primary_df.columns]
    grouped = primary_df.groupby(["family", "universe", "algorithm"], dropna=False)[available].agg(["mean", "std", "min", "max"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else str(col)
        for col in grouped.columns
    ]
    return grouped

def main() -> None:
    fixed_budget_pop_sizes = calibrate_budget_pop_sizes()
    tasks = build_tasks(fixed_budget_pop_sizes)

    rows = []
    max_workers = min(4, max(1, os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task): task for task in tasks}
        for future in as_completed(future_map):
            rows.extend(future.result())

    raw_df = pd.DataFrame(rows)
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

    raw_df.drop(columns=["decision_dates", "selected_weights", "final_front", "final_pf"]).to_csv(RESULTS_DIR / "asoc_portfolio_raw_metrics.csv", index=False)
    coverage_df.to_csv(RESULTS_DIR / "asoc_portfolio_algorithm_coverage.csv", index=False)
    activation_summary.to_csv(RESULTS_DIR / "asoc_portfolio_activation_audit_summary.csv", index=False)
    activation_events.to_csv(RESULTS_DIR / "asoc_portfolio_activation_audit_events.csv", index=False)
    mhv_summary.to_csv(RESULTS_DIR / "asoc_portfolio_mhv_summary.csv", index=False)
    migd_summary.to_csv(RESULTS_DIR / "asoc_portfolio_migd_summary.csv", index=False)
    deployment_summary.to_csv(RESULTS_DIR / "asoc_portfolio_deployment_summary.csv", index=False)
    sensitivity_full.to_csv(RESULTS_DIR / "asoc_portfolio_sensitivity_full.csv", index=False)
    sensitivity_counts.to_csv(RESULTS_DIR / "asoc_portfolio_sensitivity_counts.csv", index=False)
    benchmark_df.to_csv(RESULTS_DIR / "asoc_portfolio_benchmarks.csv", index=False)
    stats_df.to_csv(RESULTS_DIR / "asoc_portfolio_stats.csv", index=False)
    budget_stats_df.to_csv(RESULTS_DIR / "asoc_portfolio_budget_stats.csv", index=False)
    (RESULTS_DIR / "asoc_portfolio_budget_calibration.json").write_text(json.dumps(fixed_budget_pop_sizes, indent=2), encoding="utf-8")

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
            caption="Main-family walk-forward deployment metrics for both portfolio universes under the closest-to-utopia selection rule at 10 bps. Values are generated from the flattened \\texttt{asoc\\_portfolio\\_deployment\\_summary.csv}; both universes and all seven ASOC algorithms are shown.",
            label="tab:portfolio-deployment",
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "asoc_portfolio_benchmark_table.tex").write_text(
        benchmark_table_latex(
            benchmark_df,
            caption="Deterministic deployment references for the secondary walk-forward portfolio stress test under the primary 10 bps cost setting. These references provide scale only and are not included in FITO-versus-DMOEA statistical tests.",
            label="tab:portfolio-benchmarks",
        ),
        encoding="utf-8",
    )

    write_summary(primary_df, sensitivity_counts, benchmark_df, stats_df, budget_stats_df, fixed_budget_pop_sizes, RESULTS_DIR / "asoc_portfolio_summary.txt")


if __name__ == "__main__":
    main()
