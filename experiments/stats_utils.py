from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy.stats import mannwhitneyu, t


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted_value = (m - rank) * float(p_values[idx])
        running = max(running, adjusted_value)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, mean
    stderr = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    margin = float(t.ppf(0.975, df=arr.size - 1) * stderr)
    return mean - margin, mean + margin


def cliffs_delta(sample_a: Iterable[float], sample_b: Iterable[float]) -> float:
    a = np.asarray(list(sample_a), dtype=float)
    b = np.asarray(list(sample_b), dtype=float)
    if a.size == 0 or b.size == 0:
        return math.nan
    greater = 0
    lower = 0
    for value in a:
        greater += int(np.sum(value > b))
        lower += int(np.sum(value < b))
    return float((greater - lower) / (a.size * b.size))


def mann_whitney_summary(
    sample_a: Iterable[float],
    sample_b: Iterable[float],
    higher_is_better: bool,
) -> dict[str, float | bool]:
    a = np.asarray(list(sample_a), dtype=float)
    b = np.asarray(list(sample_b), dtype=float)
    stat, p_value = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    better = mean_a > mean_b if higher_is_better else mean_a < mean_b
    ci_low_a, ci_high_a = mean_ci95(a)
    ci_low_b, ci_high_b = mean_ci95(b)
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mann_whitney_u": float(stat),
        "p_value": float(p_value),
        "a_better": bool(better),
        "cliffs_delta": cliffs_delta(a, b),
        "a_ci95_low": ci_low_a,
        "a_ci95_high": ci_high_a,
        "b_ci95_low": ci_low_b,
        "b_ci95_high": ci_high_b,
    }
