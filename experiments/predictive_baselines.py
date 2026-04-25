from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.algorithms.moo.nsga2 import NSGA2


def environment_clock(problem):
    """Effective dynamic clock; prefer problem.time, fall back to problem.tau."""
    value = getattr(problem, "time", getattr(problem, "tau", 0))
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return float(arr[0]) if arr.size else 0.0
    except Exception:
        return value


def bounds(problem):
    xl = np.asarray(problem.xl, dtype=float)
    xu = np.asarray(problem.xu, dtype=float)
    return xl, xu, np.maximum(xu - xl, 1e-12)


def normal_noise(algorithm, scale, size):
    rng = getattr(algorithm, "random_state", None)
    if rng is not None and hasattr(rng, "normal"):
        return rng.normal(0.0, scale, size=size)
    return np.random.normal(0.0, scale, size=size)


def choice_indices(algorithm, n_items: int, size: int):
    rng = getattr(algorithm, "random_state", None)
    if rng is not None and hasattr(rng, "integers"):
        return rng.integers(0, n_items, size=size)
    return np.random.randint(0, n_items, size=size)


def worst_population_indices(pop, fraction: float = 0.5) -> np.ndarray:
    F = np.asarray(pop.get("F"), dtype=float)
    n = len(F)
    if n == 0:
        return np.asarray([], dtype=int)
    keep = int(np.floor(n * (1.0 - fraction)))
    keep = min(max(1, keep), n - 1)
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        fronts = NonDominatedSorting().do(F)
        order = np.concatenate([np.asarray(front, dtype=int) for front in fronts if len(front)])
        if len(order) != n:
            missing = np.setdiff1d(np.arange(n), order, assume_unique=False)
            order = np.concatenate([order, missing])
    except Exception:
        order = np.lexsort(tuple(F[:, j] for j in reversed(range(F.shape[1]))))
    return np.asarray(order[keep:], dtype=int)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _attached_counter(algorithm):
    callback = getattr(getattr(algorithm, "problem", None), "callback", None)
    return callback if hasattr(callback, "count") else None


def force_evaluate_population(algorithm, *, count_in_budget: bool = False) -> int:
    """Force objective re-evaluation after direct X edits inside pymoo algorithms.

    Directly editing ``X`` leaves objective values stale.  The refresh is required
    for correct survival/selection, but the ASOC generation-matched protocol
    should not silently inflate nominal ``n_evals`` because of this bookkeeping
    refresh.  By default, pymoo's evaluator counter and the project-level
    EvaluationCounter are restored after the refresh.  The refresh cost is still
    reported separately as ``response_evaluation_count`` in the activation audit.

    Set ``count_in_budget=True`` only for experiments that intentionally charge
    response refreshes to the optimization budget.
    """
    evaluator = getattr(algorithm, "evaluator", None)
    counter = _attached_counter(algorithm)
    evaluator_before = _safe_int(getattr(evaluator, "n_eval", 0), 0) if evaluator is not None else 0
    counter_before = _safe_int(getattr(counter, "count", 0), 0) if counter is not None else 0

    def _restore_if_needed() -> int:
        evaluator_after = _safe_int(getattr(evaluator, "n_eval", evaluator_before), evaluator_before) if evaluator is not None else evaluator_before
        counter_after = _safe_int(getattr(counter, "count", counter_before), counter_before) if counter is not None else counter_before
        response_evals = max(0, evaluator_after - evaluator_before, counter_after - counter_before)
        # If no external counter is attached, fall back to population size.
        if response_evals == 0:
            try:
                response_evals = len(algorithm.pop)
            except Exception:
                response_evals = 0
        algorithm.response_evaluation_count = int(getattr(algorithm, "response_evaluation_count", 0)) + int(response_evals)
        if not count_in_budget:
            if evaluator is not None and hasattr(evaluator, "n_eval"):
                evaluator.n_eval = evaluator_before
            if counter is not None:
                counter.count = counter_before
        return int(response_evals)

    try:
        algorithm.evaluator.eval(algorithm.problem, algorithm.pop, skip_already_evaluated=False)
        return _restore_if_needed()
    except TypeError:
        pass
    for ind in algorithm.pop:
        try:
            ind.evaluated.clear()
        except Exception:
            try:
                ind.evaluated = set()
            except Exception:
                pass
        try:
            ind.set("F", None)
        except Exception:
            pass
    algorithm.evaluator.eval(algorithm.problem, algorithm.pop)
    return _restore_if_needed()


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def init_activation_audit(obj, mode: str) -> None:
    """Attach a uniform change-response audit state to a pymoo algorithm object."""
    obj.audit_response_mode = mode
    obj.previous_time = None
    obj.environment_change_count = 0
    obj.change_response_count = 0
    obj.prediction_count = 0
    obj.replaced_count = 0
    obj.kde_success_count = 0
    obj.kde_fallback_count = 0
    obj.response_evaluation_count = 0
    obj.response_event_log: list[dict[str, Any]] = []


def algorithm_generation(obj) -> int:
    for attr in ("n_gen", "generation"):
        value = getattr(obj, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return -1


def mark_environment_change(obj, current_time) -> tuple[bool, int | None, Any]:
    previous_time = getattr(obj, "previous_time", None)
    changed = previous_time is not None and current_time != previous_time
    if changed:
        obj.environment_change_count = int(getattr(obj, "environment_change_count", 0)) + 1
        return True, obj.environment_change_count, previous_time
    return False, None, previous_time


def record_activation_event(
    obj,
    *,
    event_index: int,
    previous_time,
    current_time,
    response_activated: bool,
    response_type: str,
    replaced_count: int = 0,
    prediction_used: bool = False,
    kde_success: bool = False,
    kde_fallback: bool = False,
    response_evaluation_count: int = 0,
    note: str = "",
) -> None:
    if response_activated:
        obj.change_response_count = int(getattr(obj, "change_response_count", 0)) + 1
    if prediction_used:
        obj.prediction_count = int(getattr(obj, "prediction_count", 0)) + 1
    if kde_success:
        obj.kde_success_count = int(getattr(obj, "kde_success_count", 0)) + 1
    if kde_fallback:
        obj.kde_fallback_count = int(getattr(obj, "kde_fallback_count", 0)) + 1
    obj.replaced_count = int(getattr(obj, "replaced_count", 0)) + int(replaced_count)
    # force_evaluate_population already adds response_evaluation_count to the object;
    # keep this event-level value for traceability without double-counting object total.
    event = {
        "event_index": int(event_index),
        "generation": algorithm_generation(obj),
        "previous_time": previous_time,
        "current_time": current_time,
        "response_activated": int(bool(response_activated)),
        "response_type": response_type,
        "replaced_count": int(replaced_count),
        "prediction_used": int(bool(prediction_used)),
        "kde_success": int(bool(kde_success)),
        "kde_fallback": int(bool(kde_fallback)),
        "response_evaluation_count": int(response_evaluation_count),
        "note": note,
    }
    obj.response_event_log.append(event)


def activation_audit_summary(algorithm) -> dict[str, object]:
    events = list(getattr(algorithm, "response_event_log", []))
    environment_change_count = int(getattr(algorithm, "environment_change_count", 0))
    change_response_count = int(getattr(algorithm, "change_response_count", 0))
    return {
        "environment_change_count": environment_change_count,
        "change_response_count": change_response_count,
        "prediction_count": int(getattr(algorithm, "prediction_count", 0)),
        "replaced_count": int(getattr(algorithm, "replaced_count", 0)),
        "kde_success_count": int(getattr(algorithm, "kde_success_count", 0)),
        "kde_fallback_count": int(getattr(algorithm, "kde_fallback_count", 0)),
        "response_evaluation_count": int(getattr(algorithm, "response_evaluation_count", 0)),
        "response_activation_rate": float(change_response_count / environment_change_count) if environment_change_count else 0.0,
        "response_audit_mode": str(getattr(algorithm, "audit_response_mode", "unavailable")),
        "activation_events_json": json.dumps(events, default=_json_default, ensure_ascii=False),
    }


def _audit_environment_change(obj, *, response_activated: bool, response_type: str, prediction_used: bool = False, note: str = ""):
    current_time = environment_clock(obj.problem)
    changed, event_index, previous_time = mark_environment_change(obj, current_time)
    if changed:
        record_activation_event(
            obj,
            event_index=int(event_index),
            previous_time=previous_time,
            current_time=current_time,
            response_activated=response_activated,
            response_type=response_type,
            prediction_used=prediction_used,
            note=note,
        )
    obj.previous_time = current_time
    return changed, event_index, previous_time, current_time


class AuditedNSGA2(NSGA2):
    """Plain NSGA-II with audit counters; environment changes are observed but no explicit response is activated."""

    def __init__(self, *args, audit_response_mode: str = "none_plain_nsga2", **kwargs):
        super().__init__(*args, **kwargs)
        init_activation_audit(self, audit_response_mode)

    def _audit(self):
        _audit_environment_change(
            self,
            response_activated=False,
            response_type="none",
            note="Plain NSGA-II is intentionally audited as a zero-response baseline.",
        )

    def _advance(self, infills=None, **kwargs):
        self._audit()
        return super()._advance(infills=infills, **kwargs)

    def _step(self):  # compatibility for older/custom pymoo forks
        self._audit()
        parent_step = getattr(super(), "_step", None)
        if parent_step is not None:
            return parent_step()
        return None


class AuditedDNSGA2(DNSGA2):
    """DNSGA-II wrapper exposing dynamic-change activation counters."""

    def __init__(self, *args, audit_response_mode: str = "dnsga2_internal_dynamic_response", **kwargs):
        super().__init__(*args, **kwargs)
        init_activation_audit(self, audit_response_mode)

    def _audit(self):
        _audit_environment_change(
            self,
            response_activated=True,
            response_type=self.audit_response_mode,
            note="DNSGA-II dynamic response is handled inside pymoo; replacement count is not exposed by the library.",
        )

    def _advance(self, infills=None, **kwargs):
        self._audit()
        return super()._advance(infills=infills, **kwargs)

    def _step(self):
        self._audit()
        parent_step = getattr(super(), "_step", None)
        if parent_step is not None:
            return parent_step()
        return None


class AuditedKGB(KGB):
    """KGB-DMOEA wrapper exposing dynamic-change activation counters."""

    def __init__(self, *args, audit_response_mode: str = "kgb_internal_knowledge_guided_response", **kwargs):
        super().__init__(*args, **kwargs)
        init_activation_audit(self, audit_response_mode)

    def _audit(self):
        _audit_environment_change(
            self,
            response_activated=True,
            response_type=self.audit_response_mode,
            prediction_used=True,
            note="KGB-DMOEA dynamic response is handled inside pymoo; replacement count is not exposed by the library.",
        )

    def _advance(self, infills=None, **kwargs):
        self._audit()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.naive_bayes")
            return super()._advance(infills=infills, **kwargs)

    def _step(self):
        self._audit()
        parent_step = getattr(super(), "_step", None)
        if parent_step is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.naive_bayes")
                return parent_step()
        return None


class PPS(NSGA2):
    """Audited Population Prediction Strategy dynamic baseline."""

    def __init__(self, pop_size=100, response_fraction=0.5, jitter=0.015, **kwargs):
        super().__init__(pop_size=pop_size, **kwargs)
        init_activation_audit(self, "pps_centroid_drift_tail_replacement")
        self.response_fraction = response_fraction
        self.jitter = jitter
        self.previous_centroid = None

    def _apply_prediction_response(self, event_index: int, previous_time, current_time):
        X = np.asarray(self.pop.get("X"), dtype=float)
        if len(X) == 0:
            record_activation_event(
                self,
                event_index=event_index,
                previous_time=previous_time,
                current_time=current_time,
                response_activated=False,
                response_type="pps_skipped_empty_population",
                note="PPS observed a change but the population was empty.",
            )
            return
        centroid = np.mean(X, axis=0)
        drift = np.zeros_like(centroid) if self.previous_centroid is None else centroid - self.previous_centroid
        replace_idx = worst_population_indices(self.pop, self.response_fraction)
        if len(replace_idx) == 0:
            self.previous_centroid = centroid.copy()
            record_activation_event(
                self,
                event_index=event_index,
                previous_time=previous_time,
                current_time=current_time,
                response_activated=False,
                response_type="pps_skipped_no_tail_indices",
                note="PPS observed a change but no replaceable tail indices were available.",
            )
            return
        xl, xu, span = bounds(self.problem)
        X_new = X.copy()
        noise = normal_noise(self, self.jitter, size=(len(replace_idx), X.shape[1])) * span
        X_new[replace_idx] = np.clip(X[replace_idx] + drift + noise, xl, xu)
        self.pop.set("X", X_new)
        response_evals = force_evaluate_population(self)
        record_activation_event(
            self,
            event_index=event_index,
            previous_time=previous_time,
            current_time=current_time,
            response_activated=True,
            response_type=self.audit_response_mode,
            replaced_count=int(len(replace_idx)),
            prediction_used=True,
            response_evaluation_count=int(response_evals),
            note="PPS replaced dominated-tail individuals using centroid drift and jitter.",
        )
        self.previous_centroid = centroid.copy()

    def _audit_and_respond(self):
        current_time = environment_clock(self.problem)
        changed, event_index, previous_time = mark_environment_change(self, current_time)
        if changed:
            self._apply_prediction_response(int(event_index), previous_time, current_time)
        self.previous_time = current_time

    def _update_memory(self):
        try:
            self.previous_centroid = np.mean(np.asarray(self.pop.get("X"), dtype=float), axis=0).copy()
        except Exception:
            pass

    def _advance(self, infills=None, **kwargs):
        self._audit_and_respond()
        result = super()._advance(infills=infills, **kwargs)
        self._update_memory()
        return result

    def _step(self):
        self._audit_and_respond()
        parent_step = getattr(super(), "_step", None)
        result = parent_step() if parent_step is not None else None
        self._update_memory()
        return result


class MDDM(NSGA2):
    """Audited KDE-based MDDM surrogate dynamic baseline."""

    def __init__(self, pop_size=100, response_fraction=0.5, jitter=0.01, **kwargs):
        super().__init__(pop_size=pop_size, **kwargs)
        init_activation_audit(self, "mddm_kde_density_tail_replacement")
        self.response_fraction = response_fraction
        self.jitter = jitter
        self.previous_X = None
        self.previous_centroid = None

    def _sample_density_candidates(self, n_candidates: int, current_X: np.ndarray, drift: np.ndarray) -> tuple[np.ndarray, str]:
        xl, xu, span = bounds(self.problem)
        source = self.previous_X if self.previous_X is not None and len(self.previous_X) > 1 else current_X
        try:
            from scipy.stats import gaussian_kde

            kde_noise = normal_noise(self, 1e-6, size=source.shape)
            kde = gaussian_kde((source + kde_noise).T)
            try:
                sampled = kde.resample(n_candidates, seed=getattr(self, "random_state", None)).T
            except TypeError:
                sampled = kde.resample(n_candidates).T
            sample_mode = "kde_success"
        except Exception:
            pick = choice_indices(self, len(source), n_candidates)
            sampled = source[pick] + normal_noise(self, self.jitter, size=(n_candidates, current_X.shape[1])) * span
            sample_mode = "kde_fallback"
        sampled = sampled + drift
        sampled = sampled + normal_noise(self, self.jitter, size=sampled.shape) * span
        return np.clip(sampled, xl, xu), sample_mode

    def _apply_density_response(self, event_index: int, previous_time, current_time):
        X = np.asarray(self.pop.get("X"), dtype=float)
        if len(X) == 0:
            record_activation_event(
                self,
                event_index=event_index,
                previous_time=previous_time,
                current_time=current_time,
                response_activated=False,
                response_type="mddm_skipped_empty_population",
                note="MDDM observed a change but the population was empty.",
            )
            return
        centroid = np.mean(X, axis=0)
        drift = np.zeros_like(centroid) if self.previous_centroid is None else centroid - self.previous_centroid
        replace_idx = worst_population_indices(self.pop, self.response_fraction)
        if len(replace_idx) == 0:
            self.previous_centroid = centroid.copy()
            self.previous_X = X.copy()
            record_activation_event(
                self,
                event_index=event_index,
                previous_time=previous_time,
                current_time=current_time,
                response_activated=False,
                response_type="mddm_skipped_no_tail_indices",
                note="MDDM observed a change but no replaceable tail indices were available.",
            )
            return
        candidates, sample_mode = self._sample_density_candidates(len(replace_idx), X, drift)
        X_new = X.copy()
        X_new[replace_idx] = candidates
        self.pop.set("X", X_new)
        response_evals = force_evaluate_population(self)
        record_activation_event(
            self,
            event_index=event_index,
            previous_time=previous_time,
            current_time=current_time,
            response_activated=True,
            response_type=self.audit_response_mode,
            replaced_count=int(len(replace_idx)),
            prediction_used=True,
            kde_success=sample_mode == "kde_success",
            kde_fallback=sample_mode == "kde_fallback",
            response_evaluation_count=int(response_evals),
            note=f"MDDM replaced dominated-tail individuals using {sample_mode}.",
        )
        self.previous_centroid = centroid.copy()
        self.previous_X = X.copy()

    def _audit_and_respond(self):
        current_time = environment_clock(self.problem)
        changed, event_index, previous_time = mark_environment_change(self, current_time)
        if changed:
            self._apply_density_response(int(event_index), previous_time, current_time)
        self.previous_time = current_time

    def _update_memory(self):
        try:
            X_after = np.asarray(self.pop.get("X"), dtype=float)
            self.previous_X = X_after.copy()
            self.previous_centroid = np.mean(X_after, axis=0).copy()
        except Exception:
            pass

    def _advance(self, infills=None, **kwargs):
        self._audit_and_respond()
        result = super()._advance(infills=infills, **kwargs)
        self._update_memory()
        return result

    def _step(self):
        self._audit_and_respond()
        parent_step = getattr(super(), "_step", None)
        result = parent_step() if parent_step is not None else None
        self._update_memory()
        return result
