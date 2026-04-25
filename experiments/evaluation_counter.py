from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _batch_size(X) -> int:
    if isinstance(X, np.ndarray) and X.dtype != object:
        return int(np.atleast_2d(X).shape[0])
    if isinstance(X, (list, tuple)):
        if not X:
            return 0
        first = X[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return len(X)
        return 1
    return 1


@dataclass
class EvaluationCounter:
    previous_callback: object | None = None
    count: int = 0

    def __call__(self, X, out) -> None:
        self.count += _batch_size(X)
        if self.previous_callback is not None:
            self.previous_callback(X, out)


def attach_evaluation_counter(problem):
    counter = EvaluationCounter(problem.callback)
    problem.callback = counter
    return counter
