"""Correlation scoring, mapping, and metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def cosine_distance_matrix(ingress: np.ndarray, egress: np.ndarray) -> np.ndarray:
    ingress = ingress / np.clip(np.linalg.norm(ingress, axis=1, keepdims=True), 1e-12, None)
    egress = egress / np.clip(np.linalg.norm(egress, axis=1, keepdims=True), 1e-12, None)
    return 1.0 - ingress @ egress.T


def one_to_one_mapping(distances: np.ndarray) -> list[tuple[int, int, float]]:
    rows, columns = linear_sum_assignment(distances)
    return [(int(i), int(j), float(distances[i, j])) for i, j in zip(rows, columns)]


def threshold_metrics(distances: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("metrics require a square matrix with ground-truth matches on the diagonal")
    positives = np.diag(distances)
    negatives = distances[~np.eye(len(distances), dtype=bool)]
    return np.asarray(
        [(threshold, np.mean(positives < threshold), np.mean(negatives < threshold))
         for threshold in thresholds],
        dtype=float,
    )

