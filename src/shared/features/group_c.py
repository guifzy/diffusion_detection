from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import kurtosis

from src.shared.features.common import entropy_from_hist, masked_values, region_contrasts


REGION_NAMES = ("face", "border", "background")


def compute_residual(image, diameter=9, sigma_color=75, sigma_space=75):
    smooth = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return image.astype(np.float32) - smooth.astype(np.float32)


def _lag_correlation(residual: np.ndarray, mask: np.ndarray, axis: int) -> float:
    if axis == 0:
        left = residual[:-1, :]
        right = residual[1:, :]
        valid = (mask[:-1, :] == 1) & (mask[1:, :] == 1)
    else:
        left = residual[:, :-1]
        right = residual[:, 1:]
        valid = (mask[:, :-1] == 1) & (mask[:, 1:] == 1)

    left_values = left[valid].astype(float)
    right_values = right[valid].astype(float)
    if left_values.size < 3 or np.std(left_values) < 1e-8 or np.std(right_values) < 1e-8:
        return np.nan
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _channel_correlation_mean(color_residual: np.ndarray, mask: np.ndarray) -> float:
    values = color_residual[mask == 1]
    if values.shape[0] < 3:
        return np.nan
    correlations = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if np.std(values[:, left]) < 1e-8 or np.std(values[:, right]) < 1e-8:
            continue
        correlations.append(np.corrcoef(values[:, left], values[:, right])[0, 1])
    return float(np.mean(correlations)) if correlations else np.nan


def _residual_region(gray_residual, color_residual, mask) -> dict:
    values = masked_values(gray_residual, mask).astype(float)
    if values.size == 0:
        return {}

    absolute = np.abs(values)
    max_abs = max(float(np.percentile(absolute, 99.5)), 1.0)
    hist, _ = np.histogram(np.clip(absolute, 0, max_abs), bins=32, range=(0, max_abs))
    entropy = entropy_from_hist(hist)
    std = float(np.std(values))
    return {
        "signed_mean": float(np.mean(values)),
        "std": std,
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "abs_p95": float(np.percentile(absolute, 95)),
        "entropy_norm": entropy / np.log2(len(hist)) if np.isfinite(entropy) else np.nan,
        "kurtosis": (
            float(kurtosis(values, fisher=True, bias=False, nan_policy="omit"))
            if values.size >= 4 and std >= 1e-8
            else 0.0
        ),
        "horizontal_lag1_corr": _lag_correlation(gray_residual, mask, axis=1),
        "vertical_lag1_corr": _lag_correlation(gray_residual, mask, axis=0),
        "channel_corr_mean": _channel_correlation_mean(color_residual, mask),
    }


def compute_residual_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_residual = compute_residual(gray)
    color_residual = compute_residual(frame)
    region_features = {
        name: _residual_region(gray_residual, color_residual, regions[name])
        for name in REGION_NAMES
    }

    out = {}
    for name, features in region_features.items():
        out.update({f"residual_{name}_{key}": value for key, value in features.items()})

    for left, right, prefix in (
        ("face", "background", "face_bg_residual"),
        ("face", "border", "face_border_residual"),
        ("border", "background", "border_bg_residual"),
    ):
        out.update(region_contrasts(region_features[left], region_features[right], prefix))
    return out


# Backward-compatible callable name for external consumers. New columns use
# "residual" to avoid claiming that the bilateral residual is sensor noise.
compute_noise_metrics = compute_residual_metrics

GROUP_C_FUNCTIONS = [compute_residual_metrics]
GROUP_C_PREFIXES = ("residual",)
