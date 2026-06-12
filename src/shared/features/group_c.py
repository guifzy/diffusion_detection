from __future__ import annotations

import cv2
import numpy as np

from src.shared.features.common import entropy_from_hist, masked_values, region_diff


def compute_residual(frame, diameter=9, sigma_color=75, sigma_space=75):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)
    return gray.astype(np.float32) - smooth.astype(np.float32)


def _noise_region(residual, mask) -> dict:
    values = masked_values(residual, mask)
    if values.size == 0:
        return {}
    abs_values = np.abs(values)
    hist, _ = np.histogram(abs_values, bins=32)
    entropy = entropy_from_hist(hist)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "mad": float(np.median(np.abs(values - np.median(values)))),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "entropy_norm": entropy / np.log2(len(hist)) if len(hist) > 1 and np.isfinite(entropy) else np.nan,
    }


def compute_noise_metrics(frame, regions) -> dict:
    residual = compute_residual(frame)
    region_features = {name: _noise_region(residual, regions[name]) for name in ["face", "border", "background"]}
    out = {}
    for name, features in region_features.items():
        out.update({f"noise_{name}_{k}": v for k, v in features.items()})
    out.update(region_diff(region_features["face"], region_features["background"], "face_bg_noise"))
    out.update(region_diff(region_features["face"], region_features["border"], "face_border_noise"))
    out.update(region_diff(region_features["border"], region_features["background"], "border_bg_noise"))
    return out


GROUP_C_FUNCTIONS = [compute_noise_metrics]
GROUP_C_PREFIXES = ("noise",)

