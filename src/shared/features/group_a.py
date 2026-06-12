from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import kurtosis
from skimage.feature import local_binary_pattern

from src.shared.features.common import entropy_from_hist, masked_values, region_diff


def _hist(values: np.ndarray, bins: int, value_range) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    return hist.astype(float)


def _basic_stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "energy": float(np.mean(np.square(values.astype(float)))),
        "kurtosis": float(kurtosis(values.reshape(-1), fisher=True, nan_policy="omit")),
    }


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.sum() == 0 or right.sum() == 0:
        return np.nan
    return float(cosine(left, right))


def compute_lbp_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    region_features = {}
    hists = {}

    for name in ["face", "border", "background"]:
        values = masked_values(lbp, regions[name])
        hist = _hist(values, bins=10, value_range=(0, 10))
        hists[name] = hist
        entropy = entropy_from_hist(hist)
        stats = _basic_stats(values)
        stats.update(
            {
                "entropy_norm": entropy / np.log2(len(hist)) if len(hist) > 1 and np.isfinite(entropy) else np.nan,
                "uniformity": float(np.sum(np.square(hist / hist.sum()))) if hist.sum() > 0 else np.nan,
                "sparsity": float(np.mean(hist == 0)),
            }
        )
        region_features[name] = stats

    out = {}
    for name, features in region_features.items():
        out.update({f"lbp_{name}_{k}": v for k, v in features.items()})

    out.update(region_diff(region_features["face"], region_features["background"], "face_bg_lbp"))
    out.update(region_diff(region_features["face"], region_features["border"], "face_border_lbp"))
    out.update(region_diff(region_features["border"], region_features["background"], "border_bg_lbp"))
    out["face_bg_lbp_hist_dist"] = _safe_cosine(hists["face"], hists["background"])
    out["face_border_lbp_hist_dist"] = _safe_cosine(hists["face"], hists["border"])
    out["border_bg_lbp_hist_dist"] = _safe_cosine(hists["border"], hists["background"])
    return out


def compute_sobel_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    max_mag = float(np.max(mag)) if np.max(mag) > 0 else 1.0
    region_features = {}

    for name in ["face", "border", "background"]:
        values = masked_values(mag, regions[name])
        stats = _basic_stats(values)
        hist = _hist(values, bins=32, value_range=(0, max_mag))
        entropy = entropy_from_hist(hist)
        stats["entropy_norm"] = entropy / np.log2(len(hist)) if len(hist) > 1 and np.isfinite(entropy) else np.nan
        stats["coherence"] = float(np.mean(values > np.mean(mag))) if values.size else np.nan
        region_features[name] = stats

    out = {}
    for name, features in region_features.items():
        out.update({f"sobel_{name}_{k}": v for k, v in features.items()})
    out.update(region_diff(region_features["face"], region_features["background"], "face_bg_sobel"))
    out.update(region_diff(region_features["face"], region_features["border"], "face_border_sobel"))
    out.update(region_diff(region_features["border"], region_features["background"], "border_bg_sobel"))
    return out


def compute_laplacian_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    abs_lap = np.abs(lap)
    max_lap = float(np.max(abs_lap)) if np.max(abs_lap) > 0 else 1.0
    region_features = {}
    hists = {}

    for name in ["face", "border", "background"]:
        values = masked_values(abs_lap, regions[name])
        hist = _hist(values, bins=32, value_range=(0, max_lap))
        hists[name] = hist
        stats = _basic_stats(values)
        stats["variance"] = float(np.var(values)) if values.size else np.nan
        region_features[name] = stats

    out = {}
    for name, features in region_features.items():
        out.update({f"lap_{name}_{k}": v for k, v in features.items()})
    out.update(region_diff(region_features["face"], region_features["background"], "face_bg_lap"))
    out.update(region_diff(region_features["face"], region_features["border"], "face_border_lap"))
    out.update(region_diff(region_features["border"], region_features["background"], "border_bg_lap"))
    out["face_bg_lap_hist_dist"] = _safe_cosine(hists["face"], hists["background"])
    out["face_border_lap_hist_dist"] = _safe_cosine(hists["face"], hists["border"])
    out["border_bg_lap_hist_dist"] = _safe_cosine(hists["border"], hists["background"])
    return out


GROUP_A_FUNCTIONS = [compute_lbp_metrics, compute_sobel_metrics, compute_laplacian_metrics]
GROUP_A_PREFIXES = ("lbp", "sobel", "lap")

