from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import kurtosis
from skimage.feature import local_binary_pattern

from src.shared.features.common import entropy_from_hist, masked_values, region_contrasts


REGION_NAMES = ("face", "border", "background")
LBP_SCALES = ((8, 1), (16, 2), (24, 3))


def _normalized_hist(values: np.ndarray, bins: int, value_range) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(float)
    return hist / hist.sum() if hist.sum() > 0 else hist


def _entropy_norm(hist: np.ndarray) -> float:
    entropy = entropy_from_hist(hist)
    return entropy / np.log2(len(hist)) if len(hist) > 1 and np.isfinite(entropy) else np.nan


def _safe_js_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.sum() <= 0 or right.sum() <= 0:
        return np.nan
    return float(jensenshannon(left, right, base=2.0))


def _safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4 or np.std(values) < 1e-8:
        return 0.0
    return float(kurtosis(values, fisher=True, bias=False, nan_policy="omit"))


def compute_lbp_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out = {}

    for points, radius in LBP_SCALES:
        lbp = local_binary_pattern(gray, P=points, R=radius, method="uniform")
        bins = points + 2
        base = f"lbp_r{radius}_p{points}"
        region_features = {}
        hists = {}

        for name in REGION_NAMES:
            values = masked_values(lbp, regions[name])
            hist = _normalized_hist(values, bins=bins, value_range=(0, bins))
            hists[name] = hist
            region_features[name] = {
                "entropy_norm": _entropy_norm(hist),
                "uniformity": float(np.sum(np.square(hist))) if hist.sum() > 0 else np.nan,
                "active_bin_ratio": float(np.mean(hist > 0.01)) if hist.sum() > 0 else np.nan,
            }
            out.update({f"{base}_{name}_{key}": value for key, value in region_features[name].items()})

        for left, right, prefix in (
            ("face", "background", "face_bg"),
            ("face", "border", "face_border"),
            ("border", "background", "border_bg"),
        ):
            out.update(region_contrasts(region_features[left], region_features[right], f"{prefix}_{base}"))
            out[f"{prefix}_{base}_hist_js_distance"] = _safe_js_distance(hists[left], hists[right])

    return out


def _sobel_region(magnitude, angle, mask, strong_threshold: float) -> tuple[dict, np.ndarray]:
    mag_values = masked_values(magnitude, mask)
    angle_values = masked_values(angle, mask)
    if mag_values.size == 0:
        return {}, np.array([])

    max_magnitude = max(float(np.max(magnitude)), 1.0)
    magnitude_hist = _normalized_hist(mag_values, bins=32, value_range=(0, max_magnitude))
    orientation_hist, _ = np.histogram(
        angle_values,
        bins=18,
        range=(-np.pi, np.pi),
        weights=mag_values,
    )
    orientation_hist = orientation_hist.astype(float)
    if orientation_hist.sum() > 0:
        orientation_hist /= orientation_hist.sum()

    weights = mag_values / (mag_values.sum() + 1e-12)
    vector = np.sum(weights * np.exp(1j * angle_values))
    features = {
        "magnitude_mean": float(np.mean(mag_values)),
        "magnitude_std": float(np.std(mag_values)),
        "magnitude_median": float(np.median(mag_values)),
        "magnitude_p95": float(np.percentile(mag_values, 95)),
        "gradient_energy": float(np.mean(np.square(mag_values))),
        "magnitude_entropy_norm": _entropy_norm(magnitude_hist),
        "orientation_entropy_norm": _entropy_norm(orientation_hist),
        "orientation_coherence": float(abs(vector)),
        "strong_gradient_ratio": float(np.mean(mag_values > strong_threshold)),
    }
    return features, orientation_hist


def compute_sobel_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sx, sy)
    angle = np.arctan2(sy, sx)
    strong_threshold = float(np.mean(magnitude))

    region_features = {}
    orientation_hists = {}
    out = {}
    for name in REGION_NAMES:
        region_features[name], orientation_hists[name] = _sobel_region(
            magnitude,
            angle,
            regions[name],
            strong_threshold,
        )
        out.update({f"sobel_{name}_{key}": value for key, value in region_features[name].items()})

    for left, right, prefix in (
        ("face", "background", "face_bg_sobel"),
        ("face", "border", "face_border_sobel"),
        ("border", "background", "border_bg_sobel"),
    ):
        out.update(region_contrasts(region_features[left], region_features[right], prefix))
        out[f"{prefix}_orientation_hist_js_distance"] = _safe_js_distance(
            orientation_hists[left],
            orientation_hists[right],
        )
    return out


def _laplacian_region(laplacian, mask) -> tuple[dict, np.ndarray]:
    values = masked_values(laplacian, mask).astype(float)
    if values.size == 0:
        return {}, np.array([])

    std = float(np.std(values))
    normalized = (values - np.mean(values)) / std if std >= 1e-8 else np.zeros_like(values)
    hist = _normalized_hist(normalized, bins=40, value_range=(-5, 5))
    absolute = np.abs(values)
    features = {
        "signed_mean": float(np.mean(values)),
        "signed_std": std,
        "signed_variance": float(np.var(values)),
        "signed_energy": float(np.mean(np.square(values))),
        "abs_mean": float(np.mean(absolute)),
        "abs_median": float(np.median(absolute)),
        "abs_p95": float(np.percentile(absolute, 95)),
        "entropy_norm": _entropy_norm(hist),
        "kurtosis": _safe_kurtosis(values),
        "standardized_tail_ratio": float(np.mean(np.abs(normalized) > 1.0)),
    }
    return features, hist


def compute_laplacian_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    region_features = {}
    hists = {}
    out = {}

    for name in REGION_NAMES:
        region_features[name], hists[name] = _laplacian_region(laplacian, regions[name])
        out.update({f"lap_{name}_{key}": value for key, value in region_features[name].items()})

    for left, right, prefix in (
        ("face", "background", "face_bg_lap"),
        ("face", "border", "face_border_lap"),
        ("border", "background", "border_bg_lap"),
    ):
        out.update(region_contrasts(region_features[left], region_features[right], prefix))
        out[f"{prefix}_hist_js_distance"] = _safe_js_distance(hists[left], hists[right])
    return out


GROUP_A_FUNCTIONS = [compute_lbp_metrics, compute_sobel_metrics, compute_laplacian_metrics]
GROUP_A_PREFIXES = ("lbp", "sobel", "lap")
