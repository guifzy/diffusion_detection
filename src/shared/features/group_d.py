from __future__ import annotations

import cv2
import numpy as np

from src.shared.features.common import region_contrasts


REGION_NAMES = ("face", "border", "background")
FFT_SIZE = 128


def _valid_patches(patches):
    return [
        patch
        for patch in patches
        if patch is not None and patch.size > 0 and min(patch.shape[:2]) >= 8
    ]


def extract_fft_region_crops(frame, bbox, border_padding=0.25, background_padding=0.90):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * border_padding)
    py = int(bh * border_padding)
    bx1 = max(0, x1 - px)
    by1 = max(0, y1 - py)
    bx2 = min(w, x2 + px)
    by2 = min(h, y2 + py)

    cx1 = max(0, x1 - int(bw * background_padding))
    cy1 = max(0, y1 - int(bh * background_padding))
    cx2 = min(w, x2 + int(bw * background_padding))
    cy2 = min(h, y2 + int(bh * background_padding))

    return {
        "face": _valid_patches([frame[y1:y2, x1:x2]]),
        "border": _valid_patches(
            [
                frame[by1:y1, bx1:bx2],
                frame[y2:by2, bx1:bx2],
                frame[y1:y2, bx1:x1],
                frame[y1:y2, x2:bx2],
            ]
        ),
        "background": _valid_patches(
            [
                frame[cy1:by1, cx1:cx2],
                frame[by2:cy2, cx1:cx2],
                frame[cy1:cy2, cx1:bx1],
                frame[cy1:cy2, bx2:cx2],
            ]
        ),
    }


def _spectral_slope(power: np.ndarray, radius: np.ndarray) -> float:
    edges = np.linspace(0.05, 1.0, 25)
    centers = (edges[:-1] + edges[1:]) / 2.0
    radial_power = []
    valid_centers = []
    for low, high, center in zip(edges[:-1], edges[1:], centers):
        values = power[(radius >= low) & (radius < high)]
        if values.size and np.mean(values) > 0:
            valid_centers.append(center)
            radial_power.append(float(np.mean(values)))
    if len(radial_power) < 3:
        return np.nan
    return float(np.polyfit(np.log(valid_centers), np.log(radial_power), 1)[0])


def _fft_patch(crop, fft_size: int = FFT_SIZE) -> dict:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray = cv2.resize(gray, (fft_size, fft_size), interpolation=cv2.INTER_AREA)
    centered = gray - np.mean(gray)
    window_1d = np.hanning(fft_size)
    window = np.outer(window_1d, window_1d)
    spectrum = np.fft.fftshift(np.fft.fft2(centered * window))
    amplitude = np.abs(spectrum)
    power = np.square(amplitude)
    log_amplitude = np.log1p(amplitude)

    yy, xx = np.indices(power.shape)
    center = float(fft_size // 2)
    dy = yy - center
    dx = xx - center
    radius = np.sqrt(dx * dx + dy * dy)
    radius /= radius.max() + 1e-12
    angle = np.arctan2(dy, dx)

    non_dc = radius >= 0.02
    total_power = float(np.sum(power[non_dc]) + 1e-12)
    low = (radius >= 0.02) & (radius < 0.20)
    mid = (radius >= 0.20) & (radius < 0.55)
    high = radius >= 0.55

    probabilities = power[non_dc] / total_power
    probabilities = probabilities[probabilities > 0]
    spectral_entropy = (
        -np.sum(probabilities * np.log2(probabilities)) / np.log2(np.sum(non_dc))
        if probabilities.size
        else np.nan
    )

    positive_power = power[non_dc]
    flatness = float(
        np.exp(np.mean(np.log(positive_power + 1e-12)))
        / (np.mean(positive_power) + 1e-12)
    )

    orientation_mask = (radius >= 0.10) & (radius <= 0.95)
    orientation_weights = power[orientation_mask]
    orientation_vector = np.sum(
        orientation_weights * np.exp(2j * angle[orientation_mask])
    ) / (np.sum(orientation_weights) + 1e-12)

    return {
        "mean_log_amplitude": float(np.mean(log_amplitude[non_dc])),
        "std_log_amplitude": float(np.std(log_amplitude[non_dc])),
        "low_power_ratio": float(np.sum(power[low]) / total_power),
        "mid_power_ratio": float(np.sum(power[mid]) / total_power),
        "high_power_ratio": float(np.sum(power[high]) / total_power),
        "radial_centroid": float(np.sum(radius[non_dc] * power[non_dc]) / total_power),
        "spectral_entropy_norm": float(spectral_entropy),
        "spectral_flatness": flatness,
        "angular_anisotropy": float(abs(orientation_vector)),
        "spectral_slope": _spectral_slope(power, radius),
    }


def _fft_region(crops) -> dict:
    patch_features = [_fft_patch(crop) for crop in crops]
    if not patch_features:
        return {}
    keys = patch_features[0].keys()
    aggregated = {}
    for key in keys:
        values = np.asarray([features[key] for features in patch_features], dtype=float)
        finite = values[np.isfinite(values)]
        aggregated[key] = float(np.mean(finite)) if finite.size else np.nan
    aggregated["patch_count"] = float(len(patch_features))
    return aggregated


def compute_fft_metrics(frame, regions) -> dict:
    crops = extract_fft_region_crops(frame, regions["bbox"])
    region_features = {
        name: _fft_region(crops[name])
        for name in REGION_NAMES
    }

    out = {}
    for name, features in region_features.items():
        for key, value in features.items():
            if key == "patch_count":
                out[f"qc_fft_{name}_{key}"] = value
            else:
                out[f"fft_{name}_{key}"] = value

    forensic_features = {
        name: {key: value for key, value in features.items() if key != "patch_count"}
        for name, features in region_features.items()
    }
    for left, right, prefix in (
        ("face", "background", "face_bg_fft"),
        ("face", "border", "face_border_fft"),
        ("border", "background", "border_bg_fft"),
    ):
        out.update(region_contrasts(forensic_features[left], forensic_features[right], prefix))
    return out


GROUP_D_FUNCTIONS = [compute_fft_metrics]
GROUP_D_PREFIXES = ("fft",)
