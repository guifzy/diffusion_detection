from __future__ import annotations

import cv2
import numpy as np

from src.shared.features.common import entropy_from_hist, region_diff


def _largest_valid_patch(patches):
    valid = [patch for patch in patches if patch is not None and patch.size > 0 and min(patch.shape[:2]) >= 8]
    if not valid:
        return None
    return max(valid, key=lambda patch: patch.shape[0] * patch.shape[1])


def extract_fft_region_crops(frame, bbox, border_padding=0.25, background_padding=0.90):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    face_crop = frame[y1:y2, x1:x2]

    px = int(bw * border_padding)
    py = int(bh * border_padding)
    bx1 = max(0, x1 - px)
    by1 = max(0, y1 - py)
    bx2 = min(w, x2 + px)
    by2 = min(h, y2 + py)
    border_patches = [
        frame[by1:y1, bx1:bx2],
        frame[y2:by2, bx1:bx2],
        frame[y1:y2, bx1:x1],
        frame[y1:y2, x2:bx2],
    ]

    cx1 = max(0, x1 - int(bw * background_padding))
    cy1 = max(0, y1 - int(bh * background_padding))
    cx2 = min(w, x2 + int(bw * background_padding))
    cy2 = min(h, y2 + int(bh * background_padding))
    background_patches = [
        frame[cy1:by1, cx1:cx2],
        frame[by2:cy2, cx1:cx2],
        frame[cy1:cy2, cx1:bx1],
        frame[cy1:cy2, bx2:cx2],
    ]

    return {
        "face": face_crop,
        "border": _largest_valid_patch(border_patches),
        "background": _largest_valid_patch(background_patches),
    }


def _fft_region(crop) -> dict:
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = gray - np.mean(gray)
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    power = np.log1p(magnitude)

    h, w = power.shape
    yy, xx = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr_norm = rr / (rr.max() + 1e-6)

    low = power[rr_norm < 0.20]
    mid = power[(rr_norm >= 0.20) & (rr_norm < 0.55)]
    high = power[rr_norm >= 0.55]
    total = float(np.sum(power) + 1e-6)
    hist, _ = np.histogram(power.reshape(-1), bins=32)
    entropy = entropy_from_hist(hist)

    horizontal = power[np.abs(yy - cy) < max(1, h * 0.05)].sum()
    vertical = power[np.abs(xx - cx) < max(1, w * 0.05)].sum()

    return {
        "mean_power": float(np.mean(power)),
        "std_power": float(np.std(power)),
        "low_freq_ratio": float(np.sum(low) / total),
        "mid_freq_ratio": float(np.sum(mid) / total),
        "high_freq_ratio": float(np.sum(high) / total),
        "radial_centroid": float(np.sum(rr_norm * power) / total),
        "entropy_norm": entropy / np.log2(len(hist)) if len(hist) > 1 and np.isfinite(entropy) else np.nan,
        "flatness": float(np.exp(np.mean(np.log(power + 1e-6))) / (np.mean(power) + 1e-6)),
        "anisotropy": float(abs(horizontal - vertical) / (horizontal + vertical + 1e-6)),
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
    }


def compute_fft_metrics(frame, regions) -> dict:
    crops = extract_fft_region_crops(frame, regions["bbox"])
    region_features = {name: _fft_region(crops[name]) for name in ["face", "border", "background"]}
    out = {}
    for name, features in region_features.items():
        out.update({f"fft_{name}_{k}": v for k, v in features.items()})
    out.update(region_diff(region_features["face"], region_features["background"], "face_bg_fft"))
    out.update(region_diff(region_features["face"], region_features["border"], "face_border_fft"))
    out.update(region_diff(region_features["border"], region_features["background"], "border_bg_fft"))
    return out


GROUP_D_FUNCTIONS = [compute_fft_metrics]
GROUP_D_PREFIXES = ("fft",)

