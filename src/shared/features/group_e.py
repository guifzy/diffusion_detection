from __future__ import annotations

import cv2
import numpy as np

from src.shared.features.common import (
    circular_distance,
    entropy_from_hist,
    masked_values,
    region_contrasts,
)


REGION_NAMES = ("face", "border", "background")
SHADOW_ILLUMINATION_RATIO_THRESHOLD = 0.65


def _region_values(array, mask):
    return masked_values(array, mask).astype(float)


def _photometry_region(lab, hsv, illumination, grad_mag, grad_angle, mask) -> dict:
    l_values = _region_values(lab[:, :, 0], mask)
    if l_values.size == 0:
        return {}

    a_values = _region_values(lab[:, :, 1], mask)
    b_values = _region_values(lab[:, :, 2], mask)
    sat_values = _region_values(hsv[:, :, 1], mask)
    value_values = _region_values(hsv[:, :, 2], mask)
    illumination_values = _region_values(illumination, mask)
    gradient_values = _region_values(grad_mag, mask)
    angle_values = _region_values(grad_angle, mask)

    hist, _ = np.histogram(l_values, bins=32, range=(0, 255))
    entropy = entropy_from_hist(hist)
    p05, p25, p75, p95 = np.percentile(l_values, [5, 25, 75, 95])
    l_mean = float(np.mean(l_values))

    if gradient_values.sum() > 1e-12:
        weights = gradient_values / gradient_values.sum()
        gradient_vector = np.sum(weights * np.exp(1j * angle_values))
        gradient_coherence = float(abs(gradient_vector))
        gradient_direction = float(np.angle(gradient_vector))
    else:
        gradient_coherence = 0.0
        gradient_direction = np.nan

    return {
        "l_mean": l_mean,
        "l_std": float(np.std(l_values)),
        "l_contrast_p90_norm": float((p95 - p05) / (abs(l_mean) + 1e-6)),
        "l_iqr": float(p75 - p25),
        "l_entropy_norm": entropy / np.log2(len(hist)) if np.isfinite(entropy) else np.nan,
        "a_mean": float(np.mean(a_values)),
        "a_std": float(np.std(a_values)),
        "b_mean": float(np.mean(b_values)),
        "b_std": float(np.std(b_values)),
        "saturation_mean": float(np.mean(sat_values)),
        "saturation_std": float(np.std(sat_values)),
        "value_mean": float(np.mean(value_values)),
        "illumination_mean": float(np.mean(illumination_values)),
        "illumination_std": float(np.std(illumination_values)),
        "illumination_gradient_energy": float(np.mean(np.square(gradient_values))),
        "illumination_gradient_std": float(np.std(gradient_values)),
        "illumination_gradient_coherence": gradient_coherence,
        "illumination_gradient_direction": gradient_direction,
        "dark_pixel_ratio": float(np.mean(l_values <= 0.15 * 255)),
        "bright_pixel_ratio": float(np.mean(l_values >= 0.85 * 255)),
        "black_clip_ratio": float(np.mean(l_values <= 5)),
        "white_clip_ratio": float(np.mean(l_values >= 250)),
    }


def _masked_mean(channel, mask) -> float:
    values = _region_values(channel, mask)
    return float(np.mean(values)) if values.size else np.nan


def _normalized_abs_difference(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return float(abs(left - right) / (abs(left) + abs(right) + 1e-6))


def _face_photometric_asymmetry(l_channel, face_mask) -> dict:
    ys, xs = np.where(face_mask == 1)
    if xs.size == 0:
        return {}

    x_mid = int((xs.min() + xs.max() + 1) / 2)
    y_mid = int((ys.min() + ys.max() + 1) / 2)
    masks = {}
    for name, x_slice, y_slice in (
        ("left", slice(None, x_mid), slice(None)),
        ("right", slice(x_mid, None), slice(None)),
        ("top", slice(None), slice(None, y_mid)),
        ("bottom", slice(None), slice(y_mid, None)),
        ("top_left", slice(None, x_mid), slice(None, y_mid)),
        ("top_right", slice(x_mid, None), slice(None, y_mid)),
        ("bottom_left", slice(None, x_mid), slice(y_mid, None)),
        ("bottom_right", slice(x_mid, None), slice(y_mid, None)),
    ):
        submask = face_mask.copy()
        keep = np.zeros_like(face_mask)
        keep[y_slice, x_slice] = 1
        masks[name] = submask * keep

    means = {name: _masked_mean(l_channel, mask) for name, mask in masks.items()}
    quadrants = np.array(
        [
            means["top_left"],
            means["top_right"],
            means["bottom_left"],
            means["bottom_right"],
        ],
        dtype=float,
    )
    valid_quadrants = quadrants[np.isfinite(quadrants)]
    quadrant_imbalance = (
        float(np.std(valid_quadrants) / (np.mean(np.abs(valid_quadrants)) + 1e-6))
        if valid_quadrants.size
        else np.nan
    )
    return {
        "photo_face_lr_luma_asymmetry": _normalized_abs_difference(means["left"], means["right"]),
        "photo_face_tb_luma_asymmetry": _normalized_abs_difference(means["top"], means["bottom"]),
        "photo_face_quadrant_luma_imbalance": quadrant_imbalance,
    }


def compute_photometry_metrics(frame, regions) -> dict:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_channel = lab[:, :, 0].astype(np.float32)
    x1, y1, x2, y2 = regions["bbox"]
    face_scale = max(1, min(x2 - x1, y2 - y1))
    sigma = max(3.0, face_scale * 0.04)

    illumination = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gx = cv2.Sobel(illumination, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(illumination, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(gx, gy)
    gradient_angle = np.arctan2(gy, gx)

    region_features = {
        name: _photometry_region(
            lab,
            hsv,
            illumination,
            gradient_magnitude,
            gradient_angle,
            regions[name],
        )
        for name in REGION_NAMES
    }

    out = {"qc_photo_illumination_sigma_px": float(sigma)}
    for name, features in region_features.items():
        out.update({f"photo_{name}_{key}": value for key, value in features.items()})

    direction_key = "illumination_gradient_direction"
    for left, right, prefix in (
        ("face", "background", "face_bg_photo"),
        ("face", "border", "face_border_photo"),
        ("border", "background", "border_bg_photo"),
    ):
        left_linear = {key: value for key, value in region_features[left].items() if key != direction_key}
        right_linear = {key: value for key, value in region_features[right].items() if key != direction_key}
        out.update(region_contrasts(left_linear, right_linear, prefix))
        out[f"{prefix}_{direction_key}_circular_distance"] = circular_distance(
            region_features[left].get(direction_key, np.nan),
            region_features[right].get(direction_key, np.nan),
        )

    out.update(_face_photometric_asymmetry(l_channel, regions["face"]))
    return out


def _linear_rgb_and_luminance(frame):
    rgb = frame[:, :, ::-1].astype(np.float32) / 255.0
    linear_rgb = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )
    luminance = (
        0.2126 * linear_rgb[:, :, 0]
        + 0.7152 * linear_rgb[:, :, 1]
        + 0.0722 * linear_rgb[:, :, 2]
    )
    return linear_rgb, luminance


def _shadow_region(
    luminance,
    illumination_ratio,
    illumination_gradient,
    chromaticity,
    shadow_candidates,
    shadow_boundary,
    mask,
) -> dict:
    valid = mask == 1
    if not np.any(valid):
        return {}

    candidate = valid & shadow_candidates
    lit = valid & ~shadow_candidates
    boundary = valid & shadow_boundary
    ratio_values = illumination_ratio[valid]
    candidate_count = int(np.sum(candidate))
    lit_count = int(np.sum(lit))

    if candidate_count:
        candidate_depth = float(np.mean(1.0 - np.clip(illumination_ratio[candidate], 0.0, 1.0)))
        candidate_luminance = float(np.mean(luminance[candidate]))
    else:
        candidate_depth = 0.0
        candidate_luminance = np.nan

    lit_luminance = float(np.mean(luminance[lit])) if lit_count else np.nan
    luminance_ratio = (
        float(candidate_luminance / (lit_luminance + 1e-6))
        if np.isfinite(candidate_luminance) and np.isfinite(lit_luminance)
        else 1.0
    )

    if candidate_count and lit_count:
        candidate_chroma = np.mean(chromaticity[candidate], axis=0)
        lit_chroma = np.mean(chromaticity[lit], axis=0)
        chromaticity_shift = float(np.linalg.norm(candidate_chroma - lit_chroma))
    else:
        chromaticity_shift = 0.0

    return {
        "candidate_ratio": float(candidate_count / np.sum(valid)),
        "illumination_ratio_mean": float(np.mean(ratio_values)),
        "illumination_ratio_std": float(np.std(ratio_values)),
        "illumination_ratio_p10": float(np.percentile(ratio_values, 10)),
        "candidate_depth_mean": candidate_depth,
        "candidate_to_lit_luminance_ratio": luminance_ratio,
        "candidate_chromaticity_shift": chromaticity_shift,
        "boundary_density": float(np.sum(boundary) / np.sum(valid)),
        "boundary_gradient_mean": (
            float(np.mean(illumination_gradient[boundary])) if np.any(boundary) else 0.0
        ),
    }


def compute_shadow_metrics(frame, regions) -> dict:
    linear_rgb, luminance = _linear_rgb_and_luminance(frame)
    x1, y1, x2, y2 = regions["bbox"]
    face_scale = max(1, min(x2 - x1, y2 - y1))
    sigma = max(3.0, face_scale * 0.08)

    log_luminance = np.log(luminance + 1e-4)
    log_illumination = cv2.GaussianBlur(log_luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
    illumination = np.exp(log_illumination)
    ex1, ey1, ex2, ey2 = regions["bbox_expanded"]
    reference = float(np.median(illumination[ey1:ey2, ex1:ex2]))
    if not np.isfinite(reference) or reference <= 1e-8:
        reference = float(np.median(illumination) + 1e-8)
    illumination_ratio = illumination / (reference + 1e-8)

    shadow_candidates = (
        (illumination_ratio < SHADOW_ILLUMINATION_RATIO_THRESHOLD)
        & (luminance > 0.01)
    )
    candidate_u8 = shadow_candidates.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    shadow_boundary = cv2.morphologyEx(candidate_u8, cv2.MORPH_GRADIENT, kernel) > 0

    gx = cv2.Sobel(illumination_ratio, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(illumination_ratio, cv2.CV_64F, 0, 1, ksize=3)
    illumination_gradient = cv2.magnitude(gx, gy)
    chromaticity = linear_rgb / (np.sum(linear_rgb, axis=2, keepdims=True) + 1e-6)

    region_features = {
        name: _shadow_region(
            luminance,
            illumination_ratio,
            illumination_gradient,
            chromaticity,
            shadow_candidates,
            shadow_boundary,
            regions[name],
        )
        for name in REGION_NAMES
    }

    out = {
        "qc_shadow_illumination_sigma_px": float(sigma),
        "qc_shadow_ratio_threshold": float(SHADOW_ILLUMINATION_RATIO_THRESHOLD),
    }
    for name, features in region_features.items():
        out.update({f"shadow_{name}_{key}": value for key, value in features.items()})

    for left, right, prefix in (
        ("face", "background", "face_bg_shadow"),
        ("face", "border", "face_border_shadow"),
        ("border", "background", "border_bg_shadow"),
    ):
        out.update(region_contrasts(region_features[left], region_features[right], prefix))
    return out


# Compatibility for callers that still import the old function name. The
# emitted schema deliberately uses "photo", because these are regional
# photometric proxies rather than a complete physical illumination model.
compute_physics_metrics = compute_photometry_metrics

GROUP_E_FUNCTIONS = [compute_photometry_metrics, compute_shadow_metrics]
GROUP_E_PREFIXES = ("photo", "shadow")
