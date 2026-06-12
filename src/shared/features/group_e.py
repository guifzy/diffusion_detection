from __future__ import annotations

import cv2
import numpy as np

from src.shared.features.common import entropy_from_hist, masked_values, region_diff


def _region_values(array, mask):
    return masked_values(array, mask).astype(float)


def _illumination_region(lab, hsv, illumination, grad_mag, grad_angle, mask) -> dict:
    l_values = _region_values(lab[:, :, 0], mask)
    a_values = _region_values(lab[:, :, 1], mask)
    b_values = _region_values(lab[:, :, 2], mask)
    sat_values = _region_values(hsv[:, :, 1], mask)
    val_values = _region_values(hsv[:, :, 2], mask)
    illum_values = _region_values(illumination, mask)
    grad_values = _region_values(grad_mag, mask)
    angle_values = _region_values(grad_angle, mask)

    if l_values.size == 0:
        return {}

    hist, _ = np.histogram(l_values, bins=32, range=(0, 255))
    entropy = entropy_from_hist(hist)
    p20 = np.percentile(l_values, 20)
    p80 = np.percentile(l_values, 80)

    return {
        "l_mean": float(np.mean(l_values)),
        "l_std": float(np.std(l_values)),
        "l_contrast": float(p80 - p20),
        "l_entropy_norm": entropy / np.log2(len(hist)) if len(hist) > 1 and np.isfinite(entropy) else np.nan,
        "a_mean": float(np.mean(a_values)),
        "b_mean": float(np.mean(b_values)),
        "sat_mean": float(np.mean(sat_values)),
        "value_mean": float(np.mean(val_values)),
        "illum_mean": float(np.mean(illum_values)),
        "illum_std": float(np.std(illum_values)),
        "grad_energy": float(np.mean(np.square(grad_values))),
        "grad_std": float(np.std(grad_values)),
        "grad_direction": float(np.angle(np.mean(np.exp(1j * angle_values)))) if angle_values.size else np.nan,
        "shadow_ratio": float(np.mean(l_values < p20)),
        "highlight_ratio": float(np.mean(l_values > p80)),
    }


def _face_asymmetry(l_channel, face_mask) -> dict:
    ys, xs = np.where(face_mask == 1)
    if len(xs) == 0:
        return {}
    x_mid = int((xs.min() + xs.max()) / 2)
    y_mid = int((ys.min() + ys.max()) / 2)
    left = face_mask.copy()
    left[:, x_mid:] = 0
    right = face_mask.copy()
    right[:, :x_mid] = 0
    top = face_mask.copy()
    top[y_mid:, :] = 0
    bottom = face_mask.copy()
    bottom[:y_mid, :] = 0

    left_values = _region_values(l_channel, left)
    right_values = _region_values(l_channel, right)
    top_values = _region_values(l_channel, top)
    bottom_values = _region_values(l_channel, bottom)

    return {
        "phys_face_lr_luma_diff": float(np.mean(left_values) - np.mean(right_values))
        if left_values.size and right_values.size
        else np.nan,
        "phys_face_tb_luma_diff": float(np.mean(top_values) - np.mean(bottom_values))
        if top_values.size and bottom_values.size
        else np.nan,
        "phys_face_luma_quadrant_imbalance": float(
            np.std(
                [
                    np.mean(left_values) if left_values.size else np.nan,
                    np.mean(right_values) if right_values.size else np.nan,
                    np.mean(top_values) if top_values.size else np.nan,
                    np.mean(bottom_values) if bottom_values.size else np.nan,
                ]
            )
        ),
    }


def compute_physics_metrics(frame, regions) -> dict:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_channel = lab[:, :, 0]
    illumination = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=21, sigmaY=21)
    gx = cv2.Sobel(illumination, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(illumination, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_angle = np.arctan2(gy, gx)

    region_features = {
        name: _illumination_region(lab, hsv, illumination, grad_mag, grad_angle, regions[name])
        for name in ["face", "border", "background"]
    }

    out = {}
    for name, features in region_features.items():
        out.update({f"phys_{name}_{k}": v for k, v in features.items()})
    out.update(region_diff(region_features["face"], region_features["background"], "face_bg_phys"))
    out.update(region_diff(region_features["face"], region_features["border"], "face_border_phys"))
    out.update(region_diff(region_features["border"], region_features["background"], "border_bg_phys"))
    out.update(_face_asymmetry(l_channel, regions["face"]))
    return out


GROUP_E_FUNCTIONS = [compute_physics_metrics]
GROUP_E_PREFIXES = ("phys",)

