from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import cdist, pdist

from src.shared.features.common import entropy_from_hist, region_diff


def create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    raise RuntimeError("OpenCV SIFT is not available. Install opencv-contrib-python.")


def _descriptor_entropy(descriptors) -> float:
    if descriptors is None or len(descriptors) == 0:
        return np.nan
    hist, _ = np.histogram(descriptors.reshape(-1), bins=32)
    entropy = entropy_from_hist(hist)
    return entropy / np.log2(len(hist)) if len(hist) > 1 and np.isfinite(entropy) else np.nan


def _safe_descriptor_distance(left, right) -> float:
    if left is None or right is None or len(left) == 0 or len(right) == 0:
        return np.nan
    return float(np.mean(cdist(left, right, metric="cosine")))


def _region_keypoints(gray, mask, sift):
    keypoints, descriptors = sift.detectAndCompute(gray, mask.astype(np.uint8) * 255)
    keypoints = keypoints or []
    area = float(np.sum(mask))
    responses = np.array([kp.response for kp in keypoints], dtype=float)
    sizes = np.array([kp.size for kp in keypoints], dtype=float)
    return {
        "kp_count": float(len(keypoints)),
        "kp_density": float(len(keypoints) / area) if area > 0 else np.nan,
        "response_mean": float(np.mean(responses)) if responses.size else 0.0,
        "response_std": float(np.std(responses)) if responses.size else 0.0,
        "size_mean": float(np.mean(sizes)) if sizes.size else 0.0,
        "size_std": float(np.std(sizes)) if sizes.size else 0.0,
        "desc_entropy_norm": _descriptor_entropy(descriptors),
        "_descriptors": descriptors,
    }


def compute_sift_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = create_sift()
    region_features = {name: _region_keypoints(gray, regions[name], sift) for name in ["face", "border", "background"]}

    out = {}
    for name, features in region_features.items():
        public = {k: v for k, v in features.items() if not k.startswith("_")}
        out.update({f"sift_{name}_{k}": v for k, v in public.items()})

    public_features = {name: {k: v for k, v in vals.items() if not k.startswith("_")} for name, vals in region_features.items()}
    out.update(region_diff(public_features["face"], public_features["background"], "face_bg_sift"))
    out.update(region_diff(public_features["face"], public_features["border"], "face_border_sift"))
    out.update(region_diff(public_features["border"], public_features["background"], "border_bg_sift"))
    out["face_bg_sift_desc_dist"] = _safe_descriptor_distance(
        region_features["face"]["_descriptors"], region_features["background"]["_descriptors"]
    )
    out["face_border_sift_desc_dist"] = _safe_descriptor_distance(
        region_features["face"]["_descriptors"], region_features["border"]["_descriptors"]
    )
    out["border_bg_sift_desc_dist"] = _safe_descriptor_distance(
        region_features["border"]["_descriptors"], region_features["background"]["_descriptors"]
    )
    return out


def extract_patches(gray, mask, patch_size=16, stride=16, max_patches=128):
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return np.empty((0, patch_size * patch_size), dtype=float)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    patches = []
    for y in range(y_min, max(y_min + 1, y_max - patch_size + 1), stride):
        for x in range(x_min, max(x_min + 1, x_max - patch_size + 1), stride):
            patch_mask = mask[y : y + patch_size, x : x + patch_size]
            if patch_mask.shape != (patch_size, patch_size) or patch_mask.mean() < 0.75:
                continue
            patch = gray[y : y + patch_size, x : x + patch_size].astype(float).reshape(-1)
            patch = (patch - patch.mean()) / (patch.std() + 1e-6)
            patches.append(patch)
            if len(patches) >= max_patches:
                return np.array(patches)
    return np.array(patches)


def _patch_region(gray, mask) -> dict:
    patches = extract_patches(gray, mask)
    if len(patches) < 2:
        return {"sim_mean": np.nan, "sim_std": np.nan, "sim_median": np.nan, "sim_p95": np.nan, "patch_count": float(len(patches))}
    distances = pdist(patches, metric="cosine")
    similarities = 1 - distances
    similarities = similarities[np.isfinite(similarities)]
    if similarities.size == 0:
        return {"sim_mean": np.nan, "sim_std": np.nan, "sim_median": np.nan, "sim_p95": np.nan, "patch_count": float(len(patches))}
    return {
        "sim_mean": float(np.mean(similarities)),
        "sim_std": float(np.std(similarities)),
        "sim_median": float(np.median(similarities)),
        "sim_p95": float(np.percentile(similarities, 95)),
        "patch_count": float(len(patches)),
    }


def compute_patch_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    region_features = {name: _patch_region(gray, regions[name]) for name in ["face", "border", "background"]}
    out = {}
    for name, features in region_features.items():
        out.update({f"patch_{name}_{k}": v for k, v in features.items()})
    out.update(region_diff(region_features["face"], region_features["background"], "face_bg_patch"))
    out.update(region_diff(region_features["face"], region_features["border"], "face_border_patch"))
    out.update(region_diff(region_features["border"], region_features["background"], "border_bg_patch"))
    return out


GROUP_B_FUNCTIONS = [compute_sift_metrics, compute_patch_metrics]
GROUP_B_PREFIXES = ("sift", "patch")
