from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.distance import pdist

from src.shared.features.common import entropy_from_hist, region_contrasts


REGION_NAMES = ("face", "border", "background")


def create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=500)
    raise RuntimeError("OpenCV SIFT is not available. Install opencv-contrib-python.")


def _normalized_entropy(values, bins: int, value_range=None, weights=None) -> float:
    if values is None or len(values) == 0:
        return np.nan
    hist, _ = np.histogram(values, bins=bins, range=value_range, weights=weights)
    entropy = entropy_from_hist(hist)
    return entropy / np.log2(bins) if np.isfinite(entropy) and bins > 1 else np.nan


def _keypoint_coverage(keypoints, mask, grid_size: int = 4) -> float:
    if not keypoints:
        return 0.0
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return np.nan

    x_min, x_max = float(xs.min()), float(xs.max() + 1)
    y_min, y_max = float(ys.min()), float(ys.max() + 1)
    occupied = set()
    for keypoint in keypoints:
        x, y = keypoint.pt
        gx = min(grid_size - 1, int(grid_size * (x - x_min) / max(x_max - x_min, 1.0)))
        gy = min(grid_size - 1, int(grid_size * (y - y_min) / max(y_max - y_min, 1.0)))
        occupied.add((gx, gy))
    return float(len(occupied) / (grid_size * grid_size))


def _descriptor_self_similarity(descriptors, max_descriptors: int = 80) -> float:
    if descriptors is None or len(descriptors) < 2:
        return np.nan
    if len(descriptors) > max_descriptors:
        indices = np.linspace(0, len(descriptors) - 1, max_descriptors).astype(int)
        descriptors = descriptors[indices]
    distances = pdist(descriptors, metric="cosine")
    distances = distances[np.isfinite(distances)]
    return float(1.0 - np.mean(distances)) if distances.size else np.nan


def _region_keypoints(gray, mask, sift):
    keypoints, descriptors = sift.detectAndCompute(gray, mask.astype(np.uint8) * 255)
    keypoints = keypoints or []
    area = float(np.sum(mask))
    responses = np.array([kp.response for kp in keypoints], dtype=float)
    sizes = np.array([kp.size for kp in keypoints], dtype=float)
    angles = np.deg2rad(np.array([kp.angle for kp in keypoints if kp.angle >= 0], dtype=float))
    angles = (angles + np.pi) % (2 * np.pi) - np.pi

    orientation_coherence = float(abs(np.mean(np.exp(1j * angles)))) if angles.size else np.nan
    descriptor_values = descriptors.reshape(-1) if descriptors is not None else np.array([])
    return {
        "kp_count": float(len(keypoints)),
        "kp_density": float(len(keypoints) / area) if area > 0 else np.nan,
        "kp_coverage": _keypoint_coverage(keypoints, mask),
        "response_mean": float(np.mean(responses)) if responses.size else 0.0,
        "response_std": float(np.std(responses)) if responses.size else 0.0,
        "size_mean": float(np.mean(sizes)) if sizes.size else 0.0,
        "size_std": float(np.std(sizes)) if sizes.size else 0.0,
        "orientation_entropy_norm": _normalized_entropy(angles, bins=18, value_range=(-np.pi, np.pi)),
        "orientation_coherence": orientation_coherence,
        "descriptor_entropy_norm": _normalized_entropy(descriptor_values, bins=32),
        "descriptor_self_similarity": _descriptor_self_similarity(descriptors),
    }


def compute_sift_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sift = create_sift()
    region_features = {
        name: _region_keypoints(gray, regions[name], sift)
        for name in REGION_NAMES
    }

    out = {}
    for name, features in region_features.items():
        out.update({f"sift_{name}_{key}": value for key, value in features.items()})

    for left, right, prefix in (
        ("face", "background", "face_bg_sift"),
        ("face", "border", "face_border_sift"),
        ("border", "background", "border_bg_sift"),
    ):
        out.update(region_contrasts(region_features[left], region_features[right], prefix))
    return out


def _relative_patch_geometry(regions) -> tuple[int, int]:
    x1, y1, x2, y2 = regions["bbox"]
    face_size = max(1, min(x2 - x1, y2 - y1))
    patch_size = int(np.clip(round(face_size * 0.08), 8, 32))
    stride = max(4, patch_size // 2)
    return patch_size, stride


def extract_patches(gray, mask, patch_size: int, stride: int, max_patches: int = 128):
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return np.empty((0, patch_size * patch_size), dtype=float), 0

    x_min, x_max = int(xs.min()), int(xs.max() + 1)
    y_min, y_max = int(ys.min()), int(ys.max() + 1)
    candidates = []
    for y in range(y_min, max(y_min + 1, y_max - patch_size + 1), stride):
        for x in range(x_min, max(x_min + 1, x_max - patch_size + 1), stride):
            patch_mask = mask[y : y + patch_size, x : x + patch_size]
            if patch_mask.shape == (patch_size, patch_size) and patch_mask.mean() >= 0.75:
                candidates.append((x, y))

    candidate_count = len(candidates)
    if candidate_count > max_patches:
        selected = np.linspace(0, candidate_count - 1, max_patches).round().astype(int)
        candidates = [candidates[index] for index in selected]

    patches = []
    for x, y in candidates:
        patch = gray[y : y + patch_size, x : x + patch_size].astype(float).reshape(-1)
        patch = (patch - patch.mean()) / (patch.std() + 1e-6)
        patches.append(patch)
    return np.asarray(patches), candidate_count


def _patch_region(gray, mask, patch_size: int, stride: int) -> dict:
    patches, candidate_count = extract_patches(gray, mask, patch_size, stride)
    base = {
        "sampled_patch_count": float(len(patches)),
        "candidate_patch_count": float(candidate_count),
        "sampling_coverage": float(len(patches) / candidate_count) if candidate_count else 0.0,
    }
    if len(patches) < 2:
        return {
            **base,
            "sim_mean": np.nan,
            "sim_std": np.nan,
            "sim_median": np.nan,
            "sim_p95": np.nan,
        }

    similarities = 1.0 - pdist(patches, metric="cosine")
    similarities = similarities[np.isfinite(similarities)]
    if similarities.size == 0:
        return {
            **base,
            "sim_mean": np.nan,
            "sim_std": np.nan,
            "sim_median": np.nan,
            "sim_p95": np.nan,
        }
    return {
        **base,
        "sim_mean": float(np.mean(similarities)),
        "sim_std": float(np.std(similarities)),
        "sim_median": float(np.median(similarities)),
        "sim_p95": float(np.percentile(similarities, 95)),
    }


def compute_patch_metrics(frame, regions) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    patch_size, stride = _relative_patch_geometry(regions)
    region_features = {
        name: _patch_region(gray, regions[name], patch_size, stride)
        for name in REGION_NAMES
    }

    out = {
        "qc_patch_size_px": float(patch_size),
        "qc_patch_stride_px": float(stride),
    }
    for name, features in region_features.items():
        for key, value in features.items():
            prefix = "patch" if key.startswith("sim_") else "qc_patch"
            out[f"{prefix}_{name}_{key}"] = value

    forensic_keys = ("sim_mean", "sim_std", "sim_median", "sim_p95")
    for left, right, prefix in (
        ("face", "background", "face_bg_patch"),
        ("face", "border", "face_border_patch"),
        ("border", "background", "border_bg_patch"),
    ):
        left_forensic = {key: region_features[left][key] for key in forensic_keys}
        right_forensic = {key: region_features[right][key] for key in forensic_keys}
        out.update(region_contrasts(left_forensic, right_forensic, prefix))
    return out


GROUP_B_FUNCTIONS = [compute_sift_metrics, compute_patch_metrics]
GROUP_B_PREFIXES = ("sift", "patch")
