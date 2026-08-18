from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from src.shared.features.common import aggregate_video_metrics, prepare_annotated_region_contexts, region_contrasts
from src.shared.features.extractor import aggregate_video_region_features
from src.shared.features.group_a import (
    compute_laplacian_metrics,
    compute_lbp_metrics,
    compute_sobel_metrics,
)
from src.shared.features.group_b import compute_sift_metrics
from src.shared.features.group_c import compute_residual_metrics
from src.shared.features.group_d import compute_fft_metrics
from src.shared.features.group_e import compute_photometry_metrics, compute_shadow_metrics
from src.shared.video import create_face_regions, metadata_for_frame
from src.data_engineering.preprocessing.metadata import _assign_track_id


def _frame_and_regions(size: int = 128):
    x = np.linspace(0, 255, size, dtype=np.uint8)
    gray = np.tile(x, (size, 1))
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    regions = create_face_regions(frame, [32, 32, 96, 96])
    return frame, regions


def test_region_contrasts_preserve_direction_magnitude_and_scale() -> None:
    values = region_contrasts({"energy": 6.0}, {"energy": 2.0}, "face_bg")

    assert values["face_bg_energy_signed_diff"] == 4.0
    assert values["face_bg_energy_abs_diff"] == 4.0
    assert np.isclose(values["face_bg_energy_norm_diff"], 0.5)


def test_metadata_lookup_uses_persisted_frame_ids() -> None:
    metadata = [
        {"frame_id": 0, "bbox": [0, 0, 10, 10]},
        {"frame_id": 20, "bbox": [1, 1, 11, 11]},
        {"frame_id": 90, "bbox": [2, 2, 12, 12]},
    ]

    exact, exact_index = metadata_for_frame(20, 100, metadata)
    nearest, nearest_index = metadata_for_frame(80, 100, metadata)

    assert exact["frame_id"] == 20
    assert exact_index == 1
    assert nearest["frame_id"] == 90
    assert nearest_index == 2


def test_video_aggregation_excludes_quality_control_columns() -> None:
    frame_metrics = pd.DataFrame(
        {
            "patch_face_sim_mean": [0.1, 0.3],
            "qc_patch_face_sampled_patch_count": [10.0, 12.0],
        }
    )

    aggregated = aggregate_video_metrics(frame_metrics, ("patch",))

    assert "patch_face_sim_mean_mean" in aggregated
    assert not any(key.startswith("qc_") for key in aggregated)


def test_annotated_regions_expand_to_region_contexts() -> None:
    frame, _regions = _frame_and_regions()
    metadata = {
        "regions": [
            {
                "region": "rosto_completo_1",
                "region_id": "rosto_completo_1",
                "region_label": "rosto completo 1",
                "region_type": "rosto_completo",
                "track_id": "face_1",
                "bbox": [32, 32, 96, 96],
                "polygon": [[32, 32], [96, 32], [96, 96], [32, 96]],
                "source": "mediapipe_face_landmarker",
            },
            {
                "region": "olhos_1",
                "region_id": "olhos_1",
                "region_label": "olhos 1",
                "region_type": "olhos",
                "track_id": "face_1",
                "bbox": [42, 45, 86, 60],
                "polygon": [[42, 45], [86, 45], [86, 60], [42, 60]],
                "source": "mediapipe_face_landmarker",
            },
            {
                "region": "fundo",
                "region_id": "fundo",
                "region_label": "fundo",
                "region_type": "fundo",
                "track_id": "global",
                "bbox": [0, 0, 128, 128],
                "source": "computed_background",
            },
        ]
    }

    _frame_std, contexts = prepare_annotated_region_contexts(frame, metadata)

    assert [context["region"] for context in contexts] == ["rosto_completo_1", "olhos_1", "fundo"]
    assert all(context["regions"]["face"].sum() > 0 for context in contexts)


def test_region_aggregation_keeps_one_row_per_video_region() -> None:
    frame_metrics = pd.DataFrame(
        {
            "video_id": ["video_01", "video_01", "video_01", "video_01"],
            "frame_id": [0, 1, 0, 1],
            "metadata_idx": [0, 1, 0, 1],
            "region": ["rosto_completo_1", "rosto_completo_1", "fundo", "fundo"],
            "region_id": ["rosto_completo_1", "rosto_completo_1", "fundo", "fundo"],
            "region_label": ["rosto completo 1", "rosto completo 1", "fundo", "fundo"],
            "region_type": ["rosto_completo", "rosto_completo", "fundo", "fundo"],
            "track_id": ["face_1", "face_1", "global", "global"],
            "label": ["Fake", "Fake", "Fake", "Fake"],
            "lbp_r1_p8_face_uniformity": [0.20, 0.30, 0.10, 0.14],
        }
    )

    aggregated = aggregate_video_region_features(frame_metrics, groups="a", video_id="video_01", label="Fake")

    assert set(aggregated["region"]) == {"rosto_completo_1", "fundo"}
    assert set(aggregated["video_id"]) == {"video_01"}
    assert "lbp_r1_p8_face_uniformity_mean" in aggregated.columns


def test_track_assignment_does_not_reuse_id_within_same_frame() -> None:
    tracks = {}
    used = set()

    first = _assign_track_id([10, 10, 40, 40], tracks, frame_id=0, used_track_ids=used)
    second = _assign_track_id([14, 14, 44, 44], tracks, frame_id=0, used_track_ids=used)

    assert first == 1
    assert second == 2
    assert used == {1, 2}


def test_lbp_uses_histogram_features_at_multiple_scales() -> None:
    frame, regions = _frame_and_regions()
    features = compute_lbp_metrics(frame, regions)

    assert "lbp_r1_p8_face_entropy_norm" in features
    assert "lbp_r2_p16_face_entropy_norm" in features
    assert "lbp_r3_p24_face_entropy_norm" in features
    assert "lbp_r1_p8_face_mean" not in features
    assert 0.0 <= features["lbp_r1_p8_face_uniformity"] <= 1.0


def test_sobel_reports_true_orientation_statistics() -> None:
    frame, regions = _frame_and_regions()
    features = compute_sobel_metrics(frame, regions)

    assert features["sobel_face_orientation_coherence"] > 0.9
    assert "sobel_face_strong_gradient_ratio" in features
    assert "sobel_face_coherence" not in features


def test_laplacian_separates_signed_and_absolute_statistics() -> None:
    frame, regions = _frame_and_regions()
    features = compute_laplacian_metrics(frame, regions)

    assert "lap_face_signed_energy" in features
    assert "lap_face_abs_mean" in features
    assert features["lap_face_signed_energy"] >= 0.0
    assert features["lap_face_abs_mean"] >= 0.0


def test_sift_does_not_compare_unrelated_cross_region_descriptors() -> None:
    frame, regions = _frame_and_regions()
    features = compute_sift_metrics(frame, regions)

    assert "sift_face_kp_density" in features
    assert "sift_face_kp_coverage" in features
    assert not any(key.endswith("sift_desc_dist") for key in features)


def test_bilateral_residual_is_explicitly_a_residual_baseline() -> None:
    frame = np.full((128, 128, 3), 127, dtype=np.uint8)
    regions = create_face_regions(frame, [32, 32, 96, 96])
    features = compute_residual_metrics(frame, regions)

    assert np.isclose(features["residual_face_rms"], 0.0)
    assert "residual_face_variance" not in features
    assert not any(key.startswith("noise_") for key in features)


def test_fft_ratios_use_power_and_form_a_partition() -> None:
    frame, regions = _frame_and_regions()
    features = compute_fft_metrics(frame, regions)
    total = sum(
        features[f"fft_face_{band}_power_ratio"]
        for band in ("low", "mid", "high")
    )

    assert np.isclose(total, 1.0, atol=1e-6)
    assert "fft_face_spectral_flatness" in features
    assert "fft_face_mean_intensity" not in features


def test_photometry_does_not_force_shadow_highlight_percentages() -> None:
    frame = np.full((128, 128, 3), 127, dtype=np.uint8)
    regions = create_face_regions(frame, [32, 32, 96, 96])
    features = compute_photometry_metrics(frame, regions)

    assert features["photo_face_dark_pixel_ratio"] == 0.0
    assert features["photo_face_bright_pixel_ratio"] == 0.0
    assert "photo_face_shadow_ratio" not in features
    assert "photo_face_highlight_ratio" not in features


def test_shadow_candidates_are_based_on_linearized_retinex_illumination() -> None:
    uniform = np.full((128, 128, 3), 180, dtype=np.uint8)
    uniform_regions = create_face_regions(uniform, [16, 16, 112, 112])
    uniform_features = compute_shadow_metrics(uniform, uniform_regions)

    illuminated = uniform.copy()
    illuminated[:, :48] = 45
    illuminated_regions = create_face_regions(illuminated, [16, 16, 112, 112])
    shadow_features = compute_shadow_metrics(illuminated, illuminated_regions)

    assert uniform_features["shadow_face_candidate_ratio"] == 0.0
    assert shadow_features["shadow_face_candidate_ratio"] > 0.0
    assert shadow_features["shadow_face_candidate_depth_mean"] > 0.0
