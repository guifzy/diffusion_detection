import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
from itertools import combinations
from pathlib import Path
from typing import Callable, cast

from scipy.spatial.distance import cosine
from skimage.feature import local_binary_pattern

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return np.array(frames)

## Funcoes auxiliares
def draw_text_block(img, texts, x=10, y=20):
    overlay = img.copy()

    scale = max(0.45, min(0.75, img.shape[1] / 900.0))
    thickness = 1 if img.shape[1] < 900 else 2
    line_h = int(22 * scale) + 8

    text_sizes = [cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0] for text in texts]
    max_text_w = max((size[0] for size in text_sizes), default=220)
    w = min(max_text_w + 24, img.shape[1] - x - 10)
    h = line_h * len(texts) + 10

    cv2.rectangle(overlay, (x - 5, y - 20), (x + w, y + h), (0, 0, 0), -1)

    # transparência
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    # texto
    for i, text in enumerate(texts):
        cv2.putText(
            img,
            text,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

    return img


def _normalize_to_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))

    if max_value - min_value < 1e-6:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (array - min_value) * 255.0 / (max_value - min_value)
    return normalized.astype(np.uint8)


def spatial_entropy(lbp, grid=4, n_bins=10):
    h, w = lbp.shape
    h_step = h // grid
    w_step = w // grid

    entropies = []

    for i in range(grid):
        for j in range(grid):
            patch = lbp[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]

            hist, _ = np.histogram(patch.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype("float")
            hist /= hist.sum() + 1e-6

            entropies.append(-np.sum(hist * np.log(hist + 1e-6)))

    return np.mean(entropies), np.std(entropies)


def compare_region_histograms(hists):
    dists = []

    for i in range(len(hists)):
        for j in range(i + 1, len(hists)):
            dists.append(cosine(hists[i], hists[j]))

    return np.mean(dists), np.std(dists)


def compute_lbp_visual(gray):
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    lbp_norm = _normalize_to_uint8(lbp)
    lbp_color = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_TURBO)

    return lbp, lbp_color


def compute_lbp_image(gray):
    r = 1
    n_points = 8 * r
    n_bins = n_points + 2

    lbp = local_binary_pattern(gray, n_points, r, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6

    features = {}

    features["entropy"] = -np.sum(hist * np.log(hist + 1e-6))

    _, ent_std = spatial_entropy(lbp, grid=4, n_bins=n_bins)
    features["entropy_spatial_std"] = ent_std

    return features, hist


def compute_lbp_frame(frame):
    results = {}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))

    full_feats, _ = compute_lbp_image(gray)

    for k, v in full_feats.items():
        results[f"full_{k}"] = v

    h, w = gray.shape

    regions = {
        "top": gray[:h // 2, :],
        "bottom": gray[h // 2:, :],
        "left": gray[:, :w // 2],
        "right": gray[:, w // 2:],
    }

    region_hists = {}

    for name, region in regions.items():
        region = cv2.resize(region, (128, 128))
        _, hist = compute_lbp_image(region)
        region_hists[name] = hist

    mean_dist, std_dist = compare_region_histograms(list(region_hists.values()))

    results["lbp_region_distance_mean"] = mean_dist
    results["lbp_region_distance_std"] = std_dist

    _, lbp_map_full = compute_lbp_visual(gray)

    return results, lbp_map_full


def compute_lbp_metrics_video(sample):
    return compute_family_video_report(sample, "lbp")


## Player
def play_video_lbp_image(sample, interval=40, show_players=True):
    frames = load_video_frames(sample)

    if len(frames) == 0:
        raise ValueError("Nao ha frames para exibir.")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    _, lbp0 = compute_lbp_frame(frames[0])

    img = ax.imshow(lbp0)

    if show_players:
        def update(i):
            frame = frames[i].copy()

            metrics, lbp_map = compute_lbp_frame(frame)

            texts = [
                f"FULL Entropy: {metrics['full_entropy']:.3f}",
                f"FULL Entropia Espacial: {metrics['full_entropy_spatial_std']:.3f}",
                f"Distancia entre Regioes: {metrics['lbp_region_distance_mean']:.3f} / {metrics['lbp_region_distance_std']:.3f}",
            ]
            lbp_map = draw_text_block(lbp_map, texts)

            img.set_data(lbp_map)
            return [img]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=interval,
            blit=True
        )

        plt.close(fig)
        display(HTML(ani.to_jshtml()))

    return compute_lbp_metrics_video(sample)


# Laplacian
def compute_laplacian_forensic(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(lap)

    lap_norm = _normalize_to_uint8(lap_abs)
    lap_color = cv2.applyColorMap(lap_norm, cv2.COLORMAP_TURBO)

    return lap_abs, lap_color


def laplacian_spatial_features(lap_abs, grid=3):
    h, w = lap_abs.shape
    h_bins = np.linspace(0, h, grid + 1, dtype=int)
    w_bins = np.linspace(0, w, grid + 1, dtype=int)

    patch_means = []
    patch_stds = []

    for i in range(grid):
        for j in range(grid):
            patch = lap_abs[h_bins[i]:h_bins[i + 1], w_bins[j]:w_bins[j + 1]]
            patch_means.append(np.mean(patch))
            patch_stds.append(np.std(patch))

    patch_means = np.array(patch_means)
    patch_stds = np.array(patch_stds)

    return np.std(patch_means), np.std(patch_stds)


def projection_features_laplacian(lap_abs, grid=3):
    h, w = lap_abs.shape
    h_bins = np.linspace(0, h, grid + 1, dtype=int)
    w_bins = np.linspace(0, w, grid + 1, dtype=int)

    grid_vals = np.zeros((grid, grid))

    for i in range(grid):
        for j in range(grid):
            patch = lap_abs[h_bins[i]:h_bins[i + 1], w_bins[j]:w_bins[j + 1]]
            grid_vals[i, j] = np.mean(patch)

    row_means = np.mean(grid_vals, axis=1)
    col_means = np.mean(grid_vals, axis=0)

    return np.std(row_means), np.std(col_means)


def compute_laplacian_image(gray):
    lap_abs, _ = compute_laplacian_forensic(gray)

    features = {
        "lap_var": np.var(lap_abs),
        "lap_mean_abs": np.mean(lap_abs),
        "lap_p90": np.percentile(lap_abs, 90),
    }

    lap_spatial_mean_std, lap_spatial_std_std = laplacian_spatial_features(lap_abs, grid=3)
    features["lap_spatial_mean_std"] = lap_spatial_mean_std
    features["lap_spatial_std_std"] = lap_spatial_std_std

    lap_horizontal_var, lap_vertical_var = projection_features_laplacian(lap_abs, grid=3)
    features["lap_horizontal_var"] = lap_horizontal_var
    features["lap_vertical_var"] = lap_vertical_var

    hist, _ = np.histogram(lap_abs.ravel(), bins=12)
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6

    features["lap_entropy"] = -np.sum(hist * np.log(hist + 1e-6))

    return features, hist


def laplacian_region_heterogeneity(metrics):
    regions = ["full_frame", "upper_frame", "lower_frame", "left_frame", "right_frame"]

    values = [
        metrics[f"{r}_lap_var"]
        for r in regions
        if f"{r}_lap_var" in metrics
    ]

    if len(values) < 2:
        return {}

    values = np.array(values)

    return {
        "lap_region_std": np.std(values),
        "lap_region_range": np.max(values) - np.min(values),
    }


def laplacian_region_consistency(region_hists):
    dists = []

    for a, b in combinations(region_hists.keys(), 2):
        dists.append(cosine(region_hists[a], region_hists[b]))

    if len(dists) == 0:
        return {}

    dists = np.array(dists)

    return {
        "lap_region_consistency_mean": np.mean(dists),
        "lap_region_consistency_std": np.std(dists),
    }


def laplacian_score_frame(metrics):
    return float(score_laplacian_summary(metrics)["score"])


def compute_laplacian_frame(frame):
    results = {}
    region_hists = {}

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_full = cv2.resize(gray_full, (256, 256))

    h, w = gray_full.shape

    all_regions = {
        "full_frame": gray_full,
        "upper_frame": gray_full[:h // 2, :],
        "lower_frame": gray_full[h // 2:, :],
        "left_frame": gray_full[:, :w // 2],
        "right_frame": gray_full[:, w // 2:],
    }

    for region_name, gray in all_regions.items():
        if gray.shape[0] < 32 or gray.shape[1] < 32:
            continue

        features, hist = compute_laplacian_image(gray)

        for k, v in features.items():
            results[f"{region_name}_{k}"] = v

        region_hists[region_name] = hist

    results.update(laplacian_region_heterogeneity(results))
    results.update(laplacian_region_consistency(region_hists))

    results["laplacian_score"] = laplacian_score_frame(results)

    lap_abs_full, lap_color_full = compute_laplacian_forensic(gray_full)
    _ = lap_abs_full

    return results, lap_color_full


def compute_laplacian_metrics_video(sample):
    return compute_family_video_report(sample, "laplacian")


# Sobel
def compute_sobel_forensic(gray):
    # gradientes
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # magnitude do gradiente
    mag = cv2.magnitude(gx, gy)
    mag = mag / (np.mean(mag) + 1e-6) # normalização para evitar dependência da iluminação

    angle = cv2.phase(gx, gy, angleInDegrees=True)

    # normalização
    mag_norm = _normalize_to_uint8(mag)

    # colormap forte (visual forense)
    mag_color = cv2.applyColorMap(mag_norm, cv2.COLORMAP_TURBO)

    return mag, angle, mag_color

def sobel_spatial_features(mag, grid=3):
    h, w = mag.shape
    h_bins = np.linspace(0, h, grid+1, dtype=int)
    w_bins = np.linspace(0, w, grid+1, dtype=int)

    patch_means = []
    patch_stds = []

    for i in range(grid):
        for j in range(grid):
            patch = mag[h_bins[i]:h_bins[i+1], w_bins[j]:w_bins[j+1]]

            patch_means.append(np.mean(patch))
            patch_stds.append(np.std(patch))

    patch_means = np.array(patch_means)
    patch_stds = np.array(patch_stds)

    # variação entre regiões
    spatial_mean_std = np.std(patch_means)
    spatial_std_std = np.std(patch_stds)

    return spatial_mean_std, spatial_std_std

def directional_coherence(angle, grid=3):
    h, w = angle.shape
    h_bins = np.linspace(0, h, grid+1, dtype=int)
    w_bins = np.linspace(0, w, grid+1, dtype=int)

    region_vectors = []

    for i in range(grid):
        for j in range(grid):
            patch = angle[h_bins[i]:h_bins[i+1], w_bins[j]:w_bins[j+1]]

            radians = np.deg2rad(patch)

            mean_sin = np.mean(np.sin(radians))
            mean_cos = np.mean(np.cos(radians))

            region_vectors.append([mean_sin, mean_cos])

    region_vectors = np.array(region_vectors)

    std_sin = np.std(region_vectors[:, 0])
    std_cos = np.std(region_vectors[:, 1])

    return std_sin + std_cos

def projection_features(mag, grid=3):
    h, w = mag.shape
    h_bins = np.linspace(0, h, grid+1, dtype=int)
    w_bins = np.linspace(0, w, grid+1, dtype=int)

    grid_vals = np.zeros((grid, grid))

    for i in range(grid):
        for j in range(grid):
            patch = mag[h_bins[i]:h_bins[i+1], w_bins[j]:w_bins[j+1]]
            grid_vals[i, j] = np.mean(patch)

    # média por linha e coluna
    row_means = np.mean(grid_vals, axis=1)
    col_means = np.mean(grid_vals, axis=0)

    horizontal_var = np.std(row_means)
    vertical_var = np.std(col_means)

    return horizontal_var, vertical_var

def sobel_artifact_features(mag, gray, bins=24):
    p90 = float(np.percentile(mag, 90))
    p99 = float(np.percentile(mag, 99))

    # Perfect/artificial edges tend to over-populate the high-tail of gradient magnitude.
    edge_tail_ratio = p99 / (p90 + 1e-6)

    dark_threshold = float(np.percentile(gray, 25))
    dark_mask = gray <= dark_threshold
    shadow_edge_strength = float(np.mean(mag[dark_mask])) if np.any(dark_mask) else 0.0
    global_edge_strength = float(np.mean(mag)) + 1e-6
    shadow_edge_ratio = shadow_edge_strength / global_edge_strength

    h, w = mag.shape
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    center = mag[cy0:cy1, cx0:cx1].ravel()

    outer_mask = np.ones_like(mag, dtype=bool)
    outer_mask[cy0:cy1, cx0:cx1] = False
    background = mag[outer_mask].ravel()

    c_hist, _ = np.histogram(center, bins=bins, range=(0, float(np.max(mag) + 1e-6)))
    b_hist, _ = np.histogram(background, bins=bins, range=(0, float(np.max(mag) + 1e-6)))
    c_hist = c_hist.astype(np.float64)
    b_hist = b_hist.astype(np.float64)
    c_hist /= c_hist.sum() + 1e-6
    b_hist /= b_hist.sum() + 1e-6

    # Repeated skin-like gradient patterns in background increase this similarity.
    center_background_grad_sim = 1.0 - float(cosine(c_hist, b_hist))

    return {
        "edge_tail_ratio": edge_tail_ratio,
        "shadow_edge_ratio": shadow_edge_ratio,
        "center_background_grad_sim": center_background_grad_sim,
    }

def compute_sobel_image(gray):
    features = {}

    mag, angle, mag_color = compute_sobel_forensic(gray)

    # intensidade do gradiente
    features["grad_mean"] = np.mean(mag)
    features["grad_std"] = np.std(mag)
    features["grad_p90"] = np.percentile(mag, 90)

    # distribuição de direções
    angle_hist, _ = np.histogram(angle.ravel(), bins=12, range=(0, 360))
    angle_hist = angle_hist.astype("float")
    angle_hist /= angle_hist.sum() + 1e-6

    features["grad_dir_entropy"] = -np.sum(angle_hist * np.log(angle_hist + 1e-6))

    # coerência direcional entre regiões
    features["grad_dir_coherence"] = directional_coherence(angle, grid=3)

    # estatísticas espaciais
    spatial_mean_std, spatial_std_std = sobel_spatial_features(mag, grid=3)

    features["grad_spatial_mean_std"] = spatial_mean_std
    features["grad_spatial_std_std"] = spatial_std_std

    # estatísticas de projeção, estrutura global
    h_var, v_var = projection_features(mag, grid=3)

    features["grad_horizontal_var"] = h_var
    features["grad_vertical_var"] = v_var

    artifact_features = sobel_artifact_features(mag, gray)
    features.update(artifact_features)
    features["sobel_artifact_score"] = (
        0.35 * artifact_features["edge_tail_ratio"]
        + 0.35 * artifact_features["shadow_edge_ratio"]
        + 0.30 * artifact_features["center_background_grad_sim"]
    )

    return features, angle_hist, mag_color

def region_heterogeneity(metrics):
    regions = ["upper_frame", "lower_frame", "left_frame", "right_frame"]

    values = [
        metrics[f"{r}_grad_std"]
        for r in regions
        if f"{r}_grad_std" in metrics
    ]

    if len(values) < 2:
        return {}

    values = np.array(values)

    return {
        "sobel_region_std": np.std(values),              # dispersão global
        "sobel_region_range": np.max(values) - np.min(values)  # max-min (ranking)
    }

def directional_asymmetry(metrics):
    results = {}

    if "upper_frame_grad_dir_coherence" in metrics and "lower_frame_grad_dir_coherence" in metrics:
        results["sobel_tb_coherence_diff"] = abs(
            metrics["upper_frame_grad_dir_coherence"] -
            metrics["lower_frame_grad_dir_coherence"]
        )

    if "left_frame_grad_dir_coherence" in metrics and "right_frame_grad_dir_coherence" in metrics:
        results["sobel_lr_coherence_diff"] = abs(
            metrics["left_frame_grad_dir_coherence"] -
            metrics["right_frame_grad_dir_coherence"]
        )

    return results

def angular_consistency(region_angle_hists):
    dists = []

    for a, b in combinations(region_angle_hists.keys(), 2):
        dists.append(cosine(region_angle_hists[a], region_angle_hists[b]))

    if len(dists) == 0:
        return {}

    dists = np.array(dists)

    return {
        "sobel_angle_consistency_mean": np.mean(dists),
        "sobel_angle_consistency_std": np.std(dists),
    }

def sobel_score_frame(metrics):
    return float(score_sobel_summary(metrics)["score"])
def compute_sobel_frame(frame):

    results = {}
    region_angle_hists = {}

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray_full.shape

    # regiões: frame inteiro, metade superior, metade inferior, metade esquerda, metade direita
    all_regions = {
        "full_frame": gray_full,
        "upper_frame": gray_full[:gray_full.shape[0]//2, :],
        "lower_frame": gray_full[gray_full.shape[0]//2:, :],
        "left_frame": gray_full[:, :gray_full.shape[1]//2],
        "right_frame": gray_full[:, gray_full.shape[1]//2:]
    }

    # extrair features de cada região
    for region_name, gray in all_regions.items():

        if gray.shape[0] < 32 or gray.shape[1] < 32:
            continue

        features, angle_hist, _ = compute_sobel_image(gray)

        for k, v in features.items():
            results[f"{region_name}_{k}"] = v

        region_angle_hists[region_name] = angle_hist

    # mede a heterogeneidade entre as regiões, o quanto elas são diferentes em termos de textura, quanto maior, mais heterogêneo é o frame
    results.update(region_heterogeneity(results))
    # mede a assimetria direcional entre as regiões, o quanto as direções dos gradientes são diferentes entre as regiões opostas, quanto maior, mais assimétrico é o frame
    results.update(directional_asymmetry(results))
    # mede a consistência angular entre as regiões, o quanto as direções dos gradientes são similares entre as regiões, quanto menor, mais consistente é o frame
    results.update(angular_consistency(region_angle_hists))

    #score
    results["sobel_score"] = sobel_score_frame(results)

    # mapa visual do frame inteiro
    _, _, sobel_map_full = compute_sobel_forensic(gray_full)

    return results, sobel_map_full
def compute_sobel_scores(sample):
    return compute_family_video_report(sample, "sobel")
## Player
def play_video_sobel_image(sample, interval=40, show_players=True):
    frames = load_video_frames(sample)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")

    _, sobel0 = compute_sobel_frame(frames[0])

    img = ax.imshow(sobel0)

    if show_players:
        def update(i):
            frame = frames[i].copy()

            metrics, sobel_map = compute_sobel_frame(frame)

            # overlay de texto
            texts = [
                "[INTENSITY]",
                f"Mean: {metrics['full_frame_grad_mean']:.3f}",
                f"Std: {metrics['full_frame_grad_std']:.3f}",
                f"P90: {metrics['full_frame_grad_p90']:.3f}",

                "",

                "[DIRECTION]",
                f"Entropy: {metrics['full_frame_grad_dir_entropy']:.3f}",
                f"Coherence: {metrics['full_frame_grad_dir_coherence']:.3f}",

                "",

                "[SPATIAL]",
                f"Mean STD: {metrics['full_frame_grad_spatial_mean_std']:.3f}",
                f"STD STD: {metrics['full_frame_grad_spatial_std_std']:.3f}",

                "[STRUCTURE]",
                f"Horizontal-Var: {metrics['full_frame_grad_horizontal_var']:.3f}",
                f"Vertical-Var: {metrics['full_frame_grad_vertical_var']:.3f}",

                "",

                "[CONSISTENCY]",
                f"Angle Mean: {metrics.get('sobel_angle_consistency_mean', 0):.5f}",
                f"Angle Std: {metrics.get('sobel_angle_consistency_std', 0):.5f}",

                "[SCORE]",
                f"Frame Score: {metrics['sobel_score']:.3f}"
            ]
            sobel_map = draw_text_block(sobel_map, texts)

            img.set_data(sobel_map)
            return [img]

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=interval,
            blit=True
        )

        plt.close(fig)

        import matplotlib as mpl
        mpl.rcParams["animation.embed_limit"] = 500000000

        display(HTML(ani.to_jshtml()))

    return compute_sobel_scores(sample)


# Metadatas
df = pd.read_csv("C:\\Users\\Guilherme Monteiro\\Desktop\\TCC\\data\\video-metadata-publish-with-links.csv")
df['video_path'] = df['Filename'].apply(lambda x: "C:\\Users\\Guilherme Monteiro\\Desktop\\TCC\\data\\videos\\" + x)

metadata_true = df[df['Video Ground Truth'] == "Real"]
metadata_fake = df[df['Video Ground Truth'] == "Fake"]


# Grupo A: scoring e exportacao por dataframe de metadados
LBP_AGG_METRICS = [
    "full_entropy",
    "full_entropy_spatial_std",
    "lbp_region_distance_mean",
    "lbp_region_distance_std",
]

SOBEL_AGG_METRICS = [
    "full_frame_grad_mean",
    "full_frame_grad_std",
    "full_frame_grad_p90",
    "full_frame_grad_dir_entropy",
    "full_frame_grad_dir_coherence",
    "full_frame_edge_tail_ratio",
    "full_frame_shadow_edge_ratio",
    "full_frame_center_background_grad_sim",
    "full_frame_grad_spatial_mean_std",
    "full_frame_grad_spatial_std_std",
    "sobel_region_std",
    "sobel_region_range",
    "full_frame_sobel_artifact_score",
    "sobel_tb_coherence_diff",
    "sobel_lr_coherence_diff",
    "sobel_angle_consistency_mean",
    "sobel_angle_consistency_std",
]

LAPLACIAN_AGG_METRICS = [
    "full_frame_lap_var",
    "full_frame_lap_mean_abs",
    "full_frame_lap_p90",
    "full_frame_lap_entropy",
    "full_frame_lap_spatial_mean_std",
    "full_frame_lap_spatial_std_std",
    "lap_region_std",
    "lap_region_range",
    "lap_region_consistency_mean",
    "lap_region_consistency_std",
]


def _finite_values(values):
    return np.array([value for value in values if value is not None and np.isfinite(value)])


def aggregate_metric_series(frame_metrics, metric_names, reducer=np.median):
    summary = {}

    for metric_name in metric_names:
        values = _finite_values([metrics.get(metric_name) for metrics in frame_metrics])
        summary[metric_name] = float(reducer(values)) if len(values) else 0.0

    return summary


def _bounded_positive(value):
    value = max(float(value), 0.0)
    return 1.0 - np.exp(-value)


def _inverse_log(value):
    value = max(float(value), 0.0)
    return 1.0 / (1.0 + np.log1p(value))


def _bounded_log_positive(value):
    value = max(float(value), 0.0)
    return 1.0 - np.exp(-np.log1p(value))


def score_lbp_summary(summary):
    basic = (
        0.65 * _inverse_log(summary.get("full_entropy", 0.0))
        + 0.35 * _bounded_positive(summary.get("full_entropy_spatial_std", 0.0))
    )

    coherence = (
        0.70 * _bounded_positive(summary.get("lbp_region_distance_mean", 0.0))
        + 0.30 * _bounded_positive(summary.get("lbp_region_distance_std", 0.0))
    )

    score = 0.45 * basic + 0.55 * coherence

    return {
        "score": float(score),
        "components": {
            "basic": float(basic),
            "coherence": float(coherence),
        },
        "metric_contributions": {
            "full_entropy": float(_inverse_log(summary.get("full_entropy", 0.0))),
            "full_entropy_spatial_std": float(_bounded_positive(summary.get("full_entropy_spatial_std", 0.0))),
            "lbp_region_distance_mean": float(_bounded_positive(summary.get("lbp_region_distance_mean", 0.0))),
            "lbp_region_distance_std": float(_bounded_positive(summary.get("lbp_region_distance_std", 0.0))),
        },
    }


def score_sobel_summary(summary):
    basic = (
        0.30 * _bounded_positive(summary.get("full_frame_grad_std", 0.0))
        + 0.35 * _bounded_positive(summary.get("full_frame_grad_p90", 0.0))
        + 0.35 * _bounded_positive(summary.get("full_frame_grad_dir_entropy", 0.0))
    )

    spatial = (
        0.50 * _bounded_positive(summary.get("full_frame_grad_spatial_mean_std", 0.0))
        + 0.50 * _bounded_positive(summary.get("full_frame_grad_spatial_std_std", 0.0))
    )

    coherence = (
        0.25 * _bounded_positive(summary.get("sobel_region_std", 0.0))
        + 0.20 * _bounded_positive(summary.get("sobel_region_range", 0.0))
        + 0.20 * _bounded_positive(summary.get("sobel_tb_coherence_diff", 0.0))
        + 0.15 * _bounded_positive(summary.get("sobel_lr_coherence_diff", 0.0))
        + 0.10 * _bounded_positive(summary.get("full_frame_grad_dir_coherence", 0.0))
        + 0.08 * _bounded_positive(summary.get("sobel_angle_consistency_mean", 0.0))
        + 0.02 * _bounded_positive(summary.get("sobel_angle_consistency_std", 0.0))
    )

    artifact = (
        0.20 * _bounded_positive(summary.get("full_frame_sobel_artifact_score", 0.0))
        + 0.30 * _bounded_positive(summary.get("full_frame_edge_tail_ratio", 0.0))
        + 0.30 * _bounded_positive(summary.get("full_frame_shadow_edge_ratio", 0.0))
        + 0.20 * _bounded_positive(summary.get("full_frame_center_background_grad_sim", 0.0))
    )

    score = 0.20 * basic + 0.25 * spatial + 0.30 * coherence + 0.25 * artifact

    return {
        "score": float(score),
        "components": {
            "basic": float(basic),
            "spatial": float(spatial),
            "coherence": float(coherence),
            "artifact": float(artifact),
        },
        "metric_contributions": {
            "full_frame_grad_std": float(_bounded_positive(summary.get("full_frame_grad_std", 0.0))),
            "full_frame_grad_p90": float(_bounded_positive(summary.get("full_frame_grad_p90", 0.0))),
            "full_frame_grad_dir_entropy": float(_bounded_positive(summary.get("full_frame_grad_dir_entropy", 0.0))),
            "full_frame_grad_dir_coherence": float(_bounded_positive(summary.get("full_frame_grad_dir_coherence", 0.0))),
            "full_frame_grad_spatial_mean_std": float(_bounded_positive(summary.get("full_frame_grad_spatial_mean_std", 0.0))),
            "full_frame_grad_spatial_std_std": float(_bounded_positive(summary.get("full_frame_grad_spatial_std_std", 0.0))),
            "full_frame_edge_tail_ratio": float(_bounded_positive(summary.get("full_frame_edge_tail_ratio", 0.0))),
            "full_frame_shadow_edge_ratio": float(_bounded_positive(summary.get("full_frame_shadow_edge_ratio", 0.0))),
            "full_frame_center_background_grad_sim": float(_bounded_positive(summary.get("full_frame_center_background_grad_sim", 0.0))),
            "sobel_region_std": float(_bounded_positive(summary.get("sobel_region_std", 0.0))),
            "sobel_region_range": float(_bounded_positive(summary.get("sobel_region_range", 0.0))),
            "sobel_tb_coherence_diff": float(_bounded_positive(summary.get("sobel_tb_coherence_diff", 0.0))),
            "sobel_lr_coherence_diff": float(_bounded_positive(summary.get("sobel_lr_coherence_diff", 0.0))),
            "sobel_angle_consistency_mean": float(_bounded_positive(summary.get("sobel_angle_consistency_mean", 0.0))),
            "sobel_angle_consistency_std": float(_bounded_positive(summary.get("sobel_angle_consistency_std", 0.0))),
            "full_frame_sobel_artifact_score": float(_bounded_positive(summary.get("full_frame_sobel_artifact_score", 0.0))),
        },
    }


def score_laplacian_summary(summary):
    basic = (
        0.35 * _bounded_log_positive(summary.get("full_frame_lap_var", 0.0))
        + 0.30 * _bounded_log_positive(summary.get("full_frame_lap_mean_abs", 0.0))
        + 0.25 * _bounded_log_positive(summary.get("full_frame_lap_p90", 0.0))
        + 0.10 * _bounded_positive(summary.get("full_frame_lap_entropy", 0.0))
    )

    spatial = (
        0.50 * _bounded_positive(summary.get("full_frame_lap_spatial_mean_std", 0.0))
        + 0.50 * _bounded_positive(summary.get("full_frame_lap_spatial_std_std", 0.0))
    )

    coherence = (
        0.25 * _bounded_positive(summary.get("lap_region_std", 0.0))
        + 0.25 * _bounded_positive(summary.get("lap_region_range", 0.0))
        + 0.30 * _bounded_positive(summary.get("lap_region_consistency_mean", 0.0))
        + 0.20 * _bounded_positive(summary.get("lap_region_consistency_std", 0.0))
    )

    score = 0.30 * basic + 0.30 * spatial + 0.40 * coherence

    return {
        "score": float(score),
        "components": {
            "basic": float(basic),
            "spatial": float(spatial),
            "coherence": float(coherence),
        },
        "metric_contributions": {
            "full_frame_lap_var": float(_bounded_log_positive(summary.get("full_frame_lap_var", 0.0))),
            "full_frame_lap_mean_abs": float(_bounded_log_positive(summary.get("full_frame_lap_mean_abs", 0.0))),
            "full_frame_lap_p90": float(_bounded_log_positive(summary.get("full_frame_lap_p90", 0.0))),
            "full_frame_lap_entropy": float(_bounded_positive(summary.get("full_frame_lap_entropy", 0.0))),
            "full_frame_lap_spatial_mean_std": float(_bounded_positive(summary.get("full_frame_lap_spatial_mean_std", 0.0))),
            "full_frame_lap_spatial_std_std": float(_bounded_positive(summary.get("full_frame_lap_spatial_std_std", 0.0))),
            "lap_region_std": float(_bounded_positive(summary.get("lap_region_std", 0.0))),
            "lap_region_range": float(_bounded_positive(summary.get("lap_region_range", 0.0))),
            "lap_region_consistency_mean": float(_bounded_positive(summary.get("lap_region_consistency_mean", 0.0))),
            "lap_region_consistency_std": float(_bounded_positive(summary.get("lap_region_consistency_std", 0.0))),
        },
    }


FAMILY_PIPELINE_CONFIG = {
    "lbp": {
        "frame_fn": compute_lbp_frame,
        "agg_metrics": LBP_AGG_METRICS,
        "score_fn": score_lbp_summary,
        "important_metrics": [
            "full_entropy",
            "full_entropy_spatial_std",
            "lbp_region_distance_mean",
            "lbp_region_distance_std",
        ],
    },
    "sobel": {
        "frame_fn": compute_sobel_frame,
        "agg_metrics": SOBEL_AGG_METRICS,
        "score_fn": score_sobel_summary,
        "important_metrics": [
            "full_frame_grad_std",
            "full_frame_grad_p90",
            "full_frame_edge_tail_ratio",
            "full_frame_shadow_edge_ratio",
            "full_frame_sobel_artifact_score",
            "full_frame_grad_dir_entropy",
            "full_frame_center_background_grad_sim",
        ],
    },
    "laplacian": {
        "frame_fn": compute_laplacian_frame,
        "agg_metrics": LAPLACIAN_AGG_METRICS,
        "score_fn": score_laplacian_summary,
        "important_metrics": [
            "full_frame_lap_var",
            "full_frame_lap_p90",
            "lap_region_std",
            "lap_region_consistency_mean",
        ],
    },
}


def collect_family_outputs(sample, frame_fn, include_previews=False):
    frames = load_video_frames(sample)
    frame_metrics = []
    preview_frames = [] if include_previews else None

    for frame in frames:
        metrics, preview = frame_fn(frame)
        frame_metrics.append(metrics)

        if preview_frames is not None:
            preview_frames.append(preview)

    return {
        "frame_metrics": frame_metrics,
        "preview_frames": preview_frames,
    }


def compute_family_video_report(sample, family_name, include_previews=False):
    config = FAMILY_PIPELINE_CONFIG[family_name]
    outputs = collect_family_outputs(sample, config["frame_fn"], include_previews=include_previews)
    summary = aggregate_metric_series(outputs["frame_metrics"], config["agg_metrics"])
    scored = config["score_fn"](summary)

    return {
        "summary": summary,
        "score": scored["score"],
        "components": scored["components"],
        "metric_contributions": scored["metric_contributions"],
        "frame_metrics": outputs["frame_metrics"],
        "preview_frames": outputs["preview_frames"],
    }


def _safe_video_stem(filename):
    stem = Path(str(filename)).stem
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return clean or "video"


def _get_video_fps(video_path, default_fps=25.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or not np.isfinite(fps) or fps <= 0:
        return float(default_fps)

    return float(fps)


def _video_writer_fourcc():
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(fourcc_fn):
        typed_fourcc_fn = cast(Callable[..., int], fourcc_fn)
        return typed_fourcc_fn(*"mp4v")

    fallback_fourcc_fn = cast(Callable[..., int], cv2.VideoWriter.fourcc)
    return fallback_fourcc_fn(*"mp4v")


def _score_from_frame_metrics(family_name, frame_metrics):
    score_fn = FAMILY_PIPELINE_CONFIG[family_name]["score_fn"]
    return float(score_fn(frame_metrics)["score"])


def _resize_frame_for_export(frame, family_name):
    min_side_target = 512 if family_name in ("lbp", "laplacian") else 360
    h, w = frame.shape[:2]
    min_side = min(h, w)

    if min_side >= min_side_target:
        return frame

    scale = float(min_side_target) / float(min_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _build_overlay_lines(family_name, frame_metrics, label, filename, frame_idx):
    important_keys = FAMILY_PIPELINE_CONFIG[family_name]["important_metrics"]
    short_name = str(filename)
    if len(short_name) > 42:
        short_name = short_name[:39] + "..."

    lines = [
        f"Label: {label}",
        f"Video: {short_name}",
        f"Frame: {frame_idx}",
        f"Family: {family_name.upper()}",
        f"Score(frame): {_score_from_frame_metrics(family_name, frame_metrics):.4f}",
    ]

    for key in important_keys:
        lines.append(f"{key}: {frame_metrics.get(key, 0.0):.4f}")

    return lines


def _export_annotated_family_video(preview_frames, frame_metrics, family_name, label, filename, output_path, fps):
    if preview_frames is None or len(preview_frames) == 0:
        return False

    if frame_metrics is None or len(frame_metrics) == 0:
        return False

    n_frames = min(len(preview_frames), len(frame_metrics))
    if n_frames == 0:
        return False

    first = _resize_frame_for_export(preview_frames[0], family_name)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        _video_writer_fourcc(),
        fps,
        (width, height),
    )

    for idx in range(n_frames):
        frame = _resize_frame_for_export(preview_frames[idx], family_name)
        metrics = frame_metrics[idx]
        overlay_lines = _build_overlay_lines(family_name, metrics, label, filename, idx)
        rendered = draw_text_block(frame.copy(), overlay_lines)
        writer.write(rendered)

    writer.release()
    return True


def _select_rows_from_metadata(metadata_df, n_real, n_fake):
    real_rows = metadata_df[metadata_df["Video Ground Truth"] == "Real"].head(n_real)
    fake_rows = metadata_df[metadata_df["Video Ground Truth"] == "Fake"].head(n_fake)

    selected = []

    for _, row in real_rows.iterrows():
        selected.append((row, "Real"))

    for _, row in fake_rows.iterrows():
        selected.append((row, "Fake"))

    return selected


def process_group_a_from_metadata(metadata_df, n_real=3, n_fake=3, output_dir="output"):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for family_name in FAMILY_PIPELINE_CONFIG.keys():
        (output_root / family_name).mkdir(parents=True, exist_ok=True)

    results = []

    for row, label in _select_rows_from_metadata(metadata_df, n_real, n_fake):
        sample_path = row["video_path"]
        filename = row.get("Filename", Path(sample_path).name)

        if not Path(sample_path).exists():
            continue

        fps = _get_video_fps(sample_path)
        family_scores = {}
        output_videos = {}

        for family_name in FAMILY_PIPELINE_CONFIG.keys():
            report = compute_family_video_report(sample_path, family_name, include_previews=True)
            family_scores[family_name] = report["score"]

            out_name = f"{label.lower()}_{_safe_video_stem(filename)}_{family_name}.mp4"
            out_path = output_root / family_name / out_name

            exported = _export_annotated_family_video(
                report["preview_frames"],
                report["frame_metrics"],
                family_name,
                label,
                filename,
                out_path,
                fps,
            )

            if exported:
                output_videos[family_name] = str(out_path)

        results.append(
            {
                "label": label,
                "family_scores": family_scores,
                "output_videos": output_videos,
            }
        )

    return {
        "results": results,
        "output_dir": str(output_root),
    }

