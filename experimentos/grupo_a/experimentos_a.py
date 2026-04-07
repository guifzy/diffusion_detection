import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
from itertools import combinations
from pathlib import Path

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

    h = 20 * len(texts) + 10
    w = 350

    # retângulo escuro
    cv2.rectangle(overlay, (x-5, y-20), (x+w, y+h), (0,0,0), -1)

    # transparência
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    # texto
    for i, text in enumerate(texts):
        cv2.putText(
            img,
            text,
            (x, y + i*20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1,
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


def summarize_group_a_results(results):
    label_groups = {}

    for result in results:
        label_groups.setdefault(result["label"], []).append(result)

    summary = {}

    for label, label_results in label_groups.items():
        summary[label] = {
            "count": len(label_results),
            "lbp_score_median": float(np.median([result["family_scores"]["lbp"] for result in label_results])),
            "sobel_score_median": float(np.median([result["family_scores"]["sobel"] for result in label_results])),
            "laplacian_score_median": float(np.median([result["family_scores"]["laplacian"] for result in label_results])),
            "global_score_median": float(np.median([result["global_score"] for result in label_results])),
        }

    return summary

## Player
def play_video_lbp_image(sample, interval=40, show_players=True, max_frames=120):
    frames = load_video_frames(sample)
    _ = max_frames

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
    basic = (
        metrics.get("full_frame_lap_var", 0)
        + metrics.get("full_frame_lap_mean_abs", 0)
        + metrics.get("full_frame_lap_p90", 0)
    )

    spatial = (
        metrics.get("lap_region_std", 0)
        + metrics.get("lap_region_range", 0)
        + metrics.get("full_frame_lap_spatial_mean_std", 0)
        + metrics.get("full_frame_lap_spatial_std_std", 0)
    )

    consistency = (
        metrics.get("lap_region_consistency_mean", 0) * 2
        + metrics.get("lap_region_consistency_std", 0)
    )

    return 0.35 * basic + 0.35 * spatial + 0.30 * consistency


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

    # heterogeneidade entre regiões, mede a falta de coerência local, quanto mais heterogêneo, mais provável de ser fake
    heterogeneity = (
        metrics.get("sobel_region_std", 0) +
        metrics.get("sobel_region_range", 0)
    )

    # assimetria direcional entre regiões, mede a falta de coerência local, quanto mais assimétrico, mais provável de ser fake
    asymmetry = (
        metrics.get("sobel_tb_coherence_diff", 0) +
        metrics.get("sobel_lr_coherence_diff", 0)
    )

    # inconsistência angular entre regiões, mede a falta de coerência global, quanto mais inconsistente, mais provável de ser fake
    angular = (
        metrics.get("sobel_angle_consistency_mean", 0) * 2 +
        metrics.get("sobel_angle_consistency_std", 0)
    )

    # score final ponderado
    score = (
        0.4 * heterogeneity +
        0.3 * asymmetry +
        0.3 * angular
    )

    return score
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
def play_video_sobel_image(sample, interval=40, show_players=True, max_frames=500):
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

                "",

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


# Grupo A: agregacao e benchmark
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
    "full_frame_grad_spatial_mean_std",
    "full_frame_grad_spatial_std_std",
    "sobel_region_std",
    "sobel_region_range",
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
        0.20 * _bounded_positive(summary.get("full_frame_grad_mean", 0.0))
        + 0.25 * _bounded_positive(summary.get("full_frame_grad_std", 0.0))
        + 0.30 * _bounded_positive(summary.get("full_frame_grad_p90", 0.0))
        + 0.25 * _bounded_positive(summary.get("full_frame_grad_dir_entropy", 0.0))
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

    score = 0.25 * basic + 0.30 * spatial + 0.45 * coherence

    return {
        "score": float(score),
        "components": {
            "basic": float(basic),
            "spatial": float(spatial),
            "coherence": float(coherence),
        },
        "metric_contributions": {
            "full_frame_grad_mean": float(_bounded_positive(summary.get("full_frame_grad_mean", 0.0))),
            "full_frame_grad_std": float(_bounded_positive(summary.get("full_frame_grad_std", 0.0))),
            "full_frame_grad_p90": float(_bounded_positive(summary.get("full_frame_grad_p90", 0.0))),
            "full_frame_grad_dir_entropy": float(_bounded_positive(summary.get("full_frame_grad_dir_entropy", 0.0))),
            "full_frame_grad_dir_coherence": float(_bounded_positive(summary.get("full_frame_grad_dir_coherence", 0.0))),
            "full_frame_grad_spatial_mean_std": float(_bounded_positive(summary.get("full_frame_grad_spatial_mean_std", 0.0))),
            "full_frame_grad_spatial_std_std": float(_bounded_positive(summary.get("full_frame_grad_spatial_std_std", 0.0))),
            "sobel_region_std": float(_bounded_positive(summary.get("sobel_region_std", 0.0))),
            "sobel_region_range": float(_bounded_positive(summary.get("sobel_region_range", 0.0))),
            "sobel_tb_coherence_diff": float(_bounded_positive(summary.get("sobel_tb_coherence_diff", 0.0))),
            "sobel_lr_coherence_diff": float(_bounded_positive(summary.get("sobel_lr_coherence_diff", 0.0))),
            "sobel_angle_consistency_mean": float(_bounded_positive(summary.get("sobel_angle_consistency_mean", 0.0))),
            "sobel_angle_consistency_std": float(_bounded_positive(summary.get("sobel_angle_consistency_std", 0.0))),
        },
    }


def score_laplacian_summary(summary):
    basic = (
        0.40 * _inverse_log(summary.get("full_frame_lap_var", 0.0))
        + 0.35 * _inverse_log(summary.get("full_frame_lap_mean_abs", 0.0))
        + 0.25 * _inverse_log(summary.get("full_frame_lap_p90", 0.0))
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
            "full_frame_lap_var": float(_inverse_log(summary.get("full_frame_lap_var", 0.0))),
            "full_frame_lap_mean_abs": float(_inverse_log(summary.get("full_frame_lap_mean_abs", 0.0))),
            "full_frame_lap_p90": float(_inverse_log(summary.get("full_frame_lap_p90", 0.0))),
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
    },
    "sobel": {
        "frame_fn": compute_sobel_frame,
        "agg_metrics": SOBEL_AGG_METRICS,
        "score_fn": score_sobel_summary,
    },
    "laplacian": {
        "frame_fn": compute_laplacian_frame,
        "agg_metrics": LAPLACIAN_AGG_METRICS,
        "score_fn": score_laplacian_summary,
    },
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


def classify_group_a_video(family_scores):
    suspicious_threshold = 0.60
    coherent_threshold = 0.35
    coherent_spread_threshold = 0.15

    score_values = np.array(list(family_scores.values()), dtype=float)

    if len(score_values) == 0:
        return "inconclusivo", []

    suspicious_families = [
        family_name
        for family_name, score in family_scores.items()
        if score >= suspicious_threshold
    ]

    coherent_families = [
        family_name
        for family_name, score in family_scores.items()
        if score <= coherent_threshold
    ]

    spread = float(np.max(score_values) - np.min(score_values))
    global_score = float(np.mean(score_values))

    if len(suspicious_families) >= 2:
        return "suspeito", suspicious_families

    if (
        len(coherent_families) == len(family_scores)
        and spread <= coherent_spread_threshold
        and global_score <= coherent_threshold
    ):
        return "coerente", []

    return "inconclusivo", suspicious_families


def collect_family_outputs(sample, frame_fn, include_previews=False):
    frames = load_video_frames(sample)

    return collect_family_outputs_from_frames(
        frames,
        frame_fn,
        include_previews=include_previews,
    )


def collect_family_outputs_from_frames(frames, frame_fn, include_previews=False):
    if len(frames) == 0:
        return {
            "frame_metrics": [],
            "preview_frames": [] if include_previews else None,
        }

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


def collect_group_a_video(sample, include_previews=False):
    frames = load_video_frames(sample)

    family_outputs = {}

    for family_name, config in FAMILY_PIPELINE_CONFIG.items():
        outputs = collect_family_outputs_from_frames(
            frames,
            config["frame_fn"],
            include_previews=include_previews,
        )
        family_outputs[family_name] = {
            "summary": aggregate_metric_series(outputs["frame_metrics"], config["agg_metrics"]),
            "frame_metrics": outputs["frame_metrics"],
            "preview_frames": outputs["preview_frames"],
        }

    return {"sample": sample, **family_outputs}


def score_group_a_record(record):
    family_scores = {}
    family_scored_outputs = {}

    for family_name, config in FAMILY_PIPELINE_CONFIG.items():
        scored = config["score_fn"](record[family_name]["summary"])
        family_scores[family_name] = scored["score"]
        family_scored_outputs[family_name] = {
            "summary": record[family_name]["summary"],
            "score": scored["score"],
            "components": scored["components"],
            "metric_contributions": scored["metric_contributions"],
            "preview_frames": record[family_name].get("preview_frames"),
        }

    verdict, suspicious_families = classify_group_a_video(family_scores)
    global_score = float(np.mean(list(family_scores.values()))) if len(family_scores) else 0.0

    scored_record = {
        "sample": record["sample"],
        "verdict": verdict,
        "suspicious_families": suspicious_families,
        "global_score": global_score,
        "family_scores": family_scores,
    }

    scored_record.update(family_scored_outputs)
    return scored_record


def _resolve_sample_rows(metadata, n_rows):
    if n_rows is None:
        return metadata

    return metadata.head(n_rows)


def _score_row_video(row, label, include_previews):
    sample_path = row["video_path"]
    filename = row.get("Filename")

    if not Path(sample_path).exists():
        return None, {"label": label, "filename": filename, "video_path": sample_path}

    record = collect_group_a_video(sample_path, include_previews=include_previews)
    scored_record = score_group_a_record(record)
    scored_record.update({"label": label, "filename": filename})

    return scored_record, None


def _run_rows_benchmark(rows, label, include_previews):
    results = []
    skipped = []

    for _, row in rows.iterrows():
        scored_record, skipped_record = _score_row_video(
            row,
            label=label,
            include_previews=include_previews,
        )

        if skipped_record is not None:
            skipped.append(skipped_record)
            continue

        results.append(scored_record)

    return results, skipped


def run_group_a_benchmark(metadata_true_df, metadata_fake_df, n_true=5, n_fake=5, max_frames=120, include_previews=False):
    _ = max_frames
    true_rows = _resolve_sample_rows(metadata_true_df, n_true)
    fake_rows = _resolve_sample_rows(metadata_fake_df, n_fake)

    true_results, true_skipped = _run_rows_benchmark(
        true_rows,
        label="Real",
        include_previews=include_previews,
    )
    fake_results, fake_skipped = _run_rows_benchmark(
        fake_rows,
        label="Fake",
        include_previews=include_previews,
    )

    results = true_results + fake_results
    skipped = true_skipped + fake_skipped

    return {
        "results": results,
        "summary": summarize_group_a_results(results),
        "skipped": skipped,
    }