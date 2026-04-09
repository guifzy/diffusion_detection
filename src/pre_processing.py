# Landmarks de face, contorno e fundo
import os
from typing import Any
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from retinaface import RetinaFace
import tensorflow as tf
import json
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets
from matplotlib import animation
from IPython.display import HTML, Video, display
import torch
import tqdm as tqdm

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU ativa: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("Rodando em CPU")
        return False

tf.get_logger().setLevel("ERROR")

def load_video_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()

    return np.array(frames)

def standardize_frame(frame, max_size=640):
    h, w = frame.shape[:2]

    scale = min(max_size / max(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)

    frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame_resized, scale

def detect_face(frame):
    detections = RetinaFace.detect_faces(frame)

    if len(detections) == 0:
        return None

    # pega a face com maior score
    face = list(detections.values())[0]

    bbox = face["facial_area"]
    landmarks = face["landmarks"]

    return {
        "bbox": bbox,
        "landmarks": landmarks,
        "score": face["score"]
    }

def align_face(frame, landmarks, output_size=256):

    left_eye = landmarks["left_eye"]
    right_eye = landmarks["right_eye"]

    # ângulo
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = (
        (left_eye[0] + right_eye[0]) // 2,
        (left_eye[1] + right_eye[1]) // 2
    )

    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    aligned = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    return aligned, M

def create_face_regions(frame, bbox, padding=0.2):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    # aplica padding
    px = int(bw * padding)
    py = int(bh * padding)

    x1p = max(0, x1 - px)
    y1p = max(0, y1 - py)
    x2p = min(w, x2 + px)
    y2p = min(h, y2 + py)

    # máscara face interna
    face_mask = np.zeros((h, w), dtype=np.uint8)
    face_mask[y1:y2, x1:x2] = 1

    # máscara expandida
    expanded_mask = np.zeros((h, w), dtype=np.uint8)
    expanded_mask[y1p:y2p, x1p:x2p] = 1

    # borda = expandida - interna
    border_mask = expanded_mask - face_mask

    # fundo = resto
    background_mask = 1 - expanded_mask

    return {
        "face": face_mask,
        "border": border_mask,
        "background": background_mask,
        "bbox_expanded": (x1p, y1p, x2p, y2p)
    }

def save_metadata(metadata, video_path):
    save_path = video_path.replace(".mp4", "_meta.json")

    # cria pasta 'metadata' se não existir
    if not os.path.exists("/home/guilherme_monteiro/projetos/tcc/data/metadata"):
        os.makedirs("/home/guilherme_monteiro/projetos/tcc/data/metadata")

    save_path = os.path.join("/home/guilherme_monteiro/projetos/tcc/data/metadata", os.path.basename(save_path))

    with open(save_path, "w") as f:
        json.dump(metadata, f)

    print(f"Salvo em: {save_path}")

## Processamento e salvamento de metadados
def draw_face_box(frame, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame

def overlay_regions(frame, regions):
    overlay = frame.copy()

    face_mask = regions["face"]
    border_mask = regions["border"]
    background_mask = regions["background"]

    overlay[face_mask == 1] = (0, 255, 0)       # verde
    overlay[border_mask == 1] = (0, 0, 255)     # vermelho
    overlay[background_mask == 1] = (255, 0, 0) # azul

    alpha = 0.3
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

def sanitize_bbox_xyxy(frame, bbox, min_size=2):
    h, w = frame.shape[:2]

    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    return [x1, y1, x2, y2]

def create_csrt_tracker():
    tracker_factory = getattr(cv2, "TrackerCSRT_create", None)
    if callable(tracker_factory):
        return tracker_factory()

    legacy = getattr(cv2, "legacy", None)
    if legacy is not None:
        legacy_factory = getattr(legacy, "TrackerCSRT_create", None)
        if callable(legacy_factory):
            return legacy_factory()

    return None

# Visualiza o vídeo com as caixas e regiões sobrepostas
def saving_metadata(video_path,  max_frames=2000):
    gpu = check_gpu()
    if gpu:
        print("Usando GPU para detecção")
    else: print("Usando CPU para detecção")

    print(f"\n Video: {os.path.basename(video_path)} \n")

    frames = load_video_frames(video_path)

    if len(frames) == 0:
        raise ValueError("Sem frames")

    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, int(max_frames)).astype(int)
        frames = frames[idx]

    detections = []
    regions_list = []
    metadata = []

    tracker: Any = None
    last_bbox = None

    DETECT_EVERY = 1

    for i, frame in enumerate(tqdm.tqdm(frames, desc="Processando frames")):

        frame_std, scale = standardize_frame(frame)

        use_detection = (i % DETECT_EVERY == 0) or (tracker is None)

        bbox_original = None

        # detecção a cada N frames ou se o tracker falhar
        if use_detection:
            det = detect_face(frame_std)

            if det is not None:
                bbox_scaled = det["bbox"]
                bbox_original = [int(x / scale) for x in bbox_scaled]
                bbox_original = sanitize_bbox_xyxy(frame, bbox_original)

                if bbox_original is not None:
                    # OpenCV tracker usa bbox como (x, y, w, h).
                    x1, y1, x2, y2 = bbox_original
                    tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
                    tracker = create_csrt_tracker()
                    if tracker is not None:
                        try:
                            tracker.init(frame, tracker_bbox)
                        except cv2.error:
                            tracker = None

                    if tracker is not None:
                        last_bbox = bbox_original

        # faz o tracking nos frames intermediários
        else:
            if tracker is None:
                bbox_original = None
            else:
                success, tracked_box = tracker.update(frame)

                if success:
                    x, y, w, h = map(int, tracked_box)
                    bbox_original = [x, y, x + w, y + h]
                    bbox_original = sanitize_bbox_xyxy(frame, bbox_original)
                    if bbox_original is not None:
                        last_bbox = bbox_original
                else:
                    bbox_original = None

        # fallbacks caso detecção e tracking falhem
        if bbox_original is None:

            if last_bbox is not None:
                bbox_original = last_bbox

            else:
                # fallback: centro da imagem
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                size = min(h, w) // 4

                bbox_original = [
                    cx - size, cy - size,
                    cx + size, cy + size
                ]

        bbox_original = sanitize_bbox_xyxy(frame, bbox_original)
        if bbox_original is None:
            continue

        # extraindo as regiões para visualização
        regions = create_face_regions(frame, bbox_original)

        detections.append(bbox_original)
        regions_list.append(regions)

        metadata.append({
            "frame_id": i,
            "bbox": bbox_original,
            "bbox_expanded": regions["bbox_expanded"],
            "source": "detector" if use_detection else "tracker"
        })

    print("\nProcessamento concluído")

    print("\nSalvando metadados...")
    save_metadata(metadata, video_path)
    print("Metadados salvos")

df = pd.read_csv("/home/guilherme_monteiro/projetos/tcc/data/metadata/video-metadata-publish-with-links.csv")
df['video_path'] = df['Filename'].apply(lambda x: "/home/guilherme_monteiro/projetos/tcc/data/videos/" + x)

#executando em todas as linhas do dataframe
for idx, row in df.iterrows():
    # verifica se ja tem metadados salvos para esse vídeo
    meta_path = row['video_path'].replace(".mp4", "_meta.json").replace("/videos/", "/metadata/")
    if os.path.exists(meta_path):
        print(f"Metadados já existem para {row['Filename']}, pulando...")
    else:
        saving_metadata(row['video_path'])

