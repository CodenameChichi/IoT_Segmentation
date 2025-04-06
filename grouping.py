import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import cv2
import os
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist

# CLIP model initialization
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


def get_clip_embedding(text):
    """클래스 이름 텍스트 -> CLIP 임베딩 벡터"""
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    return embeddings.squeeze().cpu().numpy()  # shape: (512,)


def cluster_by_clip_and_dbscan(filtered_objects, depth_weight=5.0, clip_weight=0.1, eps=1.5, min_samples=1):
    """
    CLIP + DBSCAN 기반 의미-거리 그룹화
    - filtered_objects: 필터링된 객체 리스트
    - depth_weight: depth 가중치 비율
    - eps, min_samples: DBSCAN 파라미터
    """

    # 1. get CLIP embedding
    for obj in filtered_objects:
        obj["clip_embed"] = get_clip_embedding(obj["class_name"])

    clip_matrix = np.array([obj["clip_embed"] for obj in filtered_objects]) * clip_weight  # (N, 512)
    
    if clip_matrix.ndim == 1: # class가 하나일 경우 error 방지
        clip_matrix = clip_matrix.reshape(-1, 1)

    semantic_sim = cosine_similarity(clip_matrix)
    semantic_dist = np.clip(1 - semantic_sim, 0, 2) # 의미적 거리

    xys = np.array([obj["center"] for obj in filtered_objects]) # x, y 정규화
    xys = (xys - xys.mean(axis=0)) / (xys.std(axis=0) + 1e-8)

    depths = np.array([[obj["depth"] * depth_weight] for obj in filtered_objects]) # depth 가중치 적용
    coords = np.concatenate([xys, depths], axis=1)
    spatial_dist = cdist(coords, coords) # 위치적 거리

    # transformer 사용을 위한 feature
    features = [np.concatenate([obj["clip_embed"], obj["center"], [obj["depth"]]]) for obj in filtered_objects]
    features = torch.tensor(features).unsqueeze(1).float()  # (N, 1, D)

    D = features.shape[-1]

    encoder_layer = TransformerEncoderLayer(d_model=D, nhead=5)
    transformer = TransformerEncoder(encoder_layer, num_layers=2)
    out = transformer(features)  # (N, 1, D)

    attention_dist = cosine_similarity(out.squeeze(1).detach().numpy()) # attention을 적용한 의미적 거리

    alpha, beta, gamma = 0.7, 0.3, 0.15
    hybrid_dist = alpha * spatial_dist + beta * semantic_dist + gamma * attention_dist # 의미적 거리 + 위치적 거리
    hybrid_dist = np.clip(hybrid_dist, 0, None)

    # 3. DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(hybrid_dist)
    labels = clustering.fit_predict(hybrid_dist)

    for i, obj in enumerate(filtered_objects):
        obj["cluster_id"] = labels[i]

    print(f"DBSCAN + CLIP clustering done: 총 {len(set(labels)) - (1 if -1 in labels else 0)} groups")
    return filtered_objects

def draw_clustered_objects(image, clustered_objects, save_path="output/clustered_result.jpg"):
    image_vis = image.copy()
    color_map = {}

    def get_color(cluster_id):
        if cluster_id not in color_map:
            color_map[cluster_id] = [random.randint(0, 255) for _ in range(3)]
        return color_map[cluster_id]

    for obj in clustered_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cls_name = obj["class_name"]
        cluster_id = obj["cluster_id"]
        color = get_color(cluster_id)
        label = f"[{cluster_id}] {cls_name}"

        cv2.rectangle(image_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image_vis)
    print(f"clustering visualization saved: {save_path}")