from ultralytics import YOLO
import cv2
import os
import numpy as np


def load_yolo(model_path='object/yolov8s.pt'):
    return YOLO(model_path)

def detect_objects(model, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, iou=0.5)
    return results

def visualize_detections(image, results, depth_map, class_names, save_path="output/result.jpg"):
    image_with_boxes = image.copy()

    print(f"# of object detected: {len(results[0].boxes)}")

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        obj_depth = depth_map[cy, cx]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls_id]} {obj_depth:.2f}"

        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        print(f"[{i}] Class: {class_names[cls_id]}, Conf: {conf:.2f}, Depth: {obj_depth:.2f}, BBox: ({x1}, {y1}) ~ ({x2}, {y2})")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image_with_boxes)
    print(f"Result image saved: {save_path}")

def filter_detections(results, depth_map, conf_thresh=0.5, depth_thresh=15.0, area_thresh=250, class_names=None):
    class_ids = [int(box.cls[0]) for box in results[0].boxes]
    unique_classes = set(class_ids)

    if len(unique_classes) <= 5:
        conf_thresh = 0.25

    filtered = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        area = (x2 - x1) * (y2 - y1)

        # depth calculation (median depth value in the bbox)
        # depth_map[cy, cx] -> center depth
        bbox_depth = depth_map[y1:y2, x1:x2]
        depth_val = np.median(bbox_depth)  

        if conf < conf_thresh:
            continue
        if depth_val > depth_thresh:
            continue
        if area < area_thresh:
            continue

        filtered.append({
            "bbox": (x1, y1, x2, y2),
            "cls_id": cls_id,
            "class_name": class_names[cls_id],
            "conf": conf,
            "depth": depth_val,
            "center": (cx, cy)
        })
    print(f"# of filtered objects: {len(filtered)}")
    return filtered


def draw_filtered_objects(image, filtered_objects, class_names, save_path="output/filtered_result.jpg"):
    image_vis = image.copy()

    for obj in filtered_objects:
        x1, y1, x2, y2 = obj["bbox"]
        cls_id = obj["cls_id"]
        depth = obj["depth"]

        label = f"{class_names[cls_id]} {depth:.2f}"

        cv2.rectangle(image_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image_vis)
    print(f"filtered result image saved: {save_path}")

