# utils/detection_utils.py
import os, json
import numpy as np
import cv2
from ultralytics import YOLO
from typing import Tuple, List, Dict

MODEL_PATH = "models/best.pt"  # default; change if needed

def load_model(path: str = None):
    p = path or MODEL_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model weights not found: {p}")
    return YOLO(p)

# run inference and return structured output
def run_yolo_inference(model_or_path, img_bgr, conf=0.25, imgsz=640) -> Tuple[List[Dict], np.ndarray]:
    """
    model_or_path: YOLO() object or path to weights
    img_bgr: OpenCV BGR image
    returns:
       detections: list of dict {class_name, conf, bbox:[x1,y1,x2,y2]}
       annotated: annotated BGR image
    """
    # load model if given as path
    model = model_or_path if hasattr(model_or_path, "predict") else load_model(model_or_path)
    # ultralytics accepts BGR / numpy; safe to pass
    results = model.predict(source=img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    annotated = img_bgr.copy()
    detections = []
    for r in results:
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int)[0].tolist()  # [x1,y1,x2,y2]
                conf_s = float(box.conf.cpu().numpy()[0])
                cls_id = int(box.cls.cpu().numpy()[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                detections.append({"class_name": cls_name, "conf": conf_s, "bbox": xyxy})
                # draw
                x1,y1,x2,y2 = xyxy
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(annotated, f"{cls_name} {conf_s:.2f}", (x1, max(15,y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return detections, annotated

def generate_heatmap_from_detections(img_shape, detections, sigma=50):
    """
    Create a gaussian-accumulated heatmap from detection centers.
    img_shape: (H,W,...) or (H,W)
    detections: list with bbox key
    """
    H = img_shape[0]; W = img_shape[1]
    acc = np.zeros((H,W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for det in detections:
        x1,y1,x2,y2 = det["bbox"]
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        d2 = (xx - cx)**2 + (yy - cy)**2
        g = np.exp(-d2 / (2 * (sigma**2))) * det["conf"]
        acc += g
    if acc.max() > 0:
        acc = acc / acc.max()
    heatmap = (acc * 255).astype(np.uint8)
    return heatmap

def overlay_heatmap(img_bgr, heatmap, alpha=0.45):
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1-alpha, heatmap_color, alpha, 0)
    return overlay

def compute_severity_score(detections):
    """
    Example score: sum of confidences * sqrt(count)
    Normalized by a factor so typical scores are in 0-100
    """
    if not detections:
        return 0.0
    s = sum(d["conf"] for d in detections)
    score = s * np.sqrt(len(detections))
    # normalize using a heuristic constant
    norm = 1.0
    return float(score / norm)
