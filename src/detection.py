"""
YOLO Object Detection Module for NightVisionAI
Uses Ultralytics YOLOv8 lightweight models for real-time detection of pedestrians and vehicles.
"""

import os
import cv2
import numpy as np
from .utils import logger, get_device

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# COCO Dataset Default Class Mapping for Standard Pretrained Models
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

class ObjectDetector:
    """
    Ultralytics YOLO Wrapper for Real-time Vehicle & Pedestrian Detection.
    """
    def __init__(self, config=None):
        config = config or {}
        det_cfg = config.get("detection", {})
        
        self.model_name = det_cfg.get("model_name", "yolov8n.pt")
        self.conf_threshold = float(det_cfg.get("conf_threshold", 0.35))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.45))
        self.requested_device = det_cfg.get("device", "auto")
        self.img_size = int(det_cfg.get("img_size", 640))
        self.target_class_names = det_cfg.get(
            "target_classes", ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        )
        
        self.device = get_device(self.requested_device)
        self.model = None
        self.target_class_ids = []
        
        self._load_model()

    def _load_model(self):
        """
        Loads pretrained YOLO weights into memory on specified device (CPU/CUDA).
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.error("Ultralytics library is not installed. Object detection disabled.")
            return

        try:
            logger.info(f"Loading YOLO model '{self.model_name}' on device '{self.device}'...")
            self.model = YOLO(self.model_name)
            
            # Map configured class names to numeric class IDs in loaded model
            names_dict = self.model.names if hasattr(self.model, 'names') else COCO_CLASSES
            
            self.target_class_ids = []
            for cid, cname in names_dict.items():
                if cname.lower() in [t.lower() for t in self.target_class_names]:
                    self.target_class_ids.append(int(cid))
                    
            logger.info(f"YOLO model initialized successfully. Filtered class IDs: {self.target_class_ids}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{self.model_name}': {e}")
            self.model = None

    def detect(self, frame_bgr):
        """
        Performs object detection on a single video frame.
        
        Returns:
            list of dicts: [
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int,
                    "class_name": str,
                    "confidence": float
                }, ...
            ]
        """
        if self.model is None or frame_bgr is None:
            return []

        try:
            # Run inference in lightweight mode
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                classes=self.target_class_ids if self.target_class_ids else None,
                verbose=False
            )
            
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                        
                        detections.append({
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf
                        })
            return detections
        except Exception as e:
            logger.error(f"Inference error during object detection: {e}")
            return []

    def set_confidence(self, new_conf):
        """
        Dynamically adjusts the confidence score cutoff threshold.
        """
        self.conf_threshold = max(0.01, min(1.0, float(new_conf)))
        logger.info(f"Detection confidence threshold updated to: {self.conf_threshold:.2f}")

    def get_supported_classes(self):
        """
        Returns list of class names supported by active YOLO model weights.
        """
        if self.model and hasattr(self.model, 'names'):
            return list(self.model.names.values())
        return list(COCO_CLASSES.values())
