"""
Utility Functions for NightVisionAI
Handles YAML configuration loading, logging, device detection, screenshot capture,
and synthetic frame generation for testing.
"""

import os
import time
import logging
import yaml
import numpy as np
import cv2

def setup_logger(name="NightVisionAI", log_level=logging.INFO):
    """
    Configures and returns a structured logger for the project.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

def load_config(config_path="config.yaml"):
    """
    Loads project settings from a YAML configuration file.
    Falls back to safe default settings if file is missing or invalid.
    """
    default_config = {
        "camera": {
            "source_type": "webcam",
            "camera_index": 0,
            "stream_url": "http://192.168.1.50:4747/video",
            "video_path": "data/sample_night.mp4",
            "frame_width": 640,
            "frame_height": 480,
            "fps": 30
        },
        "enhancement": {
            "enabled": True,
            "method": "clahe",
            "gamma_value": 1.8,
            "clahe_clip_limit": 2.5,
            "clahe_tile_grid": [8, 8],
            "weights_path": "models/zero_dce.pth"
        },
        "detection": {
            "model_name": "yolov8n.pt",
            "conf_threshold": 0.35,
            "iou_threshold": 0.45,
            "device": "auto",
            "img_size": 640,
            "target_classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        },
        "display": {
            "show_fps": True,
            "show_counts": True,
            "show_labels": True,
            "side_by_side": False,
            "output_dir": "outputs"
        },
        "distance": {
            "enabled": True,
            "unit": "meters",
            "focal_length": 700.0,
            "danger_threshold": 1.5,
            "warning_threshold": 3.0,
            "max_pairs": 6,
            "show_camera_distance": True,
            "real_heights": {
                "person": 1.70,
                "bicycle": 1.05,
                "car": 1.50,
                "motorcycle": 1.20,
                "bus": 3.20,
                "truck": 3.00,
                "default": 1.50
            }
        }
    }

    if not os.path.exists(config_path):
        logger.warning(f"Config file '{config_path}' not found. Using default configurations.")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
            if not user_config:
                return default_config
            
            # Deep merge defaults with user_config
            for section, values in default_config.items():
                if section not in user_config:
                    user_config[section] = values
                elif isinstance(values, dict):
                    for k, v in values.items():
                        user_config[section].setdefault(k, v)
            return user_config
    except Exception as e:
        logger.error(f"Error loading '{config_path}': {e}. Falling back to default settings.")
        return default_config

def get_device(requested_device="auto"):
    """
    Determines whether to use GPU (CUDA) or CPU for model execution.
    """
    try:
        import torch
        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        else:
            return "cpu"
    except ImportError:
        return "cpu"

def save_screenshot(frame, output_dir="outputs"):
    """
    Saves an annotated live frame to disk with a timestamped filename.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"night_vision_capture_{timestamp}.png")
    success = cv2.imwrite(filepath, frame)
    if success:
        logger.info(f"Saved screenshot: {filepath}")
        return filepath
    else:
        logger.error(f"Failed to save screenshot to {filepath}")
        return None

def generate_synthetic_night_frame(width=640, height=480):
    """
    Generates a low-light dark synthetic image frame with low contrast
    for offline testing and automated test suites.
    """
    # Create dark background (low illumination values around 15-40)
    frame = np.random.normal(loc=25, scale=10, size=(height, width, 3)).astype(np.uint8)
    
    # Draw faint vehicle silhouette (dark gray box)
    cv2.rectangle(frame, (width // 4, height // 2), (width // 2, height * 4 // 5), (45, 45, 45), -1)
    
    # Draw faint headlights (small dim yellow circles)
    cv2.circle(frame, (width // 4 + 20, height * 4 // 5 - 20), 8, (100, 120, 140), -1)
    cv2.circle(frame, (width // 2 - 20, height * 4 // 5 - 20), 8, (100, 120, 140), -1)
    
    # Draw faint pedestrian outline
    cv2.circle(frame, (width * 3 // 4, height // 2), 12, (35, 35, 35), -1)
    cv2.rectangle(frame, (width * 3 // 4 - 10, height // 2 + 12), (width * 3 // 4 + 10, height * 3 // 4), (30, 30, 30), -1)
    
    return frame
