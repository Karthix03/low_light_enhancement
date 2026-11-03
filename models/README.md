# Models Directory

This directory stores pretrained machine learning model weights.

## Models Overview

### 1. YOLOv8 Nano (`yolov8n.pt`)
* **Framework:** Ultralytics PyTorch / ONNX
* **Description:** Pretrained object detection model trained on the COCO dataset (80 object classes).
* **Target Classes Filtered:** `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`.
* **Automatic Download:** If missing, Ultralytics downloads `yolov8n.pt` automatically on the first run into the root directory or this folder.

### 2. Zero-DCE (`models/zero_dce.pth`)
* **Framework:** PyTorch (`src/enhancement.py`)
* **Description:** Zero-Reference Deep Curve Estimation neural network for zero-shot image illumination enhancement.
* **Fallback behavior:** If `.pth` weights are absent, the application initializes curve parameter maps or falls back seamlessly to Contrast Limited Adaptive Histogram Equalization (CLAHE).

## Adding Custom Trained Weights
If you train a custom YOLO model (e.g. for Indian traffic detecting auto-rickshaws, e-rickshaws, or specific local vehicles):
1. Place your `.pt` file here (e.g., `models/custom_indian_vehicles.pt`).
2. Update `config.yaml`:
   ```yaml
   detection:
     model_name: "models/custom_indian_vehicles.pt"
   ```
