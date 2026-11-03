# System Architecture & Technical Specifications

**Project Title:** Low-Light Image Enhancement and Real-Time Multi-Class Vehicle & Pedestrian Detection Using Deep Learning

---

## 1. High-Level Data Flow Architecture

```
+-----------------------------------------------------------------------+
|                            INPUT MODULE                               |
|  [Android Phone Camera / DroidCam] OR [Webcam] OR [Prerecorded Video] |
+-----------------------------------+-----------------------------------+
                                    |
                                    v (Raw BGR Frame)
+-----------------------------------+-----------------------------------+
|                     LOW-LIGHT ENHANCEMENT MODULE                      |
|       Selectable: [CLAHE] | [Gamma Correction] | [Zero-DCE Neural]    |
+-----------------------------------+-----------------------------------+
                                    |
                                    v (Enhanced High-Contrast Frame)
+-----------------------------------+-----------------------------------+
|                       YOLO OBJECT DETECTOR                            |
|    Ultralytics YOLOv8 (Class Filtering: Pedestrian, Cars, Buses, etc) |
+-----------------------------------+-----------------------------------+
                                    |
                                    v (Bounding Boxes, Conf Scores, Counts)
+-----------------------------------+-----------------------------------+
|                     MONITORING & HUD OVERLAY                          |
|    Renders Bounding Boxes, Live Counters, FPS, Translucent Telemetry |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                        DISPLAY INTERFACE                              |
|       [OpenCV Standalone Window]  OR  [Streamlit Web Dashboard]       |
+-----------------------------------------------------------------------+
```

---

## 2. Detailed Component Breakdown

### A. Camera Input (`src/camera.py`)
- **Purpose:** Abstract video capture across different devices.
- **Backends:**
  1. `DirectShow` (`cv2.CAP_DSHOW`) for low-latency Windows webcams and DroidCam virtual desktop camera driver.
  2. `HTTP Stream` reader for IP cameras and DroidCam WiFi connection (`http://<PHONE_IP>:4747/video`).
  3. `VideoCapture` for local `.mp4` video files.
  4. `Synthetic Generator` for headless testing when physical cameras are unplugged.

### B. Low-Light Image Enhancement (`src/enhancement.py`)
Enhances dark road scenes **before** passing images to the object detection model.
1. **Gamma Correction:** Non-linear contrast curve mapping:
   $$\text{Output}(x,y) = 255 \times \left(\frac{\text{Input}(x,y)}{255}\right)^{1/\gamma}$$
   Implemented using a pre-computed 256-element lookup table (LUT) for sub-millisecond per-frame throughput.
2. **LAB-CLAHE:** Converts BGR to LAB color space, applies Contrast Limited Adaptive Histogram Equalization to the luminance ($L$) channel with grid size $8 \times 8$ and clip limit $2.5$, then recombines into BGR. This brightens shadow details without distorting color tint.
3. **Zero-DCE (Zero-Reference Deep Curve Estimation):** 7-layer convolutional PyTorch neural network estimating pixel-wise higher-order illumination curves.

### C. Object Detection (`src/detection.py`)
- **Engine:** Ultralytics YOLOv8 Nano (`yolov8n.pt`).
- **Processing Resolution:** Configurable $640 \times 640$ pixels.
- **Hardware Acceleration:** Auto-selects NVIDIA CUDA GPU if PyTorch CUDA drivers are detected; defaults gracefully to CPU.
- **Supported Target Classes (COCO Dataset):**
  - `person` (Pedestrians)
  - `bicycle`
  - `car`
  - `motorcycle`
  - `bus`
  - `truck`

### D. Visualizer & Telemetry HUD (`src/monitoring.py`)
- Draws distinct color-coded bounding boxes per class.
- Computes exponential moving average live **FPS** and frame processing latency in milliseconds.
- Renders live object count metrics and class breakdowns.
- Supports **Side-by-Side Split Screen Comparison** (`Original Raw Frame` vs `Enhanced Frame`) for empirical evaluation.

---

## 3. Evaluation & Metrics Plan

When evaluating night vision performance:
* **Precision ($P$):** $\frac{TP}{TP + FP}$ (Fraction of valid detections out of all detected objects).
* **Recall ($R$):** $\frac{TP}{TP + FN}$ (Fraction of actual night objects detected).
* **F1-Score:** $2 \times \frac{P \times R}{P + R}$.
* **FPS:** Benchmark processing speed (Frames Per Second) across methods (Raw vs Gamma vs CLAHE vs Zero-DCE).
