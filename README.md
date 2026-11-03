# Low-Light Image Enhancement and Real-Time Multi-Class Vehicle & Pedestrian Detection Using Deep Learning

A real-time computer vision system built in Python for night-time road monitoring, low-light image enhancement, and multi-class vehicle & pedestrian detection.

---

## 1. Project Overview & Problem Statement

### Problem Statement
Monitoring road traffic and detecting pedestrians at night is challenging due to low illumination, high noise, poor contrast, and glare from streetlights or vehicle headlights. Standard object detection algorithms often fail or miss pedestrians and vehicles in low-light conditions.

### Proposed Solution
This system combines **low-light image enhancement** (Gamma Correction, LAB-CLAHE, Zero-DCE Neural Network) with **Ultralytics YOLOv8** real-time object detection.
Live video frames captured from a laptop webcam, Android phone camera (via DroidCam), or IP stream are first enhanced to reveal shadow details and increase contrast before running object detection.

---

## 2. Key Features

* **Multi-Source Camera Support:** Works with laptop webcams, Android phone cameras (DroidCam USB & WiFi IP streams), prerecorded video files, and synthetic test generators.
* **Low-Light Enhancement Pipeline:**
  * **Gamma Correction:** Fast non-linear exponential intensity correction using lookup tables (LUT).
  * **LAB-CLAHE:** Contrast Limited Adaptive Histogram Equalization applied to the luminance ($L$) channel in LAB color space to preserve natural colors.
  * **Zero-DCE (Deep Curve Estimation):** 7-layer PyTorch neural network estimating pixel-wise illumination curves.
* **Multi-Class Vehicle & Pedestrian Detection:** Detects `person`, `bicycle`, `car`, `motorcycle`, `bus`, and `truck` using pretrained YOLOv8.
* **Real-Time Telemetry HUD:** Displays live FPS, frame latency (ms), total detected objects, class breakdown counters, active camera source, and hardware device (`CPU` / `CUDA`).
* **Inter-Object Distance & Proximity Calculator:** Real-time 3D Euclidean distance estimation between detected objects (pedestrians, cars, bikes) using pinhole camera geometry, safety classification (`DANGER` < 1.5m, `WARNING` < 3.0m, `SAFE`), dynamic color-coded connecting lines, midpoint distance pills, and automated collision warning banners.
* **Side-by-Side Comparison Mode:** Enables split-screen comparison between raw low-light footage and enhanced detection frames with synchronous bounding boxes and distance markers.
* **Dual Interfaces:** Offers both an OpenCV desktop GUI (`main.py`) and an interactive web dashboard (`app.py` built with Streamlit).

---

## 3. Supported Classes

Initially, the pretrained YOLO model supports the following object classes from the COCO dataset:

| Class Name | Description | Color Tag in HUD |
|---|---|---|
| `person` | Pedestrians on or near the road | Spring Green |
| `bicycle` | Cyclists and non-motorized two-wheelers | Gold Yellow |
| `car` | Passenger cars, sedans, SUVs | Deep Sky Blue |
| `motorcycle` | Motorbikes and scooters | Vivid Orange |
| `bus` | Transit and school buses | Violet / Purple |
| `truck` | Commercial delivery and freight trucks | Cyan |

> **Extensibility Note:** Additional vehicle classes (such as auto-rickshaws, e-rickshaws, vans, and ambulances) can be added by loading custom-trained YOLO weights (`.pt`) as explained in Section 7.

---

## 4. Hardware & Software Requirements

* **Operating System:** Windows 11 (64-bit)
* **Python Version:** Python 3.11.x
* **Hardware:**
  * Minimum: Intel Core i5 or AMD Ryzen 5 CPU, 8GB RAM.
  * Recommended: NVIDIA GPU with CUDA support for high-FPS neural enhancement and detection.
* **Camera:** Integrated laptop webcam OR Android smartphone running the DroidCam app.

---

## 5. Windows 11 PowerShell Setup Instructions

Follow these exact steps in PowerShell to set up and run the project:

### Step 1: Open PowerShell in Project Directory
```powershell
cd "c:\Users\karth\OneDrive\Desktop\Project(Sem 7)"
```

### Step 2: (Optional) Set Execution Policy if Script Activation is Blocked
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Step 3: Create and Activate Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies
```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### Step 5: Run System Diagnostics & Environment Checks
```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py
```

### Step 6: Run Automated Unit Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

---

## 6. How to Run the Application

### Option A: Standalone Live OpenCV Desktop Application (Recommended)
Launch the main monitoring interface:
```powershell
.\.venv\Scripts\python.exe main.py
```

#### Keyboard Controls:
* **`E`**: Toggle low-light enhancement ON / OFF.
* **`M`**: Cycle enhancement method (`CLAHE` -> `Gamma` -> `Zero-DCE` -> `None`).
* **`D`**: Toggle Inter-Object Distance Calculator ON / OFF.
* **`U`**: Toggle Distance Measurement Unit (`Meters` <-> `Pixels`).
* **`C`**: Toggle Side-by-Side Split Screen comparison mode.
* **`S`**: Save an annotated frame screenshot to `outputs/`.
* **`+` / `-`**: Increase / Decrease confidence score cutoff.
* **`Q` or `ESC`**: Exit safely.

#### Command-Line Arguments:
```powershell
# Run with split-screen comparison mode enabled
.\.venv\Scripts\python.exe main.py --side-by-side

# Run with distance calculator enabled in meters or pixels
.\.venv\Scripts\python.exe main.py --distance --unit meters

# Run using DroidCam WiFi stream URL
.\.venv\Scripts\python.exe main.py --source ip_camera --url "http://192.168.1.50:4747/video"

# Run on a prerecorded night driving video
.\.venv\Scripts\python.exe main.py --source video --video "data/sample_night.mp4"
```

---

### Option B: Interactive Streamlit Web Dashboard
Launch the web interface:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Camera Setup Guide (Android Phone as Webcam)

### DroidCam WiFi Setup:
1. Download **DroidCam** from Google Play Store on your Android phone.
2. Connect your phone and Windows 11 laptop to the same WiFi.
3. Open DroidCam on your phone. Note the **WiFi IP** (e.g. `192.168.1.50`).
4. Edit `config.yaml`:
   ```yaml
   camera:
     source_type: "ip_camera"
     stream_url: "http://192.168.1.50:4747/video"
   ```
5. Run `python main.py`.

*For detailed DroidCam USB debugging instructions, refer to [`docs/troubleshooting.md`](docs/troubleshooting.md).*

---

## 8. Adding Custom Vehicle Classes (e.g. Auto-Rickshaws)

To detect non-standard vehicle categories:
1. Train a YOLOv8 model on custom labelled images (e.g. Indian driving dataset) containing auto-rickshaws, vans, or ambulances.
2. Save the trained weights to `models/custom_indian_traffic.pt`.
3. Update `config.yaml`:
   ```yaml
   detection:
     model_name: "models/custom_indian_traffic.pt"
     target_classes:
       - "person"
       - "car"
       - "auto_rickshaw"
       - "motorcycle"
       - "bus"
   ```

---

## 9. Evaluation & Performance Benchmarking

To measure whether enhancement improves detection accuracy:
1. **Prepare a Test Video or Dataset:** Place test night driving videos in `data/`.
2. **Run Side-by-Side Mode:** Run `python main.py --side-by-side`.
3. **Metrics Recorded:**
   * **Precision & Recall:** Ratio of true detected objects vs missed objects.
   * **FPS Impact:** Compare processing speed across raw vs CLAHE vs Zero-DCE.

---

## 10. Privacy, Safety & Disclaimers

> [!IMPORTANT]
> **Experimental Academic Project Disclaimer:**
> This system is built for educational, research, and experimental computer vision purposes. It is **not** certified for autonomous driving, life-safety critical road warning systems, or legal enforcement.

* **Local Processing:** All camera streams and video processing occur 100% locally on your machine. No video data is uploaded to cloud servers or third-party APIs.
* **Continuous Recording:** Continuous recording is disabled by default. Images are only saved when you explicitly press `S`.

---

## 11. Project Folder Structure

```
NightVisionAI/
├── app.py                     # Streamlit interactive web dashboard
├── main.py                    # Main standalone OpenCV execution script
├── config.yaml                # Configuration settings for camera, enhancement, YOLO, display
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore file
├── src/
│   ├── __init__.py
│   ├── camera.py              # Flexible camera loader (Webcam, DroidCam, Video file, Synthetic)
│   ├── enhancement.py         # Gamma, LAB-CLAHE, and Zero-DCE PyTorch algorithms
│   ├── detection.py           # YOLO object detector wrapper & COCO class filter
│   ├── distance.py            # Monocular 3D depth & inter-object distance calculator
│   ├── pipeline.py            # End-to-end processing pipeline & metrics
│   ├── monitoring.py         # Bounding boxes renderer, HUD overlay & side-by-side mode
│   └── utils.py               # Config loader, logger, device detector, synthetic generator
├── tests/
│   ├── test_camera.py         # Camera unit tests
│   ├── test_enhancement.py    # Enhancement algorithms unit tests
│   ├── test_detection.py      # Object detector unit tests
│   ├── test_distance.py       # Distance & proximity unit tests
│   └── test_pipeline.py       # Integration tests
├── scripts/
│   ├── check_environment.py   # Diagnostic script for system & PyTorch check
│   └── test_camera.py         # Camera connection test script
├── data/
│   ├── README.md              # Dataset & test video documentation
│   └── sample_night.jpg       # Synthetic low-light sample image
├── models/
│   └── README.md              # Model weights documentation
├── outputs/
│   └── README.md              # Screenshot storage folder
└── docs/
    ├── architecture.md        # Technical system architecture document
    └── troubleshooting.md    # DroidCam, PyTorch & Windows troubleshooting guide
```

---

## 12. License & Credits

Built with Python 3.11, OpenCV, PyTorch, Ultralytics YOLOv8, and Streamlit.
Developed as a B.Tech Information Technology AI/ML Computer Vision Project.
