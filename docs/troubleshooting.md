# Troubleshooting & Camera Setup Guide

This guide provides step-by-step solutions for common issues, Android phone camera setup, and PyTorch execution on Windows 11.

---

## 1. How to Connect your Android Phone as a Live Camera

### Method A: DroidCam over WiFi (Easiest)
1. Install **DroidCam** from Google Play Store on your Android phone.
2. Connect your phone and Windows 11 laptop to the **same WiFi network**.
3. Open the DroidCam app on your phone. Note the **WiFi IP** (e.g., `192.168.1.50`) and **Port** (default `4747`).
4. Update `config.yaml`:
   ```yaml
   camera:
     source_type: "ip_camera"
     stream_url: "http://192.168.1.50:4747/video"
   ```
5. Run the application:
   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

### Method B: DroidCam Client over USB (Low Latency)
1. Install **DroidCam Client** for Windows on your laptop.
2. Enable **USB Debugging** on Android:
   - Go to `Settings` -> `About Phone` -> Tap `Build Number` 7 times to enable Developer Options.
   - Go to `Settings` -> `Developer Options` -> Enable `USB Debugging`.
3. Connect phone to laptop via USB cable.
4. Launch DroidCam Client on Windows, select the USB option, and click **Start**.
5. DroidCam creates a virtual webcam on your laptop (usually Camera Index `0` or `1`).
6. Update `config.yaml`:
   ```yaml
   camera:
     source_type: "webcam"
     camera_index: 0
   ```

---

## 2. Common Issues & Fixes

### Q1: PowerShell script execution blocked (`Execution_Policies`)
If PowerShell blocks virtual environment script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
*(This changes execution policy safely only for the current PowerShell session without modifying system security globally).*

### Q2: "Unable to open Webcam at index 0"
* Ensure no other application (Zoom, Skype, Teams, Windows Camera App) is currently using the camera.
* Run `.\.venv\Scripts\python.exe scripts/test_camera.py` to inspect available camera indices on your system.

### Q3: "PyTorch CUDA not available"
* If you have an NVIDIA GPU and want GPU acceleration, install the CUDA-enabled PyTorch wheel:
  ```powershell
  .\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
* If you don't have an NVIDIA GPU, the project runs on CPU.

### Q4: Frame rate (FPS) is low
* Lower the processing frame resolution in `config.yaml`:
  ```yaml
  camera:
    frame_width: 480
    frame_height: 360
  ```
* Turn off Side-by-Side comparison mode (press `C` key during execution) as processing two frames doubles inference workload.
