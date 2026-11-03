"""
Interactive / Automated Camera Diagnostics Test Script for NightVisionAI
Checks webcam indices, IP stream connection, and synthetic camera generator.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import CameraStream, list_available_cameras
from src.utils import load_config, logger

def test_camera_module():
    print("=" * 60)
    print("            NightVisionAI - Camera Connection Test         ")
    print("=" * 60)
    
    config = load_config()
    
    # 1. Discover working camera indices
    print("\n[1] Probing system video indices...")
    available = list_available_cameras(max_tested=4)
    print(f"    Available Camera Devices: {available}")
    
    # 2. Test Synthetic Camera Mode
    print("\n[2] Testing Synthetic Camera Stream...")
    synthetic_config = {"camera": {"source_type": "synthetic", "frame_width": 640, "frame_height": 480}}
    cam_synth = CameraStream(synthetic_config)
    success, frame = cam_synth.read()
    if success and frame is not None:
        print(f"    [SUCCESS] Captured synthetic frame with shape: {frame.shape}")
    else:
        print("    [FAIL] Failed to capture synthetic frame.")
    cam_synth.release()

    # 3. Test Configured Camera Mode
    print(f"\n[3] Testing Configured Source Mode ({config['camera']['source_type']})...")
    cam = CameraStream(config)
    if cam.is_opened():
        success, frame = cam.read()
        if success and frame is not None:
            print(f"    [SUCCESS] Connected! Captured frame shape: {frame.shape}")
        else:
            print(f"    [WARNING] Connected, but could not grab frame: {cam.error_message}")
    else:
        print(f"    [INFO] Physical camera connection failed: {cam.error_message}")
        print("    [INFO] (This is expected if no physical camera or phone stream is currently active).")
    cam.release()

    print("\n" + "=" * 60)
    print("Camera diagnostics completed safely.")
    print("=" * 60)

if __name__ == "__main__":
    test_camera_module()
