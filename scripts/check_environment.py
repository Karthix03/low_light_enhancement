"""
Environment Diagnostic Script for NightVisionAI
Checks Python version, PyTorch, CUDA GPU support, OpenCV, and Ultralytics YOLO.
"""

import sys
import os
import platform

def run_diagnostics():
    print("=" * 60)
    print("         NightVisionAI - System Diagnostics Report         ")
    print("=" * 60)
    
    # 1. System & Python Info
    print(f"\n[1] Python Environment:")
    print(f"    - Python Version : {sys.version.split()[0]}")
    print(f"    - Executable Path: {sys.executable}")
    print(f"    - Operating Sys  : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    
    # 2. PyTorch & GPU acceleration
    print(f"\n[2] PyTorch & Hardware Acceleration:")
    try:
        import torch
        print(f"    - PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"    - CUDA Available : {cuda_available}")
        if cuda_available:
            print(f"    - GPU Device Name: {torch.cuda.get_device_name(0)}")
            print(f"    - Device Count   : {torch.cuda.device_count()}")
        else:
            print("    - Running Mode   : CPU (PyTorch CUDA not detected or CPU wheel installed)")
    except ImportError as e:
        print(f"    [ERROR] PyTorch not installed properly: {e}")
        
    # 3. OpenCV
    print(f"\n[3] OpenCV Computer Vision Library:")
    try:
        import cv2
        print(f"    - OpenCV Version : {cv2.__version__}")
    except ImportError as e:
        print(f"    [ERROR] OpenCV not installed: {e}")
        
    # 4. Ultralytics YOLO
    print(f"\n[4] Ultralytics YOLO Object Detection:")
    try:
        import ultralytics
        print(f"    - Ultralytics    : {ultralytics.__version__}")
    except ImportError as e:
        print(f"    [ERROR] Ultralytics YOLO not installed: {e}")
        
    # 5. Folder structure check
    print(f"\n[5] Directory Structure Check:")
    required_dirs = ['src', 'tests', 'scripts', 'data', 'models', 'outputs', 'docs']
    for d in required_dirs:
        exists = os.path.exists(d)
        status = "EXISTS" if exists else "MISSING (Will be created)"
        print(f"    - Directory '{d}/': {status}")
        if not exists:
            os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 60)
    print("Diagnostics completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()
