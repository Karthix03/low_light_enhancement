"""
Train YOLOv8 Object Detection Model on Sample Night Traffic Dataset
Fine-tunes yolov8n.pt on night-time road traffic imagery using CUDA GPU acceleration.
Saves weights to models/trained_night_yolov8.pt and updates config.yaml.
"""

import os
import sys
import shutil
import argparse
import yaml
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
DATASET_YAML = os.path.join(BASE_DIR, "datasets", "sample_night_traffic", "data.yaml")
OUTPUT_WEIGHTS = os.path.join(BASE_DIR, "models", "trained_night_yolov8.pt")

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Night Traffic Dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', '0', 'cpu'")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Pretrained base model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n" + "=" * 65)
    print("      NIGHTVISION AI - YOLOv8 MODEL TRAINING PIPELINE")
    print("=" * 65)
    
    # Check dataset
    if not os.path.exists(DATASET_YAML):
        print(f"[ERROR] Dataset configuration not found at {DATASET_YAML}")
        print("Please run 'python scripts/prepare_sample_dataset.py' first.")
        sys.exit(1)

    # Determine hardware device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f"Base Weights: {args.base_model}")
    print(f"Dataset YAML: {DATASET_YAML}")
    print(f"Hyperparameters: Epochs={args.epochs}, Batch={args.batch}, ImgSize={args.imgsz}")
    print("=" * 65 + "\n")

    # Load base YOLO model
    print("[1/4] Loading base YOLOv8 model...")
    base_model_path = os.path.join(BASE_DIR, args.base_model) if not os.path.isabs(args.base_model) else args.base_model
    model = YOLO(base_model_path)

    # Train model
    print("\n[2/4] Starting training loop...")
    results = model.train(
        data=DATASET_YAML,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=os.path.join(BASE_DIR, "models", "runs"),
        name="train_night_yolo",
        exist_ok=True,
        save=True,
        verbose=True,
        workers=2
    )

    # Validate trained model
    print("\n[3/4] Evaluating trained model on validation split...")
    metrics = model.val()
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    print("\n" + "-" * 50)
    print("  VALIDATION PERFORMANCE METRICS")
    print("-" * 50)
    print(f"  Precision (P):       {mp * 100:.2f}%")
    print(f"  Recall (R):          {mr * 100:.2f}%")
    print(f"  mAP @ 0.50:          {map50 * 100:.2f}%")
    print(f"  mAP @ 0.50:0.95:     {map50_95 * 100:.2f}%")
    print("-" * 50 + "\n")

    # Save best weights
    best_weights_path = os.path.join(BASE_DIR, "models", "runs", "train_night_yolo", "weights", "best.pt")
    if os.path.exists(best_weights_path):
        shutil.copy2(best_weights_path, OUTPUT_WEIGHTS)
        print(f"[4/4] Successfully saved optimized model weights to:")
        print(f"      -> {OUTPUT_WEIGHTS}")
    else:
        # Fallback to last.pt if best.pt wasn't written
        last_weights_path = os.path.join(BASE_DIR, "models", "runs", "train_night_yolo", "weights", "last.pt")
        if os.path.exists(last_weights_path):
            shutil.copy2(last_weights_path, OUTPUT_WEIGHTS)
            print(f"[4/4] Saved last checkpoint weights to -> {OUTPUT_WEIGHTS}")

    # Update config.yaml with relative path to the trained model
    rel_weights_path = os.path.relpath(OUTPUT_WEIGHTS, BASE_DIR).replace(os.sep, "/")
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        
        cfg.setdefault("detection", {})["model_name"] = rel_weights_path
        
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            
        print(f"\n[CONFIG] Updated config.yaml -> detection.model_name: '{rel_weights_path}'")
    except Exception as e:
        print(f"[WARNING] Could not automatically update config.yaml: {e}")

    print("\nTraining and integration completed successfully!")

if __name__ == "__main__":
    main()

