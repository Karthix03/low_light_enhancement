"""
Main Entry Point for NightVisionAI Standalone Desktop Application
Launches real-time night-time road monitoring with interactive keyboard controls.
"""

import sys
import os
import argparse
import cv2

from src.utils import load_config, logger
from src.pipeline import NightVisionPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="NightVisionAI Real-Time Night Road Monitoring System")
    parser.add_argument("--source", type=str, default=None,
                        help="Camera source type ('webcam', 'ip_camera', 'video', 'synthetic')")
    parser.add_argument("--index", type=int, default=None,
                        help="Camera device index for webcam mode (e.g. 0)")
    parser.add_argument("--url", type=str, default=None,
                        help="Stream URL for DroidCam / IP camera mode")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to prerecorded input video file")
    parser.add_argument("--method", type=str, default=None,
                        help="Enhancement method ('clahe', 'gamma', 'zero_dce', 'none')")
    parser.add_argument("--conf", type=float, default=None,
                        help="YOLO detection confidence threshold (0.0 - 1.0)")
    parser.add_argument("--device", type=str, default=None,
                        help="Processing hardware device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--side-by-side", action="store_true",
                        help="Enable split-screen comparison mode")
    parser.add_argument("--distance", action="store_true", default=None,
                        help="Enable inter-object distance calculator")
    parser.add_argument("--unit", type=str, choices=["meters", "pixels"], default=None,
                        help="Distance unit ('meters' or 'pixels')")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config()

    # Override config with CLI arguments if supplied
    if args.source:
        config["camera"]["source_type"] = args.source
    if args.index is not None:
        config["camera"]["camera_index"] = args.index
    if args.url:
        config["camera"]["stream_url"] = args.url
    if args.video:
        config["camera"]["video_path"] = args.video
    if args.method:
        config["enhancement"]["method"] = args.method
    if args.conf:
        config["detection"]["conf_threshold"] = args.conf
    if args.device:
        config["detection"]["device"] = args.device
    if args.side_by_side:
        config["display"]["side_by_side"] = True
    if args.distance is not None:
        config.setdefault("distance", {})["enabled"] = args.distance
    if args.unit:
        config.setdefault("distance", {})["unit"] = args.unit

    print("\n" + "=" * 65)
    print("      NIGHT VISION AI - REAL-TIME ROAD MONITORING SYSTEM       ")
    print("=" * 65)
    print("Controls:")
    print("  [E] Toggle Enhancement ON/OFF")
    print("  [M] Cycle Enhancement Method (CLAHE -> Gamma -> Zero-DCE -> None)")
    print("  [D] Toggle Distance Calculator ON/OFF")
    print("  [A] Toggle Collision Audio Alarm ON/OFF")
    print("  [U] Toggle Distance Unit (Meters <-> Pixels)")
    print("  [C] Toggle Side-by-Side Comparison Split Screen")
    print("  [S] Save Frame Screenshot to outputs/")
    print("  [+] / [-] Increase / Decrease Confidence Threshold")
    print("  [Q] or [ESC] Exit Safely")
    print("=" * 65 + "\n")

    pipeline = NightVisionPipeline(config)

    window_name = "NightVisionAI Live Monitoring"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280 if pipeline.side_by_side else 800, 600)

    try:
        while pipeline.is_running:
            display_frame, metrics = pipeline.step()

            if display_frame is None:
                err = metrics.get("error", "Unknown camera error.")
                logger.error(f"Stream error: {err}")
                print("\n[CAMERA ERROR] Unable to read frame from active source.")
                print("Falling back to synthetic camera generator mode...")
                pipeline.camera.source_type = "synthetic"
                pipeline.camera.start()
                continue

            cv2.imshow(window_name, display_frame)

            # Poll for keypresses (10ms timeout)
            key = cv2.waitKey(10) & 0xFF
            
            if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                logger.info("User requested application shutdown.")
                break
            elif key in [ord('e'), ord('E')]:
                pipeline.toggle_enhancement()
            elif key in [ord('m'), ord('M')]:
                method = pipeline.cycle_enhancement_method()
                print(f"[HUD] Active Enhancement Method: {method.upper()}")
            elif key in [ord('d'), ord('D')]:
                dist_state = pipeline.toggle_distance_calc()
                print(f"[HUD] Distance Calculator: {'ENABLED' if dist_state else 'DISABLED'}")
            elif key in [ord('a'), ord('A')]:
                audio_state = pipeline.toggle_audio()
                print(f"[HUD] Collision Audio Alarm: {'ENABLED' if audio_state else 'MUTED'}")
            elif key in [ord('u'), ord('U')]:
                unit = pipeline.toggle_distance_unit()
                print(f"[HUD] Distance Measurement Unit: {unit.upper()}")
            elif key in [ord('c'), ord('C')]:
                sbs = pipeline.toggle_side_by_side()
                cv2.resizeWindow(window_name, 1280 if sbs else 800, 600)
            elif key in [ord('s'), ord('S')]:
                saved_path = pipeline.capture_screenshot(display_frame)
                if saved_path:
                    print(f"[HUD] Screenshot saved to: {saved_path}")
            elif key in [ord('+'), ord('=')]:
                pipeline.detector.set_confidence(pipeline.detector.conf_threshold + 0.05)
            elif key in [ord('-'), ord('_')]:
                pipeline.detector.set_confidence(pipeline.detector.conf_threshold - 0.05)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        pipeline.release()
        cv2.destroyAllWindows()
        print("\n[SYSTEM] NightVisionAI shutdown complete.")

if __name__ == "__main__":
    main()
