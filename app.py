"""
Streamlit Web Application Dashboard for NightVisionAI
Provides an interactive graphical browser interface to configure camera, enhancement,
and YOLO detection parameters while rendering live night monitoring feed and analytics.
"""

import os
import sys
import time
import cv2
import numpy as np
import streamlit as st

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, save_screenshot
from src.camera import CameraStream
from src.enhancement import LowLightEnhancer
from src.detection import ObjectDetector
from src.distance import DistanceCalculator
from src.monitoring import MonitoringHUD

st.set_page_config(
    page_title="NightVisionAI Dashboard",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #00E5FF;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #B0BEC5;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-title">🌙 NightVisionAI: Real-Time Road Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Low-Light Enhancement, YOLO Detection & Inter-Object Distance Calculator</div>', unsafe_allow_html=True)

    config = load_config()

    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    st.sidebar.header("📹 Camera Configuration")
    source_choice = st.sidebar.selectbox(
        "Camera Source Type",
        ["DroidCam Desktop App", "Webcam", "IP Stream / WiFi URL", "Synthetic (Test Mode)", "Prerecorded Video"],
        index=0
    )

    if source_choice == "DroidCam Desktop App":
        config["camera"]["source_type"] = "droidcam_app"
    elif source_choice == "Synthetic (Test Mode)":
        config["camera"]["source_type"] = "synthetic"
    elif source_choice == "Webcam":
        config["camera"]["source_type"] = "webcam"
        cam_idx = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=0)
        config["camera"]["camera_index"] = int(cam_idx)
    elif source_choice == "IP Stream / WiFi URL":
        config["camera"]["source_type"] = "ip_camera"
        stream_url = st.sidebar.text_input("Stream URL", value="http://10.110.18.126:4747/video")
        config["camera"]["stream_url"] = stream_url
    elif source_choice == "Prerecorded Video":
        config["camera"]["source_type"] = "video"
        v_path = st.sidebar.text_input("Video File Path", value="data/sample_night.mp4")
        config["camera"]["video_path"] = v_path

    st.sidebar.markdown("---")
    st.sidebar.header("✨ Low-Light Enhancement")
    enh_enabled = st.sidebar.checkbox("Enable Enhancement", value=True)
    config["enhancement"]["enabled"] = enh_enabled

    enh_method = st.sidebar.selectbox(
        "Enhancement Method",
        ["CLAHE (Adaptive Histogram)", "Gamma Correction", "Zero-DCE (Deep Neural Network)"],
        index=0
    )
    if "CLAHE" in enh_method:
        config["enhancement"]["method"] = "clahe"
        clip_lim = st.sidebar.slider("CLAHE Clip Limit", 1.0, 5.0, 2.5, 0.1)
        config["enhancement"]["clahe_clip_limit"] = clip_lim
    elif "Gamma" in enh_method:
        config["enhancement"]["method"] = "gamma"
        gamma_val = st.sidebar.slider("Gamma Factor", 0.5, 3.0, 1.8, 0.1)
        config["enhancement"]["gamma_value"] = gamma_val
    elif "Zero-DCE" in enh_method:
        config["enhancement"]["method"] = "zero_dce"

    st.sidebar.markdown("---")
    st.sidebar.header("🤖 YOLO Object Detection")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
    config["detection"]["conf_threshold"] = conf_thresh
    
    device_opt = st.sidebar.selectbox("Processing Hardware", ["Auto (CUDA/CPU)", "CPU Only"], index=0)
    config["detection"]["device"] = "cpu" if "CPU" in device_opt else "auto"

    st.sidebar.markdown("---")
    st.sidebar.header("📐 Distance & Proximity Calculator")
    dist_enabled = st.sidebar.checkbox("Enable Distance Calculator", value=True)
    dist_unit_choice = st.sidebar.selectbox(
        "Distance Measurement Unit",
        ["Meters (Real-World 3D)", "Pixels (Screen 2D)"],
        index=0
    )
    dist_unit = "meters" if "Meters" in dist_unit_choice else "pixels"
    
    danger_thresh = st.sidebar.slider(
        "Danger Threshold (m)" if dist_unit == "meters" else "Danger Threshold (px)",
        0.5 if dist_unit == "meters" else 20.0,
        5.0 if dist_unit == "meters" else 200.0,
        1.5 if dist_unit == "meters" else 100.0,
        0.1 if dist_unit == "meters" else 5.0
    )
    warning_thresh = st.sidebar.slider(
        "Warning Threshold (m)" if dist_unit == "meters" else "Warning Threshold (px)",
        1.0 if dist_unit == "meters" else 50.0,
        10.0 if dist_unit == "meters" else 400.0,
        3.0 if dist_unit == "meters" else 200.0,
        0.2 if dist_unit == "meters" else 10.0
    )
    
    config.setdefault("distance", {})["enabled"] = dist_enabled
    config["distance"]["unit"] = dist_unit
    config["distance"]["danger_threshold"] = danger_thresh
    config["distance"]["warning_threshold"] = warning_thresh

    sbs_mode = st.sidebar.checkbox("Side-by-Side Comparison Mode", value=False)
    config["display"]["side_by_side"] = sbs_mode

    # =========================================================================
    # MAIN LAYOUT METRICS & VIDEO STREAM CONTAINER
    # =========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    metric_fps = col1.metric("Live FPS", "0.0")
    metric_lat = col2.metric("Latency", "0 ms")
    metric_objs = col3.metric("Objects", "0")
    metric_dist = col4.metric("Closest Distance", "--")
    metric_enh = col5.metric("Enhancer", config["enhancement"]["method"].upper() if enh_enabled else "OFF")

    proximity_box = st.empty()
    video_placeholder = st.empty()
    status_box = st.empty()

    run_pipeline = st.button("🚀 Start Live Monitoring Feed", type="primary")

    if run_pipeline:
        camera = CameraStream(config)
        enhancer = LowLightEnhancer(config)
        detector = ObjectDetector(config)
        distance_calc = DistanceCalculator(config)
        hud = MonitoringHUD(config)

        if not camera.is_opened():
            status_box.error(f"Failed to open camera source. Details: {camera.error_message}")
            return

        status_box.success("Camera connected! Stream processing active...")
        stop_btn = st.button("⏹ Stop Stream")

        try:
            while not stop_btn:
                start_time = time.time()
                success, raw_frame = camera.read()
                
                if not success or raw_frame is None:
                    status_box.warning("End of stream or frame lost.")
                    break

                # Enhancement
                enhanced_frame = enhancer.enhance(raw_frame) if enhancer.enabled else raw_frame

                # Detection & Distance Render
                if sbs_mode:
                    raw_dets = detector.detect(raw_frame)
                    enh_dets = detector.detect(enhanced_frame)
                    
                    pairwise = distance_calc.calculate_pairwise_distances(enh_dets, enhanced_frame.shape) if distance_calc.enabled else []
                    alerts = distance_calc.get_proximity_alerts(pairwise)

                    display_bgr = hud.create_side_by_side(
                        raw_frame, raw_dets,
                        enhanced_frame, enh_dets,
                        enh_method=enhancer.method if enhancer.enabled else "OFF",
                        distance_calc=distance_calc
                    )
                    total_count = len(enh_dets)
                else:
                    dets = detector.detect(enhanced_frame)
                    
                    if distance_calc.enabled and len(dets) > 0:
                        distance_calc.compute_all_object_depths(dets, enhanced_frame.shape)
                        pairwise = distance_calc.calculate_pairwise_distances(dets, enhanced_frame.shape)
                        alerts = distance_calc.get_proximity_alerts(pairwise)
                    else:
                        pairwise = []
                        alerts = []

                    annotated, class_counts = hud.draw_detections(
                        enhanced_frame, dets,
                        show_distance=distance_calc.enabled
                    )
                    
                    if distance_calc.enabled and pairwise:
                        annotated = distance_calc.draw_distance_overlays(annotated, pairwise)

                    top_alert = alerts[0]["message"] if alerts else None
                    unit_sym = "m" if distance_calc.unit == "meters" else "px"

                    display_bgr = hud.draw_hud(
                        annotated, class_counts,
                        source_info=camera.source_type,
                        enh_method=enhancer.method if enhancer.enabled else "OFF",
                        device=detector.device,
                        dist_active=distance_calc.enabled,
                        dist_unit=unit_sym,
                        proximity_alert=top_alert
                    )
                    total_count = len(dets)

                # Convert BGR to RGB for Streamlit image renderer
                display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
                video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

                elapsed_ms = (time.time() - start_time) * 1000.0
                fps_val = hud.calculate_fps()

                # Update live metrics
                min_dist_str = f"{pairwise[0]['distance']}{pairwise[0]['unit']}" if pairwise else "N/A"
                metric_fps.metric("Live FPS", f"{fps_val:.1f}")
                metric_lat.metric("Latency", f"{elapsed_ms:.0f} ms")
                metric_objs.metric("Objects", f"{total_count}")
                metric_dist.metric("Closest Dist", min_dist_str)
                metric_enh.metric("Enhancer", enhancer.method.upper() if enhancer.enabled else "OFF")

                # Show proximity alert banner if danger detected
                if alerts:
                    top = alerts[0]
                    if top["severity"] == "CRITICAL":
                        proximity_box.error(f"🚨 {top['message']}")
                    else:
                        proximity_box.warning(f"⚠️ {top['message']}")
                else:
                    proximity_box.empty()

                time.sleep(0.01)

        except Exception as e:
            status_box.error(f"Stream interrupted: {e}")
        finally:
            camera.release()
            status_box.info("Camera stream closed.")

if __name__ == "__main__":
    main()
