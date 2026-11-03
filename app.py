"""
Streamlit Web Application Dashboard for NightVisionAI
Provides an interactive graphical browser interface to configure camera, enhancement,
and YOLO detection parameters while rendering live night monitoring feed and analytics.
Cloud-ready with Sample Video Demo, File Uploader, and Browser Device Camera support.
"""

import os
import sys
import time
import tempfile
import cv2
import numpy as np
from PIL import Image
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

def format_closest_distance(pairwise, dets):
    """Formats closest distance string: pairwise inter-object distance if pairs exist, else camera-to-object radial distance."""
    if pairwise:
        return f"{pairwise[0]['distance']}{pairwise[0]['unit']}"
    if dets:
        cam_dists = [d.get("distance_cam") for d in dets if d.get("distance_cam") is not None]
        if cam_dists:
            return f"{min(cam_dists):.2f}m (Cam)"
    return "--"

def process_single_frame(frame, enhancer, detector, distance_calc, hud, sbs_mode=False):
    """Processes a single BGR frame and returns annotated display frame, metrics, and alerts."""
    start_t = time.time()
    
    enhanced = enhancer.enhance(frame) if enhancer.enabled else frame
    
    if sbs_mode:
        raw_dets = detector.detect(frame)
        enh_dets = detector.detect(enhanced)
        pairwise = distance_calc.calculate_pairwise_distances(enh_dets, enhanced.shape) if distance_calc.enabled else []
        alerts = distance_calc.get_proximity_alerts(pairwise)
        display_bgr = hud.create_side_by_side(
            frame, raw_dets,
            enhanced, enh_dets,
            enh_method=enhancer.method if enhancer.enabled else "OFF",
            distance_calc=distance_calc
        )
        total_count = len(enh_dets)
        class_counts = {}
        for d in enh_dets:
            c = d["class_name"]
            class_counts[c] = class_counts.get(c, 0) + 1
    else:
        dets = detector.detect(enhanced)
        if distance_calc.enabled and len(dets) > 0:
            distance_calc.compute_all_object_depths(dets, enhanced.shape)
            pairwise = distance_calc.calculate_pairwise_distances(dets, enhanced.shape)
            alerts = distance_calc.get_proximity_alerts(pairwise)
        else:
            pairwise = []
            alerts = []

        annotated, class_counts = hud.draw_detections(
            enhanced, dets,
            show_distance=distance_calc.enabled
        )
        if distance_calc.enabled and pairwise:
            annotated = distance_calc.draw_distance_overlays(annotated, pairwise)

        top_alert = alerts[0]["message"] if alerts else None
        unit_sym = "m" if distance_calc.unit == "meters" else "px"

        display_bgr = hud.draw_hud(
            annotated, class_counts,
            source_info="Snapshot",
            enh_method=enhancer.method if enhancer.enabled else "OFF",
            device=detector.device,
            dist_active=distance_calc.enabled,
            dist_unit=unit_sym,
            proximity_alert=top_alert
        )
        total_count = len(dets)

    latency_ms = (time.time() - start_t) * 1000.0
    return display_bgr, total_count, pairwise, alerts, latency_ms, (enh_dets if sbs_mode else dets)

def main():
    st.markdown('<div class="main-title">🌙 NightVisionAI: Real-Time Road Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Low-Light Enhancement, YOLO Detection & Inter-Object Distance Calculator</div>', unsafe_allow_html=True)

    config = load_config()

    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    st.sidebar.header("📹 Camera & Stream Source")
    source_choice = st.sidebar.selectbox(
        "Select Video / Ingestion Source",
        [
            "🎬 Sample Night Video (Cloud Demo)",
            "📁 Upload Video or Image File",
            "📸 Browser Device Camera (Snapshot)",
            "🤖 Synthetic Generator (Test Mode)",
            "📹 Local Webcam (USB Hardware)",
            "📱 DroidCam Desktop App",
            "🌐 IP Stream / WiFi URL"
        ],
        index=0,
        help="Use 'Sample Night Video' or 'Upload File' when accessing via Cloud (Streamlit Community Cloud). Select 'Webcam' or 'DroidCam' when running locally on your laptop."
    )

    uploaded_file = None
    snapshot_file = None

    if source_choice == "🎬 Sample Night Video (Cloud Demo)":
        config["camera"]["source_type"] = "video"
        sample_path = "data/sample_night.mp4"
        if not os.path.exists(sample_path):
            sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_night.mp4")
        config["camera"]["video_path"] = sample_path
        st.sidebar.info("💡 Streaming pre-recorded nighttime traffic clip. Continuous loop active.")

    elif source_choice == "📁 Upload Video or Image File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a video (.mp4, .avi, .mov) or image (.jpg, .png)",
            type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
        )

    elif source_choice == "📸 Browser Device Camera (Snapshot)":
        st.sidebar.info("Take a photo using your laptop/phone camera right through your browser.")
        snapshot_file = st.camera_input("Capture Road Scene Snapshot")

    elif source_choice == "🤖 Synthetic Generator (Test Mode)":
        config["camera"]["source_type"] = "synthetic"

    elif source_choice == "📹 Local Webcam (USB Hardware)":
        config["camera"]["source_type"] = "webcam"
        cam_idx = st.sidebar.number_input("Webcam Hardware Index", min_value=0, max_value=5, value=0)
        config["camera"]["camera_index"] = int(cam_idx)
        st.sidebar.caption("Requires a physical webcam connected to the host machine.")

    elif source_choice == "📱 DroidCam Desktop App":
        config["camera"]["source_type"] = "droidcam_app"
        st.sidebar.caption("Requires DroidCam client active on your local Windows PC.")

    elif source_choice == "🌐 IP Stream / WiFi URL":
        config["camera"]["source_type"] = "ip_camera"
        stream_url = st.sidebar.text_input("Stream URL", value="http://10.110.18.126:4747/video")
        config["camera"]["stream_url"] = stream_url

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
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 0.90, float(config.get("detection", {}).get("conf_threshold", 0.30)), 0.05, help="Lower threshold detects dim/distant objects; higher values reduce false positives.")
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
    # MAIN LAYOUT METRICS
    # =========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)
    metric_fps = col1.metric("Live FPS", "0.0")
    metric_lat = col2.metric("Latency", "0 ms")
    metric_objs = col3.metric("Objects", "0")
    metric_dist = col4.metric("Closest Distance", "--")
    metric_enh = col5.metric("Enhancer", config["enhancement"]["method"].upper() if enh_enabled else "OFF")

    proximity_box = st.empty()
    status_box = st.empty()
    video_placeholder = st.empty()

    # =========================================================================
    # MODE A: BROWSER WEBCAM SNAPSHOT MODE
    # =========================================================================
    if source_choice == "📸 Browser Device Camera (Snapshot)":
        if snapshot_file is not None:
            enhancer = LowLightEnhancer(config)
            detector = ObjectDetector(config)
            distance_calc = DistanceCalculator(config)
            hud = MonitoringHUD(config)

            img_pil = Image.open(snapshot_file).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            display_bgr, total_count, pairwise, alerts, lat_ms, active_dets = process_single_frame(
                frame_bgr, enhancer, detector, distance_calc, hud, sbs_mode=sbs_mode
            )

            min_dist_str = format_closest_distance(pairwise, active_dets)
            metric_lat.metric("Latency", f"{lat_ms:.0f} ms")
            metric_objs.metric("Objects", f"{total_count}")
            metric_dist.metric("Closest Dist", min_dist_str)

            if alerts:
                top = alerts[0]
                if top["severity"] == "CRITICAL":
                    proximity_box.error(f"🚨 {top['message']}")
                else:
                    proximity_box.warning(f"⚠️ {top['message']}")

            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)
            status_box.success("Snapshot successfully processed through AI perception pipeline!")
        else:
            status_box.info("👆 Use the camera widget above to take a photo of a vehicle or road scene.")
        return

    # =========================================================================
    # MODE B: UPLOADED FILE MODE (IMAGE OR VIDEO)
    # =========================================================================
    if source_choice == "📁 Upload Video or Image File":
        if uploaded_file is None:
            status_box.info("👈 Upload a night video or image from the sidebar to begin analysis.")
            return

        is_video = uploaded_file.name.lower().endswith((".mp4", ".avi", ".mov"))

        if not is_video:
            # Process uploaded image once
            enhancer = LowLightEnhancer(config)
            detector = ObjectDetector(config)
            distance_calc = DistanceCalculator(config)
            hud = MonitoringHUD(config)

            img_pil = Image.open(uploaded_file).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            display_bgr, total_count, pairwise, alerts, lat_ms, active_dets = process_single_frame(
                frame_bgr, enhancer, detector, distance_calc, hud, sbs_mode=sbs_mode
            )

            min_dist_str = format_closest_distance(pairwise, active_dets)
            metric_lat.metric("Latency", f"{lat_ms:.0f} ms")
            metric_objs.metric("Objects", f"{total_count}")
            metric_dist.metric("Closest Dist", min_dist_str)

            if alerts:
                top = alerts[0]
                if top["severity"] == "CRITICAL":
                    proximity_box.error(f"🚨 {top['message']}")
                else:
                    proximity_box.warning(f"⚠️ {top['message']}")

            display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)
            status_box.success(f"Analyzed uploaded image: {uploaded_file.name}")
            return
        else:
            # Save uploaded video to temporary file for OpenCV VideoCapture
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            config["camera"]["source_type"] = "video"
            config["camera"]["video_path"] = tfile.name

    # =========================================================================
    # MODE C: CONTINUOUS STREAMING (SAMPLE VIDEO, SYNTHETIC, WEBCAM, DROIDCAM)
    # =========================================================================
    run_pipeline = st.button("🚀 Start Live Monitoring Feed", type="primary")

    if run_pipeline:
        camera = CameraStream(config)
        enhancer = LowLightEnhancer(config)
        detector = ObjectDetector(config)
        distance_calc = DistanceCalculator(config)
        hud = MonitoringHUD(config)

        if not camera.is_opened():
            status_box.error(
                f"⚠️ Unable to open camera source: {camera.error_message}\n\n"
                "💡 **Note for Cloud Hosting:** Cloud servers (Streamlit Community Cloud) do not have physical USB webcams plugged in. "
                "Please select **'🎬 Sample Night Video (Cloud Demo)'**, **'📁 Upload Video or Image File'**, or **'📸 Browser Device Camera'** in the sidebar!"
            )
            return

        status_box.success("Source connected! Stream processing active...")
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
                min_dist_str = format_closest_distance(pairwise, enh_dets if sbs_mode else dets)
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
