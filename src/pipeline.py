"""
End-to-End Night Vision Pipeline Module for NightVisionAI
Integrates Camera Input, Low-Light Enhancement, YOLO Object Detector, and Telemetry HUD.
"""

import time
import cv2
from .camera import CameraStream
from .enhancement import LowLightEnhancer
from .detection import ObjectDetector
from .monitoring import MonitoringHUD
from .distance import DistanceCalculator
from .utils import load_config, logger, save_screenshot

class NightVisionPipeline:
    """
    Core Pipeline orchestrating video capture, image enhancement, object detection,
    distance calculation / proximity alerts, and visual output.
    """
    def __init__(self, config=None):
        self.config = load_config() if config is None else config
        
        display_cfg = self.config.get("display", {})
        self.side_by_side = display_cfg.get("side_by_side", False)
        
        # Instantiate sub-modules
        logger.info("Initializing NightVision Pipeline components...")
        self.camera = CameraStream(self.config)
        self.enhancer = LowLightEnhancer(self.config)
        self.detector = ObjectDetector(self.config)
        self.distance_calc = DistanceCalculator(self.config)
        self.hud = MonitoringHUD(self.config)
        
        self.is_running = True

    def process_frame(self, raw_frame):
        """
        Processes a single input image frame through the pipeline.
        
        Returns:
            display_image (numpy array), metrics_dict (dict)
        """
        start_t = time.time()
        
        if raw_frame is None:
            return None, {}

        # 1. Enhancement Stage
        if self.enhancer.enabled:
            enhanced_frame = self.enhancer.enhance(raw_frame)
        else:
            enhanced_frame = raw_frame

        # 2. Side-by-Side Comparison Mode vs Single Enhanced Mode
        if self.side_by_side:
            # Detect on raw frame
            raw_detections = self.detector.detect(raw_frame)
            # Detect on enhanced frame
            enh_detections = self.detector.detect(enhanced_frame)
            
            # Enrich and compute pairwise on enhanced detections for metrics & alerts
            if self.distance_calc.enabled and len(enh_detections) >= 2:
                pairwise = self.distance_calc.calculate_pairwise_distances(enh_detections, enhanced_frame.shape)
                alerts = self.distance_calc.get_proximity_alerts(pairwise)
            else:
                pairwise = []
                alerts = []

            display_frame = self.hud.create_side_by_side(
                raw_frame, raw_detections,
                enhanced_frame, enh_detections,
                enh_method=self.enhancer.method if self.enhancer.enabled else "OFF",
                distance_calc=self.distance_calc
            )
            total_detections = enh_detections
        else:
            # Standard Single Mode
            detections = self.detector.detect(enhanced_frame)
            
            # Distance calculation & proximity alerts
            if self.distance_calc.enabled and len(detections) > 0:
                self.distance_calc.compute_all_object_depths(detections, enhanced_frame.shape)
                pairwise = self.distance_calc.calculate_pairwise_distances(detections, enhanced_frame.shape)
                alerts = self.distance_calc.get_proximity_alerts(pairwise)
            else:
                pairwise = []
                alerts = []

            annotated_frame, class_counts = self.hud.draw_detections(
                enhanced_frame, detections,
                show_distance=self.distance_calc.enabled
            )
            
            # Draw distance lines and pills between detected objects
            if self.distance_calc.enabled and pairwise:
                annotated_frame = self.distance_calc.draw_distance_overlays(annotated_frame, pairwise)

            source_info = self.camera.source_type
            enh_name = self.enhancer.method if self.enhancer.enabled else "OFF"
            top_alert = alerts[0]["message"] if alerts else None
            unit_sym = "m" if self.distance_calc.unit == "meters" else "px"

            display_frame = self.hud.draw_hud(
                annotated_frame,
                class_counts,
                source_info=source_info,
                enh_method=enh_name,
                device=self.detector.device,
                dist_active=self.distance_calc.enabled,
                dist_unit=unit_sym,
                proximity_alert=top_alert
            )
            total_detections = detections

        latency_ms = (time.time() - start_t) * 1000.0
        
        metrics = {
            "fps": self.hud.current_fps,
            "latency_ms": latency_ms,
            "num_objects": len(total_detections),
            "enhancement_active": self.enhancer.enabled,
            "enhancement_method": self.enhancer.method,
            "distance_calculator_active": self.distance_calc.enabled,
            "distance_unit": self.distance_calc.unit,
            "min_distance": pairwise[0]["distance"] if pairwise else None,
            "proximity_alerts": [a["message"] for a in alerts],
            "num_pairs": len(pairwise),
            "device": self.detector.device,
            "camera_source": self.camera.get_source_info()
        }
        
        return display_frame, metrics

    def step(self):
        """
        Reads one frame from camera, runs pipeline, and returns (display_frame, metrics).
        """
        success, raw_frame = self.camera.read()
        if not success or raw_frame is None:
            return None, {"error": self.camera.error_message or "Failed to read frame."}

        return self.process_frame(raw_frame)

    def toggle_enhancement(self):
        """
        Toggles enhancement on or off.
        """
        self.enhancer.enabled = not self.enhancer.enabled
        logger.info(f"Enhancement toggled: Enabled = {self.enhancer.enabled}")
        return self.enhancer.enabled

    def cycle_enhancement_method(self):
        """
        Cycles through available enhancement methods: clahe -> gamma -> zero_dce -> none -> clahe.
        """
        methods = ["clahe", "gamma", "zero_dce", "none"]
        curr_idx = methods.index(self.enhancer.method) if self.enhancer.method in methods else 0
        next_method = methods[(curr_idx + 1) % len(methods)]
        self.enhancer.set_method(next_method)
        return self.enhancer.method

    def toggle_distance_calc(self):
        """
        Toggles distance calculation on or off.
        """
        self.distance_calc.enabled = not self.distance_calc.enabled
        logger.info(f"Distance Calculator toggled: Enabled = {self.distance_calc.enabled}")
        return self.distance_calc.enabled

    def toggle_distance_unit(self):
        """
        Toggles distance unit between 'meters' and 'pixels'.
        """
        unit = self.distance_calc.toggle_unit()
        logger.info(f"Distance Calculator unit toggled to: {unit}")
        return unit

    def toggle_side_by_side(self):
        """
        Toggles side-by-side raw vs enhanced split screen comparison mode.
        """
        self.side_by_side = not self.side_by_side
        logger.info(f"Side-by-Side comparison mode set to: {self.side_by_side}")
        return self.side_by_side

    def capture_screenshot(self, current_display_frame):
        """
        Saves current frame screenshot.
        """
        if current_display_frame is not None:
            return save_screenshot(current_display_frame, self.config.get("display", {}).get("output_dir", "outputs"))
        return None

    def release(self):
        """
        Safely shuts down camera resources.
        """
        if self.camera:
            self.camera.release()
        logger.info("NightVision Pipeline shut down cleanly.")
