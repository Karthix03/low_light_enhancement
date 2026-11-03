"""
Live Monitoring Screen & HUD Renderer for NightVisionAI
Handles visual annotations, bounding boxes, live counters, FPS rendering,
and Side-by-Side original vs enhanced comparison layouts.
"""

import time
import cv2
import numpy as np

# Color Palette per Target Class (BGR format)
CLASS_COLORS = {
    "person": (0, 255, 127),      # Bright Spring Green
    "bicycle": (255, 215, 0),     # Gold Yellow
    "car": (255, 144, 30),        # Deep Sky Blue
    "motorcycle": (0, 165, 255),  # Vivid Orange
    "bus": (238, 130, 238),       # Violet / Purple
    "truck": (255, 255, 0),       # Cyan
    "default": (0, 255, 255)      # Yellow default
}

class MonitoringHUD:
    """
    Renders Head-Up Display (HUD) overlays and object detection bounding boxes.
    """
    def __init__(self, config=None):
        config = config or {}
        display_cfg = config.get("display", {})
        
        self.show_fps = display_cfg.get("show_fps", True)
        self.show_counts = display_cfg.get("show_counts", True)
        self.show_labels = display_cfg.get("show_labels", True)
        
        # FPS Calculation variables
        self.prev_time = time.time()
        self.current_fps = 0.0

    def calculate_fps(self):
        """
        Calculates moving average FPS.
        """
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        if delta > 0:
            fps = 1.0 / delta
            # Smooth out FPS variation
            self.current_fps = 0.8 * self.current_fps + 0.2 * fps if self.current_fps > 0 else fps
        return self.current_fps

    def draw_detections(self, frame, detections, show_distance=True):
        """
        Draws bounding boxes, class labels, confidence scores, and optional camera distance tags.
        
        Returns:
            annotated_frame, class_counts (dict)
        """
        annotated = frame.copy()
        class_counts = {}

        for det in detections:
            bbox = det["bbox"]
            cls_name = det["class_name"].lower()
            conf = det["confidence"]
            
            # Count class occurrence
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            x1, y1, x2, y2 = bbox
            color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])
            
            # Draw primary bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            if self.show_labels:
                dist_str = ""
                if show_distance and "distance_cam" in det:
                    dist_str = f" | {det['distance_cam']}m"
                
                label = f"{cls_name} {int(conf * 100)}%{dist_str}"
                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Fill label background tag
                tag_y1 = max(0, y1 - h - 6)
                cv2.rectangle(annotated, (x1, tag_y1), (x1 + w + 8, tag_y1 + h + 6), color, -1)
                
                # Draw dark label text for contrast
                cv2.putText(annotated, label, (x1 + 4, tag_y1 + h + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
        return annotated, class_counts

    def draw_hud(self, frame, class_counts, source_info="Webcam", enh_method="CLAHE",
                 device="CPU", dist_active=True, dist_unit="m", proximity_alert=None):
        """
        Overlays telemetry stats, FPS, device status, object counts, and proximity alerts.
        """
        annotated = frame
        h, w = annotated.shape[:2]
        
        fps = self.calculate_fps()
        total_objects = sum(class_counts.values())
        
        # Adjust HUD height if there is an active proximity alert
        banner_h = 95 if proximity_alert else 75
        overlay = annotated.copy()
        
        # Color top banner with reddish tint if danger alert is active
        banner_bg = (15, 15, 60) if (proximity_alert and "CRITICAL" in proximity_alert) else (20, 20, 20)
        cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_bg, -1)
        cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)
        
        # Line 1: Title & Telemetry
        title = "NIGHT VISION AI - REAL-TIME ROAD MONITORING"
        cv2.putText(annotated, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
        
        dist_status = f"DistCalc: ON ({dist_unit})" if dist_active else "DistCalc: OFF"
        telemetry = f"FPS: {fps:.1f} | Dev: {device.upper()} | Enh: {enh_method.upper()} | {dist_status} | Src: {source_info}"
        cv2.putText(annotated, telemetry, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Line 2: Object Counts Summary
        if class_counts:
            counts_str = " | ".join([f"{k.capitalize()}: {v}" for k, v in class_counts.items()])
        else:
            counts_str = "No objects detected"
            
        summary_text = f"Total Detected: {total_objects}  ({counts_str})"
        cv2.putText(annotated, summary_text, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)

        # Line 3 (Optional): Proximity / Collision Alert Banner
        if proximity_alert:
            is_crit = "CRITICAL" in proximity_alert
            alert_color = (0, 0, 255) if is_crit else (0, 165, 255)
            alert_prefix = "[!] CRITICAL PROXIMITY ALERT:" if is_crit else "[*] PROXIMITY WARNING:"
            alert_msg = f"{alert_prefix} {proximity_alert.replace('CRITICAL: ', '').replace('WARNING: ', '')}"
            cv2.putText(annotated, alert_msg, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, alert_color, 2, cv2.LINE_AA)

        # Safety Disclaimer Footer
        cv2.putText(annotated, "EXPERIMENTAL AI SYSTEM - NOT FOR SAFEGUARD / CERTIFIED AUTONOMOUS USE",
                    (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        return annotated

    def create_side_by_side(self, raw_frame, raw_detections, enh_frame, enh_detections,
                            enh_method="CLAHE", distance_calc=None):
        """
        Renders a side-by-side split screen comparing raw low-light vs enhanced image,
        with bounding boxes, labels, and inter-object distance overlays on both sides.
        """
        # Annotate raw frame
        annotated_raw, counts_raw = self.draw_detections(
            raw_frame, raw_detections,
            show_distance=(distance_calc.enabled if distance_calc else False)
        )
        if distance_calc and distance_calc.enabled and len(raw_detections) >= 2:
            raw_pairs = distance_calc.calculate_pairwise_distances(raw_detections, raw_frame.shape)
            annotated_raw = distance_calc.draw_distance_overlays(annotated_raw, raw_pairs)

        # Annotate enhanced frame
        annotated_enh, counts_enh = self.draw_detections(
            enh_frame, enh_detections,
            show_distance=(distance_calc.enabled if distance_calc else False)
        )
        if distance_calc and distance_calc.enabled and len(enh_detections) >= 2:
            enh_pairs = distance_calc.calculate_pairwise_distances(enh_detections, enh_frame.shape)
            annotated_enh = distance_calc.draw_distance_overlays(annotated_enh, enh_pairs)
        
        # Header banners for each side
        h, w = raw_frame.shape[:2]
        
        cv2.putText(annotated_raw, f"ORIGINAL RAW FRAME (Obj: {len(raw_detections)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    
        cv2.putText(annotated_enh, f"ENHANCED [{enh_method.upper()}] (Obj: {len(enh_detections)})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Stack horizontally
        combined = np.hstack((annotated_raw, annotated_enh))
        return combined
