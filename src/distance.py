"""
Distance & Proximity Calculator Module for NightVisionAI
Calculates estimated 3D real-world depth from camera to detected objects,
computes pairwise Euclidean distances between detected objects,
and provides collision / proximity warning alerts and visual overlays.
"""

import math
import cv2
import numpy as np
from .utils import logger

# Real-world estimated heights in meters for standard COCO road and pedestrian classes
DEFAULT_REAL_HEIGHTS = {
    "person": 1.70,
    "bicycle": 1.05,
    "car": 1.50,
    "motorcycle": 1.20,
    "bus": 3.20,
    "truck": 3.00,
    "default": 1.50
}

# Visual styling colors for safety zones (BGR format)
COLOR_DANGER = (0, 0, 255)      # Red for hazardous proximity (< danger_threshold)
COLOR_WARNING = (0, 165, 255)   # Amber / Orange for caution (< warning_threshold)
COLOR_SAFE = (0, 255, 127)      # Spring Green for safe distance

class DistanceCalculator:
    """
    Computes camera-to-object depths and inter-object spatial distances
    for collision warning and ADAS road safety monitoring.
    """
    def __init__(self, config=None):
        config = config or {}
        dist_cfg = config.get("distance", {})

        self.enabled = dist_cfg.get("enabled", True)
        self.unit = dist_cfg.get("unit", "meters").lower()  # "meters" or "pixels"
        self.focal_length = float(dist_cfg.get("focal_length", 700.0))
        self.danger_threshold = float(dist_cfg.get("danger_threshold", 1.5))
        self.warning_threshold = float(dist_cfg.get("warning_threshold", 3.0))
        self.max_pairs = int(dist_cfg.get("max_pairs", 6))
        self.show_camera_distance = dist_cfg.get("show_camera_distance", True)

        # Merge user custom heights with defaults
        user_heights = dist_cfg.get("real_heights", {})
        self.real_heights = {**DEFAULT_REAL_HEIGHTS, **user_heights}

        logger.info(
            f"DistanceCalculator initialized: enabled={self.enabled}, unit='{self.unit}', "
            f"danger_thresh={self.danger_threshold}{'m' if self.unit == 'meters' else 'px'}, "
            f"warning_thresh={self.warning_threshold}{'m' if self.unit == 'meters' else 'px'}"
        )

    def estimate_object_depth(self, bbox, class_name, frame_shape=(480, 640)):
        """
        Estimates the 3D position (X, Y, Z) and radial distance in meters of an object
        relative to the camera using a monocular pinhole camera geometry model.

        Args:
            bbox: [x1, y1, x2, y2]
            class_name: string label (e.g. 'person', 'car')
            frame_shape: (height, width)

        Returns:
            dict containing:
                - 'x_3d': float (meters)
                - 'y_3d': float (meters)
                - 'depth_z': float (meters)
                - 'distance_cam': float (meters, radial distance from camera)
                - 'center': (x, y) pixel coordinates
                - 'bottom_center': (x, y) ground contact point
        """
        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        cx_box = (x1 + x2) / 2.0
        cy_box = (y1 + y2) / 2.0
        bottom_cx = cx_box
        bottom_cy = float(y2)

        real_h = self.real_heights.get(class_name.lower(), self.real_heights.get("default", 1.5))

        # Optical center of image plane
        frame_h, frame_w = frame_shape[:2]
        c_x = frame_w / 2.0
        c_y = frame_h / 2.0

        # Monocular depth: Z = (focal_length * real_height) / bbox_height_in_pixels
        depth_z = (self.focal_length * real_h) / float(box_h)

        # 3D spatial projection in camera coordinate frame (in meters)
        x_3d = (cx_box - c_x) * depth_z / self.focal_length
        y_3d = (bottom_cy - c_y) * depth_z / self.focal_length

        # Radial Euclidean distance from camera origin
        distance_cam = math.sqrt(x_3d ** 2 + y_3d ** 2 + depth_z ** 2)

        return {
            "x_3d": round(x_3d, 2),
            "y_3d": round(y_3d, 2),
            "depth_z": round(depth_z, 2),
            "distance_cam": round(distance_cam, 2),
            "center": (int(cx_box), int(cy_box)),
            "bottom_center": (int(bottom_cx), int(bottom_cy))
        }

    def compute_all_object_depths(self, detections, frame_shape=(480, 640)):
        """
        Enriches a list of detection dictionaries with estimated 3D coordinates and camera distances.
        """
        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cname = det.get("class_name", "object")
            depth_info = self.estimate_object_depth(bbox, cname, frame_shape)
            det.update(depth_info)
        return detections

    def calculate_pairwise_distances(self, detections, frame_shape=(480, 640)):
        """
        Calculates distances between all unique pairs of detected objects.

        Args:
            detections: list of detection dicts (enriched or raw)
            frame_shape: (height, width)

        Returns:
            list of dicts sorted by distance ascending:
                [
                    {
                        "pair_idx": (i, j),
                        "obj_a": detections[i],
                        "obj_b": detections[j],
                        "distance": float,
                        "unit": str ("m" or "px"),
                        "status": "danger" | "warning" | "safe",
                        "color": (B, G, R),
                        "pt_a": (x, y),
                        "pt_b": (x, y),
                        "alert_text": str
                    }, ...
                ]
        """
        if not self.enabled or len(detections) < 2:
            return []

        # Ensure all detections have 3D and anchor coordinates
        self.compute_all_object_depths(detections, frame_shape)

        pairs = []
        n = len(detections)

        for i in range(n):
            for j in range(i + 1, n):
                det_a = detections[i]
                det_b = detections[j]

                pt_a = det_a["bottom_center"]
                pt_b = det_b["bottom_center"]

                if self.unit == "meters":
                    # Real-world 3D Euclidean distance (in meters)
                    dx = det_a["x_3d"] - det_b["x_3d"]
                    dy = det_a["y_3d"] - det_b["y_3d"]
                    dz = det_a["depth_z"] - det_b["depth_z"]
                    dist = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    dist = round(dist, 2)
                    unit_label = "m"
                else:
                    # 2D Screen Pixel distance
                    dx = pt_a[0] - pt_b[0]
                    dy = pt_a[1] - pt_b[1]
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    dist = round(dist, 1)
                    unit_label = "px"

                # Classify proximity status
                if dist <= self.danger_threshold:
                    status = "danger"
                    color = COLOR_DANGER
                elif dist <= self.warning_threshold:
                    status = "warning"
                    color = COLOR_WARNING
                else:
                    status = "safe"
                    color = COLOR_SAFE

                alert_text = f"{det_a['class_name'].capitalize()} <-> {det_b['class_name'].capitalize()}: {dist}{unit_label}"

                pairs.append({
                    "pair_idx": (i, j),
                    "obj_a": det_a,
                    "obj_b": det_b,
                    "distance": dist,
                    "unit": unit_label,
                    "status": status,
                    "color": color,
                    "pt_a": pt_a,
                    "pt_b": pt_b,
                    "alert_text": alert_text
                })

        # Sort pairs so closest objects appear first
        pairs.sort(key=lambda p: p["distance"])
        return pairs

    def draw_distance_overlays(self, frame, pairwise_results):
        """
        Draws dynamic connecting lines, distance pills, and safety warning indicators
        between pairs of objects on the given image frame.
        """
        if not self.enabled or not pairwise_results:
            return frame

        annotated = frame.copy()
        drawn_count = 0

        for pair in pairwise_results:
            if drawn_count >= self.max_pairs:
                break

            pt_a = pair["pt_a"]
            pt_b = pair["pt_b"]
            color = pair["color"]
            dist_val = pair["distance"]
            unit = pair["unit"]
            status = pair["status"]

            # 1. Draw connecting line between objects
            line_thickness = 3 if status == "danger" else 2
            cv2.line(annotated, pt_a, pt_b, color, line_thickness, cv2.LINE_AA)

            # 2. Draw sleek anchor points on each object's ground contact point
            cv2.circle(annotated, pt_a, 5, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, pt_a, 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(annotated, pt_b, 5, color, -1, cv2.LINE_AA)
            cv2.circle(annotated, pt_b, 7, (255, 255, 255), 1, cv2.LINE_AA)

            # 3. Midpoint Distance Badge Pill
            mid_x = int((pt_a[0] + pt_b[0]) / 2.0)
            mid_y = int((pt_a[1] + pt_b[1]) / 2.0)

            if status == "danger":
                tag_text = f"! {dist_val}{unit} !"
            elif status == "warning":
                tag_text = f"{dist_val}{unit}"
            else:
                tag_text = f"{dist_val}{unit}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (t_w, t_h), baseline = cv2.getTextSize(tag_text, font, font_scale, thickness)

            # Pill box coordinates
            pad_x, pad_y = 6, 4
            box_x1 = mid_x - (t_w // 2) - pad_x
            box_y1 = mid_y - (t_h // 2) - pad_y
            box_x2 = mid_x + (t_w // 2) + pad_x
            box_y2 = mid_y + (t_h // 2) + pad_y

            # Keep pill within frame boundaries
            h, w = frame.shape[:2]
            box_x1 = max(2, min(box_x1, w - t_w - 2 * pad_x - 2))
            box_y1 = max(2, min(box_y1, h - t_h - 2 * pad_y - 2))
            box_x2 = box_x1 + t_w + 2 * pad_x
            box_y2 = box_y1 + t_h + 2 * pad_y

            # Fill pill background with contrast border
            bg_color = (0, 0, 160) if status == "danger" else ((0, 80, 160) if status == "warning" else (20, 40, 20))
            cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
            cv2.rectangle(annotated, (box_x1, box_y1), (box_x2, box_y2), color, 1, cv2.LINE_AA)

            # Text inside pill
            text_x = box_x1 + pad_x
            text_y = box_y2 - pad_y
            text_color = (255, 255, 255)
            cv2.putText(annotated, tag_text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            drawn_count += 1

        return annotated

    def get_proximity_alerts(self, pairwise_results):
        """
        Returns list of active collision / danger warning messages.
        """
        alerts = []
        for pair in pairwise_results:
            if pair["status"] == "danger":
                alerts.append({
                    "severity": "CRITICAL",
                    "message": f"COLLISION RISK: {pair['alert_text']}",
                    "pair": pair["alert_text"],
                    "distance": pair["distance"],
                    "unit": pair["unit"]
                })
            elif pair["status"] == "warning":
                alerts.append({
                    "severity": "WARNING",
                    "message": f"CLOSE PROXIMITY: {pair['alert_text']}",
                    "pair": pair["alert_text"],
                    "distance": pair["distance"],
                    "unit": pair["unit"]
                })
        return alerts

    def set_unit(self, new_unit):
        """
        Toggles or sets distance measurement unit ('meters' or 'pixels').
        """
        if new_unit.lower() in ["meters", "pixels"]:
            self.unit = new_unit.lower()
            logger.info(f"Distance calculation unit set to: {self.unit}")

    def toggle_unit(self):
        """
        Cycles between 'meters' and 'pixels'.
        """
        self.unit = "pixels" if self.unit == "meters" else "meters"
        logger.info(f"Distance calculation unit toggled to: {self.unit}")
        return self.unit
