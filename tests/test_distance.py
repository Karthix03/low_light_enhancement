"""
Unit Tests for Distance & Proximity Calculator Module
"""

import unittest
import numpy as np
from src.distance import DistanceCalculator

class TestDistanceCalculator(unittest.TestCase):
    def setUp(self):
        self.config = {
            "distance": {
                "enabled": True,
                "unit": "meters",
                "focal_length": 700.0,
                "danger_threshold": 1.5,
                "warning_threshold": 3.0,
                "max_pairs": 4,
                "show_camera_distance": True,
                "real_heights": {
                    "person": 1.70,
                    "car": 1.50
                }
            }
        }
        self.calculator = DistanceCalculator(self.config)
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_initialization(self):
        self.assertTrue(self.calculator.enabled)
        self.assertEqual(self.calculator.unit, "meters")
        self.assertEqual(self.calculator.danger_threshold, 1.5)
        self.assertEqual(self.calculator.warning_threshold, 3.0)

    def test_estimate_depth(self):
        # Tall person bbox (closer) vs short person bbox (farther)
        close_bbox = [200, 100, 300, 400]   # height = 300px
        far_bbox = [200, 100, 250, 200]     # height = 100px

        depth_close = self.calculator.estimate_object_depth(close_bbox, "person", (480, 640))
        depth_far = self.calculator.estimate_object_depth(far_bbox, "person", (480, 640))

        self.assertIn("depth_z", depth_close)
        self.assertIn("distance_cam", depth_close)
        self.assertIn("x_3d", depth_close)
        self.assertIn("bottom_center", depth_close)

        # Closer object (larger pixel height) must have smaller depth Z
        self.assertLess(depth_close["depth_z"], depth_far["depth_z"])
        self.assertLess(depth_close["distance_cam"], depth_far["distance_cam"])

    def test_pairwise_distances_empty_or_single(self):
        # 0 detections
        self.assertEqual(self.calculator.calculate_pairwise_distances([]), [])
        
        # 1 detection
        single_det = [{"bbox": [100, 100, 200, 300], "class_name": "person", "confidence": 0.9}]
        self.assertEqual(self.calculator.calculate_pairwise_distances(single_det), [])

    def test_pairwise_distances_two_objects(self):
        detections = [
            {"bbox": [100, 150, 180, 350], "class_name": "person", "confidence": 0.85},
            {"bbox": [350, 150, 480, 350], "class_name": "car", "confidence": 0.90}
        ]

        pairs = self.calculator.calculate_pairwise_distances(detections, (480, 640))
        self.assertEqual(len(pairs), 1)

        pair = pairs[0]
        self.assertIn("distance", pair)
        self.assertEqual(pair["unit"], "m")
        self.assertIn(pair["status"], ["danger", "warning", "safe"])
        self.assertIn("alert_text", pair)

    def test_danger_proximity_alert(self):
        # Two bounding boxes very close together (touching or overlapping)
        detections = [
            {"bbox": [200, 200, 260, 400], "class_name": "person", "confidence": 0.8},
            {"bbox": [210, 200, 280, 400], "class_name": "person", "confidence": 0.85}
        ]

        pairs = self.calculator.calculate_pairwise_distances(detections, (480, 640))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["status"], "danger")

        alerts = self.calculator.get_proximity_alerts(pairs)
        self.assertTrue(len(alerts) >= 1)
        self.assertEqual(alerts[0]["severity"], "CRITICAL")
        self.assertIn("COLLISION RISK", alerts[0]["message"])

    def test_pixel_unit_calculation(self):
        self.calculator.set_unit("pixels")
        self.assertEqual(self.calculator.unit, "pixels")

        detections = [
            {"bbox": [100, 200, 150, 300], "class_name": "person", "confidence": 0.8},
            {"bbox": [200, 200, 250, 300], "class_name": "car", "confidence": 0.9}
        ]

        pairs = self.calculator.calculate_pairwise_distances(detections, (480, 640))
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["unit"], "px")
        # Horizontal difference is 100 pixels between centers (125 vs 225)
        self.assertAlmostEqual(pairs[0]["distance"], 100.0, places=0)

    def test_toggle_unit(self):
        self.calculator.unit = "meters"
        new_unit = self.calculator.toggle_unit()
        self.assertEqual(new_unit, "pixels")
        new_unit2 = self.calculator.toggle_unit()
        self.assertEqual(new_unit2, "meters")

    def test_draw_distance_overlays(self):
        detections = [
            {"bbox": [100, 150, 180, 350], "class_name": "person", "confidence": 0.85},
            {"bbox": [350, 150, 480, 350], "class_name": "car", "confidence": 0.90}
        ]
        pairs = self.calculator.calculate_pairwise_distances(detections, (480, 640))
        overlay_frame = self.calculator.draw_distance_overlays(self.dummy_frame, pairs)

        self.assertIsNotNone(overlay_frame)
        self.assertEqual(overlay_frame.shape, self.dummy_frame.shape)
        self.assertEqual(overlay_frame.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
