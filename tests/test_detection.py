"""
Unit Tests for YOLO Object Detection Module
"""

import unittest
import numpy as np
from src.detection import ObjectDetector
from src.utils import generate_synthetic_night_frame

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.config = {
            "detection": {
                "model_name": "yolov8n.pt",
                "conf_threshold": 0.25,
                "device": "cpu",
                "target_classes": ["person", "car"]
            }
        }
        self.detector = ObjectDetector(self.config)
        self.test_frame = generate_synthetic_night_frame(width=640, height=480)

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIn("cpu", self.detector.device)

    def test_detection_on_frame(self):
        detections = self.detector.detect(self.test_frame)
        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIn("bbox", det)
            self.assertIn("class_name", det)
            self.assertIn("confidence", det)
            self.assertEqual(len(det["bbox"]), 4)

    def test_set_confidence(self):
        self.detector.set_confidence(0.50)
        self.assertEqual(self.detector.conf_threshold, 0.50)

if __name__ == "__main__":
    unittest.main()
