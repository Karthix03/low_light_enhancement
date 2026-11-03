"""
Integration Tests for NightVision Pipeline
"""

import unittest
import os
import shutil
from src.pipeline import NightVisionPipeline

class TestNightVisionPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "camera": {
                "source_type": "synthetic",
                "frame_width": 640,
                "frame_height": 480
            },
            "enhancement": {
                "enabled": True,
                "method": "clahe"
            },
            "detection": {
                "model_name": "yolov8n.pt",
                "conf_threshold": 0.3,
                "device": "cpu"
            },
            "display": {
                "side_by_side": False,
                "output_dir": "tests/test_outputs"
            }
        }
        self.pipeline = NightVisionPipeline(self.config)

    def tearDown(self):
        self.pipeline.release()
        if os.path.exists("tests/test_outputs"):
            shutil.rmtree("tests/test_outputs")

    def test_pipeline_step(self):
        display_frame, metrics = self.pipeline.step()
        self.assertIsNotNone(display_frame)
        self.assertIn("fps", metrics)
        self.assertIn("num_objects", metrics)
        self.assertIn("distance_calculator_active", metrics)
        self.assertTrue(metrics["distance_calculator_active"])

    def test_toggle_functions(self):
        initial_state = self.pipeline.enhancer.enabled
        new_state = self.pipeline.toggle_enhancement()
        self.assertNotEqual(initial_state, new_state)

        initial_sbs = self.pipeline.side_by_side
        new_sbs = self.pipeline.toggle_side_by_side()
        self.assertNotEqual(initial_sbs, new_sbs)

        initial_dist = self.pipeline.distance_calc.enabled
        new_dist = self.pipeline.toggle_distance_calc()
        self.assertNotEqual(initial_dist, new_dist)

        unit1 = self.pipeline.distance_calc.unit
        unit2 = self.pipeline.toggle_distance_unit()
        self.assertNotEqual(unit1, unit2)

    def test_screenshot_capture(self):
        display_frame, _ = self.pipeline.step()
        path = self.pipeline.capture_screenshot(display_frame)
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))

if __name__ == "__main__":
    unittest.main()
