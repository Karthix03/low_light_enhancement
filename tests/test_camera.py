"""
Unit Tests for Camera Stream Module
"""

import unittest
import numpy as np
from src.camera import CameraStream

class TestCameraStream(unittest.TestCase):
    def test_synthetic_camera_stream(self):
        config = {
            "camera": {
                "source_type": "synthetic",
                "frame_width": 640,
                "frame_height": 480
            }
        }
        cam = CameraStream(config)
        self.assertTrue(cam.is_opened())
        
        success, frame = cam.read()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)
        
        cam.release()
        self.assertFalse(cam.is_connected)

    def test_invalid_webcam_index(self):
        config = {
            "camera": {
                "source_type": "webcam",
                "camera_index": 999  # Non-existent camera index
            }
        }
        cam = CameraStream(config)
        # Should gracefully fail or report error message
        self.assertIsNotNone(cam.error_message)
        cam.release()

if __name__ == "__main__":
    unittest.main()
