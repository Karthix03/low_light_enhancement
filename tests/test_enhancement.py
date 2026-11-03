"""
Unit Tests for Low-Light Enhancement Module
"""

import unittest
import numpy as np
from src.enhancement import LowLightEnhancer
from src.utils import generate_synthetic_night_frame

class TestLowLightEnhancer(unittest.TestCase):
    def setUp(self):
        self.dark_frame = generate_synthetic_night_frame(width=640, height=480)

    def test_gamma_enhancement(self):
        config = {"enhancement": {"enabled": True, "method": "gamma", "gamma_value": 2.0}}
        enhancer = LowLightEnhancer(config)
        enhanced = enhancer.enhance(self.dark_frame)
        
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, self.dark_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Enhanced frame mean brightness should generally be higher than original dark frame
        self.assertGreaterEqual(np.mean(enhanced), np.mean(self.dark_frame))

    def test_clahe_enhancement(self):
        config = {"enhancement": {"enabled": True, "method": "clahe", "clahe_clip_limit": 3.0}}
        enhancer = LowLightEnhancer(config)
        enhanced = enhancer.enhance(self.dark_frame)
        
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, self.dark_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_zero_dce_enhancement(self):
        config = {"enhancement": {"enabled": True, "method": "zero_dce"}}
        enhancer = LowLightEnhancer(config)
        enhanced = enhancer.enhance(self.dark_frame)
        
        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape, self.dark_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_disabled_enhancement(self):
        config = {"enhancement": {"enabled": False, "method": "clahe"}}
        enhancer = LowLightEnhancer(config)
        enhanced = enhancer.enhance(self.dark_frame)
        
        # When disabled, should return exact raw frame
        np.testing.assert_array_equal(enhanced, self.dark_frame)

if __name__ == "__main__":
    unittest.main()
