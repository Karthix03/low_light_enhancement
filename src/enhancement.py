"""
Low-Light Image Enhancement Module for NightVisionAI
Implements Baseline Algorithms (Gamma Correction, LAB-CLAHE) and
Deep Learning Neural Enhancement (Zero-DCE PyTorch architecture).
"""

import os
import cv2
import numpy as np
from .utils import logger

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# 1. Zero-DCE Deep Learning Architecture (PyTorch Implementation)
# ============================================================================
if TORCH_AVAILABLE:
    class ZeroDCENetwork(nn.Module):
        """
        Zero-Reference Deep Curve Estimation (Zero-DCE) PyTorch Network.
        Estimates higher-order pixel-wise curve parameter maps to enhance dark images
        without requiring paired dark/bright training images.
        """
        def __init__(self, number_f=32, n_curves=8):
            super(ZeroDCENetwork, self).__init__()
            self.n_curves = n_curves
            self.relu = nn.ReLU(inplace=True)
            
            # 7-layer Convolutional Feature Extractor
            self.conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
            self.conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            self.conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            self.conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
            
            # Skip connection concats
            self.conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
            self.conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
            self.conv7 = nn.Conv2d(number_f * 2, 3 * n_curves, 3, 1, 1, bias=True)

        def forward(self, x):
            x1 = self.relu(self.conv1(x))
            x2 = self.relu(self.conv2(x1))
            x3 = self.relu(self.conv3(x2))
            x4 = self.relu(self.conv4(x3))
            
            x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
            x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
            r = torch.tanh(self.conv7(torch.cat([x1, x6], 1))) # Curve parameters [-1, 1]
            
            # Apply iteration curves LE_n(x) = LE_{n-1} + r_n * LE_{n-1} * (1 - LE_{n-1})
            r_list = torch.split(r, 3, dim=1)
            x_enhanced = x
            for i in range(self.n_curves):
                x_enhanced = x_enhanced + r_list[i] * x_enhanced * (1.0 - x_enhanced)
                
            return torch.clamp(x_enhanced, 0.0, 1.0)


# ============================================================================
# 2. Main Enhancer Class
# ============================================================================
class LowLightEnhancer:
    """
    Unified Image Enhancer supporting CLAHE, Gamma Correction, and Zero-DCE.
    """
    def __init__(self, config=None):
        config = config or {}
        enh_cfg = config.get("enhancement", {})
        
        self.enabled = enh_cfg.get("enabled", True)
        self.method = enh_cfg.get("method", "clahe").lower()
        self.gamma_value = float(enh_cfg.get("gamma_value", 1.8))
        self.clahe_clip_limit = float(enh_cfg.get("clahe_clip_limit", 2.5))
        grid_size = enh_cfg.get("clahe_tile_grid", [8, 8])
        self.clahe_tile_grid = (int(grid_size[0]), int(grid_size[1]))
        self.weights_path = enh_cfg.get("weights_path", "models/zero_dce.pth")
        
        # Build Gamma Lookup Table (LUT) for real-time speed
        self._build_gamma_lut()
        
        # Initialize CLAHE solver
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid
        )
        
        # Zero-DCE model state
        self.zero_dce_model = None
        self.device = "cpu"
        
        if TORCH_AVAILABLE and self.method == "zero_dce":
            self._init_zero_dce()

    def _build_gamma_lut(self):
        """
        Creates a fast 256-element lookup table for exponential gamma mapping.
        """
        inv_gamma = 1.0 / self.gamma_value if self.gamma_value > 0 else 1.0
        self.gamma_lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")

    def _init_zero_dce(self):
        """
        Initializes the Zero-DCE PyTorch network and loads weights if available.
        """
        try:
            self.device = "cuda" if (torch.cuda.is_available()) else "cpu"
            self.zero_dce_model = ZeroDCENetwork().to(self.device)
            self.zero_dce_model.eval()
            
            if os.path.exists(self.weights_path):
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                self.zero_dce_model.load_state_dict(checkpoint)
                logger.info(f"Loaded Zero-DCE pretrained weights from '{self.weights_path}'.")
            else:
                logger.info(f"Zero-DCE weights not found at '{self.weights_path}'. Initializing Zero-DCE zero-shot curve model.")
        except Exception as e:
            logger.warning(f"Failed to initialize Zero-DCE PyTorch model: {e}. Falling back to CLAHE.")
            self.zero_dce_model = None
            self.method = "clahe"

    def set_method(self, method_name):
        """
        Dynamically changes the active enhancement method.
        Supported: 'none', 'clahe', 'gamma', 'zero_dce'
        """
        valid_methods = ['none', 'clahe', 'gamma', 'zero_dce']
        method_name = method_name.lower()
        if method_name in valid_methods:
            self.method = method_name
            if method_name == "zero_dce" and self.zero_dce_model is None and TORCH_AVAILABLE:
                self._init_zero_dce()
            logger.info(f"Enhancement method changed to: '{self.method}'")
        else:
            logger.warning(f"Invalid enhancement method '{method_name}'. Keeping '{self.method}'.")

    def enhance_gamma(self, frame_bgr):
        """
        Applies non-linear Gamma Correction using pre-calculated Lookup Table (LUT).
        """
        return cv2.LUT(frame_bgr, self.gamma_lut)

    def enhance_clahe(self, frame_bgr):
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        to the Luminance (L) channel in LAB color space to preserve color harmony.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def enhance_zero_dce(self, frame_bgr):
        """
        Applies Zero-DCE Neural Deep Curve Enhancement using PyTorch.
        """
        if not TORCH_AVAILABLE or self.zero_dce_model is None:
            # Safe fallback
            return self.enhance_clahe(frame_bgr)
            
        try:
            # Convert BGR (numpy uint8) -> RGB float32 [0, 1] tensor
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(self.device)
            
            with torch.no_grad():
                enhanced_tensor = self.zero_dce_model(img_tensor)
                
            # Convert tensor back -> BGR numpy uint8
            enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            enhanced_np = (enhanced_np * 255.0).clip(0, 255).astype(np.uint8)
            return cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error executing Zero-DCE neural enhancement: {e}. Falling back to CLAHE.")
            return self.enhance_clahe(frame_bgr)

    def enhance(self, frame_bgr):
        """
        Main pipeline entry point. Enhances input frame based on current configuration.
        Guarantees returned image matches input dimensions.
        """
        if not self.enabled or frame_bgr is None or self.method == "none":
            return frame_bgr

        try:
            if self.method == "gamma":
                enhanced = self.enhance_gamma(frame_bgr)
            elif self.method == "clahe":
                enhanced = self.enhance_clahe(frame_bgr)
            elif self.method == "zero_dce":
                enhanced = self.enhance_zero_dce(frame_bgr)
            else:
                enhanced = frame_bgr
                
            # Validate output dimensions match input
            if enhanced.shape != frame_bgr.shape:
                logger.warning("Enhanced frame shape mismatch. Resizing to original dimensions.")
                enhanced = cv2.resize(enhanced, (frame_bgr.shape[1], frame_bgr.shape[0]))
                
            return enhanced
        except Exception as e:
            logger.error(f"Enhancement error ({self.method}): {e}. Returning raw frame.")
            return frame_bgr
