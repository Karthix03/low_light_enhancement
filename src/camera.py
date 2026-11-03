"""
Flexible Camera Input Module for NightVisionAI
Supports Laptop Webcams, DroidCam Desktop App, WiFi IP Streams, Prerecorded Video Files,
and Synthetic Dark Frame Generators for testing.
"""

import time
import cv2
import numpy as np
import urllib.request
import requests
import threading
from .utils import logger, generate_synthetic_night_frame

def list_available_cameras(max_tested=5):
    """
    Scans camera indices 0 through max_tested-1 on Windows
    to discover connected webcams, USB cameras, or DroidCam virtual cameras.
    Returns a list of working integer camera indices.
    """
    working_indices = []
    logger.info("Scanning for connected camera devices...")
    for index in range(max_tested):
        for api in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(index, api)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    working_indices.append((index, api))
                    cap.release()
                    break
                cap.release()
    logger.info(f"Available camera indices detected: {working_indices}")
    return working_indices


class HTTPStreamReader:
    """
    Threaded HTTP MJPEG / JPEG Stream Reader for IP cameras and DroidCam streams.
    Decodes JPEG frames asynchronously in a background thread for sub-millisecond latency.
    """
    def __init__(self, url):
        self.url = url
        self.current_frame = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self.open()

    def open(self):
        self.release()
        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        
        # Wait up to 1.5 seconds for first frame decoding
        start_t = time.time()
        while time.time() - start_t < 1.5:
            with self.lock:
                if self.current_frame is not None:
                    return True
            time.sleep(0.05)
        return self.is_opened()

    def _update_loop(self):
        while self.is_running:
            try:
                resp = requests.get(self.url, stream=True, timeout=4, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    bytes_data = b''
                    for chunk in resp.iter_content(chunk_size=4096):
                        if not self.is_running:
                            break
                        bytes_data += chunk
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')
                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]
                            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                with self.lock:
                                    self.current_frame = img
                resp.close()
            except Exception:
                pass
            time.sleep(0.02)

    def is_opened(self):
        with self.lock:
            has_frame = self.current_frame is not None
        return self.is_running and (has_frame or (self.thread and self.thread.is_alive()))

    def read(self):
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
        return False, None

    def release(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        with self.lock:
            self.current_frame = None


class CameraStream:
    """
    Robust Camera Capture Class handling Webcams, DroidCam Desktop App, Video Files, and Synthetic Fallbacks.
    """
    def __init__(self, config=None):
        config = config or {}
        camera_cfg = config.get("camera", {})
        
        self.source_type = camera_cfg.get("source_type", "droidcam_app").lower()
        self.camera_index = int(camera_cfg.get("camera_index", 0))
        self.stream_url = camera_cfg.get("stream_url", "http://10.110.18.126:4747/video")
        self.video_path = camera_cfg.get("video_path", "data/sample_night.mp4")
        self.target_width = camera_cfg.get("frame_width", 640)
        self.target_height = camera_cfg.get("frame_height", 480)
        
        self.cap = None
        self.http_reader = None
        self.is_connected = False
        self.error_message = None
        self.frame_count = 0
        
        self.start()

    def _resolve_droidcam_url(self, base_url):
        """
        Checks if DroidCam URL returns HTML 'DroidCam is Busy' or valid stream endpoints.
        """
        urls_to_try = [base_url]
        if not base_url.endswith("/mjpegfeed"):
            urls_to_try.append(base_url.rstrip("/") + "/mjpegfeed")
        if not base_url.endswith("/video"):
            urls_to_try.append(base_url.rstrip("/") + "/video")

        for url in urls_to_try:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=2) as resp:
                    body_sample = resp.read(200)
                    if b"DroidCam is Busy" in body_sample:
                        self.error_message = (
                            "DroidCam is BUSY on phone! Close any other DroidCam Windows client, "
                            "browser tab, or restart the DroidCam app on your phone."
                        )
                        logger.error(self.error_message)
                        return None
                    return url
            except Exception:
                pass
        return base_url

    def start(self):
        """
        Initializes and opens the camera capture device or stream based on configuration.
        """
        self.release()
        self.error_message = None
        logger.info(f"Connecting to camera source type: '{self.source_type}'...")

        if self.source_type == "synthetic":
            self.is_connected = True
            logger.info("Using Synthetic Test Frame Generator.")
            return True

        if self.source_type == "webcam":
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
            
            if not self.cap.isOpened():
                self.error_message = (
                    f"Unable to open Webcam at index {self.camera_index}. "
                    "Ensure camera is plugged in."
                )
                logger.error(self.error_message)
                self.is_connected = False
                return False

        elif self.source_type == "droidcam_app":
            logger.info("Connecting to DroidCam Desktop App Virtual Camera (MSMF Index 1)...")
            # DroidCam Desktop App maps to MSMF Index 1 on Windows 11
            self.cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.camera_index = 1
                    logger.info("SUCCESSFULLY CONNECTED to DroidCam Desktop App Virtual Camera!")
                else:
                    self.cap.release()
                    self.cap = None

            if not self.cap or not self.cap.isOpened():
                # Try DSHOW Index 1 or Index 0 as fallback
                for idx in [1, 0, 2]:
                    cap_test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cap_test.isOpened():
                        ret, f = cap_test.read()
                        if ret and f is not None:
                            self.cap = cap_test
                            self.camera_index = idx
                            logger.info(f"Connected to DroidCam / Webcam (DSHOW Index {idx}).")
                            break
                        cap_test.release()

            if not self.cap or not self.cap.isOpened():
                self.error_message = (
                    "DroidCam Desktop App is not active or connected! "
                    "Open DroidCam Client on laptop, connect your phone, and click START."
                )
                logger.error(self.error_message)
                self.is_connected = False
                return False

        elif self.source_type == "ip_camera":
            valid_url = self._resolve_droidcam_url(self.stream_url)
            if self.error_message:
                self.is_connected = False
                return False

            url_to_open = valid_url if valid_url else self.stream_url
            logger.info(f"Opening IP Camera stream: {url_to_open}")
            
            self.cap = cv2.VideoCapture(url_to_open)
            if self.cap.isOpened():
                ret, test_f = self.cap.read()
                if not ret or test_f is None:
                    self.cap.release()
                    self.cap = None

            if self.cap is None or not self.cap.isOpened():
                logger.info(f"Initializing Threaded HTTP Stream Reader for '{url_to_open}'...")
                self.http_reader = HTTPStreamReader(url_to_open)
                if not self.http_reader.is_opened():
                    self.error_message = (
                        f"Unable to connect to IP Camera stream URL: '{self.stream_url}'. "
                        "Check if DroidCam app is open on phone and 'DroidCam is Busy' error is cleared."
                    )
                    logger.error(self.error_message)
                    self.is_connected = False
                    return False

        elif self.source_type == "video":
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error_message = f"Unable to open video file at path: '{self.video_path}'."
                logger.error(self.error_message)
                self.is_connected = False
                return False

        else:
            self.error_message = f"Unknown camera source type: '{self.source_type}'."
            logger.error(self.error_message)
            self.is_connected = False
            return False

        if (self.cap and self.cap.isOpened()) or (self.http_reader and self.http_reader.is_opened()):
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.is_connected = True
            logger.info(f"Camera source successfully initialized ({self.source_type}).")
            return True

        return False

    def read(self):
        """
        Reads the next video frame. Returns (success_flag, frame_bgr).
        """
        if self.source_type == "synthetic":
            self.frame_count += 1
            frame = generate_synthetic_night_frame(self.target_width, self.target_height)
            return True, frame

        if not self.is_connected:
            return False, None

        ret, frame = False, None
        if self.http_reader and self.http_reader.is_opened():
            ret, frame = self.http_reader.read()
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret and self.source_type == "video":
                logger.info("End of video stream reached. Looping video...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

        if ret and frame is not None:
            self.frame_count += 1
            if (frame.shape[1], frame.shape[0]) != (self.target_width, self.target_height):
                frame = cv2.resize(frame, (self.target_width, self.target_height))
            return True, frame
        else:
            logger.warning("Failed to grab video frame from camera stream.")
            return False, None

    def release(self):
        """
        Safely releases OpenCV VideoCapture device or HTTP stream resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.http_reader is not None:
            self.http_reader.release()
            self.http_reader = None
        self.is_connected = False
        logger.info("Camera stream released.")

    def is_opened(self):
        """
        Checks if camera stream is open and active.
        """
        if self.source_type == "synthetic":
            return True
        if self.http_reader and self.http_reader.is_opened():
            return True
        return self.cap is not None and self.cap.isOpened()

    def get_source_info(self):
        """
        Returns human-readable details about the camera stream connection.
        """
        if self.source_type == "synthetic":
            return "Synthetic Test Stream (No Physical Hardware)"
        elif self.source_type in ["droidcam_app", "webcam"]:
            return f"DroidCam Desktop App / Webcam (Index {self.camera_index})"
        elif self.source_type == "ip_camera":
            return f"IP Camera / DroidCam WiFi ({self.stream_url})"
        elif self.source_type == "video":
            return f"Video File ({self.video_path})"
        return "Unknown"
