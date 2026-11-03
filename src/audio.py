"""
Audio Notification & Collision Alert Sound System for NightVisionAI
Synthesizes and delivers audible warning chimes and alarms for ADAS proximity alerts.
Supports both in-browser playback (Streamlit HTML5 / Web Audio API) and local desktop playback.
"""

import os
import io
import time
import math
import struct
import wave
import base64
import threading
from .utils import logger

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False


def synthesize_wav_bytes(pattern="critical", sample_rate=22050):
    """
    Synthesizes standard PCM WAV bytes in-memory for automotive alert chimes.
    'critical': 3 rapid urgent collision alarm beeps (1050 Hz)
    'warning': 2-tone melodic proximity warning chime (850 Hz -> 1250 Hz)
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        if pattern == "critical":
            # 3 rapid urgent collision alarm beeps (1050 Hz)
            for _ in range(3):
                n_on = int(sample_rate * 0.08)
                for i in range(n_on):
                    t = float(i) / sample_rate
                    env = min(1.0, (n_on - i) / (sample_rate * 0.015))
                    sample = int(32767.0 * 0.55 * env * math.sin(2.0 * math.pi * 1050.0 * t))
                    wf.writeframes(struct.pack("<h", sample))
                n_off = int(sample_rate * 0.035)
                for _ in range(n_off):
                    wf.writeframes(struct.pack("<h", 0))
        else:
            # 2-tone melodic proximity warning chime (850 Hz -> 1250 Hz)
            tones = [(850, 0.11, 0.50), (1250, 0.16, 0.60)]
            for idx, (freq, dur, amp) in enumerate(tones):
                n_on = int(sample_rate * dur)
                for i in range(n_on):
                    t = float(i) / sample_rate
                    env = math.exp(-5.5 * (t / dur))
                    sample = int(32767.0 * amp * env * math.sin(2.0 * math.pi * freq * t))
                    wf.writeframes(struct.pack("<h", sample))
                if idx == 0:
                    for _ in range(int(sample_rate * 0.03)):
                        wf.writeframes(struct.pack("<h", 0))

    return buf.getvalue()


class CollisionAudioAlert:
    """
    Manages collision sound notification chimes, volume, throttling,
    browser HTML5 audio embedding, and desktop native playback.
    """
    def __init__(self, config=None):
        config = config or {}
        audio_cfg = config.get("audio", {})

        self.enabled = audio_cfg.get("enabled", True)
        self.cooldown_seconds = float(audio_cfg.get("cooldown_seconds", 1.5))
        self.critical_only = audio_cfg.get("critical_only", False)
        self.volume = float(audio_cfg.get("volume", 0.8))

        self.last_played_time = 0.0
        self.crit_wav_path = os.path.join("data", "sounds", "collision_alert.wav")
        self.warn_wav_path = os.path.join("data", "sounds", "proximity_warning.wav")

        # Cache base64 audio strings in memory
        self._crit_b64 = self._load_or_synthesize_b64(self.crit_wav_path, "critical")
        self._warn_b64 = self._load_or_synthesize_b64(self.warn_wav_path, "warning")

        logger.info(
            f"CollisionAudioAlert initialized: enabled={self.enabled}, "
            f"cooldown={self.cooldown_seconds}s, crit_only={self.critical_only}"
        )

    def _load_or_synthesize_b64(self, path, pattern):
        """Loads WAV from disk or synthesizes in-memory, caching as base64 string."""
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    raw = f.read()
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                raw = synthesize_wav_bytes(pattern)
                try:
                    with open(path, "wb") as f:
                        f.write(raw)
                except Exception:
                    pass
            return base64.b64encode(raw).decode("ascii")
        except Exception as e:
            logger.warning(f"Audio sound cache fallback ({e})")
            raw = synthesize_wav_bytes(pattern)
            return base64.b64encode(raw).decode("ascii")

    def should_play(self, alerts):
        """
        Determines whether an audible alert should be triggered given active alerts and cooldown throttle.
        Returns:
            tuple: (bool should_play, str severity)
        """
        if not self.enabled or not alerts:
            return False, None

        has_critical = any(a.get("severity") == "CRITICAL" for a in alerts)
        has_warning = any(a.get("severity") == "WARNING" for a in alerts)

        if not has_critical and not has_warning:
            return False, None

        if self.critical_only and not has_critical:
            return False, None

        severity = "CRITICAL" if has_critical else "WARNING"
        now = time.time()

        if now - self.last_played_time >= self.cooldown_seconds:
            self.last_played_time = now
            return True, severity

        return False, None

    def get_browser_audio_html(self, severity="CRITICAL"):
        """
        Generates an HTML/JavaScript audio snippet for browser playback in Streamlit.
        Uses dual-mode Web Audio API synthesis + HTML5 <audio autoplay> fallback.
        """
        b64 = self._crit_b64 if severity == "CRITICAL" else self._warn_b64
        nonce = int(time.time() * 1000)
        freq = 1050 if severity == "CRITICAL" else 950
        dur = 0.25 if severity == "CRITICAL" else 0.30
        vol = min(1.0, max(0.1, self.volume))

        html = f"""
        <div id="audio-alert-{nonce}" style="display:none; height:0; width:0;">
            <audio autoplay playsinline style="display:none;">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            <script>
            (function() {{
                try {{
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    if (AudioContext) {{
                        const ctx = new AudioContext();
                        if (ctx.state === 'suspended') {{
                            ctx.resume();
                        }}
                        const osc = ctx.createOscillator();
                        const gain = ctx.createGain();
                        osc.type = 'sine';
                        osc.frequency.setValueAtTime({freq}, ctx.currentTime);
                        gain.gain.setValueAtTime({vol * 0.4}, ctx.currentTime);
                        gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + {dur});
                        osc.connect(gain);
                        gain.connect(ctx.destination);
                        osc.start();
                        osc.stop(ctx.currentTime + {dur});
                    }}
                }} catch(e) {{}}
            }})();
            </script>
        </div>
        """
        return html

    def play_desktop_sound(self, severity="CRITICAL"):
        """
        Plays notification sound asynchronously on local desktop speakers (for main.py).
        """
        if not self.enabled:
            return

        def _worker():
            try:
                wav_path = self.crit_wav_path if severity == "CRITICAL" else self.warn_wav_path
                if WINSOUND_AVAILABLE and os.path.exists(wav_path):
                    winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
                elif WINSOUND_AVAILABLE:
                    freq = 1050 if severity == "CRITICAL" else 850
                    dur = 200 if severity == "CRITICAL" else 150
                    winsound.Beep(freq, dur)
                else:
                    print("\a", end="", flush=True)
            except Exception as e:
                logger.debug(f"Desktop audio playback skipped: {e}")

        threading.Thread(target=_worker, daemon=True).start()
