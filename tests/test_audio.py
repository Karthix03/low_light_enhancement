"""
Unit Tests for CollisionAudioAlert Module
"""

import unittest
import time
from src.audio import CollisionAudioAlert, synthesize_wav_bytes

class TestCollisionAudioAlert(unittest.TestCase):

    def setUp(self):
        self.config = {
            "audio": {
                "enabled": True,
                "cooldown_seconds": 0.5,
                "critical_only": False,
                "volume": 0.8
            }
        }
        self.audio = CollisionAudioAlert(self.config)

    def test_synthesize_wav_bytes(self):
        crit_wav = synthesize_wav_bytes("critical")
        warn_wav = synthesize_wav_bytes("warning")
        self.assertGreater(len(crit_wav), 1000)
        self.assertGreater(len(warn_wav), 1000)
        # Check WAV header RIFF and WAVE
        self.assertTrue(crit_wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", crit_wav[:16])

    def test_should_play_empty_alerts(self):
        should_play, sev = self.audio.should_play([])
        self.assertFalse(should_play)
        self.assertIsNone(sev)

    def test_should_play_critical_alert(self):
        alerts = [{"severity": "CRITICAL", "message": "COLLISION IMMINENT"}]
        should_play, sev = self.audio.should_play(alerts)
        self.assertTrue(should_play)
        self.assertEqual(sev, "CRITICAL")

    def test_cooldown_throttling(self):
        alerts = [{"severity": "CRITICAL", "message": "COLLISION IMMINENT"}]
        # First trigger should succeed
        should_play, _ = self.audio.should_play(alerts)
        self.assertTrue(should_play)

        # Immediate second trigger within cooldown (0.5s) must be throttled
        should_play_2, _ = self.audio.should_play(alerts)
        self.assertFalse(should_play_2)

        # After cooldown passes
        time.sleep(0.55)
        should_play_3, _ = self.audio.should_play(alerts)
        self.assertTrue(should_play_3)

    def test_critical_only_filter(self):
        audio_crit = CollisionAudioAlert({
            "audio": {"enabled": True, "critical_only": True}
        })
        warn_alerts = [{"severity": "WARNING", "message": "PROXIMITY WARNING"}]
        should_play, _ = audio_crit.should_play(warn_alerts)
        self.assertFalse(should_play)

        crit_alerts = [{"severity": "CRITICAL", "message": "COLLISION IMMINENT"}]
        should_play_crit, sev = audio_crit.should_play(crit_alerts)
        self.assertTrue(should_play_crit)
        self.assertEqual(sev, "CRITICAL")

    def test_browser_audio_html(self):
        html = self.audio.get_browser_audio_html("CRITICAL")
        self.assertIn("<audio", html)
        self.assertIn("data:audio/wav;base64,", html)
        self.assertIn("AudioContext", html)

    def test_desktop_playback_no_crash(self):
        # Should not raise exception
        self.audio.play_desktop_sound("WARNING")

if __name__ == "__main__":
    unittest.main()
