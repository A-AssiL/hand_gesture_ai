"""Tests for configuration loading."""
import tempfile
import unittest
from pathlib import Path

from config.settings import Config, load_config


class LoadConfigTest(unittest.TestCase):
    def test_defaults_when_file_missing(self) -> None:
        cfg = load_config(Path("/nonexistent/config.yaml"))
        self.assertIsInstance(cfg, Config)
        self.assertEqual(cfg.camera.width, 1280)
        self.assertEqual(cfg.hand_tracking.max_num_hands, 2)

    def test_loads_and_overrides_from_yaml(self) -> None:
        content = (
            "camera:\n"
            "  index: 2\n"
            "  width: 640\n"
            "  height: 480\n"
            "hand_tracking:\n"
            "  max_num_hands: 1\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual(cfg.camera.index, 2)
        self.assertEqual(cfg.camera.width, 640)
        self.assertEqual(cfg.hand_tracking.max_num_hands, 1)
        # Untouched sections keep defaults.
        self.assertTrue(cfg.display.draw_landmarks)

    def test_unknown_keys_are_ignored(self) -> None:
        content = "camera:\n  index: 0\n  bogus_key: 123\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            cfg = load_config(path)
        self.assertEqual(cfg.camera.index, 0)


if __name__ == "__main__":
    unittest.main()
