"""Configuration management for UniFace Unlock."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "camera": {
        "device": "/dev/video0",
        "resolution": [640, 480],
        "fps": 10,
    },
    "detection": {
        "model": "SCRFD",
        "threshold": 0.7,
        "max_faces": 1,
        "enable_landmarks": True,
        "min_face_size": 10000,
    },
    "recognition": {
        "model": "MobileFace",
        "threshold": 0.7,  # Will be calibrated
        "embedding_dim": 512,
    },
    "liveness": {
        "model": "MiniFASNet",
        "threshold": 0.5,
        "require_real": True,
    },
    "authentication": {
        "required_consecutive_frames": 3,
        "frame_skip": 2,
        "timeout_seconds": 30,
    },
    "quality": {
        "enabled": True,
        "min_laplacian_variance": 12.0,
        "min_mean_brightness": 35.0,
        "max_mean_brightness": 225.0,
        "max_clipped_fraction": 0.15,
        "min_eye_dist_ratio": 0.18,
        "max_eye_dist_ratio": 0.60,
        "landmarks_margin_px": 2.0,
    },
    "data": {
        "enrollment_path": "~/.config/uniface-unlock",
        "embedding_file": "enrollment.dat",
        "config_file": "config.json",
    },
    "performance": {
        "max_cpu_percent": 50,
        "log_latency": True,
    },
}


class Config:
    """Configuration manager with defaults and file persistence."""

    def __init__(self, config_path: str | None = None) -> None:
        if config_path is None:
            config_dir = Path.home() / ".config" / "uniface-unlock"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.json"

        self.config_path = Path(config_path).expanduser()
        self._config = DEFAULT_CONFIG.copy()
        self.load()

    def load(self) -> None:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    loaded = json.load(f)
                    self._deep_update(self._config, loaded)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        else:
            logger.info("No config file found, using defaults")
            self.save()

    def save(self) -> None:
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self._config, f, indent=2)
        logger.info(f"Configuration saved to {self.config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key: Dot-separated key path, e.g. "camera.device"
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.

        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dict."""
        return self._config.copy()

    @staticmethod
    def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep update dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_update(base[key], value)
            else:
                base[key] = value


# Global config instance
_config_instance: Config | None = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config() -> None:
    """Reset global config instance."""
    global _config_instance
    _config_instance = None
