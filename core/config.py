"""
Configuration management for CageMetrics.

Handles user preferences, telemetry settings, and persistent storage.
"""

import os
import uuid
import json
from pathlib import Path


def get_config_dir() -> Path:
    """
    Get the configuration directory for CageMetrics.

    Returns:
        Path to config directory (created if doesn't exist)
    """
    if os.name == 'nt':  # Windows
        config_dir = Path(os.environ.get('APPDATA', '~')) / 'CageMetrics'
    else:  # macOS/Linux
        config_dir = Path.home() / '.config' / 'cagemetrics'

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _get_config_file() -> Path:
    """Get the path to the config file."""
    return get_config_dir() / 'config.json'


def _load_config() -> dict:
    """Load configuration from file."""
    config_file = _get_config_file()
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_config(config: dict):
    """Save configuration to file."""
    config_file = _get_config_file()
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save config: {e}")


def get_user_id() -> str:
    """
    Get or create a unique anonymous user ID.

    This is used for telemetry to count unique users without
    collecting any personal information.

    Returns:
        Anonymous UUID string
    """
    config = _load_config()

    if 'user_id' not in config:
        config['user_id'] = str(uuid.uuid4())
        _save_config(config)

    return config['user_id']


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled."""
    config = _load_config()
    return config.get('telemetry_enabled', True)  # Default to enabled


def set_telemetry_enabled(enabled: bool):
    """Enable or disable telemetry."""
    config = _load_config()
    config['telemetry_enabled'] = enabled
    _save_config(config)


def is_crash_reports_enabled() -> bool:
    """Check if crash reports are enabled."""
    config = _load_config()
    return config.get('crash_reports_enabled', True)  # Default to enabled


def set_crash_reports_enabled(enabled: bool):
    """Enable or disable crash reports."""
    config = _load_config()
    config['crash_reports_enabled'] = enabled
    _save_config(config)


def is_first_launch() -> bool:
    """Check if this is the first launch of the application."""
    config = _load_config()
    return not config.get('first_launch_completed', False)


def set_first_launch_completed():
    """Mark first launch as completed."""
    config = _load_config()
    config['first_launch_completed'] = True
    _save_config(config)


def get_last_update_check() -> str:
    """Get the timestamp of the last update check."""
    config = _load_config()
    return config.get('last_update_check', '')


def set_last_update_check(timestamp: str):
    """Set the timestamp of the last update check."""
    config = _load_config()
    config['last_update_check'] = timestamp
    _save_config(config)
