"""
GitHub Release Update Checker for CageMetrics.

Checks for new releases on GitHub and notifies users.
"""

import urllib.request
import json
from typing import Optional, Tuple
from version_info import VERSION_STRING


# GitHub repository info
GITHUB_REPO_OWNER = "RyanSeanPhillips"
GITHUB_REPO_NAME = "CageMetrics"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest"
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases"


def parse_version(version_str: str) -> Tuple[int, ...]:
    """
    Parse semantic version string into tuple of integers.

    Args:
        version_str: Version string like "1.0.0" or "v1.0.0"

    Returns:
        Tuple of integers (1, 0, 0)
    """
    version_str = version_str.lstrip('v')

    try:
        return tuple(int(x) for x in version_str.split('.'))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def compare_versions(current: str, latest: str) -> bool:
    """
    Compare two version strings.

    Args:
        current: Current version string
        latest: Latest version string

    Returns:
        True if latest > current, False otherwise
    """
    current_tuple = parse_version(current)
    latest_tuple = parse_version(latest)

    return latest_tuple > current_tuple


def check_for_updates(timeout: float = 5.0) -> Optional[dict]:
    """
    Check GitHub for new releases.

    Args:
        timeout: Request timeout in seconds

    Returns:
        Dictionary with update info if available:
        {
            'version': '1.1.0',
            'url': 'https://github.com/.../releases/tag/v1.1.0',
            'name': 'Release Name',
            'published_at': '2025-01-07T12:00:00Z',
            'body': 'Release notes...'
        }
        Returns None if no update available or on error.
    """
    try:
        req = urllib.request.Request(GITHUB_API_URL)
        req.add_header('Accept', 'application/vnd.github.v3+json')
        req.add_header('User-Agent', f'CageMetrics/{VERSION_STRING}')

        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))

        latest_version = data.get('tag_name', '').lstrip('v')
        release_name = data.get('name', '')
        release_url = data.get('html_url', GITHUB_RELEASES_URL)
        published_at = data.get('published_at', '')
        release_notes = data.get('body', '')

        if compare_versions(VERSION_STRING, latest_version):
            return {
                'version': latest_version,
                'url': release_url,
                'name': release_name,
                'published_at': published_at,
                'body': release_notes
            }

        return None

    except urllib.error.URLError as e:
        print(f"[Update Check] Network error: {e}")
        return None
    except Exception as e:
        print(f"[Update Check] Error checking for updates: {e}")
        return None


def get_update_message(update_info: dict) -> str:
    """
    Format update info into user-friendly HTML message.

    Args:
        update_info: Dictionary from check_for_updates()

    Returns:
        Formatted HTML message
    """
    version = update_info.get('version', 'Unknown')
    name = update_info.get('name', f'Version {version}')

    message = f"""
    <div style="padding: 15px; background-color: #2a4a2a; border: 3px solid #4CAF50; border-radius: 8px; margin: 10px 0;">
        <h3 style="color: #4CAF50; margin-top: 0;">Update Available!</h3>
        <p style="margin: 8px 0;"><b>New version:</b> {version}</p>
        <p style="margin: 8px 0;"><b>Release:</b> {name}</p>
        <p style="margin-bottom: 0;">
            <a href="{update_info.get('url', GITHUB_RELEASES_URL)}" style="color: #4CAF50; font-weight: bold; text-decoration: underline;">
                Download the latest version
            </a>
        </p>
    </div>
    """
    return message


def get_main_window_update_message(update_info: dict) -> Tuple[str, str]:
    """
    Get simple update message for main window banner.

    Args:
        update_info: Dictionary from check_for_updates()

    Returns:
        Tuple of (text, url) for display
    """
    version = update_info.get('version', 'Unknown')
    url = update_info.get('url', GITHUB_RELEASES_URL)
    return (f"Update Available: v{version}", url)


def get_no_update_message() -> str:
    """Get message when no update is available."""
    return f"""
    <div style="padding: 10px; background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 5px;">
        <p style="color: #4CAF50; margin: 5px 0;"><b>You're up to date!</b></p>
        <p style="margin: 5px 0;">Current version: {VERSION_STRING}</p>
    </div>
    """


def get_check_failed_message() -> str:
    """Get message when update check fails."""
    return f"""
    <div style="padding: 10px; background-color: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 5px;">
        <p style="margin: 5px 0;">Current version: {VERSION_STRING}</p>
        <p style="margin: 5px 0; color: #888;">
            <i>Could not check for updates (network error)</i>
        </p>
        <p style="margin: 5px 0;">
            <a href="{GITHUB_RELEASES_URL}" style="color: #2a7fff;">
                View releases on GitHub
            </a>
        </p>
    </div>
    """
