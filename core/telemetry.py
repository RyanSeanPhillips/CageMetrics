"""
Anonymous usage tracking and telemetry for CageMetrics.

Uses Google Analytics 4 for usage statistics.

No personal information, file names, or experimental data is collected.
"""

import sys
import platform
import json
import threading
from datetime import datetime

from core.config import (
    get_config_dir,
    get_user_id,
    is_telemetry_enabled
)
from version_info import VERSION_STRING


# ============================================================================
# CONFIGURATION - Add your GA4 credentials here
# ============================================================================

# Google Analytics 4 Measurement Protocol
# Get these from: https://analytics.google.com/
# Admin -> Data Streams -> Your stream -> Measurement Protocol API secrets
GA4_MEASUREMENT_ID = "G-8S60JNR9HY"
GA4_API_SECRET = "Aq51rAIZQpSMbSSoSnM1eQ"


# ============================================================================
# Global telemetry state
# ============================================================================

_telemetry_initialized = False
_geo_data = None
_geo_fetch_attempted = False
_user_ip = None

_session_data = {
    'files_loaded': 0,
    'consolidations_run': 0,
    'comparisons_run': 0,
    'exports': {},
    'features_used': set(),
    'session_start': None,
    'last_engagement_time': None,
    'total_engagement_time_ms': 0,
}


# ============================================================================
# Initialization
# ============================================================================

def init_telemetry():
    """
    Initialize telemetry system.

    Call this once at app startup (after first-launch dialog).
    """
    global _telemetry_initialized

    if not is_telemetry_enabled():
        return

    try:
        _session_data['session_start'] = datetime.now().isoformat()
        _telemetry_initialized = True

        # Fetch geolocation in background
        geo_thread = threading.Thread(target=_fetch_geolocation, daemon=True)
        geo_thread.start()

        # Sync any cached events
        sync_cached_events()

        # Send session start event
        log_event('session_start', {
            'version': VERSION_STRING,
            'platform': sys.platform,
            'python_version': platform.python_version()
        })

        print("Telemetry: Initialized")

    except Exception as e:
        print(f"Warning: Could not initialize telemetry: {e}")


def _update_engagement_time():
    """Update engagement time tracking."""
    import time

    current_time = time.time()
    last_time = _session_data.get('last_engagement_time')

    if last_time is None:
        _session_data['last_engagement_time'] = current_time
        return 0

    elapsed_ms = int((current_time - last_time) * 1000)
    elapsed_ms = min(elapsed_ms, 60000)

    _session_data['last_engagement_time'] = current_time
    _session_data['total_engagement_time_ms'] += elapsed_ms

    return elapsed_ms


# ============================================================================
# IP Geolocation (for GA4 geographic reports)
# ============================================================================

def _fetch_geolocation():
    """Fetch user's approximate geographic location."""
    global _geo_data, _geo_fetch_attempted, _user_ip

    if _geo_data is not None or _geo_fetch_attempted:
        return _geo_data

    _geo_fetch_attempted = True

    try:
        import requests

        response = requests.get(
            'http://ip-api.com/json/?fields=status,query,country,countryCode,regionName,city',
            timeout=3
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                _user_ip = data.get('query')
                _geo_data = {
                    'country': data.get('country', ''),
                    'country_code': data.get('countryCode', ''),
                    'region': data.get('regionName', ''),
                    'city': data.get('city', '')
                }
                return _geo_data

    except Exception as e:
        print(f"Telemetry: Could not fetch geolocation: {e}")

    return None


def _get_geo_params():
    """Get geographic parameters for GA4 events."""
    geo = _fetch_geolocation()

    if geo is None:
        return {}

    params = {}
    if geo.get('country'):
        params['geo_country'] = geo['country']
    if geo.get('country_code'):
        params['geo_country_code'] = geo['country_code']
    if geo.get('region'):
        params['geo_region'] = geo['region']
    if geo.get('city'):
        params['geo_city'] = geo['city']

    return params


# ============================================================================
# Google Analytics 4 Integration
# ============================================================================

def _send_to_google_analytics(event_name, params=None):
    """Send event to Google Analytics 4 in background thread."""
    if not GA4_MEASUREMENT_ID or not GA4_API_SECRET:
        _log_event_locally({
            'event': event_name,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
        return

    thread = threading.Thread(
        target=_send_to_ga4_blocking,
        args=(event_name, params),
        daemon=True
    )
    thread.start()


def _send_to_ga4_blocking(event_name, params):
    """Actually send event to GA4 (runs in background thread)."""
    try:
        import requests

        url = f"https://www.google-analytics.com/mp/collect?measurement_id={GA4_MEASUREMENT_ID}&api_secret={GA4_API_SECRET}"

        payload = {
            "client_id": get_user_id(),
            "events": [{
                "name": event_name,
                "params": params or {}
            }]
        }

        headers = {'Content-Type': 'application/json'}
        if _user_ip:
            headers['X-Forwarded-For'] = _user_ip

        response = requests.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()

        if response.status_code in (200, 204):
            print(f"Telemetry: Sent '{event_name}' to GA4")

    except ImportError:
        print("Telemetry: 'requests' module not available")
        _log_event_locally({
            'event': event_name,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Telemetry: GA4 send failed: {e}")
        _log_event_locally({
            'event': event_name,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })


def _log_event_locally(event):
    """Log event to local file (fallback when GA4 unavailable)."""
    try:
        log_file = get_config_dir() / 'telemetry.log'
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    except Exception:
        pass


def sync_cached_events():
    """Upload any cached events from telemetry.log to GA4."""
    if not GA4_MEASUREMENT_ID or not GA4_API_SECRET:
        return

    try:
        log_file = get_config_dir() / 'telemetry.log'

        if not log_file.exists():
            return

        with open(log_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            log_file.unlink()
            return

        uploaded = 0
        for line in lines:
            try:
                event_data = json.loads(line.strip())
                event_name = event_data.get('event')
                params = event_data.get('params', {})

                if event_name:
                    _send_to_google_analytics(event_name, params)
                    uploaded += 1
            except Exception:
                continue

        log_file.unlink()

        if uploaded > 0:
            print(f"Telemetry: Synced {uploaded} cached event(s)")

    except Exception as e:
        print(f"Telemetry: Could not sync cached events: {e}")


# ============================================================================
# Public API - Usage Statistics
# ============================================================================

def log_event(event_name, params=None):
    """
    Log a usage event to Google Analytics 4.

    Args:
        event_name (str): Event name (e.g., "file_loaded", "consolidation_run")
        params (dict, optional): Event parameters (must not contain PII)
    """
    if not is_telemetry_enabled():
        return

    try:
        if params is None:
            params = {}

        engagement_time_ms = _update_engagement_time()
        if engagement_time_ms > 0:
            params['engagement_time_msec'] = engagement_time_ms

        params['app_version'] = VERSION_STRING
        params['platform'] = sys.platform

        geo_params = _get_geo_params()
        params.update(geo_params)

        _send_to_google_analytics(event_name, params)

    except Exception as e:
        print(f"Warning: Telemetry event failed: {e}")


def log_file_loaded(file_type, animal_count=0, **extra_params):
    """
    Log that files were loaded.

    Args:
        file_type (str): File type ('npz', 'xlsx', 'consolidated_npz')
        animal_count (int): Number of animals/files loaded
    """
    _session_data['files_loaded'] += 1

    params = {
        'file_type': file_type,
        'animal_count': animal_count
    }
    params.update(extra_params)

    log_event('file_loaded', params)


def log_consolidation_run(animal_count, filter_criteria=None, **extra_params):
    """
    Log that a consolidation was run.

    Args:
        animal_count (int): Number of animals consolidated
        filter_criteria (str, optional): Description of filters applied
    """
    _session_data['consolidations_run'] += 1

    params = {
        'animal_count': animal_count,
        'filter_criteria': filter_criteria or 'none'
    }
    params.update(extra_params)

    log_event('consolidation_run', params)


def log_comparison_run(dataset_count, **extra_params):
    """
    Log that a comparison was run.

    Args:
        dataset_count (int): Number of datasets compared
    """
    _session_data['comparisons_run'] += 1

    params = {
        'dataset_count': dataset_count
    }
    params.update(extra_params)

    log_event('comparison_run', params)


def log_export(export_type, **extra_params):
    """
    Log that data was exported.

    Args:
        export_type (str): Export type ('pdf', 'xlsx', 'npz', 'csv')
    """
    _session_data['exports'][export_type] = \
        _session_data['exports'].get(export_type, 0) + 1

    params = {'export_type': export_type}
    params.update(extra_params)

    log_event('export', params)


def log_feature_used(feature_name, **extra_params):
    """
    Log that a feature was used.

    Args:
        feature_name (str): Feature identifier
            Examples: 'statistics_enabled', 'filter_applied', 'metadata_edited'
    """
    _session_data['features_used'].add(feature_name)

    params = {'feature': feature_name}
    params.update(extra_params)

    log_event('feature_used', params)


def log_screen_view(screen_name, **extra_params):
    """
    Log when user views a screen/tab.

    Args:
        screen_name (str): Name of the screen/tab
    """
    params = {
        'screen_name': screen_name,
        'firebase_screen': screen_name
    }
    params.update(extra_params)

    log_event('screen_view', params)


def log_error(error, context=None):
    """
    Log an error or exception.

    Args:
        error (Exception): Exception object
        context (dict, optional): Additional context
    """
    if not is_telemetry_enabled():
        return

    try:
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error)[:100],
        }
        if context:
            error_data.update(context)

        log_event('error', error_data)

    except Exception:
        pass


def log_session_end():
    """Log session summary when app closes."""
    if not is_telemetry_enabled():
        return

    try:
        if _session_data['session_start']:
            start = datetime.fromisoformat(_session_data['session_start'])
            duration_minutes = (datetime.now() - start).total_seconds() / 60
        else:
            duration_minutes = 0

        log_event('session_end', {
            'session_duration_minutes': round(duration_minutes, 1),
            'files_loaded': _session_data['files_loaded'],
            'consolidations_run': _session_data['consolidations_run'],
            'comparisons_run': _session_data['comparisons_run'],
            'exports_count': sum(_session_data['exports'].values()),
            'features_used_count': len(_session_data['features_used'])
        })

    except Exception as e:
        print(f"Warning: Could not log session end: {e}")


def is_active():
    """Check if telemetry is initialized and enabled."""
    return _telemetry_initialized and is_telemetry_enabled()
