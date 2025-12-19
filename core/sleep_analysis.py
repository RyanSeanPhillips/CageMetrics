"""
Sleep bout analysis module for CageMetrics.

Analyzes sleep fragmentation from the sleeping % metric by detecting
sleep bouts (threshold crossings) and computing sleep quality metrics.

Key metrics:
- Total sleep time (light/dark)
- Number of sleep bouts (fragmentation)
- Bout duration statistics (mean, median, max)
- Bout duration histograms
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class SleepBout:
    """Represents a single sleep bout."""
    day: int
    bout_num: int
    start_minute: int  # ZT minute (0-1439)
    end_minute: int    # ZT minute (0-1439)
    duration: float    # Duration in minutes
    phase: str         # 'light' or 'dark'


def detect_bouts_single_day(
    trace: np.ndarray,
    threshold: float = 0.5,
    day_index: int = 0
) -> List[SleepBout]:
    """
    Detect sleep bouts in a single day's trace using threshold crossings.

    Args:
        trace: 1D array of 1440 values (one per minute), range 0-1
        threshold: Value above which is considered "asleep"
        day_index: Day number (for labeling bouts)

    Returns:
        List of SleepBout objects
    """
    if len(trace) != 1440:
        raise ValueError(f"Expected 1440 values, got {len(trace)}")

    # Handle NaN values - treat as "awake"
    trace_clean = np.nan_to_num(trace, nan=0.0)

    # Normalize if values are 0-100 instead of 0-1
    if np.nanmax(trace_clean) > 1.5:
        trace_clean = trace_clean / 100.0

    # Binary sleep state
    is_asleep = trace_clean >= threshold

    bouts = []
    bout_num = 0
    in_bout = False
    bout_start = 0

    for minute in range(1440):
        if is_asleep[minute] and not in_bout:
            # Bout starts
            in_bout = True
            bout_start = minute
        elif not is_asleep[minute] and in_bout:
            # Bout ends
            in_bout = False
            bout_end = minute
            duration = bout_end - bout_start

            # Determine phase based on bout midpoint
            midpoint = (bout_start + bout_end) / 2
            phase = 'light' if midpoint < 720 else 'dark'

            bouts.append(SleepBout(
                day=day_index,
                bout_num=bout_num,
                start_minute=bout_start,
                end_minute=bout_end,
                duration=duration,
                phase=phase
            ))
            bout_num += 1

    # Handle bout that extends to end of day
    if in_bout:
        bout_end = 1440
        duration = bout_end - bout_start
        midpoint = (bout_start + bout_end) / 2
        phase = 'light' if midpoint < 720 else 'dark'

        bouts.append(SleepBout(
            day=day_index,
            bout_num=bout_num,
            start_minute=bout_start,
            end_minute=bout_end,
            duration=duration,
            phase=phase
        ))

    return bouts


def detect_bouts_all_days(
    daily_data: np.ndarray,
    threshold: float = 0.5
) -> List[SleepBout]:
    """
    Detect sleep bouts across all days.

    Args:
        daily_data: 2D array of shape (n_days, 1440)
        threshold: Sleep detection threshold (0-1)

    Returns:
        List of all SleepBout objects across all days
    """
    all_bouts = []
    n_days = daily_data.shape[0]

    for day_idx in range(n_days):
        day_trace = daily_data[day_idx, :]
        day_bouts = detect_bouts_single_day(day_trace, threshold, day_idx)
        all_bouts.extend(day_bouts)

    return all_bouts


def compute_phase_stats(bouts: List[SleepBout], phase: str, n_days: int = None) -> Dict[str, float]:
    """
    Compute statistics for bouts in a specific phase.

    Args:
        bouts: List of all bouts
        phase: 'light', 'dark', or 'all'
        n_days: Total number of recording days (used for accurate % calculation)

    Returns:
        Dictionary with statistics
    """
    if phase == 'all':
        phase_bouts = bouts
    else:
        phase_bouts = [b for b in bouts if b.phase == phase]

    if not phase_bouts:
        return {
            'total_minutes': 0.0,
            'percent_time': 0.0,
            'bout_count': 0,
            'mean_duration': 0.0,
            'median_duration': 0.0,
            'max_duration': 0.0,
            'min_duration': 0.0,
            'std_duration': 0.0,
        }

    durations = [b.duration for b in phase_bouts]
    total_minutes = sum(durations)

    # Calculate percent time
    # Use provided n_days if available, otherwise infer from bouts
    if n_days is None:
        days = set(b.day for b in phase_bouts)
        n_days = len(days) if days else 1

    # Use 720 minutes (12 hours) per phase for light/dark,
    # but 1440 minutes (24 hours) for total ('all')
    if phase == 'all':
        phase_minutes = 1440 * n_days  # Full day
    else:
        phase_minutes = 720 * n_days  # 12 hours per phase per day

    percent_time = (total_minutes / phase_minutes) * 100 if phase_minutes > 0 else 0

    return {
        'total_minutes': total_minutes,
        'percent_time': percent_time,
        'bout_count': len(phase_bouts),
        'mean_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'max_duration': np.max(durations),
        'min_duration': np.min(durations),
        'std_duration': np.std(durations) if len(durations) > 1 else 0.0,
    }


def compute_histogram(
    bouts: List[SleepBout],
    bin_width: float = 5.0,
    max_duration: float = None,
    phase: str = 'all'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of bout durations.

    Args:
        bouts: List of sleep bouts
        bin_width: Width of histogram bins in minutes
        max_duration: Maximum duration for histogram (auto if None)
        phase: 'light', 'dark', or 'all'

    Returns:
        Tuple of (bin_edges, counts)
    """
    if phase == 'all':
        durations = [b.duration for b in bouts]
    else:
        durations = [b.duration for b in bouts if b.phase == phase]

    if not durations:
        return np.array([0, bin_width]), np.array([0])

    if max_duration is None:
        max_duration = max(durations) + bin_width

    # Create bins
    bins = np.arange(0, max_duration + bin_width, bin_width)
    counts, bin_edges = np.histogram(durations, bins=bins)

    return bin_edges, counts


def compute_per_day_stats(bouts: List[SleepBout], n_days: int) -> List[Dict[str, Any]]:
    """
    Compute statistics broken down by day.

    Args:
        bouts: List of all bouts
        n_days: Total number of days

    Returns:
        List of dictionaries, one per day
    """
    per_day = []

    for day_idx in range(n_days):
        day_bouts = [b for b in bouts if b.day == day_idx]

        light_stats = compute_phase_stats(
            [b for b in day_bouts if b.phase == 'light'], 'all'
        )
        dark_stats = compute_phase_stats(
            [b for b in day_bouts if b.phase == 'dark'], 'all'
        )

        per_day.append({
            'day': day_idx + 1,
            'light': light_stats,
            'dark': dark_stats,
            'total_bouts': len(day_bouts),
            'total_sleep_minutes': light_stats['total_minutes'] + dark_stats['total_minutes'],
        })

    return per_day


def analyze_sleep(
    daily_data: np.ndarray,
    threshold: float = 0.5,
    bin_width: float = 5.0
) -> Dict[str, Any]:
    """
    Perform complete sleep bout analysis.

    Args:
        daily_data: 2D array of shape (n_days, 1440) with sleeping % values
        threshold: Sleep detection threshold (0-1)
        bin_width: Histogram bin width in minutes

    Returns:
        Dictionary containing:
        - 'bouts': List of SleepBout objects
        - 'light_stats': Statistics for light phase
        - 'dark_stats': Statistics for dark phase
        - 'total_stats': Combined statistics
        - 'per_day_stats': List of per-day statistics
        - 'histogram_light': (bin_edges, counts) for light phase
        - 'histogram_dark': (bin_edges, counts) for dark phase
        - 'histogram_combined': (bin_edges, counts) for all bouts
        - 'per_day_histograms': List of (bin_edges, counts) per day
        - 'parameters': {'threshold': ..., 'bin_width': ...}
    """
    if daily_data is None or daily_data.size == 0:
        return None

    # Ensure 2D
    if daily_data.ndim == 1:
        daily_data = daily_data.reshape(1, -1)

    n_days = daily_data.shape[0]

    # Detect all bouts
    bouts = detect_bouts_all_days(daily_data, threshold)

    # Compute statistics by phase (pass n_days for accurate % calculation)
    light_stats = compute_phase_stats(bouts, 'light', n_days)
    dark_stats = compute_phase_stats(bouts, 'dark', n_days)
    total_stats = compute_phase_stats(bouts, 'all', n_days)

    # Per-day stats
    per_day_stats = compute_per_day_stats(bouts, n_days)

    # Find max duration for consistent histogram scaling
    if bouts:
        max_dur = max(b.duration for b in bouts)
        max_dur = np.ceil(max_dur / bin_width) * bin_width + bin_width
    else:
        max_dur = 60  # Default to 60 minutes

    # Compute histograms
    hist_light = compute_histogram(bouts, bin_width, max_dur, 'light')
    hist_dark = compute_histogram(bouts, bin_width, max_dur, 'dark')
    hist_combined = compute_histogram(bouts, bin_width, max_dur, 'all')

    # Per-day histograms
    per_day_histograms = []
    for day_idx in range(n_days):
        day_bouts = [b for b in bouts if b.day == day_idx]
        hist = compute_histogram(day_bouts, bin_width, max_dur, 'all')
        per_day_histograms.append(hist)

    # Compute additional quality metrics
    quality_metrics = compute_quality_metrics(bouts, daily_data, light_stats, dark_stats, threshold)

    return {
        'bouts': bouts,
        'light_stats': light_stats,
        'dark_stats': dark_stats,
        'total_stats': total_stats,
        'per_day_stats': per_day_stats,
        'quality_metrics': quality_metrics,
        'histogram_light': hist_light,
        'histogram_dark': hist_dark,
        'histogram_combined': hist_combined,
        'per_day_histograms': per_day_histograms,
        'n_days': n_days,
        'parameters': {
            'threshold': threshold,
            'bin_width': bin_width,
        }
    }


def compute_long_bout_percentage(bouts: List[SleepBout], threshold_minutes: float = 10.0,
                                  phase: str = 'all') -> float:
    """
    Compute percentage of total sleep time occurring in long bouts.

    Args:
        bouts: List of sleep bouts
        threshold_minutes: Minimum duration to count as "long" (default 10 min)
        phase: 'light', 'dark', or 'all'

    Returns:
        Percentage of total sleep in long bouts (0-100)
    """
    if phase == 'all':
        phase_bouts = bouts
    else:
        phase_bouts = [b for b in bouts if b.phase == phase]

    if not phase_bouts:
        return 0.0

    total_sleep = sum(b.duration for b in phase_bouts)
    long_bout_sleep = sum(b.duration for b in phase_bouts if b.duration >= threshold_minutes)

    return (long_bout_sleep / total_sleep) * 100 if total_sleep > 0 else 0.0


def compute_sleep_stability(daily_data: np.ndarray) -> float:
    """
    Compute Sleep Stability Index from continuous sleep data.

    Stability = mean(|s_t - s_{t-1}|)

    Lower values indicate more stable/consolidated sleep-wake states.
    Higher values indicate frequent state flickering or instability.

    Args:
        daily_data: 2D array of shape (n_days, 1440) with sleep values 0-1

    Returns:
        Stability index (0-1 scale, lower = more stable)
    """
    if daily_data is None or daily_data.size == 0:
        return 0.0

    # Flatten all days into continuous trace
    flat = daily_data.flatten()

    # Handle NaN values
    flat = np.nan_to_num(flat, nan=0.0)

    # Normalize if 0-100 scale
    if np.max(flat) > 1.5:
        flat = flat / 100.0

    # Compute mean absolute difference between consecutive minutes
    diffs = np.abs(np.diff(flat))
    stability = np.mean(diffs)

    return float(stability)


def compute_transition_rate(daily_data: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute sleep-wake transition rate (transitions per hour).

    Args:
        daily_data: 2D array of shape (n_days, 1440) with sleep values
        threshold: Sleep detection threshold

    Returns:
        Transitions per hour
    """
    if daily_data is None or daily_data.size == 0:
        return 0.0

    # Flatten and binarize
    flat = daily_data.flatten()
    flat = np.nan_to_num(flat, nan=0.0)

    # Normalize if 0-100 scale
    if np.max(flat) > 1.5:
        flat = flat / 100.0

    # Binarize
    binary = (flat >= threshold).astype(int)

    # Count transitions
    transitions = np.sum(np.abs(np.diff(binary)))

    # Total hours
    total_hours = len(flat) / 60.0

    return transitions / total_hours if total_hours > 0 else 0.0


def compute_quality_metrics(bouts: List[SleepBout], daily_data: np.ndarray,
                            light_stats: Dict, dark_stats: Dict,
                            threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute additional sleep quality metrics.

    Returns:
        Dictionary with:
        - long_bout_pct_light: % of light sleep in bouts ≥10 min
        - long_bout_pct_dark: % of dark sleep in bouts ≥10 min
        - long_bout_pct_total: % of total sleep in bouts ≥10 min
        - light_dark_ratio: TST_light / TST_dark
        - stability_index: Mean absolute change in sleep signal (lower = more stable)
        - transition_rate: Sleep-wake transitions per hour
    """
    # Long bout percentages
    long_bout_pct_light = compute_long_bout_percentage(bouts, 10.0, 'light')
    long_bout_pct_dark = compute_long_bout_percentage(bouts, 10.0, 'dark')
    long_bout_pct_total = compute_long_bout_percentage(bouts, 10.0, 'all')

    # Light/Dark ratio
    tst_light = light_stats.get('total_minutes', 0)
    tst_dark = dark_stats.get('total_minutes', 0)
    light_dark_ratio = tst_light / tst_dark if tst_dark > 0 else float('inf') if tst_light > 0 else 0.0

    # Stability index (threshold-independent)
    stability_index = compute_sleep_stability(daily_data)

    # Transition rate
    transition_rate = compute_transition_rate(daily_data, threshold)

    return {
        'long_bout_pct_light': long_bout_pct_light,
        'long_bout_pct_dark': long_bout_pct_dark,
        'long_bout_pct_total': long_bout_pct_total,
        'light_dark_ratio': light_dark_ratio,
        'stability_index': stability_index,
        'transition_rate': transition_rate,
    }


def bouts_to_array(bouts: List[SleepBout]) -> np.ndarray:
    """
    Convert list of SleepBout objects to numpy array for export.

    Returns:
        2D array with columns: [day, bout_num, start_min, end_min, duration, phase_code]
        phase_code: 0 = light, 1 = dark
    """
    if not bouts:
        return np.array([]).reshape(0, 6)

    data = []
    for b in bouts:
        phase_code = 0 if b.phase == 'light' else 1
        data.append([b.day, b.bout_num, b.start_minute, b.end_minute, b.duration, phase_code])

    return np.array(data)


def bouts_to_dataframe(bouts: List[SleepBout]):
    """
    Convert bouts to pandas DataFrame for Excel export.

    Returns:
        DataFrame with bout information
    """
    import pandas as pd

    if not bouts:
        return pd.DataFrame(columns=[
            'Day', 'Bout', 'Start_ZT', 'End_ZT', 'Duration_min', 'Phase'
        ])

    data = []
    for b in bouts:
        data.append({
            'Day': b.day + 1,  # 1-indexed for display
            'Bout': b.bout_num + 1,
            'Start_ZT': b.start_minute / 60,  # Convert to ZT hours
            'End_ZT': b.end_minute / 60,
            'Start_min': b.start_minute,
            'End_min': b.end_minute,
            'Duration_min': b.duration,
            'Phase': b.phase.capitalize(),
        })

    return pd.DataFrame(data)


def stats_to_dataframe(analysis_result: Dict[str, Any]):
    """
    Convert sleep analysis statistics to DataFrame for Excel export.

    Returns:
        DataFrame with summary statistics
    """
    import pandas as pd

    if not analysis_result:
        return pd.DataFrame()

    light = analysis_result['light_stats']
    dark = analysis_result['dark_stats']
    total = analysis_result['total_stats']

    rows = [
        {'Metric': 'Total Sleep (min)', 'Light': light['total_minutes'],
         'Dark': dark['total_minutes'], 'Total': total['total_minutes']},
        {'Metric': 'Sleep %', 'Light': light['percent_time'],
         'Dark': dark['percent_time'], 'Total': total['percent_time']},
        {'Metric': 'Bout Count', 'Light': light['bout_count'],
         'Dark': dark['bout_count'], 'Total': total['bout_count']},
        {'Metric': 'Mean Duration (min)', 'Light': light['mean_duration'],
         'Dark': dark['mean_duration'], 'Total': total['mean_duration']},
        {'Metric': 'Median Duration (min)', 'Light': light['median_duration'],
         'Dark': dark['median_duration'], 'Total': total['median_duration']},
        {'Metric': 'Max Duration (min)', 'Light': light['max_duration'],
         'Dark': dark['max_duration'], 'Total': total['max_duration']},
        {'Metric': 'Min Duration (min)', 'Light': light['min_duration'],
         'Dark': dark['min_duration'], 'Total': total['min_duration']},
    ]

    return pd.DataFrame(rows)
