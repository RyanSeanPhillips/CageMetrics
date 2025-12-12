"""
Parallel processing utilities for behavioral analysis.

Uses concurrent.futures for parallel processing of animals and metrics.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, List, Optional, Callable
import multiprocessing


def get_optimal_workers() -> int:
    """Get optimal number of worker threads/processes."""
    cpu_count = multiprocessing.cpu_count()
    # Use half of available CPUs to avoid overwhelming the system
    return max(1, min(cpu_count // 2, 4))


def parallel_map(func: Callable, items: List, max_workers: int = None,
                use_threads: bool = True) -> List:
    """
    Apply function to items in parallel.

    Args:
        func: Function to apply
        items: List of items to process
        max_workers: Maximum number of workers (default: auto)
        use_threads: Use threads (True) or processes (False)

    Returns:
        List of results in same order as items
    """
    if max_workers is None:
        max_workers = get_optimal_workers()

    # For small lists, just use sequential processing
    if len(items) <= 2:
        return [func(item) for item in items]

    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))

    return results


def vectorized_cta(data_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute CTA statistics using vectorized operations.

    Args:
        data_matrix: 2D array of shape (n_days, n_minutes)

    Returns:
        Dictionary with cta, std, and sem arrays
    """
    n_days = data_matrix.shape[0]

    with np.errstate(invalid='ignore'):
        cta = np.nanmean(data_matrix, axis=0)
        cta_std = np.nanstd(data_matrix, axis=0)
        cta_sem = cta_std / np.sqrt(n_days)

    return {
        'cta': cta,
        'cta_std': cta_std,
        'cta_sem': cta_sem
    }


def vectorized_align_to_zt(data: np.ndarray, zt0_minute: int) -> np.ndarray:
    """
    Align data array to ZT time using vectorized roll.

    Args:
        data: Array of values
        zt0_minute: Minute when ZT0 occurs

    Returns:
        Array aligned so index 0 = ZT0
    """
    if zt0_minute == 0:
        return data.copy() if isinstance(data, np.ndarray) else np.array(data)

    return np.roll(data, -zt0_minute)


def batch_align_to_zt(data_list: List[np.ndarray], zt0_minute: int) -> List[np.ndarray]:
    """
    Align multiple arrays to ZT time.

    Args:
        data_list: List of arrays to align
        zt0_minute: Minute when ZT0 occurs

    Returns:
        List of aligned arrays
    """
    if zt0_minute == 0:
        return [d.copy() for d in data_list]

    return [np.roll(d, -zt0_minute) for d in data_list]


def compute_dark_light_means(cta_zt: np.ndarray, light_cycle_zt: np.ndarray) -> Dict[str, float]:
    """
    Compute dark and light cycle means from ZT-aligned data.

    Args:
        cta_zt: ZT-aligned CTA array
        light_cycle_zt: ZT-aligned light cycle labels

    Returns:
        Dictionary with dark_mean and light_mean
    """
    dark_mask = light_cycle_zt == 'Dark'
    light_mask = light_cycle_zt == 'Light'

    dark_mean = np.nanmean(cta_zt[dark_mask]) if np.any(dark_mask) else np.nan
    light_mean = np.nanmean(cta_zt[light_mask]) if np.any(light_mask) else np.nan

    return {
        'dark_mean': dark_mean,
        'light_mean': light_mean
    }
