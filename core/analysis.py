"""
Analysis module for behavioral data processing.

Handles ZT alignment, CTA computation, and statistical analysis of
behavioral metrics from Allentown cage monitoring data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from .data_loader import DataLoader
from .parallel_utils import (
    get_optimal_workers, vectorized_cta, vectorized_align_to_zt,
    batch_align_to_zt, compute_dark_light_means
)


class BehaviorAnalyzer:
    """Analyze behavioral data for individual animals."""

    MINUTES_PER_DAY = 1440
    MAX_DAYS_FOR_CTA = 7  # Use up to 7 days for CTA computation

    def __init__(self):
        self.data_loader = DataLoader()

    def analyze_all_animals(self, df: pd.DataFrame, cohort_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all animals in a dataset using parallel processing.

        Args:
            df: Full DataFrame with all animal data
            cohort_name: Name of the cohort

        Returns:
            Dictionary mapping animal_id to analysis results
        """
        results = {}

        animal_ids = self.data_loader.get_animals(df)

        # Filter valid animal IDs and prepare data
        valid_animals = []
        animal_dfs = {}

        for animal_id in animal_ids:
            if pd.isna(animal_id):
                continue

            animal_df = self.data_loader.get_animal_data(df, animal_id)

            if len(animal_df) < self.MINUTES_PER_DAY:
                print(f"Skipping {animal_id}: insufficient data ({len(animal_df)} minutes)")
                continue

            valid_animals.append(animal_id)
            animal_dfs[animal_id] = animal_df

        # Process animals in parallel using threads
        n_workers = get_optimal_workers()
        print(f"Processing {len(valid_animals)} animals using {n_workers} workers...")

        def process_animal(animal_id):
            return (animal_id, self.analyze_animal(animal_dfs[animal_id], cohort_name))

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for animal_id, analysis in executor.map(process_animal, valid_animals):
                if analysis:
                    results[animal_id] = analysis

        # Find companion animals (animals in same cage)
        self._identify_companions(results, df)

        return results

    def analyze_animal(self, animal_df: pd.DataFrame, cohort_name: str) -> Optional[Dict[str, Any]]:
        """
        Perform full analysis for a single animal.

        Args:
            animal_df: DataFrame for single animal
            cohort_name: Cohort name

        Returns:
            Dictionary with analysis results, or None if analysis failed
        """
        try:
            # Get metadata
            metadata = self.data_loader.get_animal_metadata(animal_df, cohort_name)

            # Calculate number of complete days
            n_days = min(self.MAX_DAYS_FOR_CTA, len(animal_df) // self.MINUTES_PER_DAY)
            if n_days == 0:
                return None

            # Truncate to complete days
            animal_df = animal_df.head(n_days * self.MINUTES_PER_DAY).copy()
            metadata['n_days_analyzed'] = n_days

            # Find ZT0 (lights on)
            zt0_minute = self._find_zt0(animal_df)
            metadata['zt0_minute'] = zt0_minute

            # Compute data quality
            quality = self.data_loader.compute_data_quality(animal_df)

            # Get available metrics
            available_metrics = self.data_loader.get_available_metrics(animal_df)

            # Compute per-metric analysis
            metrics_data = {}
            for metric_name, col_name in available_metrics.items():
                metric_analysis = self._analyze_metric(
                    animal_df, col_name, metric_name, n_days, zt0_minute
                )
                if metric_analysis:
                    metrics_data[metric_name] = metric_analysis

            return {
                'metadata': metadata,
                'quality': quality,
                'metrics': metrics_data,
                'genotype': metadata['genotype'],
                'sex': metadata['sex'],
                'n_days': n_days,
                'zt0_minute': zt0_minute,
            }

        except Exception as e:
            print(f"Analysis failed for animal: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_zt0(self, animal_df: pd.DataFrame) -> int:
        """
        Find ZT0 (lights on) minute from light cycle data.

        Args:
            animal_df: DataFrame for single animal

        Returns:
            Minute of day when ZT0 occurs (lights turn on)
        """
        first_day = animal_df.head(self.MINUTES_PER_DAY)
        light_cycle = first_day['light.cycle'].values

        # Look for Dark->Light transition (ZT0)
        for i in range(1, len(light_cycle)):
            if light_cycle[i - 1] == 'Dark' and light_cycle[i] == 'Light':
                return i

        # If starts with Light, ZT0 is at minute 0
        if light_cycle[0] == 'Light':
            return 0

        # Look for Light->Dark transition (that's ZT12, so ZT0 is 12 hours before)
        for i in range(1, len(light_cycle)):
            if light_cycle[i - 1] == 'Light' and light_cycle[i] == 'Dark':
                return (i - 720) % self.MINUTES_PER_DAY

        # Default to 0 if no transition found
        return 0

    def _analyze_metric(self, animal_df: pd.DataFrame, col_name: str,
                        metric_name: str, n_days: int, zt0_minute: int) -> Optional[Dict[str, Any]]:
        """
        Analyze a single behavioral metric using vectorized operations.

        Args:
            animal_df: DataFrame for single animal
            col_name: Column name for the metric
            metric_name: Display name for the metric
            n_days: Number of days to analyze
            zt0_minute: ZT0 minute offset

        Returns:
            Dictionary with metric analysis results
        """
        # Get metric values - use numpy for speed
        values = pd.to_numeric(animal_df[col_name], errors='coerce').values

        # Skip if all NaN
        if np.all(np.isnan(values)):
            return None

        # Convert respiration rate from breaths/min to Hz
        if 'Respiration Rate' in metric_name:
            values = values / 60.0

        # Reshape into days x minutes matrix
        data_matrix = values[:n_days * self.MINUTES_PER_DAY].reshape(n_days, self.MINUTES_PER_DAY)

        # Compute CTA using vectorized function
        cta_stats = vectorized_cta(data_matrix)
        cta = cta_stats['cta']
        cta_std = cta_stats['cta_std']
        cta_sem = cta_stats['cta_sem']

        # Convert to ZT-aligned using vectorized operations
        cta_zt = vectorized_align_to_zt(cta, zt0_minute)
        cta_std_zt = vectorized_align_to_zt(cta_std, zt0_minute)
        cta_sem_zt = vectorized_align_to_zt(cta_sem, zt0_minute)

        # Get light cycle info (ZT-aligned)
        light_cycle = animal_df['light.cycle'].values[:self.MINUTES_PER_DAY]
        light_cycle_zt = vectorized_align_to_zt(light_cycle, zt0_minute)

        # Compute dark/light means using vectorized function
        dl_means = compute_dark_light_means(cta_zt, light_cycle_zt)

        # Per-day data for stacked plots - batch align
        daily_data = [data_matrix[day_idx, :] for day_idx in range(n_days)]
        daily_data_zt = batch_align_to_zt(daily_data, zt0_minute)

        return {
            'cta': cta_zt,
            'cta_std': cta_std_zt,
            'cta_sem': cta_sem_zt,
            'daily_data': daily_data_zt,
            'dark_mean': dl_means['dark_mean'],
            'light_mean': dl_means['light_mean'],
            'overall_mean': np.nanmean(cta_zt),
            'overall_std': np.nanstd(cta_zt),
        }

    def _align_to_zt(self, data: np.ndarray, zt0_minute: int) -> np.ndarray:
        """
        Align data array to ZT time.

        Args:
            data: Array of values (1440 elements for one day)
            zt0_minute: Minute when ZT0 occurs

        Returns:
            Array aligned so index 0 = ZT0
        """
        if zt0_minute == 0:
            return data.copy()

        return np.roll(data, -zt0_minute)

    def _identify_companions(self, results: Dict[str, Dict], df: pd.DataFrame):
        """
        Identify companion animals (same cage) for each animal.

        Args:
            results: Analysis results dictionary
            df: Original full DataFrame
        """
        # Check if cage ID column exists - cage.name is most common
        cage_col = None
        for col in ['cage.name', 'cage.id', 'cage', 'Cage']:
            if col in df.columns:
                cage_col = col
                break

        if cage_col is None:
            # No cage info available
            return

        # Group animals by cage
        cage_groups = {}
        for animal_id in results.keys():
            animal_data = df[df['animal.id'] == animal_id]
            if len(animal_data) > 0 and cage_col in animal_data.columns:
                cage_id = animal_data[cage_col].iloc[0]
                if pd.notna(cage_id):
                    if cage_id not in cage_groups:
                        cage_groups[cage_id] = []
                    cage_groups[cage_id].append(animal_id)

        # Assign companions
        for cage_id, animals in cage_groups.items():
            for animal_id in animals:
                companions = [a for a in animals if a != animal_id]
                if companions and animal_id in results:
                    results[animal_id]['metadata']['companion'] = companions[0] if len(companions) == 1 else companions
                    results[animal_id]['metadata']['cage_id'] = cage_id
