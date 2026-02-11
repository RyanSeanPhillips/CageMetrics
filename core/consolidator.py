"""
Consolidator module for combining multiple animal NPZ datasets.

Loads NPZ files exported by DataExporter and consolidates them into:
- Consolidated NPZ file with combined CTAs, grand means, and full metadata
- Excel workbook with summary statistics by group (genotype, sex, etc.)
- Combined CTA data for population-level analysis

Consolidated NPZ files use naming convention:
    consolidated_<filters>_<n>animals_<date>.npz
    e.g., consolidated_WT_housed-with-DS_Male_12animals_20250115.npz
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime


def generate_consolidated_filename(filter_criteria=None, n_animals: int = 0) -> str:
    """
    Generate a descriptive filename for consolidated data based on filter criteria.

    Args:
        filter_criteria: FilterCriteria object or dict with filter settings
        n_animals: Number of animals in the consolidated dataset

    Returns:
        Suggested filename (without extension)
        Format: CageMetrics_Consolidated_<filters>_<n>animals_<date>
    """
    parts = ["CageMetrics_Consolidated"]

    if filter_criteria:
        # Handle FilterCriteria object or dict
        if hasattr(filter_criteria, 'filters'):
            filters = filter_criteria.filters
        elif isinstance(filter_criteria, dict):
            filters = filter_criteria
        else:
            filters = {}

        # Add genotype info
        genotypes = filters.get('genotype', set())
        if genotypes and len(genotypes) < 3:
            parts.append('_'.join(sorted(genotypes)))

        # Add cagemate genotype info
        cagemate_genos = filters.get('cagemate_genotype', set())
        if cagemate_genos and len(cagemate_genos) < 3:
            cagemate_str = '_'.join(sorted(cagemate_genos))
            parts.append(f"housed-with-{cagemate_str}")

        # Add sex info
        sexes = filters.get('sex', set())
        if sexes and len(sexes) == 1:
            parts.append(list(sexes)[0])

        # Add treatment info
        treatments = filters.get('treatment', set())
        if treatments and len(treatments) < 3:
            parts.append('_'.join(sorted(treatments)))

    # Add animal count
    if n_animals > 0:
        parts.append(f"{n_animals}animals")

    # Add date
    parts.append(datetime.now().strftime("%Y%m%d"))

    # Join with underscores, clean up
    filename = '_'.join(parts)
    # Remove any double underscores
    while '__' in filename:
        filename = filename.replace('__', '_')

    return filename


class Consolidator:
    """Consolidate multiple animal NPZ datasets."""

    MINUTES_PER_DAY = 1440

    def __init__(self):
        pass

    def consolidate(self, npz_paths: List[str], output_path: str,
                    filter_criteria=None, save_npz: bool = True,
                    save_pdf: bool = True, layout_style: str = "classic",
                    smoothing_window: int = 15) -> Dict[str, Any]:
        """
        Consolidate multiple NPZ files into NPZ, Excel, and PDF formats.

        Args:
            npz_paths: List of paths to NPZ data files
            output_path: Base path for output files (extension determines format)
            filter_criteria: Optional FilterCriteria used for selection
            save_npz: Also save consolidated NPZ file (default: True)
            save_pdf: Also save PDF figures (default: True)
            layout_style: Figure layout style ("classic" or "matrix")

        Returns:
            Dictionary with consolidation results including output paths
        """
        # Load all NPZ files
        animals = []
        for path in npz_paths:
            animal_data = self._load_npz(path)
            if animal_data:
                animal_data['source_file'] = path
                animals.append(animal_data)

        if not animals:
            raise ValueError("No valid NPZ files found")

        # Get all unique metrics
        all_metrics = set()
        for animal in animals:
            all_metrics.update(animal.get('metric_names', []))
        all_metrics = sorted(all_metrics)

        output_base = Path(output_path)
        results = {
            'n_animals': len(animals),
            'n_metrics': len(all_metrics),
            'output_paths': []
        }

        # Determine output paths based on provided extension
        stem = output_base.stem
        parent = output_base.parent
        npz_path = parent / f"{stem}.npz"
        excel_path = parent / f"{stem}.xlsx"
        pdf_path = parent / f"{stem}.pdf"

        # Write consolidated Excel
        with pd.ExcelWriter(str(excel_path), engine='openpyxl') as writer:
            # Tab 1: Animal Summary
            self._write_summary_tab(writer, animals)

            # Tab 2: Light/Dark Means (all animals, all metrics)
            self._write_light_dark_tab(writer, animals, all_metrics)

            # Tab 3: Group Summary (mean ± SEM by genotype for Prism)
            self._write_group_summary_tab(writer, animals, all_metrics)

            # Tab 4: Sleep Statistics (if sleep data available)
            self._write_sleep_stats_tab(writer, animals)

            # Tab 5: Sleep Histogram Bins (if sleep data available)
            self._write_histogram_tab(writer, animals)

            # Tabs 6+: Per-metric CTA data
            for metric_name in all_metrics:
                self._write_metric_cta_tab(writer, animals, metric_name)

            # Tabs: Per-metric Daily Data (long format with light/dark/age)
            for metric_name in all_metrics:
                self._write_daily_data_tab(writer, animals, metric_name)

            # Tabs: Per-metric Age-pivoted data (rows=age, cols=animals, for Prism)
            has_age = any(a.get('metadata', {}).get('age_days_at_start') is not None for a in animals)
            if has_age:
                for metric_name in all_metrics:
                    self._write_age_pivot_tab(writer, animals, metric_name)

        results['output_paths'].append(str(excel_path))
        results['excel_path'] = str(excel_path)

        # Write consolidated NPZ
        if save_npz:
            self._write_consolidated_npz(npz_path, animals, all_metrics, filter_criteria)
            results['output_paths'].append(str(npz_path))
            results['npz_path'] = str(npz_path)

        # Write consolidated PDF figures
        if save_pdf:
            filter_desc = ""
            if filter_criteria:
                if hasattr(filter_criteria, 'to_description'):
                    filter_desc = filter_criteria.to_description()
            self._write_consolidated_pdf(pdf_path, animals, filter_desc, layout_style, smoothing_window)
            results['output_paths'].append(str(pdf_path))
            results['pdf_path'] = str(pdf_path)

        return results

    def _write_consolidated_pdf(self, pdf_path: Path, animals: List[Dict],
                                 filter_description: str, layout_style: str = "classic",
                                 smoothing_window: int = 15):
        """
        Write consolidated figures to PDF.

        Args:
            pdf_path: Output PDF path
            animals: List of animal data dicts
            filter_description: Human-readable filter description
            layout_style: Figure layout style ("classic" or "matrix")
            smoothing_window: Rolling average window size in minutes
        """
        from matplotlib.backends.backend_pdf import PdfPages
        from core.consolidation_figure_generator import ConsolidationFigureGenerator

        generator = ConsolidationFigureGenerator(smoothing_window=smoothing_window)
        pages = generator.generate_all_pages(animals, filter_description, layout_style=layout_style)

        with PdfPages(str(pdf_path)) as pdf:
            for title, fig in pages:
                pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
                # Close figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)

    def _write_consolidated_npz(self, npz_path: Path, animals: List[Dict],
                                 metrics: List[str], filter_criteria=None):
        """
        Write consolidated data to NPZ format for use in comparison tab.

        The NPZ file contains:
        - consolidation_metadata: JSON with filter criteria, source files, date
        - animal_metadata: JSON array of per-animal metadata
        - metric_names: List of metric names
        - For each metric:
          - {metric}_grand_cta: Mean CTA across all animals (1440,)
          - {metric}_grand_sem: SEM across animals (1440,)
          - {metric}_all_ctas: Matrix of all animal CTAs (n_animals, 1440)
          - {metric}_dark_means: Array of dark means per animal
          - {metric}_light_means: Array of light means per animal
        """
        data = {}

        # Consolidation metadata
        consolidation_meta = {
            'consolidation_date': datetime.now().isoformat(),
            'n_animals': len(animals),
            'n_metrics': len(metrics),
            'source_files': [a.get('source_file', '') for a in animals],
            'filter_description': '',
            'filters': {}
        }

        if filter_criteria:
            if hasattr(filter_criteria, 'to_description'):
                consolidation_meta['filter_description'] = filter_criteria.to_description()
            if hasattr(filter_criteria, 'filters'):
                # Convert sets to lists for JSON serialization
                consolidation_meta['filters'] = {
                    k: list(v) for k, v in filter_criteria.filters.items()
                }

        data['consolidation_metadata'] = json.dumps(consolidation_meta)

        # Per-animal metadata
        animal_metadata = [a.get('metadata', {}) for a in animals]
        data['animal_metadata'] = json.dumps(animal_metadata)

        # Metric names
        data['metric_names'] = np.array(metrics, dtype=object)

        # Per-metric data
        for metric_name in metrics:
            key_base = self._clean_metric_name(metric_name)

            # Collect all CTAs for this metric
            ctas = []
            dark_means = []
            light_means = []

            for animal in animals:
                m_data = animal.get('metrics', {}).get(metric_name, {})
                cta = m_data.get('cta', np.array([]))

                if len(cta) == self.MINUTES_PER_DAY:
                    ctas.append(cta)
                else:
                    ctas.append(np.full(self.MINUTES_PER_DAY, np.nan))

                dark_means.append(m_data.get('dark_mean', np.nan))
                light_means.append(m_data.get('light_mean', np.nan))

            cta_matrix = np.array(ctas)  # (n_animals, 1440)

            # Compute grand mean and SEM
            grand_cta = np.nanmean(cta_matrix, axis=0)
            n_valid = np.sum(~np.isnan(cta_matrix), axis=0)
            grand_sem = np.nanstd(cta_matrix, axis=0) / np.sqrt(np.maximum(n_valid, 1))

            # Store arrays
            data[f'{key_base}_grand_cta'] = grand_cta
            data[f'{key_base}_grand_sem'] = grand_sem
            data[f'{key_base}_all_ctas'] = cta_matrix
            data[f'{key_base}_dark_means'] = np.array(dark_means)
            data[f'{key_base}_light_means'] = np.array(light_means)

            # Store per-animal daily data for actograms and age trending
            # Each animal may have different number of days, so store separately
            n_days_list = []
            for animal_idx, animal in enumerate(animals):
                m_data = animal.get('metrics', {}).get(metric_name, {})
                daily_data = m_data.get('daily_data', None)
                if daily_data is not None and len(daily_data) > 0:
                    daily_arr = np.array(daily_data)
                    data[f'{key_base}_daily_{animal_idx}'] = daily_arr
                    n_days_list.append(len(daily_data))
                else:
                    n_days_list.append(0)
            data[f'{key_base}_n_days_per_animal'] = np.array(n_days_list)

        # Aggregate and save sleep data if available
        self._write_sleep_data_to_npz(data, animals)

        # Save NPZ
        np.savez_compressed(str(npz_path), **data)

    def _write_sleep_data_to_npz(self, data: Dict, animals: List[Dict]):
        """
        Aggregate sleep data from all animals and add to NPZ data dict.

        Saves:
        - Per-animal arrays for bar charts with statistics
        - Pooled bout durations for histograms
        - Sleep parameters metadata
        """
        # Collect animals with sleep data
        animals_with_sleep = [a for a in animals if a.get('sleep_analysis')]

        if not animals_with_sleep:
            return

        n_animals = len(animals_with_sleep)

        # Per-animal metric arrays (for bar charts and statistics)
        total_minutes_light = []
        total_minutes_dark = []
        bout_count_light = []
        bout_count_dark = []
        mean_duration_light = []
        mean_duration_dark = []
        frag_index_light = []
        frag_index_dark = []
        percent_time_light = []
        percent_time_dark = []

        # Quality metrics (per animal)
        long_bout_pct_light = []
        long_bout_pct_dark = []
        light_dark_ratio = []
        transition_rate = []

        # Pooled bout durations (for histograms)
        all_bout_durations_light = []
        all_bout_durations_dark = []

        # Collect sleep parameters (use first animal's settings)
        first_sleep = animals_with_sleep[0].get('sleep_analysis', {})
        threshold = first_sleep.get('threshold', 0.5)
        bin_width = first_sleep.get('bin_width', 5.0)

        for animal in animals_with_sleep:
            sleep = animal.get('sleep_analysis', {})
            light_stats = sleep.get('light_stats', {})
            dark_stats = sleep.get('dark_stats', {})

            # Total sleep minutes
            total_minutes_light.append(light_stats.get('total_minutes', 0))
            total_minutes_dark.append(dark_stats.get('total_minutes', 0))

            # Bout counts
            l_bouts = light_stats.get('bout_count', 0)
            d_bouts = dark_stats.get('bout_count', 0)
            bout_count_light.append(l_bouts)
            bout_count_dark.append(d_bouts)

            # Mean bout duration
            mean_duration_light.append(light_stats.get('mean_duration', 0))
            mean_duration_dark.append(dark_stats.get('mean_duration', 0))

            # Fragmentation index (bouts per hour of sleep)
            l_sleep_hrs = light_stats.get('total_minutes', 0) / 60
            d_sleep_hrs = dark_stats.get('total_minutes', 0) / 60
            frag_index_light.append(l_bouts / l_sleep_hrs if l_sleep_hrs > 0 else 0)
            frag_index_dark.append(d_bouts / d_sleep_hrs if d_sleep_hrs > 0 else 0)

            # Percent time asleep
            percent_time_light.append(light_stats.get('percent_time', 0))
            percent_time_dark.append(dark_stats.get('percent_time', 0))

            # Quality metrics
            qm = sleep.get('quality_metrics', {})
            long_bout_pct_light.append(qm.get('long_bout_pct_light', 0))
            long_bout_pct_dark.append(qm.get('long_bout_pct_dark', 0))
            light_dark_ratio.append(qm.get('light_dark_ratio', 0))
            transition_rate.append(qm.get('transition_rate', 0))

            # Pool bout durations for histograms
            bout_dur_light = sleep.get('bout_durations_light', np.array([]))
            bout_dur_dark = sleep.get('bout_durations_dark', np.array([]))
            if len(bout_dur_light) > 0:
                all_bout_durations_light.extend(bout_dur_light.tolist())
            if len(bout_dur_dark) > 0:
                all_bout_durations_dark.extend(bout_dur_dark.tolist())

        # Store per-animal arrays
        data['sleep_total_minutes_light'] = np.array(total_minutes_light)
        data['sleep_total_minutes_dark'] = np.array(total_minutes_dark)
        data['sleep_bout_count_light'] = np.array(bout_count_light)
        data['sleep_bout_count_dark'] = np.array(bout_count_dark)
        data['sleep_mean_duration_light'] = np.array(mean_duration_light)
        data['sleep_mean_duration_dark'] = np.array(mean_duration_dark)
        data['sleep_frag_index_light'] = np.array(frag_index_light)
        data['sleep_frag_index_dark'] = np.array(frag_index_dark)
        data['sleep_percent_time_light'] = np.array(percent_time_light)
        data['sleep_percent_time_dark'] = np.array(percent_time_dark)

        # Store pooled bout durations
        data['sleep_bout_durations_light'] = np.array(all_bout_durations_light)
        data['sleep_bout_durations_dark'] = np.array(all_bout_durations_dark)

        # Store quality metrics (per-animal arrays)
        data['sleep_long_bout_pct_light'] = np.array(long_bout_pct_light)
        data['sleep_long_bout_pct_dark'] = np.array(long_bout_pct_dark)
        data['sleep_light_dark_ratio'] = np.array(light_dark_ratio)
        data['sleep_transition_rate'] = np.array(transition_rate)

        # Store metadata
        data['sleep_n_animals'] = n_animals
        data['sleep_parameters_json'] = json.dumps({
            'threshold': threshold,
            'bin_width': bin_width
        })

    def _clean_metric_name(self, metric_name: str) -> str:
        """Clean metric name for use as NPZ key."""
        return metric_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'pct')

    def _load_npz(self, path: str) -> Optional[Dict[str, Any]]:
        """Load an NPZ file and extract its data."""
        try:
            data = np.load(path, allow_pickle=True)

            # Parse metadata
            metadata = json.loads(str(data['metadata_json']))
            quality = json.loads(str(data['quality_json']))

            # Get metric names
            metric_names = list(data['metric_names'])

            # Extract metric data
            metrics = {}
            for metric_name in metric_names:
                key_base = metric_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'pct')

                metrics[metric_name] = {
                    'cta': data.get(f'{key_base}_cta', np.array([])),
                    'cta_sem': data.get(f'{key_base}_cta_sem', np.array([])),
                    'dark_mean': float(data.get(f'{key_base}_dark_mean', np.nan)),
                    'light_mean': float(data.get(f'{key_base}_light_mean', np.nan)),
                    'overall_mean': float(data.get(f'{key_base}_overall_mean', np.nan)),
                    'overall_std': float(data.get(f'{key_base}_overall_std', np.nan)),
                }

                # Try to load daily data if present
                daily_key = f'{key_base}_daily'
                if daily_key in data:
                    metrics[metric_name]['daily_data'] = data[daily_key]

            result = {
                'metadata': metadata,
                'quality': quality,
                'metric_names': metric_names,
                'metrics': metrics
            }

            # Extract sleep analysis data if present
            sleep_data = self._extract_sleep_data(data)
            if sleep_data:
                result['sleep_analysis'] = sleep_data

            return result

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _extract_sleep_data(self, data) -> Optional[Dict[str, Any]]:
        """
        Extract sleep analysis data from an individual NPZ file.

        Args:
            data: Loaded NPZ file data

        Returns:
            Dictionary with sleep stats or None if no sleep data
        """
        # Check if sleep data exists
        if 'sleep_stats_json' not in data:
            return None

        try:
            stats = json.loads(str(data['sleep_stats_json']))

            sleep_data = {
                'threshold': float(data.get('sleep_threshold', 0.5)),
                'bin_width': float(data.get('sleep_bin_width', 5.0)),
                'n_days': int(data.get('sleep_n_days', 0)),
                'light_stats': stats.get('light', {}),
                'dark_stats': stats.get('dark', {}),
                'total_stats': stats.get('total', {}),
                'quality_metrics': stats.get('quality_metrics', {}),
            }

            # Extract bout data for histogram aggregation
            if 'sleep_bouts' in data:
                from core.sleep_analysis import SleepBout
                bout_array = data['sleep_bouts']
                # bouts is structured array: (day, bout_num, start_minute, end_minute, duration, phase)
                # phase: 0 = light, 1 = dark
                light_durations = []
                dark_durations = []
                bouts_list = []
                for bout in bout_array:
                    duration = float(bout['duration'])
                    phase_str = 'light' if bout['phase'] == 0 else 'dark'
                    if bout['phase'] == 0:
                        light_durations.append(duration)
                    else:
                        dark_durations.append(duration)
                    # Reconstruct SleepBout object for histogram methods
                    bouts_list.append(SleepBout(
                        day=int(bout['day']),
                        bout_num=int(bout['bout_num']),
                        start_minute=int(bout['start_minute']),
                        end_minute=int(bout['end_minute']),
                        duration=duration,
                        phase=phase_str
                    ))
                sleep_data['bout_durations_light'] = np.array(light_durations)
                sleep_data['bout_durations_dark'] = np.array(dark_durations)
                sleep_data['bouts'] = bouts_list
            else:
                sleep_data['bout_durations_light'] = np.array([])
                sleep_data['bout_durations_dark'] = np.array([])
                sleep_data['bouts'] = []

            return sleep_data

        except Exception as e:
            print(f"Error extracting sleep data: {e}")
            return None

    def _write_summary_tab(self, writer: pd.ExcelWriter, animals: List[Dict]):
        """Write summary of all animals."""
        rows = []
        for animal in animals:
            meta = animal.get('metadata', {})
            row = {
                'Animal ID': meta.get('animal_id', ''),
                'Genotype': meta.get('genotype', ''),
                'Genotype (Raw)': meta.get('genotype_raw', ''),
                'Sex': meta.get('sex', ''),
                'Cohort': meta.get('cohort', ''),
                'Cage ID': meta.get('cage_id', ''),
                'Cagemate': meta.get('companion', ''),
                'DOB': meta.get('dob', ''),
                'Age at Start (days)': meta.get('age_days_at_start', ''),
                'Days Analyzed': meta.get('n_days_analyzed', 0),
                'Total Minutes': meta.get('total_minutes', 0),
                'Source File': animal.get('source_file', '')
            }
            # Compute age at end if age at start is available
            age_start = meta.get('age_days_at_start', None)
            n_days = meta.get('n_days_analyzed', 0)
            if age_start is not None and n_days:
                row['Age at End (days)'] = age_start + n_days - 1
            else:
                row['Age at End (days)'] = ''
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='Animal Summary', index=False)

    def _write_light_dark_tab(self, writer: pd.ExcelWriter, animals: List[Dict], metrics: List[str]):
        """Write light/dark means for all animals and metrics."""
        rows = []

        for animal in animals:
            meta = animal.get('metadata', {})
            animal_metrics = animal.get('metrics', {})

            row = {
                'Animal ID': meta.get('animal_id', ''),
                'Genotype': meta.get('genotype', ''),
                'Sex': meta.get('sex', ''),
                'Cohort': meta.get('cohort', ''),
                'Cage ID': meta.get('cage_id', ''),
            }

            # Add light/dark means for each metric
            for metric_name in metrics:
                m_data = animal_metrics.get(metric_name, {})
                row[f'{metric_name} (Dark)'] = m_data.get('dark_mean', np.nan)
                row[f'{metric_name} (Light)'] = m_data.get('light_mean', np.nan)
                row[f'{metric_name} (Overall)'] = m_data.get('overall_mean', np.nan)

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='Light-Dark Means', index=False)

    def _write_metric_cta_tab(self, writer: pd.ExcelWriter, animals: List[Dict], metric_name: str):
        """Write CTA data for a single metric across all animals."""
        # Clean sheet name
        sheet_name = metric_name[:28].replace('/', '_').replace('\\', '_').replace('*', '').replace('?', '').replace('[', '').replace(']', '')
        sheet_name = f"CTA_{sheet_name}"[:31]

        # Build data frame with ZT time and each animal's CTA
        data = {
            'ZT_Minute': list(range(self.MINUTES_PER_DAY)),
            'ZT_Hour': [m / 60 for m in range(self.MINUTES_PER_DAY)],
        }

        # Add each animal's CTA
        for animal in animals:
            meta = animal.get('metadata', {})
            animal_id = meta.get('animal_id', 'Unknown')
            genotype = meta.get('genotype', '')
            sex = meta.get('sex', '')

            col_name = f"{animal_id}_{genotype}_{sex}"

            m_data = animal.get('metrics', {}).get(metric_name, {})
            cta = m_data.get('cta', np.array([]))

            if len(cta) == self.MINUTES_PER_DAY:
                data[col_name] = cta
            else:
                data[col_name] = [np.nan] * self.MINUTES_PER_DAY

        df = pd.DataFrame(data)

        # Calculate group means by genotype
        # First, identify unique genotypes
        genotypes = set()
        for animal in animals:
            genotypes.add(animal.get('metadata', {}).get('genotype', 'Unknown'))

        for genotype in sorted(genotypes):
            # Get columns for this genotype
            geno_cols = [col for col in df.columns
                        if col not in ['ZT_Minute', 'ZT_Hour']
                        and f'_{genotype}_' in col]

            if geno_cols:
                df[f'Mean_{genotype}'] = df[geno_cols].mean(axis=1)
                df[f'SEM_{genotype}'] = df[geno_cols].sem(axis=1)

        df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _write_sleep_stats_tab(self, writer: pd.ExcelWriter, animals: List[Dict]):
        """Write sleep statistics for all animals to Excel tab."""
        # Check if any animals have sleep data
        animals_with_sleep = [a for a in animals if a.get('sleep_analysis')]
        if not animals_with_sleep:
            return

        rows = []
        for animal in animals_with_sleep:
            meta = animal.get('metadata', {})
            sleep = animal.get('sleep_analysis', {})
            light_stats = sleep.get('light_stats', {})
            dark_stats = sleep.get('dark_stats', {})
            total_stats = sleep.get('total_stats', {})
            quality = sleep.get('quality_metrics', {})

            row = {
                'Animal ID': meta.get('animal_id', ''),
                'Genotype': meta.get('genotype', ''),
                'Sex': meta.get('sex', ''),
                'Cohort': meta.get('cohort', ''),
                'N Days': sleep.get('n_days', 0),
                # Light phase
                'Light Total Sleep (min)': light_stats.get('total_minutes', np.nan),
                'Light Sleep %': light_stats.get('percent_time', np.nan),
                'Light Bout Count': light_stats.get('bout_count', np.nan),
                'Light Mean Bout (min)': light_stats.get('mean_duration', np.nan),
                'Light Median Bout (min)': light_stats.get('median_duration', np.nan),
                'Light Max Bout (min)': light_stats.get('max_duration', np.nan),
                # Dark phase
                'Dark Total Sleep (min)': dark_stats.get('total_minutes', np.nan),
                'Dark Sleep %': dark_stats.get('percent_time', np.nan),
                'Dark Bout Count': dark_stats.get('bout_count', np.nan),
                'Dark Mean Bout (min)': dark_stats.get('mean_duration', np.nan),
                'Dark Median Bout (min)': dark_stats.get('median_duration', np.nan),
                'Dark Max Bout (min)': dark_stats.get('max_duration', np.nan),
                # Total
                'Total Sleep (min)': total_stats.get('total_minutes', np.nan),
                'Total Bout Count': total_stats.get('bout_count', np.nan),
                'Total Mean Bout (min)': total_stats.get('mean_duration', np.nan),
                # Quality metrics
                '% Long Bouts (Light)': quality.get('long_bout_pct_light', np.nan),
                '% Long Bouts (Dark)': quality.get('long_bout_pct_dark', np.nan),
                'L/D Ratio': quality.get('light_dark_ratio', np.nan),
                'Transition Rate': quality.get('transition_rate', np.nan),
            }
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name='Sleep Statistics', index=False)

    def _write_group_summary_tab(self, writer: pd.ExcelWriter, animals: List[Dict], metrics: List[str]):
        """Write group summary with mean ± SEM by genotype (Prism-ready format)."""
        # Group animals by genotype
        by_genotype = {}
        for animal in animals:
            geno = animal.get('metadata', {}).get('genotype', 'Unknown')
            if geno not in by_genotype:
                by_genotype[geno] = []
            by_genotype[geno].append(animal)

        rows = []
        for genotype in sorted(by_genotype.keys()):
            geno_animals = by_genotype[genotype]
            n = len(geno_animals)

            for metric_name in metrics:
                # Collect light/dark/overall means for this genotype
                light_vals = []
                dark_vals = []
                overall_vals = []

                for animal in geno_animals:
                    m_data = animal.get('metrics', {}).get(metric_name, {})
                    if not np.isnan(m_data.get('light_mean', np.nan)):
                        light_vals.append(m_data.get('light_mean'))
                    if not np.isnan(m_data.get('dark_mean', np.nan)):
                        dark_vals.append(m_data.get('dark_mean'))
                    if not np.isnan(m_data.get('overall_mean', np.nan)):
                        overall_vals.append(m_data.get('overall_mean'))

                # Calculate mean ± SEM
                def calc_stats(vals):
                    if not vals:
                        return np.nan, np.nan
                    mean = np.mean(vals)
                    sem = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                    return mean, sem

                l_mean, l_sem = calc_stats(light_vals)
                d_mean, d_sem = calc_stats(dark_vals)
                o_mean, o_sem = calc_stats(overall_vals)

                rows.append({
                    'Genotype': genotype,
                    'N': n,
                    'Metric': metric_name,
                    'Light Mean': l_mean,
                    'Light SEM': l_sem,
                    'Dark Mean': d_mean,
                    'Dark SEM': d_sem,
                    'Overall Mean': o_mean,
                    'Overall SEM': o_sem,
                })

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name='Group Summary', index=False)

    def _write_histogram_tab(self, writer: pd.ExcelWriter, animals: List[Dict]):
        """Write sleep bout histogram bins (pre-binned for Prism)."""
        # Collect all bout durations
        light_durations = []
        dark_durations = []

        for animal in animals:
            sleep = animal.get('sleep_analysis', {})
            bouts = sleep.get('bouts', [])
            for bout in bouts:
                if hasattr(bout, 'duration') and hasattr(bout, 'phase'):
                    if bout.phase == 'light':
                        light_durations.append(bout.duration)
                    else:
                        dark_durations.append(bout.duration)

        if not light_durations and not dark_durations:
            return

        # Create histogram bins (5-minute bins, 0-120 minutes)
        bin_width = 5.0
        max_dur = 120.0
        bins = np.arange(0, max_dur + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute histograms
        light_counts, _ = np.histogram(light_durations, bins=bins) if light_durations else (np.zeros(len(bins)-1), bins)
        dark_counts, _ = np.histogram(dark_durations, bins=bins) if dark_durations else (np.zeros(len(bins)-1), bins)

        # Time-weighted histograms (sum of durations in each bin)
        light_time = np.zeros(len(bins) - 1)
        dark_time = np.zeros(len(bins) - 1)
        for dur in light_durations:
            idx = min(int(dur // bin_width), len(light_time) - 1)
            light_time[idx] += dur
        for dur in dark_durations:
            idx = min(int(dur // bin_width), len(dark_time) - 1)
            dark_time[idx] += dur

        # Get number of animals for normalization
        n_animals = len([a for a in animals if a.get('sleep_analysis')])

        df = pd.DataFrame({
            'Bin Start (min)': bins[:-1],
            'Bin End (min)': bins[1:],
            'Bin Center (min)': bin_centers,
            'Light Bout Count': light_counts,
            'Dark Bout Count': dark_counts,
            'Light Time in Bouts (min)': light_time,
            'Dark Time in Bouts (min)': dark_time,
            'Light Count per Animal': light_counts / n_animals if n_animals > 0 else light_counts,
            'Dark Count per Animal': dark_counts / n_animals if n_animals > 0 else dark_counts,
            'Light Time per Animal': light_time / n_animals if n_animals > 0 else light_time,
            'Dark Time per Animal': dark_time / n_animals if n_animals > 0 else dark_time,
        })
        df.to_excel(writer, sheet_name='Histogram Bins', index=False)

    def _write_daily_data_tab(self, writer: pd.ExcelWriter, animals: List[Dict], metric_name: str):
        """Write per-day data for a metric with overall, light, and dark means plus age."""
        # Clean sheet name
        sheet_name = metric_name[:20].replace('/', '_').replace('\\', '_').replace('*', '').replace('?', '').replace('[', '').replace(']', '')
        sheet_name = f"Daily_{sheet_name}"[:31]

        # Collect daily data from all animals
        all_daily_data = []
        max_days = 0

        for animal in animals:
            meta = animal.get('metadata', {})
            animal_id = meta.get('animal_id', 'Unknown')
            genotype = meta.get('genotype', '')
            age_at_start = meta.get('age_days_at_start', None)

            m_data = animal.get('metrics', {}).get(metric_name, {})
            daily_data = m_data.get('daily_data', None)

            if daily_data is not None and len(daily_data) > 0:
                n_days = len(daily_data)
                max_days = max(max_days, n_days)
                all_daily_data.append({
                    'animal_id': animal_id,
                    'genotype': genotype,
                    'daily_data': daily_data,
                    'n_days': n_days,
                    'age_at_start': age_at_start
                })

        if not all_daily_data:
            return

        # Build long-format data frame: one row per animal per day
        # Columns: Animal ID, Genotype, Day, Age, Overall Mean, Light Mean, Dark Mean
        rows = []
        for animal_data in all_daily_data:
            for day_idx in range(animal_data['n_days']):
                day_trace = animal_data['daily_data'][day_idx]
                if len(day_trace) == 0:
                    continue

                day_arr = np.array(day_trace, dtype=float)

                # Light phase = first 720 min (ZT0-ZT12), Dark phase = last 720 min (ZT12-ZT24)
                light_mean = np.nanmean(day_arr[:720]) if len(day_arr) >= 720 else np.nan
                dark_mean = np.nanmean(day_arr[720:]) if len(day_arr) > 720 else np.nan
                overall_mean = np.nanmean(day_arr)

                row = {
                    'Animal ID': animal_data['animal_id'],
                    'Genotype': animal_data['genotype'],
                    'Day': day_idx + 1,
                    'Overall Mean': overall_mean,
                    'Light Mean': light_mean,
                    'Dark Mean': dark_mean,
                }

                # Add age column if available
                if animal_data['age_at_start'] is not None:
                    row['Age (postnatal day)'] = animal_data['age_at_start'] + day_idx
                else:
                    row['Age (postnatal day)'] = ''

                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    def _write_age_pivot_tab(self, writer: pd.ExcelWriter, animals: List[Dict], metric_name: str):
        """Write age-pivoted data: rows=postnatal age, cols=animals (light/dark/overall).

        This format is ideal for copy-paste into Prism for age-based trending plots.
        """
        sheet_base = metric_name[:16].replace('/', '_').replace('\\', '_').replace('*', '').replace('?', '').replace('[', '').replace(']', '')
        sheet_name = f"Age_{sheet_base}"[:31]

        # Collect per-animal data keyed by age
        animal_age_data = []  # list of {animal_id, genotype, age_map: {age: (light, dark, overall)}}

        for animal in animals:
            meta = animal.get('metadata', {})
            age_at_start = meta.get('age_days_at_start', None)
            if age_at_start is None:
                continue

            animal_id = meta.get('animal_id', 'Unknown')
            genotype = meta.get('genotype', '')

            m_data = animal.get('metrics', {}).get(metric_name, {})
            daily_data = m_data.get('daily_data', None)

            if daily_data is None or len(daily_data) == 0:
                continue

            age_map = {}
            for day_idx, day_trace in enumerate(daily_data):
                if len(day_trace) == 0:
                    continue
                day_arr = np.array(day_trace, dtype=float)
                age = age_at_start + day_idx
                light_mean = np.nanmean(day_arr[:720]) if len(day_arr) >= 720 else np.nan
                dark_mean = np.nanmean(day_arr[720:]) if len(day_arr) > 720 else np.nan
                overall_mean = np.nanmean(day_arr)
                age_map[age] = (light_mean, dark_mean, overall_mean)

            if age_map:
                animal_age_data.append({
                    'animal_id': animal_id,
                    'genotype': genotype,
                    'age_map': age_map
                })

        if not animal_age_data:
            return

        # Determine full age range
        all_ages = set()
        for ad in animal_age_data:
            all_ages.update(ad['age_map'].keys())
        min_age = min(all_ages)
        max_age = max(all_ages)
        age_range = list(range(min_age, max_age + 1))

        # Build three sub-tables: Overall, Light, Dark
        # Each has rows=ages, cols=animals
        for phase, phase_idx in [('Overall', 2), ('Light', 0), ('Dark', 1)]:
            rows = []
            for age in age_range:
                row = {'Age (P)': f'P{age}'}
                for ad in animal_age_data:
                    col_name = f"{ad['animal_id']}_{ad['genotype']}"
                    vals = ad['age_map'].get(age)
                    row[col_name] = vals[phase_idx] if vals else np.nan
                rows.append(row)

            df = pd.DataFrame(rows)

            # Add group mean/SEM by genotype
            genotypes = set(ad['genotype'] for ad in animal_age_data)
            for genotype in sorted(genotypes):
                geno_cols = [f"{ad['animal_id']}_{ad['genotype']}" for ad in animal_age_data if ad['genotype'] == genotype]
                if geno_cols:
                    df[f'Mean_{genotype}'] = df[geno_cols].mean(axis=1)
                    df[f'SEM_{genotype}'] = df[geno_cols].sem(axis=1)
                    df[f'N_{genotype}'] = df[geno_cols].count(axis=1)

            # Write with phase label as sub-sheet or combined
            if phase == 'Overall':
                # First phase gets the sheet, write header row
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)
                ws = writer.sheets[sheet_name]
                ws.cell(row=1, column=1, value=f'{metric_name} - {phase} Mean by Age')
            else:
                # Append below with a gap
                startrow = writer.sheets[sheet_name].max_row + 2
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow + 1)
                ws = writer.sheets[sheet_name]
                ws.cell(row=startrow + 1, column=1, value=f'{metric_name} - {phase} Mean by Age')
