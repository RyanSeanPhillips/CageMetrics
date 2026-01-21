"""
Data exporter module for saving analyzed behavioral data.

Exports per-animal data including:
- Multi-panel PDF figures
- Excel workbook with timeseries data, light/dark means, and metadata
- NPZ file for fast consolidation

Supports parallel export for improved performance.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def _convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


class DataExporter:
    """Export analyzed behavioral data to files."""

    MINUTES_PER_DAY = 1440
    MINUTES_PER_HOUR = 60

    def __init__(self):
        pass

    def export_all_animals(self, analyzed_data: Dict[str, Dict[str, Any]],
                          output_dir: str, source_file: str = None,
                          figure_generator=None, parallel: bool = True,
                          progress_callback=None) -> List[str]:
        """
        Export data for all analyzed animals.

        Args:
            analyzed_data: Dictionary mapping animal_id to analysis results
            output_dir: Directory to save files (will create 'analysis' subfolder)
            source_file: Path to source data file (used to create output folder)
            figure_generator: FigureGenerator instance for PDF export
            parallel: Use parallel processing for faster export
            progress_callback: Optional callback(message, percent) for progress updates

        Returns:
            List of saved file paths
        """
        # Create analyzed folder next to data files
        if source_file:
            source_path = Path(source_file)
            analysis_dir = source_path.parent / "analyzed"
        else:
            analysis_dir = Path(output_dir) / "analyzed"

        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Prepare export tasks
        export_tasks = [
            (animal_id, animal_data, analysis_dir, figure_generator)
            for animal_id, animal_data in analyzed_data.items()
        ]

        n_animals = len(export_tasks)

        if parallel and n_animals > 1:
            # Parallel export using threads (I/O bound)
            max_workers = min(multiprocessing.cpu_count(), n_animals, 8)
            print(f"[Export] Exporting {n_animals} animals using {max_workers} workers...")

            saved_files = []
            completed = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self._export_animal_task, task): task[0]
                    for task in export_tasks
                }

                # Collect results as they complete
                from concurrent.futures import as_completed
                for future in as_completed(futures):
                    animal_id = futures[future]
                    try:
                        files = future.result()
                        saved_files.extend(files)
                    except Exception as e:
                        print(f"[Export] Error exporting {animal_id}: {e}")

                    completed += 1
                    if progress_callback:
                        pct = int(30 + (60 * completed / n_animals))
                        progress_callback(f"Exported {completed}/{n_animals}...", pct)

            return saved_files
        else:
            # Sequential export
            saved_files = []
            for i, (animal_id, animal_data, analysis_dir, fig_gen) in enumerate(export_tasks):
                files = self.export_animal(animal_id, animal_data, analysis_dir, fig_gen)
                saved_files.extend(files)

                if progress_callback:
                    pct = int(30 + (60 * (i + 1) / n_animals))
                    progress_callback(f"Exported {i + 1}/{n_animals}...", pct)

            return saved_files

    def _export_animal_task(self, task: Tuple) -> List[str]:
        """Worker function for parallel export."""
        animal_id, animal_data, output_dir, figure_generator = task
        return self.export_animal(animal_id, animal_data, output_dir, figure_generator)

    def export_animal(self, animal_id: str, animal_data: Dict[str, Any],
                     output_dir: Path, figure_generator=None) -> List[str]:
        """
        Export data for a single animal.

        Args:
            animal_id: Animal identifier
            animal_data: Analysis results for this animal
            output_dir: Directory to save files
            figure_generator: FigureGenerator instance for PDF export

        Returns:
            List of saved file paths
        """
        saved_files = []

        # Generate filename base from animal info
        base_name = self._create_filename(animal_data)

        # Export PDF figures
        if figure_generator:
            pdf_file = self._export_pdf(animal_id, animal_data, output_dir, base_name, figure_generator)
            if pdf_file:
                saved_files.append(str(pdf_file))

        # Export Excel workbook with all data
        excel_file = self._export_excel(animal_id, animal_data, output_dir, base_name)
        if excel_file:
            saved_files.append(str(excel_file))

        # Export NPZ for fast consolidation
        npz_file = self._export_npz(animal_id, animal_data, output_dir, base_name)
        if npz_file:
            saved_files.append(str(npz_file))

        return saved_files

    def _create_filename(self, animal_data: Dict[str, Any]) -> str:
        """Create filename from animal information.

        Format: CageMetrics_<AnimalID>_<Sex>_<Cohort>_<CageID>_Cagemate-<CagemateID>_<CagemateSex>
        """
        metadata = animal_data.get('metadata', {})

        animal_id = metadata.get('animal_id', 'Unknown')
        sex = metadata.get('sex', 'Unknown')
        cohort = metadata.get('cohort', 'Unknown')
        cage_id = metadata.get('cage_id', 'Unknown')

        # Get companion/cagemate ID
        companion = metadata.get('companion', None)
        if isinstance(companion, list):
            companion = '-'.join(str(c) for c in companion)
        companion = companion if companion else 'Single'

        # Get cagemate sex
        cagemate_sex = metadata.get('cagemate_sex', 'Unknown')
        if not cagemate_sex or cagemate_sex == 'Unknown':
            # If no cagemate, don't include sex
            cagemate_sex = '' if companion == 'Single' else 'Unknown'

        # Clean up values for filename (remove spaces, special chars)
        def clean(s):
            return str(s).replace(' ', '_').replace('/', '-').replace('\\', '-')

        # Build filename: AnimalID_Sex_Cohort_CageID_Cagemate-ID_CagemateSex
        parts = [
            f"CageMetrics_{clean(animal_id)}",
            clean(sex),
            clean(cohort),
            f"Cage-{clean(cage_id)}",
            f"Cagemate-{clean(companion)}"
        ]

        # Add cagemate sex if applicable
        if cagemate_sex and companion != 'Single':
            parts.append(clean(cagemate_sex))

        return '_'.join(parts)

    def get_filenames_preview(self, analyzed_data: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get preview of filenames that will be created for all animals."""
        filenames = []
        for animal_id, animal_data in analyzed_data.items():
            base_name = self._create_filename(animal_data)
            filenames.append(f"{base_name}_Figures.pdf")
            filenames.append(f"{base_name}_Data.xlsx")
            filenames.append(f"{base_name}_Data.npz")
        return filenames

    # File pattern for consolidation tab to search for
    FILE_PATTERN = "CageMetrics_*_Data.npz"

    def _export_pdf(self, animal_id: str, animal_data: Dict[str, Any],
                   output_dir: Path, base_name: str, figure_generator) -> Optional[Path]:
        """Export all figures to a multi-page PDF."""
        try:
            output_file = output_dir / f"{base_name}_Figures.pdf"

            # Generate all figure pages
            pages = figure_generator.generate_all_pages(animal_id, animal_data)

            with PdfPages(output_file) as pdf:
                for title, fig in pages:
                    pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor='none')
                    # Close figure to free memory
                    import matplotlib.pyplot as plt
                    plt.close(fig)

            return output_file

        except Exception as e:
            print(f"Error exporting PDF for {animal_id}: {e}")
            return None

    def _export_excel(self, animal_id: str, animal_data: Dict[str, Any],
                     output_dir: Path, base_name: str) -> Optional[Path]:
        """Export all data to Excel workbook with multiple tabs."""
        try:
            output_file = output_dir / f"{base_name}_Data.xlsx"

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Tab 1: Summary (Animal Info, Recording Info, Data Quality, Summary Stats)
                self._write_summary_tab(writer, animal_data)

                # Tabs 2+: One tab per metric with timeseries and light/dark means
                metrics = animal_data.get('metrics', {})
                for metric_name, metric_data in metrics.items():
                    self._write_metric_tab(writer, metric_name, metric_data, animal_data)

                # Sleep Analysis tabs (if sleep analysis was performed)
                sleep_analysis = animal_data.get('sleep_analysis', {})
                if sleep_analysis:
                    self._write_sleep_bouts_tab(writer, sleep_analysis)
                    self._write_sleep_stats_tab(writer, sleep_analysis)

            return output_file

        except Exception as e:
            print(f"Error exporting Excel for {animal_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _write_summary_tab(self, writer: pd.ExcelWriter, animal_data: Dict[str, Any]):
        """Write summary information to the first Excel tab."""
        metadata = animal_data.get('metadata', {})
        quality = animal_data.get('quality', {})
        metrics = animal_data.get('metrics', {})

        rows = []

        # Animal Information Section
        rows.append({'Category': 'ANIMAL INFORMATION', 'Field': '', 'Value': ''})
        rows.append({'Category': '', 'Field': 'Animal ID', 'Value': metadata.get('animal_id', '')})
        rows.append({'Category': '', 'Field': 'Genotype', 'Value': metadata.get('genotype', '')})
        rows.append({'Category': '', 'Field': 'Genotype (Raw)', 'Value': metadata.get('genotype_raw', '')})
        rows.append({'Category': '', 'Field': 'Sex', 'Value': metadata.get('sex', '')})
        rows.append({'Category': '', 'Field': 'Cohort', 'Value': metadata.get('cohort', '')})
        rows.append({'Category': '', 'Field': 'Cage ID', 'Value': metadata.get('cage_id', '')})
        cagemate = metadata.get('companion', None)
        if isinstance(cagemate, list):
            cagemate = ', '.join(cagemate)
        rows.append({'Category': '', 'Field': 'Cagemate', 'Value': cagemate or 'None'})
        rows.append({'Category': '', 'Field': '', 'Value': ''})

        # Recording Information Section
        rows.append({'Category': 'RECORDING INFORMATION', 'Field': '', 'Value': ''})
        rows.append({'Category': '', 'Field': 'Start Time', 'Value': metadata.get('start_time', '')})
        rows.append({'Category': '', 'Field': 'End Time', 'Value': metadata.get('end_time', '')})
        rows.append({'Category': '', 'Field': 'Days Analyzed', 'Value': metadata.get('n_days_analyzed', 0)})
        rows.append({'Category': '', 'Field': 'Total Minutes', 'Value': metadata.get('total_minutes', 0)})
        rows.append({'Category': '', 'Field': 'ZT0 at Minute', 'Value': metadata.get('zt0_minute', 0)})
        rows.append({'Category': '', 'Field': '', 'Value': ''})

        # Data Quality Section
        rows.append({'Category': 'DATA QUALITY', 'Field': 'Metric', 'Value': 'Coverage %', 'Extra1': 'Missing', 'Extra2': 'Rating'})
        for metric_name, q_info in quality.items():
            rows.append({
                'Category': '',
                'Field': metric_name,
                'Value': f"{q_info.get('coverage', 0):.1f}%",
                'Extra1': q_info.get('missing', 0),
                'Extra2': q_info.get('rating', '')
            })
        rows.append({'Category': '', 'Field': '', 'Value': ''})

        # Summary Statistics Section
        rows.append({'Category': 'SUMMARY STATISTICS', 'Field': 'Metric', 'Value': 'Dark Mean', 'Extra1': 'Light Mean', 'Extra2': 'Diff', 'Extra3': 'Ratio'})
        for metric_name, metric_data in metrics.items():
            dark_m = metric_data.get('dark_mean', np.nan)
            light_m = metric_data.get('light_mean', np.nan)
            diff = light_m - dark_m if not (np.isnan(dark_m) or np.isnan(light_m)) else np.nan
            ratio = light_m / dark_m if dark_m != 0 and not np.isnan(dark_m) else np.nan

            rows.append({
                'Category': '',
                'Field': metric_name,
                'Value': f"{dark_m:.4f}" if not np.isnan(dark_m) else 'N/A',
                'Extra1': f"{light_m:.4f}" if not np.isnan(light_m) else 'N/A',
                'Extra2': f"{diff:+.4f}" if not np.isnan(diff) else 'N/A',
                'Extra3': f"{ratio:.4f}" if not np.isnan(ratio) else 'N/A'
            })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='Summary', index=False)

    def _write_metric_tab(self, writer: pd.ExcelWriter, metric_name: str,
                         metric_data: Dict[str, Any], animal_data: Dict[str, Any]):
        """Write metric data to an Excel tab with timeseries and light/dark means."""
        daily_data = metric_data.get('daily_data', [])
        cta = metric_data.get('cta', np.array([]))
        cta_sem = metric_data.get('cta_sem', np.array([]))
        n_days = len(daily_data)

        if n_days == 0:
            return

        # Clean sheet name (Excel has 31 char limit and restrictions)
        sheet_name = metric_name[:31].replace('/', '_').replace('\\', '_').replace('*', '').replace('?', '').replace('[', '').replace(']', '')

        # === Build timeseries data ===
        # Columns: ZT_Minute, ZT_Hour, D1, D2, ..., Dn, Mean, SEM
        data = {
            'ZT_Minute': list(range(self.MINUTES_PER_DAY)),
            'ZT_Hour': [m / self.MINUTES_PER_HOUR for m in range(self.MINUTES_PER_DAY)],
        }

        # Add each day's data
        for day_idx, day_values in enumerate(daily_data):
            data[f'D{day_idx + 1}'] = day_values

        # Add Mean and SEM
        data['Mean'] = cta if len(cta) == self.MINUTES_PER_DAY else [np.nan] * self.MINUTES_PER_DAY
        data['SEM'] = cta_sem if len(cta_sem) == self.MINUTES_PER_DAY else [np.nan] * self.MINUTES_PER_DAY

        # Create DataFrame for timeseries
        df_timeseries = pd.DataFrame(data)

        # === Build light/dark means section ===
        # Calculate per-day light and dark means
        light_dark_data = []

        for day_idx, day_values in enumerate(daily_data):
            day_arr = np.array(day_values)
            # Light phase: ZT0-12 (minutes 0-719)
            light_mean = np.nanmean(day_arr[:720])
            # Dark phase: ZT12-24 (minutes 720-1439)
            dark_mean = np.nanmean(day_arr[720:])

            light_dark_data.append({
                'Day': f'D{day_idx + 1}',
                'Light_Mean': light_mean,
                'Dark_Mean': dark_mean
            })

        # Add overall means
        all_light_means = [d['Light_Mean'] for d in light_dark_data]
        all_dark_means = [d['Dark_Mean'] for d in light_dark_data]
        light_dark_data.append({
            'Day': 'MEAN',
            'Light_Mean': np.nanmean(all_light_means),
            'Dark_Mean': np.nanmean(all_dark_means)
        })

        df_light_dark = pd.DataFrame(light_dark_data)

        # === Combine into one sheet ===
        # Write timeseries starting at column A
        df_timeseries.to_excel(writer, sheet_name=sheet_name, index=False, startcol=0)

        # Write light/dark means starting a few columns to the right
        # Leave 2 empty columns for visual separation
        start_col = len(df_timeseries.columns) + 2
        df_light_dark.to_excel(writer, sheet_name=sheet_name, index=False, startcol=start_col)

    def _write_sleep_bouts_tab(self, writer: pd.ExcelWriter, sleep_analysis: Dict[str, Any]):
        """Write individual sleep bout data to Excel tab."""
        bouts = sleep_analysis.get('bouts', [])

        if not bouts:
            # Write empty tab with message
            df = pd.DataFrame({'Message': ['No sleep bouts detected']})
            df.to_excel(writer, sheet_name='Sleep Bouts', index=False)
            return

        # Convert bouts to DataFrame
        bout_data = []
        for bout in bouts:
            bout_data.append({
                'Day': bout.day + 1,  # 1-indexed
                'Bout_Number': bout.bout_num + 1,  # 1-indexed
                'Start_ZT_Hour': bout.start_minute / 60,
                'End_ZT_Hour': bout.end_minute / 60,
                'Start_Minute': bout.start_minute,
                'End_Minute': bout.end_minute,
                'Duration_Minutes': bout.duration,
                'Phase': bout.phase.capitalize(),
            })

        df = pd.DataFrame(bout_data)
        df.to_excel(writer, sheet_name='Sleep Bouts', index=False)

    def _write_sleep_stats_tab(self, writer: pd.ExcelWriter, sleep_analysis: Dict[str, Any]):
        """Write sleep analysis statistics summary to Excel tab."""
        light_stats = sleep_analysis.get('light_stats', {})
        dark_stats = sleep_analysis.get('dark_stats', {})
        total_stats = sleep_analysis.get('total_stats', {})
        per_day_stats = sleep_analysis.get('per_day_stats', [])
        params = sleep_analysis.get('parameters', {})
        n_days = sleep_analysis.get('n_days', 0)

        rows = []

        # Analysis Parameters Section
        rows.append({'Category': 'ANALYSIS PARAMETERS', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})
        rows.append({'Category': '', 'Metric': 'Threshold', 'Light': params.get('threshold', 0.5), 'Dark': '', 'Total': ''})
        rows.append({'Category': '', 'Metric': 'Bin Width (min)', 'Light': params.get('bin_width', 5.0), 'Dark': '', 'Total': ''})
        rows.append({'Category': '', 'Metric': 'Number of Days', 'Light': n_days, 'Dark': '', 'Total': ''})
        rows.append({'Category': '', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})

        # Summary Statistics Section
        rows.append({'Category': 'SUMMARY STATISTICS', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})

        stat_metrics = [
            ('Total Sleep (min)', 'total_minutes'),
            ('Sleep %', 'percent_time'),
            ('Bout Count', 'bout_count'),
            ('Mean Duration (min)', 'mean_duration'),
            ('Median Duration (min)', 'median_duration'),
            ('Max Duration (min)', 'max_duration'),
            ('Min Duration (min)', 'min_duration'),
            ('Std Dev (min)', 'std_duration'),
        ]

        for label, key in stat_metrics:
            rows.append({
                'Category': '',
                'Metric': label,
                'Light': light_stats.get(key, 0),
                'Dark': dark_stats.get(key, 0),
                'Total': total_stats.get(key, 0),
            })

        rows.append({'Category': '', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})

        # Derived Metrics / Interpretation Section
        rows.append({'Category': 'INTERPRETATION METRICS', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})

        # Recording hours per phase
        recording_hours_light = n_days * 12
        recording_hours_dark = n_days * 12
        recording_hours_total = n_days * 24

        # Bouts per recording hour
        light_bouts = light_stats.get('bout_count', 0)
        dark_bouts = dark_stats.get('bout_count', 0)
        total_bouts = total_stats.get('bout_count', 0)

        bouts_per_hr_light = light_bouts / recording_hours_light if recording_hours_light > 0 else 0
        bouts_per_hr_dark = dark_bouts / recording_hours_dark if recording_hours_dark > 0 else 0
        bouts_per_hr_total = total_bouts / recording_hours_total if recording_hours_total > 0 else 0

        rows.append({
            'Category': '',
            'Metric': 'Bouts per Recording Hour',
            'Light': f'{bouts_per_hr_light:.2f}',
            'Dark': f'{bouts_per_hr_dark:.2f}',
            'Total': f'{bouts_per_hr_total:.2f}',
        })

        # Fragmentation Index (bouts per hour of sleep)
        light_sleep_hrs = light_stats.get('total_minutes', 0) / 60
        dark_sleep_hrs = dark_stats.get('total_minutes', 0) / 60
        total_sleep_hrs = total_stats.get('total_minutes', 0) / 60

        frag_light = light_bouts / light_sleep_hrs if light_sleep_hrs > 0 else 0
        frag_dark = dark_bouts / dark_sleep_hrs if dark_sleep_hrs > 0 else 0
        frag_total = total_bouts / total_sleep_hrs if total_sleep_hrs > 0 else 0

        rows.append({
            'Category': '',
            'Metric': 'Fragmentation Index (bouts/hr sleep)',
            'Light': f'{frag_light:.2f}',
            'Dark': f'{frag_dark:.2f}',
            'Total': f'{frag_total:.2f}',
        })

        rows.append({'Category': '', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})

        # Definitions
        rows.append({'Category': 'DEFINITIONS', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})
        rows.append({'Category': '', 'Metric': 'Bouts/Recording Hour: Number of sleep bouts per hour of total recording time', 'Light': '', 'Dark': '', 'Total': ''})
        rows.append({'Category': '', 'Metric': 'Fragmentation Index: Number of sleep bouts per hour of actual sleep (higher = more fragmented)', 'Light': '', 'Dark': '', 'Total': ''})
        rows.append({'Category': '', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})

        # Per-Day Statistics Section
        if per_day_stats:
            rows.append({'Category': 'PER-DAY STATISTICS', 'Metric': '', 'Light': '', 'Dark': '', 'Total': ''})
            for day_stat in per_day_stats:
                day_num = day_stat.get('day', 0)
                light = day_stat.get('light', {})
                dark = day_stat.get('dark', {})

                rows.append({
                    'Category': f'Day {day_num}',
                    'Metric': 'Total Sleep (min)',
                    'Light': light.get('total_minutes', 0),
                    'Dark': dark.get('total_minutes', 0),
                    'Total': day_stat.get('total_sleep_minutes', 0),
                })
                rows.append({
                    'Category': '',
                    'Metric': 'Bout Count',
                    'Light': light.get('bout_count', 0),
                    'Dark': dark.get('bout_count', 0),
                    'Total': day_stat.get('total_bouts', 0),
                })
                rows.append({
                    'Category': '',
                    'Metric': 'Mean Duration (min)',
                    'Light': f"{light.get('mean_duration', 0):.1f}",
                    'Dark': f"{dark.get('mean_duration', 0):.1f}",
                    'Total': '',
                })

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name='Sleep Stats', index=False)

    def _export_npz(self, animal_id: str, animal_data: Dict[str, Any],
                   output_dir: Path, base_name: str) -> Optional[Path]:
        """Export all data to NPZ file for fast consolidation."""
        try:
            output_file = output_dir / f"{base_name}_Data.npz"

            # Prepare data for NPZ
            npz_data = {}

            # Metadata as JSON string - convert numpy types to native Python
            metadata = animal_data.get('metadata', {})
            npz_data['metadata_json'] = json.dumps(_convert_to_serializable(metadata))

            # Quality as JSON string - convert numpy types to native Python
            quality = animal_data.get('quality', {})
            quality_export = {}
            for metric_name, q_info in quality.items():
                quality_export[metric_name] = {
                    'coverage': float(q_info.get('coverage', 0)),
                    'missing': int(q_info.get('missing', 0)),
                    'rating': str(q_info.get('rating', 'Unknown'))
                }
            npz_data['quality_json'] = json.dumps(quality_export)

            # Metrics data
            metrics = animal_data.get('metrics', {})
            metric_names = list(metrics.keys())
            npz_data['metric_names'] = np.array(metric_names, dtype=object)

            for metric_name in metric_names:
                metric_data = metrics[metric_name]
                # Clean metric name for use as key
                key_base = metric_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'pct')

                # CTA and SEM
                npz_data[f'{key_base}_cta'] = metric_data.get('cta', np.array([]))
                npz_data[f'{key_base}_cta_sem'] = metric_data.get('cta_sem', np.array([]))

                # Daily data as 2D array
                daily_data = metric_data.get('daily_data', [])
                if daily_data:
                    npz_data[f'{key_base}_daily'] = np.array(daily_data)

                # Light/dark means
                npz_data[f'{key_base}_dark_mean'] = metric_data.get('dark_mean', np.nan)
                npz_data[f'{key_base}_light_mean'] = metric_data.get('light_mean', np.nan)
                npz_data[f'{key_base}_overall_mean'] = metric_data.get('overall_mean', np.nan)
                npz_data[f'{key_base}_overall_std'] = metric_data.get('overall_std', np.nan)

            # Sleep analysis data (if performed)
            sleep_analysis = animal_data.get('sleep_analysis', {})
            if sleep_analysis:
                # Parameters
                params = sleep_analysis.get('parameters', {})
                npz_data['sleep_threshold'] = params.get('threshold', 0.5)
                npz_data['sleep_bin_width'] = params.get('bin_width', 5.0)
                npz_data['sleep_n_days'] = sleep_analysis.get('n_days', 0)

                # Bout data as structured array
                bouts = sleep_analysis.get('bouts', [])
                if bouts:
                    bout_array = np.array([
                        (b.day, b.bout_num, b.start_minute, b.end_minute, b.duration, 0 if b.phase == 'light' else 1)
                        for b in bouts
                    ], dtype=[
                        ('day', 'i4'), ('bout_num', 'i4'), ('start_minute', 'i4'),
                        ('end_minute', 'i4'), ('duration', 'f4'), ('phase', 'i4')
                    ])
                    npz_data['sleep_bouts'] = bout_array

                # Statistics as JSON (including quality_metrics)
                stats_export = {
                    'light': _convert_to_serializable(sleep_analysis.get('light_stats', {})),
                    'dark': _convert_to_serializable(sleep_analysis.get('dark_stats', {})),
                    'total': _convert_to_serializable(sleep_analysis.get('total_stats', {})),
                    'per_day': _convert_to_serializable(sleep_analysis.get('per_day_stats', [])),
                    'quality_metrics': _convert_to_serializable(sleep_analysis.get('quality_metrics', {})),
                }
                npz_data['sleep_stats_json'] = json.dumps(stats_export)

                # Histograms
                hist_light = sleep_analysis.get('histogram_light', ([], []))
                hist_dark = sleep_analysis.get('histogram_dark', ([], []))
                npz_data['sleep_hist_light_edges'] = np.array(hist_light[0])
                npz_data['sleep_hist_light_counts'] = np.array(hist_light[1])
                npz_data['sleep_hist_dark_edges'] = np.array(hist_dark[0])
                npz_data['sleep_hist_dark_counts'] = np.array(hist_dark[1])

            # Save
            np.savez_compressed(output_file, **npz_data)

            return output_file

        except Exception as e:
            print(f"Error exporting NPZ for {animal_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
