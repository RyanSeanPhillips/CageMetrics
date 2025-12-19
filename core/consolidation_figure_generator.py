"""
Consolidation figure generator for multi-experiment preview.

Creates matplotlib figures for consolidated data preview:
- Summary page with filter criteria, experiment list, completeness
- Consolidated traces pages with grand mean CTA + SEM (no individual traces)
- Statistics page with bar charts showing SEM across experiments
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


class ConsolidationFigureGenerator:
    """Generate matplotlib figures for consolidated experiment preview."""

    # Dark theme colors (consistent with FigureGenerator)
    BG_COLOR = '#2d2d2d'
    TEXT_COLOR = '#ffffff'
    GRID_COLOR = '#4d4d4d'
    DARK_PHASE_COLOR = '#2d2d2d'
    LIGHT_PHASE_COLOR = '#4a4a2a'

    # Color for grand mean CTA
    CTA_COLOR = '#3daee9'  # Blue
    SEM_ALPHA = 0.3

    def __init__(self):
        pass

    def generate_all_pages(self, experiments: List[Dict[str, Any]],
                           filter_description: str) -> List[Tuple[str, Figure]]:
        """
        Generate all figure pages for consolidated experiments.

        Args:
            experiments: List of experiment data dicts (from _load_npz_full)
            filter_description: Human-readable filter criteria string

        Returns:
            List of (title, figure) tuples
        """
        pages = []

        # Page 1: Summary with experiment list and completeness
        fig_summary = self.create_summary_page(experiments, filter_description)
        pages.append(("Summary", fig_summary))

        # Get common metrics across all experiments
        common_metrics = self._get_common_metrics(experiments)
        metrics_per_page = 3

        # Pages 2+: Consolidated traces (3 metrics per page)
        for page_idx in range(0, len(common_metrics), metrics_per_page):
            page_metrics = common_metrics[page_idx:page_idx + metrics_per_page]
            page_num = page_idx // metrics_per_page + 1
            fig_traces = self.create_consolidated_traces_page(experiments, page_metrics, page_num)
            pages.append((f"Traces {page_num}", fig_traces))

        # Final page: Statistics with error bars
        fig_stats = self.create_statistics_page(experiments, common_metrics)
        pages.append(("Statistics", fig_stats))

        # Sleep analysis page (if sleep data available)
        fig_sleep = self.create_sleep_analysis_page(experiments)
        if fig_sleep is not None:
            pages.append(("Sleep Analysis", fig_sleep))

        return pages

    def _get_common_metrics(self, experiments: List[Dict[str, Any]]) -> List[str]:
        """Get list of metrics common to all experiments."""
        if not experiments:
            return []

        # Start with metrics from first experiment
        common = set(experiments[0].get('metric_names', []))

        # Intersect with other experiments
        for exp in experiments[1:]:
            common &= set(exp.get('metric_names', []))

        return sorted(list(common))

    def create_summary_page(self, experiments: List[Dict[str, Any]],
                            filter_description: str) -> Figure:
        """Create summary page with filter criteria, experiment list, and completeness."""
        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

        n_experiments = len(experiments)
        fig.suptitle(f"Consolidation Summary ({n_experiments} Experiments)",
                     fontsize=16, fontweight='bold', color=self.TEXT_COLOR, y=0.96)

        # Create grid: filter info (top), experiment table (middle), quality (bottom)
        gs = GridSpec(3, 1, figure=fig, height_ratios=[0.15, 1.2, 0.6],
                      hspace=0.2, left=0.06, right=0.94, top=0.90, bottom=0.06)

        # === Filter Criteria Box (top) ===
        ax_filter = fig.add_subplot(gs[0])
        ax_filter.set_facecolor(self.BG_COLOR)
        ax_filter.axis('off')

        ax_filter.text(0.02, 0.8, "Filter Criteria:", fontsize=11, fontweight='bold',
                       transform=ax_filter.transAxes, color=self.TEXT_COLOR, va='top')
        ax_filter.text(0.15, 0.8, filter_description, fontsize=10,
                       transform=ax_filter.transAxes, color='#aaaaaa', va='top')

        # === Experiment Table (middle) ===
        ax_table = fig.add_subplot(gs[1])
        ax_table.set_facecolor(self.BG_COLOR)
        ax_table.axis('off')

        ax_table.text(0.02, 0.98, "Selected Experiments", fontsize=12, fontweight='bold',
                      transform=ax_table.transAxes, color=self.TEXT_COLOR, va='top')

        # Table header
        header = f"{'#':<3}  {'Animal ID':<12}  {'Genotype':<10}  {'Sex':<5}  {'Cohort':<10}  {'Cage':<6}  {'Cagemate':<12}  {'Days':>4}"
        ax_table.text(0.02, 0.92, header, fontsize=9, fontweight='bold',
                      transform=ax_table.transAxes, color=self.TEXT_COLOR,
                      family='monospace', va='top')

        ax_table.text(0.02, 0.88, "-" * 80, fontsize=9,
                      transform=ax_table.transAxes, color=self.GRID_COLOR,
                      family='monospace', va='top')

        # Table rows
        y_pos = 0.84
        for idx, exp in enumerate(experiments, 1):
            meta = exp.get('metadata', {})
            animal_id = str(meta.get('animal_id', 'Unknown'))[:10]
            genotype = str(meta.get('genotype', 'Unknown'))[:8]
            sex = str(meta.get('sex', 'Unknown'))[:3]
            cohort = str(meta.get('cohort', 'Unknown'))[:8]
            cage = str(meta.get('cage_id', 'Unknown'))[:4]
            days = meta.get('n_days_analyzed', 0)

            # Get cagemate ID (companion field)
            companion = meta.get('companion', '')
            if isinstance(companion, list):
                cagemate = ', '.join(str(c) for c in companion)[:10]
            else:
                cagemate = str(companion)[:10] if companion else '-'

            row = f"{idx:<3}  {animal_id:<12}  {genotype:<10}  {sex:<5}  {cohort:<10}  {cage:<6}  {cagemate:<12}  {days:>4}"
            ax_table.text(0.02, y_pos, row, fontsize=8,
                          transform=ax_table.transAxes, color=self.TEXT_COLOR,
                          family='monospace', va='top')
            y_pos -= 0.04

            if y_pos < 0.05:  # Stop if running out of space
                remaining = n_experiments - idx
                if remaining > 0:
                    ax_table.text(0.02, y_pos, f"... and {remaining} more experiments",
                                  fontsize=8, transform=ax_table.transAxes,
                                  color='#888888', family='monospace', va='top')
                break

        # === Overall Quality Summary (bottom) ===
        ax_quality = fig.add_subplot(gs[2])
        ax_quality.set_facecolor(self.BG_COLOR)
        ax_quality.axis('off')

        ax_quality.text(0.02, 0.95, "Data Coverage Summary", fontsize=12, fontweight='bold',
                        transform=ax_quality.transAxes, color=self.TEXT_COLOR, va='top')

        # Calculate average completeness across experiments
        common_metrics = self._get_common_metrics(experiments)
        if common_metrics:
            quality_header = f"{'Metric':<22}  {'Avg Coverage':>12}  {'Min':>8}  {'Max':>8}"
            ax_quality.text(0.02, 0.80, quality_header, fontsize=9, fontweight='bold',
                            transform=ax_quality.transAxes, color=self.TEXT_COLOR,
                            family='monospace', va='top')

            ax_quality.text(0.02, 0.72, "-" * 56, fontsize=9,
                            transform=ax_quality.transAxes, color=self.GRID_COLOR,
                            family='monospace', va='top')

            y_pos = 0.64
            for metric_name in common_metrics[:8]:  # Show up to 8 metrics
                coverages = []
                for exp in experiments:
                    q = exp.get('quality', {}).get(metric_name, {})
                    cov = q.get('coverage', 100) if q else 100
                    coverages.append(cov)

                avg_cov = np.mean(coverages)
                min_cov = np.min(coverages)
                max_cov = np.max(coverages)

                short_name = metric_name[:20] if len(metric_name) > 20 else metric_name
                row = f"{short_name:<22}  {avg_cov:>10.1f}%  {min_cov:>7.1f}%  {max_cov:>7.1f}%"
                ax_quality.text(0.02, y_pos, row, fontsize=8,
                                transform=ax_quality.transAxes, color=self.TEXT_COLOR,
                                family='monospace', va='top')
                y_pos -= 0.08

        return fig

    def create_consolidated_traces_page(self, experiments: List[Dict[str, Any]],
                                         page_metrics: List[str], page_num: int) -> Figure:
        """
        Create consolidated traces page with grand mean CTA + SEM.

        Shows only the grand mean (mean of all experiments' CTAs) with SEM
        shading representing variability across experiments.
        """
        n_metrics = len(page_metrics)
        n_experiments = len(experiments)

        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

        fig.suptitle(f"Consolidated CTA (n={n_experiments}) - Page {page_num}",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

        # Create grid: 2 rows x 3 columns
        # Top: Stacked daily means (mean trace per day across all experiments)
        # Bottom: Grand mean CTA with SEM + completeness on y2
        gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1.2],
                      hspace=0.15, wspace=0.25, left=0.06, right=0.94, top=0.92, bottom=0.08)

        for metric_idx, metric_name in enumerate(page_metrics):
            # === Top row: Stacked daily means ===
            ax_traces = fig.add_subplot(gs[0, metric_idx])
            ax_traces.set_facecolor(self.BG_COLOR)

            self._draw_stacked_daily_means(ax_traces, experiments, metric_name,
                                            show_ylabel=(metric_idx == 0))

            # === Bottom row: Grand CTA with SEM + completeness ===
            ax_cta = fig.add_subplot(gs[1, metric_idx])
            ax_cta.set_facecolor(self.BG_COLOR)

            self._draw_grand_cta_with_sem(ax_cta, experiments, metric_name,
                                           show_xlabel=True, show_y2_label=True)

        return fig

    def _get_daily_data_length(self, daily_data):
        """Get the number of days in daily_data, handling both lists and numpy arrays."""
        if daily_data is None:
            return 0
        if hasattr(daily_data, 'shape'):
            return daily_data.shape[0] if len(daily_data.shape) > 0 else 0
        return len(daily_data) if daily_data else 0

    def _draw_stacked_daily_means(self, ax, experiments: List[Dict[str, Any]],
                                   metric_name: str, show_ylabel: bool = True):
        """Draw stacked daily mean traces (mean per day across all experiments)."""
        # Collect all daily data from all experiments
        all_daily = []
        for exp in experiments:
            metric_data = exp.get('metrics', {}).get(metric_name, {})
            daily_data = metric_data.get('daily_data', None)
            if daily_data is not None and self._get_daily_data_length(daily_data) > 0:
                # Handle numpy arrays - iterate over first dimension
                if hasattr(daily_data, 'shape'):
                    for day_row in daily_data:
                        all_daily.append(np.array(day_row))
                else:
                    all_daily.extend(daily_data)

        if not all_daily:
            ax.text(0.5, 0.5, "No data", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10)
            ax.axis('off')
            return

        # Find global min/max for consistent scaling
        all_values = np.concatenate(all_daily)
        valid_values = all_values[~np.isnan(all_values)]
        if len(valid_values) == 0:
            return

        data_min = np.percentile(valid_values, 1)
        data_max = np.percentile(valid_values, 99)
        data_range = data_max - data_min if data_max > data_min else 1

        # Group days across experiments and compute mean per day
        # Find max days across all experiments
        max_days = 0
        for exp in experiments:
            daily_data = exp.get('metrics', {}).get(metric_name, {}).get('daily_data', None)
            n_days = self._get_daily_data_length(daily_data)
            if n_days > max_days:
                max_days = n_days

        if max_days == 0:
            return

        # Create 36-hour x-axis
        x_minutes_36h = np.arange(2160)
        x_hours_36h = x_minutes_36h / 60

        # DAY_COLORS from FigureGenerator
        DAY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        for day_idx in range(max_days):
            # Collect day data from all experiments
            day_traces = []
            for exp in experiments:
                metric_data = exp.get('metrics', {}).get(metric_name, {})
                daily_data = metric_data.get('daily_data', None)
                if daily_data is not None:
                    # Handle both list and numpy array
                    if hasattr(daily_data, 'shape'):
                        # Numpy array - check first dimension
                        if day_idx < daily_data.shape[0]:
                            day_traces.append(daily_data[day_idx])
                    elif day_idx < len(daily_data):
                        day_traces.append(daily_data[day_idx])

            if not day_traces:
                continue

            # Compute mean across experiments for this day
            # Pad traces to same length (1440 minutes)
            padded_traces = []
            for trace in day_traces:
                if trace is None or len(trace) == 0:
                    continue
                if len(trace) >= 1440:
                    padded_traces.append(trace[:1440])
                else:
                    padded = np.full(1440, np.nan)
                    padded[:len(trace)] = trace
                    padded_traces.append(padded)

            if not padded_traces:
                continue

            # Suppress warnings for empty slices (expected when no valid data)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                day_mean = np.nanmean(np.array(padded_traces), axis=0)

            # Skip if all NaN
            if np.all(np.isnan(day_mean)):
                continue

            # Create 36-hour trace (add first 12h for continuity)
            day_36h = np.concatenate([day_mean, day_mean[:720]])

            # Normalize to 0-1 range
            normalized = (day_36h - data_min) / data_range
            normalized = np.clip(normalized, 0, 1)
            y_offset = day_idx
            y_values = normalized * 0.85 + y_offset + 0.075

            color = DAY_COLORS[day_idx % len(DAY_COLORS)]
            ax.plot(x_hours_36h, y_values, color=color, linewidth=0.7, alpha=0.9)

            # Day label
            ax.text(-0.3, y_offset + 0.5, f'D{day_idx + 1}', ha='right', va='center',
                    fontsize=7, color=color, fontweight='bold')

        # Add light/dark shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)

        # Formatting
        ax.set_xlim(0, 36)
        ax.set_ylim(0, max_days)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_xticklabels([])
        ax.set_yticks([])

        ax.set_title(metric_name, fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_grand_cta_with_sem(self, ax, experiments: List[Dict[str, Any]],
                                  metric_name: str, show_xlabel: bool = True,
                                  show_y2_label: bool = False):
        """
        Draw grand mean CTA with SEM shading across experiments.

        Grand mean = mean of all experiments' CTAs
        SEM = standard error across experiments (not within-experiment)
        """
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.collections import LineCollection

        # Collect CTAs from all experiments
        ctas = []
        for exp in experiments:
            metric_data = exp.get('metrics', {}).get(metric_name, {})
            cta = metric_data.get('cta', None)
            if cta is not None and len(cta) > 0 and not np.all(np.isnan(cta)):
                ctas.append(np.array(cta))

        if not ctas:
            ax.set_facecolor(self.BG_COLOR)
            ax.text(0.5, 0.5, f"No CTA data for {metric_name}", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(self.GRID_COLOR)
            return

        # Ensure all CTAs have same length (should be 1440)
        min_len = min(len(c) for c in ctas)
        ctas = [c[:min_len] for c in ctas]

        # Stack CTAs and compute grand mean + SEM across experiments
        cta_matrix = np.array(ctas)  # (n_experiments, 1440)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            grand_mean = np.nanmean(cta_matrix, axis=0)
            n_exp = len(ctas)
            grand_sem = np.nanstd(cta_matrix, axis=0) / np.sqrt(n_exp)

        # Create 36-hour version
        grand_mean_36h = np.concatenate([grand_mean, grand_mean[:720]])
        grand_sem_36h = np.concatenate([grand_sem, grand_sem[:720]])

        x_hours = np.arange(len(grand_mean_36h)) / 60

        # Smooth (15-min rolling average)
        window = 15
        mean_smooth = pd.Series(grand_mean_36h).rolling(window=window, min_periods=1, center=True).mean().values
        sem_smooth = pd.Series(grand_sem_36h).rolling(window=window, min_periods=1, center=True).mean().values

        # Add light/dark shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)

        # Plot grand CTA with SEM shading
        ax.plot(x_hours, mean_smooth, color=self.CTA_COLOR, linewidth=1.2, zorder=3)
        ax.fill_between(x_hours, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                        color=self.CTA_COLOR, alpha=self.SEM_ALPHA, zorder=2)

        # === Completeness on y2 axis ===
        ax2 = ax.twinx()

        # Compute average completeness across experiments
        completeness_list = []
        for exp in experiments:
            metric_data = exp.get('metrics', {}).get(metric_name, {})
            daily_data = metric_data.get('daily_data', [])
            if daily_data is not None and len(daily_data) > 0:
                all_data = np.concatenate(daily_data)
                # Compute per-minute completeness in bins
                bin_size = 15
                n_bins = min(len(all_data) // bin_size, 96)  # max 96 bins for 1 day
                comp = np.zeros(n_bins)
                for i in range(n_bins):
                    chunk = all_data[i * bin_size:(i + 1) * bin_size]
                    comp[i] = np.sum(~np.isnan(chunk)) / len(chunk) if len(chunk) > 0 else 0
                completeness_list.append(comp)

        if completeness_list:
            # Average completeness across experiments
            min_bins = min(len(c) for c in completeness_list)
            completeness_matrix = np.array([c[:min_bins] for c in completeness_list])
            avg_completeness = np.nanmean(completeness_matrix, axis=0)

            # Extend to 36 hours
            avg_completeness_36h = np.concatenate([avg_completeness, avg_completeness,
                                                    avg_completeness[:len(avg_completeness)//2]])
            x_complete = np.linspace(0, 36, len(avg_completeness_36h))

            # Smooth
            comp_smooth = pd.Series(avg_completeness_36h).rolling(window=3, min_periods=1, center=True).mean().values

            # Color-coded line
            cmap = LinearSegmentedColormap.from_list('completeness',
                                                     ['#e74c3c', '#f39c12', '#27ae60'])

            points = np.array([x_complete, comp_smooth * 100]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = (comp_smooth[:-1] + comp_smooth[1:]) / 2

            lc = LineCollection(segments, cmap=cmap, linewidth=1, alpha=0.6, zorder=1)
            lc.set_array(colors)
            lc.set_clim(0, 1)
            ax2.add_collection(lc)

        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y', colors='#888888', labelsize=6)
        ax2.spines['right'].set_color('#888888')
        ax2.set_yticks([0, 50, 100])

        if show_y2_label:
            ax2.set_ylabel('Completeness (%)', fontsize=6, color='#888888',
                           rotation=270, labelpad=12)

        # Formatting
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 12, 24, 36])

        if show_xlabel:
            ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36'], fontsize=7)
        else:
            ax.set_xticklabels([])

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

        # Y-axis label
        y_label = self._get_short_metric_label(metric_name)
        if y_label:
            ax.set_ylabel(y_label, fontsize=6, color=self.TEXT_COLOR)

        for spine in ['top', 'bottom', 'left']:
            ax.spines[spine].set_color(self.GRID_COLOR)

    def _get_short_metric_label(self, metric_name: str) -> str:
        """Create shortened metric label for y-axis."""
        label_map = {
            'Inactive %': 'Inactive (%)',
            'Active %': 'Active (%)',
            'Locomotion %': 'Locomotion (%)',
            'Climbing %': 'Climbing (%)',
            'Drinking %': 'Drinking (%)',
            'Feeding %': 'Feeding (%)',
            'Sleeping %': 'Sleeping (%)',
            'Distance (cm/min)': 'Distance (cm/min)',
            'Speed (cm/s)': 'Speed (cm/s)',
            'Social Distance (cm)': 'Social Dist (cm)',
            'Respiration Rate (Hz)': 'Resp Rate (Hz)',
        }
        return label_map.get(metric_name, metric_name)

    def create_statistics_page(self, experiments: List[Dict[str, Any]],
                                common_metrics: List[str]) -> Figure:
        """
        Create statistics page with bar charts showing SEM across experiments.

        Error bars represent SEM across experiments (not within-experiment).
        """
        n_experiments = len(experiments)
        n_metrics = len(common_metrics)

        fig = Figure(figsize=(14, 8.5), facecolor=self.BG_COLOR)

        fig.suptitle(f"Consolidated Statistics (n={n_experiments})",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

        if n_metrics == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No common metrics", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=14)
            ax.set_facecolor(self.BG_COLOR)
            ax.axis('off')
            return fig

        # Layout: max 7 bar charts per row, table in last row
        max_per_row = 7
        n_rows = (n_metrics + max_per_row - 1) // max_per_row

        # Add space for table
        plots_in_last_row = n_metrics - (n_rows - 1) * max_per_row
        last_row_cols = plots_in_last_row + 1

        n_cols = max(max_per_row, last_row_cols)
        gs = GridSpec(n_rows, n_cols, figure=fig,
                      hspace=0.4, wspace=0.3, left=0.04, right=0.98, top=0.92, bottom=0.06)

        # Compute statistics for each metric
        stats = {}
        for metric_name in common_metrics:
            dark_means = []
            light_means = []

            for exp in experiments:
                metric_data = exp.get('metrics', {}).get(metric_name, {})
                dark_m = metric_data.get('dark_mean', np.nan)
                light_m = metric_data.get('light_mean', np.nan)

                if not np.isnan(dark_m):
                    dark_means.append(dark_m)
                if not np.isnan(light_m):
                    light_means.append(light_m)

            dark_mean = np.mean(dark_means) if dark_means else np.nan
            light_mean = np.mean(light_means) if light_means else np.nan
            dark_sem = np.std(dark_means) / np.sqrt(len(dark_means)) if len(dark_means) > 1 else 0
            light_sem = np.std(light_means) / np.sqrt(len(light_means)) if len(light_means) > 1 else 0

            stats[metric_name] = {
                'dark_mean': dark_mean,
                'light_mean': light_mean,
                'dark_sem': dark_sem,
                'light_sem': light_sem,
                'n_dark': len(dark_means),
                'n_light': len(light_means)
            }

        # Draw bar charts
        for idx, metric_name in enumerate(common_metrics):
            row = idx // max_per_row
            col = idx % max_per_row

            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(self.BG_COLOR)

            verbose_label = self._get_short_metric_label(metric_name)
            short_title = metric_name.split(' (')[0] if ' (' in metric_name else metric_name
            short_title = short_title.replace(' %', '')

            self._draw_bar_chart_with_sem(ax, stats[metric_name], short_title, verbose_label,
                                           show_legend=(idx == n_metrics - 1))

        # Summary table
        table_start_col = plots_in_last_row
        ax_table = fig.add_subplot(gs[n_rows - 1, table_start_col:])
        ax_table.set_facecolor(self.BG_COLOR)
        ax_table.axis('off')

        self._draw_consolidated_stats_table(ax_table, stats, common_metrics)

        return fig

    def _draw_bar_chart_with_sem(self, ax, stat_data: Dict, title: str, ylabel: str,
                                  show_legend: bool = False):
        """Draw bar chart with SEM error bars across experiments."""
        dark_mean = stat_data.get('dark_mean', 0)
        light_mean = stat_data.get('light_mean', 0)
        dark_sem = stat_data.get('dark_sem', 0)
        light_sem = stat_data.get('light_sem', 0)

        # Handle NaN
        dark_mean = 0 if np.isnan(dark_mean) else dark_mean
        light_mean = 0 if np.isnan(light_mean) else light_mean

        x = np.array([0])
        width = 0.15

        ax.bar(x - width/2, [dark_mean], width, yerr=[dark_sem], capsize=3,
               label='Dark', color='#4a4a4a', edgecolor='white', linewidth=0.5,
               error_kw={'ecolor': '#888888', 'capthick': 1})
        ax.bar(x + width/2, [light_mean], width, yerr=[light_sem], capsize=3,
               label='Light', color='#f0c040', edgecolor='white', linewidth=0.5,
               error_kw={'ecolor': '#888888', 'capthick': 1})

        ax.set_ylabel(ylabel, fontsize=6, color=self.TEXT_COLOR)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        if show_legend:
            ax.legend(facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                      labelcolor=self.TEXT_COLOR, loc='upper right', fontsize=5)

        ax.grid(True, alpha=0.3, color=self.GRID_COLOR, axis='y')
        ax.set_title(title, fontsize=7, color=self.TEXT_COLOR, fontweight='bold')

        ax.set_ylim(bottom=0)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)

    def _draw_consolidated_stats_table(self, ax, stats: Dict, metric_names: List[str]):
        """Draw consolidated statistics table with SEM values."""
        header = f"{'Metric':<18}  {'Dark':>10}  {'Light':>10}  {'Diff':>9}  {'Ratio':>7}"
        ax.text(0.02, 0.95, header, fontsize=7, fontweight='bold',
                transform=ax.transAxes, color=self.TEXT_COLOR,
                family='monospace', va='top')

        ax.text(0.02, 0.88, "-" * 62, fontsize=7,
                transform=ax.transAxes, color=self.GRID_COLOR,
                family='monospace', va='top')

        y_pos = 0.80
        for metric_name in metric_names:
            s = stats.get(metric_name, {})
            dark_m = s.get('dark_mean', np.nan)
            light_m = s.get('light_mean', np.nan)
            dark_sem = s.get('dark_sem', 0)
            light_sem = s.get('light_sem', 0)

            if np.isnan(dark_m) or np.isnan(light_m):
                diff = np.nan
                ratio = np.nan
            else:
                diff = light_m - dark_m
                ratio = light_m / dark_m if dark_m != 0 else np.nan

            short_name = metric_name[:16] if len(metric_name) > 16 else metric_name

            # Format with SEM
            if not np.isnan(dark_m):
                dark_str = f"{dark_m:.1f}±{dark_sem:.1f}"
            else:
                dark_str = "N/A"

            if not np.isnan(light_m):
                light_str = f"{light_m:.1f}±{light_sem:.1f}"
            else:
                light_str = "N/A"

            diff_str = f"{diff:+.2f}" if not np.isnan(diff) else "N/A"
            ratio_str = f"{ratio:.2f}x" if not np.isnan(ratio) else "N/A"

            row = f"{short_name:<18}  {dark_str:>10}  {light_str:>10}  {diff_str:>9}  {ratio_str:>7}"

            ax.text(0.02, y_pos, row, fontsize=6,
                    transform=ax.transAxes, color=self.TEXT_COLOR,
                    family='monospace', va='top')
            y_pos -= 0.065

            if y_pos < 0.05:
                break

    def create_sleep_analysis_page(self, experiments: List[Dict[str, Any]]) -> Optional[Figure]:
        """
        Create sleep analysis page with aggregated data across all animals.

        Layout: 3 columns
        - Left: Stacked daily traces (mean ± SEM per day) + Grand CTA
        - Middle: Stacked per-day histograms + Combined histogram
        - Right: Aggregated statistics table

        Returns:
            Figure or None if no sleep data available
        """
        # Check if any experiments have sleep analysis data
        experiments_with_sleep = []
        for exp in experiments:
            sleep_data = exp.get('sleep_analysis', None)
            if sleep_data is not None:
                experiments_with_sleep.append(exp)

        if not experiments_with_sleep:
            return None

        n_animals = len(experiments_with_sleep)

        fig = Figure(figsize=(11, 10.625), facecolor=self.BG_COLOR)  # 25% taller
        fig.suptitle(f"Sleep Bout Analysis (n={n_animals})",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

        # Create grid: 3 rows x 3 columns
        # Row 0: Traces, Per-day bout count, Per-day time-in-bouts
        # Row 1: CTA, Combined bout count, Combined time-in-bouts
        # Row 2: Stats table, Bar charts grid (2x5)
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.6, 0.9, 3.0],
                      hspace=0.30, wspace=0.20, left=0.06, right=0.98, top=0.93, bottom=0.03)

        # === ROW 0 ===
        # Stacked daily traces
        ax_traces = fig.add_subplot(gs[0, 0])
        ax_traces.set_facecolor(self.BG_COLOR)
        self._draw_sleep_stacked_daily_means(ax_traces, experiments_with_sleep)

        # Per-day bout count histograms (will share x-axis with row 1)
        ax_day_count = fig.add_subplot(gs[0, 1])
        ax_day_count.set_facecolor(self.BG_COLOR)
        self._draw_sleep_stacked_histograms(ax_day_count, experiments_with_sleep, weighted=False,
                                            show_xlabel=False)

        # Per-day time-in-bouts histograms (will share x-axis with row 1)
        ax_day_time = fig.add_subplot(gs[0, 2])
        ax_day_time.set_facecolor(self.BG_COLOR)
        self._draw_sleep_stacked_histograms(ax_day_time, experiments_with_sleep, weighted=True,
                                            show_xlabel=False)

        # === ROW 1 ===
        # CTA trace
        ax_cta = fig.add_subplot(gs[1, 0])
        ax_cta.set_facecolor(self.BG_COLOR)
        self._draw_sleep_grand_cta(ax_cta, experiments_with_sleep)

        # Combined bout count histogram (shares x-axis with row 0)
        ax_count_hist = fig.add_subplot(gs[1, 1], sharex=ax_day_count)
        ax_count_hist.set_facecolor(self.BG_COLOR)
        self._draw_sleep_combined_histogram(ax_count_hist, experiments_with_sleep, weighted=False)

        # Combined time-in-bouts histogram (shares x-axis with row 0)
        ax_time_hist = fig.add_subplot(gs[1, 2], sharex=ax_day_time)
        ax_time_hist.set_facecolor(self.BG_COLOR)
        self._draw_sleep_combined_histogram(ax_time_hist, experiments_with_sleep, weighted=True)

        # === ROW 2 ===
        # Stats table (left)
        ax_stats = fig.add_subplot(gs[2, 0])
        ax_stats.set_facecolor(self.BG_COLOR)
        ax_stats.axis('off')
        self._draw_sleep_stats_compact(ax_stats, experiments_with_sleep)

        # Quality metric bar charts (spans 2 columns)
        ax_bars = fig.add_subplot(gs[2, 1:])
        ax_bars.set_facecolor(self.BG_COLOR)
        self._draw_sleep_quality_bars(ax_bars, experiments_with_sleep)

        return fig

    def _draw_sleep_stacked_daily_means(self, ax, experiments: List[Dict[str, Any]]):
        """Draw stacked daily traces showing mean ± SEM across animals per day."""
        # Collect daily data for sleeping % from all experiments
        all_daily_by_day = {}  # day_idx -> list of traces from all animals
        max_days = 0

        for exp in experiments:
            # Get sleeping % daily data
            metric_data = exp.get('metrics', {}).get('Sleeping %', {})
            daily_data = metric_data.get('daily_data', None)

            if daily_data is not None:
                n_days = self._get_daily_data_length(daily_data)
                max_days = max(max_days, n_days)

                for day_idx in range(n_days):
                    if day_idx not in all_daily_by_day:
                        all_daily_by_day[day_idx] = []

                    if hasattr(daily_data, 'shape'):
                        all_daily_by_day[day_idx].append(daily_data[day_idx])
                    else:
                        all_daily_by_day[day_idx].append(daily_data[day_idx])

        if max_days == 0:
            ax.text(0.5, 0.5, "No sleep data", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10)
            ax.axis('off')
            return

        # DAY_COLORS
        DAY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        # Create 36-hour x-axis
        x_hours_36h = np.arange(2160) / 60

        for day_idx in range(max_days):
            traces = all_daily_by_day.get(day_idx, [])
            if not traces:
                continue

            # Pad to 1440 minutes
            padded = []
            for trace in traces:
                if trace is None or len(trace) == 0:
                    continue
                if len(trace) >= 1440:
                    padded.append(trace[:1440])
                else:
                    pad_trace = np.full(1440, np.nan)
                    pad_trace[:len(trace)] = trace
                    padded.append(pad_trace)

            if not padded:
                continue

            # Compute mean and SEM across animals
            stacked = np.array(padded)
            day_mean = np.nanmean(stacked, axis=0)
            day_sem = np.nanstd(stacked, axis=0) / np.sqrt(len(padded))

            # Create 36-hour version
            mean_36h = np.concatenate([day_mean, day_mean[:720]])
            sem_36h = np.concatenate([day_sem, day_sem[:720]])

            # Normalize to 0-1 for stacking (sleeping % already 0-100)
            mean_norm = mean_36h / 100 if np.nanmax(mean_36h) > 1.5 else mean_36h
            sem_norm = sem_36h / 100 if np.nanmax(mean_36h) > 1.5 else sem_36h

            y_offset = day_idx
            y_mean = mean_norm * 0.85 + y_offset + 0.075
            y_upper = (mean_norm + sem_norm) * 0.85 + y_offset + 0.075
            y_lower = (mean_norm - sem_norm) * 0.85 + y_offset + 0.075

            color = DAY_COLORS[day_idx % len(DAY_COLORS)]
            ax.fill_between(x_hours_36h, y_lower, y_upper, color=color, alpha=0.3)
            ax.plot(x_hours_36h, y_mean, color=color, linewidth=0.8, alpha=0.9)

            # Day label
            ax.text(-0.5, y_offset + 0.5, f'D{day_idx + 1}', ha='right', va='center',
                    fontsize=7, color=color, fontweight='bold')

        # Light/dark shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)

        ax.set_xlim(0, 36)
        ax.set_ylim(0, max_days)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_title("Daily Sleep Traces (Mean ± SEM)", fontsize=9,
                     color=self.TEXT_COLOR, fontweight='bold', pad=5)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_sleep_grand_cta(self, ax, experiments: List[Dict[str, Any]]):
        """Draw grand CTA for sleeping % across all animals."""
        # Collect CTAs from all experiments
        ctas = []
        for exp in experiments:
            metric_data = exp.get('metrics', {}).get('Sleeping %', {})
            cta = metric_data.get('cta', None)
            if cta is not None and len(cta) > 0 and not np.all(np.isnan(cta)):
                ctas.append(np.array(cta))

        if not ctas:
            ax.text(0.5, 0.5, "No CTA data", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Ensure same length
        min_len = min(len(c) for c in ctas)
        ctas = [c[:min_len] for c in ctas]

        # Compute grand mean and SEM
        cta_matrix = np.array(ctas)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            grand_mean = np.nanmean(cta_matrix, axis=0)
            grand_sem = np.nanstd(cta_matrix, axis=0) / np.sqrt(len(ctas))

        # 36-hour version
        mean_36h = np.concatenate([grand_mean, grand_mean[:720]])
        sem_36h = np.concatenate([grand_sem, grand_sem[:720]])
        x_hours = np.arange(len(mean_36h)) / 60

        # Smooth
        import pandas as pd
        window = 15
        mean_smooth = pd.Series(mean_36h).rolling(window=window, min_periods=1, center=True).mean().values
        sem_smooth = pd.Series(sem_36h).rolling(window=window, min_periods=1, center=True).mean().values

        # Light/dark shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)

        # Plot
        ax.plot(x_hours, mean_smooth, color=self.CTA_COLOR, linewidth=1.2, zorder=3)
        ax.fill_between(x_hours, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                        color=self.CTA_COLOR, alpha=self.SEM_ALPHA, zorder=2)

        ax.set_xlim(0, 36)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36'], fontsize=7)
        ax.set_ylabel('Sleeping (%)', fontsize=7, color=self.TEXT_COLOR)
        ax.set_title("Grand Mean CTA", fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_sleep_stacked_histograms(self, ax, experiments: List[Dict[str, Any]], weighted: bool = False,
                                          show_xlabel: bool = True):
        """
        Draw stacked per-day histograms combining bouts from all animals (vertical bars).

        Args:
            ax: Matplotlib axes
            experiments: List of experiment data
            weighted: If True, show time-in-bouts (duration sum) instead of count
            show_xlabel: If True, show x-axis label
        """
        # Collect all bouts from all experiments
        all_bouts_by_day = {}
        max_days = 0
        bin_width = 5.0

        for exp in experiments:
            sleep_data = exp.get('sleep_analysis', {})
            bouts = sleep_data.get('bouts', [])
            n_days = sleep_data.get('n_days', 0)
            max_days = max(max_days, n_days)

            for bout in bouts:
                day = bout.day
                if day not in all_bouts_by_day:
                    all_bouts_by_day[day] = {'light': [], 'dark': []}
                all_bouts_by_day[day][bout.phase].append(bout)

        if max_days == 0:
            ax.text(0.5, 0.5, "No bout data", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10)
            ax.axis('off')
            return

        # Find max duration for consistent bins
        all_durations = []
        for day_data in all_bouts_by_day.values():
            all_durations.extend([b.duration for b in day_data['light']])
            all_durations.extend([b.duration for b in day_data['dark']])

        max_dur = max(all_durations) if all_durations else 60
        max_dur = np.ceil(max_dur / bin_width) * bin_width + bin_width
        bins = np.arange(0, max_dur + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        DAY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        # Compute histogram data for each day
        day_hist_data = []
        max_val = 1
        for day_idx in range(max_days):
            day_data = all_bouts_by_day.get(day_idx, {'light': [], 'dark': []})

            if weighted:
                # Time-weighted: sum durations in each bin
                light_vals = np.zeros(len(bins) - 1)
                dark_vals = np.zeros(len(bins) - 1)
                for b in day_data['light']:
                    bin_idx = np.searchsorted(bins[1:], b.duration)
                    if bin_idx < len(light_vals):
                        light_vals[bin_idx] += b.duration
                for b in day_data['dark']:
                    bin_idx = np.searchsorted(bins[1:], b.duration)
                    if bin_idx < len(dark_vals):
                        dark_vals[bin_idx] += b.duration
            else:
                # Count-based histogram
                light_durs = [b.duration for b in day_data['light']]
                dark_durs = [b.duration for b in day_data['dark']]
                light_vals, _ = np.histogram(light_durs, bins=bins) if light_durs else (np.zeros(len(bins)-1), bins)
                dark_vals, _ = np.histogram(dark_durs, bins=bins) if dark_durs else (np.zeros(len(bins)-1), bins)

            day_hist_data.append((light_vals, dark_vals))
            max_val = max(max_val, np.max(light_vals), np.max(dark_vals))

        # Draw stacked vertical bar histograms
        bar_width = bin_width * 0.35
        for day_idx in range(max_days):
            y_offset = day_idx
            light_vals, dark_vals = day_hist_data[day_idx]

            # Normalize to 0-0.8 range for stacking
            light_norm = (light_vals / max_val) * 0.8
            dark_norm = (dark_vals / max_val) * 0.8

            # Draw bars - light phase (yellow)
            for x, h in zip(bin_centers, light_norm):
                if h > 0:
                    ax.bar(x - bar_width/2, h, bar_width, bottom=y_offset + 0.1,
                           color='#f0c040', edgecolor='none', alpha=0.8)

            # Draw bars - dark phase (gray)
            for x, h in zip(bin_centers, dark_norm):
                if h > 0:
                    ax.bar(x + bar_width/2, h, bar_width, bottom=y_offset + 0.1,
                           color='#6a6a6a', edgecolor='none', alpha=0.8)

            # Day label
            color = DAY_COLORS[day_idx % len(DAY_COLORS)]
            ax.text(-max_dur * 0.02, y_offset + 0.5, f'D{day_idx + 1}', ha='right', va='center',
                    fontsize=7, color=color, fontweight='bold')

        # Formatting
        ax.set_xlim(0, max_dur)
        ax.set_ylim(0, max_days)
        ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel('Duration (min)', fontsize=7, color=self.TEXT_COLOR)
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)
        else:
            # Hide x-tick labels when sharing axis with plot below
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=6, labelbottom=False)

        title = 'Time in Bouts' if weighted else 'Bout Count'
        ax.set_title(f'Per-Day {title}', fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#f0c040', label='Light'),
            Patch(facecolor='#6a6a6a', label='Dark')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=5,
                  facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR, labelcolor=self.TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_sleep_combined_histogram(self, ax, experiments: List[Dict[str, Any]], weighted: bool = False,
                                        show_title: bool = True):
        """
        Draw combined histogram with light/dark overlay across all animals.

        Args:
            ax: Matplotlib axes
            experiments: List of experiment data
            weighted: If True, show time-weighted (duration × count) instead of count
            show_title: If True, show the title
        """
        # Collect all bouts
        light_durations = []
        dark_durations = []
        bin_width = 5.0

        for exp in experiments:
            sleep_data = exp.get('sleep_analysis', {})
            bouts = sleep_data.get('bouts', [])
            for bout in bouts:
                if bout.phase == 'light':
                    light_durations.append(bout.duration)
                else:
                    dark_durations.append(bout.duration)

        if not light_durations and not dark_durations:
            ax.text(0.5, 0.5, "No bout data", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Compute bins
        all_durations = light_durations + dark_durations
        max_dur = max(all_durations) if all_durations else 60
        max_dur = np.ceil(max_dur / bin_width) * bin_width + bin_width
        bins = np.arange(0, max_dur + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if weighted:
            # Time-weighted: compute total time in each bin (sum of durations)
            light_weights = np.zeros(len(bins) - 1)
            dark_weights = np.zeros(len(bins) - 1)

            for dur in light_durations:
                bin_idx = np.searchsorted(bins[1:], dur)
                if bin_idx < len(light_weights):
                    light_weights[bin_idx] += dur

            for dur in dark_durations:
                bin_idx = np.searchsorted(bins[1:], dur)
                if bin_idx < len(dark_weights):
                    dark_weights[bin_idx] += dur

            # Plot as bars
            bar_width = bin_width * 0.4
            if np.any(light_weights > 0):
                ax.bar(bin_centers - bar_width/2, light_weights, bar_width,
                       alpha=0.7, color='#f0c040', edgecolor='white', linewidth=0.5, label='Light')
            if np.any(dark_weights > 0):
                ax.bar(bin_centers + bar_width/2, dark_weights, bar_width,
                       alpha=0.7, color='#4a4a4a', edgecolor='white', linewidth=0.5, label='Dark')

            ax.set_ylabel('Time (min)', fontsize=7, color=self.TEXT_COLOR)
            if show_title:
                ax.set_title("Time in Bouts", fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)
        else:
            # Standard count histogram
            if light_durations:
                ax.hist(light_durations, bins=bins, alpha=0.7, color='#f0c040',
                        edgecolor='white', linewidth=0.5, label='Light')
            if dark_durations:
                ax.hist(dark_durations, bins=bins, alpha=0.7, color='#4a4a4a',
                        edgecolor='white', linewidth=0.5, label='Dark')

            ax.set_ylabel('Count', fontsize=7, color=self.TEXT_COLOR)
            if show_title:
                ax.set_title("Bout Count", fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)

        ax.set_xlabel('Duration (min)', fontsize=7, color=self.TEXT_COLOR)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)
        ax.legend(loc='upper right', fontsize=5, facecolor=self.BG_COLOR,
                  edgecolor=self.GRID_COLOR, labelcolor=self.TEXT_COLOR)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_sleep_aggregated_stats(self, ax, experiments: List[Dict[str, Any]]):
        """Draw aggregated sleep statistics across all animals."""
        # Collect stats from all experiments
        light_stats_list = []
        dark_stats_list = []
        total_stats_list = []

        for exp in experiments:
            sleep_data = exp.get('sleep_analysis', {})
            if sleep_data:
                light_stats_list.append(sleep_data.get('light_stats', {}))
                dark_stats_list.append(sleep_data.get('dark_stats', {}))
                total_stats_list.append(sleep_data.get('total_stats', {}))

        n_animals = len(experiments)

        # Helper to compute mean ± SEM
        def mean_sem(values):
            values = [v for v in values if v is not None and not np.isnan(v)]
            if not values:
                return np.nan, np.nan
            mean = np.mean(values)
            sem = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            return mean, sem

        # Compute aggregated stats
        def get_values(stats_list, key):
            return [s.get(key, np.nan) for s in stats_list if s]

        y_pos = 0.95

        # Title
        ax.text(0.5, y_pos, f"Sleep Statistics (n={n_animals})", fontsize=10, fontweight='bold',
                transform=ax.transAxes, color=self.TEXT_COLOR, ha='center', va='top')
        y_pos -= 0.06

        # Table header
        header = f"{'Metric':<18} {'Light':>14} {'Dark':>14} {'Total':>14}"
        ax.text(0.02, y_pos, header, fontsize=6.5, fontweight='bold',
                transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')
        y_pos -= 0.025
        ax.text(0.02, y_pos, "-" * 62, fontsize=6, transform=ax.transAxes,
                color=self.GRID_COLOR, family='monospace', va='top')
        y_pos -= 0.03

        # Key metrics (bold)
        key_metrics = ['total_minutes', 'percent_time', 'bout_count', 'mean_duration']

        # Stats rows
        stats_rows = [
            ('Total Sleep (min)', 'total_minutes', False),
            ('Sleep %', 'percent_time', True),
            ('Number of Bouts', 'bout_count', True),
            ('Bouts/rec hour', None, True),  # Computed
            ('Mean Duration', 'mean_duration', True),
            ('Median Duration', 'median_duration', False),
            ('Max Duration', 'max_duration', False),
            ('Frag Index', None, True),  # Computed
        ]

        for label, key, is_bold in stats_rows:
            if key == 'bout_count':
                # Sum bout counts
                light_vals = get_values(light_stats_list, key)
                dark_vals = get_values(dark_stats_list, key)
                total_vals = get_values(total_stats_list, key)

                l_mean, l_sem = mean_sem(light_vals)
                d_mean, d_sem = mean_sem(dark_vals)
                t_mean, t_sem = mean_sem(total_vals)

                light_str = f"{l_mean:.0f}±{l_sem:.0f}" if not np.isnan(l_mean) else "N/A"
                dark_str = f"{d_mean:.0f}±{d_sem:.0f}" if not np.isnan(d_mean) else "N/A"
                total_str = f"{t_mean:.0f}±{t_sem:.0f}" if not np.isnan(t_mean) else "N/A"

            elif key is None and 'Bouts/rec' in label:
                # Compute bouts per recording hour
                # Assume n_days average across experiments
                n_days_list = [exp.get('sleep_analysis', {}).get('n_days', 1) for exp in experiments]
                avg_days = np.mean(n_days_list) if n_days_list else 1
                rec_hours = avg_days * 24

                light_bouts = get_values(light_stats_list, 'bout_count')
                dark_bouts = get_values(dark_stats_list, 'bout_count')
                total_bouts = get_values(total_stats_list, 'bout_count')

                l_rate = [b / rec_hours for b in light_bouts if not np.isnan(b)]
                d_rate = [b / rec_hours for b in dark_bouts if not np.isnan(b)]
                t_rate = [b / rec_hours for b in total_bouts if not np.isnan(b)]

                l_mean, l_sem = mean_sem(l_rate)
                d_mean, d_sem = mean_sem(d_rate)
                t_mean, t_sem = mean_sem(t_rate)

                light_str = f"{l_mean:.2f}±{l_sem:.2f}" if not np.isnan(l_mean) else "N/A"
                dark_str = f"{d_mean:.2f}±{d_sem:.2f}" if not np.isnan(d_mean) else "N/A"
                total_str = f"{t_mean:.2f}±{t_sem:.2f}" if not np.isnan(t_mean) else "N/A"

            elif key is None and 'Frag' in label:
                # Compute fragmentation index = bouts / sleep hours
                frag_light = []
                frag_dark = []
                frag_total = []

                for i in range(len(light_stats_list)):
                    l_stats = light_stats_list[i] if i < len(light_stats_list) else {}
                    d_stats = dark_stats_list[i] if i < len(dark_stats_list) else {}
                    t_stats = total_stats_list[i] if i < len(total_stats_list) else {}

                    l_sleep_hrs = l_stats.get('total_minutes', 0) / 60 if l_stats else 0
                    d_sleep_hrs = d_stats.get('total_minutes', 0) / 60 if d_stats else 0
                    t_sleep_hrs = t_stats.get('total_minutes', 0) / 60 if t_stats else 0

                    if l_sleep_hrs > 0:
                        frag_light.append(l_stats.get('bout_count', 0) / l_sleep_hrs)
                    if d_sleep_hrs > 0:
                        frag_dark.append(d_stats.get('bout_count', 0) / d_sleep_hrs)
                    if t_sleep_hrs > 0:
                        frag_total.append(t_stats.get('bout_count', 0) / t_sleep_hrs)

                l_mean, l_sem = mean_sem(frag_light)
                d_mean, d_sem = mean_sem(frag_dark)
                t_mean, t_sem = mean_sem(frag_total)

                light_str = f"{l_mean:.2f}±{l_sem:.2f}" if not np.isnan(l_mean) else "N/A"
                dark_str = f"{d_mean:.2f}±{d_sem:.2f}" if not np.isnan(d_mean) else "N/A"
                total_str = f"{t_mean:.2f}±{t_sem:.2f}" if not np.isnan(t_mean) else "N/A"

            else:
                # Standard stats
                light_vals = get_values(light_stats_list, key)
                dark_vals = get_values(dark_stats_list, key)
                total_vals = get_values(total_stats_list, key)

                l_mean, l_sem = mean_sem(light_vals)
                d_mean, d_sem = mean_sem(dark_vals)
                t_mean, t_sem = mean_sem(total_vals)

                light_str = f"{l_mean:.1f}±{l_sem:.1f}" if not np.isnan(l_mean) else "N/A"
                dark_str = f"{d_mean:.1f}±{d_sem:.1f}" if not np.isnan(d_mean) else "N/A"
                total_str = f"{t_mean:.1f}±{t_sem:.1f}" if not np.isnan(t_mean) else "N/A"

            row = f"{label:<18} {light_str:>14} {dark_str:>14} {total_str:>14}"
            weight = 'bold' if is_bold else 'normal'
            ax.text(0.02, y_pos, row, fontsize=5.5, fontweight=weight,
                    transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')
            y_pos -= 0.032

        # Separator
        y_pos -= 0.02
        ax.text(0.02, y_pos, "-" * 62, fontsize=6, transform=ax.transAxes,
                color=self.GRID_COLOR, family='monospace', va='top')
        y_pos -= 0.035

        # Definitions section
        ax.text(0.02, y_pos, "Definitions:", fontsize=6, fontweight='bold',
                transform=ax.transAxes, color=self.TEXT_COLOR, va='top')
        y_pos -= 0.032

        definitions = [
            "Frag Index = Bouts ÷ sleep hours (higher = more fragmented)",
            "Bouts/rec hour = Bouts ÷ recording hours",
            "Values shown as Mean ± SEM across animals",
        ]

        for defn in definitions:
            ax.text(0.02, y_pos, f"• {defn}", fontsize=5, transform=ax.transAxes,
                    color='#aaaaaa', va='top')
            y_pos -= 0.028

        # Per-day summary at bottom
        y_pos -= 0.03
        ax.text(0.02, y_pos, "Per-Day Summary (avg across animals):", fontsize=6, fontweight='bold',
                transform=ax.transAxes, color=self.TEXT_COLOR, va='top')
        y_pos -= 0.028

        # Get max days
        max_days = max([exp.get('sleep_analysis', {}).get('n_days', 0) for exp in experiments], default=0)

        for day_idx in range(min(max_days, 7)):
            # Collect per-day stats from all animals
            day_bouts = []
            day_sleep = []

            for exp in experiments:
                sleep_data = exp.get('sleep_analysis', {})
                per_day = sleep_data.get('per_day_stats', [])
                if day_idx < len(per_day):
                    day_bouts.append(per_day[day_idx].get('total_bouts', 0))
                    day_sleep.append(per_day[day_idx].get('total_sleep_minutes', 0))

            if day_bouts and day_sleep:
                b_mean, b_sem = mean_sem(day_bouts)
                s_mean, s_sem = mean_sem(day_sleep)

                day_text = f"D{day_idx + 1}: {b_mean:.0f}±{b_sem:.0f} bouts, {s_mean:.0f}±{s_sem:.0f} min sleep"
                ax.text(0.02, y_pos, day_text, fontsize=5, transform=ax.transAxes,
                        color='#cccccc', va='top', family='monospace')
                y_pos -= 0.022

    def _draw_sleep_stats_compact(self, ax, experiments: List[Dict[str, Any]]):
        """Draw full sleep statistics table with mean±SEM across animals."""
        n_animals = len(experiments)

        # Collect stats from all experiments
        light_stats_list = []
        dark_stats_list = []
        total_stats_list = []
        n_days_list = []

        for exp in experiments:
            sleep_data = exp.get('sleep_analysis', {})
            if sleep_data:
                light_stats_list.append(sleep_data.get('light_stats', {}))
                dark_stats_list.append(sleep_data.get('dark_stats', {}))
                total_stats_list.append(sleep_data.get('total_stats', {}))
                n_days_list.append(sleep_data.get('n_days', 1))

        avg_n_days = np.mean(n_days_list) if n_days_list else 1

        def mean_sem(values):
            values = [v for v in values if v is not None and not np.isnan(v)]
            if not values:
                return np.nan, np.nan
            mean = np.mean(values)
            sem = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            return mean, sem

        def get_values(stats_list, key):
            return [s.get(key, np.nan) for s in stats_list if s]

        # Compute derived metrics per animal then average
        bouts_per_hr_light = []
        bouts_per_hr_dark = []
        bouts_per_hr_total = []
        frag_light = []
        frag_dark = []
        frag_total = []

        for i in range(len(light_stats_list)):
            n_days = n_days_list[i] if i < len(n_days_list) else 1
            rec_hrs_phase = n_days * 12
            rec_hrs_total = n_days * 24

            l_bouts = light_stats_list[i].get('bout_count', 0)
            d_bouts = dark_stats_list[i].get('bout_count', 0)
            t_bouts = total_stats_list[i].get('bout_count', 0)

            l_sleep_hrs = light_stats_list[i].get('total_minutes', 0) / 60
            d_sleep_hrs = dark_stats_list[i].get('total_minutes', 0) / 60
            t_sleep_hrs = total_stats_list[i].get('total_minutes', 0) / 60

            bouts_per_hr_light.append(l_bouts / rec_hrs_phase if rec_hrs_phase > 0 else 0)
            bouts_per_hr_dark.append(d_bouts / rec_hrs_phase if rec_hrs_phase > 0 else 0)
            bouts_per_hr_total.append(t_bouts / rec_hrs_total if rec_hrs_total > 0 else 0)

            frag_light.append(l_bouts / l_sleep_hrs if l_sleep_hrs > 0 else 0)
            frag_dark.append(d_bouts / d_sleep_hrs if d_sleep_hrs > 0 else 0)
            frag_total.append(t_bouts / t_sleep_hrs if t_sleep_hrs > 0 else 0)

        # Title
        ax.text(0.5, 0.99, f"Sleep Bout Statistics (n={n_animals})", fontsize=9, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, va='top', ha='center')

        # Subtitle
        ax.text(0.5, 0.93, f"Mean ± SEM across animals (avg {avg_n_days:.1f} days)",
               fontsize=5.5, transform=ax.transAxes, color='#888888', va='top', ha='center')

        # Header
        y_pos = 0.87
        header = f"{'Metric':<14} {'Light':>10} {'Dark':>10} {'Total':>10}"
        ax.text(0.02, y_pos, header, fontsize=5, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')
        y_pos -= 0.04
        ax.text(0.02, y_pos, "-" * 48, fontsize=5,
               transform=ax.transAxes, color=self.GRID_COLOR, family='monospace', va='top')
        y_pos -= 0.04

        # Full statistics - (label, light_vals, dark_vals, total_vals, format, is_bold)
        stats_rows = [
            ('Sleep (min)', get_values(light_stats_list, 'total_minutes'),
             get_values(dark_stats_list, 'total_minutes'),
             get_values(total_stats_list, 'total_minutes'), '.0f', True),
            ('Sleep %', get_values(light_stats_list, 'percent_time'),
             get_values(dark_stats_list, 'percent_time'),
             get_values(total_stats_list, 'percent_time'), '.1f', True),
            ('Bouts', get_values(light_stats_list, 'bout_count'),
             get_values(dark_stats_list, 'bout_count'),
             get_values(total_stats_list, 'bout_count'), '.0f', False),
            ('Bouts/rec hr', bouts_per_hr_light, bouts_per_hr_dark, bouts_per_hr_total, '.2f', True),
            ('Frag Index', frag_light, frag_dark, frag_total, '.2f', True),
            ('Mean Bout', get_values(light_stats_list, 'mean_duration'),
             get_values(dark_stats_list, 'mean_duration'),
             get_values(total_stats_list, 'mean_duration'), '.1f', False),
            ('Median Bout', get_values(light_stats_list, 'median_duration'),
             get_values(dark_stats_list, 'median_duration'),
             get_values(total_stats_list, 'median_duration'), '.1f', False),
            ('Max Bout', get_values(light_stats_list, 'max_duration'),
             get_values(dark_stats_list, 'max_duration'),
             get_values(total_stats_list, 'max_duration'), '.1f', False),
        ]

        for label, light_vals, dark_vals, total_vals, fmt, is_bold in stats_rows:
            l_mean, l_sem = mean_sem(light_vals)
            d_mean, d_sem = mean_sem(dark_vals)
            t_mean, t_sem = mean_sem(total_vals)

            l_str = f"{l_mean:{fmt}}±{l_sem:{fmt}}" if not np.isnan(l_mean) else "N/A"
            d_str = f"{d_mean:{fmt}}±{d_sem:{fmt}}" if not np.isnan(d_mean) else "N/A"
            t_str = f"{t_mean:{fmt}}±{t_sem:{fmt}}" if not np.isnan(t_mean) else "N/A"

            row = f"{label:<14} {l_str:>10} {d_str:>10} {t_str:>10}"
            weight = 'bold' if is_bold else 'normal'
            ax.text(0.02, y_pos, row, fontsize=4.5,
                   transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace',
                   fontweight=weight, va='top')
            y_pos -= 0.042

        # Definitions
        y_pos -= 0.015
        ax.text(0.02, y_pos, "Definitions:", fontsize=5, fontweight='bold',
               transform=ax.transAxes, color='#3daee9', va='top')
        y_pos -= 0.035

        definitions = [
            "Bouts/rec hr = Bouts ÷ recording hours",
            "Frag Index = Bouts ÷ sleep hrs (higher=fragmented)",
        ]

        for defn in definitions:
            ax.text(0.02, y_pos, f"• {defn}", fontsize=4,
                   transform=ax.transAxes, color='#888888', va='top')
            y_pos -= 0.03

    def _draw_sleep_quality_bars(self, ax, experiments: List[Dict[str, Any]]):
        """Draw 2x5 grid of bar charts for all sleep quality metrics with SEM."""
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        def mean_sem(values):
            values = [v for v in values if v is not None and not np.isnan(v) and v != float('inf')]
            if not values:
                return 0, 0
            mean = np.mean(values)
            sem = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            return mean, sem

        def get_stat_values(key, stats_type='light'):
            vals = []
            for exp in experiments:
                sleep_data = exp.get('sleep_analysis', {})
                stats = sleep_data.get(f'{stats_type}_stats', {})
                vals.append(stats.get(key, np.nan))
            return vals

        # Collect all metrics per experiment
        sleep_min_l, sleep_min_d, sleep_min_t = [], [], []
        sleep_pct_l, sleep_pct_d = [], []
        bouts_l, bouts_d, bouts_t = [], [], []
        bouts_hr_l, bouts_hr_d, bouts_hr_t = [], [], []
        frag_l, frag_d, frag_t = [], [], []
        mean_bout_l, mean_bout_d, mean_bout_t = [], [], []
        med_bout_l, med_bout_d, med_bout_t = [], [], []
        max_bout_l, max_bout_d, max_bout_t = [], [], []
        long_pct_l, long_pct_d = [], []
        ld_ratios, trans_rates = [], []

        for exp in experiments:
            sleep_data = exp.get('sleep_analysis', {})
            ls = sleep_data.get('light_stats', {})
            ds = sleep_data.get('dark_stats', {})
            ts = sleep_data.get('total_stats', {})
            qm = sleep_data.get('quality_metrics', {})
            n_days = sleep_data.get('n_days', 1)

            # Basic stats
            sleep_min_l.append(ls.get('total_minutes', np.nan))
            sleep_min_d.append(ds.get('total_minutes', np.nan))
            sleep_min_t.append(ts.get('total_minutes', np.nan))

            sleep_pct_l.append(ls.get('percent_time', np.nan))
            sleep_pct_d.append(ds.get('percent_time', np.nan))

            l_bouts = ls.get('bout_count', 0)
            d_bouts = ds.get('bout_count', 0)
            t_bouts = ts.get('bout_count', 0)
            bouts_l.append(l_bouts)
            bouts_d.append(d_bouts)
            bouts_t.append(t_bouts)

            # Derived metrics
            rec_hrs_phase = n_days * 12
            rec_hrs_total = n_days * 24
            bouts_hr_l.append(l_bouts / rec_hrs_phase if rec_hrs_phase > 0 else 0)
            bouts_hr_d.append(d_bouts / rec_hrs_phase if rec_hrs_phase > 0 else 0)
            bouts_hr_t.append(t_bouts / rec_hrs_total if rec_hrs_total > 0 else 0)

            l_sleep_hrs = ls.get('total_minutes', 0) / 60
            d_sleep_hrs = ds.get('total_minutes', 0) / 60
            t_sleep_hrs = ts.get('total_minutes', 0) / 60
            frag_l.append(l_bouts / l_sleep_hrs if l_sleep_hrs > 0 else 0)
            frag_d.append(d_bouts / d_sleep_hrs if d_sleep_hrs > 0 else 0)
            frag_t.append(t_bouts / t_sleep_hrs if t_sleep_hrs > 0 else 0)

            mean_bout_l.append(ls.get('mean_duration', np.nan))
            mean_bout_d.append(ds.get('mean_duration', np.nan))
            mean_bout_t.append(ts.get('mean_duration', np.nan))

            med_bout_l.append(ls.get('median_duration', np.nan))
            med_bout_d.append(ds.get('median_duration', np.nan))
            med_bout_t.append(ts.get('median_duration', np.nan))

            max_bout_l.append(ls.get('max_duration', np.nan))
            max_bout_d.append(ds.get('max_duration', np.nan))
            max_bout_t.append(ts.get('max_duration', np.nan))

            long_pct_l.append(qm.get('long_bout_pct_light', np.nan))
            long_pct_d.append(qm.get('long_bout_pct_dark', np.nan))
            ld_ratios.append(qm.get('light_dark_ratio', np.nan))
            trans_rates.append(qm.get('transition_rate', np.nan))

        # Create 2x5 subgrid
        gs_inner = GridSpecFromSubplotSpec(2, 5, subplot_spec=ax.get_subplotspec(),
                                           wspace=0.35, hspace=0.45)
        ax.axis('off')
        fig = ax.figure

        # Row A: Sleep min, Sleep %, Bout Count, Bouts/hr, Frag Index
        ax1 = fig.add_subplot(gs_inner[0, 0])
        ax1.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar_sem(ax1, mean_sem(sleep_min_l), mean_sem(sleep_min_d),
                               mean_sem(sleep_min_t), 'Sleep (min)', fmt='.0f')

        ax2 = fig.add_subplot(gs_inner[0, 1])
        ax2.set_facecolor(self.BG_COLOR)
        self._draw_ld_bar_sem(ax2, mean_sem(sleep_pct_l), mean_sem(sleep_pct_d), 'Sleep %')

        ax3 = fig.add_subplot(gs_inner[0, 2])
        ax3.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar_sem(ax3, mean_sem(bouts_l), mean_sem(bouts_d),
                               mean_sem(bouts_t), 'Bouts', fmt='.0f')

        ax4 = fig.add_subplot(gs_inner[0, 3])
        ax4.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar_sem(ax4, mean_sem(bouts_hr_l), mean_sem(bouts_hr_d),
                               mean_sem(bouts_hr_t), 'Bouts/hr', fmt='.2f')

        ax5 = fig.add_subplot(gs_inner[0, 4])
        ax5.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar_sem(ax5, mean_sem(frag_l), mean_sem(frag_d),
                               mean_sem(frag_t), 'Frag Idx', fmt='.2f')

        # Row B: Mean Bout, Median Bout, Max Bout, % Long, L/D + Trans
        ax6 = fig.add_subplot(gs_inner[1, 0])
        ax6.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar_sem(ax6, mean_sem(mean_bout_l), mean_sem(mean_bout_d),
                               mean_sem(mean_bout_t), 'Mean Bout', fmt='.1f')

        ax7 = fig.add_subplot(gs_inner[1, 1])
        ax7.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar_sem(ax7, mean_sem(med_bout_l), mean_sem(med_bout_d),
                               mean_sem(med_bout_t), 'Med Bout', fmt='.1f')

        ax8 = fig.add_subplot(gs_inner[1, 2])
        ax8.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar_sem(ax8, mean_sem(max_bout_l), mean_sem(max_bout_d),
                               mean_sem(max_bout_t), 'Max Bout', fmt='.0f')

        ax9 = fig.add_subplot(gs_inner[1, 3])
        ax9.set_facecolor(self.BG_COLOR)
        self._draw_ld_bar_sem(ax9, mean_sem(long_pct_l), mean_sem(long_pct_d), '% Long')

        ax10 = fig.add_subplot(gs_inner[1, 4])
        ax10.set_facecolor(self.BG_COLOR)
        self._draw_dual_bar_sem(ax10, mean_sem(ld_ratios), mean_sem(trans_rates),
                                'L/D', 'Trans')

    def _draw_ldt_bar_sem(self, ax, l_stats, d_stats, t_stats, title: str, fmt: str = '.1f'):
        """Draw Light/Dark/Total bar chart with SEM error bars."""
        l_mean, l_sem = l_stats
        d_mean, d_sem = d_stats
        t_mean, t_sem = t_stats

        x = [0, 1, 2]
        vals = [l_mean, d_mean, t_mean]
        errs = [l_sem, d_sem, t_sem]
        colors = ['#f0c040', '#4a4a4a', '#3daee9']

        bars = ax.bar(x, vals, yerr=errs, color=colors, edgecolor='white',
                     linewidth=0.3, width=0.7, capsize=2,
                     error_kw={'ecolor': 'white', 'capthick': 0.5, 'elinewidth': 0.5})

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['L', 'D', 'T'], fontsize=5, color=self.TEXT_COLOR)
        ax.set_title(title, fontsize=6, color=self.TEXT_COLOR, fontweight='bold', pad=2)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)

    def _draw_ld_bar_sem(self, ax, l_stats, d_stats, title: str):
        """Draw Light/Dark bar chart with SEM error bars."""
        l_mean, l_sem = l_stats
        d_mean, d_sem = d_stats

        x = [0, 1]
        vals = [l_mean, d_mean]
        errs = [l_sem, d_sem]
        colors = ['#f0c040', '#4a4a4a']

        bars = ax.bar(x, vals, yerr=errs, color=colors, edgecolor='white',
                     linewidth=0.3, width=0.6, capsize=2,
                     error_kw={'ecolor': 'white', 'capthick': 0.5, 'elinewidth': 0.5})

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['L', 'D'], fontsize=5, color=self.TEXT_COLOR)
        ax.set_title(title, fontsize=6, color=self.TEXT_COLOR, fontweight='bold', pad=2)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)

    def _draw_dual_bar_sem(self, ax, stats1, stats2, label1: str, label2: str):
        """Draw two metrics side by side with SEM error bars."""
        mean1, sem1 = stats1
        mean2, sem2 = stats2

        x = [0, 1]
        vals = [mean1, mean2]
        errs = [sem1, sem2]
        colors = ['#3daee9', '#e74c3c']

        bars = ax.bar(x, vals, yerr=errs, color=colors, edgecolor='white',
                     linewidth=0.3, width=0.6, capsize=2,
                     error_kw={'ecolor': 'white', 'capthick': 0.5, 'elinewidth': 0.5})

        ax.set_xticks([0, 1])
        ax.set_xticklabels([label1, label2], fontsize=5, color=self.TEXT_COLOR)
        ax.set_title('Quality', fontsize=6, color=self.TEXT_COLOR, fontweight='bold', pad=2)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)
