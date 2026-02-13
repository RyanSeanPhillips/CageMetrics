"""
Comparison figure generator for comparing multiple consolidated datasets.

Creates matplotlib figures for comparing consolidated NPZ files:
- Summary page with datasets being compared
- Overlay CTA plots showing grand means from each dataset with SEM bands (36hr view)
- Bar chart comparisons for dark/light means
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from typing import Dict, Any, List, Tuple, Optional
import json


class ComparisonFigureGenerator:
    """Generate matplotlib figures for comparing consolidated datasets."""

    # Dark theme colors
    DARK_THEME = {
        'bg': '#2d2d2d',
        'text': '#ffffff',
        'grid': '#4d4d4d',
        'dark_phase': '#2d2d2d',
        'light_phase': '#4a4a2a',
    }

    # Light theme colors (for publications)
    LIGHT_THEME = {
        'bg': '#ffffff',
        'text': '#000000',
        'grid': '#cccccc',
        'dark_phase': '#e0e0e0',
        'light_phase': '#fffde7',
    }

    # Color palette for different datasets (up to 8)
    DATASET_COLORS = [
        '#3daee9',  # Blue
        '#e74c3c',  # Red
        '#2ecc71',  # Green
        '#f39c12',  # Orange
        '#9b59b6',  # Purple
        '#1abc9c',  # Teal
        '#e91e63',  # Pink
        '#00bcd4',  # Cyan
    ]

    SEM_ALPHA = 0.25
    MINUTES_PER_DAY = 1440

    # Marker shapes for scatter points (one per dataset, up to 8)
    DATASET_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    def __init__(self, smoothing_window: int = 5, bar_grouping: str = 'dataset',
                 light_mode: bool = False, dataset_colors: List[str] = None,
                 show_statistics: bool = True):
        """
        Initialize the comparison figure generator.

        Args:
            smoothing_window: Rolling average window size in minutes (default 5)
            bar_grouping: 'dataset' or 'phase' - how to group bar charts
            light_mode: If True, use light background for figures
            dataset_colors: Optional list of colors for each dataset
            show_statistics: If True, add statistical annotations to bar charts
        """
        self.smoothing_window = smoothing_window
        self.bar_grouping = bar_grouping
        self.light_mode = light_mode
        self.show_statistics = show_statistics

        # Store all statistics results for export
        self.statistics_results = []

        # Use custom colors if provided, otherwise use defaults
        if dataset_colors:
            self.DATASET_COLORS = dataset_colors
        # Keep default DATASET_COLORS if not overridden

        # Set theme colors
        theme = self.LIGHT_THEME if light_mode else self.DARK_THEME
        self.BG_COLOR = theme['bg']
        self.TEXT_COLOR = theme['text']
        self.GRID_COLOR = theme['grid']
        self.DARK_PHASE_COLOR = theme['dark_phase']
        self.LIGHT_PHASE_COLOR = theme['light_phase']

    def _smooth_data(self, data: np.ndarray) -> np.ndarray:
        """Apply rolling average smoothing to data."""
        if self.smoothing_window <= 1 or len(data) == 0:
            return data

        # Use convolution for rolling average
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        # Pad to handle edges
        padded = np.pad(data, (self.smoothing_window // 2, self.smoothing_window // 2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')

        # Ensure output length matches input
        if len(smoothed) > len(data):
            smoothed = smoothed[:len(data)]
        elif len(smoothed) < len(data):
            smoothed = np.pad(smoothed, (0, len(data) - len(smoothed)), mode='edge')

        return smoothed

    def generate_all_pages(self, datasets: List[Dict[str, Any]],
                           progress_callback=None) -> List[Tuple[str, Figure]]:
        """
        Generate all comparison figure pages (all visualization modes combined).

        Produces: Summary, Age Coverage (if age data), per-metric grand CTA +
        actogram + age trend pages, bar charts, sleep analysis, statistics.

        Page titles encode metric names so the display layer can group into tabs.

        Args:
            datasets: List of loaded consolidated NPZ data dicts
            progress_callback: Optional callable(str) for progress updates

        Returns:
            List of (title, figure) tuples
        """
        def _progress(msg):
            if progress_callback:
                progress_callback(msg)

        # Clear statistics from previous runs
        self.statistics_results = []

        pages = []

        if not datasets:
            return pages

        # Page 1: Summary comparing all datasets
        _progress("Creating summary...")
        fig_summary = self.create_summary_page(datasets)
        pages.append(("Comparison Summary", fig_summary))

        # Get common metrics across all datasets
        common_metrics = self._get_common_metrics(datasets)
        n_metrics = len(common_metrics)

        # Age coverage overview (if age data exists)
        has_age = self._any_dataset_has_age_data(datasets)
        if has_age:
            _progress("Creating age coverage overview...")
            coverage_fig = self._create_age_coverage_comparison_page(datasets)
            if coverage_fig is not None:
                pages.append(("Age Coverage Overview", coverage_fig))

        # Per-metric pages: Grand CTA + Actogram + Age Trends
        for i, metric_name in enumerate(common_metrics):
            _progress(f"Metric {i+1}/{n_metrics}: {metric_name} (CTA)...")
            fig_cta = self._create_single_metric_cta_page(datasets, metric_name)
            pages.append((f"CTA: {metric_name}", fig_cta))

            _progress(f"Metric {i+1}/{n_metrics}: {metric_name} (actogram)...")
            actogram_pages = self._generate_actogram_pages(datasets, metric_name)
            pages.extend(actogram_pages)

            if has_age:
                _progress(f"Metric {i+1}/{n_metrics}: {metric_name} (age trends)...")
                age_pages = self._generate_age_trend_pages(datasets, metric_name)
                pages.extend(age_pages)

        # Dark/Light bar chart pages (all metrics, from CTA+Bars mode)
        _progress("Creating bar charts...")
        bars_per_page = 12
        for page_idx in range(0, len(common_metrics), bars_per_page):
            page_metrics = common_metrics[page_idx:page_idx + bars_per_page]
            page_num = page_idx // bars_per_page + 1
            total_bar_pages = (len(common_metrics) + bars_per_page - 1) // bars_per_page
            fig_bars = self.create_bar_comparison_page(datasets, page_metrics, page_num, total_bar_pages)
            if total_bar_pages > 1:
                pages.append((f"Dark/Light Comparison {page_num}", fig_bars))
            else:
                pages.append(("Dark/Light Comparison", fig_bars))

        # Sleep analysis comparison page
        _progress("Creating sleep analysis...")
        fig_sleep = self.create_sleep_comparison_page(datasets)
        if fig_sleep is not None:
            pages.append(("Sleep Analysis Comparison", fig_sleep))

        # Statistics summary page
        if self.show_statistics and len(datasets) >= 2 and self.statistics_results:
            _progress("Creating statistics summary...")
            fig_stats = self.create_statistics_summary_page(datasets)
            pages.append(("Statistical Analysis", fig_stats))

        _progress(f"Generated {len(pages)} figures")
        return pages

    def _create_single_metric_cta_page(self, datasets: List[Dict[str, Any]],
                                         metric_name: str) -> Figure:
        """Create a large single-metric CTA comparison page (48hr double-plot L-D-L-D)."""
        fig = Figure(figsize=(11, 5.5), dpi=150, facecolor=self.BG_COLOR)

        fig.suptitle(f"CTA Comparison: {metric_name}",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.97)

        ax = fig.add_axes([0.07, 0.12, 0.88, 0.78])
        ax.set_facecolor(self.BG_COLOR)

        # 48-hour x-axis (L-D-L-D double-plot)
        x_minutes_48h = np.arange(2880)
        x_hours_48h = x_minutes_48h / 60

        # Draw dark/light phase background (48hr: L-D-L-D)
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(36, 48, facecolor=self.DARK_PHASE_COLOR, zorder=0)

        key_base = self._clean_metric_name(metric_name)

        for ds_idx, ds in enumerate(datasets):
            color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)

            grand_cta = ds.get(f'{key_base}_grand_cta')
            grand_sem = ds.get(f'{key_base}_grand_sem')

            if grand_cta is not None and len(grand_cta) == self.MINUTES_PER_DAY:
                smoothed_cta = self._smooth_data(grand_cta)
                # Double to 48 hours (repeat full 24h cycle)
                cta_48h = np.concatenate([smoothed_cta, smoothed_cta])
                ax.plot(x_hours_48h, cta_48h, color=color, linewidth=1.5,
                       label=label, alpha=0.9)

                if grand_sem is not None and len(grand_sem) == self.MINUTES_PER_DAY:
                    smoothed_sem = self._smooth_data(grand_sem)
                    sem_48h = np.concatenate([smoothed_sem, smoothed_sem])
                    ax.fill_between(x_hours_48h,
                                   cta_48h - sem_48h,
                                   cta_48h + sem_48h,
                                   color=color, alpha=self.SEM_ALPHA)

        ax.set_xlim(0, 48)
        ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
        ax.set_xticklabels(['ZT0', 'ZT6', 'ZT12', 'ZT18', 'ZT24',
                            'ZT30', 'ZT36', 'ZT42', 'ZT48'],
                           fontsize=9, color=self.TEXT_COLOR)
        ax.set_ylabel(metric_name, fontsize=10, color=self.TEXT_COLOR)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

        ax.legend(loc='upper right', fontsize=9, framealpha=0.8,
                  facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                  labelcolor=self.TEXT_COLOR)

        return fig

    def _get_common_metrics(self, datasets: List[Dict[str, Any]]) -> List[str]:
        """Get list of metrics common to all datasets."""
        if not datasets:
            return []

        # Start with metrics from first dataset
        common = set(datasets[0].get('metric_names', []))

        # Intersect with other datasets
        for ds in datasets[1:]:
            common &= set(ds.get('metric_names', []))

        return sorted(list(common))

    def _get_dataset_label(self, dataset: Dict[str, Any]) -> str:
        """Generate a short label for a dataset based on its metadata."""
        meta = dataset.get('consolidation_metadata', {})

        # Check for custom display name first (set by user via rename)
        display_name = meta.get('display_name', '')
        if display_name:
            if len(display_name) > 40:
                return display_name[:37] + '...'
            return display_name

        # Try filter description
        filter_desc = meta.get('filter_description', '')
        if filter_desc and filter_desc != 'No filters applied':
            # Shorten if needed
            if len(filter_desc) > 40:
                return filter_desc[:37] + '...'
            return filter_desc

        # Fall back to filename
        filename = dataset.get('filename', 'Dataset')
        if len(filename) > 30:
            return filename[:27] + '...'
        return filename

    def create_summary_page(self, datasets: List[Dict[str, Any]]) -> Figure:
        """Create summary page showing all datasets being compared."""
        fig = Figure(figsize=(11, 8.5), dpi=150, facecolor=self.BG_COLOR)

        n_datasets = len(datasets)
        fig.suptitle(f"Dataset Comparison ({n_datasets} datasets)",
                     fontsize=16, fontweight='bold', color=self.TEXT_COLOR, y=0.96)

        # Single axes for the summary table
        ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
        ax.set_facecolor(self.BG_COLOR)
        ax.axis('off')

        # Build summary table data
        headers = ['#', 'Dataset', 'Animals', 'Metrics', 'Filter Criteria', 'Date']
        rows = []

        for i, ds in enumerate(datasets):
            meta = ds.get('consolidation_metadata', {})
            # Truncate filename to prevent overflow
            filename = ds.get('filename', f'Dataset {i+1}')
            if len(filename) > 25:
                filename = filename[:22] + '...'
            rows.append([
                str(i + 1),
                filename,
                str(meta.get('n_animals', '?')),
                str(meta.get('n_metrics', '?')),
                meta.get('filter_description', 'None')[:40],
                meta.get('consolidation_date', '')[:10]
            ])

        # Draw table - adjusted column widths to prevent overflow
        n_rows = len(rows)
        col_widths = [0.05, 0.22, 0.07, 0.07, 0.42, 0.12]

        # Header row
        y_start = 0.92
        row_height = 0.06
        x_pos = 0.02

        for j, (header, width) in enumerate(zip(headers, col_widths)):
            ax.text(x_pos + width/2, y_start, header,
                   fontsize=10, fontweight='bold', color=self.TEXT_COLOR,
                   ha='center', va='center',
                   transform=ax.transAxes)
            x_pos += width

        # Draw header line using plot (axhline doesn't support transform)
        ax.plot([0.02, 0.98], [y_start - 0.02, y_start - 0.02],
                color=self.GRID_COLOR, linewidth=1, transform=ax.transAxes)

        # Data rows with color indicators
        for i, row in enumerate(rows):
            y_pos = y_start - (i + 1) * row_height
            x_pos = 0.02

            # Color indicator
            color = self.DATASET_COLORS[i % len(self.DATASET_COLORS)]
            ax.add_patch(Rectangle((0.01, y_pos - row_height/2 + 0.01),
                                   0.015, row_height - 0.02,
                                   facecolor=color, transform=ax.transAxes))

            for j, (value, width) in enumerate(zip(row, col_widths)):
                ax.text(x_pos + width/2, y_pos, value,
                       fontsize=9, color=self.TEXT_COLOR,
                       ha='center', va='center',
                       transform=ax.transAxes)
                x_pos += width

        # Add legend explaining colors
        y_legend = y_start - (n_rows + 2) * row_height
        ax.text(0.02, y_legend, "Legend:", fontsize=10, fontweight='bold',
               color=self.TEXT_COLOR, transform=ax.transAxes)

        for i, ds in enumerate(datasets):
            color = self.DATASET_COLORS[i % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)
            x_leg = 0.02 + (i % 4) * 0.24
            y_leg = y_legend - 0.05 - (i // 4) * 0.05

            ax.add_patch(Rectangle((x_leg, y_leg - 0.01), 0.02, 0.025,
                                   facecolor=color, transform=ax.transAxes))
            ax.text(x_leg + 0.03, y_leg, label, fontsize=8, color=self.TEXT_COLOR,
                   va='center', transform=ax.transAxes)

        return fig

    def create_cta_comparison_page(self, datasets: List[Dict[str, Any]],
                                    metrics: List[str], page_num: int) -> Figure:
        """Create page with overlaid CTA traces for multiple datasets (36hr view, 2x3 grid)."""
        fig = Figure(figsize=(11, 8.5), dpi=150, facecolor=self.BG_COLOR)

        fig.suptitle(f"CTA Comparison (Page {page_num}) - Shading = SEM",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

        # Create 2x3 grid for up to 6 metrics per page
        n_cols = 3
        n_rows = 2
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.25,
                      left=0.06, right=0.94, top=0.92, bottom=0.08)

        # 36-hour x-axis
        x_minutes_36h = np.arange(2160)
        x_hours_36h = x_minutes_36h / 60

        for idx, metric_name in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(self.BG_COLOR)

            # Draw dark/light phase background (36hr)
            ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
            ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
            ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)

            # Clean metric name for array key
            key_base = self._clean_metric_name(metric_name)

            # Plot each dataset
            for ds_idx, ds in enumerate(datasets):
                color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
                label = self._get_dataset_label(ds)

                # Get grand CTA and SEM
                grand_cta = ds.get(f'{key_base}_grand_cta')
                grand_sem = ds.get(f'{key_base}_grand_sem')

                if grand_cta is not None and len(grand_cta) == self.MINUTES_PER_DAY:
                    # Apply smoothing
                    smoothed_cta = self._smooth_data(grand_cta)

                    # Extend to 36 hours (add first 12 hours)
                    cta_36h = np.concatenate([smoothed_cta, smoothed_cta[:720]])

                    # Plot mean line
                    ax.plot(x_hours_36h, cta_36h, color=color, linewidth=1.2,
                           label=label, alpha=0.9)

                    # Plot SEM band
                    if grand_sem is not None and len(grand_sem) == self.MINUTES_PER_DAY:
                        smoothed_sem = self._smooth_data(grand_sem)
                        sem_36h = np.concatenate([smoothed_sem, smoothed_sem[:720]])
                        ax.fill_between(x_hours_36h,
                                       cta_36h - sem_36h,
                                       cta_36h + sem_36h,
                                       color=color, alpha=self.SEM_ALPHA)

            # Styling
            ax.set_title(metric_name, fontsize=9, fontweight='bold',
                        color=self.TEXT_COLOR, pad=3)
            ax.set_xlim(0, 36)
            ax.set_xticks([0, 12, 24, 36])
            ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36'], fontsize=7)
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

            for spine in ax.spines.values():
                spine.set_color(self.GRID_COLOR)

            # Legend on first metric of each page (draggable)
            if idx == 0:
                legend = ax.legend(loc='upper right', fontsize=6, framealpha=0.8,
                                   facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                                   labelcolor=self.TEXT_COLOR)
                legend.set_draggable(True)

        return fig

    def create_bar_comparison_page(self, datasets: List[Dict[str, Any]],
                                    metrics: List[str], page_num: int = 1,
                                    total_pages: int = 1) -> Figure:
        """Create page with bar chart comparisons of dark/light means."""
        fig = Figure(figsize=(11, 8.5), dpi=150, facecolor=self.BG_COLOR)

        grouping_text = "by Dataset" if self.bar_grouping == 'dataset' else "by Light/Dark"
        if total_pages > 1:
            fig.suptitle(f"Dark vs Light Phase Comparison ({grouping_text}, Page {page_num}/{total_pages})",
                         fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.97)
        else:
            fig.suptitle(f"Dark vs Light Phase Comparison ({grouping_text})",
                         fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.97)

        # 3x4 grid for up to 12 metrics per page
        n_cols = 4
        n_rows = 3
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3,
                      left=0.06, right=0.94, top=0.90, bottom=0.08)

        n_datasets = len(datasets)

        for idx, metric_name in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(self.BG_COLOR)

            key_base = self._clean_metric_name(metric_name)

            dark_means = []
            dark_sems = []
            light_means = []
            light_sems = []
            # Store raw data for statistics
            dark_raw_data = []
            light_raw_data = []

            for ds in datasets:
                dark_arr = ds.get(f'{key_base}_dark_means', np.array([]))
                light_arr = ds.get(f'{key_base}_light_means', np.array([]))

                # Store raw arrays for statistics
                dark_raw_data.append(dark_arr if len(dark_arr) > 0 else np.array([]))
                light_raw_data.append(light_arr if len(light_arr) > 0 else np.array([]))

                if len(dark_arr) > 0:
                    valid_dark = dark_arr[~np.isnan(dark_arr)]
                    dark_means.append(np.mean(valid_dark) if len(valid_dark) > 0 else 0)
                    dark_sems.append(np.std(valid_dark) / np.sqrt(len(valid_dark)) if len(valid_dark) > 1 else 0)
                else:
                    dark_means.append(0)
                    dark_sems.append(0)

                if len(light_arr) > 0:
                    valid_light = light_arr[~np.isnan(light_arr)]
                    light_means.append(np.mean(valid_light) if len(valid_light) > 0 else 0)
                    light_sems.append(np.std(valid_light) / np.sqrt(len(valid_light)) if len(valid_light) > 1 else 0)
                else:
                    light_means.append(0)
                    light_sems.append(0)

            if self.bar_grouping == 'dataset':
                # Group by dataset: each dataset has dark/light side-by-side
                bar_width = 0.35
                x_positions = np.arange(n_datasets)

                # Use light-colored error bars in dark mode for visibility
                error_color = '#000000' if self.light_mode else '#ffffff'
                error_kw = {'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': error_color}

                bars_dark = ax.bar(x_positions - bar_width/2, dark_means, bar_width,
                                  yerr=dark_sems, capsize=3,
                                  color='#555555', label='Dark', alpha=0.8,
                                  error_kw=error_kw)
                bars_light = ax.bar(x_positions + bar_width/2, light_means, bar_width,
                                   yerr=light_sems, capsize=3,
                                   color='#ffcc00', label='Light', alpha=0.8,
                                   error_kw=error_kw)

                # Color-code the bars by dataset
                for i, (bd, bl) in enumerate(zip(bars_dark, bars_light)):
                    color = self.DATASET_COLORS[i % len(self.DATASET_COLORS)]
                    bd.set_edgecolor(color)
                    bd.set_linewidth(2)
                    bl.set_edgecolor(color)
                    bl.set_linewidth(2)

                ax.set_xticks(x_positions)
                # Use shortened dataset labels for x-axis
                xlabels = []
                for ds in datasets:
                    label = self._get_dataset_label(ds)
                    # Shorten further for x-axis (max ~10 chars)
                    if len(label) > 12:
                        label = label[:10] + '..'
                    xlabels.append(label)
                ax.set_xticklabels(xlabels, fontsize=6, rotation=15, ha='right')

                # Add statistics if enabled (compare dark values across datasets)
                if self.show_statistics and n_datasets >= 2:
                    dark_stats = self._compute_statistics(dark_raw_data)
                    light_stats = self._compute_statistics(light_raw_data)

                    # Store statistics results
                    self._store_statistics_result(metric_name, datasets, dark_stats, light_stats,
                                                  dark_means, dark_sems, light_means, light_sems)

                    # Get max bar height for positioning brackets
                    max_dark = max(dark_means[i] + dark_sems[i] for i in range(n_datasets)) if dark_means else 0
                    max_light = max(light_means[i] + light_sems[i] for i in range(n_datasets)) if light_means else 0
                    max_height = max(max_dark, max_light)

                    # Add brackets for significant comparisons (dark phase)
                    bracket_y = max_height * 1.05
                    bracket_level = 0
                    for (i, j), p_value in sorted(dark_stats['pairwise'].items()):
                        symbol = self._get_significance_symbol(p_value)
                        if symbol and symbol != 'ns':
                            y_pos = bracket_y + bracket_level * max_height * 0.12
                            self._add_significance_bracket(
                                ax, x_positions[i] - bar_width/2, x_positions[j] - bar_width/2,
                                y_pos, symbol
                            )
                            bracket_level += 1

            else:
                # Group by phase: Dark group then Light group
                bar_width = 0.8 / n_datasets
                x_dark = np.arange(n_datasets) * bar_width
                x_light = np.arange(n_datasets) * bar_width + n_datasets * bar_width + 0.3

                # Use light-colored error bars in dark mode for visibility
                error_color = '#000000' if self.light_mode else '#ffffff'
                error_kw = {'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': error_color}

                for i, ds in enumerate(datasets):
                    color = self.DATASET_COLORS[i % len(self.DATASET_COLORS)]
                    # Use actual dataset label instead of "Dataset N"
                    label = self._get_dataset_label(ds) if idx == 0 else None
                    ax.bar(x_dark[i], dark_means[i], bar_width * 0.9,
                          yerr=dark_sems[i], capsize=3,
                          color=color, alpha=0.8, label=label,
                          error_kw=error_kw)
                    ax.bar(x_light[i], light_means[i], bar_width * 0.9,
                          yerr=light_sems[i], capsize=3,
                          color=color, alpha=0.8,
                          error_kw=error_kw)

                # X-axis labels for phases
                dark_center = (n_datasets - 1) * bar_width / 2
                light_center = n_datasets * bar_width + 0.3 + (n_datasets - 1) * bar_width / 2
                ax.set_xticks([dark_center, light_center])
                ax.set_xticklabels(['Dark', 'Light'], fontsize=8)

                # Add statistics if enabled
                if self.show_statistics and n_datasets >= 2:
                    dark_stats = self._compute_statistics(dark_raw_data)
                    light_stats = self._compute_statistics(light_raw_data)

                    # Store statistics results
                    self._store_statistics_result(metric_name, datasets, dark_stats, light_stats,
                                                  dark_means, dark_sems, light_means, light_sems)

                    # Get max bar height for positioning brackets
                    max_dark = max(dark_means[i] + dark_sems[i] for i in range(n_datasets)) if dark_means else 0
                    max_light = max(light_means[i] + light_sems[i] for i in range(n_datasets)) if light_means else 0

                    # Add brackets for dark phase
                    bracket_level = 0
                    for (i, j), p_value in sorted(dark_stats['pairwise'].items()):
                        symbol = self._get_significance_symbol(p_value)
                        if symbol and symbol != 'ns':
                            y_pos = max_dark * 1.05 + bracket_level * max_dark * 0.12
                            self._add_significance_bracket(
                                ax, x_dark[i], x_dark[j], y_pos, symbol
                            )
                            bracket_level += 1

                    # Add brackets for light phase
                    bracket_level = 0
                    for (i, j), p_value in sorted(light_stats['pairwise'].items()):
                        symbol = self._get_significance_symbol(p_value)
                        if symbol and symbol != 'ns':
                            y_pos = max_light * 1.05 + bracket_level * max_light * 0.12
                            self._add_significance_bracket(
                                ax, x_light[i], x_light[j], y_pos, symbol
                            )
                            bracket_level += 1

            # Styling - shortened title
            short_title = metric_name.split(' (')[0] if ' (' in metric_name else metric_name
            short_title = short_title.replace(' %', '')
            if len(short_title) > 15:
                short_title = short_title[:12] + '...'

            ax.set_title(short_title, fontsize=8, fontweight='bold',
                        color=self.TEXT_COLOR, pad=2)
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

            for spine in ax.spines.values():
                spine.set_color(self.GRID_COLOR)

            # Legend only on first plot - draggable so user can move it
            if idx == 0:
                legend = ax.legend(loc='upper left', fontsize=5, framealpha=0.9,
                                   facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                                   labelcolor=self.TEXT_COLOR)
                legend.get_frame().set_linewidth(0.5)
                legend.set_draggable(True)

        return fig

    def _compute_statistics(self, data_groups: List[np.ndarray],
                             apply_correction: bool = True) -> dict:
        """
        Compute statistical comparisons between groups.

        Args:
            data_groups: List of arrays, one per group (each array contains values for all animals in that group)
            apply_correction: If True, apply Bonferroni correction for 3+ groups.
                              If False, return raw (uncorrected) p-values.

        Returns:
            Dictionary with p-values for pairwise comparisons
        """
        from scipy import stats

        n_groups = len(data_groups)
        results = {
            'n_groups': n_groups,
            'pairwise': {},  # (i, j) -> p-value
            'anova_p': None
        }

        # Filter out groups with insufficient data
        valid_groups = [(i, g) for i, g in enumerate(data_groups)
                        if g is not None and len(g) >= 2 and not np.all(np.isnan(g))]

        if len(valid_groups) < 2:
            return results

        if len(valid_groups) == 2:
            # Two groups: use Welch's t-test
            i, g1 = valid_groups[0]
            j, g2 = valid_groups[1]
            # Remove NaN values
            g1_clean = g1[~np.isnan(g1)]
            g2_clean = g2[~np.isnan(g2)]
            if len(g1_clean) >= 2 and len(g2_clean) >= 2:
                _, p_value = stats.ttest_ind(g1_clean, g2_clean, equal_var=False)
                results['pairwise'][(i, j)] = p_value
        else:
            # Three or more groups: use ANOVA + pairwise t-tests
            # Clean the data
            clean_groups = []
            group_indices = []
            for i, g in valid_groups:
                g_clean = g[~np.isnan(g)]
                if len(g_clean) >= 2:
                    clean_groups.append(g_clean)
                    group_indices.append(i)

            if len(clean_groups) >= 2:
                # One-way ANOVA
                _, results['anova_p'] = stats.f_oneway(*clean_groups)

                # Pairwise t-tests
                n_comparisons = len(clean_groups) * (len(clean_groups) - 1) // 2
                for idx1, (i, g1) in enumerate(zip(group_indices, clean_groups)):
                    for idx2, (j, g2) in enumerate(zip(group_indices[idx1+1:], clean_groups[idx1+1:])):
                        j = group_indices[idx1 + 1 + idx2]
                        _, p_value = stats.ttest_ind(g1, g2, equal_var=False)
                        if apply_correction:
                            # Apply Bonferroni correction
                            results['pairwise'][(i, j)] = min(p_value * n_comparisons, 1.0)
                        else:
                            results['pairwise'][(i, j)] = p_value

        return results

    @staticmethod
    def _apply_fdr_correction(p_values: List[float]) -> List[float]:
        """
        Apply Benjamini-Hochberg FDR correction to a list of p-values.

        Args:
            p_values: List of raw (uncorrected) p-values

        Returns:
            List of FDR-adjusted q-values in the same order as input
        """
        m = len(p_values)
        if m == 0:
            return []
        if m == 1:
            return list(p_values)

        # Sort by p-value, keeping track of original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])

        # Compute adjusted p-values: q[i] = p[i] * m / rank
        adjusted = [0.0] * m
        for rank_0, (orig_idx, pval) in enumerate(indexed):
            rank = rank_0 + 1  # 1-indexed rank
            adjusted[orig_idx] = pval * m / rank

        # Enforce monotonicity from the largest p-value down
        # (each adjusted value must be <= the next larger one)
        sorted_orig_indices = [orig_idx for orig_idx, _ in indexed]
        for k in range(m - 2, -1, -1):
            idx_k = sorted_orig_indices[k]
            idx_k1 = sorted_orig_indices[k + 1]
            adjusted[idx_k] = min(adjusted[idx_k], adjusted[idx_k1])

        # Cap at 1.0
        adjusted = [min(q, 1.0) for q in adjusted]

        return adjusted

    def _get_significance_symbol(self, p_value: float) -> str:
        """Convert p-value to significance symbol."""
        if p_value is None or np.isnan(p_value):
            return ''
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'

    def _add_significance_bracket(self, ax, x1: float, x2: float, y: float,
                                   symbol: str, height: float = 0.03):
        """
        Add a significance bracket between two bar positions.

        Args:
            ax: Matplotlib axis
            x1, x2: X positions of the two bars
            y: Y position for the bracket (top of bars + offset)
            symbol: Significance symbol (*, **, ***, ns)
            height: Height of the bracket arms as fraction of y range
        """
        if not symbol or symbol == 'ns':
            return  # Don't show non-significant comparisons

        # Get current y limits to scale bracket
        y_min, y_max = ax.get_ylim()
        bracket_height = (y_max - y_min) * height

        # Draw the bracket
        bracket_color = self.TEXT_COLOR
        line_width = 1.0

        # Horizontal line
        ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y],
                color=bracket_color, linewidth=line_width, clip_on=False)

        # Add the significance symbol
        ax.text((x1 + x2) / 2, y + bracket_height * 1.2, symbol,
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color=bracket_color)

    def _clean_metric_name(self, metric_name: str) -> str:
        """Clean metric name for use as NPZ key."""
        return metric_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('%', 'pct')

    def _store_statistics_result(self, metric_name: str, datasets: List[Dict],
                                  dark_stats: dict, light_stats: dict,
                                  dark_means: List[float], dark_sems: List[float],
                                  light_means: List[float], light_sems: List[float]):
        """
        Store statistics results for later export.

        Args:
            metric_name: Name of the metric
            datasets: List of dataset dictionaries
            dark_stats: Statistics results for dark phase
            light_stats: Statistics results for light phase
            dark_means: Mean values for each dataset (dark)
            dark_sems: SEM values for each dataset (dark)
            light_means: Mean values for each dataset (light)
            light_sems: SEM values for each dataset (light)
        """
        n_datasets = len(datasets)
        dataset_labels = [self._get_dataset_label(ds) for ds in datasets]

        # Store dark phase results
        for (i, j), p_value in dark_stats['pairwise'].items():
            self.statistics_results.append({
                'metric': metric_name,
                'phase': 'Dark',
                'comparison': f'{dataset_labels[i]} vs {dataset_labels[j]}',
                'group1': dataset_labels[i],
                'group2': dataset_labels[j],
                'group1_mean': dark_means[i],
                'group1_sem': dark_sems[i],
                'group2_mean': dark_means[j],
                'group2_sem': dark_sems[j],
                'test': 't-test' if n_datasets == 2 else 't-test (Bonferroni)',
                'p_value': p_value,
                'significance': self._get_significance_symbol(p_value),
                'anova_p': dark_stats.get('anova_p')
            })

        # Store light phase results
        for (i, j), p_value in light_stats['pairwise'].items():
            self.statistics_results.append({
                'metric': metric_name,
                'phase': 'Light',
                'comparison': f'{dataset_labels[i]} vs {dataset_labels[j]}',
                'group1': dataset_labels[i],
                'group2': dataset_labels[j],
                'group1_mean': light_means[i],
                'group1_sem': light_sems[i],
                'group2_mean': light_means[j],
                'group2_sem': light_sems[j],
                'test': 't-test' if n_datasets == 2 else 't-test (Bonferroni)',
                'p_value': p_value,
                'significance': self._get_significance_symbol(p_value),
                'anova_p': light_stats.get('anova_p')
            })

    def create_statistics_summary_page(self, datasets: List[Dict[str, Any]]) -> Figure:
        """Create a summary page with all statistical test results."""
        fig = Figure(figsize=(11, 8.5), dpi=150, facecolor=self.BG_COLOR)

        fig.suptitle("Statistical Analysis Summary",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.97)

        ax = fig.add_axes([0.03, 0.05, 0.94, 0.88])
        ax.set_facecolor(self.BG_COLOR)
        ax.axis('off')

        if not self.statistics_results:
            ax.text(0.5, 0.5, "No statistical tests performed.\n(Enable 'Show statistics' and have 2+ datasets)",
                   ha='center', va='center', fontsize=12, color=self.TEXT_COLOR,
                   transform=ax.transAxes)
            return fig

        # Group results by metric
        from collections import defaultdict
        metrics_results = defaultdict(list)
        for result in self.statistics_results:
            metrics_results[result['metric']].append(result)

        # Create summary table
        y_pos = 0.95
        line_height = 0.022

        # Method description
        n_datasets = len(datasets)
        has_fdr = any(r.get('q_value') is not None for r in self.statistics_results)
        if n_datasets == 2:
            method_text = "Method: Welch's t-test (unequal variance)"
        else:
            method_text = "Method: One-way ANOVA + pairwise t-tests with Bonferroni correction"
        if has_fdr:
            method_text += "\nMulti-day/age bar charts: Benjamini-Hochberg FDR correction across all days/ages x phases"

        ax.text(0.0, y_pos, method_text, fontsize=9, color=self.TEXT_COLOR,
               fontweight='bold', transform=ax.transAxes)
        y_pos -= line_height * (1.5 if not has_fdr else 2.5)

        sig_label = "q" if has_fdr else "p"
        ax.text(0.0, y_pos, f"Significance: * {sig_label}<0.05, ** {sig_label}<0.01, *** {sig_label}<0.001, ns = not significant",
               fontsize=8, color=self.TEXT_COLOR, transform=ax.transAxes)
        y_pos -= line_height * 2

        # Table headers
        if has_fdr:
            headers = ['Metric', 'Phase', 'Comparison', 'p-value', 'q-value', 'Sig.']
            col_positions = [0.0, 0.22, 0.32, 0.58, 0.72, 0.86]
        else:
            headers = ['Metric', 'Phase', 'Comparison', 'p-value', 'Sig.']
            col_positions = [0.0, 0.25, 0.35, 0.68, 0.82]

        for i, (header, x) in enumerate(zip(headers, col_positions)):
            ax.text(x, y_pos, header, fontsize=8, fontweight='bold',
                   color=self.TEXT_COLOR, transform=ax.transAxes)
        y_pos -= line_height * 0.5

        # Draw header line
        ax.plot([0.0, 0.95], [y_pos, y_pos], color=self.GRID_COLOR,
               linewidth=0.5, transform=ax.transAxes)
        y_pos -= line_height

        # Results rows (limit to fit on page)
        max_rows = 30
        row_count = 0

        for metric, results in sorted(metrics_results.items()):
            if row_count >= max_rows:
                break

            # Truncate metric name for display
            display_metric = metric.split(' (')[0] if ' (' in metric else metric
            if len(display_metric) > 20:
                display_metric = display_metric[:17] + '...'

            for result in results:
                if row_count >= max_rows:
                    break

                # Format p-value
                p_val = result['p_value']
                if p_val is not None and not np.isnan(p_val):
                    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
                else:
                    p_str = 'N/A'

                # Format q-value if present
                q_val = result.get('q_value')
                if q_val is not None and not np.isnan(q_val):
                    q_str = f"{q_val:.4f}" if q_val >= 0.0001 else f"{q_val:.2e}"
                else:
                    q_str = None

                # Truncate comparison for display
                comparison = result['comparison']
                max_comp_len = 22 if has_fdr else 28
                if len(comparison) > max_comp_len:
                    comparison = comparison[:max_comp_len - 3] + '...'

                # Row data
                if has_fdr:
                    row_data = [
                        display_metric,
                        result['phase'],
                        comparison,
                        p_str,
                        q_str if q_str else p_str,
                        result['significance']
                    ]
                else:
                    row_data = [
                        display_metric,
                        result['phase'],
                        comparison,
                        p_str,
                        result['significance']
                    ]

                # Color-code significance
                sig = result['significance']
                if sig in ['***', '**', '*']:
                    row_color = '#90EE90' if self.light_mode else '#228B22'  # Green for significant
                else:
                    row_color = self.TEXT_COLOR

                for value, x in zip(row_data, col_positions):
                    ax.text(x, y_pos, value, fontsize=7,
                           color=row_color if x == col_positions[-1] else self.TEXT_COLOR,
                           transform=ax.transAxes)

                y_pos -= line_height
                row_count += 1

                # Only show metric name on first row for each metric
                display_metric = ''

        # Note if results were truncated
        total_results = sum(len(r) for r in metrics_results.values())
        if total_results > max_rows:
            ax.text(0.0, y_pos - line_height, f"... and {total_results - max_rows} more comparisons (see exported CSV for full results)",
                   fontsize=7, color=self.TEXT_COLOR, fontstyle='italic', transform=ax.transAxes)

        return fig

    # ========== Shared Daily Data Helpers ==========

    def _extract_daily_data_for_dataset(self, dataset: Dict[str, Any],
                                         key_base: str) -> List[Dict[str, Any]]:
        """
        Extract per-animal daily data from a consolidated dataset.

        Returns:
            List of dicts with keys: animal_idx, daily_data (n_days x 1440), n_days, age_at_start
        """
        n_days_arr = dataset.get(f'{key_base}_n_days_per_animal', np.array([]))
        animal_metadata = dataset.get('animal_metadata', [])
        if isinstance(animal_metadata, str):
            import json as _json
            animal_metadata = _json.loads(animal_metadata)

        animals = []
        for idx in range(len(n_days_arr)):
            n_days = int(n_days_arr[idx])
            if n_days == 0:
                continue
            daily_data = dataset.get(f'{key_base}_daily_{idx}', None)
            if daily_data is None:
                continue
            daily_data = np.array(daily_data)
            if daily_data.ndim == 1:
                daily_data = daily_data.reshape(1, -1)

            age_at_start = None
            if idx < len(animal_metadata):
                meta = animal_metadata[idx] if isinstance(animal_metadata[idx], dict) else {}
                age_at_start = meta.get('age_days_at_start', None)

            animals.append({
                'animal_idx': idx,
                'daily_data': daily_data,
                'n_days': daily_data.shape[0],
                'age_at_start': age_at_start
            })
        return animals

    def _compute_population_daily_means(self, animal_list: List[Dict],
                                         max_days: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute population-average trace and SEM for each day.

        Returns:
            (day_means, day_sems) - each a list of 1440-length arrays.
        """
        day_means = []
        day_sems = []
        for day_idx in range(max_days):
            traces = []
            for animal in animal_list:
                if day_idx < animal['n_days']:
                    row = animal['daily_data'][day_idx]
                    if len(row) == self.MINUTES_PER_DAY:
                        traces.append(row)
            if traces:
                stacked = np.array(traces)
                day_means.append(np.nanmean(stacked, axis=0))
                n_valid = np.sum(~np.isnan(stacked), axis=0)
                day_sems.append(np.nanstd(stacked, axis=0) / np.sqrt(np.maximum(n_valid, 1)))
            else:
                day_means.append(np.full(self.MINUTES_PER_DAY, np.nan))
                day_sems.append(np.full(self.MINUTES_PER_DAY, np.nan))
        return day_means, day_sems

    def _draw_overlaid_48h_cta(self, ax, datasets: List[Dict[str, Any]],
                                cta_key: str, sem_key: str, title: str,
                                show_legend: bool = True):
        """Draw overlaid 48h CTAs from multiple datasets on a single axes."""
        # L-D-L-D shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(36, 48, facecolor=self.DARK_PHASE_COLOR, zorder=0)

        x_hours = np.arange(2 * self.MINUTES_PER_DAY) / 60

        for ds_idx, ds in enumerate(datasets):
            color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)
            cta = ds.get(cta_key)
            sem = ds.get(sem_key)

            if cta is None or len(cta) != self.MINUTES_PER_DAY:
                continue

            smoothed = self._smooth_data(cta)
            cta_48h = np.concatenate([smoothed, smoothed])
            ax.plot(x_hours, cta_48h, color=color, linewidth=1.2,
                    label=label, alpha=0.9)

            if sem is not None and len(sem) == self.MINUTES_PER_DAY:
                smoothed_sem = self._smooth_data(sem)
                sem_48h = np.concatenate([smoothed_sem, smoothed_sem])
                ax.fill_between(x_hours, cta_48h - sem_48h, cta_48h + sem_48h,
                               color=color, alpha=self.SEM_ALPHA)

        ax.set_xlim(0, 48)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36', 'ZT48'], fontsize=6)
        ax.set_title(title, fontsize=8, fontweight='bold', color=self.TEXT_COLOR, pad=3)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        if show_legend:
            legend = ax.legend(loc='upper right', fontsize=5, framealpha=0.8,
                               facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                               labelcolor=self.TEXT_COLOR)
            legend.set_draggable(True)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_overlaid_day_cta(self, ax, all_day_means: List[List[np.ndarray]],
                                all_day_sems: List[List[np.ndarray]],
                                datasets: List[Dict[str, Any]],
                                day_idx: int, title: str,
                                show_legend: bool = False):
        """
        Draw overlaid 48h double-plotted CTA for a specific day pair across datasets.

        all_day_means[ds_idx] = list of 1440-length arrays per day for that dataset.
        all_day_sems[ds_idx] = corresponding SEM arrays.
        day_idx = the starting day index for the double plot.
        """
        # L-D-L-D shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(36, 48, facecolor=self.DARK_PHASE_COLOR, zorder=0)

        x_hours = np.arange(2 * self.MINUTES_PER_DAY) / 60

        for ds_idx, ds in enumerate(datasets):
            color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)
            day_means = all_day_means[ds_idx]
            day_sems = all_day_sems[ds_idx]

            if day_idx >= len(day_means):
                continue

            trace_first = self._smooth_data(day_means[day_idx])
            if np.all(np.isnan(trace_first)):
                continue

            if day_idx + 1 < len(day_means):
                trace_second = self._smooth_data(day_means[day_idx + 1])
            else:
                trace_second = np.full(self.MINUTES_PER_DAY, np.nan)

            trace_48h = np.concatenate([trace_first, trace_second])
            ax.plot(x_hours, trace_48h, color=color, linewidth=1.2,
                    label=label, alpha=0.9)

            # SEM shading
            sem_first = self._smooth_data(day_sems[day_idx]) if day_idx < len(day_sems) else np.zeros(self.MINUTES_PER_DAY)
            if day_idx + 1 < len(day_sems):
                sem_second = self._smooth_data(day_sems[day_idx + 1])
            else:
                sem_second = np.full(self.MINUTES_PER_DAY, np.nan)
            sem_48h = np.concatenate([sem_first, sem_second])
            ax.fill_between(x_hours, trace_48h - sem_48h, trace_48h + sem_48h,
                           color=color, alpha=self.SEM_ALPHA)

        ax.set_xlim(0, 48)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36', 'ZT48'], fontsize=6)
        ax.set_title(title, fontsize=8, fontweight='bold', color=self.TEXT_COLOR, pad=3)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        if show_legend:
            legend = ax.legend(loc='upper right', fontsize=5, framealpha=0.8,
                               facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                               labelcolor=self.TEXT_COLOR)
            legend.set_draggable(True)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    # ========== Actogram Comparison ==========

    def _generate_actogram_pages(self, datasets: List[Dict[str, Any]],
                                  metric_name: str) -> List[Tuple[str, Figure]]:
        """
        Generate actogram comparison pages for one metric.

        Layout per page: 4 columns x 2 rows of overlaid 48h double-plotted CTAs.
        Columns = day pairs (D1-2, D2-3, ...) + Grand Average CTA as last panel.
        All datasets overlaid on each subplot.
        Final page includes bar chart of per-day light/dark means with stats.

        Returns list of (title, figure) tuples.
        """
        key_base = self._clean_metric_name(metric_name)
        pages = []

        # Extract daily data for all datasets and compute per-day population means + SEMs
        all_day_means = []  # per dataset: list of 1440-length arrays
        all_day_sems = []   # per dataset: list of 1440-length SEM arrays
        max_days = 0
        has_any_data = False

        for ds in datasets:
            animals = self._extract_daily_data_for_dataset(ds, key_base)
            if animals:
                ds_max = max(a['n_days'] for a in animals)
                max_days = max(max_days, ds_max)
                day_means, day_sems = self._compute_population_daily_means(animals, ds_max)
                all_day_means.append(day_means)
                all_day_sems.append(day_sems)
                has_any_data = True
            else:
                all_day_means.append([])
                all_day_sems.append([])

        if not has_any_data or max_days < 1:
            fig = Figure(figsize=(11, 8.5), dpi=150, facecolor=self.BG_COLOR)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_facecolor(self.BG_COLOR)
            ax.axis('off')
            ax.text(0.5, 0.5, f"No daily data available for {metric_name}",
                    ha='center', va='center', fontsize=12, color=self.TEXT_COLOR,
                    transform=ax.transAxes)
            pages.append((f"Actogram: {metric_name}", fig))
            return pages

        # Build list of panels: one per day pair + grand average
        # Day pairs: D1-2, D2-3, ..., D(n-1)-n
        n_day_pairs = max_days  # day 0..max_days-1, each double-plotted
        panel_count = n_day_pairs + 1  # +1 for grand average CTA

        # Layout: 4 columns, 2 rows of CTAs per page
        cols_per_page = 4
        rows_per_page = 2
        panels_per_page = cols_per_page * rows_per_page  # 8

        # Generate CTA grid pages
        panel_idx = 0
        page_num = 0

        while panel_idx < panel_count:
            page_num += 1
            panels_this_page = min(panels_per_page, panel_count - panel_idx)
            n_rows = (panels_this_page + cols_per_page - 1) // cols_per_page

            fig = Figure(figsize=(11, 4.5 * n_rows), dpi=150, facecolor=self.BG_COLOR)
            fig.suptitle(f"Actogram: {metric_name} (Page {page_num})",
                         fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

            gs = GridSpec(n_rows, cols_per_page, figure=fig,
                          hspace=0.35, wspace=0.25,
                          left=0.06, right=0.94, top=0.92, bottom=0.06)

            for slot in range(panels_this_page):
                cur = panel_idx + slot
                row = slot // cols_per_page
                col = slot % cols_per_page

                ax = fig.add_subplot(gs[row, col])
                ax.set_facecolor(self.BG_COLOR)

                if cur < n_day_pairs:
                    # Day pair CTA
                    day_i = cur
                    title = f'D{day_i+1}-{day_i+2}'
                    self._draw_overlaid_day_cta(ax, all_day_means, all_day_sems,
                                                 datasets, day_i, title,
                                                 show_legend=(cur == 0))
                else:
                    # Grand average CTA (last panel)
                    self._draw_overlaid_48h_cta(
                        ax, datasets,
                        f'{key_base}_grand_cta', f'{key_base}_grand_sem',
                        'Grand Average', show_legend=(panel_idx == 0 and slot == 0))

            panel_idx += panels_this_page
            pages.append((f"Actogram: {metric_name} p{page_num}", fig))

        # Bar chart page: per-day + grand average light/dark means
        bar_fig = self._create_actogram_bar_page(
            datasets, all_day_means, key_base, metric_name, max_days)
        pages.append((f"Actogram Bars: {metric_name}", bar_fig))

        return pages

    def _create_actogram_bar_page(self, datasets: List[Dict[str, Any]],
                                    all_day_means: List[List[np.ndarray]],
                                    key_base: str, metric_name: str,
                                    max_days: int) -> Figure:
        """
        Create bar chart page for actogram mode.

        Two bar charts (Dark / Light) showing per-day + grand average means
        for each dataset, with FDR-corrected statistics across all days and phases.
        """
        n_datasets = len(datasets)

        # Compute per-day light/dark means and raw values for each dataset
        day_labels = [f'D{d+1}' for d in range(max_days)] + ['Grand\nAvg']
        n_groups = max_days + 1  # days + grand avg

        # all_dark[ds_idx][group_idx] = array of per-animal means
        all_dark = [[] for _ in range(n_datasets)]
        all_light = [[] for _ in range(n_datasets)]

        for ds_idx, ds in enumerate(datasets):
            animals = self._extract_daily_data_for_dataset(ds, key_base)

            for day_idx in range(max_days):
                dark_vals = []
                light_vals = []
                for animal in animals:
                    if day_idx < animal['n_days']:
                        trace = animal['daily_data'][day_idx]
                        if len(trace) == self.MINUTES_PER_DAY:
                            lm = np.nanmean(trace[:720])
                            dm = np.nanmean(trace[720:])
                            if not np.isnan(lm):
                                light_vals.append(lm)
                            if not np.isnan(dm):
                                dark_vals.append(dm)
                all_dark[ds_idx].append(np.array(dark_vals))
                all_light[ds_idx].append(np.array(light_vals))

            # Grand average: use the precomputed dark/light means arrays
            dark_arr = ds.get(f'{key_base}_dark_means', np.array([]))
            light_arr = ds.get(f'{key_base}_light_means', np.array([]))
            all_dark[ds_idx].append(dark_arr[~np.isnan(dark_arr)] if len(dark_arr) > 0 else np.array([]))
            all_light[ds_idx].append(light_arr[~np.isnan(light_arr)] if len(light_arr) > 0 else np.array([]))

        # Figure with two rows: Dark phase, Light phase
        fig = Figure(figsize=(max(11, n_groups * 0.8 + 2), 8.5), dpi=150,
                     facecolor=self.BG_COLOR)
        fig.suptitle(f"Daily Means: {metric_name}",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.97)

        gs = GridSpec(2, 1, figure=fig, hspace=0.35,
                      left=0.07, right=0.95, top=0.91, bottom=0.08)

        error_color = '#000000' if self.light_mode else '#ffffff'
        error_kw = {'elinewidth': 1.0, 'capthick': 1.0, 'ecolor': error_color}

        phase_info = [
            ('Dark Phase', 'dark', all_dark),
            ('Light Phase', 'light', all_light)
        ]
        axes = []
        x_base = np.arange(n_groups)
        total_width = 0.8
        bar_width = total_width / n_datasets

        # --- Pass 1: Draw bars and scatter points ---
        for row_idx, (phase, phase_key, all_phase) in enumerate(phase_info):
            ax = fig.add_subplot(gs[row_idx])
            ax.set_facecolor(self.BG_COLOR)
            axes.append(ax)

            dot_color = '#000000' if self.light_mode else '#ffffff'

            for ds_idx, ds in enumerate(datasets):
                color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
                label = self._get_dataset_label(ds) if row_idx == 0 else None

                means = []
                sems = []
                for g in range(n_groups):
                    arr = all_phase[ds_idx][g]
                    if len(arr) > 0:
                        means.append(np.mean(arr))
                        sems.append(np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0)
                    else:
                        means.append(0)
                        sems.append(0)

                offset = (ds_idx - n_datasets / 2 + 0.5) * bar_width
                ax.bar(x_base + offset, means, bar_width * 0.9,
                       yerr=sems, capsize=2, color=color, alpha=0.8,
                       label=label, error_kw=error_kw)

                for g in range(n_groups):
                    arr = all_phase[ds_idx][g]
                    if len(arr) > 0:
                        jitter = np.random.default_rng(ds_idx * 1000 + g).uniform(
                            -bar_width * 0.25, bar_width * 0.25, size=len(arr))
                        marker = self.DATASET_MARKERS[ds_idx % len(self.DATASET_MARKERS)]
                        ax.scatter(np.full(len(arr), x_base[g] + offset) + jitter,
                                  arr, color=dot_color, s=12, alpha=0.6,
                                  zorder=5, edgecolors='none', marker=marker)

            ax.axvline(x=max_days - 0.5, color='#888888', linestyle='--',
                       linewidth=1.2, alpha=0.9)

            ax.set_title(phase, fontsize=10, fontweight='bold',
                        color=self.TEXT_COLOR, pad=5)
            ax.set_xticks(x_base)
            ax.set_xticklabels(day_labels, fontsize=7)
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

            if row_idx == 0:
                legend = ax.legend(loc='upper right', fontsize=6, framealpha=0.8,
                                   facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                                   labelcolor=self.TEXT_COLOR)
                legend.set_draggable(True)

            for spine in ax.spines.values():
                spine.set_color(self.GRID_COLOR)

        # --- Pass 2: FDR-corrected statistics across all days × phases ---
        if self.show_statistics and n_datasets >= 2:
            dataset_labels = [self._get_dataset_label(ds) for ds in datasets]

            # Collect raw (uncorrected) p-values from all groups × both phases
            raw_entries = []  # list of dicts with raw p-value + metadata for each test
            for row_idx, (phase, phase_key, all_phase) in enumerate(phase_info):
                for g in range(n_groups):
                    raw_groups = [all_phase[ds_idx][g] for ds_idx in range(n_datasets)]
                    stats = self._compute_statistics(raw_groups, apply_correction=False)
                    grp_label = day_labels[g].replace('\n', ' ')

                    for (i, j), p_value in sorted(stats['pairwise'].items()):
                        raw_entries.append({
                            'row_idx': row_idx,
                            'phase_key': phase_key,
                            'g': g,
                            'grp_label': grp_label,
                            'i': i, 'j': j,
                            'raw_p': p_value,
                            'raw_groups': raw_groups,
                            'anova_p': stats.get('anova_p'),
                        })

            # Apply FDR correction to the full set of p-values
            if raw_entries:
                raw_p_values = [e['raw_p'] for e in raw_entries]
                corrected_q = self._apply_fdr_correction(raw_p_values)

                # Compute global max for bracket spacing per phase
                phase_global_max = {}
                for row_idx, (phase, phase_key, all_phase) in enumerate(phase_info):
                    gmax = 0
                    for g in range(n_groups):
                        for ds_idx in range(n_datasets):
                            arr = all_phase[ds_idx][g]
                            if len(arr) > 0:
                                h = max(np.max(arr),
                                        np.mean(arr) + (np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0))
                                gmax = max(gmax, h)
                    phase_global_max[row_idx] = gmax

                # Track bracket levels per (phase, group) for staggering
                bracket_levels = {}

                for idx, entry in enumerate(raw_entries):
                    q_value = corrected_q[idx]
                    ri = entry['row_idx']
                    g = entry['g']
                    i, j = entry['i'], entry['j']
                    raw_groups = entry['raw_groups']

                    # Store statistics result with FDR-corrected q-value
                    self.statistics_results.append({
                        'metric': metric_name,
                        'phase': entry['phase_key'].capitalize(),
                        'comparison': f'{dataset_labels[i]} vs {dataset_labels[j]} ({entry["grp_label"]})',
                        'group1': dataset_labels[i],
                        'group2': dataset_labels[j],
                        'group1_mean': np.mean(raw_groups[i]) if len(raw_groups[i]) > 0 else np.nan,
                        'group1_sem': np.std(raw_groups[i]) / np.sqrt(len(raw_groups[i])) if len(raw_groups[i]) > 1 else 0,
                        'group2_mean': np.mean(raw_groups[j]) if len(raw_groups[j]) > 0 else np.nan,
                        'group2_sem': np.std(raw_groups[j]) / np.sqrt(len(raw_groups[j])) if len(raw_groups[j]) > 1 else 0,
                        'test': 't-test (FDR)' if n_datasets == 2 else 'ANOVA + t-test (FDR)',
                        'p_value': entry['raw_p'],
                        'q_value': q_value,
                        'significance': self._get_significance_symbol(q_value),
                        'anova_p': entry['anova_p']
                    })

                    # Draw bracket using corrected q-value
                    symbol = self._get_significance_symbol(q_value)
                    if symbol and symbol != 'ns':
                        ax = axes[ri]
                        gmax = phase_global_max[ri]
                        bracket_spacing = gmax * 0.08 if gmax > 0 else 0.1

                        level_key = (ri, g)
                        bracket_level = bracket_levels.get(level_key, 0)

                        offset_i = (i - n_datasets / 2 + 0.5) * bar_width
                        offset_j = (j - n_datasets / 2 + 0.5) * bar_width
                        local_max = max(
                            (np.max(raw_groups[k]) if len(raw_groups[k]) > 0 else 0)
                            for k in range(n_datasets)
                        )
                        y_pos = local_max * 1.05 + bracket_level * bracket_spacing
                        self._add_significance_bracket(
                            ax, x_base[g] + offset_i, x_base[g] + offset_j,
                            y_pos, symbol)
                        bracket_levels[level_key] = bracket_level + 1

        return fig

    # ========== Age Trend Comparison ==========

    def _any_dataset_has_age_data(self, datasets: List[Dict[str, Any]]) -> bool:
        """Check if any dataset has animal age data."""
        for ds in datasets:
            animal_metadata = ds.get('animal_metadata', [])
            if isinstance(animal_metadata, str):
                import json as _json
                animal_metadata = _json.loads(animal_metadata)
            for meta in animal_metadata:
                if isinstance(meta, dict) and meta.get('age_days_at_start') is not None:
                    return True
        return False

    def _create_no_age_data_page(self) -> Figure:
        """Create an info page indicating no age data is available."""
        fig = Figure(figsize=(11, 8.5), dpi=150, facecolor=self.BG_COLOR)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.set_facecolor(self.BG_COLOR)
        ax.axis('off')
        ax.text(0.5, 0.55, "No Age Data Available",
                ha='center', va='center', fontsize=18, fontweight='bold',
                color=self.TEXT_COLOR, transform=ax.transAxes)
        ax.text(0.5, 0.42,
                "None of the loaded datasets contain animal age information.\n"
                "Age data (age_days_at_start) must be present in the source NPZ files\n"
                "for age trend analysis.",
                ha='center', va='center', fontsize=11, color='#aaaaaa',
                transform=ax.transAxes, linespacing=1.5)
        return fig

    def _create_age_coverage_comparison_page(self, datasets: List[Dict[str, Any]]) -> Optional[Figure]:
        """
        Create an age coverage overview page comparing data availability across datasets.

        Top panel: Gantt-style chart with one row per animal, grouped by dataset.
        Each dataset uses its own color. Segments are colored where data exists.
        Bottom panel: Stacked bar chart showing animal count per age per dataset.

        Returns:
            Figure or None if no age data available
        """
        from matplotlib.patches import Patch

        NO_DATA_COLOR = '#3a3a3a' if not self.light_mode else '#e0e0e0'

        # Collect per-dataset animal coverage info
        dataset_animals = []  # list of (ds_idx, label, color, animals_list)

        for ds_idx, ds in enumerate(datasets):
            color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)
            animal_metadata = ds.get('animal_metadata', [])
            if isinstance(animal_metadata, str):
                import json as _json
                animal_metadata = _json.loads(animal_metadata)

            # Use first available metric to check daily data
            metric_names = list(ds.get('metric_names', []))
            if hasattr(metric_names, 'tolist'):
                metric_names = metric_names.tolist()

            animals = []
            for idx in range(len(animal_metadata)):
                meta = animal_metadata[idx] if isinstance(animal_metadata[idx], dict) else {}
                age_at_start = meta.get('age_days_at_start', None)
                if age_at_start is None:
                    continue
                age_at_start = int(age_at_start)

                animal_id = meta.get('animal_id', f'Animal {idx}')

                # Find daily data from any metric
                n_days = 0
                ages_with_data = set()
                for mn in metric_names[:1]:  # Check first metric
                    key_base = self._clean_metric_name(mn)
                    daily_key = f'{key_base}_daily_{idx}'
                    daily_data = ds.get(daily_key, None)
                    if daily_data is not None:
                        daily_data = np.array(daily_data)
                        if daily_data.ndim == 1:
                            daily_data = daily_data.reshape(1, -1)
                        n_days = daily_data.shape[0]
                        for d in range(n_days):
                            if (daily_data[d].shape[0] == self.MINUTES_PER_DAY and
                                    not np.all(np.isnan(daily_data[d]))):
                                ages_with_data.add(age_at_start + d)

                if n_days > 0:
                    animals.append({
                        'animal_id': animal_id,
                        'age_at_start': age_at_start,
                        'n_days': n_days,
                        'ages_with_data': ages_with_data,
                    })

            if animals:
                animals.sort(key=lambda a: a['age_at_start'])
                dataset_animals.append((ds_idx, label, color, animals))

        if not dataset_animals:
            return None

        # Compute global age range
        all_ages = set()
        for _, _, _, animals in dataset_animals:
            for a in animals:
                all_ages.update(a['ages_with_data'])
        if not all_ages:
            return None

        min_age = min(all_ages)
        max_age = max(all_ages)
        age_range = list(range(min_age, max_age + 1))
        n_ages = len(age_range)

        # Total number of animal rows (with group separators)
        total_animals = sum(len(animals) for _, _, _, animals in dataset_animals)
        n_separators = len(dataset_animals) - 1
        total_rows = total_animals + n_separators

        # Create figure
        fig_height = max(6, 2.5 + total_rows * 0.3)
        fig = Figure(figsize=(12, fig_height), dpi=150, facecolor=self.BG_COLOR)
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.3,
                      left=0.12, right=0.95, top=0.93, bottom=0.06)

        # === Top panel: Gantt-style coverage chart ===
        ax_gantt = fig.add_subplot(gs[0])
        ax_gantt.set_facecolor(self.BG_COLOR)

        y_labels = []
        y_pos = total_rows - 1  # Start from top

        for grp_idx, (ds_idx, label, color, animals) in enumerate(dataset_animals):
            if grp_idx > 0:
                # Draw separator line
                ax_gantt.axhline(y=y_pos + 0.5, color=self.GRID_COLOR,
                                 linewidth=0.5, linestyle='--', alpha=0.5)
                y_pos -= 1  # Skip a row for separator

            for animal in animals:
                animal_min = animal['age_at_start']
                animal_max = animal_min + animal['n_days'] - 1

                for age in age_range:
                    x = age - min_age
                    if age in animal['ages_with_data']:
                        ax_gantt.barh(y_pos, 1, left=x, height=0.7,
                                     color=color, edgecolor='none', linewidth=0)
                    elif animal_min <= age <= animal_max:
                        ax_gantt.barh(y_pos, 1, left=x, height=0.7,
                                     color=NO_DATA_COLOR, edgecolor='none', linewidth=0)

                y_labels.append((y_pos, animal['animal_id'][:15]))
                y_pos -= 1

        # Y-axis labels
        ytick_positions = [pos for pos, _ in y_labels]
        ytick_labels = [lbl for _, lbl in y_labels]
        ax_gantt.set_yticks(ytick_positions)
        ax_gantt.set_yticklabels(ytick_labels, fontsize=5, color=self.TEXT_COLOR)
        ax_gantt.set_ylim(-0.5, total_rows - 0.5)

        # X-axis: age labels
        ax_gantt.set_xlim(-0.5, n_ages - 0.5)
        tick_step = max(1, n_ages // 25)
        tick_positions = list(range(0, n_ages, tick_step))
        ax_gantt.set_xticks(tick_positions)
        ax_gantt.set_xticklabels([f'P{age_range[t]}' for t in tick_positions],
                                 fontsize=6, color=self.TEXT_COLOR, rotation=45, ha='right')

        ax_gantt.set_title(
            f"Age Coverage Overview (n={total_animals} animals, P{min_age}\u2013P{max_age})",
            fontsize=12, fontweight='bold', color=self.TEXT_COLOR, pad=10)
        ax_gantt.tick_params(colors=self.TEXT_COLOR, labelsize=6)
        for spine in ax_gantt.spines.values():
            spine.set_color(self.GRID_COLOR)

        # Legend
        legend_elements = [
            Patch(facecolor=color, label=label)
            for _, label, color, _ in dataset_animals
        ]
        legend_elements.append(Patch(facecolor=NO_DATA_COLOR, label='Recording period (no data)'))
        ax_gantt.legend(handles=legend_elements, fontsize=7, loc='lower right',
                       framealpha=0.3, labelcolor=self.TEXT_COLOR, edgecolor=self.GRID_COLOR)

        # === Bottom panel: Animal count per age per dataset (stacked) ===
        ax_count = fig.add_subplot(gs[1])
        ax_count.set_facecolor(self.BG_COLOR)

        x_positions = np.arange(n_ages)
        bottom = np.zeros(n_ages)

        for ds_idx, label, color, animals in dataset_animals:
            counts = np.array([
                sum(1 for a in animals if age in a['ages_with_data'])
                for age in age_range
            ])
            ax_count.bar(x_positions, counts, width=0.8, bottom=bottom,
                        color=color, alpha=0.7, edgecolor='none', label=label)
            bottom += counts

        ax_count.set_xlim(-0.5, n_ages - 0.5)
        ax_count.set_xticks(tick_positions)
        ax_count.set_xticklabels([f'P{age_range[t]}' for t in tick_positions],
                                 fontsize=6, color=self.TEXT_COLOR, rotation=45, ha='right')
        ax_count.set_xlabel("Postnatal Age", fontsize=9, color=self.TEXT_COLOR)
        ax_count.set_ylabel("n animals", fontsize=8, color=self.TEXT_COLOR)
        ax_count.set_title("Animals with Data at Each Age", fontsize=10,
                          fontweight='bold', color=self.TEXT_COLOR, pad=5)
        ax_count.tick_params(colors=self.TEXT_COLOR, labelsize=6)
        ax_count.set_ylim(0, np.max(bottom) * 1.2 if np.max(bottom) > 0 else 1)
        for spine in ax_count.spines.values():
            spine.set_color(self.GRID_COLOR)
        ax_count.legend(fontsize=6, loc='upper right', framealpha=0.3,
                       labelcolor=self.TEXT_COLOR, edgecolor=self.GRID_COLOR)

        return fig

    def _compute_age_ctas_for_dataset(self, dataset: Dict[str, Any],
                                       key_base: str) -> Dict[int, Dict[str, Any]]:
        """
        Compute per-postnatal-age data for a dataset.

        Returns:
            Dict mapping age (int) -> {
                'traces': list of 1440-length arrays (one per animal-day at this age),
                'mean_cta': 1440-length array (nanmean across traces),
                'sem_cta': 1440-length array,
                'dark_values': list of per-animal dark phase means,
                'light_values': list of per-animal light phase means,
                'dark_mean': float, 'dark_sem': float,
                'light_mean': float, 'light_sem': float,
                'n_animals': int
            }
        """
        animals = self._extract_daily_data_for_dataset(dataset, key_base)
        age_data = {}

        for animal in animals:
            age_start = animal.get('age_at_start')
            if age_start is None:
                continue
            age_start = int(age_start)

            for day_idx in range(animal['n_days']):
                age = age_start + day_idx
                day_trace = animal['daily_data'][day_idx]
                if len(day_trace) != self.MINUTES_PER_DAY:
                    continue

                if age not in age_data:
                    age_data[age] = {'traces': [], 'dark_values': [], 'light_values': []}

                age_data[age]['traces'].append(day_trace)

                light_mean = np.nanmean(day_trace[:720])
                dark_mean = np.nanmean(day_trace[720:])
                if not np.isnan(dark_mean):
                    age_data[age]['dark_values'].append(dark_mean)
                if not np.isnan(light_mean):
                    age_data[age]['light_values'].append(light_mean)

        # Compute summary stats per age
        for age, d in age_data.items():
            stacked = np.array(d['traces'])
            d['mean_cta'] = np.nanmean(stacked, axis=0)
            n_valid = np.sum(~np.isnan(stacked), axis=0)
            d['sem_cta'] = np.nanstd(stacked, axis=0) / np.sqrt(np.maximum(n_valid, 1))

            dark_arr = np.array(d['dark_values'])
            light_arr = np.array(d['light_values'])
            d['dark_mean'] = np.mean(dark_arr) if len(dark_arr) > 0 else np.nan
            d['dark_sem'] = np.std(dark_arr) / np.sqrt(len(dark_arr)) if len(dark_arr) > 1 else 0.0
            d['light_mean'] = np.mean(light_arr) if len(light_arr) > 0 else np.nan
            d['light_sem'] = np.std(light_arr) / np.sqrt(len(light_arr)) if len(light_arr) > 1 else 0.0
            d['n_animals'] = len(d['traces'])

        return age_data

    def _draw_overlaid_age_cta(self, ax, all_age_data: List[Dict],
                                datasets: List[Dict[str, Any]],
                                age: int, title: str,
                                show_legend: bool = False):
        """Draw overlaid 48h CTA for a specific postnatal age across all datasets."""
        # L-D-L-D shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(36, 48, facecolor=self.DARK_PHASE_COLOR, zorder=0)

        x_hours = np.arange(2 * self.MINUTES_PER_DAY) / 60

        for ds_idx, ds in enumerate(datasets):
            color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)
            age_data = all_age_data[ds_idx]

            if age not in age_data:
                continue

            cta = self._smooth_data(age_data[age]['mean_cta'])
            sem = self._smooth_data(age_data[age]['sem_cta'])

            if np.all(np.isnan(cta)):
                continue

            cta_48h = np.concatenate([cta, cta])
            sem_48h = np.concatenate([sem, sem])

            ax.plot(x_hours, cta_48h, color=color, linewidth=1.2,
                    label=label, alpha=0.9)
            ax.fill_between(x_hours, cta_48h - sem_48h, cta_48h + sem_48h,
                           color=color, alpha=self.SEM_ALPHA)

        ax.set_xlim(0, 48)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36', 'ZT48'], fontsize=6)
        ax.set_title(title, fontsize=8, fontweight='bold', color=self.TEXT_COLOR, pad=3)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        if show_legend:
            legend = ax.legend(loc='upper right', fontsize=5, framealpha=0.8,
                               facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                               labelcolor=self.TEXT_COLOR)
            legend.set_draggable(True)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _generate_age_trend_pages(self, datasets: List[Dict[str, Any]],
                                   metric_name: str) -> List[Tuple[str, Figure]]:
        """
        Generate age trend comparison pages for one metric.

        Same layout as actogram: 4 columns x 2 rows of overlaid CTAs per page,
        one per postnatal age + grand average CTA. Bar chart page at end.
        """
        key_base = self._clean_metric_name(metric_name)
        pages = []

        # Compute per-age data for all datasets
        all_age_data = []
        all_ages = set()
        for ds in datasets:
            age_data = self._compute_age_ctas_for_dataset(ds, key_base)
            all_age_data.append(age_data)
            all_ages.update(age_data.keys())

        if not all_ages:
            fig = Figure(figsize=(11, 8.5), dpi=150, facecolor=self.BG_COLOR)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_facecolor(self.BG_COLOR)
            ax.axis('off')
            ax.text(0.5, 0.5, f"No age data available for {metric_name}",
                    ha='center', va='center', fontsize=12, color=self.TEXT_COLOR,
                    transform=ax.transAxes)
            pages.append((f"Age Trend: {metric_name}", fig))
            return pages

        ages_sorted = sorted(all_ages)
        panel_count = len(ages_sorted) + 1  # ages + grand average CTA

        cols_per_page = 4
        rows_per_page = 2
        panels_per_page = cols_per_page * rows_per_page

        # Generate CTA grid pages
        panel_idx = 0
        page_num = 0

        while panel_idx < panel_count:
            page_num += 1
            panels_this_page = min(panels_per_page, panel_count - panel_idx)
            n_rows = (panels_this_page + cols_per_page - 1) // cols_per_page

            fig = Figure(figsize=(11, 4.5 * n_rows), dpi=150, facecolor=self.BG_COLOR)
            fig.suptitle(f"Age Trend: {metric_name} (Page {page_num})",
                         fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

            gs = GridSpec(n_rows, cols_per_page, figure=fig,
                          hspace=0.35, wspace=0.25,
                          left=0.06, right=0.94, top=0.92, bottom=0.06)

            for slot in range(panels_this_page):
                cur = panel_idx + slot
                row = slot // cols_per_page
                col = slot % cols_per_page

                ax = fig.add_subplot(gs[row, col])
                ax.set_facecolor(self.BG_COLOR)

                if cur < len(ages_sorted):
                    age = ages_sorted[cur]
                    title = f'P{age}'
                    self._draw_overlaid_age_cta(ax, all_age_data, datasets,
                                                 age, title,
                                                 show_legend=(cur == 0))
                else:
                    # Grand average CTA (last panel)
                    self._draw_overlaid_48h_cta(
                        ax, datasets,
                        f'{key_base}_grand_cta', f'{key_base}_grand_sem',
                        'Grand Average', show_legend=(panel_idx == 0 and slot == 0))

            panel_idx += panels_this_page
            pages.append((f"Age Trend: {metric_name} p{page_num}", fig))

        # Bar chart page: per-age + grand average light/dark means
        bar_fig = self._create_age_bar_page(
            datasets, all_age_data, ages_sorted, key_base, metric_name)
        pages.append((f"Age Bars: {metric_name}", bar_fig))

        return pages

    def _create_age_bar_page(self, datasets: List[Dict[str, Any]],
                              all_age_data: List[Dict],
                              ages_sorted: List[int],
                              key_base: str, metric_name: str) -> Figure:
        """
        Create bar chart page for age trend mode.

        Two bar charts (Dark / Light) showing per-age + grand average means
        for each dataset, with FDR-corrected statistics across all ages and phases.
        """
        n_datasets = len(datasets)
        n_ages = len(ages_sorted)

        group_labels = [f'P{a}' for a in ages_sorted] + ['Grand\nAvg']
        n_groups = n_ages + 1

        all_dark = [[] for _ in range(n_datasets)]
        all_light = [[] for _ in range(n_datasets)]

        for ds_idx in range(n_datasets):
            age_data = all_age_data[ds_idx]
            for age in ages_sorted:
                if age in age_data:
                    all_dark[ds_idx].append(np.array(age_data[age]['dark_values']))
                    all_light[ds_idx].append(np.array(age_data[age]['light_values']))
                else:
                    all_dark[ds_idx].append(np.array([]))
                    all_light[ds_idx].append(np.array([]))

            ds = datasets[ds_idx]
            dark_arr = ds.get(f'{key_base}_dark_means', np.array([]))
            light_arr = ds.get(f'{key_base}_light_means', np.array([]))
            all_dark[ds_idx].append(dark_arr[~np.isnan(dark_arr)] if len(dark_arr) > 0 else np.array([]))
            all_light[ds_idx].append(light_arr[~np.isnan(light_arr)] if len(light_arr) > 0 else np.array([]))

        fig = Figure(figsize=(max(11, n_groups * 0.8 + 2), 8.5), dpi=150,
                     facecolor=self.BG_COLOR)
        fig.suptitle(f"Age Means: {metric_name}",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.97)

        gs = GridSpec(2, 1, figure=fig, hspace=0.35,
                      left=0.07, right=0.95, top=0.91, bottom=0.08)

        error_color = '#000000' if self.light_mode else '#ffffff'
        error_kw = {'elinewidth': 1.0, 'capthick': 1.0, 'ecolor': error_color}

        phase_info = [
            ('Dark Phase', 'dark', all_dark),
            ('Light Phase', 'light', all_light)
        ]
        axes = []
        x_base = np.arange(n_groups)
        total_width = 0.8
        bar_width = total_width / n_datasets

        # --- Pass 1: Draw bars and scatter points ---
        for row_idx, (phase, phase_key, all_phase) in enumerate(phase_info):
            ax = fig.add_subplot(gs[row_idx])
            ax.set_facecolor(self.BG_COLOR)
            axes.append(ax)

            dot_color = '#000000' if self.light_mode else '#ffffff'

            for ds_idx, ds in enumerate(datasets):
                color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
                label = self._get_dataset_label(ds) if row_idx == 0 else None

                means = []
                sems = []
                for g in range(n_groups):
                    arr = all_phase[ds_idx][g]
                    if len(arr) > 0:
                        means.append(np.mean(arr))
                        sems.append(np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0)
                    else:
                        means.append(0)
                        sems.append(0)

                offset = (ds_idx - n_datasets / 2 + 0.5) * bar_width
                ax.bar(x_base + offset, means, bar_width * 0.9,
                       yerr=sems, capsize=2, color=color, alpha=0.8,
                       label=label, error_kw=error_kw)

                for g in range(n_groups):
                    arr = all_phase[ds_idx][g]
                    if len(arr) > 0:
                        jitter = np.random.default_rng(ds_idx * 1000 + g).uniform(
                            -bar_width * 0.25, bar_width * 0.25, size=len(arr))
                        marker = self.DATASET_MARKERS[ds_idx % len(self.DATASET_MARKERS)]
                        ax.scatter(np.full(len(arr), x_base[g] + offset) + jitter,
                                  arr, color=dot_color, s=12, alpha=0.6,
                                  zorder=5, edgecolors='none', marker=marker)

            # Vertical separator before grand average
            ax.axvline(x=n_ages - 0.5, color='#888888', linestyle='--',
                       linewidth=1.2, alpha=0.9)

            ax.set_title(phase, fontsize=10, fontweight='bold',
                        color=self.TEXT_COLOR, pad=5)
            ax.set_xticks(x_base)
            ax.set_xticklabels(group_labels, fontsize=6, rotation=45, ha='right')
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

            if row_idx == 0:
                legend = ax.legend(loc='upper right', fontsize=6, framealpha=0.8,
                                   facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                                   labelcolor=self.TEXT_COLOR)
                legend.set_draggable(True)

            for spine in ax.spines.values():
                spine.set_color(self.GRID_COLOR)

        # --- Pass 2: FDR-corrected statistics across all ages × phases ---
        if self.show_statistics and n_datasets >= 2:
            dataset_labels = [self._get_dataset_label(ds) for ds in datasets]

            # Collect raw (uncorrected) p-values from all groups × both phases
            raw_entries = []
            for row_idx, (phase, phase_key, all_phase) in enumerate(phase_info):
                for g in range(n_groups):
                    raw_groups = [all_phase[ds_idx][g] for ds_idx in range(n_datasets)]
                    stats = self._compute_statistics(raw_groups, apply_correction=False)
                    grp_label = group_labels[g].replace('\n', ' ')

                    for (i, j), p_value in sorted(stats['pairwise'].items()):
                        raw_entries.append({
                            'row_idx': row_idx,
                            'phase_key': phase_key,
                            'g': g,
                            'grp_label': grp_label,
                            'i': i, 'j': j,
                            'raw_p': p_value,
                            'raw_groups': raw_groups,
                            'anova_p': stats.get('anova_p'),
                        })

            # Apply FDR correction to the full set of p-values
            if raw_entries:
                raw_p_values = [e['raw_p'] for e in raw_entries]
                corrected_q = self._apply_fdr_correction(raw_p_values)

                # Compute global max for bracket spacing per phase
                phase_global_max = {}
                for row_idx, (phase, phase_key, all_phase) in enumerate(phase_info):
                    gmax = 0
                    for g in range(n_groups):
                        for ds_idx in range(n_datasets):
                            arr = all_phase[ds_idx][g]
                            if len(arr) > 0:
                                h = max(np.max(arr),
                                        np.mean(arr) + (np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0))
                                gmax = max(gmax, h)
                    phase_global_max[row_idx] = gmax

                bracket_levels = {}

                for idx, entry in enumerate(raw_entries):
                    q_value = corrected_q[idx]
                    ri = entry['row_idx']
                    g = entry['g']
                    i, j = entry['i'], entry['j']
                    raw_groups = entry['raw_groups']

                    self.statistics_results.append({
                        'metric': metric_name,
                        'phase': entry['phase_key'].capitalize(),
                        'comparison': f'{dataset_labels[i]} vs {dataset_labels[j]} ({entry["grp_label"]})',
                        'group1': dataset_labels[i],
                        'group2': dataset_labels[j],
                        'group1_mean': np.mean(raw_groups[i]) if len(raw_groups[i]) > 0 else np.nan,
                        'group1_sem': np.std(raw_groups[i]) / np.sqrt(len(raw_groups[i])) if len(raw_groups[i]) > 1 else 0,
                        'group2_mean': np.mean(raw_groups[j]) if len(raw_groups[j]) > 0 else np.nan,
                        'group2_sem': np.std(raw_groups[j]) / np.sqrt(len(raw_groups[j])) if len(raw_groups[j]) > 1 else 0,
                        'test': 't-test (FDR)' if n_datasets == 2 else 'ANOVA + t-test (FDR)',
                        'p_value': entry['raw_p'],
                        'q_value': q_value,
                        'significance': self._get_significance_symbol(q_value),
                        'anova_p': entry['anova_p']
                    })

                    symbol = self._get_significance_symbol(q_value)
                    if symbol and symbol != 'ns':
                        ax = axes[ri]
                        gmax = phase_global_max[ri]
                        bracket_spacing = gmax * 0.08 if gmax > 0 else 0.1

                        level_key = (ri, g)
                        bracket_level = bracket_levels.get(level_key, 0)

                        offset_i = (i - n_datasets / 2 + 0.5) * bar_width
                        offset_j = (j - n_datasets / 2 + 0.5) * bar_width
                        local_max = max(
                            (np.max(raw_groups[k]) if len(raw_groups[k]) > 0 else 0)
                            for k in range(n_datasets)
                        )
                        y_pos = local_max * 1.05 + bracket_level * bracket_spacing
                        self._add_significance_bracket(
                            ax, x_base[g] + offset_i, x_base[g] + offset_j,
                            y_pos, symbol)
                        bracket_levels[level_key] = bracket_level + 1

        return fig

    def create_sleep_comparison_page(self, datasets: List[Dict[str, Any]]) -> Optional[Figure]:
        """
        Create sleep analysis comparison page for multiple datasets.

        Layout: 3 rows
        Row 0: Light Bout Count | Dark Bout Count | Sleep CTA
        Row 1: Light Time-in-Bouts | Dark Time-in-Bouts | L/D Ratio
        Row 2: 2x3 Bar Chart Grid (Total Sleep, Bout Count, Frag Index, Mean Duration, Sleep %)

        Returns:
            Figure or None if no datasets have sleep data
        """
        # Check if any datasets have sleep data
        datasets_with_sleep = [ds for ds in datasets if ds.get('has_sleep_data', False)]

        if not datasets_with_sleep:
            return None

        fig = Figure(figsize=(11, 10), dpi=150, facecolor=self.BG_COLOR)
        fig.suptitle("Sleep Analysis Comparison",
                     fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

        # Create grid: 3 rows
        # Row 0: Light histograms + CTA
        # Row 1: Dark histograms + L/D ratio
        # Row 2: 2x3 bar chart grid
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2],
                      hspace=0.35, wspace=0.25, left=0.06, right=0.94, top=0.93, bottom=0.05)

        # === ROW 0: Light Phase ===
        # Light Bout Count Histogram
        ax_light_count = fig.add_subplot(gs[0, 0])
        ax_light_count.set_facecolor(self.BG_COLOR)
        self._draw_overlaid_bout_histogram(ax_light_count, datasets_with_sleep, weighted=False, phase='light')

        # Light Time-in-Bouts Histogram
        ax_light_time = fig.add_subplot(gs[0, 1])
        ax_light_time.set_facecolor(self.BG_COLOR)
        self._draw_overlaid_bout_histogram(ax_light_time, datasets_with_sleep, weighted=True, phase='light')

        # Sleep CTA Comparison (36hr view)
        ax_cta = fig.add_subplot(gs[0, 2])
        ax_cta.set_facecolor(self.BG_COLOR)
        self._draw_sleep_cta_comparison(ax_cta, datasets)

        # === ROW 1: Dark Phase ===
        # Dark Bout Count Histogram
        ax_dark_count = fig.add_subplot(gs[1, 0])
        ax_dark_count.set_facecolor(self.BG_COLOR)
        self._draw_overlaid_bout_histogram(ax_dark_count, datasets_with_sleep, weighted=False, phase='dark')

        # Dark Time-in-Bouts Histogram
        ax_dark_time = fig.add_subplot(gs[1, 1])
        ax_dark_time.set_facecolor(self.BG_COLOR)
        self._draw_overlaid_bout_histogram(ax_dark_time, datasets_with_sleep, weighted=True, phase='dark')

        # L/D Ratio comparison
        ax_ld = fig.add_subplot(gs[1, 2])
        ax_ld.set_facecolor(self.BG_COLOR)
        self._draw_sleep_ld_ratio_comparison(ax_ld, datasets_with_sleep)

        # === ROW 2: Bar chart grid ===
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        gs_bars = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[2, :],
                                          wspace=0.3, hspace=0.4)

        # Bar 1: Total Sleep (minutes)
        ax_bar1 = fig.add_subplot(gs_bars[0, 0])
        ax_bar1.set_facecolor(self.BG_COLOR)
        self._draw_sleep_bar_comparison(ax_bar1, datasets_with_sleep, 'sleep_total_minutes',
                                        'Total Sleep (min)', show_legend=True)

        # Bar 2: Bout Count
        ax_bar2 = fig.add_subplot(gs_bars[0, 1])
        ax_bar2.set_facecolor(self.BG_COLOR)
        self._draw_sleep_bar_comparison(ax_bar2, datasets_with_sleep, 'sleep_bout_count',
                                        'Bout Count', show_legend=False)

        # Bar 3: Fragmentation Index
        ax_bar3 = fig.add_subplot(gs_bars[0, 2])
        ax_bar3.set_facecolor(self.BG_COLOR)
        self._draw_sleep_bar_comparison(ax_bar3, datasets_with_sleep, 'sleep_frag_index',
                                        'Frag Index', show_legend=False)

        # Bar 4: Mean Bout Duration
        ax_bar4 = fig.add_subplot(gs_bars[1, 0])
        ax_bar4.set_facecolor(self.BG_COLOR)
        self._draw_sleep_bar_comparison(ax_bar4, datasets_with_sleep, 'sleep_mean_duration',
                                        'Mean Duration (min)', show_legend=False)

        # Bar 5: Sleep %
        ax_bar5 = fig.add_subplot(gs_bars[1, 1])
        ax_bar5.set_facecolor(self.BG_COLOR)
        self._draw_sleep_bar_comparison(ax_bar5, datasets_with_sleep, 'sleep_percent_time',
                                        'Sleep %', show_legend=False)

        # Bar 6: empty or additional metric
        ax_bar6 = fig.add_subplot(gs_bars[1, 2])
        ax_bar6.set_facecolor(self.BG_COLOR)
        ax_bar6.axis('off')  # Empty slot

        return fig

    def _draw_overlaid_bout_histogram(self, ax, datasets: List[Dict[str, Any]], weighted: bool = False,
                                       phase: str = 'all'):
        """
        Draw overlaid bout duration histograms from multiple datasets.

        Uses per-animal normalization (mean histogram per animal) with per-dataset
        mean line and SEM shading.

        Args:
            ax: Matplotlib axes
            datasets: List of dataset dicts with sleep data
            weighted: If True, show time-in-bouts (weighted by duration); if False, show counts
            phase: 'light', 'dark', or 'all' - which phase to plot
        """
        bin_width = 5.0
        fixed_max_dur = 120.0  # Fixed max duration for consistent bins
        bins = np.arange(0, fixed_max_dur + bin_width, bin_width)
        n_bins = len(bins) - 1
        bin_centers = (bins[:-1] + bins[1:]) / 2

        has_data = False

        # Draw histogram line with SEM for each dataset
        for ds_idx, ds in enumerate(datasets):
            color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)

            # Get durations for the specified phase
            if phase == 'light':
                durations = ds.get('sleep_bout_durations_light', np.array([]))
            elif phase == 'dark':
                durations = ds.get('sleep_bout_durations_dark', np.array([]))
            else:  # 'all'
                light_dur = ds.get('sleep_bout_durations_light', np.array([]))
                dark_dur = ds.get('sleep_bout_durations_dark', np.array([]))
                durations = np.concatenate([light_dur, dark_dur]) if len(light_dur) > 0 or len(dark_dur) > 0 else np.array([])

            if len(durations) == 0:
                continue

            has_data = True

            # Get number of animals from the dataset
            n_animals = ds.get('sleep_n_animals', 1)

            # Compute histogram (normalized by number of animals for per-animal average)
            if weighted:
                # Time-weighted: sum durations in each bin, then divide by n_animals
                hist_vals = np.zeros(n_bins)
                for dur in durations:
                    bin_idx = min(np.searchsorted(bins[1:], dur), n_bins - 1)
                    hist_vals[bin_idx] += dur
                hist_vals = hist_vals / n_animals  # Per-animal average
            else:
                # Count histogram, normalized by n_animals
                hist_vals, _ = np.histogram(durations, bins=bins)
                hist_vals = hist_vals.astype(float) / n_animals  # Per-animal average

            # Plot as continuous line
            ax.plot(bin_centers, hist_vals, color=color, linewidth=1.5, label=label, zorder=3, alpha=0.9)

            # Add light shading under the curve
            ax.fill_between(bin_centers, 0, hist_vals, color=color, alpha=0.2, zorder=2)

        if not has_data:
            ax.text(0.5, 0.5, "No bout data", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # Styling
        phase_label = phase.capitalize() if phase != 'all' else ''
        title_base = 'Time in Bouts' if weighted else 'Bout Count'
        title = f'{phase_label} {title_base}' if phase_label else title_base
        ax.set_title(title, fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)
        ax.set_xlabel('Duration (min)', fontsize=7, color=self.TEXT_COLOR)
        ylabel = 'Time (min/animal)' if weighted else 'Count/animal'
        ax.set_ylabel(ylabel, fontsize=7, color=self.TEXT_COLOR)
        ax.set_xlim(0, fixed_max_dur)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        # Legend
        legend = ax.legend(loc='upper right', fontsize=5, facecolor=self.BG_COLOR,
                          edgecolor=self.GRID_COLOR, labelcolor=self.TEXT_COLOR)
        if legend:
            legend.set_draggable(True)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_sleep_cta_comparison(self, ax, datasets: List[Dict[str, Any]]):
        """
        Draw overlaid Sleep CTA comparison (using Sleeping % metric if available).

        Args:
            ax: Matplotlib axes
            datasets: List of dataset dicts
        """
        # Light/dark phase background (36hr)
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)

        x_hours_36h = np.arange(2160) / 60

        has_data = False
        for ds_idx, ds in enumerate(datasets):
            color = self.DATASET_COLORS[ds_idx % len(self.DATASET_COLORS)]
            label = self._get_dataset_label(ds)

            # Try to get Sleeping % CTA
            grand_cta = ds.get('Sleeping_pct_grand_cta', None)
            grand_sem = ds.get('Sleeping_pct_grand_sem', None)

            if grand_cta is None or len(grand_cta) != self.MINUTES_PER_DAY:
                continue

            has_data = True

            # Apply smoothing
            smoothed_cta = self._smooth_data(grand_cta)

            # Extend to 36 hours
            cta_36h = np.concatenate([smoothed_cta, smoothed_cta[:720]])

            # Plot mean line
            ax.plot(x_hours_36h, cta_36h, color=color, linewidth=1.2, label=label, alpha=0.9)

            # Plot SEM band
            if grand_sem is not None and len(grand_sem) == self.MINUTES_PER_DAY:
                smoothed_sem = self._smooth_data(grand_sem)
                sem_36h = np.concatenate([smoothed_sem, smoothed_sem[:720]])
                ax.fill_between(x_hours_36h, cta_36h - sem_36h, cta_36h + sem_36h,
                               color=color, alpha=self.SEM_ALPHA)

        if not has_data:
            ax.text(0.5, 0.5, "No Sleeping % CTA data", ha='center', va='center',
                    color=self.TEXT_COLOR, fontsize=10, transform=ax.transAxes)

        # Styling
        ax.set_title('Sleep CTA Comparison', fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36'], fontsize=7)
        ax.set_ylabel('Sleeping %', fontsize=7, color=self.TEXT_COLOR)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        if has_data:
            legend = ax.legend(loc='upper right', fontsize=5, facecolor=self.BG_COLOR,
                              edgecolor=self.GRID_COLOR, labelcolor=self.TEXT_COLOR)
            legend.set_draggable(True)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_sleep_bar_comparison(self, ax, datasets: List[Dict[str, Any]], metric_key: str,
                                    title: str, show_legend: bool = False):
        """
        Draw bar chart comparison for a sleep metric across datasets.

        Args:
            ax: Matplotlib axes
            datasets: List of dataset dicts with sleep data
            metric_key: Base key for the metric (e.g., 'sleep_total_minutes')
            title: Chart title
            show_legend: Whether to show legend
        """
        n_datasets = len(datasets)

        # Collect data for each dataset
        light_means = []
        light_sems = []
        dark_means = []
        dark_sems = []
        light_raw_data = []
        dark_raw_data = []

        for ds in datasets:
            light_arr = ds.get(f'{metric_key}_light', np.array([]))
            dark_arr = ds.get(f'{metric_key}_dark', np.array([]))

            # Store raw arrays for statistics
            light_raw_data.append(light_arr if len(light_arr) > 0 else np.array([]))
            dark_raw_data.append(dark_arr if len(dark_arr) > 0 else np.array([]))

            if len(light_arr) > 0:
                valid_light = light_arr[~np.isnan(light_arr)]
                light_means.append(np.mean(valid_light) if len(valid_light) > 0 else 0)
                light_sems.append(np.std(valid_light) / np.sqrt(len(valid_light)) if len(valid_light) > 1 else 0)
            else:
                light_means.append(0)
                light_sems.append(0)

            if len(dark_arr) > 0:
                valid_dark = dark_arr[~np.isnan(dark_arr)]
                dark_means.append(np.mean(valid_dark) if len(valid_dark) > 0 else 0)
                dark_sems.append(np.std(valid_dark) / np.sqrt(len(valid_dark)) if len(valid_dark) > 1 else 0)
            else:
                dark_means.append(0)
                dark_sems.append(0)

        # Draw bars grouped by dataset
        bar_width = 0.35
        x_positions = np.arange(n_datasets)

        error_color = '#000000' if self.light_mode else '#ffffff'
        error_kw = {'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': error_color}

        bars_dark = ax.bar(x_positions - bar_width/2, dark_means, bar_width,
                          yerr=dark_sems, capsize=3, color='#555555', label='Dark', alpha=0.8,
                          error_kw=error_kw)
        bars_light = ax.bar(x_positions + bar_width/2, light_means, bar_width,
                           yerr=light_sems, capsize=3, color='#ffcc00', label='Light', alpha=0.8,
                           error_kw=error_kw)

        # Color-code bar edges by dataset
        for i, (bd, bl) in enumerate(zip(bars_dark, bars_light)):
            color = self.DATASET_COLORS[i % len(self.DATASET_COLORS)]
            bd.set_edgecolor(color)
            bd.set_linewidth(2)
            bl.set_edgecolor(color)
            bl.set_linewidth(2)

        # X-axis labels
        ax.set_xticks(x_positions)
        xlabels = []
        for ds in datasets:
            label = self._get_dataset_label(ds)
            if len(label) > 10:
                label = label[:8] + '..'
            xlabels.append(label)
        ax.set_xticklabels(xlabels, fontsize=5, rotation=15, ha='right')

        # Add statistics if enabled
        if self.show_statistics and n_datasets >= 2:
            dark_stats = self._compute_statistics(dark_raw_data)
            light_stats = self._compute_statistics(light_raw_data)

            # Store statistics results for sleep metrics
            self._store_sleep_statistics_result(title, datasets, dark_stats, light_stats,
                                                dark_means, dark_sems, light_means, light_sems)

            # Get max bar height for positioning brackets
            max_dark = max(dark_means[i] + dark_sems[i] for i in range(n_datasets)) if dark_means else 0
            max_light = max(light_means[i] + light_sems[i] for i in range(n_datasets)) if light_means else 0
            max_height = max(max_dark, max_light)

            # Add brackets for significant comparisons
            bracket_y = max_height * 1.05
            bracket_level = 0
            for (i, j), p_value in sorted(dark_stats['pairwise'].items()):
                symbol = self._get_significance_symbol(p_value)
                if symbol and symbol != 'ns':
                    y_pos = bracket_y + bracket_level * max_height * 0.15
                    self._add_significance_bracket(ax, x_positions[i] - bar_width/2,
                                                    x_positions[j] - bar_width/2, y_pos, symbol)
                    bracket_level += 1

        # Styling
        ax.set_title(title, fontsize=7, fontweight='bold', color=self.TEXT_COLOR, pad=2)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=5)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

        if show_legend:
            legend = ax.legend(loc='upper left', fontsize=5, framealpha=0.9,
                              facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                              labelcolor=self.TEXT_COLOR)
            legend.set_draggable(True)

    def _draw_sleep_ld_ratio_comparison(self, ax, datasets: List[Dict[str, Any]]):
        """
        Draw Light/Dark ratio comparison bar chart.

        L/D Ratio = Total Sleep Light / Total Sleep Dark
        """
        n_datasets = len(datasets)

        ratios = []
        ratio_sems = []
        raw_ratios = []

        for ds in datasets:
            light_arr = ds.get('sleep_total_minutes_light', np.array([]))
            dark_arr = ds.get('sleep_total_minutes_dark', np.array([]))

            if len(light_arr) > 0 and len(dark_arr) > 0:
                # Compute L/D ratio per animal, then average
                per_animal_ratios = []
                for l, d in zip(light_arr, dark_arr):
                    if d > 0 and not np.isnan(l) and not np.isnan(d):
                        per_animal_ratios.append(l / d)
                    elif l > 0 and (d == 0 or np.isnan(d)):
                        per_animal_ratios.append(np.nan)  # Infinite ratio

                valid_ratios = [r for r in per_animal_ratios if not np.isnan(r) and r != float('inf')]
                if valid_ratios:
                    ratios.append(np.mean(valid_ratios))
                    ratio_sems.append(np.std(valid_ratios) / np.sqrt(len(valid_ratios)) if len(valid_ratios) > 1 else 0)
                    raw_ratios.append(np.array(valid_ratios))
                else:
                    ratios.append(0)
                    ratio_sems.append(0)
                    raw_ratios.append(np.array([]))
            else:
                ratios.append(0)
                ratio_sems.append(0)
                raw_ratios.append(np.array([]))

        # Draw bars
        x_positions = np.arange(n_datasets)
        bar_width = 0.6

        error_color = '#000000' if self.light_mode else '#ffffff'
        error_kw = {'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': error_color}

        for i, (ratio, sem) in enumerate(zip(ratios, ratio_sems)):
            color = self.DATASET_COLORS[i % len(self.DATASET_COLORS)]
            ax.bar(x_positions[i], ratio, bar_width, yerr=sem, capsize=3,
                   color=color, alpha=0.8, error_kw=error_kw)

        # Add reference line at y=1 (equal light/dark)
        ax.axhline(y=1, color=self.GRID_COLOR, linestyle='--', linewidth=1, alpha=0.7)

        # X-axis labels
        ax.set_xticks(x_positions)
        xlabels = []
        for ds in datasets:
            label = self._get_dataset_label(ds)
            if len(label) > 10:
                label = label[:8] + '..'
            xlabels.append(label)
        ax.set_xticklabels(xlabels, fontsize=5, rotation=15, ha='right')

        # Add statistics if enabled
        if self.show_statistics and n_datasets >= 2:
            ratio_stats = self._compute_statistics(raw_ratios)

            # Store statistics
            dataset_labels = [self._get_dataset_label(ds) for ds in datasets]
            for (i, j), p_value in ratio_stats['pairwise'].items():
                self.statistics_results.append({
                    'metric': 'Sleep L/D Ratio',
                    'phase': 'Combined',
                    'comparison': f'{dataset_labels[i]} vs {dataset_labels[j]}',
                    'group1': dataset_labels[i],
                    'group2': dataset_labels[j],
                    'group1_mean': ratios[i],
                    'group1_sem': ratio_sems[i],
                    'group2_mean': ratios[j],
                    'group2_sem': ratio_sems[j],
                    'test': 't-test' if n_datasets == 2 else 't-test (Bonferroni)',
                    'p_value': p_value,
                    'significance': self._get_significance_symbol(p_value),
                    'anova_p': ratio_stats.get('anova_p')
                })

            # Add significance brackets
            max_height = max(ratios[i] + ratio_sems[i] for i in range(n_datasets)) if ratios else 1
            bracket_y = max_height * 1.05
            bracket_level = 0
            for (i, j), p_value in sorted(ratio_stats['pairwise'].items()):
                symbol = self._get_significance_symbol(p_value)
                if symbol and symbol != 'ns':
                    y_pos = bracket_y + bracket_level * max_height * 0.15
                    self._add_significance_bracket(ax, x_positions[i], x_positions[j], y_pos, symbol)
                    bracket_level += 1

        # Styling
        ax.set_title('L/D Ratio', fontsize=7, fontweight='bold', color=self.TEXT_COLOR, pad=2)
        ax.set_ylabel('Light / Dark', fontsize=6, color=self.TEXT_COLOR)
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=5)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _store_sleep_statistics_result(self, metric_name: str, datasets: List[Dict],
                                        dark_stats: dict, light_stats: dict,
                                        dark_means: List[float], dark_sems: List[float],
                                        light_means: List[float], light_sems: List[float]):
        """Store sleep statistics results for export."""
        n_datasets = len(datasets)
        dataset_labels = [self._get_dataset_label(ds) for ds in datasets]

        # Store dark phase results
        for (i, j), p_value in dark_stats['pairwise'].items():
            self.statistics_results.append({
                'metric': f'Sleep {metric_name}',
                'phase': 'Dark',
                'comparison': f'{dataset_labels[i]} vs {dataset_labels[j]}',
                'group1': dataset_labels[i],
                'group2': dataset_labels[j],
                'group1_mean': dark_means[i],
                'group1_sem': dark_sems[i],
                'group2_mean': dark_means[j],
                'group2_sem': dark_sems[j],
                'test': 't-test' if n_datasets == 2 else 't-test (Bonferroni)',
                'p_value': p_value,
                'significance': self._get_significance_symbol(p_value),
                'anova_p': dark_stats.get('anova_p')
            })

        # Store light phase results
        for (i, j), p_value in light_stats['pairwise'].items():
            self.statistics_results.append({
                'metric': f'Sleep {metric_name}',
                'phase': 'Light',
                'comparison': f'{dataset_labels[i]} vs {dataset_labels[j]}',
                'group1': dataset_labels[i],
                'group2': dataset_labels[j],
                'group1_mean': light_means[i],
                'group1_sem': light_sems[i],
                'group2_mean': light_means[j],
                'group2_sem': light_sems[j],
                'test': 't-test' if n_datasets == 2 else 't-test (Bonferroni)',
                'p_value': p_value,
                'significance': self._get_significance_symbol(p_value),
                'anova_p': light_stats.get('anova_p')
            })


def load_consolidated_npz(npz_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a consolidated NPZ file for comparison.

    Args:
        npz_path: Path to consolidated NPZ file

    Returns:
        Dictionary with all data from the NPZ file, or None if invalid
    """
    try:
        data = np.load(npz_path, allow_pickle=True)

        result = {
            'filename': str(npz_path).split('\\')[-1].split('/')[-1]
        }

        # Load consolidation metadata
        if 'consolidation_metadata' in data:
            result['consolidation_metadata'] = json.loads(str(data['consolidation_metadata']))
        else:
            # Not a consolidated file
            return None

        # Load animal metadata
        if 'animal_metadata' in data:
            result['animal_metadata'] = json.loads(str(data['animal_metadata']))

        # Load metric names
        if 'metric_names' in data:
            result['metric_names'] = list(data['metric_names'])
        else:
            result['metric_names'] = []

        # Load all metric arrays
        for key in data.keys():
            if key not in ['consolidation_metadata', 'animal_metadata', 'metric_names',
                           'sleep_parameters_json']:
                result[key] = data[key]

        # Load sleep data if present
        if 'sleep_n_animals' in data:
            result['has_sleep_data'] = True
            result['sleep_n_animals'] = int(data['sleep_n_animals'])

            # Load sleep parameters
            if 'sleep_parameters_json' in data:
                result['sleep_parameters'] = json.loads(str(data['sleep_parameters_json']))
            else:
                result['sleep_parameters'] = {'threshold': 0.5, 'bin_width': 5.0}

            # Sleep arrays are already loaded in the generic loop above
        else:
            result['has_sleep_data'] = False

        return result

    except Exception as e:
        print(f"Error loading consolidated NPZ {npz_path}: {e}")
        return None
