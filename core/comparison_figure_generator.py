"""
Comparison figure generator for comparing multiple consolidated datasets.

Creates matplotlib figures for comparing consolidated NPZ files:
- Summary page with datasets being compared
- Overlay CTA plots showing grand means from each dataset with SEM bands (36hr view)
- Bar chart comparisons for dark/light means
"""

import numpy as np
import matplotlib.pyplot as plt
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

    def generate_all_pages(self, datasets: List[Dict[str, Any]]) -> List[Tuple[str, Figure]]:
        """
        Generate all comparison figure pages.

        Args:
            datasets: List of loaded consolidated NPZ data dicts

        Returns:
            List of (title, figure) tuples
        """
        # Clear statistics from previous runs
        self.statistics_results = []

        pages = []

        if not datasets:
            return pages

        # Page 1: Summary comparing all datasets
        fig_summary = self.create_summary_page(datasets)
        pages.append(("Comparison Summary", fig_summary))

        # Get common metrics across all datasets
        common_metrics = self._get_common_metrics(datasets)

        # Pages 2+: Overlay CTA comparison plots (6 metrics per page in 2x3 grid)
        metrics_per_page = 6
        for page_idx in range(0, len(common_metrics), metrics_per_page):
            page_metrics = common_metrics[page_idx:page_idx + metrics_per_page]
            page_num = page_idx // metrics_per_page + 1
            fig_cta = self.create_cta_comparison_page(datasets, page_metrics, page_num)
            pages.append((f"CTA Comparison {page_num}", fig_cta))

        # Final page(s): Bar chart comparisons (show ALL metrics)
        # Use multiple pages if needed, 12 metrics per page
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

        # Statistics summary page (if statistics are enabled and we have results)
        if self.show_statistics and len(datasets) >= 2:
            fig_stats = self.create_statistics_summary_page(datasets)
            pages.append(("Statistical Analysis", fig_stats))

        # Sleep analysis comparison page (if any datasets have sleep data)
        fig_sleep = self.create_sleep_comparison_page(datasets)
        if fig_sleep is not None:
            pages.append(("Sleep Analysis Comparison", fig_sleep))

        return pages

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
        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

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
        n_cols = len(headers)
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
        n_metrics = len(metrics)
        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

        n_datasets = len(datasets)
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
        n_metrics = len(metrics)

        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

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

    def _compute_statistics(self, data_groups: List[np.ndarray]) -> dict:
        """
        Compute statistical comparisons between groups.

        Args:
            data_groups: List of arrays, one per group (each array contains values for all animals in that group)

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

                # Pairwise t-tests with Bonferroni correction
                n_comparisons = len(clean_groups) * (len(clean_groups) - 1) // 2
                for idx1, (i, g1) in enumerate(zip(group_indices, clean_groups)):
                    for idx2, (j, g2) in enumerate(zip(group_indices[idx1+1:], clean_groups[idx1+1:])):
                        j = group_indices[idx1 + 1 + idx2]
                        _, p_value = stats.ttest_ind(g1, g2, equal_var=False)
                        # Apply Bonferroni correction
                        results['pairwise'][(i, j)] = min(p_value * n_comparisons, 1.0)

        return results

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
        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

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
        if n_datasets == 2:
            method_text = "Method: Welch's t-test (unequal variance)"
        else:
            method_text = "Method: One-way ANOVA + pairwise t-tests with Bonferroni correction"

        ax.text(0.0, y_pos, method_text, fontsize=9, color=self.TEXT_COLOR,
               fontweight='bold', transform=ax.transAxes)
        y_pos -= line_height * 1.5

        ax.text(0.0, y_pos, "Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant",
               fontsize=8, color=self.TEXT_COLOR, transform=ax.transAxes)
        y_pos -= line_height * 2

        # Table headers
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

                # Truncate comparison for display
                comparison = result['comparison']
                if len(comparison) > 28:
                    comparison = comparison[:25] + '...'

                # Row data
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

        fig = Figure(figsize=(11, 10), facecolor=self.BG_COLOR)
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
