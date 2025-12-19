"""
Figure generator module for behavioral analysis visualizations.

Creates matplotlib figures for:
- Animal summary/metadata page with data availability timeline
- Stacked daily traces with CTA below (3 metrics per page)
- Summary statistics bar charts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from typing import Dict, Any, List, Tuple


class FigureGenerator:
    """Generate matplotlib figures for behavioral analysis."""

    # Color palette for daily traces
    DAY_COLORS = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
    ]

    # Dark theme colors
    BG_COLOR = '#2d2d2d'
    TEXT_COLOR = '#ffffff'
    GRID_COLOR = '#4d4d4d'
    # For phase shading: use yellow for light phase to match bar charts
    DARK_PHASE_COLOR = '#2d2d2d'   # Same as background (no visible shading)
    LIGHT_PHASE_COLOR = '#4a4a2a'  # Subtle yellow tint for light phase

    def __init__(self):
        pass

    def generate_all_pages(self, animal_id: str, animal_data: Dict[str, Any]) -> List[Tuple[str, Figure]]:
        """
        Generate all figure pages for an animal.

        Args:
            animal_id: Animal identifier
            animal_data: Analysis results for this animal

        Returns:
            List of (title, figure) tuples
        """
        pages = []

        # Page 1: Summary with data availability timeline
        fig_summary = self.create_summary_page(animal_id, animal_data)
        pages.append(("Summary", fig_summary))

        # Pages 2+: Stacked daily traces with CTA (3 metrics per page)
        metrics = animal_data.get('metrics', {})
        metric_names = list(metrics.keys())
        metrics_per_page = 3

        for page_idx in range(0, len(metric_names), metrics_per_page):
            page_metrics = metric_names[page_idx:page_idx + metrics_per_page]
            page_num = page_idx // metrics_per_page + 1
            fig_traces = self.create_stacked_traces_page(animal_id, animal_data, page_metrics, page_num)
            pages.append((f"Traces {page_num}", fig_traces))

        # Final page: Statistics
        fig_stats = self.create_statistics_page(animal_id, animal_data)
        pages.append(("Statistics", fig_stats))

        # Sleep Analysis page (if sleeping data exists)
        if 'sleep_analysis' in animal_data and animal_data['sleep_analysis']:
            fig_sleep = self.create_sleep_analysis_page(animal_id, animal_data)
            pages.append(("Sleep Analysis", fig_sleep))

        return pages

    def create_summary_page(self, animal_id: str, animal_data: Dict[str, Any]) -> Figure:
        """Create the animal summary/metadata page with data availability timeline."""
        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

        # Title
        fig.suptitle(f"Animal Summary: {animal_id}", fontsize=16, fontweight='bold',
                    color=self.TEXT_COLOR, y=0.96)

        metadata = animal_data.get('metadata', {})
        quality = animal_data.get('quality', {})
        n_days = animal_data.get('n_days', 1)

        # Create grid: left side for info/table, right side for timeline
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1], height_ratios=[1, 1.5],
                     hspace=0.25, wspace=0.15, left=0.06, right=0.96, top=0.90, bottom=0.06)

        # === Animal Information Box (top-left) ===
        ax_info = fig.add_subplot(gs[0, 0])
        ax_info.set_facecolor(self.BG_COLOR)
        ax_info.axis('off')

        cagemate = metadata.get('companion', None)
        if cagemate is None:
            cagemate_str = 'None'
        elif isinstance(cagemate, list):
            cagemate_str = ', '.join(cagemate)
        else:
            cagemate_str = str(cagemate)

        info_text = [
            f"Animal ID:     {metadata.get('animal_id', 'Unknown')}",
            f"Genotype:      {metadata.get('genotype', 'Unknown')}",
            f"Sex:           {metadata.get('sex', 'Unknown')}",
            f"Cohort:        {metadata.get('cohort', 'Unknown')}",
            f"Cage ID:       {metadata.get('cage_id', 'Unknown')}",
            f"Cagemate:      {cagemate_str}",
        ]

        ax_info.text(0.02, 0.95, "Animal Information", fontsize=12, fontweight='bold',
                    transform=ax_info.transAxes, color=self.TEXT_COLOR, va='top')

        for i, line in enumerate(info_text):
            ax_info.text(0.02, 0.78 - i * 0.13, line, fontsize=10,
                        transform=ax_info.transAxes, color=self.TEXT_COLOR,
                        family='monospace', va='top')

        # === Recording Information Box (top-right) ===
        ax_rec = fig.add_subplot(gs[0, 1])
        ax_rec.set_facecolor(self.BG_COLOR)
        ax_rec.axis('off')

        start_time = metadata.get('start_time', 'Unknown')
        end_time = metadata.get('end_time', 'Unknown')
        if len(start_time) > 19:
            start_time = start_time[:19]
        if len(end_time) > 19:
            end_time = end_time[:19]

        rec_text = [
            f"Start:         {start_time}",
            f"End:           {end_time}",
            f"Days Analyzed: {metadata.get('n_days_analyzed', 0)}",
            f"Total Minutes: {metadata.get('total_minutes', 0):,}",
            f"ZT0 at Minute: {metadata.get('zt0_minute', 0)}",
        ]

        ax_rec.text(0.02, 0.95, "Recording Information", fontsize=12, fontweight='bold',
                   transform=ax_rec.transAxes, color=self.TEXT_COLOR, va='top')

        for i, line in enumerate(rec_text):
            ax_rec.text(0.02, 0.78 - i * 0.13, line, fontsize=10,
                       transform=ax_rec.transAxes, color=self.TEXT_COLOR,
                       family='monospace', va='top')

        # === Data Quality Table (bottom-left) ===
        ax_quality = fig.add_subplot(gs[1, 0])
        ax_quality.set_facecolor(self.BG_COLOR)
        ax_quality.axis('off')

        ax_quality.text(0.02, 0.98, "Data Quality Summary", fontsize=12, fontweight='bold',
                       transform=ax_quality.transAxes, color=self.TEXT_COLOR, va='top')

        # Table header - fixed width columns
        header = f"{'Metric':<22}{'Coverage':>9}{'Missing':>9}  {'Quality':>7}"
        ax_quality.text(0.02, 0.90, header, fontsize=9, fontweight='bold',
                       transform=ax_quality.transAxes, color=self.TEXT_COLOR,
                       family='monospace', va='top')

        # Separator line
        ax_quality.text(0.02, 0.86, "-" * 52, fontsize=9,
                       transform=ax_quality.transAxes, color=self.GRID_COLOR,
                       family='monospace', va='top')

        # Table rows
        y_pos = 0.82
        for metric_name, q_info in quality.items():
            coverage = q_info['coverage']
            missing = q_info['missing']
            rating = q_info['rating']

            # Color code rating
            if rating == 'OK':
                rating_color = '#27ae60'  # green
            elif rating == 'WARN':
                rating_color = '#f39c12'  # orange
            else:
                rating_color = '#e74c3c'  # red

            # Truncate metric name
            short_name = metric_name[:20] if len(metric_name) > 20 else metric_name

            row = f"{short_name:<22}{coverage:>8.1f}%{missing:>9}"
            ax_quality.text(0.02, y_pos, row, fontsize=8,
                           transform=ax_quality.transAxes, color=self.TEXT_COLOR,
                           family='monospace', va='top')

            # Add colored rating aligned
            ax_quality.text(0.78, y_pos, rating, fontsize=8,
                           transform=ax_quality.transAxes, color=rating_color,
                           family='monospace', va='top', fontweight='bold')

            y_pos -= 0.065

        # === Data Availability Timeline (bottom-right) ===
        ax_timeline = fig.add_subplot(gs[1, 1])
        ax_timeline.set_facecolor(self.BG_COLOR)

        self._draw_data_timeline(ax_timeline, animal_data, n_days)

        return fig

    def _draw_data_timeline(self, ax, animal_data: Dict[str, Any], n_days: int):
        """Draw data availability timeline showing green/red for present/missing data."""
        metrics = animal_data.get('metrics', {})
        metric_names = list(metrics.keys())
        n_metrics = len(metric_names)

        if n_metrics == 0:
            ax.text(0.5, 0.5, "No metrics", ha='center', va='center',
                   color=self.TEXT_COLOR, fontsize=12)
            ax.axis('off')
            return

        # Each metric gets a horizontal bar
        bar_height = 0.8
        total_minutes = n_days * 1440

        for idx, metric_name in enumerate(metric_names):
            y_pos = n_metrics - idx - 1
            metric_data = metrics[metric_name]
            daily_data = metric_data.get('daily_data', [])

            if not daily_data:
                continue

            # Concatenate all days
            all_data = np.concatenate(daily_data)

            # Downsample for visualization (1 pixel per ~10 minutes)
            chunk_size = max(1, len(all_data) // 200)
            n_chunks = len(all_data) // chunk_size

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = start_idx + chunk_size
                chunk = all_data[start_idx:end_idx]

                # Check if chunk has valid data
                valid_ratio = np.sum(~np.isnan(chunk)) / len(chunk)

                if valid_ratio > 0.5:
                    color = '#27ae60'  # green - data present
                else:
                    color = '#e74c3c'  # red - data missing

                x_start = chunk_idx / n_chunks * n_days
                x_width = 1 / n_chunks * n_days

                rect = Rectangle((x_start, y_pos + 0.1), x_width, bar_height,
                                 facecolor=color, edgecolor='none', alpha=0.8)
                ax.add_patch(rect)

            # Add metric label
            short_name = metric_name[:15] if len(metric_name) > 15 else metric_name
            ax.text(-0.05, y_pos + 0.5, short_name, ha='right', va='center',
                   fontsize=7, color=self.TEXT_COLOR, transform=ax.get_yaxis_transform())

        # Add day markers and light/dark shading
        for day in range(n_days):
            # Light phase (first half of day in ZT time) = light gray
            ax.axvspan(day, day + 0.5, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
            # Dark phase (second half of day in ZT time) = same as background
            ax.axvspan(day + 0.5, day + 1, facecolor=self.DARK_PHASE_COLOR, zorder=0)

            # Day label
            ax.axvline(day, color=self.GRID_COLOR, linewidth=0.5, alpha=0.5)
            ax.text(day + 0.5, -0.3, f'Day {day + 1}', ha='center', va='top',
                   fontsize=8, color=self.TEXT_COLOR)

        ax.set_xlim(0, n_days)
        ax.set_ylim(-0.5, n_metrics)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title('Data Availability Timeline', fontsize=10, fontweight='bold',
                    color=self.TEXT_COLOR, pad=10)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def create_stacked_traces_page(self, animal_id: str, animal_data: Dict[str, Any],
                                   page_metrics: List[str], page_num: int) -> Figure:
        """Create stacked daily traces page with CTA (with completeness on y2 axis)."""
        metrics = animal_data.get('metrics', {})
        n_days = animal_data.get('n_days', 1)
        n_metrics = len(page_metrics)

        # Figure with 3 columns (one per metric), 2 rows (traces + CTA with completeness on y2)
        fig = Figure(figsize=(11, 8.5), facecolor=self.BG_COLOR)

        fig.suptitle(f"Daily Traces & CTA: {animal_id} (Page {page_num})",
                    fontsize=14, fontweight='bold', color=self.TEXT_COLOR, y=0.98)

        # Create grid: 2 rows x 3 columns (traces and CTA with completeness curve)
        gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1.2],
                     hspace=0.15, wspace=0.25, left=0.06, right=0.94, top=0.92, bottom=0.08)

        for metric_idx, metric_name in enumerate(page_metrics):
            if metric_name not in metrics:
                continue

            metric_data = metrics[metric_name]
            daily_data = metric_data.get('daily_data', [])
            cta = metric_data.get('cta', np.array([]))
            cta_sem = metric_data.get('cta_sem', np.array([]))

            # === Top row: Stacked Daily Traces ===
            ax_traces = fig.add_subplot(gs[0, metric_idx])
            ax_traces.set_facecolor(self.BG_COLOR)

            if daily_data:
                self._draw_stacked_traces(ax_traces, daily_data, n_days, metric_name,
                                         show_ylabel=(metric_idx == 0))

            # === Bottom row: CTA with completeness curve on y2 axis ===
            ax_cta = fig.add_subplot(gs[1, metric_idx])
            ax_cta.set_facecolor(self.BG_COLOR)

            if len(cta) > 0:
                self._draw_cta_with_completeness(ax_cta, cta, cta_sem, daily_data,
                                                 metric_name=metric_name,
                                                 show_xlabel=True, show_y2_label=True)

        return fig

    def _draw_stacked_traces(self, ax, daily_data: List[np.ndarray], n_days: int,
                             metric_name: str, show_ylabel: bool = True):
        """Draw stacked daily traces where each day is normalized to 0-1 range.

        Shows 36 hours (ZT0-36) to better visualize light/dark transitions.
        Each day's data is normalized based on 1st-99th percentile of all days.
        """
        # Find global min/max for consistent scaling across all days
        all_values = np.concatenate(daily_data)
        valid_values = all_values[~np.isnan(all_values)]
        if len(valid_values) == 0:
            return

        data_min = np.percentile(valid_values, 1)
        data_max = np.percentile(valid_values, 99)
        data_range = data_max - data_min if data_max > data_min else 1

        # Create 36-hour x-axis (0-36 hours = 0-2160 minutes)
        # We'll show ZT0 to ZT36 (1.5 days)
        x_minutes_36h = np.arange(2160)  # 36 hours worth of minutes
        x_hours_36h = x_minutes_36h / 60

        # Plot each day stacked vertically
        for day_idx, day_values in enumerate(daily_data):
            # Create 36-hour trace by concatenating current day + first 12h of next day
            if day_idx < len(daily_data) - 1:
                # Use first 12 hours (720 min) of next day
                next_day_start = daily_data[day_idx + 1][:720]
                day_36h = np.concatenate([day_values, next_day_start])
            else:
                # For last day, repeat first 12 hours of same day
                day_36h = np.concatenate([day_values, day_values[:720]])

            # Normalize to 0-1 range, then offset by day index
            normalized = (day_36h - data_min) / data_range
            normalized = np.clip(normalized, 0, 1)  # Clip outliers
            y_offset = day_idx
            y_values = normalized * 0.85 + y_offset + 0.075  # Leave small margin

            color = self.DAY_COLORS[day_idx % len(self.DAY_COLORS)]
            ax.plot(x_hours_36h, y_values, color=color, linewidth=0.5, alpha=0.8)

            # Day label on left
            ax.text(-0.3, y_offset + 0.5, f'D{day_idx + 1}', ha='right', va='center',
                   fontsize=7, color=color, fontweight='bold')

        # Add light/dark shading for 36 hours
        # Light phase = light gray, Dark phase = same as background (invisible)
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)    # Light ZT0-12
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)    # Dark ZT12-24
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)   # Light ZT24-36

        # Formatting
        ax.set_xlim(0, 36)
        ax.set_ylim(0, n_days)
        ax.set_xticks([0, 12, 24, 36])
        ax.set_xticklabels([])  # Hide x labels, CTA below will show them
        ax.set_yticks([])

        # Title with metric name
        ax.set_title(metric_name, fontsize=9, color=self.TEXT_COLOR, fontweight='bold', pad=5)

        # Don't show "Days" ylabel - the D1, D2, D3 labels are sufficient
        # and having both causes overlap

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_completeness_bar(self, ax, daily_data: List[np.ndarray], show_xlabel: bool = True):
        """Draw a compact bar showing data completeness over 36 hours.

        Green = data present, Red = data missing.
        """
        # Concatenate all days and create 36-hour version
        all_data = np.concatenate(daily_data)

        # Extend to show 36 hours (add first 12h worth)
        first_12h = all_data[:720] if len(all_data) >= 720 else all_data
        all_data_36h = np.concatenate([all_data, first_12h])

        # Downsample to ~100 bins for visualization
        n_bins = 100
        bin_size = max(1, len(all_data_36h) // n_bins)

        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = min(start_idx + bin_size, len(all_data_36h))

            if start_idx >= len(all_data_36h):
                break

            chunk = all_data_36h[start_idx:end_idx]
            valid_ratio = np.sum(~np.isnan(chunk)) / len(chunk) if len(chunk) > 0 else 0

            # Color based on completeness
            if valid_ratio >= 0.95:
                color = '#27ae60'  # green - complete
            elif valid_ratio >= 0.5:
                color = '#f39c12'  # orange - partial
            else:
                color = '#e74c3c'  # red - missing

            x_start = bin_idx / n_bins * 36
            x_width = 36 / n_bins

            rect = Rectangle((x_start, 0), x_width, 1,
                             facecolor=color, edgecolor='none', alpha=0.8)
            ax.add_patch(rect)

        # Add light/dark shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)

        # Formatting
        ax.set_xlim(0, 36)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 12, 24, 36])

        if show_xlabel:
            ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36'], fontsize=7)
        else:
            ax.set_xticklabels([])

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)
            spine.set_linewidth(0.5)

    def _draw_cta_with_completeness(self, ax, cta: np.ndarray, cta_sem: np.ndarray,
                                     daily_data: List[np.ndarray], metric_name: str = '',
                                     show_xlabel: bool = True, show_y2_label: bool = False):
        """Draw CTA with SEM shading over 36 hours, with data completeness on y2 axis.

        The completeness curve is color-coded: green (100%), yellow (50%), red (0%).
        """
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.collections import LineCollection

        # Create 36-hour CTA by extending with first 12 hours
        cta_36h = np.concatenate([cta, cta[:720]])  # Add first 12h (720 min)
        sem_36h = np.concatenate([cta_sem, cta_sem[:720]])

        x_hours = np.arange(len(cta_36h)) / 60  # 0-36 hours

        # Apply smoothing (15-min rolling average)
        window = 15
        cta_smooth = pd.Series(cta_36h).rolling(window=window, min_periods=1, center=True).mean().values
        sem_smooth = pd.Series(sem_36h).rolling(window=window, min_periods=1, center=True).mean().values

        # Add light/dark shading for 36 hours (before plotting data)
        # Light phase = light gray, Dark phase = same as background (invisible)
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)    # Light ZT0-12
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)    # Dark ZT12-24
        ax.axvspan(24, 36, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)   # Light ZT24-36

        # Plot CTA with SEM shading
        ax.plot(x_hours, cta_smooth, color='#3daee9', linewidth=1.2, zorder=3)
        ax.fill_between(x_hours, cta_smooth - sem_smooth, cta_smooth + sem_smooth,
                       color='#3daee9', alpha=0.3, zorder=2)

        # === Create y2 axis for completeness curve ===
        ax2 = ax.twinx()

        # Calculate completeness over 36 hours
        if daily_data:
            # Concatenate all days
            all_data = np.concatenate(daily_data)
            # Extend to 36 hours
            first_12h = all_data[:720] if len(all_data) >= 720 else all_data
            all_data_36h = np.concatenate([all_data, first_12h])

            # Compute completeness in small windows (e.g., 15-minute bins)
            bin_size = 15  # minutes
            n_bins = len(all_data_36h) // bin_size
            completeness = np.zeros(n_bins)

            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size
                chunk = all_data_36h[start_idx:end_idx]
                completeness[i] = np.sum(~np.isnan(chunk)) / len(chunk) if len(chunk) > 0 else 0

            # Create x values for completeness (center of each bin)
            x_complete = (np.arange(n_bins) * bin_size + bin_size / 2) / 60  # hours

            # Smooth the completeness curve
            completeness_smooth = pd.Series(completeness).rolling(window=3, min_periods=1, center=True).mean().values

            # Create color-coded line using LineCollection
            # Colors: red (0%) -> yellow (50%) -> green (100%)
            cmap = LinearSegmentedColormap.from_list('completeness',
                                                     ['#e74c3c', '#f39c12', '#27ae60'])  # red, yellow, green

            # Create segments for LineCollection
            points = np.array([x_complete, completeness_smooth * 100]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color based on completeness value (average of adjacent points)
            colors = (completeness_smooth[:-1] + completeness_smooth[1:]) / 2

            lc = LineCollection(segments, cmap=cmap, linewidth=1, alpha=0.6, zorder=1)
            lc.set_array(colors)
            lc.set_clim(0, 1)
            ax2.add_collection(lc)

        ax2.set_ylim(0, 105)
        ax2.set_ylabel('', fontsize=6, color='#888888')  # No label by default
        if show_y2_label:
            ax2.set_ylabel('Completeness (%)', fontsize=6, color='#888888', rotation=270, labelpad=12)

        # Style y2 axis
        ax2.tick_params(axis='y', colors='#888888', labelsize=6)
        ax2.spines['right'].set_color('#888888')
        ax2.set_yticks([0, 50, 100])

        # Formatting for main axis
        ax.set_xlim(0, 36)
        ax.set_xticks([0, 12, 24, 36])

        if show_xlabel:
            ax.set_xticklabels(['ZT0', 'ZT12', 'ZT24', 'ZT36'], fontsize=7)
        else:
            ax.set_xticklabels([])

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)

        # Add y1 label showing the metric name (shortened)
        y_label = self._get_short_metric_label(metric_name)
        if y_label:
            ax.set_ylabel(y_label, fontsize=6, color=self.TEXT_COLOR)

        for spine in ['top', 'bottom', 'left']:
            ax.spines[spine].set_color(self.GRID_COLOR)

    def _get_short_metric_label(self, metric_name: str) -> str:
        """Create shortened metric label for CTA y-axis."""
        # Map full names to short labels with units
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

    def _get_unit_from_metric(self, metric_name: str) -> str:
        """Extract unit label from metric name for y-axis."""
        if '%' in metric_name:
            return '%'
        elif '(cm/min)' in metric_name:
            return 'cm/min'
        elif '(cm/s)' in metric_name:
            return 'cm/s'
        elif '(cm)' in metric_name:
            return 'cm'
        elif '(Hz)' in metric_name:
            return 'Hz'
        else:
            return ''

    def create_statistics_page(self, animal_id: str, animal_data: Dict[str, Any]) -> Figure:
        """Create summary statistics page with individual bar charts (max 7 per row)."""
        metrics = animal_data.get('metrics', {})
        n_metrics = len(metrics)

        fig = Figure(figsize=(14, 8.5), facecolor=self.BG_COLOR)

        fig.suptitle(f"Summary Statistics: {animal_id}", fontsize=14, fontweight='bold',
                    color=self.TEXT_COLOR, y=0.98)

        if n_metrics == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No metrics available", ha='center', va='center',
                   color=self.TEXT_COLOR, fontsize=14)
            ax.set_facecolor(self.BG_COLOR)
            ax.axis('off')
            return fig

        metric_names = list(metrics.keys())

        # Layout: max 7 bar charts per row, table on far right of last row
        max_per_row = 7
        n_rows = (n_metrics + max_per_row - 1) // max_per_row  # Ceiling division

        # Determine how many plots in the last row and add 1 for the table
        plots_in_last_row = n_metrics - (n_rows - 1) * max_per_row
        last_row_cols = plots_in_last_row + 1  # +1 for the table

        # Create grid with consistent column count
        n_cols = max(max_per_row, last_row_cols)
        gs = GridSpec(n_rows, n_cols, figure=fig,
                     hspace=0.4, wspace=0.3, left=0.04, right=0.98, top=0.92, bottom=0.06)

        # === Individual bar chart for each metric ===
        for idx, metric_name in enumerate(metric_names):
            row = idx // max_per_row
            col = idx % max_per_row

            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor(self.BG_COLOR)

            # Use verbose label like CTA plots
            verbose_label = self._get_short_metric_label(metric_name)
            # Create short title without unit for the title
            short_title = metric_name.split(' (')[0] if ' (' in metric_name else metric_name
            short_title = short_title.replace(' %', '')

            self._draw_single_bar_chart(ax, metrics[metric_name], short_title, verbose_label,
                                        show_legend=(idx == n_metrics - 1))

        # === Summary Table (far right of last row) ===
        # Table spans from after last plot to end of row
        table_start_col = plots_in_last_row
        ax_table = fig.add_subplot(gs[n_rows - 1, table_start_col:])
        ax_table.set_facecolor(self.BG_COLOR)
        ax_table.axis('off')

        self._draw_stats_table(ax_table, metrics)

        return fig

    def _draw_single_bar_chart(self, ax, metric_data: Dict, title: str, ylabel: str,
                                show_legend: bool = False):
        """Draw a bar chart for a single metric with Dark/Light comparison."""
        dark_mean = metric_data.get('dark_mean', 0)
        light_mean = metric_data.get('light_mean', 0)

        # Handle NaN values
        dark_mean = 0 if np.isnan(dark_mean) else dark_mean
        light_mean = 0 if np.isnan(light_mean) else light_mean

        x = np.array([0])
        width = 0.15  # Reduced from 0.35 to about 1/2

        ax.bar(x - width/2, [dark_mean], width, label='Dark', color='#4a4a4a', edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, [light_mean], width, label='Light', color='#f0c040', edgecolor='white', linewidth=0.5)

        ax.set_ylabel(ylabel, fontsize=6, color=self.TEXT_COLOR)
        ax.set_xticks([])  # No x-axis labels since title shows metric name
        ax.set_xlim(-0.5, 0.5)  # Keep consistent width
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        if show_legend:
            ax.legend(facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                     labelcolor=self.TEXT_COLOR, loc='upper right', fontsize=5)

        ax.grid(True, alpha=0.3, color=self.GRID_COLOR, axis='y')
        ax.set_title(title, fontsize=7, color=self.TEXT_COLOR, fontweight='bold')

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Only show left and bottom spines (remove box)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)

    def _draw_stats_table(self, ax, metrics: Dict):
        """Draw the statistics summary table with proper alignment."""
        metric_names = list(metrics.keys())

        # Header with fixed-width columns - ensure alignment
        header = f"{'Metric':<18}  {'Dark':>8}  {'Light':>8}  {'Diff':>9}  {'Ratio':>7}"
        ax.text(0.02, 0.95, header, fontsize=7, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR,
               family='monospace', va='top')

        ax.text(0.02, 0.88, "-" * 58, fontsize=7,
               transform=ax.transAxes, color=self.GRID_COLOR,
               family='monospace', va='top')

        y_pos = 0.80
        for metric_name in metric_names:
            dark_m = metrics[metric_name].get('dark_mean', np.nan)
            light_m = metrics[metric_name].get('light_mean', np.nan)

            if np.isnan(dark_m) or np.isnan(light_m):
                diff = np.nan
                ratio = np.nan
            else:
                diff = light_m - dark_m
                ratio = light_m / dark_m if dark_m != 0 else np.nan

            # Truncate metric name
            short_name = metric_name[:16] if len(metric_name) > 16 else metric_name

            # Format values with consistent width matching header
            dark_str = f"{dark_m:>8.2f}" if not np.isnan(dark_m) else f"{'N/A':>8}"
            light_str = f"{light_m:>8.2f}" if not np.isnan(light_m) else f"{'N/A':>8}"
            diff_str = f"{diff:>+9.2f}" if not np.isnan(diff) else f"{'N/A':>9}"
            ratio_str = f"{ratio:>6.2f}x" if not np.isnan(ratio) else f"{'N/A':>7}"

            row = f"{short_name:<18}  {dark_str}  {light_str}  {diff_str}  {ratio_str}"

            ax.text(0.02, y_pos, row, fontsize=6,
                   transform=ax.transAxes, color=self.TEXT_COLOR,
                   family='monospace', va='top')
            y_pos -= 0.065

            if y_pos < 0.05:  # Stop if running out of space
                break

    def create_sleep_analysis_page(self, animal_id: str, animal_data: Dict[str, Any]) -> Figure:
        """Create sleep bout analysis page with traces, histograms, and statistics.

        Layout (3 columns):
        - Left third: Stacked daily traces (top) + CTA (bottom)
        - Middle third: Stacked per-day histograms (top) + Combined histogram (bottom)
        - Right third: Statistics table with expanded interpretation
        """
        fig = Figure(figsize=(16, 12.5), facecolor=self.BG_COLOR)  # 25% taller

        fig.suptitle(f"Sleep Bout Analysis: {animal_id}", fontsize=14, fontweight='bold',
                    color=self.TEXT_COLOR, y=0.98)

        sleep_analysis = animal_data.get('sleep_analysis', {})
        if not sleep_analysis:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No sleep analysis data available", ha='center', va='center',
                   color=self.TEXT_COLOR, fontsize=14)
            ax.set_facecolor(self.BG_COLOR)
            ax.axis('off')
            return fig

        # Extract sleep analysis data
        bouts = sleep_analysis.get('bouts', [])
        light_stats = sleep_analysis.get('light_stats', {})
        dark_stats = sleep_analysis.get('dark_stats', {})
        total_stats = sleep_analysis.get('total_stats', {})
        per_day_stats = sleep_analysis.get('per_day_stats', [])
        quality_metrics = sleep_analysis.get('quality_metrics', {})
        hist_light = sleep_analysis.get('histogram_light', ([], []))
        hist_dark = sleep_analysis.get('histogram_dark', ([], []))
        per_day_histograms = sleep_analysis.get('per_day_histograms', [])
        n_days = sleep_analysis.get('n_days', 1)
        params = sleep_analysis.get('parameters', {})
        threshold = params.get('threshold', 0.5)
        bin_width = params.get('bin_width', 5.0)

        # Get sleeping metric data for traces
        metrics = animal_data.get('metrics', {})
        sleeping_data = None
        for metric_name in ['Sleeping %', 'sleeping', 'sleep']:
            if metric_name in metrics:
                sleeping_data = metrics[metric_name]
                break

        # Create grid layout: 3 rows x 3 columns
        # Row 0: Traces, Per-day bout count, Per-day time-in-bouts
        # Row 1: CTA, Combined bout count, Combined time-in-bouts
        # Row 2: Stats table, Bar charts grid (2x5)
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.6, 0.9, 3.0],
                     hspace=0.30, wspace=0.20, left=0.06, right=0.98, top=0.93, bottom=0.03)

        # === ROW 0 ===
        # Stacked daily traces
        ax_traces = fig.add_subplot(gs[0, 0])
        ax_traces.set_facecolor(self.BG_COLOR)
        if sleeping_data:
            daily_data = sleeping_data.get('daily_data', [])
            if daily_data:
                self._draw_sleep_traces_with_bouts(ax_traces, daily_data, bouts, n_days, threshold)

        # Per-day bout count histograms (will share x-axis with row 1)
        ax_day_count = fig.add_subplot(gs[0, 1])
        ax_day_count.set_facecolor(self.BG_COLOR)
        self._draw_stacked_day_histograms(ax_day_count, bouts, n_days, bin_width, weighted=False,
                                          show_xlabel=False)

        # Per-day time-in-bouts histograms (will share x-axis with row 1)
        ax_day_time = fig.add_subplot(gs[0, 2])
        ax_day_time.set_facecolor(self.BG_COLOR)
        self._draw_stacked_day_histograms(ax_day_time, bouts, n_days, bin_width, weighted=True,
                                          show_xlabel=False)

        # === ROW 1 ===
        # CTA trace
        ax_cta = fig.add_subplot(gs[1, 0])
        ax_cta.set_facecolor(self.BG_COLOR)
        if sleeping_data:
            cta = sleeping_data.get('cta', np.array([]))
            cta_sem = sleeping_data.get('cta_sem', np.array([]))
            if len(cta) > 0:
                self._draw_sleep_cta(ax_cta, cta, cta_sem, threshold)

        # Combined bout count histogram (shares x-axis with row 0)
        ax_count_hist = fig.add_subplot(gs[1, 1], sharex=ax_day_count)
        ax_count_hist.set_facecolor(self.BG_COLOR)
        self._draw_overlayed_histogram(ax_count_hist, hist_light, hist_dark, bouts, weighted=False)

        # Combined time-in-bouts histogram (shares x-axis with row 0)
        ax_time_hist = fig.add_subplot(gs[1, 2], sharex=ax_day_time)
        ax_time_hist.set_facecolor(self.BG_COLOR)
        self._draw_overlayed_histogram(ax_time_hist, hist_light, hist_dark, bouts, weighted=True)

        # === ROW 2 ===
        # Stats table (left side)
        ax_stats = fig.add_subplot(gs[2, 0])
        ax_stats.set_facecolor(self.BG_COLOR)
        ax_stats.axis('off')
        self._draw_sleep_stats_compact(ax_stats, light_stats, dark_stats, total_stats,
                                       threshold, bin_width, n_days)

        # Quality metric bar charts (right side, spans 2 columns)
        ax_bars = fig.add_subplot(gs[2, 1:])
        ax_bars.set_facecolor(self.BG_COLOR)
        self._draw_sleep_quality_bars(ax_bars, light_stats, dark_stats, total_stats, quality_metrics, n_days)

        return fig

    def _draw_sleep_traces_with_bouts(self, ax, daily_data: List[np.ndarray],
                                       bouts: List, n_days: int, threshold: float):
        """Draw stacked daily sleeping traces with bout markers."""
        # Normalize data for consistent scaling
        all_values = np.concatenate(daily_data)
        valid_values = all_values[~np.isnan(all_values)]

        if len(valid_values) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                   color=self.TEXT_COLOR)
            ax.axis('off')
            return

        # For percentage data (0-100 or 0-1)
        data_max = np.nanmax(valid_values)
        if data_max > 1.5:
            # Data is 0-100, normalize to 0-1
            scale_factor = 100.0
        else:
            scale_factor = 1.0

        x_hours = np.arange(1440) / 60  # 24 hours

        # Plot each day
        for day_idx, day_values in enumerate(daily_data):
            y_offset = day_idx
            normalized = day_values / scale_factor
            y_values = normalized * 0.85 + y_offset + 0.075

            color = self.DAY_COLORS[day_idx % len(self.DAY_COLORS)]
            ax.plot(x_hours, y_values, color=color, linewidth=0.6, alpha=0.8)

            # Draw threshold line for this day
            thresh_y = threshold * 0.85 + y_offset + 0.075
            ax.axhline(thresh_y, color='#ff6b6b', linewidth=0.5, alpha=0.4, linestyle='--')

            # Day label
            ax.text(-0.3, y_offset + 0.5, f'D{day_idx + 1}', ha='right', va='center',
                   fontsize=7, color=color, fontweight='bold')

        # Highlight bouts with shading
        for bout in bouts:
            day_idx = bout.day
            y_base = day_idx
            x_start = bout.start_minute / 60
            x_end = bout.end_minute / 60

            bout_color = '#4CAF50' if bout.phase == 'light' else '#2196F3'
            ax.axvspan(x_start, x_end, ymin=(y_base + 0.05) / n_days,
                      ymax=(y_base + 0.95) / n_days,
                      alpha=0.15, color=bout_color, zorder=0)

        # Add light/dark phase shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)

        # Formatting
        ax.set_xlim(0, 24)
        ax.set_ylim(0, n_days)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['ZT0', 'ZT6', 'ZT12', 'ZT18', 'ZT24'], fontsize=7)
        ax.set_yticks([])
        ax.set_title(f'Daily Sleep Traces (threshold={threshold})', fontsize=10,
                    color=self.TEXT_COLOR, fontweight='bold')
        ax.set_xlabel('Zeitgeber Time', fontsize=8, color=self.TEXT_COLOR)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_sleep_cta(self, ax, cta: np.ndarray, cta_sem: np.ndarray, threshold: float):
        """Draw CTA for sleeping percentage."""
        x_hours = np.arange(len(cta)) / 60

        # Smooth the data
        window = 15
        cta_smooth = pd.Series(cta).rolling(window=window, min_periods=1, center=True).mean().values
        sem_smooth = pd.Series(cta_sem).rolling(window=window, min_periods=1, center=True).mean().values

        # Normalize if needed (convert 0-100 to 0-1)
        if np.nanmax(cta_smooth) > 1.5:
            cta_smooth = cta_smooth / 100.0
            sem_smooth = sem_smooth / 100.0

        # Add light/dark shading
        ax.axvspan(0, 12, facecolor=self.LIGHT_PHASE_COLOR, zorder=0)
        ax.axvspan(12, 24, facecolor=self.DARK_PHASE_COLOR, zorder=0)

        # Plot CTA
        ax.plot(x_hours, cta_smooth, color='#3daee9', linewidth=1.5, zorder=3)
        ax.fill_between(x_hours, cta_smooth - sem_smooth, cta_smooth + sem_smooth,
                       color='#3daee9', alpha=0.3, zorder=2)

        # Formatting
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(['ZT0', 'ZT6', 'ZT12', 'ZT18', 'ZT24'], fontsize=7)
        ax.set_ylabel('Sleep Probability', fontsize=8, color=self.TEXT_COLOR)
        ax.set_xlabel('Zeitgeber Time', fontsize=8, color=self.TEXT_COLOR)
        ax.set_title('Circadian Time Average', fontsize=10, color=self.TEXT_COLOR, fontweight='bold')

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=7)
        ax.grid(True, alpha=0.2, color=self.GRID_COLOR)
        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_histogram(self, ax, bin_edges: np.ndarray, counts: np.ndarray,
                        title: str = '', color: str = '#3daee9', show_xlabel: bool = True):
        """Draw a simple histogram for a single day or combined data."""
        if len(bin_edges) < 2 or len(counts) == 0:
            ax.text(0.5, 0.5, "No bouts", ha='center', va='center',
                   color=self.TEXT_COLOR, fontsize=8)
            ax.axis('off')
            return

        # Plot histogram as bar chart
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 5

        ax.bar(bin_centers, counts, width=bin_width * 0.8, color=color,
               edgecolor='white', linewidth=0.3, alpha=0.8)

        ax.set_xlim(0, bin_edges[-1])
        if show_xlabel:
            ax.set_xlabel('Bout Duration (min)', fontsize=6, color=self.TEXT_COLOR)
        ax.set_ylabel('Count', fontsize=6, color=self.TEXT_COLOR)
        ax.set_title(title, fontsize=8, color=self.TEXT_COLOR, fontweight='bold')

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)

    def _draw_overlayed_histogram(self, ax, hist_light: Tuple, hist_dark: Tuple,
                                    bouts: List = None, weighted: bool = False,
                                    show_title: bool = True):
        """
        Draw combined histogram with light and dark phases overlayed.

        Args:
            ax: Matplotlib axes
            hist_light: (edges, counts) tuple for light phase
            hist_dark: (edges, counts) tuple for dark phase
            bouts: List of SleepBout objects (needed for weighted mode)
            weighted: If True, show time-weighted (sum of durations) instead of count
            show_title: If True, show the title
        """
        light_edges, light_counts = hist_light
        dark_edges, dark_counts = hist_dark

        has_light = len(light_edges) > 1 and np.sum(light_counts) > 0
        has_dark = len(dark_edges) > 1 and np.sum(dark_counts) > 0

        if not has_light and not has_dark:
            ax.text(0.5, 0.5, "No sleep bouts detected", ha='center', va='center',
                   color=self.TEXT_COLOR, fontsize=10)
            ax.axis('off')
            return

        # Use the same bin edges for both
        if has_light:
            bin_edges = light_edges
        else:
            bin_edges = dark_edges

        bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 5
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if weighted and bouts:
            # Time-weighted: compute total time in each bin
            light_weights = np.zeros(len(bin_edges) - 1)
            dark_weights = np.zeros(len(bin_edges) - 1)

            for bout in bouts:
                bin_idx = np.searchsorted(bin_edges[1:], bout.duration)
                if bin_idx < len(light_weights):
                    if bout.phase == 'light':
                        light_weights[bin_idx] += bout.duration
                    else:
                        dark_weights[bin_idx] += bout.duration

            # Plot time-weighted bars
            if np.any(light_weights > 0):
                ax.bar(bin_centers - bin_width * 0.2, light_weights, width=bin_width * 0.4,
                       color='#f0c040', edgecolor='white', linewidth=0.3, alpha=0.8,
                       label='Light')

            if np.any(dark_weights > 0):
                ax.bar(bin_centers + bin_width * 0.2, dark_weights, width=bin_width * 0.4,
                       color='#4a4a4a', edgecolor='white', linewidth=0.3, alpha=0.8,
                       label='Dark')

            ax.set_ylabel('Time (min)', fontsize=7, color=self.TEXT_COLOR)
            if show_title:
                ax.set_title('Time in Bouts', fontsize=9, color=self.TEXT_COLOR, fontweight='bold')
        else:
            # Standard count histogram
            if has_light:
                ax.bar(bin_centers - bin_width * 0.2, light_counts, width=bin_width * 0.4,
                       color='#f0c040', edgecolor='white', linewidth=0.3, alpha=0.8,
                       label='Light')

            if has_dark:
                ax.bar(bin_centers + bin_width * 0.2, dark_counts, width=bin_width * 0.4,
                       color='#4a4a4a', edgecolor='white', linewidth=0.3, alpha=0.8,
                       label='Dark')

            ax.set_ylabel('Count', fontsize=7, color=self.TEXT_COLOR)
            if show_title:
                ax.set_title('Bout Count', fontsize=9, color=self.TEXT_COLOR, fontweight='bold')

        ax.set_xlim(0, bin_edges[-1])
        ax.set_xlabel('Duration (min)', fontsize=7, color=self.TEXT_COLOR)

        ax.legend(loc='upper right', fontsize=5, facecolor=self.BG_COLOR,
                 edgecolor=self.GRID_COLOR, labelcolor=self.TEXT_COLOR)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)
        ax.grid(True, alpha=0.2, color=self.GRID_COLOR, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)

    def _draw_sleep_stats_table(self, ax, light_stats: Dict, dark_stats: Dict,
                                 total_stats: Dict, threshold: float, bin_width: float,
                                 n_days: int):
        """Draw sleep analysis statistics table."""
        # Title
        ax.text(0.02, 0.98, "Sleep Bout Statistics", fontsize=11, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, va='top')

        # Parameters
        ax.text(0.02, 0.90, f"Analysis Parameters: threshold={threshold}, bin_width={bin_width}min, days={n_days}",
               fontsize=7, transform=ax.transAxes, color='#888888', va='top')

        # Header
        header = f"{'Metric':<22}  {'Light':>10}  {'Dark':>10}  {'Total':>10}"
        ax.text(0.02, 0.82, header, fontsize=8, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR,
               family='monospace', va='top')

        ax.text(0.02, 0.76, "-" * 60, fontsize=8,
               transform=ax.transAxes, color=self.GRID_COLOR,
               family='monospace', va='top')

        # Statistics rows
        stats_rows = [
            ('Total Sleep (min)', 'total_minutes', '.1f'),
            ('Sleep %', 'percent_time', '.1f'),
            ('Bout Count', 'bout_count', '.0f'),
            ('Mean Duration (min)', 'mean_duration', '.1f'),
            ('Median Duration (min)', 'median_duration', '.1f'),
            ('Max Duration (min)', 'max_duration', '.1f'),
            ('Min Duration (min)', 'min_duration', '.1f'),
            ('Std Dev (min)', 'std_duration', '.1f'),
        ]

        y_pos = 0.70
        for label, key, fmt in stats_rows:
            light_val = light_stats.get(key, 0)
            dark_val = dark_stats.get(key, 0)
            total_val = total_stats.get(key, 0)

            light_str = f"{light_val:{fmt}}" if light_val else "0"
            dark_str = f"{dark_val:{fmt}}" if dark_val else "0"
            total_str = f"{total_val:{fmt}}" if total_val else "0"

            row = f"{label:<22}  {light_str:>10}  {dark_str:>10}  {total_str:>10}"
            ax.text(0.02, y_pos, row, fontsize=7,
                   transform=ax.transAxes, color=self.TEXT_COLOR,
                   family='monospace', va='top')
            y_pos -= 0.07

        # Add interpretation note
        y_pos -= 0.05
        ax.text(0.02, y_pos, "Interpretation:", fontsize=7, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, va='top')
        y_pos -= 0.06

        # Calculate fragmentation index (bouts per hour of sleep)
        if total_stats.get('total_minutes', 0) > 0:
            frag_index = total_stats.get('bout_count', 0) / (total_stats['total_minutes'] / 60)
            frag_text = f"Fragmentation Index: {frag_index:.2f} bouts/hour of sleep"
        else:
            frag_text = "Fragmentation Index: N/A (no sleep detected)"

        ax.text(0.02, y_pos, frag_text, fontsize=7,
               transform=ax.transAxes, color='#888888', va='top')

    def _draw_stacked_day_histograms(self, ax, bouts: List, n_days: int, bin_width: float,
                                       weighted: bool = False, show_xlabel: bool = True):
        """
        Draw stacked per-day histograms with light/dark overlayed.

        Args:
            ax: Matplotlib axes
            bouts: List of SleepBout objects
            n_days: Number of days
            bin_width: Histogram bin width in minutes
            weighted: If True, show time-in-bouts (duration sum) instead of count
            show_xlabel: If True, show x-axis label
        """
        if not bouts or n_days == 0:
            ax.text(0.5, 0.5, "No bouts detected", ha='center', va='center',
                   color=self.TEXT_COLOR, fontsize=10)
            ax.axis('off')
            return

        # Get max duration for consistent x-axis
        max_dur = max(b.duration for b in bouts) if bouts else 60
        max_dur = np.ceil(max_dur / bin_width) * bin_width + bin_width

        # Create bins
        bins = np.arange(0, max_dur + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute histogram data for each day
        max_val = 0
        day_data = []
        for day_idx in range(n_days):
            day_bouts = [b for b in bouts if b.day == day_idx]
            light_bouts = [b for b in day_bouts if b.phase == 'light']
            dark_bouts = [b for b in day_bouts if b.phase == 'dark']

            if weighted:
                # Time-weighted: sum durations in each bin
                light_vals = np.zeros(len(bins) - 1)
                dark_vals = np.zeros(len(bins) - 1)
                for b in light_bouts:
                    bin_idx = np.searchsorted(bins[1:], b.duration)
                    if bin_idx < len(light_vals):
                        light_vals[bin_idx] += b.duration
                for b in dark_bouts:
                    bin_idx = np.searchsorted(bins[1:], b.duration)
                    if bin_idx < len(dark_vals):
                        dark_vals[bin_idx] += b.duration
            else:
                # Count-based histogram
                light_durs = [b.duration for b in light_bouts]
                dark_durs = [b.duration for b in dark_bouts]
                light_vals, _ = np.histogram(light_durs, bins=bins) if light_durs else (np.zeros(len(bins)-1), bins)
                dark_vals, _ = np.histogram(dark_durs, bins=bins) if dark_durs else (np.zeros(len(bins)-1), bins)

            day_data.append((light_vals, dark_vals))
            max_val = max(max_val, np.max(light_vals), np.max(dark_vals))

        if max_val == 0:
            max_val = 1  # Avoid division by zero

        # Plot each day stacked vertically
        bar_width = bin_width * 0.35
        for day_idx in range(n_days):
            y_offset = day_idx
            light_vals, dark_vals = day_data[day_idx]

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
            color = self.DAY_COLORS[day_idx % len(self.DAY_COLORS)]
            ax.text(-max_dur * 0.02, y_offset + 0.5, f'D{day_idx + 1}', ha='right', va='center',
                   fontsize=7, color=color, fontweight='bold')

        # Formatting
        ax.set_xlim(0, max_dur)
        ax.set_ylim(0, n_days)
        ax.set_yticks([])
        if show_xlabel:
            ax.set_xlabel('Duration (min)', fontsize=7, color=self.TEXT_COLOR)
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)
        else:
            # Hide x-tick labels when sharing axis with plot below
            ax.tick_params(colors=self.TEXT_COLOR, labelsize=6, labelbottom=False)

        title = 'Time in Bouts' if weighted else 'Bout Count'
        ax.set_title(f'Per-Day {title}', fontsize=9, color=self.TEXT_COLOR, fontweight='bold')

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

    def _draw_sleep_stats_table_expanded(self, ax, light_stats: Dict, dark_stats: Dict,
                                          total_stats: Dict, per_day_stats: List,
                                          threshold: float, bin_width: float, n_days: int):
        """Draw unified sleep analysis statistics table."""
        # Title
        ax.text(0.02, 0.99, "Sleep Bout Statistics", fontsize=11, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, va='top')

        # Parameters
        ax.text(0.02, 0.95, f"threshold={threshold}, bin={bin_width}min, days={n_days}",
               fontsize=6, transform=ax.transAxes, color='#888888', va='top')

        # Calculate derived metrics
        recording_hours_light = n_days * 12
        recording_hours_dark = n_days * 12
        recording_hours_total = n_days * 24

        light_bouts = light_stats.get('bout_count', 0)
        dark_bouts = dark_stats.get('bout_count', 0)
        total_bouts = total_stats.get('bout_count', 0)

        bouts_per_hr_light = light_bouts / recording_hours_light if recording_hours_light > 0 else 0
        bouts_per_hr_dark = dark_bouts / recording_hours_dark if recording_hours_dark > 0 else 0
        bouts_per_hr_total = total_bouts / recording_hours_total if recording_hours_total > 0 else 0

        light_sleep_hrs = light_stats.get('total_minutes', 0) / 60
        dark_sleep_hrs = dark_stats.get('total_minutes', 0) / 60
        total_sleep_hrs = total_stats.get('total_minutes', 0) / 60

        frag_light = light_bouts / light_sleep_hrs if light_sleep_hrs > 0 else 0
        frag_dark = dark_bouts / dark_sleep_hrs if dark_sleep_hrs > 0 else 0
        frag_total = total_bouts / total_sleep_hrs if total_sleep_hrs > 0 else 0

        # === UNIFIED TABLE ===
        header = f"{'Metric':<18} {'Light':>7} {'Dark':>7} {'Total':>7}"
        ax.text(0.02, 0.91, header, fontsize=6, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')

        ax.text(0.02, 0.885, "-" * 44, fontsize=6,
               transform=ax.transAxes, color=self.GRID_COLOR, family='monospace', va='top')

        # Define rows: (label, light_val, dark_val, total_val, format, is_bold)
        rows_data = [
            ('Total Sleep (min)', light_stats.get('total_minutes', 0), dark_stats.get('total_minutes', 0), total_stats.get('total_minutes', 0), '.1f', True),
            ('Sleep %', light_stats.get('percent_time', 0), dark_stats.get('percent_time', 0), total_stats.get('percent_time', 0), '.1f', True),
            ('Bout Count', light_bouts, dark_bouts, total_bouts, '.0f', False),
            ('Bouts/rec hour', bouts_per_hr_light, bouts_per_hr_dark, bouts_per_hr_total, '.2f', True),
            ('Frag Index', frag_light, frag_dark, frag_total, '.2f', True),
            ('Mean Bout (min)', light_stats.get('mean_duration', 0), dark_stats.get('mean_duration', 0), total_stats.get('mean_duration', 0), '.1f', False),
            ('Median Bout (min)', light_stats.get('median_duration', 0), dark_stats.get('median_duration', 0), total_stats.get('median_duration', 0), '.1f', False),
            ('Max Bout (min)', light_stats.get('max_duration', 0), dark_stats.get('max_duration', 0), total_stats.get('max_duration', 0), '.1f', False),
        ]

        y_pos = 0.86
        for label, light_val, dark_val, total_val, fmt, is_bold in rows_data:
            light_str = f"{light_val:{fmt}}" if light_val else "0"
            dark_str = f"{dark_val:{fmt}}" if dark_val else "0"
            total_str = f"{total_val:{fmt}}" if total_val else "0"

            row = f"{label:<18} {light_str:>7} {dark_str:>7} {total_str:>7}"
            weight = 'bold' if is_bold else 'normal'
            ax.text(0.02, y_pos, row, fontsize=5.5, fontweight=weight,
                   transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')
            y_pos -= 0.036

        # === DEFINITIONS ===
        y_pos -= 0.015
        ax.text(0.02, y_pos, "Definitions:", fontsize=7, fontweight='bold',
               transform=ax.transAxes, color='#3daee9', va='top')
        y_pos -= 0.028

        definitions = [
            "• Bouts/rec hour: Bouts ÷ recording hours",
            "• Frag Index: Bouts ÷ sleep hours",
            "  (higher = more fragmented)",
        ]

        for defn in definitions:
            ax.text(0.02, y_pos, defn, fontsize=5.5,
                   transform=ax.transAxes, color='#888888', va='top')
            y_pos -= 0.025

        # === PER-DAY SUMMARY ===
        if per_day_stats:
            y_pos -= 0.015
            ax.text(0.02, y_pos, "Per-Day Summary", fontsize=7, fontweight='bold',
                   transform=ax.transAxes, color='#3daee9', va='top')
            y_pos -= 0.028

            day_header = f"{'Day':<4} {'Bouts':>5} {'Sleep%':>7} {'Mean':>6} {'Frag':>5}"
            ax.text(0.02, y_pos, day_header, fontsize=5.5, fontweight='bold',
                   transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')
            y_pos -= 0.022

            for day_stat in per_day_stats[:7]:  # Limit to 7 days
                day_num = day_stat.get('day', 0)
                total_bouts_day = day_stat.get('total_bouts', 0)
                total_sleep_day = day_stat.get('total_sleep_minutes', 0)

                # Calculate average sleep % across phases
                light_pct = day_stat.get('light', {}).get('percent_time', 0)
                dark_pct = day_stat.get('dark', {}).get('percent_time', 0)
                avg_pct = (light_pct + dark_pct) / 2

                # Calculate average mean bout duration
                light_mean = day_stat.get('light', {}).get('mean_duration', 0)
                dark_mean = day_stat.get('dark', {}).get('mean_duration', 0)
                avg_mean = (light_mean + dark_mean) / 2 if (light_mean + dark_mean) > 0 else 0

                # Calculate daily fragmentation
                sleep_hrs_day = total_sleep_day / 60
                frag_day = total_bouts_day / sleep_hrs_day if sleep_hrs_day > 0 else 0

                row = f"{'D' + str(day_num):<4} {total_bouts_day:>5} {avg_pct:>6.1f}% {avg_mean:>5.1f} {frag_day:>5.1f}"
                ax.text(0.02, y_pos, row, fontsize=5.5,
                       transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')
                y_pos -= 0.022

    def _draw_sleep_stats_compact(self, ax, light_stats: Dict, dark_stats: Dict,
                                   total_stats: Dict, threshold: float,
                                   bin_width: float, n_days: int):
        """Draw full sleep statistics table for the new layout."""
        # Title
        ax.text(0.5, 0.99, "Sleep Bout Statistics", fontsize=9, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, va='top', ha='center')

        # Parameters
        ax.text(0.5, 0.93, f"threshold={threshold}, bin={bin_width}min, days={n_days}",
               fontsize=5.5, transform=ax.transAxes, color='#888888', va='top', ha='center')

        # Calculate derived metrics
        recording_hours = n_days * 12  # Per phase
        recording_hours_total = n_days * 24

        light_bouts = light_stats.get('bout_count', 0)
        dark_bouts = dark_stats.get('bout_count', 0)
        total_bouts = total_stats.get('bout_count', 0)

        bouts_per_hr_light = light_bouts / recording_hours if recording_hours > 0 else 0
        bouts_per_hr_dark = dark_bouts / recording_hours if recording_hours > 0 else 0
        bouts_per_hr_total = total_bouts / recording_hours_total if recording_hours_total > 0 else 0

        light_sleep_hrs = light_stats.get('total_minutes', 0) / 60
        dark_sleep_hrs = dark_stats.get('total_minutes', 0) / 60
        total_sleep_hrs = total_stats.get('total_minutes', 0) / 60

        frag_light = light_bouts / light_sleep_hrs if light_sleep_hrs > 0 else 0
        frag_dark = dark_bouts / dark_sleep_hrs if dark_sleep_hrs > 0 else 0
        frag_total = total_bouts / total_sleep_hrs if total_sleep_hrs > 0 else 0

        # Header
        y_pos = 0.87
        header = f"{'Metric':<16} {'Light':>7} {'Dark':>7} {'Total':>7}"
        ax.text(0.02, y_pos, header, fontsize=5.5, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace', va='top')
        y_pos -= 0.04
        ax.text(0.02, y_pos, "-" * 40, fontsize=5.5,
               transform=ax.transAxes, color=self.GRID_COLOR, family='monospace', va='top')
        y_pos -= 0.045

        # Full statistics - rows: (label, light_val, dark_val, total_val, format, is_bold)
        rows_data = [
            ('Total Sleep (min)', light_stats.get('total_minutes', 0), dark_stats.get('total_minutes', 0), total_stats.get('total_minutes', 0), '.1f', True),
            ('Sleep %', light_stats.get('percent_time', 0), dark_stats.get('percent_time', 0), total_stats.get('percent_time', 0), '.1f', True),
            ('Bout Count', light_bouts, dark_bouts, total_bouts, '.0f', False),
            ('Bouts/rec hour', bouts_per_hr_light, bouts_per_hr_dark, bouts_per_hr_total, '.2f', True),
            ('Frag Index', frag_light, frag_dark, frag_total, '.2f', True),
            ('Mean Bout (min)', light_stats.get('mean_duration', 0), dark_stats.get('mean_duration', 0), total_stats.get('mean_duration', 0), '.1f', False),
            ('Median Bout', light_stats.get('median_duration', 0), dark_stats.get('median_duration', 0), total_stats.get('median_duration', 0), '.1f', False),
            ('Max Bout (min)', light_stats.get('max_duration', 0), dark_stats.get('max_duration', 0), total_stats.get('max_duration', 0), '.1f', False),
        ]

        for label, light_val, dark_val, total_val, fmt, is_bold in rows_data:
            light_str = f"{light_val:{fmt}}" if light_val else "0"
            dark_str = f"{dark_val:{fmt}}" if dark_val else "0"
            total_str = f"{total_val:{fmt}}" if total_val else "0"

            row = f"{label:<16} {light_str:>7} {dark_str:>7} {total_str:>7}"
            weight = 'bold' if is_bold else 'normal'
            ax.text(0.02, y_pos, row, fontsize=5,
                   transform=ax.transAxes, color=self.TEXT_COLOR, family='monospace',
                   fontweight=weight, va='top')
            y_pos -= 0.045

        # Definitions
        y_pos -= 0.02
        ax.text(0.02, y_pos, "Definitions:", fontsize=5.5, fontweight='bold',
               transform=ax.transAxes, color='#3daee9', va='top')
        y_pos -= 0.04

        definitions = [
            "Bouts/rec hr = Bouts ÷ recording hours",
            "Frag Index = Bouts ÷ sleep hours (higher=fragmented)",
        ]

        for defn in definitions:
            ax.text(0.02, y_pos, f"• {defn}", fontsize=4.5,
                   transform=ax.transAxes, color='#888888', va='top')
            y_pos -= 0.035

    def _draw_sleep_quality_bars(self, ax, light_stats: Dict, dark_stats: Dict,
                                  total_stats: Dict, quality_metrics: Dict, n_days: int = 1):
        """Draw 2x5 grid of bar charts for all sleep quality metrics."""
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        # Calculate derived metrics
        rec_hrs_phase = n_days * 12
        rec_hrs_total = n_days * 24

        l_bouts = light_stats.get('bout_count', 0)
        d_bouts = dark_stats.get('bout_count', 0)
        t_bouts = total_stats.get('bout_count', 0)

        bouts_hr_l = l_bouts / rec_hrs_phase if rec_hrs_phase > 0 else 0
        bouts_hr_d = d_bouts / rec_hrs_phase if rec_hrs_phase > 0 else 0
        bouts_hr_t = t_bouts / rec_hrs_total if rec_hrs_total > 0 else 0

        l_sleep_hrs = light_stats.get('total_minutes', 0) / 60
        d_sleep_hrs = dark_stats.get('total_minutes', 0) / 60
        t_sleep_hrs = total_stats.get('total_minutes', 0) / 60

        frag_l = l_bouts / l_sleep_hrs if l_sleep_hrs > 0 else 0
        frag_d = d_bouts / d_sleep_hrs if d_sleep_hrs > 0 else 0
        frag_t = t_bouts / t_sleep_hrs if t_sleep_hrs > 0 else 0

        # Create 2x5 subgrid
        gs_inner = GridSpecFromSubplotSpec(2, 5, subplot_spec=ax.get_subplotspec(),
                                           wspace=0.35, hspace=0.45)
        ax.axis('off')
        fig = ax.figure

        # Row A (top): Sleep min, Sleep %, Bout Count, Bouts/hr, Frag Index
        # 1. Sleep minutes (L/D/T)
        ax1 = fig.add_subplot(gs_inner[0, 0])
        ax1.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar(ax1,
                          light_stats.get('total_minutes', 0),
                          dark_stats.get('total_minutes', 0),
                          total_stats.get('total_minutes', 0),
                          'Sleep (min)', 'min', fmt='.0f')

        # 2. Sleep % (L/D)
        ax2 = fig.add_subplot(gs_inner[0, 1])
        ax2.set_facecolor(self.BG_COLOR)
        self._draw_ld_bar(ax2,
                         light_stats.get('percent_time', 0),
                         dark_stats.get('percent_time', 0),
                         'Sleep %', '%')

        # 3. Bout Count (L/D/T)
        ax3 = fig.add_subplot(gs_inner[0, 2])
        ax3.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar(ax3, l_bouts, d_bouts, t_bouts, 'Bouts', '#', fmt='.0f')

        # 4. Bouts/rec hr (L/D/T)
        ax4 = fig.add_subplot(gs_inner[0, 3])
        ax4.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar(ax4, bouts_hr_l, bouts_hr_d, bouts_hr_t, 'Bouts/hr', '/hr', fmt='.2f')

        # 5. Fragmentation Index (L/D/T)
        ax5 = fig.add_subplot(gs_inner[0, 4])
        ax5.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar(ax5, frag_l, frag_d, frag_t, 'Frag Idx', '/hr', fmt='.2f')

        # Row B (bottom): Mean Bout, Median Bout, Max Bout, % Long Bouts, L/D + Trans
        # 6. Mean Bout (L/D/T)
        ax6 = fig.add_subplot(gs_inner[1, 0])
        ax6.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar(ax6,
                          light_stats.get('mean_duration', 0),
                          dark_stats.get('mean_duration', 0),
                          total_stats.get('mean_duration', 0),
                          'Mean Bout', 'min', fmt='.1f')

        # 7. Median Bout (L/D/T)
        ax7 = fig.add_subplot(gs_inner[1, 1])
        ax7.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar(ax7,
                          light_stats.get('median_duration', 0),
                          dark_stats.get('median_duration', 0),
                          total_stats.get('median_duration', 0),
                          'Med Bout', 'min', fmt='.1f')

        # 8. Max Bout (L/D/T)
        ax8 = fig.add_subplot(gs_inner[1, 2])
        ax8.set_facecolor(self.BG_COLOR)
        self._draw_ldt_bar(ax8,
                          light_stats.get('max_duration', 0),
                          dark_stats.get('max_duration', 0),
                          total_stats.get('max_duration', 0),
                          'Max Bout', 'min', fmt='.0f')

        # 9. % Long Bouts (L/D)
        ax9 = fig.add_subplot(gs_inner[1, 3])
        ax9.set_facecolor(self.BG_COLOR)
        self._draw_ld_bar(ax9,
                         quality_metrics.get('long_bout_pct_light', 0),
                         quality_metrics.get('long_bout_pct_dark', 0),
                         '% Long Bouts', '%')

        # 10. L/D Ratio + Transitions (combined)
        ax10 = fig.add_subplot(gs_inner[1, 4])
        ax10.set_facecolor(self.BG_COLOR)
        self._draw_dual_metric_bar(ax10,
                                   quality_metrics.get('light_dark_ratio', 0),
                                   quality_metrics.get('transition_rate', 0),
                                   'L/D', 'Trans', '/hr')

    def _draw_ldt_bar(self, ax, light_val: float, dark_val: float, total_val: float,
                      title: str, unit: str, fmt: str = '.1f'):
        """Draw a Light/Dark/Total comparison bar chart."""
        x = [0, 1, 2]
        vals = [light_val, dark_val, total_val]
        colors = ['#f0c040', '#4a4a4a', '#3daee9']  # Yellow, Gray, Blue

        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.3, width=0.7)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['L', 'D', 'T'], fontsize=5, color=self.TEXT_COLOR)
        ax.set_title(title, fontsize=6, color=self.TEXT_COLOR, fontweight='bold', pad=2)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:{fmt}}', ha='center', va='bottom',
                       fontsize=4, color=self.TEXT_COLOR)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)
        ax.tick_params(axis='y', labelsize=4)

    def _draw_ld_bar(self, ax, light_val: float, dark_val: float, title: str, unit: str):
        """Draw a light/dark comparison bar chart."""
        x = [0, 1]
        vals = [light_val, dark_val]
        colors = ['#f0c040', '#4a4a4a']

        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.3, width=0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['L', 'D'], fontsize=5, color=self.TEXT_COLOR)
        ax.set_title(title, fontsize=6, color=self.TEXT_COLOR, fontweight='bold', pad=2)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom',
                       fontsize=4, color=self.TEXT_COLOR)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)
        ax.tick_params(axis='y', labelsize=4)

    def _draw_dual_metric_bar(self, ax, val1: float, val2: float,
                               label1: str, label2: str, unit: str):
        """Draw two metrics side by side."""
        x = [0, 1]
        vals = [val1, val2]
        colors = ['#3daee9', '#e74c3c']  # Blue, Red

        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.3, width=0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([label1, label2], fontsize=5, color=self.TEXT_COLOR)
        ax.set_title('Quality', fontsize=6, color=self.TEXT_COLOR, fontweight='bold', pad=2)

        # Add value labels
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom',
                       fontsize=4, color=self.TEXT_COLOR)

        ax.tick_params(colors=self.TEXT_COLOR, labelsize=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.GRID_COLOR)
        ax.spines['bottom'].set_color(self.GRID_COLOR)
        ax.tick_params(axis='y', labelsize=4)
