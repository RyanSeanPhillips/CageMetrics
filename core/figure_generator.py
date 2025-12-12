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
    # For phase shading: dark phase uses same as background, light phase uses darker gray
    DARK_PHASE_COLOR = '#2d2d2d'   # Same as background (no visible shading)
    LIGHT_PHASE_COLOR = '#454545'  # Darker gray for light phase

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

        companion = metadata.get('companion', None)
        if companion is None:
            companion_str = 'None'
        elif isinstance(companion, list):
            companion_str = ', '.join(companion)
        else:
            companion_str = str(companion)

        info_text = [
            f"Animal ID:     {metadata.get('animal_id', 'Unknown')}",
            f"Genotype:      {metadata.get('genotype', 'Unknown')}",
            f"Sex:           {metadata.get('sex', 'Unknown')}",
            f"Cohort:        {metadata.get('cohort', 'Unknown')}",
            f"Cage ID:       {metadata.get('cage_id', 'Unknown')}",
            f"Companion:     {companion_str}",
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
                                                 show_xlabel=True, show_y2_label=(metric_idx == n_metrics - 1))

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
        """Create summary statistics page with individual bar charts for each metric in one row."""
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

        # Create grid: all plots in one row at top, table at bottom
        gs = GridSpec(2, n_metrics, figure=fig, height_ratios=[1.8, 1],
                     hspace=0.35, wspace=0.4, left=0.04, right=0.98, top=0.92, bottom=0.08)

        # === Individual bar chart for each metric ===
        for idx, metric_name in enumerate(metric_names):
            ax = fig.add_subplot(gs[0, idx])
            ax.set_facecolor(self.BG_COLOR)

            unit = self._get_unit_from_metric(metric_name)
            # Create short title without unit
            short_title = metric_name.split(' (')[0] if ' (' in metric_name else metric_name
            short_title = short_title.replace(' %', '')

            self._draw_single_bar_chart(ax, metrics[metric_name], short_title, unit,
                                        show_legend=(idx == n_metrics - 1))

        # === Summary Table (spans bottom row) ===
        ax_table = fig.add_subplot(gs[1, :])
        ax_table.set_facecolor(self.BG_COLOR)
        ax_table.axis('off')

        self._draw_stats_table(ax_table, metrics)

        return fig

    def _draw_single_bar_chart(self, ax, metric_data: Dict, title: str, unit: str,
                                show_legend: bool = False):
        """Draw a bar chart for a single metric with Dark/Light comparison."""
        dark_mean = metric_data.get('dark_mean', 0)
        light_mean = metric_data.get('light_mean', 0)

        # Handle NaN values
        dark_mean = 0 if np.isnan(dark_mean) else dark_mean
        light_mean = 0 if np.isnan(light_mean) else light_mean

        x = np.array([0])
        width = 0.35

        ax.bar(x - width/2, [dark_mean], width, label='Dark', color='#4a4a4a', edgecolor='white')
        ax.bar(x + width/2, [light_mean], width, label='Light', color='#f0c040', edgecolor='white')

        ax.set_ylabel(unit if unit else 'Value', fontsize=7, color=self.TEXT_COLOR)
        ax.set_xticks([])  # No x-axis labels since title shows metric name
        ax.tick_params(colors=self.TEXT_COLOR, labelsize=6)

        if show_legend:
            ax.legend(facecolor=self.BG_COLOR, edgecolor=self.GRID_COLOR,
                     labelcolor=self.TEXT_COLOR, loc='upper right', fontsize=6)

        ax.grid(True, alpha=0.3, color=self.GRID_COLOR, axis='y')
        ax.set_title(title, fontsize=8, color=self.TEXT_COLOR, fontweight='bold')

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _draw_stats_table(self, ax, metrics: Dict):
        """Draw the statistics summary table with proper alignment."""
        metric_names = list(metrics.keys())

        # Header with fixed-width columns
        header = f"{'Metric':<20}  {'Dark':>10}  {'Light':>10}  {'Diff':>10}  {'Ratio':>8}"
        ax.text(0.02, 0.95, header, fontsize=8, fontweight='bold',
               transform=ax.transAxes, color=self.TEXT_COLOR,
               family='monospace', va='top')

        ax.text(0.02, 0.87, "-" * 68, fontsize=8,
               transform=ax.transAxes, color=self.GRID_COLOR,
               family='monospace', va='top')

        y_pos = 0.78
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
            short_name = metric_name[:18] if len(metric_name) > 18 else metric_name

            # Format values with consistent width
            dark_str = f"{dark_m:>10.2f}" if not np.isnan(dark_m) else f"{'N/A':>10}"
            light_str = f"{light_m:>10.2f}" if not np.isnan(light_m) else f"{'N/A':>10}"
            diff_str = f"{diff:>+10.2f}" if not np.isnan(diff) else f"{'N/A':>10}"
            ratio_str = f"{ratio:>7.2f}x" if not np.isnan(ratio) else f"{'N/A':>8}"

            row = f"{short_name:<20}  {dark_str}  {light_str}  {diff_str}  {ratio_str}"

            ax.text(0.02, y_pos, row, fontsize=7,
                   transform=ax.transAxes, color=self.TEXT_COLOR,
                   family='monospace', va='top')
            y_pos -= 0.07

            if y_pos < 0.05:  # Stop if running out of space
                break
