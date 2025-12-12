"""
Data exporter module for saving analyzed behavioral data.

Exports per-animal data to CSV files including:
- CTA (cycle-triggered average) data
- Daily means (dark/light cycle)
- Metadata (JSON)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


class DataExporter:
    """Export analyzed behavioral data to files."""

    def __init__(self):
        pass

    def export_all_animals(self, analyzed_data: Dict[str, Dict[str, Any]],
                          output_dir: str) -> List[str]:
        """
        Export data for all analyzed animals.

        Args:
            analyzed_data: Dictionary mapping animal_id to analysis results
            output_dir: Directory to save files

        Returns:
            List of saved file paths
        """
        saved_files = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for animal_id, animal_data in analyzed_data.items():
            files = self.export_animal(animal_id, animal_data, output_path)
            saved_files.extend(files)

        return saved_files

    def export_animal(self, animal_id: str, animal_data: Dict[str, Any],
                     output_dir: Path) -> List[str]:
        """
        Export data for a single animal.

        Args:
            animal_id: Animal identifier
            animal_data: Analysis results for this animal
            output_dir: Directory to save files

        Returns:
            List of saved file paths
        """
        saved_files = []

        metadata = animal_data.get('metadata', {})
        genotype = metadata.get('genotype', 'Unknown')
        sex = metadata.get('sex', 'Unknown')
        cohort = metadata.get('cohort', 'Unknown')

        # Create filename base
        base_name = f"{animal_id}_{genotype}_{sex}_{cohort}"

        # Export CTA data
        cta_file = self._export_cta(animal_data, output_dir, base_name)
        if cta_file:
            saved_files.append(str(cta_file))

        # Export daily means
        means_file = self._export_daily_means(animal_data, output_dir, base_name)
        if means_file:
            saved_files.append(str(means_file))

        # Export metadata
        meta_file = self._export_metadata(animal_data, output_dir, base_name)
        if meta_file:
            saved_files.append(str(meta_file))

        return saved_files

    def _export_cta(self, animal_data: Dict[str, Any], output_dir: Path,
                    base_name: str) -> Path:
        """Export CTA data to CSV."""
        metrics = animal_data.get('metrics', {})

        if not metrics:
            return None

        # Build DataFrame with ZT minute as index
        data = {'zt_minute': list(range(1440))}

        for metric_name, metric_data in metrics.items():
            cta = metric_data.get('cta', None)
            cta_sem = metric_data.get('cta_sem', None)

            if cta is not None:
                # Clean column name for CSV
                col_name = metric_name.replace(' ', '_').replace('/', '_per_')
                data[col_name] = cta
                data[f'{col_name}_SEM'] = cta_sem

        df = pd.DataFrame(data)

        # Add ZT hour for convenience
        df.insert(1, 'zt_hour', df['zt_minute'] / 60)

        # Add light/dark label
        df['light_cycle'] = df['zt_minute'].apply(lambda x: 'Light' if x < 720 else 'Dark')

        # Save
        output_file = output_dir / f"{base_name}_CTA.csv"
        df.to_csv(output_file, index=False)

        return output_file

    def _export_daily_means(self, animal_data: Dict[str, Any], output_dir: Path,
                           base_name: str) -> Path:
        """Export daily means to CSV."""
        metrics = animal_data.get('metrics', {})
        metadata = animal_data.get('metadata', {})

        if not metrics:
            return None

        rows = []
        for metric_name, metric_data in metrics.items():
            row = {
                'animal_id': metadata.get('animal_id', ''),
                'genotype': metadata.get('genotype', ''),
                'sex': metadata.get('sex', ''),
                'cohort': metadata.get('cohort', ''),
                'metric': metric_name,
                'dark_mean': metric_data.get('dark_mean', np.nan),
                'light_mean': metric_data.get('light_mean', np.nan),
                'overall_mean': metric_data.get('overall_mean', np.nan),
                'overall_std': metric_data.get('overall_std', np.nan),
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save
        output_file = output_dir / f"{base_name}_DailyMeans.csv"
        df.to_csv(output_file, index=False)

        return output_file

    def _export_metadata(self, animal_data: Dict[str, Any], output_dir: Path,
                        base_name: str) -> Path:
        """Export metadata to JSON."""
        metadata = animal_data.get('metadata', {})
        quality = animal_data.get('quality', {})

        # Convert quality dict to serializable format
        quality_export = {}
        for metric_name, q_info in quality.items():
            quality_export[metric_name] = {
                'coverage_percent': q_info.get('coverage', 0),
                'missing_points': q_info.get('missing', 0),
                'quality_rating': q_info.get('rating', 'Unknown')
            }

        export_data = {
            'animal_info': {
                'animal_id': metadata.get('animal_id', ''),
                'genotype': metadata.get('genotype', ''),
                'genotype_raw': metadata.get('genotype_raw', ''),
                'sex': metadata.get('sex', ''),
                'cohort': metadata.get('cohort', ''),
                'cage_id': metadata.get('cage_id', ''),
                'companion': metadata.get('companion', None),
            },
            'recording_info': {
                'start_time': metadata.get('start_time', ''),
                'end_time': metadata.get('end_time', ''),
                'total_minutes': metadata.get('total_minutes', 0),
                'days_analyzed': metadata.get('n_days_analyzed', 0),
                'zt0_minute': metadata.get('zt0_minute', 0),
            },
            'data_quality': quality_export
        }

        # Save
        output_file = output_dir / f"{base_name}_Metadata.json"
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        return output_file
