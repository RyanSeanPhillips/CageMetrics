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
                    save_pdf: bool = True) -> Dict[str, Any]:
        """
        Consolidate multiple NPZ files into NPZ, Excel, and PDF formats.

        Args:
            npz_paths: List of paths to NPZ data files
            output_path: Base path for output files (extension determines format)
            filter_criteria: Optional FilterCriteria used for selection
            save_npz: Also save consolidated NPZ file (default: True)
            save_pdf: Also save PDF figures (default: True)

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

            # Tabs 3+: Per-metric CTA data
            for metric_name in all_metrics:
                self._write_metric_cta_tab(writer, animals, metric_name)

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
            self._write_consolidated_pdf(pdf_path, animals, filter_desc)
            results['output_paths'].append(str(pdf_path))
            results['pdf_path'] = str(pdf_path)

        return results

    def _write_consolidated_pdf(self, pdf_path: Path, animals: List[Dict],
                                 filter_description: str):
        """
        Write consolidated figures to PDF.

        Args:
            pdf_path: Output PDF path
            animals: List of animal data dicts
            filter_description: Human-readable filter description
        """
        from matplotlib.backends.backend_pdf import PdfPages
        from core.consolidation_figure_generator import ConsolidationFigureGenerator

        generator = ConsolidationFigureGenerator()
        pages = generator.generate_all_pages(animals, filter_description)

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

        # Save NPZ
        np.savez_compressed(str(npz_path), **data)

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
                    metrics[metric_name]['daily'] = data[daily_key]

            return {
                'metadata': metadata,
                'quality': quality,
                'metric_names': metric_names,
                'metrics': metrics
            }

        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _write_summary_tab(self, writer: pd.ExcelWriter, animals: List[Dict]):
        """Write summary of all animals."""
        rows = []
        for animal in animals:
            meta = animal.get('metadata', {})
            rows.append({
                'Animal ID': meta.get('animal_id', ''),
                'Genotype': meta.get('genotype', ''),
                'Genotype (Raw)': meta.get('genotype_raw', ''),
                'Sex': meta.get('sex', ''),
                'Cohort': meta.get('cohort', ''),
                'Cage ID': meta.get('cage_id', ''),
                'Cagemate': meta.get('companion', ''),
                'Days Analyzed': meta.get('n_days_analyzed', 0),
                'Total Minutes': meta.get('total_minutes', 0),
                'Source File': animal.get('source_file', '')
            })

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
