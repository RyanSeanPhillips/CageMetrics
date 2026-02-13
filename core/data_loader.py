"""
Data loader module for Allentown behavioral data files.

Handles loading Excel and CSV files containing behavioral metrics from
Allentown cage monitoring systems.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoader:
    """Load and parse behavioral data files."""

    # Column name mappings for behavioral metrics
    METRIC_COLUMNS = {
        'Inactive %': 'activity-inactive.animal.percent.min',
        'Active %': 'activity-active.animal.percent.min',
        'Locomotion %': 'activity-locomotion.animal.percent.min',
        'Climbing %': 'activity-climbing.animal.percent.min',
        'Drinking %': 'activity-drinking.animal.percent.min',
        'Feeding %': 'activity-feeding.animal.percent.min',
        'Sleeping %': 'activity-inferred-sleeping.animal.percent.min',
        'Distance (cm/min)': 'distance-traveled.animal.cm.min',
        'Speed (cm/s)': 'activity.animal.cm_s.min',
        'Social Distance (cm)': 'social_distance_closest.animal.cm.min',
        'Respiration Rate (Hz)': 'respiratory-rate.animal.breaths-per-min.min',
    }

    def __init__(self):
        self.last_error = None

    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a data file (Excel or CSV).

        Args:
            file_path: Path to the data file

        Returns:
            DataFrame with loaded data, or None if loading failed
        """
        path = Path(file_path)

        if not path.exists():
            self.last_error = f"File not found: {file_path}"
            return None

        try:
            if path.suffix.lower() in ['.xlsx', '.xls']:
                # Try calamine engine first (much faster), fall back to openpyxl
                try:
                    import python_calamine  # noqa: F401
                    df = pd.read_excel(file_path, engine='calamine')
                except ImportError:
                    # calamine not installed, try openpyxl with column filtering
                    needed_columns = self._get_needed_columns()
                    df = pd.read_excel(
                        file_path,
                        engine='openpyxl',
                        usecols=lambda col: col in needed_columns or self._is_metric_column(col)
                    )
            elif path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                self.last_error = f"Unsupported file format: {path.suffix}"
                return None

            # Validate required columns
            if not self._validate_columns(df):
                return None

            return df

        except Exception as e:
            self.last_error = f"Failed to load file: {str(e)}"
            return None

    def _get_needed_columns(self) -> set:
        """Get set of non-metric columns needed for analysis."""
        return {
            'animal.id', 'genotype', 'light.cycle', 'start', 'sex',
            'cage.name', 'cage.id', 'cage', 'Cage',
            'birth.date', 'strain', 'group.name', 'study.code'
        }

    def _is_metric_column(self, col_name: str) -> bool:
        """Check if column is a metric column we need."""
        return col_name in self.METRIC_COLUMNS.values()

    def _validate_columns(self, df: pd.DataFrame) -> bool:
        """Check that required columns are present."""
        required = ['animal.id', 'genotype', 'light.cycle', 'start']

        missing = [col for col in required if col not in df.columns]
        if missing:
            self.last_error = f"Missing required columns: {missing}"
            return False

        # Check for at least one metric column
        found_metrics = [col for col in self.METRIC_COLUMNS.values() if col in df.columns]
        if not found_metrics:
            self.last_error = "No behavioral metric columns found"
            return False

        return True

    def get_animals(self, df: pd.DataFrame) -> list:
        """
        Get list of unique animal IDs in the dataset.

        Args:
            df: Loaded DataFrame

        Returns:
            List of animal ID strings
        """
        return df['animal.id'].dropna().unique().tolist()

    def get_animal_data(self, df: pd.DataFrame, animal_id: str) -> pd.DataFrame:
        """
        Extract data for a specific animal.

        Args:
            df: Full DataFrame
            animal_id: Animal identifier

        Returns:
            DataFrame filtered to specified animal, sorted by start time
        """
        animal_df = df[df['animal.id'] == animal_id].copy()
        animal_df = animal_df.sort_values('start').reset_index(drop=True)
        return animal_df

    def get_animal_metadata(self, animal_df: pd.DataFrame, cohort: str) -> Dict[str, Any]:
        """
        Extract metadata for an animal.

        Args:
            animal_df: DataFrame for single animal
            cohort: Cohort name

        Returns:
            Dictionary with animal metadata
        """
        metadata = {
            'animal_id': animal_df['animal.id'].iloc[0],
            'cohort': cohort,
            'genotype_raw': animal_df['genotype'].iloc[0],
            'genotype': self._classify_genotype(animal_df['genotype'].iloc[0]),
            'sex': animal_df.get('sex', pd.Series(['Unknown'])).iloc[0],
            'start_time': str(animal_df['start'].iloc[0]),
            'end_time': str(animal_df['start'].iloc[-1]),
            'total_minutes': len(animal_df),
            'total_days': len(animal_df) // 1440,
        }

        # Try to get cage ID if available - check cage.name first (most common)
        if 'cage.name' in animal_df.columns:
            metadata['cage_id'] = animal_df['cage.name'].iloc[0]
        elif 'cage.id' in animal_df.columns:
            metadata['cage_id'] = animal_df['cage.id'].iloc[0]
        elif 'cage' in animal_df.columns:
            metadata['cage_id'] = animal_df['cage'].iloc[0]
        else:
            metadata['cage_id'] = 'Unknown'

        # Extract birth date and compute age at recording start
        if 'birth.date' in animal_df.columns:
            birth_raw = animal_df['birth.date'].iloc[0]
            if pd.notna(birth_raw):
                try:
                    birth_dt = pd.to_datetime(birth_raw, utc=True)
                    start_dt = pd.to_datetime(animal_df['start'].iloc[0], utc=True)
                    age_days = (start_dt - birth_dt).days
                    metadata['dob'] = birth_dt.strftime('%Y-%m-%d')
                    metadata['age'] = f'P{age_days}'
                    metadata['age_days_at_start'] = age_days
                except Exception:
                    pass

        # Extract strain if available
        if 'strain' in animal_df.columns:
            strain_val = animal_df['strain'].iloc[0]
            if pd.notna(strain_val):
                metadata['strain'] = str(strain_val)

        # Extract group name if available
        if 'group.name' in animal_df.columns:
            group_val = animal_df['group.name'].iloc[0]
            if pd.notna(group_val):
                metadata['group_name'] = str(group_val)
        elif 'study.code' in animal_df.columns:
            study_val = animal_df['study.code'].iloc[0]
            if pd.notna(study_val):
                metadata['group_name'] = str(study_val)

        return metadata

    def _classify_genotype(self, genotype_str) -> str:
        """
        Classify genotype string into WT or DS.

        Args:
            genotype_str: Raw genotype string from data

        Returns:
            'DS' or 'WT'
        """
        if pd.isna(genotype_str):
            return 'Unknown'

        genotype_str = str(genotype_str)

        if 'MeoxCre+' in genotype_str:
            return 'DS'
        else:
            return 'WT'

    def get_available_metrics(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Get dictionary of available metrics in the dataset.

        Args:
            df: Loaded DataFrame

        Returns:
            Dictionary mapping display names to column names
        """
        available = {}
        for display_name, col_name in self.METRIC_COLUMNS.items():
            if col_name in df.columns:
                available[display_name] = col_name
        return available

    def compute_data_quality(self, animal_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Compute data quality metrics for each behavioral measure.

        Args:
            animal_df: DataFrame for single animal

        Returns:
            Dictionary mapping metric names to quality info
        """
        quality = {}

        for display_name, col_name in self.METRIC_COLUMNS.items():
            if col_name not in animal_df.columns:
                continue

            values = pd.to_numeric(animal_df[col_name], errors='coerce')
            total = len(values)
            missing = values.isna().sum()
            coverage = (total - missing) / total * 100 if total > 0 else 0

            # Determine quality rating
            if coverage >= 99:
                rating = 'OK'
            elif coverage >= 90:
                rating = 'WARN'
            else:
                rating = 'LOW'

            quality[display_name] = {
                'column': col_name,
                'total': total,
                'missing': missing,
                'coverage': coverage,
                'rating': rating
            }

        return quality
