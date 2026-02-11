"""
Consolidation filtering utilities for CageMetrics.

Provides dynamic filter discovery, criteria management, and cagemate genotype
resolution for the consolidation tab.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any, List


# Define which metadata fields are filterable and their display configuration
# 'primary' fields are shown by default, 'secondary' are in "More Filters"
FILTER_FIELDS = {
    # Primary filters (shown by default)
    'genotype': {
        'label': 'Genotype',
        'primary': True,
        'metadata_key': 'genotype',
    },
    'cagemate_genotype': {
        'label': 'Cagemate Genotype',
        'primary': True,
        'metadata_key': 'cagemate_genotype',
        'use_cache': True,
    },
    'sex': {
        'label': 'Sex',
        'primary': True,
        'metadata_key': 'sex',
    },
    'treatment': {
        'label': 'Treatment',
        'primary': True,
        'metadata_key': 'treatment',
    },
    # Secondary filters (in "More Filters")
    'dose': {
        'label': 'Dose',
        'primary': False,
        'metadata_key': 'dose',
    },
    'cohort': {
        'label': 'Cohort',
        'primary': False,
        'metadata_key': 'cohort',
    },
    'cage_id': {
        'label': 'Cage',
        'primary': False,
        'metadata_key': 'cage_id',
    },
    'age': {
        'label': 'Age',
        'primary': False,
        'metadata_key': 'age',
    },
    'strain': {
        'label': 'Strain',
        'primary': False,
        'metadata_key': 'strain',
    },
    'experimenter': {
        'label': 'Experimenter',
        'primary': False,
        'metadata_key': 'experimenter',
    },
    'experiment_date': {
        'label': 'Experiment Date',
        'primary': False,
        'metadata_key': 'experiment_date',
    },
    'group_name': {
        'label': 'Group Name',
        'primary': False,
        'metadata_key': 'group_name',
    },
    # Cagemate fields (for filtering by cagemate's properties)
    'cagemate_sex': {
        'label': 'Cagemate Sex',
        'primary': False,
        'metadata_key': 'cagemate_sex',
    },
    'cagemate_treatment': {
        'label': 'Cagemate Treatment',
        'primary': False,
        'metadata_key': 'cagemate_treatment',
    },
}


@dataclass
class FilterCriteria:
    """Dynamic filter criteria for consolidation file selection.

    Supports any metadata field. Empty sets mean "all values accepted".
    """

    # Dictionary mapping field name to set of accepted values
    # e.g., {'genotype': {'WT', 'DS'}, 'sex': {'Male'}}
    filters: Dict[str, Set[str]] = field(default_factory=dict)

    def set_filter(self, field_name: str, values: Set[str]):
        """Set filter for a specific field."""
        if values:
            self.filters[field_name] = values
        elif field_name in self.filters:
            del self.filters[field_name]

    def get_filter(self, field_name: str) -> Set[str]:
        """Get filter values for a field (empty set if no filter)."""
        return self.filters.get(field_name, set())

    def matches(self, metadata: Dict[str, Any],
                cagemate_genotype: Optional[str] = None) -> bool:
        """Check if metadata matches all active filter criteria.

        Args:
            metadata: Animal metadata dictionary
            cagemate_genotype: Resolved genotype of cagemate (or None/Single)

        Returns:
            True if all active filters pass
        """
        for field_name, accepted_values in self.filters.items():
            if not accepted_values:
                continue  # Empty filter = accept all

            # Get the value to check
            if field_name == 'cagemate_genotype':
                # Special handling: use resolved cagemate_genotype
                # First check if it's in metadata (new files), then use cache value
                value = metadata.get('cagemate_genotype')
                if value is None:
                    value = cagemate_genotype if cagemate_genotype else "Single"
            else:
                # Get from metadata
                value = metadata.get(FILTER_FIELDS.get(field_name, {}).get('metadata_key', field_name), 'Unknown')

            # Normalize None to string
            if value is None:
                value = 'Unknown'

            # Handle list values (e.g., multiple cagemates)
            if isinstance(value, list):
                # Match if ANY value in the list is accepted
                if not any(v in accepted_values for v in value):
                    return False
            else:
                if str(value) not in accepted_values:
                    return False

        return True

    def is_empty(self) -> bool:
        """Check if no filters are active."""
        return not any(self.filters.values())

    def to_description(self) -> str:
        """Generate human-readable description of active filters."""
        parts = []

        for field_name, values in self.filters.items():
            if values:
                field_config = FILTER_FIELDS.get(field_name, {})
                label = field_config.get('label', field_name)
                values_str = ', '.join(sorted(values))
                parts.append(f"{label}: {values_str}")

        return " | ".join(parts) if parts else "No filters applied"

    def clear(self):
        """Clear all filters."""
        self.filters.clear()

    # Legacy property accessors for compatibility
    @property
    def animal_genotypes(self) -> Set[str]:
        return self.filters.get('genotype', set())

    @animal_genotypes.setter
    def animal_genotypes(self, value: Set[str]):
        self.set_filter('genotype', value)

    @property
    def cagemate_genotypes(self) -> Set[str]:
        return self.filters.get('cagemate_genotype', set())

    @cagemate_genotypes.setter
    def cagemate_genotypes(self, value: Set[str]):
        self.set_filter('cagemate_genotype', value)

    @property
    def sexes(self) -> Set[str]:
        return self.filters.get('sex', set())

    @sexes.setter
    def sexes(self, value: Set[str]):
        self.set_filter('sex', value)


class MetadataDiscovery:
    """Discover available metadata values from loaded NPZ files.

    Scans all loaded files to find unique values for each filterable field,
    enabling dynamic filter UI generation.
    """

    def __init__(self):
        # field_name -> set of discovered values
        self._discovered_values: Dict[str, Set[str]] = {}
        # Count of files scanned
        self._file_count: int = 0

    def scan_files(self, npz_items: List[Dict[str, Any]],
                   cagemate_cache: 'CagemateGenotypeCache' = None) -> None:
        """Scan NPZ items to discover available metadata values.

        Args:
            npz_items: List of dicts with 'path' and 'metadata' keys
            cagemate_cache: Optional cache for resolving cagemate genotypes
        """
        self._discovered_values.clear()
        self._file_count = len(npz_items)

        for field_name in FILTER_FIELDS.keys():
            self._discovered_values[field_name] = set()

        for item_data in npz_items:
            metadata = item_data.get('metadata', {})

            for field_name, field_config in FILTER_FIELDS.items():
                metadata_key = field_config.get('metadata_key', field_name)

                if field_name == 'cagemate_genotype':
                    # Special handling for cagemate genotype
                    # First check metadata (newer files have it)
                    value = metadata.get('cagemate_genotype')
                    if value is None and cagemate_cache:
                        # Fall back to cache resolution
                        value = cagemate_cache.get_cagemate_genotype(metadata)
                    if value is None:
                        value = 'Single'
                else:
                    value = metadata.get(metadata_key)

                if value is not None:
                    # Handle list values
                    if isinstance(value, list):
                        for v in value:
                            if v is not None:
                                self._discovered_values[field_name].add(str(v))
                    else:
                        self._discovered_values[field_name].add(str(value))

        # Add "Unknown" for fields that might have missing values
        for field_name in ['genotype', 'sex', 'cohort']:
            if field_name in self._discovered_values:
                # Don't add Unknown if not needed
                pass

    def get_values(self, field_name: str) -> List[str]:
        """Get sorted list of discovered values for a field."""
        values = self._discovered_values.get(field_name, set())
        return sorted(values)

    def get_primary_fields(self) -> List[str]:
        """Get list of primary (default visible) filter fields."""
        return [name for name, config in FILTER_FIELDS.items() if config.get('primary', False)]

    def get_secondary_fields(self) -> List[str]:
        """Get list of secondary (more options) filter fields."""
        return [name for name, config in FILTER_FIELDS.items() if not config.get('primary', False)]

    def get_field_label(self, field_name: str) -> str:
        """Get display label for a field."""
        return FILTER_FIELDS.get(field_name, {}).get('label', field_name)

    def has_multiple_values(self, field_name: str) -> bool:
        """Check if a field has more than one unique value (worth filtering)."""
        values = self._discovered_values.get(field_name, set())
        return len(values) > 1

    def get_file_count(self) -> int:
        """Get number of files scanned."""
        return self._file_count


class CagemateGenotypeCache:
    """Cache for resolving cagemate genotypes from NPZ metadata.

    During folder scan, all animal_id -> genotype mappings are collected.
    Then for any animal, the cagemate's genotype can be looked up.
    """

    def __init__(self):
        self._animal_genotypes: Dict[str, str] = {}  # animal_id -> genotype
        self._animal_files: Dict[str, str] = {}  # animal_id -> npz_path

    def build_from_files(self, npz_items: List[Dict[str, Any]]) -> None:
        """Build lookup cache from list of NPZ item data.

        Args:
            npz_items: List of dicts with 'path' and 'metadata' keys
        """
        self._animal_genotypes.clear()
        self._animal_files.clear()

        for item_data in npz_items:
            metadata = item_data.get('metadata', {})
            animal_id = metadata.get('animal_id')
            genotype = metadata.get('genotype', 'Unknown')
            path = item_data.get('path', '')

            if animal_id:
                # Normalize animal_id to string
                animal_id_str = str(animal_id)
                self._animal_genotypes[animal_id_str] = genotype
                self._animal_files[animal_id_str] = path

    def get_cagemate_genotype(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Get genotype of cagemate for an animal.

        Args:
            metadata: Animal metadata dictionary with 'companion' field

        Returns:
            Cagemate genotype string, or None if single-housed or not found
        """
        # First check if cagemate_genotype is already in metadata
        if 'cagemate_genotype' in metadata:
            return metadata['cagemate_genotype']

        companion = metadata.get('companion')

        if companion is None:
            return None

        # Handle list of companions (take first one)
        if isinstance(companion, list):
            if not companion:
                return None
            companion = companion[0]

        # Handle "None" string
        if companion == 'None' or companion == 'Single':
            return None

        # Normalize to string and lookup
        companion_str = str(companion)
        return self._animal_genotypes.get(companion_str)

    def get_all_genotypes(self) -> Set[str]:
        """Get all unique genotypes in the cache."""
        return set(self._animal_genotypes.values())

    def get_animal_count(self) -> int:
        """Get number of animals in cache."""
        return len(self._animal_genotypes)
