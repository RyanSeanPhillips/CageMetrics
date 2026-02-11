"""
Metadata management for CageMetrics.

Defines standard and extended metadata fields, provides metadata editor dialog,
and handles cagemate metadata storage.
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field


# === Standard Metadata Fields ===
# These are automatically extracted from data files

STANDARD_FIELDS = {
    'animal_id': {
        'label': 'Animal ID',
        'editable': False,  # Auto-extracted
        'type': 'str',
    },
    'genotype': {
        'label': 'Genotype',
        'editable': True,
        'type': 'str',
    },
    'genotype_raw': {
        'label': 'Genotype (Raw)',
        'editable': False,
        'type': 'str',
    },
    'sex': {
        'label': 'Sex',
        'editable': True,
        'type': 'str',
        'options': ['Male', 'Female', 'Unknown'],
    },
    'cohort': {
        'label': 'Cohort',
        'editable': True,
        'type': 'str',
    },
    'cage_id': {
        'label': 'Cage ID',
        'editable': True,
        'type': 'str',
    },
    'companion': {
        'label': 'Cagemate ID',
        'editable': False,
        'type': 'str',
    },
    'cagemate_genotype': {
        'label': 'Cagemate Genotype',
        'editable': False,  # Computed from cagemate
        'type': 'str',
    },
    'start_time': {
        'label': 'Start Time',
        'editable': False,
        'type': 'str',
    },
    'end_time': {
        'label': 'End Time',
        'editable': False,
        'type': 'str',
    },
    'n_days_analyzed': {
        'label': 'Days Analyzed',
        'editable': False,
        'type': 'int',
    },
    'total_minutes': {
        'label': 'Total Minutes',
        'editable': False,
        'type': 'int',
    },
}


# === Extended Metadata Fields ===
# These are user-editable fields for additional experiment information

EXTENDED_FIELDS = {
    'treatment': {
        'label': 'Treatment',
        'editable': True,
        'type': 'str',
        'description': 'Treatment condition (e.g., Vehicle, Drug A)',
        'filterable': True,
        'primary_filter': True,
    },
    'dose': {
        'label': 'Dose',
        'editable': True,
        'type': 'str',
        'description': 'Treatment dose (e.g., 10 mg/kg)',
        'filterable': True,
        'primary_filter': False,
    },
    'age': {
        'label': 'Age',
        'editable': True,
        'type': 'str',
        'description': 'Age at experiment (e.g., 12 weeks, P30)',
        'filterable': True,
        'primary_filter': False,
    },
    'dob': {
        'label': 'Date of Birth',
        'editable': True,
        'type': 'str',
        'description': 'Date of birth (YYYY-MM-DD)',
        'filterable': False,
        'primary_filter': False,
    },
    'group_name': {
        'label': 'Group Name',
        'editable': True,
        'type': 'str',
        'description': 'Study group name (e.g., 10XDS+WT)',
        'filterable': True,
        'primary_filter': False,
    },
    'experiment_date': {
        'label': 'Experiment Date',
        'editable': True,
        'type': 'str',
        'description': 'Date experiment was performed (YYYY-MM-DD)',
        'filterable': True,
        'primary_filter': False,
    },
    'experimenter': {
        'label': 'Experimenter',
        'editable': True,
        'type': 'str',
        'description': 'Name or initials of experimenter',
        'filterable': True,
        'primary_filter': False,
    },
    'strain': {
        'label': 'Strain',
        'editable': True,
        'type': 'str',
        'description': 'Mouse strain (e.g., C57BL/6J, Ts65Dn)',
        'filterable': True,
        'primary_filter': False,
    },
    'weight': {
        'label': 'Weight (g)',
        'editable': True,
        'type': 'str',
        'description': 'Body weight in grams',
        'filterable': False,
        'primary_filter': False,
    },
    'notes': {
        'label': 'Notes',
        'editable': True,
        'type': 'str',
        'description': 'Additional notes',
        'filterable': False,
        'primary_filter': False,
    },
}


# Combined field definitions
ALL_FIELDS = {**STANDARD_FIELDS, **EXTENDED_FIELDS}


def get_all_filterable_fields() -> Dict[str, Dict]:
    """Get all fields that can be used for filtering.

    Returns dict suitable for use in consolidation_filters.FILTER_FIELDS
    """
    filterable = {}

    # Standard filterable fields
    for name, config in STANDARD_FIELDS.items():
        if name in ['genotype', 'sex', 'cohort', 'cage_id', 'cagemate_genotype']:
            filterable[name] = {
                'label': config['label'],
                'primary': name in ['genotype', 'sex', 'cagemate_genotype'],
                'metadata_key': name,
            }

    # Extended filterable fields
    for name, config in EXTENDED_FIELDS.items():
        if config.get('filterable', False):
            filterable[name] = {
                'label': config['label'],
                'primary': config.get('primary_filter', False),
                'metadata_key': name,
            }

    # Add cagemate metadata fields (prefixed with cagemate_)
    cagemate_filterable = ['genotype', 'sex', 'treatment', 'dose', 'age', 'strain']
    for name in cagemate_filterable:
        cagemate_key = f'cagemate_{name}'
        if name == 'genotype':
            continue  # Already have cagemate_genotype
        if name in STANDARD_FIELDS:
            label = STANDARD_FIELDS[name]['label']
        elif name in EXTENDED_FIELDS:
            label = EXTENDED_FIELDS[name]['label']
        else:
            continue
        filterable[cagemate_key] = {
            'label': f'Cagemate {label}',
            'primary': False,
            'metadata_key': cagemate_key,
        }

    return filterable


def get_editable_fields() -> List[str]:
    """Get list of field names that are user-editable."""
    editable = []
    for name, config in ALL_FIELDS.items():
        if config.get('editable', False):
            editable.append(name)
    return editable


def get_display_columns() -> List[str]:
    """Get ordered list of columns for metadata table display."""
    # Priority order for display
    priority = [
        'animal_id', 'genotype', 'sex', 'cohort', 'cage_id',
        'treatment', 'dose', 'age', 'strain',
        'companion', 'cagemate_genotype',
        'experiment_date', 'experimenter',
        'n_days_analyzed', 'notes'
    ]

    # Include all fields, priority first
    columns = []
    for name in priority:
        if name in ALL_FIELDS:
            columns.append(name)

    # Add any remaining fields
    for name in ALL_FIELDS:
        if name not in columns:
            columns.append(name)

    return columns


def copy_cagemate_metadata(source_metadata: Dict[str, Any],
                           cagemate_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Copy relevant cagemate metadata fields into source metadata.

    Args:
        source_metadata: The animal's metadata dict (modified in place)
        cagemate_metadata: The cagemate's metadata dict

    Returns:
        Modified source_metadata with cagemate_ prefixed fields added
    """
    # Fields to copy from cagemate
    fields_to_copy = [
        'genotype', 'sex', 'treatment', 'dose', 'age', 'strain',
        'weight', 'dob', 'cohort'
    ]

    for field_name in fields_to_copy:
        cagemate_key = f'cagemate_{field_name}'
        value = cagemate_metadata.get(field_name)
        if value is not None:
            source_metadata[cagemate_key] = value

    return source_metadata


@dataclass
class MetadataField:
    """Represents a single metadata field definition."""
    name: str
    label: str
    editable: bool = True
    field_type: str = 'str'
    options: List[str] = field(default_factory=list)
    description: str = ''
    filterable: bool = False
    primary_filter: bool = False


class MetadataManager:
    """Manages metadata for analyzed experiments.

    Provides methods to:
    - Get/set metadata values
    - Add custom fields
    - Copy cagemate metadata
    - Validate metadata
    """

    def __init__(self):
        # Custom fields added by user (not in standard/extended)
        self._custom_fields: Dict[str, Dict] = {}

    def add_custom_field(self, name: str, label: str = None,
                         filterable: bool = True) -> None:
        """Add a custom metadata field.

        Args:
            name: Field name (lowercase, no spaces)
            label: Display label (defaults to name.title())
            filterable: Whether field should appear in consolidation filters
        """
        clean_name = name.lower().replace(' ', '_')
        self._custom_fields[clean_name] = {
            'label': label or name.replace('_', ' ').title(),
            'editable': True,
            'type': 'str',
            'filterable': filterable,
            'primary_filter': False,
            'custom': True,
        }

    def get_custom_fields(self) -> Dict[str, Dict]:
        """Get all custom fields."""
        return self._custom_fields.copy()

    def get_all_field_definitions(self) -> Dict[str, Dict]:
        """Get all field definitions including custom fields."""
        return {**ALL_FIELDS, **self._custom_fields}

    def update_experiment_metadata(self, analyzed_data: Dict[str, Dict],
                                   animal_id: str, field_name: str,
                                   value: Any) -> None:
        """Update a metadata field for an experiment.

        Args:
            analyzed_data: The full analyzed_data dict
            animal_id: Animal identifier
            field_name: Metadata field name
            value: New value
        """
        if animal_id in analyzed_data:
            metadata = analyzed_data[animal_id].get('metadata', {})
            metadata[field_name] = value
            analyzed_data[animal_id]['metadata'] = metadata

    def propagate_cagemate_metadata(self, analyzed_data: Dict[str, Dict]) -> None:
        """Copy cagemate metadata to all experiments.

        Should be called after metadata edits to ensure cagemate fields are updated.
        """
        # Build animal_id -> metadata lookup
        metadata_lookup = {}
        for animal_id, data in analyzed_data.items():
            metadata_lookup[animal_id] = data.get('metadata', {})

        # Update each animal with cagemate metadata
        for animal_id, data in analyzed_data.items():
            metadata = data.get('metadata', {})
            companion = metadata.get('companion')

            if companion:
                # Handle list of companions
                if isinstance(companion, list):
                    companion_id = companion[0] if companion else None
                else:
                    companion_id = companion

                if companion_id and companion_id in metadata_lookup:
                    cagemate_meta = metadata_lookup[companion_id]
                    copy_cagemate_metadata(metadata, cagemate_meta)

    def get_unique_values(self, analyzed_data: Dict[str, Dict],
                          field_name: str) -> Set[str]:
        """Get all unique values for a metadata field across experiments."""
        values = set()
        for animal_id, data in analyzed_data.items():
            metadata = data.get('metadata', {})
            value = metadata.get(field_name)
            if value is not None:
                if isinstance(value, list):
                    values.update(str(v) for v in value)
                else:
                    values.add(str(value))
        return values


# Global metadata manager instance
metadata_manager = MetadataManager()
