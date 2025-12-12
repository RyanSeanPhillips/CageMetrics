# CageMetrics

A PyQt6 application for analyzing behavioral data from Allentown cage monitoring systems. Part of the PhysioMetrics ecosystem.

## Features

- **Fast Data Loading**: Uses python-calamine for rapid Excel file loading
- **Per-Animal Analysis**: ZT (Zeitgeber Time) alignment and Cycle-Triggered Averages (CTA)
- **Behavioral Metrics**: Supports 11 behavioral metrics including:
  - Activity states (inactive, active, locomotion, climbing)
  - Consumptive behaviors (drinking, feeding)
  - Inferred sleep
  - Distance traveled and speed
  - Social distance
  - Respiration rate
- **Visualization**:
  - Summary page with metadata and data quality assessment
  - Stacked daily traces with 36-hour display (ZT0-ZT36)
  - CTA plots with SEM shading and data completeness overlay
  - Summary statistics with dark/light cycle comparisons
- **Data Export**: CSV export for all analyzed animals

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CageMetrics.git
   cd CageMetrics
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

1. Click **Browse** to select an Excel data file from Allentown cage monitoring system
2. Click **Analyze** to process the data
3. View results in per-animal tabs with scrollable figures
4. Click **Save Data** to export extracted datasets as CSV

## Requirements

- Python 3.9+
- PyQt6
- matplotlib
- numpy
- pandas
- scipy
- python-calamine (recommended for fast Excel loading)
- openpyxl (fallback Excel reader)

## Data Format

The application expects Excel files with the following columns:
- `animal.id` - Animal identifier
- `genotype` - Genotype information
- `light.cycle` - Light/Dark phase indicator
- `start` - Timestamp
- Behavioral metric columns (e.g., `activity-inactive.animal.percent.min`)

## Related Projects

- [PhysioMetrics](https://github.com/YOUR_USERNAME/PhysioMetrics) - Respiratory signal analysis

## License

MIT License
