# CageMetrics - Behavioral Analysis Application

## Overview

CageMetrics is a PyQt6-based desktop application for analyzing behavioral data from home cage monitoring systems. It processes exported NPZ files from PhysioMetrics (or similar tools) and provides:

- **Analyze**: Individual animal analysis with sleep bout detection
- **Group**: Combine multiple animal datasets into population-level summaries with filtering
- **Compare**: Compare multiple grouped datasets side-by-side with statistics
- **Visualization**: Generate publication-ready figures with circadian time averages (CTAs)
- **Statistics**: Automatic statistical analysis with t-tests, ANOVA, and Bonferroni correction

## Project Structure

```
CageMetrics/
├── main.py                     # Main application entry point
├── run_debug.py                # Debug launcher
├── run_app.bat                 # Windows launcher (activates conda env)
├── CLAUDE.md                   # This file - project overview
│
├── core/                       # Core modules
│   ├── consolidator.py         # NPZ file consolidation logic
│   ├── consolidation_filters.py # Filter criteria and metadata discovery
│   ├── consolidation_figure_generator.py  # Figure generation for consolidation
│   ├── comparison_figure_generator.py     # Figure generation for comparisons
│   └── data_exporter.py        # Data export functionality
│
└── requirements.txt            # Python dependencies (shares with PhysioMetrics)
```

## Key Classes

### main.py
- `CageMetricsApp`: Main application window with tab interface (Analyze, Group, Compare)
- `AnalysisTab`: Individual animal analysis with sleep bout detection
- `ConsolidationTab`: Group tab - loads NPZ files, applies filters, generates consolidated previews
- `ComparisonTab`: Compare tab - loads consolidated NPZ files, compares datasets, generates statistics

### core/consolidator.py
- `Consolidator`: Combines multiple animal NPZ files into consolidated outputs
- Generates Excel workbooks, NPZ files, and PDF figures

### core/consolidation_filters.py
- `FilterCriteria`: Dataclass for storing filter selections
- `MetadataDiscovery`: Scans files to discover available filter values
- `CagemateGenotypeCache`: Resolves cagemate genotypes from companion IDs

### core/comparison_figure_generator.py
- `ComparisonFigureGenerator`: Creates comparison figures
- `load_consolidated_npz()`: Utility to load consolidated NPZ files

## Running the Application

```bash
# Method 1: Using batch script (Windows)
run_app.bat

# Method 2: Using Python directly
conda activate plethapp
python run_debug.py
```

## NPZ File Format

### Individual Animal NPZ (from PhysioMetrics)
- `metadata_json`: Animal metadata (ID, genotype, sex, etc.)
- `quality_json`: Data quality metrics
- `metric_names`: List of metric names
- `{metric}_cta`: Circadian time average (1440 values, one per minute)
- `{metric}_cta_sem`: SEM for each minute
- `{metric}_dark_mean`, `{metric}_light_mean`: Phase means
- `{metric}_daily`: Daily data matrix

### Consolidated NPZ (from CageMetrics)
- `consolidation_metadata`: Filter criteria, source files, date
- `animal_metadata`: Array of per-animal metadata
- `metric_names`: Common metrics across all animals
- `{metric}_grand_cta`: Population mean CTA
- `{metric}_grand_sem`: Population SEM
- `{metric}_all_ctas`: Matrix of all animal CTAs (n_animals, 1440)
- `{metric}_dark_means`, `{metric}_light_means`: Arrays of per-animal means

## Recent Features

### Compare Tab Statistics (Latest)
- Checkbox to toggle statistics on/off
- Welch's t-test for 2-group comparisons
- ANOVA + Bonferroni-corrected pairwise t-tests for 3+ groups
- Significance brackets on bar charts (*, **, ***)
- Statistics summary page in PDF output
- CSV export of all statistical results
- Sleep analysis comparison page (histograms, CTA, bar charts)

### Group Tab
- Advanced filtering by genotype, cagemate genotype, sex, treatment
- Preview generation with scrollable figures
- Full-tab scrolling (filters scroll out of view)
- Sleep data aggregation for comparison

## Dependencies

Uses the same conda environment as PhysioMetrics (`plethapp`):
- PyQt6
- matplotlib
- numpy
- pandas
- scipy (for statistics)

## Related Projects

- **PhysioMetrics**: The breath analysis application that exports NPZ files consumed by CageMetrics
- Located at: `C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\`

## Development Notes

- Dark theme by default, light mode option for publication figures
- Error bars are white in dark mode for visibility
- Output folder structure:
  - `analyzed/` - Individual animal exports from Analyze tab
  - `grouped/` - Consolidated group files from Group tab
  - `compared/` - Comparison outputs from Compare tab
- Statistics CSV includes all pairwise comparisons with means, SEMs, p-values
