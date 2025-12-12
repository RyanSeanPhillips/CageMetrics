# Allentown Behavioral Analysis App - Design Sketch

## Overall App Structure

```
+==============================================================================+
|  Allentown Behavioral Analysis                                    [_][O][X]  |
+==============================================================================+
|  [ Analysis ]  [ Consolidation ]                                             |
+------------------------------------------------------------------------------+
|                                                                              |
|                        << ANALYSIS TAB CONTENT >>                            |
|                                                                              |
+------------------------------------------------------------------------------+
|  Status: Ready                                                               |
+==============================================================================+
```

---

## Analysis Tab Layout

```
+==============================================================================+
|                              ANALYSIS TAB                                    |
+==============================================================================+
|                                                                              |
|  +-- File Selection Bar ------------------------------------------------+   |
|  |                                                                       |   |
|  |  [ Browse... ]  | C:\path\to\Cohort4_1MinuteBouts.xlsx               |   |
|  |                                                                       |   |
|  +-----------------------------------------------------------------------+   |
|                                                                              |
|  +-- Action Buttons ----------------------------------------------------+   |
|  |                                                                       |   |
|  |  [ Analyze ]     [ Save Data ]     [ Export Figures ]                |   |
|  |                                                                       |   |
|  +-----------------------------------------------------------------------+   |
|                                                                              |
|  +-- Animal Tabs -------------------------------------------------------+   |
|  |                                                                       |   |
|  |  [ K5757 (WT) ]  [ K5752 (DS) ]                                      |   |
|  |  +---------------------------------------------------------------+   |   |
|  |  |                                                                |   |   |
|  |  |  +-- Page Navigation ----------------------------------------+ |   |   |
|  |  |  |  [ < Prev ]  Page 1 of 4: Summary  [ Next > ]             | |   |   |
|  |  |  +-----------------------------------------------------------+ |   |   |
|  |  |                                                                |   |   |
|  |  |  +-- Matplotlib Canvas (Scrollable) ------------------------+ |   |   |
|  |  |  |                                                           | |   |   |
|  |  |  |                                                           | |   |   |
|  |  |  |              << FIGURE CONTENT HERE >>                    | |   |   |
|  |  |  |                                                           | |   |   |
|  |  |  |                                                           | |   |   |
|  |  |  +-----------------------------------------------------------+ |   |   |
|  |  |                                                                |   |   |
|  |  +---------------------------------------------------------------+   |   |
|  |                                                                       |   |
|  +-----------------------------------------------------------------------+   |
|                                                                              |
+==============================================================================+
```

---

## Page 1: Animal Summary & Metadata

Shows animal information, recording details, and data quality for all 11 metrics.

- Animal ID, Genotype (WT/DS), Sex, Cohort, Cage ID, Companion
- Start/End times, Days analyzed, ZT0 offset
- Data quality table with coverage %, missing points, and quality rating

---

## Page 2: Stacked Daily Traces (ZT-Aligned)

Shows all 11 metrics with daily traces overlaid in different colors.
- X-axis: ZT0 to ZT24
- Dark phase (ZT12-24) shaded
- Up to 7 days shown with color legend

---

## Page 3: CTA (Cycle-Triggered Averages)

48-hour display (2 full cycles) showing mean ± SEM across days.
- 30-minute rolling average smoothing
- Dark phases shaded
- All 11 metrics shown

---

## Page 4: Summary Statistics

Bar charts comparing Dark vs Light cycle means for each metric.
- Grouped bars (Dark/Light)
- Summary table with Dark Mean, Light Mean, Difference, Ratio

---

## Exported Files (per animal)

```
{AnimalID}_{Genotype}_{Sex}_{Cohort}_CTA.csv         # 1440 rows, per-minute CTAs
{AnimalID}_{Genotype}_{Sex}_{Cohort}_DailyMeans.csv  # Summary statistics
{AnimalID}_{Genotype}_{Sex}_{Cohort}_Metadata.json   # Animal info + data quality
```

---

## Color Theme (Dark Mode)

```
Background:       #1e1e1e (dark gray)
Panel Background: #2d2d2d (slightly lighter)
Text:             #ffffff (white)
Accent:           #3daee9 (blue)
Success:          #27ae60 (green)
Warning:          #f39c12 (orange)
Error:            #e74c3c (red)
```

---

## Implementation Status

- [x] Main app structure with tabs
- [x] Dark theme stylesheet
- [x] File browser with directory memory
- [x] Data loader for Excel files
- [x] ZT alignment algorithm
- [x] CTA computation
- [x] Summary page with metadata table
- [x] Daily traces page
- [x] CTA plots page
- [x] Statistics page
- [x] CSV/JSON export
- [ ] Consolidation tab (future)
