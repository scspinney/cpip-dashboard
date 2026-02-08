# C-PIP Dashboard

Refactored pipeline for generating C-PIP demographic/MRI dashboard figures from REDCap CSV exports.

## What This Repo Does

Main entrypoint:

- `src/pipeline.py`

The pipeline:

1. Loads site REDCap exports (Montreal, Calgary, Toronto).
2. Selects one best baseline (`t1_arm_1`) row per participant.
3. Derives demographics (age, effective sex, gender label, income, ethnicity).
4. Builds dashboard and standalone figures.
5. Optionally builds target-vs-observed overlays if age-by-group target CSVs exist.
6. Combines preferred figures into a single PDF.

## Requirements

- Python `>=3.10`.
- Runtime dependencies are pinned in `pyproject.toml`.

Create environment and install (local checkout):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Install directly from GitHub:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/scspinney/cpip-dashboard.git@main"
```

Notes:

- If you run into matplotlib cache permission warnings, set:
  - `export MPLCONFIGDIR=$(pwd)/.mplconfig`

## Input Files Needed

### Required REDCap Site Exports

You need REDCap CSV exports containing at least:

- `record_id`
- `redcap_repeat_instrument`
- `redcap_event_name`

For auto-discovery, filenames must include site names:

- `*montreal*.csv`
- `*calgary*.csv`
- `*toronto*.csv`

Example filenames in this repo:

- `CPIPPRODUCTION_DATA_2025-12-08_1841_montreal.csv`
- `CPIPPRODUCTION_DATA_2025-12-08_1220_calgary.csv`
- `2025-12-08_sickkids_redcap_data_toronto.csv`

### Optional Target CSVs (for Overlay Plots)

If present, these enable target-vs-observed overlay figures:

- `*montreal*age_by_group*.csv`
- `*calgary*age_by_group*.csv`
- `*toronto*age_by_group*.csv`

If missing, overlays are skipped and the rest of the pipeline still runs.

## How To Run

### Auto-discover input CSVs from current directory

```bash
python -m src.pipeline
```

### Pass explicit CSV paths

```bash
python -m src.pipeline \
  CPIPPRODUCTION_DATA_2025-12-08_1841_montreal.csv \
  CPIPPRODUCTION_DATA_2025-12-08_1220_calgary.csv \
  2025-12-08_sickkids_redcap_data_toronto.csv
```

Alternative invocation also supported:

```bash
python src/pipeline.py
```

## Outputs

All outputs are written under:

- `figs/`

### Core outputs

- `figs/dashboard.png` (8-panel overview)
- `figs/dashboard_counts.csv`
- `figs/dashboard_counts_na.csv`
- `figs/dashboard_missingness.csv`
- `figs/cpip_dashboard_all_figs.pdf`

### Standalone figure outputs

- `figs/sex_panel.png`
- `figs/gender_panel.png`
- `figs/income_panel.png`
- `figs/ethnicity_full.png`
- `figs/ethnicity_white_nonwhite.png`
- `figs/ethnicity_observed_vs_target.png`
- `figs/dashboard_mri_by_age_fixed.png`
- `figs/dashboard_mh.png`
- `figs/dashboard_mh_heatmap.png`
- `figs/dashboard_mh_heatmap_update.png`
- `figs/dashboard_mh_heatmap_percent.png`
- `figs/dashboard_mh_cooccurrence.png`
- `figs/dashboard_mh_corr.png`

### Optional overlay outputs (only if target CSVs exist)

- `figs/target_vs_observed_age_by_sex.png`
- `figs/target_vs_observed_age_by_ethnicity.png`

## Other Scripts

- `pipeline_monolith.py`: legacy monolithic version (kept for comparison).
- `src/reports/age_0_9_demographics_tables.py`: optional report script that reads:
  - `figs/age_0_9_participants_by_site_COMPLETE.csv`
  and writes summary CSV tables into:
  - `reports/tables/`

## Troubleshooting

- `No site CSVs found`: ensure working directory has site files or pass paths explicitly.
- `Cannot infer site from filename`: include `montreal`, `calgary`, or `toronto` in filenames.
- Missing overlays: add `*age_by_group*.csv` files per site.
- Permission/cache warnings from matplotlib/fontconfig: set writable local cache dirs (see requirements notes).
