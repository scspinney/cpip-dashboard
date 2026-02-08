from pathlib import Path
import pandas as pd

# ================================
# General Paths & Directories
# ================================
FIGDIR = Path("figs")
FIGDIR.mkdir(exist_ok=True)

OUTPDF = FIGDIR / "cpip_dashboard_all_figs.pdf"

# ================================
# MRI / REDCap processing constants
# ================================
MIN_T1_DATE = pd.Timestamp("2023-01-01")

# REDCap repeat-instrument identifier for MRI visit rows
MRI_INSTRUMENT = "mri"

# ================================
# File discovery patterns
# ================================
SITE_FILE_PATTERNS = (
    "*montreal*.csv",
    "*calgary*.csv",
    "*toronto*.csv",
)

AGE_BY_GROUP_PATTERNS = (
    "*montreal*age_by_group*.csv",
    "*calgary*age_by_group*.csv",
    "*toronto*age_by_group*.csv",
)

# ================================
# Demographic variable discovery
# ================================
# Youth-completed form flags
YOUTH_COMPLETE_CAND_REGEXES = [
    r"^youth.*demo.*baseline.*_complete$",
    r"^youth.*demo.*_complete$",
    r"^demo.*youth.*_complete$",
]

YOUTH_COMPLETE_FALLBACKS = [
    "youth_demographics_baseline_complete",
    "youth_demographics_complete",
    "youth_demo_complete",
]

# Parent/child-completed form flags
PARENT_CHILD_COMPLETE_CAND_REGEXES = [
    r"^caregiver.*demo.*about.*child.*_complete$",
    r"^caregiver.*demo.*child.*_complete$",
    r"^caregiver.*demograph.*_complete$",
    r"^demo.*child.*_complete$",
]

PARENT_CHILD_COMPLETE_FALLBACKS = [
    "caregiver_demographics_baseline_about_child_complete",
    "caregiver_demographics_child_complete",
    "caregiver_demographics_complete",
    "demo_child_complete",
]

# ================================
# Sanity-check: accepted sex values
# ================================
SEX_VALID = {1, 2, 3, 777, 888}

