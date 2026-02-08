import re
from pathlib import Path
from typing import List, Optional, Any
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from datetime import datetime


# ==============================================================
#  SITE + COLUMN DISCOVERY HELPERS
# ==============================================================

def derive_site(p: Path) -> str:
    """
    Infer CPIP site name from a filename.
    Expected substrings (case-insensitive): montreal, calgary, toronto.
    """
    s = p.stem.lower()
    if "montreal" in s:
        return "Montreal"
    if "calgary" in s:
        return "Calgary"
    if "toronto" in s:
        return "Toronto"
    raise ValueError(f"Cannot infer site from filename: {p.name}")


def pick_first(df: pd.DataFrame, names: List[str], *, required=False) -> Optional[str]:
    """
    Return the first matching column from `names`.
    If `required=True`, raise an error if none found.
    """
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise ValueError(f"None of columns {names} found in dataframe.")
    return None


def find_complete_col(df: pd.DataFrame,
                      regexes: List[str],
                      fallbacks: List[str]) -> Optional[str]:
    """
    Find a REDCap *_complete field using:
    1) regex search (priority order)
    2) fallback column names
    """
    candidates = [c for c in df.columns if c.endswith("_complete")]

    for pat in regexes:
        reg = re.compile(pat, flags=re.IGNORECASE)
        for c in candidates:
            if reg.search(c):
                return c

    for fb in fallbacks:
        if fb in df.columns:
            return fb

    return None


# ==============================================================
# NORMALIZATION HELPERS
# ==============================================================

def _norm_int(x: Any) -> float:
    """Convert to integer if possible, else NaN."""
    try:
        return int(x)
    except Exception:
        return np.nan


def _as_int_or_nan(x: Any) -> float:
    """Convert safely to int or NaN (used for REDCap-coded categorical fields)."""
    try:
        return int(x)
    except Exception:
        return np.nan


# ==============================================================
# AGE + BINNING HELPERS
# ==============================================================

def age_to_years(age_y: Any, age_m: Any) -> float:
    """
    Convert (age in years, months) to a single continuous age variable.
    """
    if pd.isna(age_y) and pd.isna(age_m):
        return np.nan
    return (0 if pd.isna(age_y) else float(age_y)) + \
           (0 if pd.isna(age_m) else float(age_m)) / 12.0


def to_age_block(a: float) -> Optional[str]:
    """
    Assign an age (in years) to the CPIP short-form age bins.
    """
    if pd.isna(a):
        return None
    if a <= 2:   return "0-2"
    if a <= 5:   return "2-5"
    if a <= 9:   return "6-9"
    if a <= 12:  return "10-12"
    if a <= 15:  return "13-15"
    if a <= 18:  return "16-18"
    return None


def bin_label_to_block(lbl: str) -> Optional[str]:
    """
    Map target CSV age-bin labels into the CPIP short blocks.
    """
    if lbl is None:
        return None

    if lbl in ("<5wks", "0-1", "1-2"):
        return "0-2"

    if lbl == "18":
        return "16-18"

    m = re.match(r"^(\d+)-(\d+)$", str(lbl))
    if not m:
        return None

    lo = int(m.group(1))

    if lo < 2:   return "0-2"
    if lo < 6:   return "2-5"
    if lo < 10:  return "6-9"
    if lo < 13:  return "10-12"
    if lo < 16:  return "13-15"
    return "16-18"


# ==============================================================
# INCOME MAPPING HELPERS
# ==============================================================

_BRACKET_MEDIAN = {
     1:   5_000,  2:  15_000,  3:  30_000,  4:  50_000,
     5:  70_000,  6:  90_000,  7: 112_500,  8: 137_500,
     9: 175_000, 10: 250_000, 11: 350_000, 12: 450_000,
}

SPECIAL_INCOME = {777, 888}


def closest_code_from_dollars(amount: float) -> int:
    """
    Return the REDCap income bracket code whose median is closest to `amount`.
    """
    return int(min(
        _BRACKET_MEDIAN.keys(),
        key=lambda c: abs(_BRACKET_MEDIAN[c] - amount)
    ))


# ==============================================================
# PDF GENERATION HELPERS
# ==============================================================

def add_image_page(pp: PdfPages, img_path: Path, title: Optional[str] = None):
    """
    Add an image file to a PDF as a standalone page.
    """
    try:
        img = mpimg.imread(img_path)
    except Exception as e:
        print(f"⚠️ Skipping {img_path.name}: {e}")
        return

    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img)
    ax.axis("off")

    if title:
        fig.suptitle(title, y=0.98, fontsize=12)
        fig.subplots_adjust(top=0.95)

    pp.savefig(fig, dpi=300)
    plt.close(fig)


def add_cover_page(pp: PdfPages, title: str = "C-PIP Multi-site Dashboard"):
    """
    Insert the standard CPIP cover page.
    """
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    ax.text(
        0.5, 0.6,
        f"{title}\n\nGenerated: {now}",
        ha="center", va="center",
        fontsize=18
    )

    pp.savefig(fig)
    plt.close(fig)
