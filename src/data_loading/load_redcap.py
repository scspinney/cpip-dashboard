from pathlib import Path
import glob
import pandas as pd
from typing import List, Tuple

from ..util.helpers import derive_site


def load_csv(p: Path) -> pd.DataFrame:
    """
    Load a single REDCap CSV export and inject a `site` column 
    derived from the filename.

    Parameters
    ----------
    p : Path
        Path to a REDCap CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with a site column.
    """
    df = pd.read_csv(p, low_memory=False)
    df["site"] = derive_site(p)
    return df


def discover_site_paths(args: List[str]) -> List[Path]:
    """
    Determine which REDCap CSVs to load.

    Priority:
    1. Explicit user-provided paths (args)
    2. Auto-discovery using filename patterns matching site names

    Returns
    -------
    List[Path]
        Sorted list of discovered CSV paths.
    """
    if args:
        return [Path(p) for p in args]

    patterns = ("*montreal*.csv", "*calgary*.csv", "*toronto*.csv")
    files = [Path(f) for g in patterns for f in glob.glob(g)]
    return sorted(files)


def validate_redcap_structure(df: pd.DataFrame):
    """
    Validate that required REDCap fields exist.
    Raises SystemExit on error.

    Required fields:
        • record_id
        • redcap_repeat_instrument
        • redcap_event_name
    """
    required = ["record_id", "redcap_repeat_instrument", "redcap_event_name"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise SystemExit(
            f"❌ REDCap export missing required columns: {missing}\n"
            f"Columns present: {list(df.columns)[:25]} ..."
        )


def load_redcap_exports(paths: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all REDCap site CSVs and return:

    • df_main : FULL concatenated REDCap dataset  
    • non_repeat : non-repeating baseline rows (instrument==NA)

    Parameters
    ----------
    paths : List[Path]
        List of CSV file paths.

    Returns
    -------
    (df_main, non_repeat) : Tuple[pd.DataFrame, pd.DataFrame]
    """
    if not paths:
        raise SystemExit("❌ No site CSVs found.")

    # Load and concatenate
    df_main = pd.concat([load_csv(p) for p in paths], ignore_index=True)

    # Structural validation
    validate_redcap_structure(df_main)

    # Extract non-repeat rows
    non_repeat = df_main[df_main["redcap_repeat_instrument"].isna()].copy()

    return df_main, non_repeat
