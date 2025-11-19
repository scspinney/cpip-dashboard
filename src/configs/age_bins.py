from typing import Optional
import re
import pandas as pd

AGE_BLOCK_ORDER = ["0-2", "2-5", "6-9", "10-12", "13-15", "16-18"]
AGE_BIN_ORDER   = ["<5wks"] + [f"{i}-{i+1}" for i in range(0, 18)] + ["18"]

def to_age_block(a: float) -> Optional[str]:
    if pd.isna(a): return None
    if a <= 2:   return "0-2"
    if a <= 5:   return "2-5"
    if a <= 9:   return "6-9"
    if a <= 12:  return "10-12"
    if a <= 15:  return "13-15"
    if a <= 18:  return "16-18"
    return None

def bin_label_to_block(lbl: str) -> Optional[str]:
    if lbl is None:
        return None
    if lbl in ("<5wks", "0-1", "1-2"):
        return "0-2"
    if lbl == "18":
        return "16-18"
    m = re.match(r"^(\d+)-(\d+)$", str(lbl))
    if not m:
        return None
    k = int(m.group(1))
    if k < 2:   return "0-2"
    if k < 6:   return "2-5"
    if k < 10:  return "6-9"
    if k < 13:  return "10-12"
    if k < 16:  return "13-15"
    return "16-18"
