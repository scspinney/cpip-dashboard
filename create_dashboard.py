#!/usr/bin/env python3
"""
C‑PIP multi‑site dashboard  (main REDCap export only)

• Baseline (T1) rows only
• Exactly one row per participant – the row that answers the *most* key fields
• Writes ./figs/dashboard_sp.png  (6‑panel overview)
"""

import sys, glob, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import seaborn as sns

# ───────────────────────── CONFIG ─────────────────────────
FIGDIR       = Path("figs")
MIN_T1_DATE  = pd.Timestamp("2023-01-01")     # ASCII hyphens
SITE_COLOURS = {"Montreal": "#1f77b4", "Calgary": "#ff0e0e", "Toronto": "#2ca02c"}
MRI_INSTRUMENT = "mri"                        # repeat‑instrument (lower‑case)

# ───────────────────────── HELPERS ────────────────────────
def derive_site(p: Path) -> str:
    s = p.stem.lower()
    if   "montreal" in s: return "Montreal"
    elif "calgary"  in s: return "Calgary"
    elif "toronto"  in s: return "Toronto"
    raise ValueError(f"Cannot infer site from filename: {p.name}")

def pick_first(df: pd.DataFrame, names, *, required=False):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise ValueError(f"None of {names} found")
    return None

def load_csv(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, low_memory=False)
    d["site"] = derive_site(p)
    return d

# ──────────────────────── LOAD DATA ───────────────────────
paths = [Path(p) for p in (sys.argv[1:] if len(sys.argv) > 1 else
                           [f for g in ("*montreal*.csv","*calgary*.csv","*toronto*.csv")
                            for f in glob.glob(g)])]
if not paths:
    sys.exit("❌ No site CSVs found.")

df_main = pd.concat([load_csv(p) for p in paths], ignore_index=True)

# ────────────────── PICK BEST BASELINE ROW PER PARTICIPANT ─────────────────
non_repeat = df_main[df_main["redcap_repeat_instrument"].isna()].copy()

# columns later used in the panels
SEX_COL    = pick_first(non_repeat, ["sex","demo_sex"], required=True)
GENDER_COL = pick_first(non_repeat, ["gender","demo_gender","gender_oth"], required=True)
ETH_COLS   = [c for c in non_repeat.columns if re.match(r"(demo_)?ethnicity___\d+", c)]

# calculate total household income from demo_cg_income and demo_cg_prtnr_income
if "demo_cg_income" in non_repeat.columns and "demo_cg_prtnr_income" in non_repeat.columns:
    non_repeat["demo_cg_income"] = non_repeat["demo_cg_income"].fillna(0) + non_repeat["demo_cg_prtnr_income"].fillna(0)
elif "demo_cg_income" in non_repeat.columns:
    non_repeat["demo_cg_income"] = non_repeat["demo_cg_income"]
elif "demo_cg_prtnr_income" in non_repeat.columns:
    non_repeat["demo_cg_income"] = non_repeat["demo_cg_prtnr_income"]
else:
    non_repeat["demo_cg_income"] = np.nan

KEY_VARS = ["youth_age_y", "youth_age_m", SEX_COL, GENDER_COL, "demo_cg_income"] + ETH_COLS

def best_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one T1 row per participant – the row with the most answered `KEY_VARS`."""
    t1 = df[df["redcap_event_name"] == "t1_arm_1"].copy()
    # score: count non‑missing KEY_VARS
    t1["_score"] = t1[KEY_VARS].notna().sum(axis=1)
    # sort: higher score first; tie‑break by record_id to stay deterministic
    t1 = t1.sort_values(["record_id", "_score"], ascending=[True, False])
    best = t1.drop_duplicates(subset="record_id").drop(columns="_score")
    return best.reset_index(drop=True)

baseline = best_baseline_rows(non_repeat)
# print(baseline[["site","record_id"] + KEY_VARS].head(50))

# ────────────────── DERIVED DEMOGRAPHIC FIELDS ─────────────────────────────
def _age_years(row):
    y, m = row.get("youth_age_y"), row.get("youth_age_m")
    if pd.isna(y) and pd.isna(m):
        return np.nan
    return (0 if pd.isna(y) else y) + (0 if pd.isna(m) else m) / 12

baseline["age_years"] = baseline.apply(_age_years, axis=1)

sex_map    = {1:"Male",2:"Female",3:"Intersex / Non-binary",777:"Unknown",888:"Prefer not to answer", np.nan:"NA"}
gender_map = {1:"Boy/Man",2:"Girl/Woman",3:"Non-binary",4:"Two-spirit",5:"Another",
              6:"Questioning",777:"Unknown",888:"Prefer not to answer", np.nan:"NA"}
income_map = {1:"Under $10k",2:"$10‑19k",3:"$20‑39k",4:"$40‑59k",5:"$60‑79k",
              6:"$80‑99k",7:"$100‑124k",8:"$125‑149k",9:"$150‑199k",
              10:"$200‑299k",11:"$300‑399k",12:"$400k+",777:"Unknown",888:"Prefer not to say"}

baseline["sex_lbl"]    = baseline[SEX_COL].map(sex_map)
baseline["gender_lbl"] = baseline[GENDER_COL].map(gender_map)
baseline["income_lbl"] = baseline["demo_cg_income"].map(income_map)

# print(baseline)

# ────────── Ethnicity (checkbox melt) ──────────
if ETH_COLS:
    baseline[ETH_COLS] = baseline[ETH_COLS].fillna(0)
    eth_lookup = {f"{p}{i}": lbl
                  for p in ("demo_ethnicity___","demo_cg_ethnicity___")
                  for i,lbl in {
                      1:"Black",2:"East Asian",3:"First Nations",4:"Inuk/Inuit",5:"Métis",
                      6:"Latin American",7:"Middle Eastern",8:"South Asian",9:"Southeast Asian",
                      10:"White",11:"Other",888:"Prefer not to say"}.items()}
    eth_long = (baseline[["record_id","site"]+ETH_COLS]
                .melt(id_vars=["record_id","site"], value_name="val")
                .query("val == 1")
                .assign(label=lambda d: d["variable"].map(eth_lookup)))
else:
    eth_long = None

# aggregate ethnicity as White/Non-white
if eth_long is not None:
    eth_long["is_white"] = eth_long["label"].apply(lambda x: x == "White")
    eth_long = (eth_long.groupby(["site","record_id","is_white"])["label"]
                .apply(lambda x: ", ".join(sorted(x.unique())))
                .reset_index(name="label"))
    eth_long["label"] = eth_long["label"].str.replace("White, ","", regex=False)
    eth_wide = (eth_long.pivot(index=["site","record_id"], columns="is_white", values="label")
                .reset_index().rename(columns={True: "White", False: "Non-white"}))
                    
# ────────── MRI baseline rows (deduped) ──────────
mri_base = df_main.loc[
    (df_main["redcap_event_name"] == "t1_arm_1") &
    (df_main["redcap_repeat_instrument"].fillna("")
         .str.strip().str.lower() == MRI_INSTRUMENT)
].copy()

mri_base["mri_date"] = pd.to_datetime(mri_base["mri_date"],
                                      errors="coerce").dt.normalize()
mri_base = (mri_base.dropna(subset=["mri_date"])
                     .loc[mri_base["mri_date"] >= MIN_T1_DATE]
                     .sort_values("mri_complete", ascending=False)
                     .drop_duplicates(subset=["record_id","mri_date"]))

sched = (mri_base.groupby(["site","mri_date"])["record_id"]
                  .nunique().groupby(level=0).cumsum()
                  .rename("cum").reset_index().assign(status="Scheduled"))
comp  = (mri_base[mri_base["mri_complete"] == 2]
                  .groupby(["site","mri_date"])["record_id"]
                  .nunique().groupby(level=0).cumsum()
                  .rename("cum").reset_index().assign(status="Completed"))
timeline = pd.concat([sched, comp], ignore_index=True)

# ────────── sanity‑check counts ──────────
def _count_non_na(column: str) -> pd.Series:
    """# baseline participants per site with a non‑NA in `column`."""
    return (baseline[baseline[column].notna()]
            .groupby("site")["record_id"]
            .nunique())

counts = pd.DataFrame({
    "Total available counts": baseline.groupby("site")["record_id"].nunique(),
    "Age non‑NA":          _count_non_na("age_years"),
    "Sex non‑NA":          _count_non_na("sex_lbl"),
    "Gender non‑NA":       _count_non_na("gender_lbl"),
    "Income non‑NA":       _count_non_na("income_lbl"),
})

# Ethnicity: at least one checkbox ticked (==1)
if eth_long is not None:
    eth_counts = (eth_long.groupby("site")["record_id"]
                          .nunique()
                          .rename("Ethnicity non‑NA"))
    counts = counts.join(eth_counts, how="left")

counts = counts.fillna(0).astype(int)

# ethnicity missing counts
if eth_long is not None:
    eth_counts_na = baseline.groupby("site")["record_id"].nunique() - (eth_counts)

na_counts = pd.DataFrame({
    "Total available counts": baseline.groupby("site")["record_id"].nunique(),
    "Age NA":          baseline.groupby("site")["record_id"].nunique() - _count_non_na("age_years"),  
    "Income NA":       baseline.groupby("site")["record_id"].nunique() - _count_non_na("income_lbl"),
    "Ethnicity NA": eth_counts_na if eth_long is not None else 0,
})

print("\nParticipant counts used in each baseline panel")
print(counts.to_string())
print("\nParticipant counts with missing values (NA)")  
print(na_counts.to_string())
print("\nNOTE: All numbers refer to *unique* baseline participants "
      "after deduplication (one T1 row per participant).")

# save table of missing and total counts
counts.to_csv(FIGDIR / "dashboard_counts.csv")
na_counts.to_csv(FIGDIR / "dashboard_counts_na.csv")


# ────────── PLOTS ──────────
sns.set(style="whitegrid")
FIGDIR.mkdir(exist_ok=True)
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
(ax_cum, ax_age), (ax_gender, ax_sex), (ax_eth, ax_eth_wnw), (ax_inc, ax_nas) = axes
# (ax_age, ax_sex), (ax_gender, ax_eth), (ax_inc, ax_cum), (ax_eth_wnw, ax_unused) = axes

# Age
sns.histplot(baseline.dropna(subset=["age_years"]),
             x="age_years", hue="site", multiple="dodge",
             bins=15, shrink=0.8, edgecolor="black",
             palette=SITE_COLOURS, ax=ax_age)
ax_age.set_title("Age distribution by site (baseline)")
ax_age.set_xlabel("Age (years)")

# Sex
sns.countplot(baseline, x="sex_lbl", hue="site",
              palette=SITE_COLOURS, order=sex_map.values(), ax=ax_sex)
ax_sex.set_title("Sex assigned at birth (baseline)")
ax_sex.tick_params(axis="x", rotation=45)

# Gender
order_g = ["Boy/Man","Girl/Woman","Non-binary","Two-spirit",
           "Another","Questioning","Unknown","Prefer not to answer","NA"]
sns.countplot(baseline, x="gender_lbl", hue="site",
              palette=SITE_COLOURS, order=order_g, ax=ax_gender)
ax_gender.set_title("Current gender identity (baseline)")
ax_gender.tick_params(axis="x", rotation=45)

# Ethnicity
if eth_long is not None and not eth_long.empty:
    sns.countplot(eth_long, x="label", hue="site",
                  palette=SITE_COLOURS,
                  order=eth_long["label"].value_counts().index, ax=ax_eth)
    ax_eth.set_title("Ethnicity (baseline)")
    ax_eth.tick_params(axis="x", rotation=45)
else:
    ax_eth.axis("off")
    ax_eth.text(0.5,0.5,"Ethnicity\nN/A", ha="center", va="center",
                fontsize=12, color="grey", transform=ax_eth.transAxes)

# Ethnicity wide
# ...existing code...

# Ethnicity wide (White/Non-white only) in ax_eth_wnw
if 'eth_wide' in locals() and not eth_wide.empty:
    eth_wide_long = eth_wide.melt(id_vars=["site", "record_id"], 
                                  value_vars=["White", "Non-white"],
                                  var_name="Ethnicity", value_name="Label")
    eth_wide_long = eth_wide_long[eth_wide_long["Label"].notna() & (eth_wide_long["Label"] != "")]
    eth_counts = (eth_wide_long.groupby(["site", "Ethnicity"])["record_id"]
                  .nunique()
                  .reset_index(name="Count"))
    sns.barplot(data=eth_counts, x="Ethnicity", y="Count", hue="site", palette=SITE_COLOURS, ax=ax_eth_wnw)
    ax_eth_wnw.set_title("Ethnicity (White/Non-white) by site")
    ax_eth_wnw.set_ylabel("Participant count")
    ax_eth_wnw.set_xlabel("Ethnicity group")
    ax_eth_wnw.tick_params(axis="x", rotation=0)
else:
    ax_eth_wnw.axis("off")
    ax_eth_wnw.text(0.5,0.5,"Ethnicity\nN/A", ha="center", va="center",
                    fontsize=12, color="grey", transform=ax_eth_wnw.transAxes)

# MRI timeline (keep this on ax_cum)
if not timeline.empty:
    sns.lineplot(timeline, x="mri_date", y="cum",
                 hue="site", style="status",
                 palette=SITE_COLOURS,
                 dashes={"Scheduled":[2,2], "Completed":""},
                 ax=ax_cum)
    ax_cum.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y‑%m"))
    ax_cum.set_xlim(timeline["mri_date"].min(), timeline["mri_date"].max())
    plt.setp(ax_cum.get_xticklabels(), rotation=45, ha="right")
    ax_cum.set_ylabel("Cumulative T1 MRIs")
    ax_cum.set_xlabel("Date")
    ax_cum.set_title("Baseline MRI visits by site")
    ax_cum.legend(title="Site / Status", loc="upper left")
else:
    ax_cum.axis("off")
    ax_cum.text(0.5,0.5,"MRI timeline\nN/A", ha="center", va="center",
                fontsize=12, color="grey", transform=ax_cum.transAxes)

# Income
if not baseline["income_lbl"].dropna().empty:
    order_inc = [income_map[i] for i in sorted(income_map) if i not in (777,888)]
    sns.countplot(baseline.dropna(subset=["income_lbl"]), x="income_lbl",
                  hue="site", palette=SITE_COLOURS,
                  order=order_inc, ax=ax_inc)
    ax_inc.set_title("Household income (baseline)")
    ax_inc.tick_params(axis="x", rotation=45)
else:
    ax_inc.axis("off")
    ax_inc.text(0.5,0.5,"Income\nN/A", ha="center", va="center",
                fontsize=12, color="grey", transform=ax_inc.transAxes)


# add missing counts to the last panel
if not na_counts.empty:
    na_counts = na_counts.reset_index().melt(id_vars="site", var_name="Metric", value_name="Count")
    sns.barplot(data=na_counts, x="Metric", y="Count", hue="site",
                palette=SITE_COLOURS, ax=ax_nas)
    ax_nas.set_title("Missing counts per panel")
    ax_nas.set_ylabel("Count")
    ax_nas.set_xlabel("Panel")
    ax_nas.tick_params(axis="x", rotation=45)
else:
    ax_nas.axis("off")
    ax_nas.text(0.5,0.5,"Missing counts\nN/A", ha="center", va="center",
                fontsize=12, color="grey", transform=ax_nas.transAxes)  


fig.tight_layout()
fig.savefig(FIGDIR / "dashboard_sp.png", dpi=300)
plt.close(fig)
print(f"✓ Dashboard saved → {FIGDIR/'dashboard_sp.png'}")
