"""Step 2: build a most-recent-sample-per-well table for arsenic and nitrate.

Reads WaterQualityMajor.txt and WellMain.txt straight from the GWDB zip (no unzip needed),
keeps the Gulf Coast and Ogallala aquifers, samples from MIN_YEAR onward, and one result
per well per analyte (the most recent sample that gives a determinate MCL comparison).

Censoring rules (ParameterFlag "<" = below the reporting limit):
- detected value > MCL            -> exceedance = 1
- detected value <= MCL           -> exceedance = 0
- "<" with reporting limit <= MCL -> exceedance = 0
- "<" with reporting limit > MCL  -> indeterminate, dropped
For concentration statistics a nondetect is set to half the reporting limit.

Nitrate is expressed as mg/L as N. Code preference within a sample:
00618 (nitrate as N) > 00631 (nitrite plus nitrate as N) > 71851 (nitrate as NO3, converted).
Arsenic: 01000 (dissolved) > 01002 (total).
"""
import io
import zipfile

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))  # works under python -I
from common import RAW_DIR, PROC_DIR, TAB_DIR, AQUIFERS, MCL, MIN_YEAR, NO3_TO_N

ZIP = RAW_DIR / "GWDBDownload.zip"
CODES = {"01000": ("arsenic", 1, 1.0), "01002": ("arsenic", 2, 1.0),
         "00618": ("nitrate", 1, 1.0), "00631": ("nitrate", 2, 1.0),
         "71851": ("nitrate", 3, NO3_TO_N)}
WQ_COLS = ["StateWellNumber", "County", "Aquifer", "SampleYear", "SampleDate",
           "SampleNumber", "Reliability", "ParameterCode", "ParameterFlag", "ParameterValue"]
WELL_COLS = ["StateWellNumber", "County", "AquiferCodeDescription", "LatitudeDD",
             "LongitudeDD", "WellDepth", "WellUse", "WellType"]


def read_member(z, member, usecols, chunk_filter=None):
    with z.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="latin-1")
        reader = pd.read_csv(text, sep="|", usecols=usecols, dtype=str, quoting=3,
                             chunksize=500_000, on_bad_lines="skip")
        parts = [chunk_filter(c) if chunk_filter else c for c in reader]
    return pd.concat(parts, ignore_index=True)


def wq_filter(c):
    c = c[c["Aquifer"].isin(AQUIFERS) & c["ParameterCode"].isin(CODES)]
    return c


def main():
    z = zipfile.ZipFile(ZIP)
    wq = read_member(z, "GWDBDownload/WaterQualityMajor.txt", WQ_COLS, wq_filter)
    wells = read_member(z, "GWDBDownload/WellMain.txt", WELL_COLS)
    print("raw arsenic/nitrate records in the two aquifers:", len(wq))

    wq["year"] = pd.to_numeric(wq["SampleYear"], errors="coerce")
    wq["value"] = pd.to_numeric(wq["ParameterValue"], errors="coerce")
    wq["date"] = pd.to_datetime(wq["SampleDate"], errors="coerce")
    flag = wq["ParameterFlag"].fillna("").str.strip()
    n0 = len(wq)
    bad_rel = wq["Reliability"].fillna("").str.startswith("Not indicative of aquifer quality")
    wq = wq[~bad_rel & wq["value"].notna() & (wq["value"] >= 0) & flag.isin(["", "<"])
            & (wq["year"] >= MIN_YEAR) & wq["date"].notna()].copy()
    wq["censored"] = wq["ParameterFlag"].fillna("").str.strip().eq("<")
    meta = wq["ParameterCode"].map(CODES)
    wq["analyte"] = meta.str[0]
    wq["pref"] = meta.str[1]
    wq["value"] = wq["value"] * meta.str[2]
    print(f"kept {len(wq)} of {n0} records after year/reliability/flag screens")

    # one record per well-sample-analyte, using the preferred parameter code
    wq = (wq.sort_values("pref")
            .drop_duplicates(["StateWellNumber", "date", "SampleNumber", "analyte"]))
    mcl = wq["analyte"].map(MCL)
    wq["exceed"] = np.where(~wq["censored"], (wq["value"] > mcl).astype(float),
                            np.where(wq["value"] <= mcl, 0.0, np.nan))
    wq = wq[wq["exceed"].notna()]
    wq["conc"] = np.where(wq["censored"], wq["value"] / 2.0, wq["value"])
    # most recent determinate sample per well and analyte
    wq = (wq.sort_values(["date", "SampleNumber"])
            .drop_duplicates(["StateWellNumber", "analyte"], keep="last"))

    wells = wells.drop_duplicates("StateWellNumber")
    for c in ("LatitudeDD", "LongitudeDD", "WellDepth"):
        wells[c] = pd.to_numeric(wells[c], errors="coerce")
    df = wq.merge(wells.drop(columns="County"), on="StateWellNumber", how="left")
    df = df[df["LatitudeDD"].between(25, 37) & df["LongitudeDD"].between(-107, -93)]
    df.loc[df["WellDepth"] <= 0, "WellDepth"] = np.nan

    out = df[["StateWellNumber", "Aquifer", "County", "LatitudeDD", "LongitudeDD", "WellDepth",
              "WellUse", "analyte", "date", "year", "ParameterCode", "censored", "value",
              "conc", "exceed"]].rename(columns={
        "StateWellNumber": "state_well_number", "Aquifer": "aquifer", "County": "county",
        "LatitudeDD": "lat", "LongitudeDD": "lon", "WellDepth": "well_depth_ft",
        "WellUse": "well_use", "ParameterCode": "param_code", "value": "reported_value"})
    out["date"] = out["date"].dt.date
    out["well_use"] = out["well_use"].fillna("Unknown").str.strip()
    out["exceed"] = out["exceed"].astype(int)
    out = out.sort_values(["analyte", "aquifer", "state_well_number"])
    out.to_csv(PROC_DIR / "wells_most_recent_arsenic_nitrate.csv", index=False)

    # summary table
    rows = []
    for (an, aq), g in out.groupby(["analyte", "aquifer"]):
        dom = g[g["well_use"].eq("Domestic")]
        rows.append({"analyte": an, "aquifer": aq, "n_wells": len(g),
                     "year_min": int(g["year"].min()), "year_max": int(g["year"].max()),
                     "median_year": int(g["year"].median()),
                     "pct_nondetect": round(100 * g["censored"].mean(), 1),
                     "median_conc": round(g["conc"].median(), 2),
                     "p90_conc": round(g["conc"].quantile(0.9), 2),
                     "n_exceed": int(g["exceed"].sum()),
                     "pct_exceed": round(100 * g["exceed"].mean(), 1),
                     "n_domestic": len(dom),
                     "pct_exceed_domestic": round(100 * dom["exceed"].mean(), 1) if len(dom) else np.nan,
                     "n_with_depth": int(g["well_depth_ft"].notna().sum())})
    summ = pd.DataFrame(rows)
    summ.to_csv(TAB_DIR / "table1_well_summary.csv", index=False)
    print(summ.to_string(index=False))
    use = (out.groupby(["analyte", "aquifer", "well_use"]).agg(n=("exceed", "size"),
           pct_exceed=("exceed", lambda s: round(100 * s.mean(), 1))).reset_index())
    use = use[use["n"] >= 20]
    use.to_csv(TAB_DIR / "table2_exceedance_by_well_use.csv", index=False)
    print(use.to_string(index=False))


if __name__ == "__main__":
    main()
