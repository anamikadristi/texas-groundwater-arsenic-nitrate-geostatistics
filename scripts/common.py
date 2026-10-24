"""Shared paths and settings for the Texas groundwater arsenic and nitrate project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Raw downloads are large (the GWDB zip is about 83 MB and unzips to about 1.8 GB),
# so they stay outside version control. Override with the RAW_DIR environment variable.
RAW_DIR = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
GIS_DIR = ROOT / "outputs" / "gis"
for d in (PROC_DIR, FIG_DIR, TAB_DIR, GIS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "GWDBDownload.zip": "https://www.twdb.texas.gov/groundwater/data/GWDBDownload.zip",
    "major_aquifers.zip": "https://www.twdb.texas.gov/mapping/gisdata/doc/major_aquifers.zip",
    "cb_2023_us_county_500k.zip": "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip",
}

# Projected, equal-area CRS for Texas (NAD83 / Texas Centric Albers Equal Area), metres.
CRS_M = "EPSG:3083"

AQUIFERS = {"Gulf Coast": "GULF_COAST", "Ogallala": "OGALLALA"}

# EPA maximum contaminant levels
MCL = {"arsenic": 10.0,   # ug/L
       "nitrate": 10.0}   # mg/L as N
UNITS = {"arsenic": "ug/L", "nitrate": "mg/L as N"}

MIN_YEAR = 1990          # keep samples from 1990 onward
GRID_M = 5000            # prediction grid spacing (m)
NO3_TO_N = 14.0067 / 62.0049  # convert mg/L as NO3 to mg/L as N
SEED = 20261023


def load_wells(analyte=None, aquifer=None):
    """Most recent sample per well, projected to CRS_M, with x/y in km.

    Wells that share coordinates are reduced to the most recent record, because
    co-located points make the kriging system singular.
    """
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    df = pd.read_csv(PROC_DIR / "wells_most_recent_arsenic_nitrate.csv",
                     dtype={"state_well_number": str, "param_code": str})
    if analyte:
        df = df[df["analyte"] == analyte]
    if aquifer:
        df = df[df["aquifer"] == aquifer]
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                         crs="EPSG:4269").to_crs(CRS_M)
    g["x_km"] = g.geometry.x / 1000.0
    g["y_km"] = g.geometry.y / 1000.0
    g = (g.sort_values("date")
          .drop_duplicates(["analyte", "aquifer", "x_km", "y_km"], keep="last"))
    g["log10_conc"] = np.log10(g["conc"].clip(lower=0.01))
    return g.reset_index(drop=True)


def load_aquifer(aquifer):
    """Dissolved TWDB major aquifer polygon (outcrop and subsurface parts)."""
    import zipfile
    import geopandas as gpd
    a = gpd.read_file(f"zip://{RAW_DIR / 'major_aquifers.zip'}!NEW_major_aquifers_dd.shp")
    a = a[(a["AQ_NAME"] == AQUIFERS[aquifer]) & (a["AQUIFER"].isin([1, 2]))]
    return a.to_crs(CRS_M).dissolve()[["geometry"]]


def load_counties():
    import geopandas as gpd
    c = gpd.read_file(f"zip://{RAW_DIR / 'cb_2023_us_county_500k.zip'}!cb_2023_us_county_500k.shp")
    c = c[c["STATEFP"] == "48"][["GEOID", "NAME", "geometry"]].rename(columns={"NAME": "county"})
    return c.to_crs(CRS_M).reset_index(drop=True)
