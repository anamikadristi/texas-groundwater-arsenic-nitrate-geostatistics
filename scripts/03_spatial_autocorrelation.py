"""Step 3: exploratory spatial analysis, global Moran's I and local Moran's I (LISA).

Weights: k-nearest neighbours (k = 8), row-standardised, on projected well locations.
Variables: log10 concentration (nondetects at half the reporting limit) and the 0/1
exceedance indicator. Inference: 999 conditional permutations.
LISA clusters are reported at pseudo p < 0.05 and, more strictly, after a Benjamini-Hochberg
false discovery rate correction (q < 0.05).
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))  # works under python -I

import numpy as np
import pandas as pd
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from statsmodels.stats.multitest import multipletests

from common import load_wells, PROC_DIR, TAB_DIR, AQUIFERS, SEED

LABELS = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def main():
    rows, lisa_out = [], []
    for analyte in ("arsenic", "nitrate"):
        for aquifer in AQUIFERS:
            g = load_wells(analyte, aquifer)
            w = KNN.from_array(g[["x_km", "y_km"]].to_numpy(), k=8)
            w.transform = "r"
            for var in ("log10_conc", "exceed"):
                y = g[var].to_numpy(dtype=float)
                np.random.seed(SEED)
                m = Moran(y, w, permutations=999)
                rows.append({"analyte": analyte, "aquifer": aquifer, "variable": var,
                             "n_wells": len(g), "morans_I": round(m.I, 3),
                             "expected_I": round(m.EI, 4), "z_perm": round(m.z_sim, 1),
                             "p_perm": m.p_sim})
            y = g["log10_conc"].to_numpy(dtype=float)
            lm = Moran_Local(y, w, permutations=999, seed=SEED)
            q = multipletests(lm.p_sim, method="fdr_bh")[1]
            cl = pd.Series(lm.q).map(LABELS)
            g["lisa_cluster_p05"] = np.where(lm.p_sim < 0.05, cl, "Not significant")
            g["lisa_cluster_fdr"] = np.where(q < 0.05, cl, "Not significant")
            g["lisa_Ii"] = lm.Is
            g["lisa_p"] = lm.p_sim
            lisa_out.append(g[["state_well_number", "analyte", "aquifer", "county", "lon", "lat",
                               "log10_conc", "exceed", "lisa_Ii", "lisa_p",
                               "lisa_cluster_p05", "lisa_cluster_fdr"]])
    mor = pd.DataFrame(rows)
    mor.to_csv(TAB_DIR / "table3_global_morans_I.csv", index=False)
    print(mor.to_string(index=False))
    lisa = pd.concat(lisa_out, ignore_index=True)
    lisa.to_csv(PROC_DIR / "lisa_well_clusters.csv", index=False)
    cnt = (lisa.groupby(["analyte", "aquifer", "lisa_cluster_fdr"]).size()
               .unstack(fill_value=0).reset_index())
    cnt.to_csv(TAB_DIR / "table4_lisa_cluster_counts_fdr.csv", index=False)
    print(cnt.to_string(index=False))
    # counties holding the most High-High wells (FDR)
    hh = (lisa[lisa["lisa_cluster_fdr"] == "High-High"]
          .groupby(["analyte", "aquifer", "county"]).size().rename("n_high_high_wells")
          .reset_index().sort_values(["analyte", "aquifer", "n_high_high_wells"],
                                     ascending=[True, True, False]))
    hh.groupby(["analyte", "aquifer"]).head(5).to_csv(
        TAB_DIR / "table5_lisa_high_high_top_counties.csv", index=False)
    print(hh.groupby(["analyte", "aquifer"]).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
