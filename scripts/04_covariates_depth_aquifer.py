"""Step 4: covariates, well depth and aquifer.

(a) Exceedance rate by well-depth class within each aquifer.
(b) Logistic regression per analyte: exceed ~ aquifer + log10(depth) + aquifer:log10(depth),
    with standard errors clustered by county, because wells in the same county are not
    independent. This is descriptive: it is not adjusted for spatial structure beyond the
    county clustering, and depth is the reported total well depth, not the screened interval.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))  # works under python -I

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

from common import load_wells, TAB_DIR

BINS = [0, 100, 200, 400, 800, 100000]
LABS = ["<100 ft", "100-199 ft", "200-399 ft", "400-799 ft", ">=800 ft"]


def main():
    g = load_wells()
    g = g[g["well_depth_ft"].notna()].copy()
    g["depth_class"] = pd.cut(g["well_depth_ft"], BINS, labels=LABS, right=False)
    tab = (g.groupby(["analyte", "aquifer", "depth_class"], observed=True)
             .agg(n_wells=("exceed", "size"), n_exceed=("exceed", "sum"),
                  pct_exceed=("exceed", lambda s: round(100 * s.mean(), 1)),
                  median_conc=("conc", "median")).reset_index())
    tab.to_csv(TAB_DIR / "table6_exceedance_by_depth_class.csv", index=False)
    print(tab.to_string(index=False))

    rows = []
    for analyte, d in g.groupby("analyte"):
        d = d.copy()
        d["log10_depth"] = np.log10(d["well_depth_ft"])
        d["ogallala"] = (d["aquifer"] == "Ogallala").astype(int)
        d["cluster"] = pd.factorize(d["county"].fillna("NA") + d["aquifer"])[0]
        fit = smf.logit("exceed ~ ogallala * log10_depth", data=d).fit(
            disp=False, cov_type="cluster", cov_kwds={"groups": d["cluster"]})
        # depth effect within each aquifer (odds ratio per tenfold increase in depth)
        for aq, cvec in (("Gulf Coast", {"log10_depth": 1}),
                         ("Ogallala", {"log10_depth": 1, "ogallala:log10_depth": 1})):
            L = np.array([cvec.get(n, 0) for n in fit.params.index], dtype=float)
            est = float(L @ fit.params)
            se = float(np.sqrt(L @ fit.cov_params().to_numpy() @ L))
            rows.append({"analyte": analyte, "aquifer": aq, "n_wells": int(fit.nobs),
                         "term": "log10(well depth), OR per 10x depth",
                         "odds_ratio": round(np.exp(est), 3),
                         "ci_low": round(np.exp(est - 1.96 * se), 3),
                         "ci_high": round(np.exp(est + 1.96 * se), 3),
                         "p_value": round(2 * norm.sf(abs(est / se)), 4)})
        # aquifer contrast at the median depth
        md = float(np.log10(d["well_depth_ft"].median()))
        L = np.array([{"ogallala": 1, "ogallala:log10_depth": md}.get(n, 0)
                      for n in fit.params.index], dtype=float)
        est = float(L @ fit.params)
        se = float(np.sqrt(L @ fit.cov_params().to_numpy() @ L))
        rows.append({"analyte": analyte, "aquifer": "Ogallala vs Gulf Coast",
                     "n_wells": int(fit.nobs),
                     "term": f"aquifer OR at median depth ({10**md:.0f} ft)",
                     "odds_ratio": round(np.exp(est), 3),
                     "ci_low": round(np.exp(est - 1.96 * se), 3),
                     "ci_high": round(np.exp(est + 1.96 * se), 3),
                     "p_value": round(2 * norm.sf(abs(est / se)), 4)})
    res = pd.DataFrame(rows)
    res.to_csv(TAB_DIR / "table7_logistic_depth_aquifer.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
