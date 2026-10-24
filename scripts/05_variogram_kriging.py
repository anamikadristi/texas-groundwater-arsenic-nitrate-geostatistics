"""Step 5: variogram modelling, indicator kriging, ordinary kriging, and cross-validation.

For each analyte x aquifer:
1. Empirical isotropic variograms (gstools) of
   - the exceedance indicator I = 1[conc > MCL]  (indicator kriging, IK)
   - log10 concentration                         (ordinary kriging, OK)
   with 20 lag bins out to MAX_LAG_KM.
2. Fit spherical and exponential models with a nugget; keep the one with higher R^2.
3. Ordinary kriging with pykrige (moving neighbourhood of N_NEIGH nearest wells) on a
   5 km grid clipped to the TWDB aquifer boundary inside Texas.
   - IK estimate, clipped to [0, 1], is read as the local probability of exceedance.
   - The kriging variance is kept as the uncertainty layer. For IK it reflects data
     density and the indicator variogram; it is not a confidence interval on probability.
   - As a sensitivity check, P(exceed) is also derived from OK of log10 concentration under
     a Gaussian assumption: 1 - Phi((log10 MCL - mean) / sqrt(variance)).
4. 10-fold cross-validation with the variogram fixed at the full-data fit.
Outputs: tables, grid predictions (CSV) and 4-band GeoTIFFs for ArcGIS Pro / QGIS.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))  # works under python -I

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import gstools as gs
import rasterio
from rasterio.transform import from_origin
from pykrige.ok import OrdinaryKriging
from scipy.stats import norm
from shapely import contains_xy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from common import (load_wells, load_aquifer, load_counties, PROC_DIR, TAB_DIR, GIS_DIR,
                    AQUIFERS, MCL, GRID_M, SEED, CRS_M)

warnings.filterwarnings("ignore")
MAX_LAG_KM = 150.0
N_BINS = 20
N_NEIGH = 64


def fit_variogram(x, y, z):
    edges = np.linspace(0, MAX_LAG_KM, N_BINS + 1)
    centers, gamma, counts = gs.vario_estimate((x, y), z, edges, return_counts=True)
    best = None
    top = 1.2 * float(np.nanmax(gamma))
    for Model, ls_max in ((gs.Spherical, MAX_LAG_KM), (gs.Exponential, MAX_LAG_KM / 3)):
        # Bound the practical range to the fitted lag window and the sill to the
        # empirical plateau, so that a regional trend cannot push the range to thousands of km.
        m = Model(dim=2, var=0.5 * top, len_scale=0.3 * ls_max, nugget=0.3 * top)
        m.set_arg_bounds(var=[0.0, top], len_scale=[1.0, ls_max], nugget=[0.0, top])
        try:
            _, _, r2 = m.fit_variogram(centers, gamma, nugget=True, return_r2=True,
                                       weights=counts)
        except RuntimeError:
            continue
        if best is None or r2 > best[1]:
            best = (m, r2)
    emp = pd.DataFrame({"lag_km": centers, "gamma": gamma, "n_pairs": counts})
    return best[0], best[1], emp


def krige(model, x, y, z, px, py):
    ok = OrdinaryKriging(x, y, z, variogram_model=model, exact_values=True)
    est, var = ok.execute("points", px, py, n_closest_points=min(N_NEIGH, len(x) - 1),
                          backend="loop")
    return np.asarray(est), np.clip(np.asarray(var), 0, None)


def make_grid(aquifer):
    poly = load_aquifer(aquifer).geometry.iloc[0]
    tx = load_counties().union_all()
    dom = poly.intersection(tx)
    xmin, ymin, xmax, ymax = dom.bounds
    xmin = np.floor(xmin / GRID_M) * GRID_M
    ymax = np.ceil(ymax / GRID_M) * GRID_M
    nx = int(np.ceil((xmax - xmin) / GRID_M))
    ny = int(np.ceil((ymax - ymin) / GRID_M))
    xs = xmin + (np.arange(nx) + 0.5) * GRID_M
    ys = ymax - (np.arange(ny) + 0.5) * GRID_M
    XX, YY = np.meshgrid(xs, ys)
    inside = contains_xy(dom, XX, YY)
    return XX, YY, inside, (xmin, ymax, nx, ny)


def cross_validate(model_ind, model_log, g, mcl):
    x, y = g["x_km"].to_numpy(), g["y_km"].to_numpy()
    ind, lz = g["exceed"].to_numpy(float), g["log10_conc"].to_numpy(float)
    p_ik = np.full(len(g), np.nan)
    m_ok = np.full(len(g), np.nan)
    v_ok = np.full(len(g), np.nan)
    base = np.full(len(g), np.nan)
    for tr, te in KFold(10, shuffle=True, random_state=SEED).split(x):
        p_ik[te] = np.clip(krige(model_ind, x[tr], y[tr], ind[tr], x[te], y[te])[0], 0, 1)
        m_ok[te], v_ok[te] = krige(model_log, x[tr], y[tr], lz[tr], x[te], y[te])
        base[te] = ind[tr].mean()
    p_ln = 1 - norm.cdf((np.log10(mcl) - m_ok) / np.sqrt(np.maximum(v_ok, 1e-9)))
    err = m_ok - lz
    brier = np.mean((p_ik - ind) ** 2)
    brier0 = np.mean((base - ind) ** 2)
    return {"cv_brier_IK": brier, "cv_brier_baseline": brier0,
            "cv_brier_skill": 1 - brier / brier0,
            "cv_auc_IK": roc_auc_score(ind, p_ik),
            "cv_auc_OK_lognormal": roc_auc_score(ind, p_ln),
            "cv_brier_OK_lognormal": np.mean((p_ln - ind) ** 2),
            "cv_ME_log10": err.mean(), "cv_RMSE_log10": np.sqrt(np.mean(err ** 2)),
            "cv_R2_log10": 1 - np.sum(err ** 2) / np.sum((lz - lz.mean()) ** 2),
            "cv_MSSE_log10": np.mean(err ** 2 / np.maximum(v_ok, 1e-9))}, p_ik


def describe(model, r2, kind):
    return {f"{kind}_model": type(model).__name__, f"{kind}_nugget": model.nugget,
            f"{kind}_partial_sill": model.var,
            # spherical: range = len_scale; exponential: practical range = 3 * len_scale
            f"{kind}_range_km": float(model.len_scale * (3 if isinstance(model, gs.Exponential) else 1)),
            f"{kind}_range_at_upper_bound": bool(np.isclose(
                model.len_scale, MAX_LAG_KM / (3 if isinstance(model, gs.Exponential) else 1))),
            f"{kind}_nugget_ratio": model.nugget / (model.nugget + model.var),
            f"{kind}_fit_R2": r2}


def main():
    vrows, emp_all, grid_all, cv_pts = [], [], [], []
    for aquifer in AQUIFERS:
        XX, YY, inside, (x0, y0, nx, ny) = make_grid(aquifer)
        px, py = XX[inside] / 1000.0, YY[inside] / 1000.0
        for analyte in ("arsenic", "nitrate"):
            g = load_wells(analyte, aquifer)
            x, y = g["x_km"].to_numpy(), g["y_km"].to_numpy()
            mi, r2i, ei = fit_variogram(x, y, g["exceed"].to_numpy(float))
            ml, r2l, el = fit_variogram(x, y, g["log10_conc"].to_numpy(float))
            for e, kind in ((ei, "indicator"), (el, "log10_conc")):
                e = e.assign(analyte=analyte, aquifer=aquifer, variable=kind)
                emp_all.append(e)
            cv, p_cv = cross_validate(mi, ml, g, MCL[analyte])
            cv_pts.append(g[["state_well_number", "analyte", "aquifer", "exceed"]]
                          .assign(cv_p_ik=p_cv))
            row = {"analyte": analyte, "aquifer": aquifer, "n_wells": len(g),
                   "prevalence": g["exceed"].mean()}
            row.update(describe(mi, r2i, "IK"))
            row.update(describe(ml, r2l, "OK"))
            row.update(cv)
            vrows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

            p_ik, v_ik = krige(mi, x, y, g["exceed"].to_numpy(float), px, py)
            p_ik = np.clip(p_ik, 0, 1)
            m_ok, v_ok = krige(ml, x, y, g["log10_conc"].to_numpy(float), px, py)
            p_ln = 1 - norm.cdf((np.log10(MCL[analyte]) - m_ok) / np.sqrt(np.maximum(v_ok, 1e-9)))
            # distance to the nearest sampled well, a plain-language uncertainty proxy
            from scipy.spatial import cKDTree
            dnn = cKDTree(np.c_[x, y]).query(np.c_[px, py])[0]
            grid_all.append(pd.DataFrame({
                "analyte": analyte, "aquifer": aquifer, "x_m": px * 1000, "y_m": py * 1000,
                "p_exceed_ik": p_ik, "ik_variance": v_ik, "ok_log10_mean": m_ok,
                "ok_log10_variance": v_ok, "p_exceed_ok_lognormal": p_ln,
                "dist_nearest_well_km": dnn}))

            bands = []
            for arr in (p_ik, v_ik, m_ok, v_ok):
                b = np.full(XX.shape, np.nan, dtype="float32")
                b[inside] = arr
                bands.append(b)
            fn = GIS_DIR / f"{analyte}_{aquifer.replace(' ', '_').lower()}_kriging.tif"
            with rasterio.open(fn, "w", driver="GTiff", height=ny, width=nx, count=4,
                               dtype="float32", crs=CRS_M, nodata=np.nan, compress="deflate",
                               transform=from_origin(x0, y0, GRID_M, GRID_M)) as dst:
                for i, (b, nm) in enumerate(zip(bands, ["P_exceed_IK", "IK_variance",
                                                        "OK_log10_mean", "OK_log10_variance"]), 1):
                    dst.write(b, i)
                    dst.set_band_description(i, nm)

    vt = pd.DataFrame(vrows)
    vt.round(4).to_csv(TAB_DIR / "table8_variograms_and_cross_validation.csv", index=False)
    pd.concat(emp_all).round(5).to_csv(PROC_DIR / "empirical_variograms.csv", index=False)
    grid = pd.concat(grid_all, ignore_index=True)
    grid.round(5).to_csv(PROC_DIR / "kriging_grid_predictions.csv", index=False)
    pd.concat(cv_pts).round(4).to_csv(PROC_DIR / "cross_validation_well_predictions.csv",
                                      index=False)


if __name__ == "__main__":
    main()
