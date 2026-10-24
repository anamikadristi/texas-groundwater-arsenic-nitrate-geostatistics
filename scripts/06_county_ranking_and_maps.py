"""Step 6: county ranking by estimated exceedance probability, figures, and GIS export.

County ranking: each 5 km grid cell is assigned to the county containing its centre.
For every county x aquifer x analyte with at least MIN_CELLS cells we report the mean
indicator-kriging P(exceed), the share of the county's aquifer area with P > 0.5, the mean
kriging variance, the median distance to the nearest sampled well, and the observed
wells and exceedances. Counties are ranked by mean P(exceed) within each analyte.

GIS export: outputs/gis/texas_groundwater_as_no3.gpkg (wells, LISA, county ranking,
aquifer boundaries) opens directly in ArcGIS Pro and QGIS, alongside the GeoTIFFs from step 5.
"""
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))  # works under python -I

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import rasterio

from common import (load_wells, load_aquifer, load_counties, PROC_DIR, TAB_DIR, FIG_DIR,
                    GIS_DIR, AQUIFERS, MCL, UNITS, GRID_M, CRS_M)

MIN_CELLS = 4  # at least 100 km2 of aquifer in the county
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
EXC, BELOW = "#e34948", "#b9b8b2"
BLUES = LinearSegmentedColormap.from_list("blues", ["#eef4fc", "#9ec5f4", "#3987e5",
                                                    "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fdf1ea", "#f6b896", "#eb6834",
                                                        "#b4441b", "#6e2508"])
LISA_COL = {"High-High": "#e34948", "Low-Low": "#2a78d6", "High-Low": "#f2a7a6",
            "Low-High": "#9ec5f4", "Not significant": "#d6d5d0"}
NAME = {"arsenic": "Arsenic", "nitrate": "Nitrate"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight"})


def base_map(ax, counties, aqs):
    counties.boundary.plot(ax=ax, color=GRID, linewidth=0.3)
    for a in aqs.values():
        a.boundary.plot(ax=ax, color=INK2, linewidth=0.6)
    ax.set_axis_off()
    ax.set_aspect("equal")


def grid_to_gdf(grid):
    return gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid["x_m"], grid["y_m"]),
                            crs=CRS_M)


def county_ranking(grid, wells, counties):
    pts = gpd.sjoin(grid_to_gdf(grid), counties[["county", "geometry"]], predicate="within",
                    how="inner")
    agg = (pts.groupby(["analyte", "aquifer", "county"])
              .agg(n_cells=("p_exceed_ik", "size"),
                   mean_p_exceed=("p_exceed_ik", "mean"),
                   share_area_p_gt_0_5=("p_exceed_ik", lambda s: (s > 0.5).mean()),
                   mean_ik_variance=("ik_variance", "mean"),
                   mean_p_exceed_ok_lognormal=("p_exceed_ok_lognormal", "mean"),
                   median_dist_nearest_well_km=("dist_nearest_well_km", "median"))
              .reset_index())
    agg["aquifer_area_km2"] = agg["n_cells"] * (GRID_M / 1000) ** 2
    obs = (wells.groupby(["analyte", "aquifer", "county"])
                .agg(n_wells_observed=("exceed", "size"), n_wells_exceed=("exceed", "sum"))
                .reset_index())
    agg = agg.merge(obs, on=["analyte", "aquifer", "county"], how="left")
    agg[["n_wells_observed", "n_wells_exceed"]] = agg[["n_wells_observed",
                                                       "n_wells_exceed"]].fillna(0).astype(int)
    agg = agg[agg["n_cells"] >= MIN_CELLS].copy()
    agg["rank_within_analyte"] = (agg.groupby("analyte")["mean_p_exceed"]
                                     .rank(ascending=False, method="min").astype(int))
    agg = agg.sort_values(["analyte", "rank_within_analyte"])
    agg.round(4).to_csv(TAB_DIR / "table9_county_ranking_exceedance_probability.csv",
                        index=False)
    top = agg.groupby("analyte").head(10)
    top.round(3).to_csv(TAB_DIR / "table10_top10_counties.csv", index=False)
    print(top[["analyte", "aquifer", "county", "mean_p_exceed", "share_area_p_gt_0_5",
               "mean_ik_variance", "n_wells_observed", "n_wells_exceed",
               "median_dist_nearest_well_km"]].round(3).to_string(index=False))
    return agg


def raster_layer(analyte, aquifer, band):
    fn = GIS_DIR / f"{analyte}_{aquifer.replace(' ', '_').lower()}_kriging.tif"
    with rasterio.open(fn) as r:
        arr = r.read(band)
        b = r.bounds
    return arr, (b.left, b.right, b.bottom, b.top)


def fig_wells(wells, counties, aqs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, an in zip(axes, ("arsenic", "nitrate")):
        base_map(ax, counties, aqs)
        w = wells[wells["analyte"] == an]
        lo, hi = w[w["exceed"] == 0], w[w["exceed"] == 1]
        ax.scatter(lo.geometry.x, lo.geometry.y, s=3, c=BELOW, linewidths=0)
        ax.scatter(hi.geometry.x, hi.geometry.y, s=6, c=EXC, linewidths=0)
        ax.set_title(f"{NAME[an]}: most recent sample per well, 1990-2026\n"
                     f"{len(hi)} of {len(w)} wells above {MCL[an]:g} {UNITS[an]}",
                     loc="left")
    h = [Line2D([], [], marker="o", ls="", color=EXC, label="Above EPA MCL"),
         Line2D([], [], marker="o", ls="", color=BELOW, label="At or below MCL"),
         Line2D([], [], color=INK2, lw=0.8, label="Aquifer boundary (TWDB)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False)
    fig.savefig(FIG_DIR / "fig1_wells_observed_exceedance.png")
    plt.close(fig)


def fig_lisa(counties, aqs):
    lisa = pd.read_csv(PROC_DIR / "lisa_well_clusters.csv")
    lisa = gpd.GeoDataFrame(lisa, geometry=gpd.points_from_xy(lisa.lon, lisa.lat),
                            crs="EPSG:4269").to_crs(CRS_M)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, an in zip(axes, ("arsenic", "nitrate")):
        base_map(ax, counties, aqs)
        l = lisa[lisa["analyte"] == an]
        for k in ["Not significant", "Low-High", "High-Low", "Low-Low", "High-High"]:
            s = l[l["lisa_cluster_fdr"] == k]
            ax.scatter(s.geometry.x, s.geometry.y, s=3 if k == "Not significant" else 6,
                       c=LISA_COL[k], linewidths=0)
        ax.set_title(f"{NAME[an]}: local Moran's I on log10 concentration\n"
                     "k = 8 neighbours, 999 permutations, FDR q < 0.05", loc="left")
    h = [Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in LISA_COL.items()]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False)
    fig.savefig(FIG_DIR / "fig2_lisa_clusters.png")
    plt.close(fig)


def vario_curve(kind, h, nugget, psill, rng):
    if kind == "Spherical":
        r = np.minimum(h / rng, 1)
        return nugget + psill * (1.5 * r - 0.5 * r ** 3)
    return nugget + psill * (1 - np.exp(-3 * h / rng))


def fig_variograms():
    emp = pd.read_csv(PROC_DIR / "empirical_variograms.csv")
    vt = pd.read_csv(TAB_DIR / "table8_variograms_and_cross_validation.csv")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.4))
    combos = [(an, aq) for aq in AQUIFERS for an in ("arsenic", "nitrate")]
    for j, (an, aq) in enumerate(combos):
        p = vt[(vt.analyte == an) & (vt.aquifer == aq)].iloc[0]
        for i, (var, pre, lab) in enumerate((("indicator", "IK", "indicator"),
                                             ("log10_conc", "OK", "log10 conc."))):
            ax = axes[i, j]
            e = emp[(emp.analyte == an) & (emp.aquifer == aq) & (emp.variable == var)]
            h = np.linspace(0, e.lag_km.max(), 200)
            ax.plot(h, vario_curve(p[f"{pre}_model"], h, p[f"{pre}_nugget"],
                                   p[f"{pre}_partial_sill"], p[f"{pre}_range_km"]),
                    color="#2a78d6", lw=2)
            ax.scatter(e.lag_km, e.gamma, s=14, color=INK, zorder=3)
            ax.set_ylim(0, None)
            ax.grid(color=GRID, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"{NAME[an]}, {aq}: {lab}", loc="left", fontsize=8.5)
            ax.text(0.97, 0.05, f"{p[f'{pre}_model']}\nrange {p[f'{pre}_range_km']:.0f} km\n"
                    f"nugget ratio {p[f'{pre}_nugget_ratio']:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7, color=INK2)
            if i == 1:
                ax.set_xlabel("Lag distance (km)")
            if j == 0:
                ax.set_ylabel("Semivariance")
    fig.suptitle("Empirical variograms (points) and fitted models (line)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_variograms.png")
    plt.close(fig)


def fig_surface(counties, aqs, band, cmap, label, fname, title, vmax=None, wells=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8))
    for ax, an in zip(axes, ("arsenic", "nitrate")):
        vm = vmax if vmax is not None else max(
            np.nanmax(raster_layer(an, aq, band)[0]) for aq in AQUIFERS)
        for aq in AQUIFERS:
            arr, ext = raster_layer(an, aq, band)
            im = ax.imshow(arr, extent=ext, cmap=cmap, vmin=0, vmax=vm, interpolation="nearest")
        base_map(ax, counties, aqs)
        if wells is not None:
            w = wells[wells["analyte"] == an]
            ax.scatter(w.geometry.x, w.geometry.y, s=0.6, c=INK, linewidths=0, alpha=0.5)
        ax.set_title(f"{NAME[an]}: {title}", loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label(label, color=INK2)
        cb.outline.set_visible(False)
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def fig_variance(counties, aqs, wells):
    """Kriging variance per aquifer panel, each on its own scale, wells overlaid."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    for i, an in enumerate(("arsenic", "nitrate")):
        for j, aq in enumerate(AQUIFERS):
            ax = axes[i, j]
            arr, ext = raster_layer(an, aq, 2)
            im = ax.imshow(arr, extent=ext, cmap=ORANGES, vmin=np.nanmin(arr),
                           vmax=np.nanmax(arr), interpolation="nearest")
            counties.boundary.plot(ax=ax, color="white", linewidth=0.3)
            aqs[aq].boundary.plot(ax=ax, color=INK2, linewidth=0.6)
            w = wells[(wells["analyte"] == an) & (wells["aquifer"] == aq)]
            ax.scatter(w.geometry.x, w.geometry.y, s=0.8, c=INK, linewidths=0, alpha=0.6)
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_axis_off()
            ax.set_title(f"{NAME[an]}, {aq}: indicator kriging variance", loc="left")
            cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
            cb.outline.set_visible(False)
    fig.text(0.01, 0.005, "Higher variance = fewer nearby wells. Each panel has its own colour "
             "scale. Dots are sampled wells.", color=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_ik_kriging_variance.png")
    plt.close(fig)


def fig_depth():
    t = pd.read_csv(TAB_DIR / "table6_exceedance_by_depth_class.csv")
    classes = ["<100 ft", "100-199 ft", "200-399 ft", "400-799 ft", ">=800 ft"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    cols = {"Gulf Coast": "#2a78d6", "Ogallala": "#eb6834"}
    x = np.arange(len(classes))
    for ax, an in zip(axes, ("arsenic", "nitrate")):
        for k, aq in enumerate(AQUIFERS):
            d = t[(t.analyte == an) & (t.aquifer == aq)].set_index("depth_class").reindex(classes)
            bars = ax.bar(x + (k - 0.5) * 0.4, d.pct_exceed, 0.38, color=cols[aq], label=aq,
                          edgecolor="white", linewidth=1)
            for b, n in zip(bars, d.n_wells):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"n={int(n)}",
                        ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(x, classes)
        ax.set_title(f"{NAME[an]}: share of wells above the MCL by well depth", loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Reported well depth")
    axes[0].set_ylabel("Wells above MCL (%)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_exceedance_by_depth.png")
    plt.close(fig)


def fig_counties(rank):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    cols = {"Gulf Coast": "#2a78d6", "Ogallala": "#eb6834"}
    for ax, an in zip(axes, ("arsenic", "nitrate")):
        d = rank[rank.analyte == an].head(15).iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d.mean_p_exceed, color=[cols[a] for a in d.aquifer], height=0.7,
                edgecolor="white", linewidth=1)
        frac = np.where(d.n_wells_observed > 0, d.n_wells_exceed / d.n_wells_observed.clip(lower=1),
                        np.nan)
        ax.scatter(frac, y, marker="D", s=16, color=INK, zorder=3)
        ax.set_yticks(y, [f"{c} (n={n})" for c, n in zip(d.county, d.n_wells_observed)])
        ax.set_xlim(0, 1.05)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Mean estimated P(exceed MCL) across county aquifer area")
        ax.set_title(f"{NAME[an]}: top 15 county-aquifer units", loc="left")
    h = [Line2D([], [], color=c, lw=6, label=a) for a, c in cols.items()]
    h.append(Line2D([], [], marker="D", ls="", color=INK, label="Observed share of wells above MCL"))
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG_DIR / "fig7_county_ranking.png")
    plt.close(fig)


def fig_calibration():
    cv = pd.read_csv(PROC_DIR / "cross_validation_well_predictions.csv")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    bins = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0001])
    combos = [(an, aq) for aq in AQUIFERS for an in ("arsenic", "nitrate")]
    for ax, (an, aq) in zip(axes, combos):
        d = cv[(cv.analyte == an) & (cv.aquifer == aq)].copy()
        d["bin"] = pd.cut(d.cv_p_ik, bins, right=False)
        g = d.groupby("bin", observed=True).agg(p=("cv_p_ik", "mean"), o=("exceed", "mean"),
                                                n=("exceed", "size"))
        g = g[g.n >= 10]
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.plot(g.p, g.o, color="#2a78d6", lw=2, marker="o", ms=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{NAME[an]}, {aq}", loc="left", fontsize=8.5)
        ax.set_xlabel("Cross-validated P(exceed)")
    axes[0].set_ylabel("Observed share above MCL")
    fig.suptitle("Indicator kriging calibration, 10-fold cross-validation (bins with n >= 10)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig8_cv_calibration.png")
    plt.close(fig)


def main():
    counties = load_counties()
    aqs = {a: load_aquifer(a) for a in AQUIFERS}
    wells = load_wells()
    grid = pd.read_csv(PROC_DIR / "kriging_grid_predictions.csv")
    rank = county_ranking(grid, wells, counties)

    fig_wells(wells, counties, aqs)
    fig_lisa(counties, aqs)
    fig_variograms()
    fig_surface(counties, aqs, 1, BLUES, "P(exceed MCL)", "fig4_ik_exceedance_probability.png",
                "indicator kriging P(exceed MCL)", vmax=1.0, wells=wells)
    fig_variance(counties, aqs, wells)
    fig_depth()
    fig_counties(rank)
    fig_calibration()

    # GeoPackage for ArcGIS Pro / QGIS
    gp = GIS_DIR / "texas_groundwater_as_no3.gpkg"
    if gp.exists():
        gp.unlink()
    wells.drop(columns=["x_km", "y_km"]).assign(date=wells["date"].astype(str)).to_file(
        gp, layer="wells_most_recent", driver="GPKG")
    lisa = pd.read_csv(PROC_DIR / "lisa_well_clusters.csv")
    gpd.GeoDataFrame(lisa, geometry=gpd.points_from_xy(lisa.lon, lisa.lat), crs="EPSG:4269"
                     ).to_crs(CRS_M).to_file(gp, layer="lisa_clusters", driver="GPKG")
    cr = counties.merge(rank, on="county", how="inner")
    cr.to_file(gp, layer="county_ranking", driver="GPKG")
    for a, g in aqs.items():
        g.assign(aquifer=a).to_file(gp, layer=f"aquifer_{a.replace(' ', '_').lower()}",
                                    driver="GPKG")
    print("wrote", gp)


if __name__ == "__main__":
    main()
