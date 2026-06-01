#!/usr/bin/env python3
"""Generate focused Boston map for Roxbury + Mattapan with delta vs city median access.

Outputs:
- map_focus_roxbury_mattapan_delta_vs_city_median.png
- focus_pockets_summary.csv
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.font_manager import FontProperties, findSystemFonts
from shapely.geometry import MultiPolygon, Polygon
from generate_access_figures import build_street_basemap, draw_basemap_and_extent, load_icon_array

DEFAULT_BOSTON_CITY_ZIPS = {
    "02108", "02109", "02110", "02111", "02113", "02114", "02115", "02116", "02118",
    "02119", "02120", "02121", "02122", "02124", "02125", "02126", "02127", "02128",
    "02129", "02130", "02131", "02132", "02134", "02135", "02136", "02163", "02199",
    "02203", "02210", "02215",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Focused map for Roxbury+Mattapan with differences to city median")
    ap.add_argument("--data-root", default="docs/data/boston_ma_nh")
    ap.add_argument("--mode", choices=["walk", "walk_transit"], default="walk_transit")
    ap.add_argument("--minutes", type=int, default=20)
    ap.add_argument("--target-zips", nargs="+", default=["02119", "02120", "02121", "02122", "02124", "02125", "02126"])
    ap.add_argument("--city-zips", nargs="+", default=[], help="Optional ZIP list for city baseline; default uses 021* and 022* ZIPs in dataset")
    ap.add_argument("--out-dir", default="reports/figures/boston_focus")
    ap.add_argument("--hotspot-quantile", type=float, default=0.25)
    ap.add_argument("--grocery-buffer-m", type=float, default=1800.0, help="Show groceries within this distance of target tracts")
    ap.add_argument("--no-basemap", action="store_true", help="Disable OSM street basemap")
    ap.add_argument("--basemap-zoom", type=int, default=12)
    ap.add_argument("--max-basemap-tiles", type=int, default=36)
    ap.add_argument("--tile-cache-dir", default="cache/tiles/osm")
    ap.add_argument("--target-logo", default="docs/favicon_fsfn.svg")
    ap.add_argument("--target-logo-zoom", type=float, default=0.075)
    return ap.parse_args()


def _fontawesome_properties() -> FontProperties | None:
    fonts = findSystemFonts()
    fa = next((f for f in fonts if "fontawesome" in Path(f).name.lower()), None)
    if not fa:
        return None
    return FontProperties(fname=fa)


def _plot_fa_icons(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    glyph: str,
    color: str,
    size: float,
    zorder: int,
    fa_prop: FontProperties | None,
) -> bool:
    if gdf.empty or fa_prop is None:
        return False
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        ax.text(
            geom.x,
            geom.y,
            glyph,
            color=color,
            fontproperties=fa_prop,
            fontsize=size,
            ha="center",
            va="center",
            zorder=zorder,
        )
    return True


def _load_logo_icon(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    if path.suffix.lower() != ".svg":
        return load_icon_array(path, icon_bg_tolerance=26)

    tmp_png: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = Path(tmp.name)
        # Rasterize SVG with alpha retained.
        subprocess.run(
            ["convert", str(path), f"png32:{tmp_png}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return load_icon_array(tmp_png, icon_bg_tolerance=26)
    except Exception:
        return None
    finally:
        if tmp_png is not None and tmp_png.exists():
            try:
                tmp_png.unlink()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracts = gpd.read_file(data_root / f"tract_{args.mode}_{args.minutes}.geojson")
    grid = gpd.read_file(data_root / f"metro_grid_{args.mode}_{args.minutes}.geojson")
    zips = gpd.read_file(data_root / f"zip_{args.mode}_{args.minutes}.geojson")
    groceries = gpd.read_file(data_root / "groceries.geojson")
    city_pantries = gpd.read_file(data_root / "city_pantries.geojson") if (data_root / "city_pantries.geojson").exists() else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    proposed_sites = gpd.read_file(data_root / f"foodbanks_{args.mode}.geojson") if (data_root / f"foodbanks_{args.mode}.geojson").exists() else gpd.read_file(data_root / "foodbanks_walk.geojson")

    for g in [tracts, grid, zips, groceries, city_pantries, proposed_sites]:
        if g.crs is None:
            g.set_crs("EPSG:4326", inplace=True)

    # normalize ZIP strings
    zips = zips.copy()
    zips["GEOID"] = zips["GEOID"].astype(str).str.zfill(5)
    target_zips = {z.zfill(5) for z in args.target_zips}
    target_zip_polys = zips[zips["GEOID"].isin(target_zips)]
    if target_zip_polys.empty:
        raise ValueError(f"No ZIP polygons found for {sorted(target_zips)}")

    # True modeled access time from metro grid (walk+transit), aggregated to tracts by population-weighted mean.
    if "min_access_minutes" not in grid.columns:
        raise ValueError(f"metro_grid_{args.mode}_{args.minutes}.geojson missing min_access_minutes")
    if "POPULATION" not in grid.columns:
        raise ValueError(f"metro_grid_{args.mode}_{args.minutes}.geojson missing POPULATION")

    # City baseline: use supplied ZIP list, otherwise use Boston-only ZIP list.
    if args.city_zips:
        city_zips = {z.zfill(5) for z in args.city_zips}
    else:
        city_zips = {z for z in set(zips["GEOID"].astype(str).str.zfill(5)) if z in DEFAULT_BOSTON_CITY_ZIPS}
    city_zip_polys = zips[zips["GEOID"].isin(city_zips)]
    city_union = city_zip_polys.to_crs("EPSG:3857").union_all() if not city_zip_polys.empty else None

    target_union_full = target_zip_polys.to_crs("EPSG:3857").union_all()
    # Drop outlying islands by keeping the largest contiguous polygon only.
    if isinstance(target_union_full, MultiPolygon):
        target_union = max(target_union_full.geoms, key=lambda g: g.area)
    elif isinstance(target_union_full, Polygon):
        target_union = target_union_full
    else:
        target_union = target_union_full
    tracts_m = tracts.to_crs("EPSG:3857").copy()
    grid_m = grid.to_crs("EPSG:3857").copy()
    groceries_m = groceries.to_crs("EPSG:3857")
    city_pantries_m = city_pantries.to_crs("EPSG:3857") if not city_pantries.empty else city_pantries
    proposed_sites_m = proposed_sites.to_crs("EPSG:3857")

    # Attach population-weighted true access time to each tract.
    grid_m["min_access_minutes"] = pd.to_numeric(grid_m["min_access_minutes"], errors="coerce")
    grid_m["POPULATION"] = pd.to_numeric(grid_m["POPULATION"], errors="coerce").fillna(0.0)
    grid_pts = grid_m.copy()
    grid_pts["geometry"] = grid_pts.geometry.centroid
    joined = gpd.sjoin(
        grid_pts[["min_access_minutes", "POPULATION", "geometry"]],
        tracts_m[["GEOID", "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.dropna(subset=["min_access_minutes"])
    joined["w"] = joined["POPULATION"].where(joined["POPULATION"] > 0, 1.0)
    tract_time = (
        joined.assign(wx=joined["min_access_minutes"] * joined["w"])
        .groupby("GEOID", as_index=False)[["wx", "w"]]
        .sum()
    )
    tract_time["access_time_min_true"] = tract_time["wx"] / tract_time["w"].replace(0, np.nan)
    tracts_m = tracts_m.merge(tract_time[["GEOID", "access_time_min_true"]], on="GEOID", how="left")

    focus_tracts = tracts_m[tracts_m.geometry.intersects(target_union)].copy()
    if focus_tracts.empty:
        raise ValueError("No tracts intersect target community footprint")

    focus_tracts["coverage_pct"] = pd.to_numeric(focus_tracts["coverage_pct"], errors="coerce").fillna(0.0)
    focus_tracts["POPULATION"] = pd.to_numeric(focus_tracts["POPULATION"], errors="coerce").fillna(0).astype(int)
    if city_union is not None:
        city_tracts = tracts_m[tracts_m.geometry.intersects(city_union)].copy()
    else:
        city_tracts = tracts_m.copy()
    city_times = pd.to_numeric(city_tracts["access_time_min_true"], errors="coerce").dropna()
    city_median_time = float(city_times.median())
    city_mean_time = float(city_times.mean())
    city_std_time = float(city_times.std(ddof=0)) if len(city_times) else 0.0
    focus_tracts["delta_vs_city_median_min"] = focus_tracts["access_time_min_true"] - city_median_time

    # define deep-need pockets within focus communities
    q = float(focus_tracts["access_time_min_true"].quantile(1.0 - args.hotspot_quantile))
    pocket_cutoff = max(0.0, q)
    pockets = focus_tracts[(focus_tracts["access_time_min_true"] >= pocket_cutoff) & (focus_tracts["POPULATION"] > 200)].copy()
    pockets = pockets.sort_values("access_time_min_true", ascending=False)

    # clip points to focus footprint
    groceries_focus = groceries_m[groceries_m.geometry.distance(target_union) <= float(args.grocery_buffer_m)]
    city_pantries_focus = city_pantries_m[city_pantries_m.geometry.within(target_union)] if not city_pantries_m.empty else city_pantries_m
    proposed_focus = proposed_sites_m[proposed_sites_m.geometry.within(target_union)]

    boundary_focus = gpd.GeoDataFrame(geometry=[target_union], crs="EPSG:3857")
    basemap_img, basemap_extent = build_street_basemap(
        boundary_focus,
        use_basemap=(not args.no_basemap),
        basemap_zoom=args.basemap_zoom,
        tile_cache_dir=Path(args.tile_cache_dir),
        max_tiles=args.max_basemap_tiles,
    )

    # map
    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    draw_basemap_and_extent(ax, boundary_focus, basemap_img, basemap_extent)

    # context city tracts very light so street basemap stays visible
    tracts_m.plot(ax=ax, color="#f2f2f2", edgecolor="#ffffff", linewidth=0.08, alpha=0.16, zorder=1)

    vals = pd.to_numeric(focus_tracts["delta_vs_city_median_min"], errors="coerce")
    v = float(np.nanquantile(np.abs(vals), 0.95))
    if not np.isfinite(v) or v <= 0:
        v = 1.0
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    # Diverging map: below median green, median white, above median red.
    cmap = colors.LinearSegmentedColormap.from_list(
        "delta_minutes_median",
        ["#1a9850", "#ffffff", "#d73027"],
        N=256,
    )
    focus_tracts = focus_tracts.copy()
    tract_area_km2 = (focus_tracts.geometry.area / 1_000_000.0).replace(0, np.nan)
    focus_tracts["pop_density_km2"] = focus_tracts["POPULATION"] / tract_area_km2
    density = pd.to_numeric(focus_tracts["pop_density_km2"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Robust scaling avoids tiny-area outliers dominating opacity.
    dlog = np.log1p(density.clip(lower=0.0))
    dlo = float(dlog.quantile(0.05))
    dhi = float(dlog.quantile(0.95))
    if dhi > dlo:
        dnorm = ((dlog - dlo) / (dhi - dlo)).clip(0.0, 1.0)
    else:
        dnorm = pd.Series(1.0, index=focus_tracts.index)
    # Stronger visibility: keep low-density tracts readable while preserving contrast.
    focus_tracts["density_alpha"] = np.clip(0.45 + np.power(dnorm, 0.5) * 0.55, 0.45, 1.0)

    for _, row in focus_tracts.iterrows():
        tract_color = cmap(norm(float(row["delta_vs_city_median_min"])))
        gpd.GeoSeries([row.geometry], crs=focus_tracts.crs).plot(
            ax=ax,
            color=tract_color,
            edgecolor="#f4f4f4",
            linewidth=0.2,
            alpha=float(row["density_alpha"]),
            zorder=2,
        )

    # Community boundary
    gpd.GeoSeries([target_union], crs="EPSG:3857").boundary.plot(ax=ax, color="#222222", linewidth=1.2, zorder=3)

    fa_prop = _fontawesome_properties()
    used_grocery_fa = _plot_fa_icons(ax, groceries_focus, "\uf07a", "#1b9e77", size=34.5, zorder=4, fa_prop=fa_prop)  # shopping-cart
    used_pantry_fa = _plot_fa_icons(ax, city_pantries_focus, "\uf187", "#d7301f", size=34.5, zorder=5, fa_prop=fa_prop)  # archive/box

    logo_arr = _load_logo_icon(Path(args.target_logo))
    used_logo = False
    if (logo_arr is not None) and (not proposed_focus.empty):
        for geom in proposed_focus.geometry:
            if geom is None or geom.is_empty:
                continue
            ab = AnnotationBbox(
                OffsetImage(logo_arr, zoom=float(args.target_logo_zoom)),
                (geom.x, geom.y),
                frameon=False,
                box_alignment=(0.5, 0.5),
                zorder=6,
            )
            ax.add_artist(ab)
            used_logo = True
    elif not proposed_focus.empty:
        proposed_focus.plot(ax=ax, marker="s", color="#2f4858", markersize=30, alpha=0.95, zorder=6)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.82)
    cbar.set_label("Minutes Above/Below City Median Transit Time")

    ax.set_title(
        (
            "Roxbury + Mattapan + Dorchester: Estimated transit_time\n"
            f"City median transit_time ≈ {city_median_time:.1f} min (white)"
        ),
        fontsize=12,
        pad=10,
    )
    ax.set_axis_off()

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1b9e77", markersize=6, label="Large Markets/Groceries"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d7301f", markersize=6, label="Existing city pantries"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#2f4858", markersize=6, label="Target service sites"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95)

    minx, miny, maxx, maxy = gpd.GeoSeries([target_union], crs="EPSG:3857").total_bounds
    padx_left = (maxx - minx) * 0.18
    padx_right = (maxx - minx) * 0.06
    pady_bottom = (maxy - miny) * 0.06
    pady_top = (maxy - miny) * 0.18
    ax.set_xlim(minx - padx_left, maxx + padx_right)
    ax.set_ylim(miny - pady_bottom, maxy + pady_top)

    out_png = out_dir / "map_focus_roxbury_mattapan_delta_vs_city_median.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_csv = out_dir / "focus_pockets_summary.csv"
    pockets_out = pockets[["GEOID", "POPULATION", "coverage_pct", "access_time_min_true", "delta_vs_city_median_min"]].copy()
    pockets_out = pockets_out.sort_values("access_time_min_true", ascending=False)
    pockets_out.to_csv(out_csv, index=False)

    print(f"[OK] Wrote {out_png}")
    print(f"[OK] Wrote {out_csv}")
    print(f"[INFO] Focus ZIPs: {sorted(target_zips)}")
    print(f"[INFO] City ZIP baseline: {sorted(city_zips)}")
    print(f"[INFO] City median transit_time: {city_median_time:.2f} min")
    print(f"[INFO] City mean transit_time: {city_mean_time:.2f} min")
    print(f"[INFO] Pocket cutoff (access_time_min_true): {pocket_cutoff:.2f} min")
    print(
        f"[INFO] Focus tract transit_time range: "
        f"{float(pd.to_numeric(focus_tracts['access_time_min_true'], errors='coerce').dropna().min()):.2f} - "
        f"{float(pd.to_numeric(focus_tracts['access_time_min_true'], errors='coerce').dropna().max()):.2f} min"
    )
    print(f"[INFO] Markets/groceries in focus map: {len(groceries_focus)}")
    print(f"[INFO] Existing city pantries in focus map: {len(city_pantries_focus)}")
    print(f"[INFO] Target service sites in focus map: {len(proposed_focus)}")
    if basemap_img is None and not args.no_basemap:
        print("[WARN] Street basemap unavailable; rendered without tiles.")


if __name__ == "__main__":
    main()
