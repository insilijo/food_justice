#!/usr/bin/env python3
"""Generate static food-access figures for outreach and siting decisions.

Outputs per metro:
- map_01_markets_pantries_<mode>_<minutes>min.png
- map_02_underserved_pockets_<mode>_<minutes>min.png
- pockets_summary_<mode>_<minutes>min.csv
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import cm, colors
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image


PLOT_CRS = "EPSG:3857"
MIN_ALPHA_POPULATED = 0.03
MAX_ALPHA_POPULATED = 0.98
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
EARTH_R = 6378137.0
DENSITY_OPACITY_GAMMA = 1.4
DENSITY_LOW_Q = 0.02
DENSITY_HIGH_Q = 0.90


def load_manifest(data_root: Path) -> dict:
    manifest_path = data_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    with manifest_path.open() as f:
        data = json.load(f)
    metros = data.get("metros", {})
    if not isinstance(metros, dict):
        raise ValueError("manifest.json has unexpected structure for `metros`")
    return metros


def load_layer(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing layer: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def load_optional_layer(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        lon_col = next((c for c in ["lon", "lng", "longitude", "x"] if c in df.columns), None)
        lat_col = next((c for c in ["lat", "latitude", "y"] if c in df.columns), None)
        if lon_col and lat_col:
            return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        raise ValueError(f"CSV {path} must include lon/lat columns")
    return load_layer(path)


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = (x / EARTH_R) * (180.0 / math.pi)
    lat = (2.0 * math.atan(math.exp(y / EARTH_R)) - math.pi / 2.0) * (180.0 / math.pi)
    return lon, lat


def lonlat_to_mercator(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = EARTH_R * math.radians(lon)
    y = EARTH_R * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    xt = int((lon + 180.0) / 360.0 * n)
    yt = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    xt = max(0, min(xt, n - 1))
    yt = max(0, min(yt, n - 1))
    return xt, yt


def tile_to_lonlat(xt: int, yt: int, zoom: int) -> tuple[float, float]:
    n = 2**zoom
    lon = xt / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * yt / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def choose_zoom_for_bounds(min_lon: float, min_lat: float, max_lon: float, max_lat: float, max_tiles: int) -> int:
    for z in range(13, 7, -1):
        x0, y1 = lonlat_to_tile(min_lon, min_lat, z)
        x1, y0 = lonlat_to_tile(max_lon, max_lat, z)
        nx = abs(x1 - x0) + 1
        ny = abs(y1 - y0) + 1
        if nx * ny <= max_tiles:
            return z
    return 8


def fetch_osm_tile(z: int, x: int, y: int, cache_root: Path, timeout_s: float = 3.0) -> Image.Image | None:
    tile_path = cache_root / str(z) / str(x) / f"{y}.png"
    if tile_path.exists():
        try:
            return Image.open(tile_path).convert("RGB")
        except Exception:
            pass

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    url = OSM_TILE_URL.format(z=z, x=x, y=y)
    try:
        r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "food-justice-figure-generator/1.0"})
        r.raise_for_status()
        tile_path.write_bytes(r.content)
        return Image.open(tile_path).convert("RGB")
    except Exception:
        return None


def build_street_basemap(
    boundary: gpd.GeoDataFrame,
    use_basemap: bool,
    basemap_zoom: int | None,
    tile_cache_dir: Path,
    max_tiles: int,
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    if not use_basemap or boundary.empty:
        return None, None

    minx, miny, maxx, maxy = boundary.total_bounds
    min_lon, min_lat = mercator_to_lonlat(minx, miny)
    max_lon, max_lat = mercator_to_lonlat(maxx, maxy)
    zoom = basemap_zoom if basemap_zoom is not None else choose_zoom_for_bounds(min_lon, min_lat, max_lon, max_lat, max_tiles)

    # Connectivity preflight to avoid waiting on many failed requests.
    test_x, test_y = lonlat_to_tile((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0, zoom)
    if fetch_osm_tile(zoom, test_x, test_y, tile_cache_dir) is None:
        return None, None

    x0, y1 = lonlat_to_tile(min_lon, min_lat, zoom)
    x1, y0 = lonlat_to_tile(max_lon, max_lat, zoom)
    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    if nx * ny > max_tiles:
        return None, None

    canvas = Image.new("RGB", (nx * 256, ny * 256), color=(245, 245, 245))
    got_any_tile = False
    for yy in range(y_min, y_max + 1):
        for xx in range(x_min, x_max + 1):
            tile = fetch_osm_tile(zoom, xx, yy, tile_cache_dir)
            if tile is None:
                continue
            got_any_tile = True
            px = (xx - x_min) * 256
            py = (yy - y_min) * 256
            canvas.paste(tile, (px, py))
    if not got_any_tile:
        return None, None

    left_lon, top_lat = tile_to_lonlat(x_min, y_min, zoom)
    right_lon, bottom_lat = tile_to_lonlat(x_max + 1, y_max + 1, zoom)
    left_x, top_y = lonlat_to_mercator(left_lon, top_lat)
    right_x, bottom_y = lonlat_to_mercator(right_lon, bottom_lat)
    extent = (left_x, right_x, bottom_y, top_y)
    return np.asarray(canvas), extent


def draw_basemap_and_extent(
    ax,
    boundary: gpd.GeoDataFrame,
    basemap_img: np.ndarray | None,
    basemap_extent: tuple[float, float, float, float] | None,
) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    if basemap_img is not None and basemap_extent is not None:
        ax.imshow(basemap_img, extent=basemap_extent, zorder=0, interpolation="bilinear", origin="upper")


def knockout_border_background(
    rgba: np.ndarray,
    bg_tolerance: int = 26,
    min_alpha_keep: int = 8,
    do_autocrop: bool = True,
    crop_pad: int = 1,
) -> np.ndarray:
    arr = rgba.copy()
    h, w, _ = arr.shape
    rgb = arr[:, :, :3].astype(np.int16)
    alpha = arr[:, :, 3]

    # Estimate background from border pixels.
    border = np.concatenate([rgb[0, :, :], rgb[h - 1, :, :], rgb[:, 0, :], rgb[:, w - 1, :]], axis=0)
    bg = np.median(border, axis=0)

    # Candidate background by color distance from border median.
    dist = np.sqrt(((rgb - bg) ** 2).sum(axis=2))
    candidate = dist <= float(bg_tolerance)

    # Flood-fill from border over candidate pixels only, to avoid removing similar interior colors.
    flood = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if candidate[0, x]:
            q.append((0, x))
        if candidate[h - 1, x]:
            q.append((h - 1, x))
    for y in range(h):
        if candidate[y, 0]:
            q.append((y, 0))
        if candidate[y, w - 1]:
            q.append((y, w - 1))
    while q:
        y, x = q.popleft()
        if flood[y, x] or not candidate[y, x]:
            continue
        flood[y, x] = True
        if y > 0:
            q.append((y - 1, x))
        if y < h - 1:
            q.append((y + 1, x))
        if x > 0:
            q.append((y, x - 1))
        if x < w - 1:
            q.append((y, x + 1))

    arr[flood, 3] = 0

    # Also clear very low-alpha remnants.
    arr[arr[:, :, 3] < min_alpha_keep, 3] = 0

    if do_autocrop:
        nz = np.argwhere(arr[:, :, 3] > 0)
        if nz.size > 0:
            y0, x0 = nz.min(axis=0)
            y1, x1 = nz.max(axis=0)
            y0 = max(0, y0 - crop_pad)
            x0 = max(0, x0 - crop_pad)
            y1 = min(h - 1, y1 + crop_pad)
            x1 = min(w - 1, x1 + crop_pad)
            arr = arr[y0 : y1 + 1, x0 : x1 + 1, :]

    return arr


def plot_icon_points(
    ax,
    gdf: gpd.GeoDataFrame,
    icon_path: Path,
    icon_bg_tolerance: int = 26,
    zoom: float = 0.13,
    zorder: int = 4,
) -> bool:
    if gdf.empty or not icon_path.exists():
        return False
    try:
        icon_rgba = Image.open(icon_path).convert("RGBA")
        arr = knockout_border_background(np.array(icon_rgba), bg_tolerance=icon_bg_tolerance)
        icon = arr
    except Exception:
        return False
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        ab = AnnotationBbox(
            OffsetImage(icon, zoom=zoom),
            (geom.x, geom.y),
            frameon=False,
            box_alignment=(0.5, 0.5),
            zorder=zorder,
        )
        ax.add_artist(ab)
    return True


class _LogoLegendHandle:
    pass


class _LogoLegendHandler(HandlerBase):
    def __init__(self, image_arr: np.ndarray):
        super().__init__()
        self.image_arr = image_arr

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        bb = Bbox.from_bounds(xdescent, ydescent, width, height)
        img = BboxImage(bb, interpolation="bilinear", transform=trans)
        img.set_data(self.image_arr)
        return [img]


def load_icon_array(icon_path: Path, icon_bg_tolerance: int = 26) -> np.ndarray | None:
    if not icon_path.exists():
        return None
    try:
        icon_rgba = Image.open(icon_path).convert("RGBA")
        return knockout_border_background(np.array(icon_rgba), bg_tolerance=icon_bg_tolerance)
    except Exception:
        return None


def prep_metro_data(data_root: Path, metro: str, mode: str, minutes: int) -> dict:
    metro_dir = data_root / metro
    tracts = load_layer(metro_dir / f"tract_{mode}_{minutes}.geojson")
    groceries = load_layer(metro_dir / "groceries.geojson")
    proposed_pantries_path = metro_dir / f"foodbanks_{mode}.geojson"
    if not proposed_pantries_path.exists():
        fallback = metro_dir / "foodbanks_walk.geojson"
        if fallback.exists():
            proposed_pantries_path = fallback
    proposed_pantries = load_layer(proposed_pantries_path)
    boundary = load_layer(metro_dir / "metro_boundary.geojson")

    for col in ["coverage_pct", "POPULATION"]:
        if col not in tracts.columns:
            raise ValueError(f"Missing `{col}` in {metro}/tract_{mode}_{minutes}.geojson")

    tracts = tracts[["GEOID", "POPULATION", "coverage_pct", "geometry"]].copy()
    tracts["coverage_pct"] = pd.to_numeric(tracts["coverage_pct"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    tracts["POPULATION"] = pd.to_numeric(tracts["POPULATION"], errors="coerce").fillna(0).astype(int)
    tracts["pop_without_access"] = tracts["POPULATION"] * (1.0 - tracts["coverage_pct"])

    if tracts.crs is None:
        tracts = tracts.set_crs("EPSG:4326")

    tracts = tracts.to_crs(PLOT_CRS)
    groceries = groceries.to_crs(PLOT_CRS)
    proposed_pantries = proposed_pantries.to_crs(PLOT_CRS)
    boundary = boundary.to_crs(PLOT_CRS)
    tracts = add_density_fields(tracts)

    return {
        "tracts": tracts,
        "groceries": groceries,
        "proposed_pantries": proposed_pantries,
        "boundary": boundary,
    }


def add_density_fields(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tracts = tracts.copy()
    tracts["area_km2"] = tracts.geometry.area / 1_000_000.0
    tracts["area_km2"] = tracts["area_km2"].replace(0, np.nan)
    tracts["pop_density_per_km2"] = (tracts["POPULATION"] / tracts["area_km2"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dens = tracts["pop_density_per_km2"].astype(float).clip(lower=0.0)
    # Keep 0..1 endpoints, but use log scaling so dense urban cores don't squash all other tracts to ~0.
    dens_t = np.log1p(dens)
    if len(dens_t):
        dmin = float(dens_t.quantile(DENSITY_LOW_Q))
        dmax = float(dens_t.quantile(DENSITY_HIGH_Q))
    else:
        dmin, dmax = 0.0, 0.0
    if dmax > dmin:
        norm = ((dens_t - dmin) / (dmax - dmin)).clip(0.0, 1.0)
        tracts["density_opacity"] = norm ** DENSITY_OPACITY_GAMMA
    else:
        tracts["density_opacity"] = 1.0
    return tracts


def rgba_colors(hex_color: str, alpha_values: pd.Series | np.ndarray) -> list[tuple[float, float, float, float]]:
    rgb = colors.to_rgb(hex_color)
    return [(rgb[0], rgb[1], rgb[2], float(a)) for a in alpha_values]


def add_opacity_colorbar(fig, ax, pad: float = 0.08, shrink: float = 0.8, horizontal: bool = False):
    opacity_cmap = colors.LinearSegmentedColormap.from_list(
        "opacity_bar",
        [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)],
    )
    sm_opacity = cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=1.0), cmap=opacity_cmap)
    if horizontal:
        cax = inset_axes(ax, width="34%", height="2.8%", loc="lower center", borderpad=2.3)
        cbar = fig.colorbar(sm_opacity, cax=cax, orientation="horizontal")
        cbar.set_label("Density opacity", fontsize=8, labelpad=1)
        cbar.ax.tick_params(labelsize=7, pad=1)
    else:
        cbar = fig.colorbar(sm_opacity, ax=ax, fraction=0.03, pad=pad, shrink=shrink)
        cbar.set_label("Density opacity (0=lowest, 1=highest)")
        cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0.0, 0.5, 1.0])
    return cbar


def prep_walk_time_data(data_root: Path, metro: str, mode: str, long_walk_minutes: int) -> tuple[gpd.GeoDataFrame, float]:
    metro_dir = data_root / metro
    walk10 = load_layer(metro_dir / f"tract_{mode}_10.geojson")
    walk15 = load_layer(metro_dir / f"tract_{mode}_15.geojson")
    walk20 = load_layer(metro_dir / f"tract_{mode}_20.geojson")

    for gdf in [walk10, walk15, walk20]:
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

    w10 = walk10[["GEOID", "coverage_pct"]].rename(columns={"coverage_pct": "cov10"})
    w15 = walk15[["GEOID", "coverage_pct"]].rename(columns={"coverage_pct": "cov15"})
    tracts = walk20[["GEOID", "POPULATION", "coverage_pct", "geometry"]].rename(columns={"coverage_pct": "cov20"})

    tracts = tracts.merge(w10, on="GEOID", how="left").merge(w15, on="GEOID", how="left")
    for c in ["cov10", "cov15", "cov20"]:
        tracts[c] = pd.to_numeric(tracts[c], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    tracts["cov15"] = np.maximum(tracts["cov15"], tracts["cov10"])
    tracts["cov20"] = np.maximum(tracts["cov20"], tracts["cov15"])

    tracts["POPULATION"] = pd.to_numeric(tracts["POPULATION"], errors="coerce").fillna(0).astype(int)

    tracts["avg_walk_min_est"] = (
        5.0 * tracts["cov10"]
        + 12.5 * (tracts["cov15"] - tracts["cov10"])
        + 17.5 * (tracts["cov20"] - tracts["cov15"])
        + float(long_walk_minutes) * (1.0 - tracts["cov20"])
    )

    tracts = tracts.to_crs(PLOT_CRS)
    tracts = add_density_fields(tracts)

    weights = tracts["POPULATION"].astype(float)
    if float(weights.sum()) > 0:
        metro_avg = float(np.average(tracts["avg_walk_min_est"], weights=weights))
    else:
        metro_avg = float(tracts["avg_walk_min_est"].mean())
    return tracts, metro_avg


def classify_pockets(tracts: gpd.GeoDataFrame, pocket_quantile: float) -> tuple[gpd.GeoDataFrame, float, float]:
    weights = tracts["POPULATION"].astype(float)
    if float(weights.sum()) > 0:
        metro_mean = float(np.average(tracts["coverage_pct"], weights=weights))
    else:
        metro_mean = float(tracts["coverage_pct"].mean())

    tracts = tracts.copy()
    tracts["underserved"] = tracts["coverage_pct"] < metro_mean

    underserved_vals = tracts.loc[tracts["underserved"], "coverage_pct"]
    if underserved_vals.empty:
        pocket_cutoff = metro_mean
    else:
        pocket_cutoff = float(underserved_vals.quantile(pocket_quantile))

    tracts["pocket"] = tracts["underserved"] & (tracts["coverage_pct"] <= pocket_cutoff)
    return tracts, metro_mean, pocket_cutoff


def plot_markets_pantries(
    out_path: Path,
    metro: str,
    tracts: gpd.GeoDataFrame,
    groceries: gpd.GeoDataFrame,
    proposed_pantries: gpd.GeoDataFrame,
    existing_pantries: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    basemap_img: np.ndarray | None,
    basemap_extent: tuple[float, float, float, float] | None,
    proposed_logo_path: Path,
    proposed_logo_zoom: float,
    proposed_logo_bg_tolerance: int,
    mode: str,
    minutes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)
    draw_basemap_and_extent(ax, boundary, basemap_img, basemap_extent)

    cmap = plt.colormaps["YlOrRd_r"]
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    facecolors = [
        (cmap(norm(v))[0], cmap(norm(v))[1], cmap(norm(v))[2], float(a))
        for v, a in zip(tracts["coverage_pct"], tracts["density_opacity"])
    ]
    tracts.plot(
        ax=ax,
        color=facecolors,
        linewidth=0.15,
        edgecolor="#f2efe8",
        zorder=1,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.8)
    cbar.set_label(f"Healthy-food coverage share ({mode}, {minutes} min)")

    boundary.boundary.plot(ax=ax, color="#2c2c2c", linewidth=1.0, zorder=2)

    if not groceries.empty:
        groceries.plot(
            ax=ax,
            markersize=9,
            marker="o",
            color="#1b9e77",
            alpha=0.9,
            edgecolor="#ffffff",
            linewidth=0.2,
            zorder=3,
        )
    used_logo = False
    legend_logo = None
    if not proposed_pantries.empty:
        used_logo = plot_icon_points(
            ax=ax,
            gdf=proposed_pantries,
            icon_path=proposed_logo_path,
            icon_bg_tolerance=proposed_logo_bg_tolerance,
            zoom=proposed_logo_zoom,
            zorder=4,
        )
        if used_logo:
            legend_logo = load_icon_array(proposed_logo_path, proposed_logo_bg_tolerance)
    if not used_logo and not proposed_pantries.empty:
        proposed_pantries.plot(
            ax=ax,
            markersize=20,
            marker="s",
            color="#2f4858",
            alpha=0.95,
            edgecolor="#1f2f3a",
            linewidth=0.35,
            zorder=4,
        )
    if not existing_pantries.empty:
        existing_pantries.plot(
            ax=ax,
            markersize=18,
            marker="o",
            color="#d7301f",
            alpha=0.95,
            edgecolor="#7f0000",
            linewidth=0.3,
            zorder=5,
        )

    ax.set_title(
        f"{metro.replace('_', ' ').title()}\nMarkets, Pantry Candidates, and Access Coverage",
        fontsize=14,
        pad=14,
    )
    ax.set_axis_off()
    # Freeze layout before legend/icon placement so display-coordinate anchoring is stable.
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1b9e77", markeredgecolor="#ffffff", markersize=7, label="Markets / groceries"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d7301f", markeredgecolor="#7f0000", markersize=7, label="Existing city pantries"),
        Patch(facecolor="#808080", alpha=0.25, edgecolor="none", label="Lower-population-density tracts are more transparent"),
    ]
    handles.insert(
        1,
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#2f4858",
            markeredgecolor="#1f2f3a",
            markersize=7,
            label="Target service sites",
        ),
    )
    legend = ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95)

    # Place the logo icon directly beside the "Target service sites" legend label.
    if used_logo and legend_logo is not None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = legend.get_texts()
        idx = next((i for i, t in enumerate(texts) if t.get_text() == "Target service sites"), None)
        if idx is not None:
            tbox = texts[idx].get_window_extent(renderer=renderer)
            # Position icon just left of legend text center (display px -> axes coords).
            x_disp = tbox.x0 - 14.0
            y_disp = (tbox.y0 + tbox.y1) / 2.0
            ab = AnnotationBbox(
                OffsetImage(legend_logo, zoom=0.06),
                (x_disp, y_disp),
                xycoords="figure pixels",
                frameon=False,
                box_alignment=(0.5, 0.5),
                annotation_clip=False,
                zorder=20,
            )
            fig.add_artist(ab)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pockets(
    out_path: Path,
    metro: str,
    tracts: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    basemap_img: np.ndarray | None,
    basemap_extent: tuple[float, float, float, float] | None,
    mode: str,
    minutes: int,
    metro_mean: float,
    pocket_cutoff: float,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)
    draw_basemap_and_extent(ax, boundary, basemap_img, basemap_extent)

    tracts.plot(
        ax=ax,
        color=rgba_colors("#d9d9d9", tracts["density_opacity"]),
        edgecolor="#ffffff",
        linewidth=0.1,
        zorder=1,
    )
    underserved = tracts[tracts["underserved"]]
    pockets = tracts[tracts["pocket"]]

    if not underserved.empty:
        underserved.plot(
            ax=ax,
            color=rgba_colors("#fdb863", underserved["density_opacity"]),
            edgecolor="#e08214",
            linewidth=0.2,
            zorder=2,
        )
    if not pockets.empty:
        pockets.plot(
            ax=ax,
            color=rgba_colors("#d7301f", pockets["density_opacity"]),
            edgecolor="#7f0000",
            linewidth=0.35,
            zorder=3,
        )
    # Opacity bar removed per request.

    boundary.boundary.plot(ax=ax, color="#2c2c2c", linewidth=1.0, zorder=4)

    top_n = pockets.sort_values("pop_without_access", ascending=False).head(8)
    for row in top_n.itertuples(index=False):
        if row.geometry is None or row.geometry.is_empty:
            continue
        pt = row.geometry.representative_point()
        if pt.is_empty:
            continue
        ax.text(pt.x, pt.y, str(int(row.pop_without_access)), fontsize=7, color="#111111", zorder=5)

    ax.set_title(
        (
            f"{metro.replace('_', ' ').title()}\n"
            f"Underserved Tracts (< metro avg {metro_mean:.0%}) and Deep-Need Pockets (<= {pocket_cutoff:.0%})"
        ),
        fontsize=14,
        pad=14,
    )
    ax.set_axis_off()

    handles = [
        Patch(facecolor="#d9d9d9", edgecolor="#ffffff", label="Other tracts"),
        Patch(facecolor="#fdb863", edgecolor="#e08214", label="Underserved tracts"),
        Patch(facecolor="#d7301f", edgecolor="#7f0000", label="Deep-need pockets"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_avg_walk_time(
    out_path: Path,
    metro: str,
    tracts: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    basemap_img: np.ndarray | None,
    basemap_extent: tuple[float, float, float, float] | None,
    mode: str,
    metro_avg_walk: float,
    long_walk_minutes: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 11), dpi=200)
    draw_basemap_and_extent(ax, boundary, basemap_img, basemap_extent)
    vmax = max(20.0, float(np.nanpercentile(tracts["avg_walk_min_est"], 95)))
    cmap = plt.colormaps["RdYlGn_r"]
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    facecolors = [
        (cmap(norm(v))[0], cmap(norm(v))[1], cmap(norm(v))[2], float(a))
        for v, a in zip(tracts["avg_walk_min_est"], tracts["density_opacity"])
    ]
    tracts.plot(ax=ax, color=facecolors, edgecolor="#f2efe8", linewidth=0.15, zorder=1)
    boundary.boundary.plot(ax=ax, color="#2c2c2c", linewidth=1.0, zorder=2)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.8)
    cbar.set_label(f"Estimated average transit_time to grocery ({mode}, minutes)")

    ax.set_title(
        (
            f"{metro.replace('_', ' ').title()}\n"
            f"Estimated Average transit_time ({mode}; 0-10, 10-15, 15-20, >20 bins; >20 set to {long_walk_minutes} min)\n"
            f"Metro population-weighted average: {metro_avg_walk:.1f} min"
        ),
        fontsize=13,
        pad=12,
    )
    ax.set_axis_off()
    ax.legend(
        handles=[Patch(facecolor="#808080", alpha=0.25, edgecolor="none", label="Lower-population-density tracts are more transparent")],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_pocket_summary(out_csv: Path, tracts: gpd.GeoDataFrame) -> None:
    pockets = tracts[tracts["pocket"]].copy()
    if pockets.empty:
        pd.DataFrame(
            columns=["GEOID", "POPULATION", "coverage_pct", "pop_without_access", "underserved", "pocket"]
        ).to_csv(out_csv, index=False)
        return
    cols = ["GEOID", "POPULATION", "coverage_pct", "pop_without_access", "underserved", "pocket"]
    pockets[cols].sort_values("pop_without_access", ascending=False).to_csv(out_csv, index=False)


def write_criteria_summary(
    out_txt: Path,
    mode: str,
    minutes: int,
    pocket_quantile: float,
    metro_mean: float,
    pocket_cutoff: float,
    metro_avg_walk: float,
    long_walk_minutes: int,
) -> None:
    lines = [
        "Selection Criteria Used In These Figures",
        "",
        f"1) Access metric: coverage_pct in tract_{mode}_{minutes}.geojson",
        f"   Definition: share of tract population with modeled access to healthy-food locations within {minutes} minutes ({mode}).",
        "",
        "2) Underserved tract definition:",
        f"   coverage_pct < metro population-weighted mean coverage_pct ({metro_mean:.4f} = {metro_mean:.1%}).",
        "",
        "3) Deep-need pocket definition:",
        f"   Underserved tract AND coverage_pct <= {pocket_quantile:.0%} quantile among underserved tracts.",
        f"   Threshold in this run: coverage_pct <= {pocket_cutoff:.4f} ({pocket_cutoff:.1%}).",
        "",
        "4) Population-priority metric for ranking pockets:",
        "   pop_without_access = POPULATION * (1 - coverage_pct).",
        "",
        "5) Density-based opacity adjustment:",
        "   Tracts are made more transparent at lower population density, so large outlying/low-density areas are visually de-emphasized.",
        "",
        "6) Average walking time map (estimated):",
        "   Uses cumulative tract walk coverage at 10/15/20 minutes:",
        "   avg_walk_min_est = 5*cov10 + 12.5*(cov15-cov10) + 17.5*(cov20-cov15) + L*(1-cov20),",
        f"   where L={long_walk_minutes} minutes for residents beyond 20-minute walk coverage.",
        f"   Metro population-weighted estimated average walk time: {metro_avg_walk:.2f} minutes.",
    ]
    out_txt.write_text("\\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate static maps for markets/pantries and low-access pockets.")
    parser.add_argument("--data-root", default="docs/data", help="Directory with metro folders and manifest.json")
    parser.add_argument("--out", default="reports/figures", help="Output directory for figures")
    parser.add_argument("--metros", nargs="+", required=True, help="Metro slugs in docs/data (e.g., boston_ma_nh)")
    parser.add_argument("--mode", choices=["walk", "walk_transit"], default="walk_transit")
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--pocket-quantile", type=float, default=0.25, help="Quantile within underserved tracts")
    parser.add_argument(
        "--long-walk-minutes",
        type=int,
        default=25,
        help="Imputed walk minutes for population beyond 20-minute walk coverage in average-walk map",
    )
    parser.add_argument("--no-basemap", action="store_true", help="Disable OSM street basemap under area layers")
    parser.add_argument("--basemap-zoom", type=int, default=None, help="Optional fixed OSM tile zoom (8-13 recommended)")
    parser.add_argument("--max-basemap-tiles", type=int, default=36, help="Maximum OSM tiles to fetch per metro")
    parser.add_argument("--tile-cache-dir", default="cache/tiles/osm", help="Directory to cache downloaded OSM tiles")
    parser.add_argument(
        "--existing-pantries-path",
        default="",
        help="Optional GeoJSON/CSV of existing city pantries. If empty, tries docs/data/<metro>/city_pantries.geojson",
    )
    parser.add_argument("--proposed-logo", default="docs/favicon_fsfn.png", help="Logo icon for your proposed pantry sites")
    parser.add_argument("--proposed-logo-zoom", type=float, default=0.055, help="Logo zoom for proposed pantry markers")
    parser.add_argument(
        "--proposed-logo-bg-tolerance",
        type=int,
        default=26,
        help="Color-distance threshold for removing logo background",
    )
    parser.add_argument(
        "--figure-format",
        choices=["png", "svg"],
        default="png",
        help="Output format for map figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    metros_manifest = load_manifest(data_root)

    for metro in args.metros:
        if metro not in metros_manifest:
            raise ValueError(f"Unknown metro `{metro}`. Check docs/data/manifest.json")

        metro_out = out_root / metro
        metro_out.mkdir(parents=True, exist_ok=True)

        data = prep_metro_data(data_root, metro, args.mode, args.minutes)
        tracts, metro_mean, pocket_cutoff = classify_pockets(data["tracts"], args.pocket_quantile)
        walk_tracts, metro_avg_walk = prep_walk_time_data(data_root, metro, args.mode, args.long_walk_minutes)
        proposed_logo_path = Path(args.proposed_logo)
        existing_pantries_path = Path(args.existing_pantries_path) if args.existing_pantries_path else (data_root / metro / "city_pantries.geojson")
        existing_pantries = load_optional_layer(existing_pantries_path)
        if not existing_pantries.empty:
            existing_pantries = existing_pantries.to_crs(PLOT_CRS)
        basemap_img, basemap_extent = build_street_basemap(
            data["boundary"],
            use_basemap=(not args.no_basemap),
            basemap_zoom=args.basemap_zoom,
            tile_cache_dir=Path(args.tile_cache_dir),
            max_tiles=args.max_basemap_tiles,
        )
        if basemap_img is None and not args.no_basemap:
            print(f"[WARN] Street basemap unavailable for {metro}; rendered without tiles.")
        if not proposed_logo_path.exists():
            print(f"[WARN] Proposed logo not found at {proposed_logo_path}; using square fallback markers.")
        if existing_pantries.empty:
            print(f"[WARN] Existing city pantries not found/empty at {existing_pantries_path}; skipping that layer.")

        fig1 = metro_out / f"map_01_markets_pantries_{args.mode}_{args.minutes}min.{args.figure_format}"
        fig2 = metro_out / f"map_02_underserved_pockets_{args.mode}_{args.minutes}min.{args.figure_format}"
        fig3 = metro_out / f"map_03_transit_time_estimated.{args.figure_format}"
        out_csv = metro_out / f"pockets_summary_{args.mode}_{args.minutes}min.csv"
        criteria_txt = metro_out / f"criteria_{args.mode}_{args.minutes}min.txt"

        plot_markets_pantries(
            fig1,
            metro,
            tracts,
            data["groceries"],
            data["proposed_pantries"],
            existing_pantries,
            data["boundary"],
            basemap_img,
            basemap_extent,
            proposed_logo_path,
            args.proposed_logo_zoom,
            args.proposed_logo_bg_tolerance,
            args.mode,
            args.minutes,
        )
        plot_pockets(
            fig2,
            metro,
            tracts,
            data["boundary"],
            basemap_img,
            basemap_extent,
            args.mode,
            args.minutes,
            metro_mean,
            pocket_cutoff,
        )
        plot_avg_walk_time(
            fig3,
            metro,
            walk_tracts,
            data["boundary"],
            basemap_img,
            basemap_extent,
            args.mode,
            metro_avg_walk,
            args.long_walk_minutes,
        )
        write_pocket_summary(out_csv, tracts)
        write_criteria_summary(
            criteria_txt,
            args.mode,
            args.minutes,
            args.pocket_quantile,
            metro_mean,
            pocket_cutoff,
            metro_avg_walk,
            args.long_walk_minutes,
        )

        print(f"[OK] Wrote {fig1}")
        print(f"[OK] Wrote {fig2}")
        print(f"[OK] Wrote {fig3}")
        print(f"[OK] Wrote {out_csv}")
        print(f"[OK] Wrote {criteria_txt}")


if __name__ == "__main__":
    main()
