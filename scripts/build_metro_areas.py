#!/usr/bin/env python3
"""
Build metro food-access layers for static Leaflet app.

Run:
  python scripts/build_metro_areas.py --config metros.json --out docs/data --year 2020 --census-key YOURKEY

What it does per metro:
- Approx metro boundary = union of place polygons (OSM geocode)
- Walking network from OSMnx
- Grocery + transit points from OSM
- Walk isochrone union to groceries for each minutes threshold
- Walk+transit proxy = union(walk_to_groceries, 5-min walk to transit)
- Clip tract + ZCTA boundaries to metro
- Pull population via Census API (ACS 5-year B01003_001E)
  - Tracts: auto-derive counties from clipped tract GEOIDs, query once per county
  - ZCTAs: query once per ZCTA (only those appearing in metro clip)
- Export GeoJSON layers + CSV summaries
- Compute quantile breaks (33/66%) per layer -> breaks.json (for color normalization)
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import requests
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm.auto import tqdm

WALK_METERS_PER_MIN = 75  # ~4.5 km/h
GTFS_WALK_RADIUS_M = 400  # meters around stops to consider grocery access
GTFS_START_TIME = 8 * 3600
GTFS_END_TIME = 22 * 3600

# ---------------- Logging ----------------

def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

# ---------------- Network helpers ----------------

def add_travel_time(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add edge travel_time in minutes using edge length (meters)."""
    for _, _, _, data in G.edges(keys=True, data=True):
        length = float(data.get("length", 0.0))
        data["travel_time"] = length / WALK_METERS_PER_MIN
    return G

def ego_isochrone_union(
    G: nx.MultiDiGraph,
    center_nodes: list[int],
    minutes: int,
    desc: str,
    max_seeds: int = 800,
):
    """
    Fast-ish isochrone union: for each seed node, take ego graph within travel_time radius,
    build convex hull of node points, union all hulls.
    """
    import random
    if max_seeds and len(center_nodes) > max_seeds:
        center_nodes = random.sample(center_nodes, max_seeds)
        info(f"Capped seeds to {max_seeds}")

    if not center_nodes:
        return None

    polys = []
    for node in tqdm(center_nodes, desc=desc, unit="seed"):
        sg = nx.ego_graph(G, node, radius=float(minutes), distance="travel_time")
        pts = [(d["x"], d["y"]) for _, d in sg.nodes(data=True)]
        if len(pts) < 3:
            continue

        # Convex hull of node cloud (fast). Replace w/ alpha shape for nicer isochrones.
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        poly = gpd.GeoSeries(
            gpd.points_from_xy(xs, ys),
            crs=G.graph["crs"],
        ).union_all().convex_hull
        polys.append(poly)

    if not polys:
        return None
    return unary_union(polys)

# ---------------- OSM feature pulls ----------------

def get_osm_groceries(poly_4326):
    feats = ox.features_from_polygon(poly_4326, {"shop": ["supermarket", "grocery"]})
    return feats[feats.geometry.type == "Point"].copy()

def get_osm_transit(poly_4326):
    tags = {
        "highway": ["bus_stop"],
        "railway": ["station", "halt", "tram_stop", "subway_entrance"],
        "public_transport": ["platform", "station", "stop_position"],
    }
    feats = ox.features_from_polygon(poly_4326, tags)
    return feats[feats.geometry.type == "Point"].copy()

# ---------------- Census API ----------------

def census_get(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def parse_hhmmss_to_seconds(val: str) -> float | None:
    if not isinstance(val, str) or ":" not in val:
        return None
    try:
        parts = val.strip().split(":")
        if len(parts) != 3:
            return None
        h, m, s = [int(p) for p in parts]
        return float(h * 3600 + m * 60 + s)
    except Exception:
        return None

def build_gtfs_stop_graph(gtfs_dir: Path) -> tuple[nx.DiGraph, gpd.GeoDataFrame, dict] | tuple[None, None, None]:
    stops_path = gtfs_dir / "stops.txt"
    stop_times_path = gtfs_dir / "stop_times.txt"
    trips_path = gtfs_dir / "trips.txt"
    calendar_path = gtfs_dir / "calendar.txt"
    if not (stops_path.exists() and stop_times_path.exists() and trips_path.exists() and calendar_path.exists()):
        return None, None, None

    stops = pd.read_csv(stops_path, usecols=["stop_id", "stop_name", "stop_lat", "stop_lon"])
    trips = pd.read_csv(trips_path, usecols=["trip_id", "service_id", "route_id"])
    cal = pd.read_csv(calendar_path, usecols=["service_id", "monday"])
    weekday_services = set(cal.loc[cal["monday"] == 1, "service_id"].astype(str))
    trips = trips[trips["service_id"].astype(str).isin(weekday_services)]
    trip_ids = set(trips["trip_id"].astype(str))

    st = pd.read_csv(
        stop_times_path,
        usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
        dtype={"trip_id": str, "stop_id": str, "stop_sequence": int},
        low_memory=False,
    )
    st = st[st["trip_id"].astype(str).isin(trip_ids)]
    st["departure_sec"] = st["departure_time"].apply(parse_hhmmss_to_seconds)
    st["arrival_sec"] = st["arrival_time"].apply(parse_hhmmss_to_seconds)
    st = st.dropna(subset=["departure_sec", "arrival_sec"])
    st = st[(st["departure_sec"] >= GTFS_START_TIME) & (st["departure_sec"] <= GTFS_END_TIME)]

    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")
    st = st.sort_values(["trip_id", "stop_sequence"])
    st["next_stop_id"] = st.groupby("trip_id")["stop_id"].shift(-1)
    st["next_arrival_sec"] = st.groupby("trip_id")["arrival_sec"].shift(-1)
    st["route_id"] = st["route_id"].astype(str)
    st = st.dropna(subset=["next_stop_id", "next_arrival_sec"])
    st["travel_sec"] = st["next_arrival_sec"] - st["departure_sec"]
    st = st[st["travel_sec"] > 0]

    edges = (
        st.groupby(["stop_id", "next_stop_id", "route_id"])["travel_sec"]
        .mean()
        .reset_index()
    )

    Gt = nx.DiGraph()
    for row in edges.itertuples(index=False):
        u = (str(row.stop_id), str(row.route_id))
        v = (str(row.next_stop_id), str(row.route_id))
        Gt.add_edge(u, v, weight=float(row.travel_sec) / 60.0)

    stops_gdf = gpd.GeoDataFrame(
        stops,
        geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]),
        crs="EPSG:4326",
    )
    routes_by_stop = {}
    for row in edges.itertuples(index=False):
        routes_by_stop.setdefault(str(row.stop_id), set()).add(str(row.route_id))
        routes_by_stop.setdefault(str(row.next_stop_id), set()).add(str(row.route_id))
    return Gt, stops_gdf, routes_by_stop

def fetch_tract_population_by_county(year: int, state: str, county: str, api_key: str) -> pd.DataFrame:
    """
    ACS 5-year: B01003_001E total population for all tracts in a county.
    Returns GEOID (11-digit) + POPULATION.
    """
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get=B01003_001E&for=tract:*&in=state:{state}&in=county:{county}"
        f"&key={api_key}"
    )
    data = census_get(url)
    df = pd.DataFrame(data[1:], columns=data[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["POPULATION"] = df["B01003_001E"].astype(int)
    return df[["GEOID", "POPULATION"]]

def fetch_tract_commute_by_county(year: int, state: str, county: str, api_key: str) -> pd.DataFrame:
    """
    ACS 5-year: B08301 means of transportation to work.
    Returns GEOID + COMMUTERS_TOTAL + TRANSIT_COMMUTERS.
    """
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get=B08301_001E,B08301_010E&for=tract:*&in=state:{state}&in=county:{county}"
        f"&key={api_key}"
    )
    data = census_get(url)
    df = pd.DataFrame(data[1:], columns=data[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["COMMUTERS_TOTAL"] = df["B08301_001E"].astype(int)
    df["TRANSIT_COMMUTERS"] = df["B08301_010E"].astype(int)
    return df[["GEOID", "COMMUTERS_TOTAL", "TRANSIT_COMMUTERS"]]

def fetch_tract_vehicle_by_county(year: int, state: str, county: str, api_key: str) -> pd.DataFrame:
    """
    ACS 5-year: B25044 tenure by vehicles available.
    Returns GEOID + HOUSEHOLDS_TOTAL + ZERO_VEH_HH.
    """
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get=B25044_001E,B25044_003E,B25044_010E&for=tract:*&in=state:{state}&in=county:{county}"
        f"&key={api_key}"
    )
    data = census_get(url)
    df = pd.DataFrame(data[1:], columns=data[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["HOUSEHOLDS_TOTAL"] = df["B25044_001E"].astype(int)
    df["ZERO_VEH_HH"] = df["B25044_003E"].astype(int) + df["B25044_010E"].astype(int)
    return df[["GEOID", "HOUSEHOLDS_TOTAL", "ZERO_VEH_HH"]]

def fetch_tract_housing_by_county(year: int, state: str, county: str, api_key: str) -> pd.DataFrame:
    """
    ACS 5-year: median gross rent and median rooms.
    Returns GEOID + MEDIAN_GROSS_RENT + MEDIAN_ROOMS.
    """
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get=B25064_001E,B25019_001E&for=tract:*&in=state:{state}&in=county:{county}"
        f"&key={api_key}"
    )
    data = census_get(url)
    df = pd.DataFrame(data[1:], columns=data[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    rent = pd.to_numeric(df["B25064_001E"], errors="coerce")
    rooms = pd.to_numeric(df["B25019_001E"], errors="coerce")
    rent = rent.where(rent > 0)
    rooms = rooms.where(rooms > 0)
    df["MEDIAN_GROSS_RENT"] = rent.fillna(0).astype(int)
    df["MEDIAN_ROOMS"] = rooms.fillna(0)
    return df[["GEOID", "MEDIAN_GROSS_RENT", "MEDIAN_ROOMS"]]
def derive_unique_counties_from_tracts(tracts_gdf: gpd.GeoDataFrame, tract_id_col: str) -> list[tuple[str, str]]:
    geoid = tracts_gdf[tract_id_col].astype(str).str.zfill(11)
    states = geoid.str.slice(0, 2)
    counties = geoid.str.slice(2, 5)
    uniq = sorted(set(zip(states.tolist(), counties.tolist())))
    return [(s, c) for (s, c) in uniq if s.isdigit() and c.isdigit()]

def fetch_zcta_population(year: int, zctas: list[str], api_key: str, sleep: float = 0.03) -> pd.DataFrame:
    """
    ACS 5-year: B01003_001E total population for a list of ZCTAs.
    (API doesn't batch ZCTAs cleanly, so this loops only the needed ZCTAs.)
    """
    rows = []
    geo = "zip%20code%20tabulation%20area"
    zctas = [str(z).zfill(5) for z in set(zctas)]

    for z in tqdm(sorted(zctas), desc="ACS ZCTA population", unit="zcta"):
        url = (
            f"https://api.census.gov/data/{year}/acs/acs5"
            f"?get=B01003_001E&for={geo}:{z}"
            f"&key={api_key}"
        )
        try:
            data = census_get(url)
            if len(data) >= 2:
                rows.append((z, int(data[1][0])))
        except Exception:
            # Keep going; missing will be set to 0 later.
            pass
        time.sleep(sleep)

    return pd.DataFrame(rows, columns=["ZCTA5CE10", "POPULATION"])

# ---------------- Scoring + export ----------------

def compute_coverage(boundaries_gdf: gpd.GeoDataFrame, access_poly):
    g = boundaries_gdf.copy()
    if "_mask_geom" in g.columns:
        mask_geom = gpd.GeoSeries(g["_mask_geom"], crs=g.crs)
        g["area_total"] = mask_geom.area
        g["covered_area"] = 0.0 if access_poly is None else mask_geom.intersection(access_poly).area
    else:
        g["area_total"] = g.geometry.area
        g["covered_area"] = 0.0 if access_poly is None else g.geometry.intersection(access_poly).area
    g["coverage_pct"] = (g["covered_area"] / g["area_total"]).clip(0, 1).fillna(0)
    return g

def quantile_breaks(gdf: gpd.GeoDataFrame, col: str = "coverage_pct") -> list[float]:
    s = pd.to_numeric(gdf[col], errors="coerce")
    s = s[pd.notna(s)]
    if s.empty:
        return [0.0, 0.0]
    q33 = float(s.quantile(0.33))
    q66 = float(s.quantile(0.66))
    if not (pd.notna(q33) and pd.notna(q66)):
        return [0.0, 0.0]
    return [q33, q66]

def export_layer(
    gdf_proj: gpd.GeoDataFrame,
    out_geojson: Path,
    out_csv: Path,
    id_col: str,
    extra_cols: list[str] | None = None,
):
    out = gdf_proj.to_crs(4326).copy()
    keep = [id_col, "POPULATION", "coverage_pct", "pop_with_access"]
    if extra_cols:
        for c in extra_cols:
            if c in out.columns:
                keep.append(c)
    keep.append("geometry")
    if set(keep) - set(out.columns):
        missing = set(keep) - set(out.columns)
        warn(ValueError(f"Missing columns for export: {missing}"))
        return
    out = out[keep].rename(columns={id_col: "GEOID"})
    out["GEOID"] = out["GEOID"].astype(str)
    out.to_file(out_geojson, driver="GeoJSON")
    csv_cols = [c for c in ["GEOID", "POPULATION", "coverage_pct", "pop_with_access"] if c in out.columns]
    if extra_cols:
        csv_cols += [c for c in extra_cols if c in out.columns and c not in csv_cols]
    out[csv_cols].to_csv(out_csv, index=False)

def kde_smooth(gdf_proj: gpd.GeoDataFrame, value_col: str, bandwidth_m: float) -> np.ndarray:
    if bandwidth_m <= 0 or gdf_proj.empty:
        return gdf_proj[value_col].to_numpy()
    centroids = gdf_proj.geometry.centroid
    coords = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    values = gdf_proj[value_col].to_numpy()
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = diff[..., 0] ** 2 + diff[..., 1] ** 2
    bw2 = bandwidth_m ** 2
    weights = np.exp(-0.5 * d2 / bw2)
    wsum = weights.sum(axis=1)
    return (weights @ values) / wsum

def subdivide_gdf(
    gdf_proj: gpd.GeoDataFrame,
    id_col: str,
    cell_size_m: float,
    keep_square_geometry: bool = False,
) -> gpd.GeoDataFrame:
    if cell_size_m <= 0 or gdf_proj.empty:
        return gdf_proj
    rows = []
    for _, row in gdf_proj.iterrows():
        geom = row.geometry
        if geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if maxx - minx <= cell_size_m and maxy - miny <= cell_size_m:
            rows.append(row.to_dict())
            continue
        xs = np.arange(minx, maxx, cell_size_m)
        ys = np.arange(miny, maxy, cell_size_m)
        base_area = geom.area if geom.area > 0 else 0
        for x in xs:
            for y in ys:
                cell = box(x, y, x + cell_size_m, y + cell_size_m)
                if not cell.intersects(geom):
                    continue
                piece = cell.intersection(geom)
                if piece.is_empty:
                    continue
                piece_area = piece.area
                pop = row["POPULATION"]
                if base_area > 0:
                    pop = int(round(pop * (piece_area / base_area)))
                base = row.to_dict()
                base["POPULATION"] = pop
                if keep_square_geometry:
                    base["_mask_geom"] = piece
                    base["geometry"] = cell
                else:
                    base["geometry"] = piece
                rows.append(base)
    return gpd.GeoDataFrame(rows, crs=gdf_proj.crs)

def build_metro_grid(metro_geom, cell_size_m: float, crs) -> gpd.GeoDataFrame:
    if cell_size_m <= 0:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    minx, miny, maxx, maxy = metro_geom.bounds
    rows = []
    gid = 0
    xs = np.arange(minx, maxx, cell_size_m)
    ys = np.arange(miny, maxy, cell_size_m)
    for x in xs:
        for y in ys:
            cell = box(x, y, x + cell_size_m, y + cell_size_m)
            if not cell.intersects(metro_geom):
                continue
            inter = cell.intersection(metro_geom)
            if inter.is_empty:
                continue
            gid += 1
            rows.append({"GRID_ID": f"g{gid}", "_mask_geom": inter, "geometry": cell})
    return gpd.GeoDataFrame(rows, crs=crs)

def allocate_tract_attrs_to_grid(
    grid_gdf: gpd.GeoDataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    tract_id: str,
    columns: list[str],
) -> gpd.GeoDataFrame:
    if grid_gdf.empty:
        return grid_gdf
    missing = [c for c in columns if c not in tracts_gdf.columns]
    if missing:
        for c in columns:
            grid_gdf[c] = 0
        return grid_gdf

    grid_mask = grid_gdf.copy()
    if "_mask_geom" in grid_mask.columns:
        grid_mask["geometry"] = grid_mask["_mask_geom"]

    tract_area = tracts_gdf[[tract_id, "geometry"]].copy()
    tract_area["tract_area"] = tract_area.geometry.area
    inter = gpd.overlay(
        grid_mask[["GRID_ID", "geometry"]],
        tracts_gdf[[tract_id, *columns, "geometry"]],
        how="intersection",
    )
    if inter.empty:
        for c in columns:
            grid_gdf[c] = 0
        return grid_gdf

    inter["area"] = inter.geometry.area
    inter = inter.merge(tract_area[[tract_id, "tract_area"]], on=tract_id, how="left")
    for c in columns:
        inter[f"{c}_alloc"] = inter[c] * (inter["area"] / inter["tract_area"]).fillna(0)
    agg_cols = [f"{c}_alloc" for c in columns]
    agg = inter.groupby("GRID_ID", as_index=False)[agg_cols].sum()

    out = grid_gdf.merge(agg, on="GRID_ID", how="left")
    for c in columns:
        out[c] = out[f"{c}_alloc"].fillna(0).round().astype(int)
        out = out.drop(columns=[f"{c}_alloc"])
    return out

def allocate_tract_values_to_grid(
    grid_gdf: gpd.GeoDataFrame,
    tracts_gdf: gpd.GeoDataFrame,
    tract_id: str,
    columns: list[str],
) -> gpd.GeoDataFrame:
    if grid_gdf.empty:
        return grid_gdf
    missing = [c for c in columns if c not in tracts_gdf.columns]
    if missing:
        for c in columns:
            grid_gdf[c] = 0
        return grid_gdf

    grid_mask = grid_gdf.copy()
    if "_mask_geom" in grid_mask.columns:
        grid_mask["geometry"] = grid_mask["_mask_geom"]

    inter = gpd.overlay(
        grid_mask[["GRID_ID", "geometry"]],
        tracts_gdf[[tract_id, *columns, "geometry"]],
        how="intersection",
    )
    if inter.empty:
        for c in columns:
            grid_gdf[c] = 0
        return grid_gdf

    inter["area"] = inter.geometry.area
    for c in columns:
        inter[f"{c}_weighted"] = inter[c] * inter["area"]

    agg_cols = [f"{c}_weighted" for c in columns] + ["area"]
    agg = inter.groupby("GRID_ID", as_index=False)[agg_cols].sum()

    out = grid_gdf.merge(agg, on="GRID_ID", how="left")
    for c in columns:
        out[c] = (out[f"{c}_weighted"] / out["area"]).fillna(0)
        out = out.drop(columns=[f"{c}_weighted"])
    out = out.drop(columns=["area"])
    return out

def aggregate_subdivisions(gdf_sub: gpd.GeoDataFrame, gdf_orig: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    if id_col not in gdf_sub.columns:
        if gdf_sub.index.name == id_col:
            gdf_sub = gdf_sub.copy()
            gdf_sub[id_col] = gdf_sub.index
        else:
            raise KeyError(f"{id_col} missing from subgrid columns: {list(gdf_sub.columns)}")
    agg = gdf_sub.groupby(id_col, as_index=False)[["POPULATION", "pop_with_access"]].sum()
    agg["coverage_pct"] = (agg["pop_with_access"] / agg["POPULATION"]).fillna(0).clip(0, 1)
    out = gdf_orig.merge(agg, on=id_col, how="left", suffixes=("_orig", "_agg"))
    if "POPULATION" not in out.columns:
        if "POPULATION_orig" in out.columns:
            out["POPULATION"] = out["POPULATION_orig"]
        elif "POPULATION_agg" in out.columns:
            out["POPULATION"] = out["POPULATION_agg"]
    if "pop_with_access_agg" in out.columns:
        out["pop_with_access"] = out["pop_with_access_agg"]
    if "coverage_pct_agg" in out.columns:
        out["coverage_pct"] = out["coverage_pct_agg"]

    out["POPULATION"] = out.get("POPULATION", 0).fillna(0).astype(int)
    out["pop_with_access"] = out.get("pop_with_access", 0).fillna(0).astype(int)
    out["coverage_pct"] = out.get("coverage_pct", 0).fillna(0)

    drop_cols = [c for c in out.columns if c.endswith("_orig") or c.endswith("_agg")]
    out = out.drop(columns=drop_cols)
    return out

# ---------------- Metro boundary ----------------

def metro_boundary_from_places(place_names: list[str]) -> gpd.GeoDataFrame:
    info(f"Geocoding {len(place_names)} place polygons for metro boundary...")
    gdfs = [ox.geocode_to_gdf(p).to_crs(4326) for p in place_names]
    geom = unary_union([g.geometry.values[0] for g in gdfs])
    return gpd.GeoDataFrame({"name": ["metro_boundary"]}, geometry=[geom], crs="EPSG:4326")

def buffer_polygon_km(poly_4326, buffer_km: float):
    if not buffer_km or buffer_km <= 0:
        return poly_4326
    series = gpd.GeoSeries([poly_4326], crs="EPSG:4326").to_crs(3857)
    buffered = series.buffer(buffer_km * 1000).to_crs(4326).iloc[0]
    return buffered

# ---------------- Main build per metro ----------------

def build_one_metro(
    slug: str,
    meta: dict,
    out_root: str,
    year: int,
    api_key: str,
    minutes_list: list[int],
    max_seeds: int,
    osm_buffer_km: float,
    kde_smooth_enabled: bool,
    tract_subdivide_m: float,
    metro_grid_m: float,
) -> None:
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"=== Metro: {slug} ===")
    info(
        "Settings: "
        f"max_seeds={max_seeds}, "
        f"osm_buffer_km={osm_buffer_km}, "
        f"kde={kde_smooth_enabled}, "
        f"tract_subdivide_m={tract_subdivide_m}, "
        f"metro_grid_m={metro_grid_m}"
    )

    # 1) Metro boundary
    metro_gdf = metro_boundary_from_places(meta["places"])
    metro_poly_4326 = metro_gdf.geometry.values[0]
    metro_gdf.to_file(out_dir / "metro_boundary.geojson", driver="GeoJSON")
    info("Metro boundary saved.")

    # 2) Walking network
    buffer_km = float(osm_buffer_km or 0)
    osm_poly_4326 = buffer_polygon_km(metro_poly_4326, buffer_km)
    if buffer_km > 0:
        info(f"Using OSM buffer: +{buffer_km:.1f} km for network + POI pulls.")

    info("Downloading walking network from OSM...")
    G = ox.graph_from_polygon(osm_poly_4326, network_type="walk", simplify=True)
    G = ox.project_graph(G)
    G = add_travel_time(G)
    crs_proj = G.graph["crs"]
    info(f"Network loaded. Nodes={len(G.nodes):,} Edges={len(G.edges):,}")

    # 3) OSM points
    info("Fetching grocery + transit points from OSM...")
    groceries = get_osm_groceries(osm_poly_4326).to_crs(crs_proj)
    transit = get_osm_transit(osm_poly_4326).to_crs(crs_proj)
    info(f"Groceries: {len(groceries):,} points; Transit: {len(transit):,} points")

    groceries.to_crs(4326).to_file(out_dir / "groceries.geojson", driver="GeoJSON")
    stations = transit[
        (transit.get("railway") == "station") | (transit.get("public_transport") == "station")
    ].copy()
    stations.to_crs(4326).to_file(out_dir / "transit.geojson", driver="GeoJSON")

    # Vectorized nearest node lookup (much faster than Python loops)
    g_x = groceries.geometry.x.to_numpy()
    g_y = groceries.geometry.y.to_numpy()
    t_x = transit.geometry.x.to_numpy()
    t_y = transit.geometry.y.to_numpy()

    groceries_nodes = ox.distance.nearest_nodes(G, X=g_x, Y=g_y)
    transit_nodes   = ox.distance.nearest_nodes(G, X=t_x, Y=t_y)

    # OSMnx may return numpy arrays; normalize to plain lists
    try:
        groceries_nodes = list(groceries_nodes)
    except TypeError:
        groceries_nodes = [groceries_nodes]
    try:
        transit_nodes = list(transit_nodes)
    except TypeError:
        transit_nodes = [transit_nodes]


    # 4) Boundaries
    zcta_path = meta["boundaries"]["zcta_path"]
    zcta_id = meta["boundaries"]["zcta_id"]
    tract_path = meta["boundaries"]["tract_path"]
    tract_id = meta["boundaries"]["tract_id"]

    info("Loading and clipping boundaries to metro...")

    # Project metro polygon once; keep both CRS for bbox + spatial ops.
    metro_geom_4326 = metro_gdf.geometry.values[0]
    minx, miny, maxx, maxy = metro_geom_4326.bounds

    metro_proj = metro_gdf.to_crs(crs_proj)
    metro_geom = metro_proj.geometry.values[0]

    def load_prefilter(path: str, crs_proj, bbox_4326):
        """
        Read with bbox prefilter (bbox must be in the source CRS),
        then project.
        """
        bbox = bbox_4326
        src_crs = None
        try:
            import fiona
            with fiona.open(path) as src:
                src_crs = CRS.from_user_input(src.crs_wkt or src.crs)
        except Exception:
            src_crs = None

        if src_crs and src_crs != CRS.from_epsg(4326):
            try:
                transformer = Transformer.from_crs(4326, src_crs, always_xy=True)
                minx, miny = transformer.transform(bbox_4326[0], bbox_4326[1])
                maxx, maxy = transformer.transform(bbox_4326[2], bbox_4326[3])
                bbox = (minx, miny, maxx, maxy)
            except Exception:
                bbox = bbox_4326

        g = gpd.read_file(path, bbox=bbox)  # HUGE win vs reading whole file
        if g.empty:
            warn(f"{Path(path).name}: bbox prefilter returned 0 features; retrying without bbox.")
            g = gpd.read_file(path)
        return g.to_crs(crs_proj)

    # 1) Read only features near the metro (bbox filter at I/O time)
    zips_raw = load_prefilter(zcta_path, crs_proj, (minx, miny, maxx, maxy))
    tracts_raw = load_prefilter(tract_path, crs_proj, (minx, miny, maxx, maxy))
    if zips_raw.empty:
        warn("ZIP prefilter returned 0 features; check ZCTA CRS vs metro bbox CRS.")
    if tracts_raw.empty:
        warn("Tract prefilter returned 0 features; check tract CRS vs metro bbox CRS.")

    # 2) Keep only geometries that actually intersect metro (cheap filter)
    zips_raw = zips_raw[zips_raw.intersects(metro_geom)]
    tracts_raw = tracts_raw[tracts_raw.intersects(metro_geom)]

    # 3) Clip to metro polygon (cheaper than overlay)
    zips = gpd.clip(zips_raw, metro_geom)
    tracts = gpd.clip(tracts_raw, metro_geom)

    # Optional: clean invalid geometries (helps avoid GEOS errors later)
    zips["geometry"] = zips.geometry.buffer(0)
    tracts["geometry"] = tracts.geometry.buffer(0)

    info(f"Clipped: ZIPs={len(zips):,} Tracts={len(tracts):,}")


    # 5) Tract population via counties derived from GEOIDs
    info("Deriving counties from tract GEOIDs and fetching tract population (ACS)...")
    if tract_id not in tracts.columns:
        if {"STATEFP", "COUNTYFP", "TRACTCE"} <= set(tracts.columns):
            tracts[tract_id] = (
                tracts["STATEFP"].astype(str).str.zfill(2)
                + tracts["COUNTYFP"].astype(str).str.zfill(3)
                + tracts["TRACTCE"].astype(str).str.zfill(6)
            )
            warn(f"{slug}: derived {tract_id} from STATEFP/COUNTYFP/TRACTCE")
        else:
            raise ValueError(f"Missing {tract_id} and cannot derive from STATEFP/COUNTYFP/TRACTCE")

    tracts[tract_id] = tracts[tract_id].astype(str).str.zfill(11)
    uniq_counties = derive_unique_counties_from_tracts(tracts, tract_id)
    info(f"Unique counties in metro: {len(uniq_counties)}")

    tract_pops = []
    for (s, c) in tqdm(uniq_counties, desc=f"{slug}: ACS tract pop by county", unit="county"):
        tract_pops.append(fetch_tract_population_by_county(year, s, c, api_key))

    tract_pop = (
        pd.concat(tract_pops, ignore_index=True).drop_duplicates("GEOID")
        if tract_pops else pd.DataFrame(columns=["GEOID", "POPULATION"])
    )

    if tract_id == "GEOID":
        tracts = tracts.merge(tract_pop, on="GEOID", how="left")
    else:
        tracts = tracts.merge(tract_pop, left_on=tract_id, right_on="GEOID", how="left").drop(columns=["GEOID"])
    missing_tract = int(tracts["POPULATION"].isna().sum())
    if missing_tract:
        warn(f"{slug}: missing tract POPULATION for {missing_tract} features (set to 0)")
    tracts["POPULATION"] = tracts["POPULATION"].fillna(0).astype(int)

    info("Fetching tract commute, vehicle availability, and housing cost (ACS)...")
    commute_rows = []
    vehicle_rows = []
    housing_rows = []
    for (s, c) in tqdm(uniq_counties, desc=f"{slug}: ACS tract commute/vehicles", unit="county"):
        commute_rows.append(fetch_tract_commute_by_county(year, s, c, api_key))
        vehicle_rows.append(fetch_tract_vehicle_by_county(year, s, c, api_key))
        housing_rows.append(fetch_tract_housing_by_county(year, s, c, api_key))

    commute_df = (
        pd.concat(commute_rows, ignore_index=True).drop_duplicates("GEOID")
        if commute_rows else pd.DataFrame(columns=["GEOID", "COMMUTERS_TOTAL", "TRANSIT_COMMUTERS"])
    )
    vehicle_df = (
        pd.concat(vehicle_rows, ignore_index=True).drop_duplicates("GEOID")
        if vehicle_rows else pd.DataFrame(columns=["GEOID", "HOUSEHOLDS_TOTAL", "ZERO_VEH_HH"])
    )
    housing_df = (
        pd.concat(housing_rows, ignore_index=True).drop_duplicates("GEOID")
        if housing_rows else pd.DataFrame(columns=["GEOID", "MEDIAN_GROSS_RENT", "MEDIAN_ROOMS"])
    )

    tracts = tracts.merge(commute_df, on="GEOID", how="left")
    tracts = tracts.merge(vehicle_df, on="GEOID", how="left")
    tracts = tracts.merge(housing_df, on="GEOID", how="left")
    for col in ["COMMUTERS_TOTAL", "TRANSIT_COMMUTERS", "HOUSEHOLDS_TOTAL", "ZERO_VEH_HH"]:
        if col in tracts.columns:
            tracts[col] = tracts[col].fillna(0).astype(int)
    if "MEDIAN_GROSS_RENT" in tracts.columns:
        tracts["MEDIAN_GROSS_RENT"] = tracts["MEDIAN_GROSS_RENT"].fillna(0).astype(int)
    if "MEDIAN_ROOMS" in tracts.columns:
        tracts["MEDIAN_ROOMS"] = tracts["MEDIAN_ROOMS"].fillna(0)
        tracts["MEDIAN_RENT_PER_ROOM"] = (
            tracts["MEDIAN_GROSS_RENT"] / tracts["MEDIAN_ROOMS"].replace(0, np.nan)
        ).fillna(0)

    # 6) ZCTA population
    info("Fetching ZIP(ZCTA) population (ACS) for metro ZCTAs...")
    zips[zcta_id] = zips[zcta_id].astype(str).str.zfill(5)
    zcta_list = zips[zcta_id].unique().tolist()
    zcta_pop = fetch_zcta_population(year, zcta_list, api_key)

    if zcta_id == "ZCTA5CE10":
        zips = zips.merge(zcta_pop, on="ZCTA5CE10", how="left")
    else:
        zips = zips.merge(zcta_pop, left_on=zcta_id, right_on="ZCTA5CE10", how="left")
        zips = zips.drop(columns=["ZCTA5CE10"])

    missing_zip = int(zips["POPULATION"].isna().sum())
    if missing_zip:
        warn(f"{slug}: missing ZIP POPULATION for {missing_zip} features (set to 0)")
    zips["POPULATION"] = zips["POPULATION"].fillna(0).astype(int)

    tracts_for_scoring = tracts
    if tract_subdivide_m > 0:
        info(f"Subdividing tracts: cell size {tract_subdivide_m:.0f}m")
        tracts_for_scoring = subdivide_gdf(
            tracts,
            tract_id,
            tract_subdivide_m,
            keep_square_geometry=False,
        )
        info(f"Subgrid cells: {len(tracts_for_scoring):,}")

    metro_grid = gpd.GeoDataFrame()
    if metro_grid_m > 0:
        info(f"Building metro grid: cell size {metro_grid_m:.0f}m")
        metro_grid = build_metro_grid(metro_geom, metro_grid_m, crs_proj)
        metro_grid = allocate_tract_attrs_to_grid(
            metro_grid,
            tracts,
            tract_id,
            ["POPULATION", "COMMUTERS_TOTAL", "TRANSIT_COMMUTERS", "HOUSEHOLDS_TOTAL", "ZERO_VEH_HH"],
        )
        metro_grid = allocate_tract_values_to_grid(
            metro_grid,
            tracts,
            tract_id,
            ["MEDIAN_RENT_PER_ROOM"],
        )
        if "COMMUTERS_TOTAL" in metro_grid.columns:
            metro_grid["transit_commuter_share"] = (
                metro_grid["TRANSIT_COMMUTERS"] / metro_grid["COMMUTERS_TOTAL"]
            ).fillna(0).clip(0, 1)
        if "HOUSEHOLDS_TOTAL" in metro_grid.columns:
            metro_grid["zero_vehicle_share"] = (
                metro_grid["ZERO_VEH_HH"] / metro_grid["HOUSEHOLDS_TOTAL"]
            ).fillna(0).clip(0, 1)
        if len(metro_grid):
            info("Computing min travel time to groceries/transit for metro grid...")
            grocery_lengths = nx.multi_source_dijkstra_path_length(
                G,
                groceries_nodes,
                weight="travel_time",
            ) if groceries_nodes else {}
            transit_lengths = nx.multi_source_dijkstra_path_length(
                G,
                transit_nodes,
                weight="travel_time",
            ) if transit_nodes else {}

            centroids = metro_grid.geometry.centroid
            grid_nodes = ox.distance.nearest_nodes(G, X=centroids.x.to_numpy(), Y=centroids.y.to_numpy())
            try:
                grid_nodes = list(grid_nodes)
            except TypeError:
                grid_nodes = [grid_nodes]

            def min_two(a, b):
                if a is None and b is None:
                    return np.nan
                if a is None:
                    return b
                if b is None:
                    return a
                return a if a < b else b

            min_grocery = []
            min_transit = []
            min_access = []
            for n in grid_nodes:
                g = grocery_lengths.get(n)
                t = transit_lengths.get(n)
                min_grocery.append(np.nan if g is None else float(g))
                min_transit.append(np.nan if t is None else float(t))
                m = min_two(g, t)
                min_access.append(np.nan if m is None else float(m))

            metro_grid["min_grocery_minutes"] = min_grocery
            metro_grid["min_transit_minutes"] = min_transit
            metro_grid["min_access_minutes"] = min_access
        gtfs_dir = Path(str(meta.get("gtfs_path", ""))) if meta.get("gtfs_path") else None
        if gtfs_dir and gtfs_dir.exists():
            info("Computing simplified GTFS access (avg in-vehicle + transfer penalties)...")
            Gt, stops_gdf, routes_by_stop = build_gtfs_stop_graph(gtfs_dir)
            if Gt is not None and stops_gdf is not None and len(stops_gdf) and len(groceries):
                stops_proj = stops_gdf.to_crs(crs_proj)
                stops_proj["stop_id"] = stops_proj["stop_id"].astype(str)
                grocery_pts = groceries.geometry.reset_index(drop=True)
                sidx = grocery_pts.sindex

                def first_index(res):
                    if isinstance(res, tuple):
                        res = res[0]
                    if not isinstance(res, (list, tuple, np.ndarray)):
                        res = list(res)
                    arr = np.asarray(res).ravel()
                    return int(arr[0]) if arr.size else None

                def nearest_grocery_index(geom):
                    return first_index(sidx.nearest(geom, 1))

                def walk_to_grocery(geom):
                    idx = nearest_grocery_index(geom)
                    if idx is None:
                        return np.nan
                    nearest = grocery_pts.iloc[idx]
                    return float(geom.distance(nearest)) / WALK_METERS_PER_MIN

                stops_proj["walk_to_grocery_min"] = stops_proj.geometry.apply(walk_to_grocery)
                grocery_stop_ids = stops_proj.loc[
                    stops_proj["walk_to_grocery_min"] <= (GTFS_WALK_RADIUS_M / WALK_METERS_PER_MIN),
                    "stop_id",
                ].tolist()
                if grocery_stop_ids and routes_by_stop:
                    Gt_rev = Gt.reverse(copy=True)
                    for stop_id, routes in routes_by_stop.items():
                        routes = list(routes)
                        if len(routes) > 1:
                            for r1 in routes:
                                for r2 in routes:
                                    if r1 == r2:
                                        continue
                                    Gt_rev.add_edge((stop_id, r1), (stop_id, r2), weight=10.0)
                    Gt_rev.add_node("__GROCERY__")
                    grocery_walk = (
                        stops_proj.set_index("stop_id")["walk_to_grocery_min"].to_dict()
                    )
                    for sid in grocery_stop_ids:
                        for rid in routes_by_stop.get(str(sid), []):
                            w = float(grocery_walk.get(str(sid), np.nan))
                            if not np.isnan(w):
                                Gt_rev.add_edge("__GROCERY__", (str(sid), str(rid)), weight=w)

                    gtfs_lengths = nx.single_source_dijkstra_path_length(
                        Gt_rev,
                        "__GROCERY__",
                        weight="weight",
                    )
                else:
                    gtfs_lengths = {}

                stop_points = stops_proj.geometry
                stop_index = stop_points.sindex
                centroids = metro_grid.geometry.centroid

                def nearest_stop_index(geom):
                    return first_index(stop_index.nearest(geom, 1))

                gtfs_min = []
                for pt in centroids:
                    idx = nearest_stop_index(pt)
                    if idx is None:
                        gtfs_min.append(np.nan)
                        continue
                    stop_geom = stops_proj.iloc[idx].geometry
                    stop_id = str(stops_proj.iloc[idx]["stop_id"])
                    walk_to_stop = pt.distance(stop_geom) / WALK_METERS_PER_MIN
                    best = np.nan
                    for rid in routes_by_stop.get(stop_id, []):
                        val = gtfs_lengths.get((stop_id, str(rid)))
                        if val is None:
                            continue
                        if np.isnan(best) or val < best:
                            best = float(val)
                    if np.isnan(best):
                        gtfs_min.append(np.nan)
                    else:
                        gtfs_min.append(float(best) + float(walk_to_stop))

                metro_grid["min_gtfs_minutes"] = gtfs_min
                metro_grid["min_access_minutes"] = np.nanmin(
                    np.vstack([
                        np.array(metro_grid["min_grocery_minutes"], dtype=float),
                        np.array(metro_grid["min_gtfs_minutes"], dtype=float),
                    ]),
                    axis=0,
                )
        info(f"Metro grid cells: {len(metro_grid):,}")

    # 7) Transit proxy access (5-min to transit)
    info("Computing transit proxy access (5-min walk to transit)...")
    transit_access_5 = ego_isochrone_union(
        G,
        transit_nodes,
        minutes=5,
        desc="Isochrone 5 min (transit proxy)",
        max_seeds=max_seeds,
    )

    # 8) Compute and export layers + breaks
    geo_layers = [("zip", zips, zcta_id, False, None)]
    if tracts_for_scoring is not tracts:
        geo_layers.append(("tract_subgrid", tracts_for_scoring, tract_id, False, None))
    if not metro_grid.empty:
        geo_layers.append((
            "metro_grid",
            metro_grid,
            "GRID_ID",
            False,
            [
                "COMMUTERS_TOTAL",
                "TRANSIT_COMMUTERS",
                "HOUSEHOLDS_TOTAL",
                "ZERO_VEH_HH",
                "transit_commuter_share",
                "zero_vehicle_share",
                "MEDIAN_RENT_PER_ROOM",
                "min_grocery_minutes",
                "min_transit_minutes",
                "min_access_minutes",
                "min_gtfs_minutes",
            ],
        ))
    geo_layers.append(("tract", tracts_for_scoring, tract_id, tracts_for_scoring is not tracts, None))

    breaks = {g: {"walk": {}, "walk_transit": {}} for g, _, _, _, _ in geo_layers}

    for mins in minutes_list:
        info(f"Computing walk access isochrone union (minutes={mins})...")
        walk_access = ego_isochrone_union(
            G,
            groceries_nodes,
            minutes=mins,
            desc=f"Isochrone {mins} min (groceries)",
            max_seeds=max_seeds,
        )

        info(f"Computing walk+transit proxy union (minutes={mins})...")
        walk_transit_access = (
            unary_union([p for p in [walk_access, transit_access_5] if p is not None])
            if (walk_access is not None or transit_access_5 is not None)
            else None
        )

        for geo_name, gdf, id_col, aggregate, extra_cols in geo_layers:
            info(f"Scoring coverage: geo={geo_name} mode=walk mins={mins} ...")
            g_walk = compute_coverage(gdf, walk_access)
            if kde_smooth_enabled:
                bandwidth_m = mins * WALK_METERS_PER_MIN
                g_walk["coverage_pct"] = kde_smooth(g_walk, "coverage_pct", bandwidth_m)
            g_walk["pop_with_access"] = (g_walk["POPULATION"] * g_walk["coverage_pct"]).round().astype(int)

            out_walk = g_walk
            if aggregate:
                out_walk = aggregate_subdivisions(g_walk, tracts, tract_id)

            export_layer(
                out_walk,
                out_dir / f"{geo_name}_walk_{mins}.geojson",
                out_dir / (f"{geo_name}_summary.csv" if mins == minutes_list[0] else f"{geo_name}_summary_{mins}.csv"),
                id_col=id_col,
                extra_cols=extra_cols,
            )
            breaks[geo_name]["walk"][str(mins)] = quantile_breaks(out_walk)

            info(f"Scoring coverage: geo={geo_name} mode=walk_transit mins={mins} ...")
            g_wt = compute_coverage(gdf, walk_transit_access)
            if kde_smooth_enabled:
                bandwidth_m = mins * WALK_METERS_PER_MIN
                g_wt["coverage_pct"] = kde_smooth(g_wt, "coverage_pct", bandwidth_m)
            g_wt["pop_with_access"] = (g_wt["POPULATION"] * g_wt["coverage_pct"]).round().astype(int)

            out_wt = g_wt
            if aggregate:
                out_wt = aggregate_subdivisions(g_wt, tracts, tract_id)

            export_layer(
                out_wt,
                out_dir / f"{geo_name}_walk_transit_{mins}.geojson",
                out_dir / (f"{geo_name}_walk_transit_summary.csv" if mins == minutes_list[0] else f"{geo_name}_walk_transit_summary_{mins}.csv"),
                id_col=id_col,
                extra_cols=extra_cols,
            )
            breaks[geo_name]["walk_transit"][str(mins)] = quantile_breaks(out_wt)

    (out_dir / "breaks.json").write_text(json.dumps(breaks, indent=2))
    info(f"Finished metro: {slug}")

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to metros.json")
    ap.add_argument("--out", default="docs/data", help="Output directory (docs/data)")
    ap.add_argument("--year", type=int, default=2020, help="ACS 5-year year (e.g., 2020)")
    ap.add_argument("--census-key", required=True, help="Census API key")
    ap.add_argument("--minutes", default="10,15,20", help="Comma-separated minutes thresholds")
    ap.add_argument("--max-seeds", type=int, default=800, help="Max seed nodes for isochrone unions (0 = no cap)")
    ap.add_argument("--osm-buffer-km", type=float, default=None, help="Expand OSM network + POI queries by km beyond metro boundary")
    ap.add_argument("--kde", action="store_true", help="KDE smooth coverage using minutes as bandwidth (meters)")
    ap.add_argument("--tract-subdivide-m", type=float, default=0, help="Subdivide tracts into grid cells (meters)")
    ap.add_argument("--metro-grid-m", type=float, default=0, help="Build a metro-wide grid (meters)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    minutes_list = [int(x.strip()) for x in args.minutes.split(",") if x.strip()]

    for slug, meta in cfg["metros"].items():
        osm_buffer_km = (
            float(args.osm_buffer_km)
            if args.osm_buffer_km is not None
            else float(meta.get("osm_buffer_km", 0) or 0)
        )
        kde_enabled = bool(meta.get("kde", args.kde))
        tract_subdivide_m = float(meta.get("tract_subdivide_m", args.tract_subdivide_m) or 0)
        max_seeds = int(meta.get("max_seeds", args.max_seeds))
        metro_grid_m = float(meta.get("metro_grid_m", args.metro_grid_m) or 0)

        build_one_metro(
            slug,
            meta,
            args.out,
            args.year,
            args.census_key,
            minutes_list,
            max_seeds,
            osm_buffer_km,
            kde_enabled,
            tract_subdivide_m,
            metro_grid_m,
        )

if __name__ == "__main__":
    main()
