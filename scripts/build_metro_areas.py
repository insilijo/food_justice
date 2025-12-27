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
from pyproj import CRS, Transformer
from shapely.ops import unary_union
from tqdm.auto import tqdm

WALK_METERS_PER_MIN = 75  # ~4.5 km/h

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

def export_layer(gdf_proj: gpd.GeoDataFrame, out_geojson: Path, out_csv: Path, id_col: str):
    out = gdf_proj.to_crs(4326).copy()
    keep = [id_col, "POPULATION", "coverage_pct", "pop_with_access", "geometry"]
    if set(keep) - set(out.columns):
        missing = set(keep) - set(out.columns)
        warn(ValueError(f"Missing columns for export: {missing}"))
        return
    out = out[keep].rename(columns={id_col: "GEOID"})
    out["GEOID"] = out["GEOID"].astype(str)
    out.to_file(out_geojson, driver="GeoJSON")
    out[["GEOID", "POPULATION", "coverage_pct", "pop_with_access"]].to_csv(out_csv, index=False)

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
) -> None:
    out_dir = Path(out_root) / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"=== Metro: {slug} ===")

    # 1) Metro boundary
    metro_gdf = metro_boundary_from_places(meta["places"])
    metro_poly_4326 = metro_gdf.geometry.values[0]
    metro_gdf.to_file(out_dir / "metro_boundary.geojson", driver="GeoJSON")
    info("Metro boundary saved.")

    # 2) Walking network
    osm_buffer_km = float(meta.get("osm_buffer_km", 0) or 0)
    osm_poly_4326 = buffer_polygon_km(metro_poly_4326, osm_buffer_km)
    if osm_buffer_km > 0:
        info(f"Using OSM buffer: +{osm_buffer_km:.1f} km for network + POI pulls.")

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

    tracts = tracts.merge(tract_pop, left_on=tract_id, right_on="GEOID", how="left").drop(columns=["GEOID"])
    missing_tract = int(tracts["POPULATION"].isna().sum())
    if missing_tract:
        warn(f"{slug}: missing tract POPULATION for {missing_tract} features (set to 0)")
    tracts["POPULATION"] = tracts["POPULATION"].fillna(0).astype(int)

    # 6) ZCTA population
    info("Fetching ZIP(ZCTA) population (ACS) for metro ZCTAs...")
    zips[zcta_id] = zips[zcta_id].astype(str).str.zfill(5)
    zcta_list = zips[zcta_id].unique().tolist()
    zcta_pop = fetch_zcta_population(year, zcta_list, api_key)

    zips = zips.merge(zcta_pop, left_on=zcta_id, right_on="ZCTA5CE10", how="left")
    if zcta_id != "ZCTA5CE10":
        zips = zips.drop(columns=["ZCTA5CE10"])

    missing_zip = int(zips["POPULATION"].isna().sum())
    if missing_zip:
        warn(f"{slug}: missing ZIP POPULATION for {missing_zip} features (set to 0)")
    zips["POPULATION"] = zips["POPULATION"].fillna(0).astype(int)

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
    breaks = {"zip": {"walk": {}, "walk_transit": {}}, "tract": {"walk": {}, "walk_transit": {}}}

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

        for geo_name, gdf, id_col in [("zip", zips, zcta_id), ("tract", tracts, tract_id)]:
            info(f"Scoring coverage: geo={geo_name} mode=walk mins={mins} ...")
            g_walk = compute_coverage(gdf, walk_access)
            g_walk["pop_with_access"] = (g_walk["POPULATION"] * g_walk["coverage_pct"]).round().astype(int)

            export_layer(
                g_walk,
                out_dir / f"{geo_name}_walk_{mins}.geojson",
                out_dir / (f"{geo_name}_summary.csv" if mins == minutes_list[0] else f"{geo_name}_summary_{mins}.csv"),
                id_col=id_col
            )
            breaks[geo_name]["walk"][str(mins)] = quantile_breaks(g_walk)

            info(f"Scoring coverage: geo={geo_name} mode=walk_transit mins={mins} ...")
            g_wt = compute_coverage(gdf, walk_transit_access)
            g_wt["pop_with_access"] = (g_wt["POPULATION"] * g_wt["coverage_pct"]).round().astype(int)

            export_layer(
                g_wt,
                out_dir / f"{geo_name}_walk_transit_{mins}.geojson",
                out_dir / (f"{geo_name}_walk_transit_summary.csv" if mins == minutes_list[0] else f"{geo_name}_walk_transit_summary_{mins}.csv"),
                id_col=id_col
            )
            breaks[geo_name]["walk_transit"][str(mins)] = quantile_breaks(g_wt)

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
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    minutes_list = [int(x.strip()) for x in args.minutes.split(",") if x.strip()]

    for slug, meta in cfg["metros"].items():
        build_one_metro(slug, meta, args.out, args.year, args.census_key, minutes_list, args.max_seeds)

if __name__ == "__main__":
    main()
