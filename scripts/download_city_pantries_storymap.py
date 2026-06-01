#!/usr/bin/env python3
"""Download existing pantry points from an ArcGIS StoryMap.

Example:
  python3 scripts/download_city_pantries_storymap.py \
    --story-id 956debdf80c0492bbceeedff9f6a4bac \
    --metro boston_ma_nh \
    --out docs/data/boston_ma_nh/city_pantries.geojson
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

PANTRY_PATTERNS = [
    r"pantr", r"food\s*bank", r"food\s*pantry", r"distribution", r"meal\s*site", r"food\s*resource",
]
ITEM_ID_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")


def get_json(url: str, timeout: float = 30.0) -> dict:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def story_item_data(story_id: str) -> dict:
    url = f"https://www.arcgis.com/sharing/rest/content/items/{story_id}/data?f=pjson"
    return get_json(url)


def story_item_meta(item_id: str) -> dict:
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}?f=pjson"
    return get_json(url)


def story_item_resources(item_id: str) -> list[str]:
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/resources?f=pjson"
    payload = get_json(url)
    names: list[str] = []
    for res in payload.get("resources", []) or []:
        name = res.get("resource") or res.get("name")
        if name:
            names.append(str(name))
    return names


def story_resource_data(item_id: str, resource_name: str) -> dict:
    url = f"https://www.arcgis.com/sharing/rest/content/items/{item_id}/resources/{resource_name}?f=pjson"
    return get_json(url)


def collect_service_urls(obj, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            collect_service_urls(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_service_urls(v, out)
    elif isinstance(obj, str):
        if "FeatureServer" in obj or "MapServer" in obj:
            m = re.search(r"https?://[^\s\"]+/(?:FeatureServer|MapServer)(?:/\d+)?", obj)
            if m:
                out.add(m.group(0))


def collect_item_ids(obj, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            collect_item_ids(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_item_ids(v, out)
    elif isinstance(obj, str):
        for m in ITEM_ID_RE.findall(obj):
            out.add(m.lower())


def base_service_url(url: str) -> str:
    m = re.search(r"(https?://.+?/(?:FeatureServer|MapServer))", url)
    return m.group(1) if m else url


def layer_id_from_url(url: str) -> int | None:
    m = re.search(r"/(?:FeatureServer|MapServer)/(\d+)", url)
    return int(m.group(1)) if m else None


def iter_layers(service_url: str, hinted_layer: int | None):
    meta = get_json(f"{service_url}?f=pjson")
    layer_ids = []
    if hinted_layer is not None:
        layer_ids.append(hinted_layer)
    for k in ["layers", "tables"]:
        for lyr in meta.get(k, []) or []:
            lid = int(lyr.get("id"))
            if lid not in layer_ids:
                layer_ids.append(lid)
    for lid in layer_ids:
        try:
            lyr = get_json(f"{service_url}/{lid}?f=pjson")
            yield service_url, lid, lyr
        except Exception:
            continue


def is_pantry_layer(layer_meta: dict) -> bool:
    text_fields = [
        str(layer_meta.get("name", "")),
        str(layer_meta.get("description", "")),
        str(layer_meta.get("displayField", "")),
    ]
    text = " ".join(text_fields).lower()
    return any(re.search(p, text) for p in PANTRY_PATTERNS)


def fetch_layer_geojson(service_url: str, layer_id: int) -> gpd.GeoDataFrame:
    q = (
        f"{service_url}/{layer_id}/query"
        "?where=1%3D1"
        "&outFields=*"
        "&returnGeometry=true"
        "&outSR=4326"
        "&f=geojson"
    )
    gdf = gpd.read_file(q)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def keep_point_like(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    geoms = gdf.geometry
    point_mask = geoms.geom_type.isin(["Point", "MultiPoint"])
    if point_mask.any():
        gdf = gdf.loc[point_mask].copy()
    return gdf


def has_pantry_text(gdf: gpd.GeoDataFrame) -> pd.Series:
    if gdf.empty:
        return pd.Series(dtype=bool)
    text_cols = [c for c in gdf.columns if c != "geometry" and gdf[c].dtype == object]
    if not text_cols:
        return pd.Series([True] * len(gdf), index=gdf.index)
    s = gdf[text_cols].astype(str).fillna("").agg(" ".join, axis=1).str.lower()
    pat = "|".join(PANTRY_PATTERNS)
    return s.str.contains(pat, regex=True, na=False)


def maybe_clip_to_metro(gdf: gpd.GeoDataFrame, metro: str | None, data_root: Path) -> gpd.GeoDataFrame:
    if not metro or gdf.empty:
        return gdf
    boundary_path = data_root / metro / "metro_boundary.geojson"
    if not boundary_path.exists():
        return gdf
    boundary = gpd.read_file(boundary_path).to_crs(gdf.crs)
    return gpd.clip(gdf, boundary)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download city pantry points referenced by an ArcGIS StoryMap")
    ap.add_argument("--story-id", required=True, help="ArcGIS StoryMap item ID")
    ap.add_argument("--metro", default="", help="Optional metro slug for clipping, e.g. boston_ma_nh")
    ap.add_argument("--data-root", default="docs/data", help="Root containing metro folders")
    ap.add_argument("--out", required=True, help="Output GeoJSON path")
    ap.add_argument("--verbose", action="store_true", help="Print debug information while resolving story sources")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data = story_item_data(args.story_id)

    urls: set[str] = set()
    item_ids: set[str] = set()
    collect_service_urls(data, urls)
    collect_item_ids(data, item_ids)

    # StoryMaps often store source references in resources/*.json, not only in /data.
    try:
        for name in story_item_resources(args.story_id):
            if not name.lower().endswith(".json"):
                continue
            try:
                res_data = story_resource_data(args.story_id, name)
                collect_service_urls(res_data, urls)
                collect_item_ids(res_data, item_ids)
            except Exception:
                continue
    except Exception:
        pass

    # Resolve item IDs to services (handles webmaps/layers referenced by ID).
    seen_item_ids: set[str] = set()
    queue = list(sorted(item_ids))
    while queue:
        iid = queue.pop(0)
        if iid in seen_item_ids:
            continue
        seen_item_ids.add(iid)
        try:
            meta = story_item_meta(iid)
        except Exception:
            continue

        item_url = meta.get("url")
        if isinstance(item_url, str):
            collect_service_urls(item_url, urls)

        # Some item types embed layer sources in their data payload.
        try:
            item_data = get_json(f"https://www.arcgis.com/sharing/rest/content/items/{iid}/data?f=pjson")
            collect_service_urls(item_data, urls)
            new_ids: set[str] = set()
            collect_item_ids(item_data, new_ids)
            for nid in sorted(new_ids):
                if nid not in seen_item_ids:
                    queue.append(nid)
        except Exception:
            pass

    if args.verbose:
        print(f"[INFO] Resolved {len(urls)} service URL candidates from story and linked items")
        if urls:
            for u in sorted(urls)[:15]:
                print(f"[INFO] URL: {u}")

    if not urls:
        raise RuntimeError(
            "No FeatureServer/MapServer URLs found. Try --verbose; the story may use non-public or app-proxied layers."
        )

    frames: list[gpd.GeoDataFrame] = []
    for u in sorted(urls):
        service = base_service_url(u)
        hinted = layer_id_from_url(u)
        for svc, lid, meta in iter_layers(service, hinted):
            try:
                if is_pantry_layer(meta):
                    g = fetch_layer_geojson(svc, lid)
                    g = keep_point_like(g)
                    if not g.empty:
                        frames.append(g)
                    continue

                # Fallback: fetch and text-filter if layer name is ambiguous.
                g = fetch_layer_geojson(svc, lid)
                g = keep_point_like(g)
                if g.empty:
                    continue
                mask = has_pantry_text(g)
                g = g.loc[mask].copy()
                if not g.empty:
                    frames.append(g)
            except Exception:
                continue

    if not frames:
        raise RuntimeError("No pantry features found from story-linked services")

    out = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    out = out.to_crs("EPSG:4326")
    out = maybe_clip_to_metro(out, args.metro or None, Path(args.data_root))
    out = out[~out.geometry.is_empty & out.geometry.notna()].copy()

    # De-duplicate by rounded coordinates + optional name-like field.
    out["_x"] = out.geometry.x.round(6)
    out["_y"] = out.geometry.y.round(6)
    name_col = next((c for c in out.columns if c.lower() in {"name", "site", "pantry", "organization", "org_name"}), None)
    if name_col:
        out["_nm"] = out[name_col].astype(str).str.lower().str.strip()
    else:
        out["_nm"] = ""
    out = out.drop_duplicates(subset=["_x", "_y", "_nm"]).drop(columns=["_x", "_y", "_nm"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_file(out_path, driver="GeoJSON")
    print(f"[OK] Wrote {out_path} with {len(out)} points")


if __name__ == "__main__":
    main()
