#!/usr/bin/env python3
"""Download grocery/convenience layers from the Boston food-access StoryMap web map.

Example:
  python3 scripts/download_storymap_food_retail.py \
    --out-dir docs/data/boston_ma_nh
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd

WEBMAP_ID = "585215fce1fe43c798191d2194ff0ee3"
WEBMAP_DATA_URL = f"https://www.arcgis.com/sharing/rest/content/items/{WEBMAP_ID}/data?f=pjson"
LAYER_PATTERNS = {
    "storymap_grocery_stores.geojson": ["grocery stores", "specialty grocers"],
    "storymap_convenience_stores.geojson": ["convenience stores"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download grocery and convenience layers from StoryMap web map")
    ap.add_argument("--out-dir", default="docs/data/boston_ma_nh")
    ap.add_argument("--metro", default="boston_ma_nh")
    ap.add_argument("--data-root", default="docs/data")
    return ap.parse_args()


def normalize_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    point_mask = gdf.geometry.geom_type.isin(["Point", "MultiPoint"])
    if point_mask.any():
        gdf = gdf.loc[point_mask].copy()
    return gdf


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(WEBMAP_DATA_URL, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    layers = payload.get("operationalLayers", [])
    if not layers:
        raise RuntimeError("No operational layers in web map")

    for out_name, patterns in LAYER_PATTERNS.items():
        keep: list[gpd.GeoDataFrame] = []
        for lyr in layers:
            title = str(lyr.get("title", "")).strip().lower()
            url = str(lyr.get("url", "")).strip()
            if not url:
                continue
            if not any(p in title for p in patterns):
                continue
            try:
                part = gpd.read_file(f"{url}/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson")
                part = normalize_points(part)
                if not part.empty:
                    keep.append(part)
            except Exception:
                continue

        if keep:
            merged = gpd.GeoDataFrame(pd.concat(keep, ignore_index=True), crs=keep[0].crs)
            merged["_x"] = merged.geometry.x.round(6)
            merged["_y"] = merged.geometry.y.round(6)
            merged = merged.drop_duplicates(subset=["_x", "_y"]).drop(columns=["_x", "_y"])
        else:
            merged = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        # clip to metro boundary if available
        boundary_path = Path(args.data_root) / args.metro / "metro_boundary.geojson"
        if boundary_path.exists() and not merged.empty:
            boundary = gpd.read_file(boundary_path).to_crs(merged.crs)
            merged = gpd.clip(merged, boundary)

        out_path = out_dir / out_name
        merged.to_file(out_path, driver="GeoJSON")
        print(f"[OK] Wrote {out_path} with {len(merged)} points")


if __name__ == "__main__":
    main()
