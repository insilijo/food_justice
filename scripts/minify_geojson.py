#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple


def round_coords(value, decimals: int):
    if isinstance(value, list):
        return [round_coords(v, decimals) for v in value]
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, int):
        return value
    return value


def minify_geojson(
    path: Path,
    decimals: int,
    drop_crs: bool,
    drop_name: bool,
    drop_props: set[str],
) -> Tuple[int, int]:
    before = path.stat().st_size
    with path.open() as f:
        data = json.load(f)

    if drop_crs:
        data.pop("crs", None)
    if drop_name:
        data.pop("name", None)

    if isinstance(data, dict):
        if "bbox" in data:
            data["bbox"] = round_coords(data["bbox"], decimals)
        features = data.get("features")
        if isinstance(features, list):
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                props = feature.get("properties")
                if isinstance(props, dict) and drop_props:
                    for key in drop_props:
                        props.pop(key, None)
                if "bbox" in feature:
                    feature["bbox"] = round_coords(feature["bbox"], decimals)
                geom = feature.get("geometry")
                if isinstance(geom, dict) and "coordinates" in geom:
                    geom["coordinates"] = round_coords(geom["coordinates"], decimals)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, ensure_ascii=True, separators=(",", ":"))
    tmp_path.replace(path)
    after = path.stat().st_size
    return before, after


def find_targets(root: Path, min_bytes: int) -> Iterable[Path]:
    for path in root.rglob("*.geojson"):
        if path.stat().st_size >= min_bytes:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minify GeoJSON files by rounding coordinates and stripping metadata."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="GeoJSON files to process. If omitted, --root/--min-mb are used.",
    )
    parser.add_argument("--root", type=Path, default=Path("docs/data"))
    parser.add_argument("--min-mb", type=float, default=25.0)
    parser.add_argument("--decimals", type=int, default=5)
    parser.add_argument("--drop-crs", action="store_true")
    parser.add_argument("--drop-name", action="store_true")
    parser.add_argument(
        "--drop-props",
        default="",
        help="Comma-separated property keys to remove from features.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = [p for p in args.paths if p.exists()]
    if not paths:
        min_bytes = int(args.min_mb * 1024 * 1024)
        paths = list(find_targets(args.root, min_bytes))

    if not paths:
        print("No matching GeoJSON files found.")
        return 0

    drop_props = {p for p in args.drop_props.split(",") if p}
    for path in paths:
        if args.dry_run:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"DRY RUN: {path} {size_mb:.1f}MB")
            continue
        before, after = minify_geojson(
            path,
            decimals=args.decimals,
            drop_crs=args.drop_crs,
            drop_name=args.drop_name,
            drop_props=drop_props,
        )
        print(
            f"{path}: {before / (1024 * 1024):.1f}MB -> "
            f"{after / (1024 * 1024):.1f}MB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
