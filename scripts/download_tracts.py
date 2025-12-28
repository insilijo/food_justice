#!/usr/bin/env python3
"""
Bootstrap TIGER/Line inputs for this project.

Downloads (TIGER2020):
- ZCTA5 national shapefile: tl_2020_us_zcta510.zip
- Census Tracts: state-based files tl_2020_<STATEFP>_tract.zip (you choose states)

Outputs:
- data_inputs/zcta_us.gpkg    (expects column ZCTA5CE10)
- data_inputs/tracts_us.gpkg  (expects column GEOID)

Run examples:
  # Boston metro (MA + NH):
  python scripts/bootstrap_data_inputs.py --states 25,33

  # Top-20 metros across many states (bigger):
  python scripts/bootstrap_data_inputs.py --states 06,12,13,17,25,26,27,33,34,36,37,42,48,53,08,32,04,09,24,51,11
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import time
import random
import requests
from requests.exceptions import HTTPError

ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = ROOT / "data_inputs"
RAW_DIR = DATA_INPUTS / "_raw"

TIGER2020_BASE = "https://www2.census.gov/geo/tiger/TIGER2020"

ZCTA_URL = f"{TIGER2020_BASE}/ZCTA5/tl_2020_us_zcta510.zip"
TRACT_URL_TEMPLATE = f"{TIGER2020_BASE}/TRACT/tl_2020_{{statefp}}_tract.zip"

ZCTA_ZIP = RAW_DIR / "tl_2020_us_zcta510.zip"
ZCTA_GPKG = DATA_INPUTS / "zcta_us.gpkg"

TRACTS_GPKG = DATA_INPUTS / "tracts_us.gpkg"

def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)

def have_ogr2ogr() -> bool:
    return shutil.which("ogr2ogr") is not None

def download(url: str, out_path: Path, force: bool = False) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        info(f"Already downloaded: {out_path.name}")
        return True

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if force and tmp_path.exists():
        tmp_path.unlink()

    tries = 8
    chunk_size = 1024 * 1024  # 1MB

    for attempt in range(1, tries + 1):
        try:
            info(f"Downloading ({attempt}/{tries}): {url}")

            # If partial file exists, try to resume
            headers = {}
            mode = "wb"
            downloaded = 0
            if tmp_path.exists():
                downloaded = tmp_path.stat().st_size
                headers["Range"] = f"bytes={downloaded}-"
                mode = "ab"
                info(f"Resuming at byte {downloaded:,}")

            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()

                # If server ignored Range request and returned full file, restart
                if "Range" in headers and r.status_code == 200:
                    info("Server ignored Range; restarting download")
                    tmp_path.unlink(missing_ok=True)
                    downloaded = 0
                    headers = {}
                    mode = "wb"
                    r.close()
                    # redo without Range
                    with requests.get(url, stream=True, timeout=60) as r2:
                        r2.raise_for_status()
                        total = r2.headers.get("Content-Length")
                        total = int(total) if total and total.isdigit() else 0
                        with open(tmp_path, mode) as f:
                            for chunk in r2.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                        if total and tmp_path.stat().st_size != total:
                            raise IOError(f"Size mismatch: got {tmp_path.stat().st_size} expected {total}")
                else:
                    # Range requests: Content-Length is remaining bytes, not full size
                    remaining = r.headers.get("Content-Length")
                    remaining = int(remaining) if remaining and remaining.isdigit() else 0

                    with open(tmp_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)

                    if remaining:
                        expected = downloaded + remaining
                        got = tmp_path.stat().st_size
                        if got != expected:
                            raise IOError(f"Size mismatch: got {got} expected {expected}")

            tmp_path.replace(out_path)
            info(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")
            return True
        except HTTPError as e:
            warn(f"HTTP error {e.response.status_code} while downloading {url}: {e}")            
            break
        except Exception as e:
            sleep_s = min(120.0, (2 ** (attempt - 1)) * 0.8 + random.random() * 0.5)
            warn(f"Failed to download {url}: {e} — sleeping {sleep_s:.1f}s and retrying")
            time.sleep(sleep_s)

    warn(f"Failed to download {url} after {tries} attempts")
    return False


def unzip(zip_path: Path, dest_dir: Path, force: bool = False) -> Path:
    out_dir = dest_dir / zip_path.stem
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # reuse if already unzipped
    if not force and list(out_dir.glob("*.shp")):
        info(f"Already unzipped: {out_dir}")
        return out_dir

    info(f"Unzipping: {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    return out_dir

def find_single_shp(folder: Path) -> Path:
    shps = list(folder.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No .shp found in {folder}")
    if len(shps) == 1:
        return shps[0]
    # prefer TIGER tl_*.shp if multiple
    for shp in shps:
        if shp.name.startswith("tl_"):
            return shp
    raise RuntimeError(f"Multiple .shp files found in {folder}: {[p.name for p in shps]}")

def ogr2ogr_write(gpkg: Path, shp: Path, layer: str, append: bool) -> None:
    cmd = ["ogr2ogr", "-f", "GPKG"]
    if append:
        cmd += ["-append"]
    cmd += [str(gpkg), str(shp), "-nln", layer]
    subprocess.check_call(cmd)

def convert_shp_to_gpkg(shp: Path, gpkg: Path, layer: str, append: bool = False) -> None:
    gpkg.parent.mkdir(parents=True, exist_ok=True)

    if have_ogr2ogr():
        ogr2ogr_write(gpkg, shp, layer=layer, append=append)
        return

    # geopandas fallback
    import geopandas as gpd
    gdf = gpd.read_file(shp)
    mode = "a" if append else "w"
    gdf.to_file(gpkg, layer=layer, driver="GPKG", mode=mode)

def validate_columns(gpkg_path: Path, required: list[str], layer: str | None = None) -> None:
    import geopandas as gpd
    info(f"Validating: {gpkg_path.name}")
    gdf = gpd.read_file(gpkg_path, layer=layer) if layer else gpd.read_file(gpkg_path)
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise RuntimeError(f"{gpkg_path.name} missing required columns: {missing}\nFound: {list(gdf.columns)}")
    info(f"OK: required columns present: {required}")

def parse_states(s: str) -> list[str]:
    vals = []
    
    if s == "all":
        vals = [str(i).zfill(2) for i in range(1, 79)]
    else:
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise ValueError(f"State FIPS must be numeric, got: {part}")
            vals.append(part.zfill(2))
    if not vals:
        raise ValueError("No states provided. Example: --states 25,33")
    return sorted(set(vals))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", help="Comma-separated 2-digit state FIPS (e.g. 25,33)", default="all")
    ap.add_argument("--force", action="store_true", help="Re-download/rebuild even if files exist")
    args = ap.parse_args()

    states = parse_states(args.states)
    force = args.force

    DATA_INPUTS.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- ZCTA5 (single national file) ---
    download(ZCTA_URL, ZCTA_ZIP, force=force)
    zcta_dir = unzip(ZCTA_ZIP, RAW_DIR, force=force)
    zcta_shp = find_single_shp(zcta_dir)

    if ZCTA_GPKG.exists() and force:
        ZCTA_GPKG.unlink()

    info(f"Converting ZCTA5 -> {ZCTA_GPKG}")
    convert_shp_to_gpkg(zcta_shp, ZCTA_GPKG, layer="zcta_us", append=False)
    validate_columns(ZCTA_GPKG, required=["ZCTA5CE10"], layer="zcta_us")

    # --- Tracts (state-based) ---
    if TRACTS_GPKG.exists() and force:
        TRACTS_GPKG.unlink()

    info(f"Downloading + merging tracts for states: {states}")
    first = True
    for st in states:
        url = TRACT_URL_TEMPLATE.format(statefp=st)
        zip_path = RAW_DIR / f"tl_2020_{st}_tract.zip"
        success = download(url, zip_path, force=force)
        
        if success:            
            st_dir = unzip(zip_path, RAW_DIR, force=force)
            shp = find_single_shp(st_dir)

            info(f"Appending tracts state={st} from {shp.name}")
            convert_shp_to_gpkg(shp, TRACTS_GPKG, layer="tracts_us", append=(not first))
            first = False

    validate_columns(TRACTS_GPKG, required=["GEOID"], layer="tracts_us")

    info("Bootstrap complete.")
    info(f"Created: {ZCTA_GPKG} (layer=zcta_us)")
    info(f"Created: {TRACTS_GPKG} (layer=tracts_us)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
