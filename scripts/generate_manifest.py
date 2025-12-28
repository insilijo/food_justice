
import json, re
from pathlib import Path
import geopandas as gpd

DATA_ROOT = Path("docs/data")
OUT = DATA_ROOT / "manifest.json"
PATTERN = re.compile(r"^(zip|tract|tract_subgrid|metro_grid)_(walk|walk_transit)_(\d+)\.geojson$")

def center_zoom(bounds):
    minx, miny, maxx, maxy = bounds
    center = [(miny+maxy)/2, (minx+maxx)/2]
    span = max(maxx-minx, maxy-miny)
    zoom = 12 if span < 0.08 else 11 if span < 0.15 else 10 if span < 0.30 else 9
    return center, zoom

manifest = {"metros": {}}

for metro in sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()]):
    boundary = metro / "metro_boundary.geojson"
    if not boundary.exists():
        continue
    g = gpd.read_file(boundary).to_crs(4326)
    center, zoom = center_zoom(g.total_bounds)

    geos, modes, mins = set(), set(), set()
    for f in metro.glob("*.geojson"):
        m = PATTERN.match(f.name)
        if not m:
            continue
        geos.add(m.group(1))
        modes.add(m.group(2))
        mins.add(int(m.group(3)))

    meta = {
        "name": metro.name.replace("_"," ").title(),
        "center": center,
        "zoom": zoom,
        "geographies": sorted(geos),
        "modes": sorted(modes),
        "minutes": sorted(mins),
        "has_boundary": True,
        "has_usda": (metro / "usda_fara_tracts.geojson").exists()
    }

    breaks_path = metro / "breaks.json"
    if breaks_path.exists():
        try:
            meta["breaks"] = json.loads(breaks_path.read_text())
        except Exception:
            pass

    manifest["metros"][metro.name] = meta

OUT.write_text(json.dumps(manifest, indent=2))
print("Wrote", OUT)
