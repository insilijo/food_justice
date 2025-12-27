# Food Access Explorer (Static Leaflet)

## Build data (no counties required)
1) Put national boundaries in `data_inputs/`:
   - ZCTA (ZIP tabulation areas) with field `ZCTA5CE10`
   - Census tracts with field `GEOID` (11 digits)
2) Get a Census API key: https://api.census.gov/data/key_signup.html
3) Use the included metro list:
   - `metros.top20.json`
4) Run:
   python scripts/build_metro_areas.py --config metros.top20.json --out docs/data --year 2020 --census-key YOURKEY
5) Generate manifest:
   python scripts/generate_manifest.py

Deploy `docs/` with GitHub Pages.
