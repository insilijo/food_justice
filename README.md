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

## Data & hosting
Large data (`docs/data/`, `data_inputs/`, `data_cache/`, GTFS) is **not stored
in git** — it exceeded the Git LFS quota. It is regenerated locally with the
build step above and hosted on **Cloudflare R2** for production. To publish:

```
DATA_BASE_URL=<r2-public-url> UPDATE_DATA=1 scripts/deploy_pages_r2.sh
```

This uploads the local `docs/data/` to R2, injects the R2 URL into the
`data-base-url` meta tag, and deploys `docs/` to Cloudflare Pages. (The repo no
longer hosts on GitHub Pages.)
