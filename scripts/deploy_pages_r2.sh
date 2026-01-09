#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="${PROJECT_NAME:-fresh-start-food-network}"
BUCKET_NAME="${BUCKET_NAME:-food-justice}"
DATA_BASE_URL="${DATA_BASE_URL:-}"
DOCS_DIR="${DOCS_DIR:-docs}"
DATA_DIR="${DOCS_DIR}/data"
STASH_DIR="${STASH_DIR:-data_cache/r2-data}"
CREATE_BUCKET="${CREATE_BUCKET:-0}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data directory: $DATA_DIR" >&2
  exit 1
fi

if [[ -e "$STASH_DIR" ]] && [[ -n "$(ls -A "$STASH_DIR" 2>/dev/null)" ]]; then
  echo "STASH_DIR already exists and is not empty: $STASH_DIR" >&2
  echo "Move it aside or set STASH_DIR to an empty path." >&2
  exit 1
fi

INDEX_FILE="${DOCS_DIR}/index.html"
INDEX_BAK="${INDEX_FILE}.bak"
mkdir -p "$(dirname "$STASH_DIR")"

restore_state() {
  if [[ -f "$INDEX_BAK" ]]; then
    mv "$INDEX_BAK" "$INDEX_FILE"
  fi
  if [[ -d "$STASH_DIR" ]] && [[ ! -d "$DATA_DIR" ]]; then
    mv "$STASH_DIR" "$DATA_DIR"
  fi
}
trap restore_state EXIT

cp "$INDEX_FILE" "$INDEX_BAK"
if [[ -n "$DATA_BASE_URL" ]]; then
  python3 - <<PY
from pathlib import Path
import re

index = Path("$INDEX_FILE")
html = index.read_text()
pattern = r'(<meta\\s+name="data-base-url"\\s+content=")([^"]*)(")'
replacement = r'\\1' + "$DATA_BASE_URL" + r'\\3'
if re.search(pattern, html):
    html = re.sub(pattern, replacement, html, count=1)
else:
    html = html.replace("<head>", "<head>\\n  <meta name=\\"data-base-url\\" content=\\"" + "$DATA_BASE_URL" + "\\">", 1)
index.write_text(html)
PY
fi

if [[ "$CREATE_BUCKET" == "1" ]]; then
  npx wrangler@latest r2 bucket create "$BUCKET_NAME" --remote
fi

find "$DATA_DIR" -type f -print0 | while IFS= read -r -d '' f; do
  key="${f#${DOCS_DIR}/}"
  npx wrangler@latest r2 object put "${BUCKET_NAME}/${key}" --file "$f" --remote
done

mv "$DATA_DIR" "$STASH_DIR"
npx wrangler@latest pages deploy "$DOCS_DIR" --project-name "$PROJECT_NAME"
