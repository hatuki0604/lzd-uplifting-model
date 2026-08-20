#!/bin/sh
set -eu

REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_DIR"

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export LZD_API_URL=${LZD_API_URL:-http://localhost:18000}

exec .venv/bin/streamlit run demo_ba/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
