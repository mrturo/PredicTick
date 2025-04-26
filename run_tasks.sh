#!/usr/bin/env bash
# Fail fast + strict mode
set -euo pipefail

# ---------- Colors ----------
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}PredicTick Task Runner${NC}"

# ---------- Resolve project root & venv ----------
cd "$(dirname "$0")"

# Pick Python/Streamlit from venv if present, else from PATH
if [[ -x ".venv/bin/python" ]]; then
  PYBIN=".venv/bin/python"
else
  if command -v python3 >/dev/null 2>&1; then PYBIN="python3"
  elif command -v python >/dev/null 2>&1; then PYBIN="python"
  else
    echo "No Python found. Install Python 3 or create .venv"; exit 1
  fi
fi

if [[ -x ".venv/bin/streamlit" ]]; then
  STREAMLIT=".venv/bin/streamlit"
else
  STREAMLIT="streamlit"
fi

# Ensure src is importable
export PYTHONPATH="${PYTHONPATH:-}:./src"

# ---------- Helpers ----------
maybe_caffeinate() {
  # Only on macOS and if caffeinate exists
  if [[ "$(uname -s)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu &
    CAFFE_PID=$!
    # Stop caffeinate on exit/signals
    trap '[[ -n "${CAFFE_PID:-}" ]] && kill -TERM "$CAFFE_PID" 2>/dev/null || true; exit 143' TERM INT
  fi
}

unset_proxies() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true
}

runm() { "${PYBIN}" -m "$@"; }   # run python module
runp() { "${PYBIN}" "$@"; }      # run python file

# ---------- Commands ----------
CMD="${1:-update}"

case "${CMD}" in
  update)
    echo -e "${GREEN}Updating market data...${NC}"
    unset_proxies
    maybe_caffeinate
    runm src.market_data.updater
    ;;

  auto-update)
    echo -e "${GREEN}Starting automatic market data updates every hour...${NC}"
    unset_proxies
    maybe_caffeinate
    runm src.market_data.updater_cron
    ;;

  train)
    echo -e "${GREEN}Training model...${NC}"
    maybe_caffeinate
    runm src.training.train
    ;;

  evaluate)
    echo -e "${GREEN}Evaluating model...${NC}"
    runm src.evaluation.evaluate_model
    ;;

  predict)
    echo -e "${GREEN}Making daily prediction...${NC}"
    runm src.prediction.predict
    ;;

  backtest)
    echo -e "${GREEN}Running backtest...${NC}"
    maybe_caffeinate
    runm src.backtesting.backtest
    ;;

  dashboard)
    echo -e "${GREEN}Launching Streamlit Dashboard...${NC}"
    "${STREAMLIT}" run src/dashboard/streamlit_dashboard.py
    ;;

  all)
    echo -e "${GREEN}Running full pipeline: Update -> Train -> Evaluate -> Predict${NC}"
    unset_proxies
    maybe_caffeinate
    runm src.market_data.updater
    runm src.training.train
    runm src.evaluation.evaluate_model
    runm src.prediction.predict
    ;;

  *)
    echo "Usage: bash run_tasks.sh {update|auto-update|train|evaluate|predict|backtest|dashboard|all}"
    exit 2
    ;;
esac
