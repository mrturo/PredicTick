#!/bin/bash

# Color formatting
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}PredicTick Task Runner${NC}"

# Check virtual environment existence
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}No virtual environment found. Please create one using 'python3 -m venv .venv'${NC}"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH=./src

case $1 in
    update)
        echo -e "${GREEN}Updating market data...${NC}"
        unset http_proxy
        unset https_proxy
        /usr/bin/caffeinate -dimsu &
        .venv/bin/python src/market_data/updater.py
        ;;
    auto-update)
        echo -e "${GREEN}Starting automatic market data updates every hour...${NC}"
        unset http_proxy
        unset https_proxy
        /usr/bin/caffeinate -dimsu &
        .venv/bin/python src/market_data/updater_cron.py
        ;;
    train)
        echo -e "${GREEN}Training model...${NC}"
        /usr/bin/caffeinate -dimsu &
        .venv/bin/python src/training/train.py
        ;;
    evaluate)
        echo -e "${GREEN}Evaluating model...${NC}"
        .venv/bin/python src/evaluation/evaluate_model.py
        ;;
    predict)
        echo -e "${GREEN}Making daily prediction...${NC}"
        .venv/bin/python src/prediction/predict.py
        ;;
    backtest)
        echo -e "${GREEN}Running backtest...${NC}"
        /usr/bin/caffeinate -dimsu &
        .venv/bin/python src/backtesting/backtest.py
        ;;
    dashboard)
        echo -e "${GREEN}Launching Streamlit Dashboard...${NC}"
        .venv/bin/streamlit run src/dashboard/streamlit_dashboard.py
        ;;
    all)
        echo -e "${GREEN}Running full pipeline: Update -> Train -> Evaluate -> Predict${NC}"
        unset http_proxy
        unset https_proxy
        /usr/bin/caffeinate -dimsu &
        .venv/bin/python src/market_data/updater.py
        .venv/bin/python src/training/train.py
        .venv/bin/python src/evaluation/evaluate_model.py
        .venv/bin/python src/to_review/prediction__predict_model.py
        ;;
    *)
        echo "Usage: bash run_tasks.sh {update|auto-update|train|evaluate|predict|backtest|dashboard|all}"
        ;;
esac
