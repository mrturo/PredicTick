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
        .venv/bin/python src/updater/data_updater.py
        ;;
    train)
        echo -e "${GREEN}Training model...${NC}"
        /usr/bin/caffeinate -dimsu &
        .venv/bin/python src/training__train_model.py
        ;;
    evaluate)
        echo -e "${GREEN}Evaluating model...${NC}"
        .venv/bin/python src/evaluation__evaluate_model.py
        ;;
    predict)
        echo -e "${GREEN}Making daily prediction...${NC}"
        .venv/bin/python src/prediction__predict_model.py
        ;;
    backtest)
        echo -e "${GREEN}Running backtest...${NC}"
        .venv/bin/python src/backtesting__backtest.py
        ;;
    dashboard)
        echo -e "${GREEN}Launching Streamlit Dashboard...${NC}"
        .venv/bin/streamlit run src/dashboard__streamlit_dashboard.py
        ;;
    all)
        echo -e "${GREEN}Running full pipeline: Update -> Train -> Evaluate -> Predict${NC}"
        unset http_proxy
        unset https_proxy
        /usr/bin/caffeinate -dimsu
        .venv/bin/python src/updater/data_updater.py
        .venv/bin/python src/training__train_model.py
        .venv/bin/python src/evaluation__evaluate_model.py
        .venv/bin/python src/prediction__predict_model.py
        ;;
    *)
        echo "Usage: bash run_tasks.sh {update|train|evaluate|predict|backtest|dashboard|all}"
        ;;
esac
