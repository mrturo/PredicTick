#!/bin/bash

set -euo pipefail
cd "$(dirname "$0")"

# Color formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔧 Project Utility Script${NC}"

# Default paths to use in cleaning phase if none are provided
DEFAULT_PATHS=("src/" "tests/")

function install_project() {
    local mode="${1:-}"
    local PYTHON_BINARY="${PYTHON_BINARY_OVERRIDE:-python3.13}"
    local REQUIRED_MAJOR=3
    local REQUIRED_MINOR=10

    if [[ -z "$mode" || ( "$mode" != "prod" && "$mode" != "dev" ) ]]; then
        echo -e "${RED}❌ You must specify the installation mode: 'prod' or 'dev'.${NC}"
        echo -e "${RED}   Example: bash envtool.sh install prod${NC}"
        echo -e "${RED}   Example: bash envtool.sh reinstall prod${NC}"
        echo -e "${RED}   Example: bash envtool.sh install dev${NC}"
        echo -e "${RED}   Example: bash envtool.sh reinstall dev${NC}"
        exit 1
    fi

    echo -e "${GREEN}🚀 Installing project using $PYTHON_BINARY...${NC}"
    find . -name '__pycache__' -exec rm -rf {} +

    PY_VERSION=$($PYTHON_BINARY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

    if [ "$PY_MAJOR" -lt "$REQUIRED_MAJOR" ] || { [ "$PY_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$PY_MINOR" -lt "$REQUIRED_MINOR" ]; }; then
        echo -e "${RED}❌ Python version must be >= $REQUIRED_MAJOR.$REQUIRED_MINOR. Found: $PY_VERSION${NC}"
        exit 1
    fi

    if [ ! -d ".venv" ]; then
        echo -e "${GREEN}📦 Creating virtual environment (.venv) using $PYTHON_BINARY...${NC}"
        $PYTHON_BINARY -m venv .venv
    else
        echo -e "${GREEN}📁 Virtual environment already exists. Skipping creation.${NC}"
    fi

    echo -e "${GREEN}💡 Activating virtual environment...${NC}"
    source .venv/bin/activate

    echo -e "${GREEN}⬆️  Upgrading pip...${NC}"
    pip install --upgrade pip

    if [ -f "requirements.txt" ]; then
        echo -e "${GREEN}📄 Installing dependencies from requirements.txt...${NC}"
        pip install -r requirements.txt
    else
        echo -e "${RED}❌ No requirements.txt found. Please provide one.${NC}"
        exit 1
    fi

    if [[ "$mode" == "dev" && -f "requirements-dev.txt" ]]; then
        echo -e "${GREEN}🛠️  Installing development dependencies from requirements-dev.txt...${NC}"
        pip install -r requirements-dev.txt
    fi

    echo -e "${GREEN}✅ Installation completed.${NC}"
}

function clean_cache() {
    echo -e "${GREEN}🧹 Cleaning project cache and build artifacts...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .pytest_cache .mypy_cache .cache dist build *.egg-info htmlcov .coverage nonexistent.json tests/data
    echo -e "${GREEN}✅ Cache and artifacts removed.${NC}"
}

function clean_env() {
    if [ -d ".venv" ]; then
        echo -e "${GREEN}🧨 Removing virtual environment (.venv)...${NC}"
        rm -rf .venv
        echo -e "${GREEN}✅ .venv removed successfully.${NC}"
    else
        echo -e "${GREEN}ℹ️  No .venv directory found. Nothing to remove.${NC}"
    fi
}

function clean_all() {
    clean_cache
    clean_env
}

function code_check() {
    local paths=("${@:-${DEFAULT_PATHS[@]}}")

    echo -e "${GREEN}📁 Using paths: ${paths[*]}${NC}"

    echo -e "${GREEN}🔧 Running isort to sort and organize imports...${NC}"
    isort "${paths[@]}"

    echo -e "${GREEN}🧹 Running autoflake to remove unused imports and variables...${NC}"
    autoflake --remove-all-unused-imports --remove-unused-variables --in-place --recursive "${paths[@]}"

    echo -e "${GREEN}📝 Running pydocstringformatter to clean and format docstrings (PEP 257)...${NC}"
    pydocstringformatter src/ --write

    echo -e "${GREEN}🎨 Running black to automatically format code (PEP 8)...${NC}"
    black "${paths[@]}"

    echo -e "${GREEN}🛡️  Running bandit to detect potential security issues...${NC}"
    bandit -r "${paths[@]}"

    echo -e "${GREEN}🔍 Running pylint for static code analysis and linting...${NC}"
    pylint --persistent=no "${paths[@]}"

    echo -e "${GREEN}✅ Code quality checks completed.${NC}"
}

function check_status() {
    echo -e "${GREEN}🔎 Checking project environment status...${NC}"

    if [ -d ".venv" ]; then
        echo -e "${GREEN}✔️  Virtual environment (.venv) exists.${NC}"
    else
        echo -e "${GREEN}❌ Virtual environment (.venv) is missing.${NC}"
    fi

    if [ -f "requirements.txt" ]; then
        echo -e "${GREEN}✔️  requirements.txt found.${NC}"
    else
        echo -e "${GREEN}❌ requirements.txt is missing.${NC}"
    fi

    if [ -f "requirements-dev.txt" ]; then
        echo -e "${GREEN}✔️  requirements-dev.txt found.${NC}"
    else
        echo -e "${RED}⚠️  requirements-dev.txt is missing (development dependencies won't be installed).${NC}"
    fi

    if [ -x ".venv/bin/python" ]; then
        VENV_PYTHON_VERSION=$(.venv/bin/python --version 2>&1)
        VENV_PIP_VERSION=$(.venv/bin/pip --version 2>&1)
        echo -e "${GREEN}🐍 Python version in .venv: ${VENV_PYTHON_VERSION}${NC}"
        echo -e "${GREEN}📦 Pip version in .venv: ${VENV_PIP_VERSION}${NC}"
    fi

    echo -e "${GREEN}🔚 Status check complete.${NC}"
}

function test_project() {
    local verbose_flag=""
    [[ "${1:-}" == "--verbose" ]] && verbose_flag="-v" && shift
    echo -e "${GREEN}💡 Activating virtual environment...${NC}"
    source .venv/bin/activate
    echo -e "${GREEN}🧪 Running tests with coverage (source=src, filtered)...${NC}"
    PYTHONPATH=. coverage run --rcfile=.coveragerc -m pytest $verbose_flag "$@" --cache-clear
    echo -e "${GREEN}📊 Generating filtered coverage report...${NC}"
    coverage report -m
    coverage html
    echo -e "${GREEN}🌐 HTML report available at htmlcov/index.html${NC}"
}

case ${1:-} in
    install)
        unset http_proxy
        unset https_proxy
        shift
        install_project "$@"
        ;;
    reinstall)
        unset http_proxy
        unset https_proxy
        clean_all
        shift
        install_project "$@"
        ;;
    uninstall)
        clean_all
        ;;
    clean-env)
        clean_env
        ;;
    clean-cache)
        clean_cache
        ;;
    code-check)
        shift
        code_check "$@"
        ;;
    status)
        check_status
        ;;
    test)
        shift
        unset http_proxy
        unset https_proxy
        test_project --verbose tests/
        rm -rf tests/data
        ;;
    *)
        echo -e "${GREEN}Usage:"
        echo -e "${GREEN}  bash envtool.sh {install|reinstall} {dev|prod}${NC}"
        echo -e "${GREEN}  bash envtool.sh code-check [optional: specific paths]${NC}"
        echo -e "${GREEN}  bash envtool.sh {uninstall|clean-env|clean-cache|status|test}${NC}"
        exit 1
        ;;
esac
