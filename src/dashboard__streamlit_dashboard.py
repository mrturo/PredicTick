"""
Streamlit dashboard for Optuna optimization study visualization.

To run: streamlit run src/dashboard__streamlit_dashboard.py
"""

from typing import Any

import joblib
import streamlit as st
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

from updater.data_updater import FileManager
from utils.logger import Logger
from utils.parameters import ParameterLoader

# Global parameter instance
parameters = ParameterLoader(FileManager.last_update())


def safe_get(key: str, default: Any):
    """Safe parameter retrieval with fallback to default."""
    try:
        return parameters[key]
    except KeyError:
        return default


def get_dashboard_config():
    """Retrieve dashboard configuration."""
    return {
        "page_title": safe_get("dashboard_page_title", "Optuna Dashboard"),
        "layout": safe_get("dashboard_layout", "wide"),
        "study_filepath": safe_get("optuna_filepath", None),
        "footer_caption": safe_get(
            "dashboard_footer_caption", "Created with ❤️ using Streamlit and Optuna."
        ),
    }


def load_study(study_path: str):
    """Load Optuna study from file."""
    try:
        study = joblib.load(study_path)
        Logger.success("Study loaded successfully.")
        return study
    except (OSError, ValueError) as exc:
        Logger.error(f"Failed to load study: {exc}")
        st.error(f"❌ Failed to load study: {exc}")
        return None


def display_best_trial(study):
    """Display the best trial information."""
    st.subheader("🏆 Best Trial Summary")
    best = study.best_trial
    st.markdown(
        f"""
    - **Average Accuracy:** `{best.value:.4f}`
    - **Hyperparameters:**
    """
    )
    for key, value in best.params.items():
        st.write(f"  - `{key}`: `{value}`")


def select_plot_type():
    """Display plot selection sidebar."""
    st.sidebar.title("🔧 Visualization Options")
    return st.sidebar.radio(
        "Choose a plot type",
        (
            "Optimization History",
            "Parameter Importances",
            "Parallel Coordinates",
            "Slice Plot",
        ),
    )


def render_plot(plot_type: str, study):
    """Render the selected plot."""
    plot_mapping = {
        "Optimization History": plot_optimization_history,
        "Parameter Importances": plot_param_importances,
        "Parallel Coordinates": plot_parallel_coordinate,
        "Slice Plot": plot_slice,
    }

    st.subheader(f"📊 {plot_type}")
    plot_function = plot_mapping.get(plot_type)

    if plot_function:
        st.plotly_chart(plot_function(study), use_container_width=True)
    else:
        Logger.warning(f"No plot function mapped for: {plot_type}")
        st.warning(f"⚠️ No plot function available for {plot_type}")


if __name__ == "__main__":
    # Main entry point for Streamlit dashboard
    config = get_dashboard_config()
    st.set_page_config(page_title=config["page_title"], layout=config["layout"])

    st.title("📈 Optuna Hyperparameter Tuning Dashboard")

    loaded_study = load_study(config["study_filepath"])
    if not loaded_study:
        st.stop()

    display_best_trial(loaded_study)
    selected_plot = select_plot_type()
    render_plot(selected_plot, loaded_study)

    st.markdown("---")
    st.caption(config["footer_caption"])
