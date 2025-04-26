"""
Streamlit dashboard for Optuna optimization study visualization.

To run: streamlit run src/dashboard/streamlit_dashboard.py
"""

import joblib
import streamlit as st
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

from market_data.gateway import Gateway
from utils.logger import Logger
from utils.parameters import ParameterLoader

_PARAMS = ParameterLoader(Gateway.get_last_update())
_DASHBOARD_PAGE_TITLE = _PARAMS.get("dashboard_page_title")
_LAYOUT = _PARAMS.get("dashboard_layout")
_STUDY_FILEPATH = _PARAMS.get("optuna_filepath")
_FOOTER_CAPTION = _PARAMS.get("dashboard_footer_caption")


def get_dashboard_config():
    """Retrieve dashboard configuration."""
    return {
        "page_title": _DASHBOARD_PAGE_TITLE,
        "layout": _LAYOUT,
        "study_filepath": _STUDY_FILEPATH,
        "footer_caption": _FOOTER_CAPTION,
    }


def load_study(study_filepath: str):
    """Load Optuna study from file."""
    try:
        study = joblib.load(study_filepath)
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
