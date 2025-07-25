"""Plot styling utilities for consistent chart formatting across evaluation modules."""

from typing import Optional

import matplotlib.pyplot as plt  # type: ignore

from utils.parameters import ParameterLoader


# pylint: disable=too-few-public-methods
class Plots:
    """Applies consistent formatting to evaluation plots for model performance visualization."""

    _PARAMS = ParameterLoader()
    _PLOT_TITLE = _PARAMS.get("f1_score_plot_title")

    @staticmethod
    def format_f1_score_plot(title: Optional[str] = None):
        """Applies standardized formatting to F1-score plots."""
        plot_title = title or Plots._PLOT_TITLE
        plt.title(plot_title)
        plt.ylabel("F1-Score")
        plt.ylim(0, 1)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
