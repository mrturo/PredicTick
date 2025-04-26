"""Plot styling utilities for consistent chart formatting across evaluation modules."""

import matplotlib.pyplot as plt


def format_f1_score_plot(title: str = "F1-Score per Class"):
    """Applies standardized formatting to F1-score plots."""
    plt.title(title)
    plt.ylabel("F1-Score")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
