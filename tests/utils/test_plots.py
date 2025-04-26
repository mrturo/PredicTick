"""Unit tests for F1 score plot formatting utility."""

import unittest

import matplotlib.pyplot as plt

from utils.plots import Plots


class TestF1ScorePlotFormatting(unittest.TestCase):
    """Test cases for the `format_f1_score_plot` function."""

    def test_format_f1_score_plot_defaults(self):
        """Verify default formatting: title, label, limits, and grid style."""
        plt.figure()
        Plots.format_f1_score_plot()

        ax = plt.gca()
        self.assertEqual(ax.get_title(), "F1-Score per Class")
        self.assertEqual(ax.get_ylabel(), "F1-Score")
        self.assertTupleEqual(ax.get_ylim(), (0, 1))

        grid_lines = ax.get_xgridlines() + ax.get_ygridlines()
        self.assertTrue(all(line.get_linestyle() == "--" for line in grid_lines))
        self.assertTrue(all(line.get_alpha() == 0.5 for line in grid_lines))

        plt.close()

    def test_format_f1_score_plot_custom_title(self):
        """Check that a custom title is correctly applied to the plot."""
        custom_title = "Custom F1 Title"
        plt.figure()
        Plots.format_f1_score_plot(title=custom_title)

        ax = plt.gca()
        self.assertEqual(ax.get_title(), custom_title)

        plt.close()
