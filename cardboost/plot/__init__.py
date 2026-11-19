"""Plotting subpackage (matplotlib optional; HTML report needs nothing)."""
from .figures import compare_with_ebm, plot_1d, plot_2d
from .report import html_report

__all__ = ["plot_1d", "plot_2d", "compare_with_ebm", "html_report"]
