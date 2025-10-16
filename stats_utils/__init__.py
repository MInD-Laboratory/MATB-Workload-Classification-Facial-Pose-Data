"""
Statistical analysis utilities for MATB workload classification project.

This module provides shared statistical functions for analyzing data across
all modalities (pose, eye tracking, ECG, GSR, performance).
"""

from .stats_figures import run_rpy2_lmer, barplot_ax, print_means

__all__ = ['run_rpy2_lmer', 'barplot_ax', 'print_means']
