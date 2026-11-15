"""Visualization utilities for RetTune benchmark diagnostics."""

from .benchmark_plots import (
    render_stage2_bm25_plus_figure,
    render_stage3_spectral_and_mrl_figure,
    render_stage4_calibration_and_transfer_figure,
    render_stage5_latency_and_ceiling_figure,
)
from .console_tables import (
    display_coverage_summary_table,
    display_dense_summary_table,
    display_fusion_summary_table,
    display_length_summary_table,
    display_reranker_summary_table,
    display_sparse_summary_table,
)
from .eda_plots import render_coverage_density, render_summary_table, render_token_length_ecdf

__all__ = [
    "render_token_length_ecdf",
    "render_coverage_density",
    "render_summary_table",
    "render_stage2_bm25_plus_figure",
    "render_stage3_spectral_and_mrl_figure",
    "render_stage4_calibration_and_transfer_figure",
    "render_stage5_latency_and_ceiling_figure",
    "display_length_summary_table",
    "display_coverage_summary_table",
    "display_sparse_summary_table",
    "display_dense_summary_table",
    "display_fusion_summary_table",
    "display_reranker_summary_table",
]
