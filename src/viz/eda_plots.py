"""Publication-grade visual figure generation for Stage 1.3 Lexical EDA.

This module is strictly isolated from core numerical computation and executes
exclusively via the headless Matplotlib 'Agg' backend. It renders:
1. Multi-domain log-scale ECDF curves of token lengths (Passages vs. Queries).
2. Multi-domain step-histograms of IDF-weighted coverage (Relevant vs. Background).
"""

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
# Enforce headless raster/vector rendering with zero X11/GUI display dependencies
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..eda import LexicalProfile

logger = logging.getLogger("rettune.viz")

# Palette constants: colorblind-safe Okabe-Ito / ColorBrewer inspired
COLOR_PASSAGE = "#1f77b4"     # Steel blue
COLOR_QUERY = "#ff7f0e"       # Amber orange
COLOR_RELEVANT = "#2ca02c"    # Green
COLOR_BACKGROUND = "#7f7f7f"  # Neutral gray


def _setup_figure_style() -> None:
    """Set clean typography and minimal publication aesthetics."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "svg.fonttype": "none",        # Preserves searchable text in SVGs
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })


def _compute_ecdf(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical cumulative distribution function coordinates."""
    if arr.size == 0:
        return np.array([0.0]), np.array([0.0])
    s = np.sort(arr)
    y = np.arange(1, len(s) + 1, dtype=np.float64) / len(s)
    return s, y


def render_token_length_ecdf(
    profiles: Dict[str, LexicalProfile],
    output_dir: Path | str,
    filename_stem: str = "eda_token_length_distributions",
) -> Dict[str, Path]:
    """Render multi-panel log-scale ECDF curves contrasting Passages vs. Queries.

    Saves dual output: Scalable Vector Graphics (.svg) and high-res raster (.png).
    """
    if not profiles:
        logger.warning("No profiles provided to render_token_length_ecdf; skipping figure export.")
        return {}

    _setup_figure_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = list(profiles.keys())
    n_panels = len(dataset_names)

    ncols = min(n_panels, 3)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        sharey=(nrows == 1),
        constrained_layout=True,
    )
    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten().tolist()
    else:
        ax_list = [axes]

    for extra_ax in ax_list[n_panels:]:
        extra_ax.set_visible(False)

    # Derive dynamic upper bound across all profiles (rounded up to next power of 10)
    max_observed = max(
        (max(p.doc_summary.max, p.query_summary_all.max) for p in profiles.values()),
        default=1000.0,
    )
    right_bound = max(1000.0, 10 ** math.ceil(math.log10(max(max_observed, 10.0))))

    try:
        for idx, (ax, name) in enumerate(zip(ax_list[:n_panels], dataset_names)):
            prof = profiles[name]

            # 1. Plot ECDF for passages
            d_x, d_y = _compute_ecdf(prof.doc_lengths)
            ax.plot(
                d_x,
                d_y,
                label=f"Passages (N={prof.doc_summary.count:,})",
                color=COLOR_PASSAGE,
                linewidth=2.0,
            )

            # 2. Plot ECDF for queries (all)
            q_x, q_y = _compute_ecdf(prof.query_lengths_all)
            ax.plot(
                q_x,
                q_y,
                label=f"Queries (N={prof.query_summary_all.count:,})",
                color=COLOR_QUERY,
                linewidth=2.0,
                linestyle="--",
            )

            # 3. Annotate passage landmarks
            ax.axvline(
                prof.doc_summary.median,
                color=COLOR_PASSAGE,
                linestyle=":",
                alpha=0.7,
                label=f"Passage p50 ({prof.doc_summary.median:.0f})",
            )
            ax.axvline(
                prof.doc_summary.mean,
                color="#000000",
                linestyle="-.",
                alpha=0.6,
                label=f"Passage avgdl ({prof.doc_summary.mean:.1f})",
            )
            if "p99" in prof.doc_summary.percentiles:
                p99_val = prof.doc_summary.percentiles["p99"]
                p99_label = f"Passage p99 ({p99_val:.0f})"
            else:
                p99_val = prof.doc_summary.max
                p99_label = f"Passage max ({p99_val:.0f})"

            ax.axvline(
                p99_val,
                color="#d62728",
                linestyle=":",
                alpha=0.7,
                label=p99_label,
            )

            ax.set_xscale("log")
            ax.set_xlim(left=1.0, right=right_bound)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.5)

            skew_str = f"Skewness g₁ = {prof.doc_summary.skewness:.2f}"
            ax.set_title(f"{name.upper()}\n({skew_str})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Token Length (Log₁₀ Scale)", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Empirical Cumulative Probability (ECDF)", fontsize=10)

            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

        svg_path = out_dir / f"{filename_stem}.svg"
        png_path = out_dir / f"{filename_stem}.png"

        fig.savefig(svg_path, format="svg", metadata={"Date": None})
        fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
        logger.info("Exported token length distributions to %s and %s", svg_path, png_path)

        return {"svg": svg_path, "png": png_path}

    finally:
        plt.close(fig)


def render_coverage_density(
    profiles: Dict[str, LexicalProfile],
    output_dir: Path | str,
    filename_stem: str = "eda_coverage_density",
) -> Dict[str, Path]:
    """Render multi-panel step histograms contrasting Relevant vs. Random Background overlap.

    Saves dual output: Scalable Vector Graphics (.svg) and high-res raster (.png).
    """
    if not profiles:
        logger.warning("No profiles provided to render_coverage_density; skipping figure export.")
        return {}

    _setup_figure_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = list(profiles.keys())
    n_panels = len(dataset_names)

    ncols = min(n_panels, 3)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        sharey=False,
        constrained_layout=True,
    )
    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten().tolist()
    else:
        ax_list = [axes]

    for extra_ax in ax_list[n_panels:]:
        extra_ax.set_visible(False)

    # 50 uniform bins strictly clamped in [0.0, 1.0] to preserve point mass at 0.0
    bins = np.linspace(0.0, 1.0, 51)

    try:
        for idx, (ax, name) in enumerate(zip(ax_list[:n_panels], dataset_names)):
            prof = profiles[name]

            # 1. Random background null distribution
            if prof.coverage_random_background.size > 0:
                ax.hist(
                    prof.coverage_random_background,
                    bins=bins,
                    density=True,
                    histtype="stepfilled",
                    alpha=0.3,
                    color=COLOR_BACKGROUND,
                    label=f"Background Noise (N={len(prof.coverage_random_background):,})",
                    edgecolor=COLOR_BACKGROUND,
                    linewidth=1.2,
                )

            # 2. Ground-truth relevant pairs distribution
            if prof.coverage_relevant_all.size > 0:
                ax.hist(
                    prof.coverage_relevant_all,
                    bins=bins,
                    density=True,
                    histtype="stepfilled",
                    alpha=0.5,
                    color=COLOR_RELEVANT,
                    label=f"Relevant Pairs (N={len(prof.coverage_relevant_all):,})",
                    edgecolor=COLOR_RELEVANT,
                    linewidth=1.5,
                )

            ax.set_xlim(-0.02, 1.02)
            ax.grid(True, alpha=0.5)

            sep = prof.separation_all
            stats_title = (
                f"{name.upper()}\n"
                f"W₁ = {sep.wasserstein_distance:.3f} | Cohen's d = {sep.cohens_d:.2f} | "
                f"Δmed = {sep.delta_median:.2f}"
            )
            ax.set_title(stats_title, fontsize=10, fontweight="bold")
            ax.set_xlabel("IDF-Weighted Query Coverage in Passage", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Probability Density", fontsize=10)

            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        svg_path = out_dir / f"{filename_stem}.svg"
        png_path = out_dir / f"{filename_stem}.png"

        fig.savefig(svg_path, format="svg", metadata={"Date": None})
        fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
        logger.info("Exported coverage density figures to %s and %s", svg_path, png_path)

        return {"svg": svg_path, "png": png_path}

    finally:
        plt.close(fig)
