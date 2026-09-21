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
            ax.set_xlabel("Word Length (Log₁₀ Scale)", fontsize=10)
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


def render_summary_table(
    profiles: Dict[str, LexicalProfile],
    output_dir: Path | str,
    filename_stem: str = "eda_summary_table",
) -> Dict[str, Path]:
    """Render publication-grade graphic summary tables with clear units.

    Saves dual output: Scalable Vector Graphics (.svg) and high-res raster (.png).
    """
    if not profiles:
        logger.warning("No profiles provided to render_summary_table; skipping table export.")
        return {}

    _setup_figure_style()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build Table 1 Rows (Length Dynamics - Words per Item)
    col_labels_1 = [
        "Dataset", "Unit", "Items (N)", "Mean (w)", "Med (w)",
        "IQR (w)", "p95 (w)", "p99 (w)", "Max (w)", "Skewness (g₁)"
    ]
    rows_1: list[list[str]] = []
    for ds_name, prof in profiles.items():
        units = [
            ("Passages", prof.doc_summary),
            ("Queries (All)", prof.query_summary_all),
        ]
        for idx, (unit_label, summary) in enumerate(units):
            ds_col = ds_name.upper() if idx == 0 else ""
            p99_val = summary.percentiles.get("p99", summary.max)
            rows_1.append([
                ds_col,
                unit_label,
                f"{summary.count:,}",
                f"{summary.mean:.1f}",
                f"{summary.median:.1f}",
                f"{summary.iqr:.1f}",
                f"{summary.percentiles.get('p95', 0.0):.1f}",
                f"{p99_val:.1f}",
                f"{summary.max:.1f}",
                f"{summary.skewness:.2f}",
            ])

    # 2. Build Table 2 Rows (Coverage & Separation - Units in headers, bare numbers in cells)
    col_labels_2 = [
        "Dataset", "Rel Pairs", "Rel Mean (%)", "Rel Med (%)",
        "Noise Med (%)", "Δ Med (%)", "Cohen's d (σ)", "Wasserstein (W₁)"
    ]
    rows_2: list[list[str]] = []
    for ds_name, prof in profiles.items():
        rel = prof.coverage_summary_relevant_all
        bg = prof.coverage_summary_random
        sep = prof.separation_all
        rows_2.append([
            ds_name.upper(),
            f"{rel.count:,}",
            f"{rel.mean * 100:.1f}",
            f"{rel.median * 100:.1f}",
            f"{bg.median * 100:.1f}",
            f"{sep.delta_median * 100:+.1f}",
            f"{sep.cohens_d:.2f}",
            f"{sep.wasserstein_distance:.3f}",
        ])

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(11.5, max(5.2, 1.8 + 0.45 * (len(rows_1) + len(rows_2)))),
        gridspec_kw={"height_ratios": [max(1.3, len(rows_1) * 0.45), max(1.0, len(rows_2) * 0.45)]},
    )
    ax1.axis("off")
    ax2.axis("off")

    try:
        # Table 1: Length Dynamics
        t1 = ax1.table(cellText=rows_1, colLabels=col_labels_1, loc="center", cellLoc="center")
        t1.auto_set_font_size(False)
        t1.set_fontsize(9.5)
        t1.scale(1.0, 1.5)

        for (row, col), cell in t1.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#1e293b")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                bg = "#ffffff" if ((row - 1) // 2) % 2 == 0 else "#f8fafc"
                cell.set_facecolor(bg)
                if col == 9:  # Skewness
                    try:
                        skew_val = float(rows_1[row - 1][9])
                        if skew_val > 2.0:
                            cell.set_facecolor("#fee2e2")
                            cell.get_text().set_color("#991b1b")
                            cell.get_text().set_weight("bold")
                    except ValueError:
                        pass

        ax1.set_title("Table 1: Corpus & Query Length Dynamics", fontsize=11, fontweight="bold", pad=10, loc="left")

        # Table 2: Separation
        t2 = ax2.table(cellText=rows_2, colLabels=col_labels_2, loc="center", cellLoc="center")
        t2.auto_set_font_size(False)
        t2.set_fontsize(9.5)
        t2.scale(1.0, 1.5)

        for (row, col), cell in t2.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#1e293b")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                bg = "#ffffff" if row % 2 == 1 else "#f8fafc"
                cell.set_facecolor(bg)
                if col == 6:  # Cohen's d
                    try:
                        d_val = float(rows_2[row - 1][6])
                        if d_val > 2.0:
                            cell.set_facecolor("#dcfce7")
                            cell.get_text().set_color("#166534")
                            cell.get_text().set_weight("bold")
                    except ValueError:
                        pass
                elif col == 5:  # Delta Median
                    try:
                        delta_val = float(rows_2[row - 1][5].replace("+", ""))
                        if delta_val > 20.0:
                            cell.set_facecolor("#fef3c7")
                            cell.get_text().set_color("#92400e")
                            cell.get_text().set_weight("bold")
                    except ValueError:
                        pass

        ax2.set_title("Table 2: IDF-Weighted Query Coverage & Separation vs. Noise Floor", fontsize=11, fontweight="bold", pad=10, loc="left")

        fig.tight_layout()
        svg_path = out_dir / f"{filename_stem}.svg"
        png_path = out_dir / f"{filename_stem}.png"

        fig.savefig(svg_path, format="svg", metadata={"Date": None})
        fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
        logger.info("Exported summary table to %s and %s", svg_path, png_path)

        return {"svg": svg_path, "png": png_path}

    finally:
        plt.close(fig)
