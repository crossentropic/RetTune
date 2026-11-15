"""Publication-grade SVG and PNG figure generators for Stages 2 to 5 (`results/figures/`).

Executes strictly via the headless Matplotlib 'Agg' backend with zero GUI dependencies.
Exports dual vector (.svg) and high-DPI raster (.png) figures:
1. `stage2_bm25_plus_surface.{svg,png}`
2. `stage3_spectral_and_mrl.{svg,png}`
3. `stage4_calibration_and_transfer.{svg,png}`
4. `stage5_latency_and_ceiling.{svg,png}`
"""

import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..dense import DenseDatasetReport
from ..fusion import FusionDatasetReport, TransferCellResult
from ..reranker import RerankerDatasetReport
from ..sparse import SparseDatasetReport
from .eda_plots import _setup_figure_style

logger = logging.getLogger("rettune.viz.benchmark_plots")


def _save_dual_figure(fig: plt.Figure, out_dir: Path, stem: str) -> Dict[str, Path]:
    """Save figure as both deterministic SVG and 180-DPI PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{stem}.svg"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(svg_path, format="svg", metadata={"Date": None}, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=180, bbox_inches="tight")
    logger.info("Exported figure %s.{svg,png} to %s", stem, out_dir)
    return {"svg": svg_path, "png": png_path}


def render_stage2_bm25_plus_figure(
    reports: Mapping[str, SparseDatasetReport],
    output_dir: Path | str,
    filename_stem: str = "stage2_bm25_plus_surface",
) -> Dict[str, Path]:
    """Render Stage 2 BM25+ (b, delta) response surfaces, long-passage stratification & Recall@K saturation."""
    if not reports:
        return {}

    _setup_figure_style()
    out_dir = Path(output_dir)
    ds_names = list(reports.keys())
    n_ds = len(ds_names)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=max(n_ds, 2),
        figsize=(5.5 * max(n_ds, 2), 9.0),
        constrained_layout=True,
    )

    try:
        # Top Row: 2D (b, delta) Response Surface Heatmaps on Dev
        for idx, ds_name in enumerate(ds_names):
            ax = axes[0, idx]
            rep = reports[ds_name]
            surf = np.asarray(rep.response_surface_ndcg_dev, dtype=np.float64)
            b_grid = rep.response_surface_b_grid
            d_grid = rep.response_surface_delta_grid

            im = ax.imshow(surf, cmap="YlGnBu", aspect="auto", origin="lower")
            ax.set_xticks(range(len(d_grid)))
            ax.set_xticklabels([f"{d:.2f}" for d in d_grid], fontsize=8.5)
            ax.set_yticks(range(len(b_grid)))
            ax.set_yticklabels([f"{b:.2f}" for b in b_grid], fontsize=8.5)

            for r_i in range(len(b_grid)):
                for c_i in range(len(d_grid)):
                    val = surf[r_i, c_i]
                    mean_v = float(np.mean(surf))
                    txt_color = "white" if val > mean_v else "#1e293b"
                    ax.text(
                        c_i,
                        r_i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=txt_color,
                        fontweight="bold" if (
                            b_grid[r_i] == rep.winning_sparse_params.b
                            and d_grid[c_i] == rep.winning_sparse_params.delta
                        ) else "normal",
                    )

            win = rep.winning_sparse_params
            ax.set_title(
                f"{ds_name.upper()} — Dev NDCG@10 Surface\n"
                f"Winner: b*={win.b:.2f}, δ*={win.delta:.2f} (k₁*={win.k1:.1f})",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlabel("BM25+ Lower-Bound Floor (δ)", fontsize=9.5)
            if idx == 0:
                ax.set_ylabel("Length Normalization (b)", fontsize=9.5)
            fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)

        for extra_idx in range(n_ds, axes.shape[1]):
            axes[0, extra_idx].set_visible(False)

        # Bottom Left: Recall@K Saturation Curves across Datasets (justifying K=100)
        ax_sat = axes[1, 0]
        palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
        for idx, ds_name in enumerate(ds_names):
            rep = reports[ds_name]
            rec_map = rep.test_winning_eval.recall_at_k
            ks = sorted(rec_map.keys())
            vals = [rec_map[k] * 100.0 for k in ks]
            ax_sat.plot(
                ks,
                vals,
                marker="o",
                linewidth=2.2,
                color=palette[idx % len(palette)],
                label=f"{ds_name.upper()} (R@100={rec_map.get(100, 0.0)*100:.1f}%)",
            )
        ax_sat.axvline(100, color="#ef4444", linestyle="--", linewidth=1.4, label="Candidate Cutoff (K=100)")
        ax_sat.set_title("Recall@K Saturation on Test Split", fontsize=10.5, fontweight="bold")
        ax_sat.set_xlabel("Candidate Pool Depth (K)", fontsize=9.5)
        ax_sat.set_ylabel("Recall@K (%)", fontsize=9.5)
        ax_sat.grid(True, alpha=0.5)
        ax_sat.legend(loc="lower right", fontsize=8.5)

        # Bottom Middle / Right: Long-Passage Tertile Stratification & Single-Term Rank Invariance
        ax_strat = axes[1, 1]
        x_pos = np.arange(3)
        width = 0.25
        for idx, ds_name in enumerate(ds_names):
            rep = reports[ds_name]
            buckets = rep.long_passage_stratification_test
            if len(buckets) == 3:
                deltas = [b.delta_ndcg_at_10 * 100.0 for b in buckets]
                offset = (idx - (n_ds - 1) / 2.0) * width
                ax_strat.bar(
                    x_pos + offset,
                    deltas,
                    width=width * 0.9,
                    color=palette[idx % len(palette)],
                    alpha=0.85,
                    label=f"{ds_name.upper()}",
                )
        ax_strat.axhline(0.0, color="#334155", linewidth=1.0)
        ax_strat.set_xticks(x_pos)
        ax_strat.set_xticklabels(["Short (T1)", "Medium (T2)", "Long (T3)"], fontsize=9)
        ax_strat.set_title(
            "Long-Passage Stratification (Test)\nΔ NDCG@10 (%) vs. Default BM25Okapi (b=0.75)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_strat.set_xlabel("Relevant Passage Length Tertile", fontsize=9.5)
        ax_strat.set_ylabel("Δ NDCG@10 (% points)", fontsize=9.5)
        ax_strat.grid(True, axis="y", alpha=0.5)
        ax_strat.legend(loc="best", fontsize=8.5)

        if axes.shape[1] >= 3:
            ax_inv = axes[1, 2]
            target_rep = reports.get("nfcorpus", next(iter(reports.values())))
            st_diag = target_rep.single_term_diagnostic_test
            labels = [
                f"Single-Term (|q|=1)\n(n={st_diag.single_term_query_count})",
                f"Multi-Term (|q|≥2)\n(n={st_diag.multi_term_query_count})",
            ]
            ok_vals = [st_diag.single_term_ndcg_okapi, st_diag.multi_term_ndcg_okapi]
            pl_vals = [st_diag.single_term_ndcg_bm25_plus, st_diag.multi_term_ndcg_bm25_plus]
            bx = np.arange(2)
            bw = 0.32
            ax_inv.bar(bx - bw / 2, ok_vals, width=bw, color="#64748b", label="BM25Okapi (δ=0.0)")
            ax_inv.bar(bx + bw / 2, pl_vals, width=bw, color="#0ea5e9", label="BM25+ (δ>0.0)")
            ax_inv.set_xticks(bx)
            ax_inv.set_xticklabels(labels, fontsize=9)
            ax_inv.set_ylabel("Test NDCG@10", fontsize=9.5)
            ax_inv.set_title(
                f"{target_rep.dataset_name.upper()} — Single-Term δ Invariance\n"
                f"Exact Rank Match on |q|=1: {st_diag.single_term_exact_rank_match_rate*100:.1f}% "
                f"(max |Δ|={st_diag.single_term_max_abs_ndcg_diff:.1e})",
                fontsize=10.0,
                fontweight="bold",
            )
            ax_inv.grid(True, axis="y", alpha=0.5)
            ax_inv.legend(loc="upper right", fontsize=8.5)
            for extra_idx in range(3, axes.shape[1]):
                axes[1, extra_idx].set_visible(False)

        return _save_dual_figure(fig, out_dir, filename_stem)
    finally:
        plt.close(fig)


def render_stage3_spectral_and_mrl_figure(
    reports: Mapping[str, DenseDatasetReport],
    output_dir: Path | str,
    filename_stem: str = "stage3_spectral_and_mrl",
) -> Dict[str, Path]:
    """Render Stage 3 active 767-d singular value spectrum, 4-regime comparison & MRL retention curve."""
    if not reports:
        return {}

    _setup_figure_style()
    out_dir = Path(output_dir)
    ds_names = list(reports.keys())
    palette = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(17.0, 5.2),
        constrained_layout=True,
    )

    try:
        # Panel 1: Active 767-d Normalized Singular Value Spectrum
        for idx, ds_name in enumerate(ds_names):
            rep = reports[ds_name]
            diag = rep.spectral_diagnostics
            s_vals = np.asarray(diag.singular_values_active, dtype=np.float64)
            s_norm = s_vals / max(1e-12, s_vals[0])
            ranks = np.arange(1, len(s_norm) + 1)
            ax1.plot(
                ranks,
                s_norm,
                linewidth=2.0,
                color=palette[idx % len(palette)],
                label=(
                    f"{ds_name.upper()} (erank={diag.effective_rank:.0f}, "
                    f"κ={diag.condition_number_active:.1f}, σ₇₆₈={diag.null_space_singular_value:.1e})"
                ),
            )
        ax1.set_yscale("log")
        ax1.set_title(
            "Stage 3A: Active 767-d Singular Value Spectrum\n(LayerNorm Hyperplane: xᵀ1₇₆₈ = 0 ⇒ σ₇₆₈ = 0)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax1.set_xlabel("Principal Component Index i ∈ [1, 767]", fontsize=9.5)
        ax1.set_ylabel("Normalized Singular Value (σᵢ / σ₁, log scale)", fontsize=9.5)
        ax1.grid(True, alpha=0.5)
        ax1.legend(loc="lower left", fontsize=8.0)

        # Panel 2: 4 Closed-Form Spectral Regimes Head-to-Head on Test
        regime_order = [
            ("raw_cosine", "Raw Cosine", "#1e293b"),
            ("split_mean_centering", "Split-Mean Center", "#0ea5e9"),
            ("split_half_whitening", "Split Half-Whiten (Σ⁻⁰·⁵)", "#10b981"),
            ("naive_svd_whitening", "Naive Full Whiten (Σ⁻¹)", "#ef4444"),
        ]
        x = np.arange(len(ds_names))
        bw = 0.19
        for r_idx, (r_key, r_lbl, r_col) in enumerate(regime_order):
            vals = [
                reports[ds].regimes[r_key].test_eval.ndcg_at_k
                if r_key in reports[ds].regimes
                else 0.0
                for ds in ds_names
            ]
            offset = (r_idx - 1.5) * bw
            ax2.bar(x + offset, vals, width=bw * 0.92, color=r_col, label=r_lbl)

        ax2.set_xticks(x)
        ax2.set_xticklabels([d.upper() for d in ds_names], fontsize=9.5, fontweight="bold")
        ax2.set_ylabel("Test NDCG@10", fontsize=9.5)
        ax2.set_title(
            "Stage 3A: Closed-Form Spectral Regimes (768-d)\n"
            "(Split Centering vs. Trailing Singular Value Noise Blow-Up)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax2.grid(True, axis="y", alpha=0.5)
        ax2.legend(loc="best", fontsize=8.0)

        # Panel 3: Matryoshka (MRL) Dimensionality Slicing Retention Curve
        for idx, ds_name in enumerate(ds_names):
            rep = reports[ds_name]
            dims = sorted(rep.mrl_slices.keys())
            ret_ndcg = [rep.mrl_slices[d].ndcg10_retention_pct for d in dims]
            ax3.plot(
                dims,
                ret_ndcg,
                marker="s",
                linewidth=2.2,
                color=palette[idx % len(palette)],
                label=f"{ds_name.upper()} NDCG@10 Retention",
            )
        ax3.axhline(100.0, color="#64748b", linestyle=":", linewidth=1.2)
        ax3.axvline(256, color="#f59e0b", linestyle="--", linewidth=1.3, label="d=256 (3× RAM/FLOPs Reduction)")
        ax3.set_xticks([64, 128, 256, 512, 768])
        ax3.set_title(
            "Stage 3B: Matryoshka (MRL) Prefix Slicing\nAccuracy Retention (% of Full d=768)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax3.set_xlabel("Matryoshka Slice Dimension (d)", fontsize=9.5)
        ax3.set_ylabel("Test NDCG@10 Retention (% of d=768)", fontsize=9.5)
        ax3.grid(True, alpha=0.5)
        ax3.legend(loc="lower right", fontsize=8.0)

        return _save_dual_figure(fig, out_dir, filename_stem)
    finally:
        plt.close(fig)


def render_stage4_calibration_and_transfer_figure(
    reports: Mapping[str, FusionDatasetReport],
    transfer_matrix: Sequence[TransferCellResult],
    output_dir: Path | str,
    filename_stem: str = "stage4_calibration_and_transfer",
) -> Dict[str, Path]:
    """Render Stage 4 alpha-sweep calibration curves, imputation paradox & 3x3 OOD transfer heatmap."""
    if not reports:
        return {}

    _setup_figure_style()
    out_dir = Path(output_dir)
    ds_names = list(reports.keys())
    n_ds = len(ds_names)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=max(n_ds, 2),
        figsize=(5.5 * max(n_ds, 2), 9.2),
        constrained_layout=True,
    )

    cal_styles = [
        ("raw", "Uncalibrated Raw", "#ef4444", "--"),
        ("minmax", "Min-Max [0, 1]", "#f59e0b", "-."),
        ("zscore", "Z-Score", "#3b82f6", "-"),
        ("zscore_sigmoid", "Z-Score + Sigmoid", "#10b981", "-"),
    ]

    try:
        # Top Row: Alpha-Sweep Curves on Test with Dev-Selected alpha* Star
        for idx, ds_name in enumerate(ds_names):
            ax = axes[0, idx]
            rep = reports[ds_name]
            rrf_ndcg = rep.rrf_test_eval_by_k0[rep.default_rrf_k0].ndcg_at_k
            ax.axhline(
                rrf_ndcg,
                color="#1e293b",
                linestyle=":",
                linewidth=2.0,
                label=f"RRF (k₀={rep.default_rrf_k0}): {rrf_ndcg:.4f}",
            )

            for cal_key, cal_lbl, cal_col, cal_ls in cal_styles:
                if cal_key not in rep.calibration_progression_4b:
                    continue
                c_rep = rep.calibration_progression_4b[cal_key]
                ax.plot(
                    c_rep.alpha_grid,
                    c_rep.test_ndcg10_curve,
                    color=cal_col,
                    linestyle=cal_ls,
                    linewidth=2.0,
                    label=f"{cal_lbl} (α*={c_rep.optimal_alpha_dev:.2f} → {c_rep.test_eval_at_opt.ndcg_at_k:.4f})",
                )
                # Mark dev-selected alpha* on the test curve
                ax.scatter(
                    [c_rep.optimal_alpha_dev],
                    [c_rep.test_eval_at_opt.ndcg_at_k],
                    color=cal_col,
                    s=65,
                    zorder=5,
                    edgecolors="black",
                    linewidths=0.8,
                )

            ax.set_title(
                f"{ds_name.upper()} — Hybrid Convex α-Sweep (Test)\n"
                "(α=0.0 Pure Dense → α=1.0 Pure Sparse)",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlabel("Sparse Weight (α)", fontsize=9.5)
            if idx == 0:
                ax.set_ylabel("Test NDCG@10", fontsize=9.5)
            ax.grid(True, alpha=0.5)
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

        for extra_idx in range(n_ds, axes.shape[1]):
            axes[0, extra_idx].set_visible(False)

        # Bottom Left: Experiment 4A Missing-Candidate Imputation Paradox (Z-Score at alpha=0.5)
        ax_imp = axes[1, 0]
        imp_order = [
            ("naive_zero", "Naive 0.0 Imputation\n(Bug: 0.0 = Top-K Mean)", "#ef4444"),
            ("topk_floor", "Top-K Cutoff Floor\n(ANN Production Default)", "#0ea5e9"),
            ("exact_dual_lookup", "Exact Dual-Lookup\n(Full Matrix Upper Bound)", "#10b981"),
        ]
        x = np.arange(len(ds_names))
        bw = 0.25
        for i_idx, (imp_key, imp_lbl, imp_col) in enumerate(imp_order):
            vals = [
                reports[ds].imputation_experiment_4a["zscore"][imp_key].alpha_fixed_05_test_ndcg10
                for ds in ds_names
            ]
            offset = (i_idx - 1.0) * bw
            ax_imp.bar(x + offset, vals, width=bw * 0.9, color=imp_col, label=imp_lbl)

        ax_imp.set_xticks(x)
        ax_imp.set_xticklabels([d.upper() for d in ds_names], fontsize=9.5, fontweight="bold")
        ax_imp.set_ylabel("Test NDCG@10 (Z-Score, α=0.50)", fontsize=9.5)
        ax_imp.set_title(
            "Stage 4A: Missing-Candidate Imputation Paradox\n"
            "(Naive `.get(id, 0.0)` vs. Cutoff Floor & Dual-Lookup)",
            fontsize=10.5,
            fontweight="bold",
        )
        ax_imp.grid(True, axis="y", alpha=0.5)
        ax_imp.legend(loc="best", fontsize=8.0)

        # Bottom Middle/Right: Experiment 4C 3x3 Out-of-Domain Zero-Shot Transfer Heatmap
        ax_ood = axes[1, 1]
        mat_size = len(ds_names)
        delta_rrf_grid = np.zeros((mat_size, mat_size), dtype=np.float64)
        alpha_map = {(c.source_dataset, c.target_dataset): c for c in transfer_matrix}

        for r_i, src in enumerate(ds_names):
            for c_i, tgt in enumerate(ds_names):
                cell = alpha_map.get((src, tgt))
                if cell is not None:
                    delta_rrf_grid[r_i, c_i] = cell.transfer_delta_vs_rrf * 100.0

        im_ood = ax_ood.imshow(delta_rrf_grid, cmap="RdYlGn", vmin=-4.0, vmax=4.0, aspect="auto")
        ax_ood.set_xticks(range(mat_size))
        ax_ood.set_xticklabels([f"Eval: {d.upper()}" for d in ds_names], fontsize=8.5, fontweight="bold")
        ax_ood.set_yticks(range(mat_size))
        ax_ood.set_yticklabels([f"Tune: {d.upper()}" for d in ds_names], fontsize=8.5, fontweight="bold")

        for r_i, src in enumerate(ds_names):
            for c_i, tgt in enumerate(ds_names):
                cell = alpha_map.get((src, tgt))
                if cell is None:
                    continue
                tag = "In-Domain" if src == tgt else "OOD Zero-Shot"
                ax_ood.text(
                    c_i,
                    r_i,
                    f"α*={cell.source_alpha_star:.2f}\n"
                    f"NDCG={cell.target_test_ndcg10_transfer:.3f}\n"
                    f"({cell.transfer_delta_vs_rrf*100:+.1f}% vs RRF)\n[{tag}]",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold" if src == tgt else "normal",
                    color="#0f172a",
                )

        ax_ood.set_title(
            "Stage 4C: Out-of-Domain α* Transfer vs. RRF (k₀=60)\n"
            "(Diagonal = In-Domain Tuned; Off-Diagonal = Domain Shift)",
            fontsize=10.5,
            fontweight="bold",
        )
        fig.colorbar(im_ood, ax=ax_ood, shrink=0.82, label="Δ NDCG@10 vs. RRF (% pts)")

        for extra_idx in range(2, axes.shape[1]):
            axes[1, extra_idx].set_visible(False)

        return _save_dual_figure(fig, out_dir, filename_stem)
    finally:
        plt.close(fig)


def render_stage5_latency_and_ceiling_figure(
    reports: Mapping[str, RerankerDatasetReport],
    output_dir: Path | str,
    filename_stem: str = "stage5_latency_and_ceiling",
) -> Dict[str, Path]:
    """Render Stage 5 NDCG@10 vs p95 Latency tradeoff & Candidate Pool Ceiling stacked breakdown."""
    if not reports:
        return {}

    _setup_figure_style()
    out_dir = Path(output_dir)
    ds_names = list(reports.keys())
    n_ds = len(ds_names)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=max(n_ds, 2),
        figsize=(5.6 * max(n_ds, 2), 9.4),
        constrained_layout=True,
    )

    family_colors = {
        "sparse_winning": "#64748b",
        "dense_mrl_256": "#f59e0b",
        "dense_full_768": "#3b82f6",
        "hybrid_rrf": "#8b5cf6",
        "hybrid_calibrated": "#10b981",
        "sparse": "#64748b",
        "hybrid": "#10b981",
    }

    try:
        # Top Row: NDCG@10 vs. p95 Latency Tradeoff per Dataset
        for idx, ds_name in enumerate(ds_names):
            ax = axes[0, idx]
            rep = reports[ds_name]

            for pt in rep.system_tradeoffs:
                col = family_colors.get(pt.pipeline_family, "#1f77b4")
                marker = "*" if pt.is_non_dominated else ("o" if pt.rerank_k == 0 else "^")
                size = 130 if pt.is_non_dominated else 65
                ax.scatter(
                    max(0.05, pt.total_p95_ms),
                    pt.ndcg10,
                    color=col,
                    marker=marker,
                    s=size,
                    edgecolors="black" if pt.is_non_dominated else "none",
                    linewidths=0.9,
                    zorder=4 if pt.is_non_dominated else 3,
                )
                if pt.is_non_dominated or pt.rerank_k in (0, 20, 100):
                    short_lbl = f"{pt.pipeline_family.replace('hybrid_', '').replace('_winning', '')}"
                    if pt.rerank_k > 0:
                        short_lbl += f"@K{pt.rerank_k}"
                    ax.annotate(
                        short_lbl,
                        (max(0.05, pt.total_p95_ms), pt.ndcg10),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7.0,
                        alpha=0.88,
                    )

            envelope_pts = sorted(
                (p for p in rep.system_tradeoffs if p.is_non_dominated),
                key=lambda p: p.total_p95_ms,
            )
            if len(envelope_pts) >= 2:
                ax.plot(
                    [max(0.05, p.total_p95_ms) for p in envelope_pts],
                    [p.ndcg10 for p in envelope_pts],
                    color="#ef4444",
                    linestyle="--",
                    linewidth=1.8,
                    label="Non-Dominated Envelope ★",
                )

            ax.set_xscale("log")
            ax.set_title(
                f"{ds_name.upper()} — NDCG@10 vs. p95 Latency Tradeoff",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlabel("Decomposed End-to-End p95 Latency (ms, log scale)", fontsize=9.5)
            if idx == 0:
                ax.set_ylabel("Test NDCG@10", fontsize=9.5)
            ax.grid(True, alpha=0.5)
            if len(envelope_pts) >= 2:
                ax.legend(loc="lower right", fontsize=8.0)

        # Bottom Row: Candidate Pool Ceiling & Error Decomposition Stacked Bars
        for idx, ds_name in enumerate(ds_names):
            ax = axes[1, idx]
            rep = reports[ds_name]

            bar_labels = []
            reranked_vals = []
            scoring_err_vals = []
            pool_miss_vals = []

            # Show Sparse vs Calibrated Hybrid across K in candidate_ks
            target_pipes = [
                p for p in ("sparse_winning", "dense_full_768", "hybrid_calibrated", "sparse", "hybrid")
                if p in rep.ceiling_decompositions
            ]
            for p_name in target_pipes:
                k_map = rep.ceiling_decompositions[p_name]
                short_p = (
                    p_name.replace("sparse_winning", "Sparse")
                    .replace("dense_full_768", "Dense768")
                    .replace("hybrid_calibrated", "CalHybrid")
                )
                for k_val, pt in sorted(k_map.items()):
                    bar_labels.append(f"{short_p}\nK={k_val}")
                    reranked_vals.append(pt.reranked_ndcg10)
                    scoring_err_vals.append(pt.reranker_scoring_error)
                    pool_miss_vals.append(pt.candidate_pool_miss_error)

            x = np.arange(len(bar_labels))
            r_arr = np.array(reranked_vals)
            s_arr = np.array(scoring_err_vals)
            m_arr = np.array(pool_miss_vals)

            ax.bar(x, r_arr, color="#10b981", label="Actual Reranked NDCG@10")
            ax.bar(x, s_arr, bottom=r_arr, color="#f59e0b", label="Stage-2 Reranker Scoring Error")
            ax.bar(
                x,
                m_arr,
                bottom=r_arr + s_arr,
                color="#ef4444",
                alpha=0.75,
                label="Stage-1 Candidate Pool Misses",
            )
            ax.axhline(
                rep.max_corpus_ndcg10,
                color="#0f172a",
                linestyle=":",
                linewidth=1.5,
                label=f"MaxCorpusNDCG@10 ({rep.max_corpus_ndcg10:.4f})",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(bar_labels, fontsize=7.5)
            ax.set_ylim(0.0, 1.08)
            ax.set_title(
                f"{ds_name.upper()} — Candidate Pool Ceiling Decomposition\n"
                "(Reranked + Reranker Error + Stage-1 Pool Misses = MaxCorpusNDCG)",
                fontsize=10.0,
                fontweight="bold",
            )
            if idx == 0:
                ax.set_ylabel("NDCG@10 (Full-Qrels IDCG Normalized)", fontsize=9.5)
            ax.grid(True, axis="y", alpha=0.5)
            if idx == 0:
                ax.legend(loc="lower right", fontsize=7.5, framealpha=0.92)

        for extra_idx in range(n_ds, axes.shape[1]):
            axes[0, extra_idx].set_visible(False)
            axes[1, extra_idx].set_visible(False)

        return _save_dual_figure(fig, out_dir, filename_stem)
    finally:
        plt.close(fig)
