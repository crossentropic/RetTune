"""Rich console table rendering for RetTune benchmark diagnostics (Stages 1 to 5)."""

from typing import Dict, Mapping, Sequence
from rich.console import Console
from rich.table import Table

from ..dense import DenseDatasetReport
from ..eda import LexicalProfile
from ..fusion import FusionDatasetReport, TransferCellResult
from ..reranker import RerankerDatasetReport
from ..sparse import SparseDatasetReport

console = Console()


def display_length_summary_table(profiles: Dict[str, LexicalProfile]) -> None:
    """Render formatted console table summarizing document and query length dynamics."""
    table = Table(
        title="[bold cyan]RetTune Stage 1: Corpus & Query Length Dynamics[/bold cyan]",
        title_justify="left",
    )
    table.add_column("Dataset", style="bold white")
    table.add_column("Unit", style="cyan")
    table.add_column("Items (N)", justify="right")
    table.add_column("Mean (w)", justify="right")
    table.add_column("Med (w)", justify="right")
    table.add_column("IQR (w)", justify="right")
    table.add_column("p95 (w)", justify="right")
    table.add_column("p99 (w)", justify="right")
    table.add_column("Max (w)", justify="right")
    table.add_column("Skewness (g₁)", justify="right")

    for ds_name, prof in profiles.items():
        units = [
            ("Passages", prof.doc_summary),
            ("Queries (Dev)", prof.query_summary_dev),
            ("Queries (Test)", prof.query_summary_test),
            ("Queries (All)", prof.query_summary_all),
        ]
        for idx, (unit_label, summary) in enumerate(units):
            ds_col = ds_name.upper() if idx == 0 else ""
            skew_style = "bold red" if summary.skewness > 2.0 else "green"
            table.add_row(
                ds_col,
                unit_label,
                f"{summary.count:,}",
                f"{summary.mean:.1f}",
                f"{summary.median:.1f}",
                f"{summary.iqr:.1f}",
                f"{summary.percentiles.get('p95', 0.0):.1f}",
                f"{summary.percentiles.get('p99', summary.max):.1f}",
                f"{summary.max:.1f}",
                f"[{skew_style}]{summary.skewness:.2f}[/{skew_style}]",
            )
        table.add_section()

    console.print(table)


def display_coverage_summary_table(profiles: Dict[str, LexicalProfile]) -> None:
    """Render formatted console table contrasting relevant vs. background coverage."""
    table = Table(
        title="[bold cyan]RetTune Stage 1: IDF-Weighted Query Coverage & Separation[/bold cyan]",
        title_justify="left",
    )
    table.add_column("Dataset", style="bold white")
    table.add_column("Rel Pairs", justify="right")
    table.add_column("Rel Mean (%)", justify="right")
    table.add_column("Rel Med (%)", justify="right")
    table.add_column("Noise Med (%)", justify="right")
    table.add_column("Δ Med (%)", justify="right", style="bold yellow")
    table.add_column("Cohen's d (σ)", justify="right", style="bold green")
    table.add_column("Wasserstein (W₁)", justify="right", style="bold magenta")

    for ds_name, prof in profiles.items():
        rel = prof.coverage_summary_relevant_all
        bg = prof.coverage_summary_random
        sep = prof.separation_all

        table.add_row(
            ds_name.upper(),
            f"{rel.count:,}",
            f"{rel.mean * 100:.1f}",
            f"{rel.median * 100:.1f}",
            f"{bg.median * 100:.1f}",
            f"{sep.delta_median * 100:+.1f}",
            f"{sep.cohens_d:.2f}",
            f"{sep.wasserstein_distance:.3f}",
        )

    console.print(table)


def display_sparse_summary_table(reports: Mapping[str, SparseDatasetReport]) -> None:
    """Render Stage 2 sparse BM25Okapi vs BM25+ benchmark summary table."""
    table = Table(
        title="[bold cyan]RetTune Stage 2: Sparse Term-Saturation & Length-Normalization (Test Split)[/bold cyan]",
        title_justify="left",
    )
    table.add_column("Dataset", style="bold white")
    table.add_column("Configuration", style="cyan")
    table.add_column("Preset / (k₁, b, δ)", justify="left")
    table.add_column("Dev NDCG@10", justify="right")
    table.add_column("Test NDCG@10", justify="right", style="bold green")
    table.add_column("Test MRR@10", justify="right")
    table.add_column("Test Recall@100", justify="right", style="bold yellow")
    table.add_column("Δ vs Default (p-val)", justify="right")
    table.add_column("p95 Lat (ms)", justify="right")

    for ds_name, rep in reports.items():
        rows = [
            (
                "Default BM25Okapi",
                rep.default_okapi_params,
                rep.dev_default_okapi_eval,
                rep.test_default_okapi_eval,
                "—",
            ),
            (
                "No-Stem BM25Okapi",
                rep.default_nostem_params,
                rep.dev_default_nostem_eval,
                rep.test_default_nostem_eval,
                f"{rep.test_nostem_vs_standard_sig.delta:+.4f} (p={rep.test_nostem_vs_standard_sig.p_value:.3f})",
            ),
            (
                "Tuned BM25Okapi (δ=0)",
                rep.best_okapi_params,
                rep.dev_best_okapi_eval,
                rep.test_best_okapi_eval,
                f"{rep.test_best_okapi_eval.ndcg_at_k - rep.test_default_okapi_eval.ndcg_at_k:+.4f}",
            ),
            (
                "Tuned BM25+ (δ>0)",
                rep.best_bm25_plus_params,
                rep.dev_best_bm25_plus_eval,
                rep.test_best_bm25_plus_eval,
                f"{rep.test_bm25_plus_vs_default_sig.delta:+.4f} (p={rep.test_bm25_plus_vs_default_sig.p_value:.3f})",
            ),
            (
                "Winning Sparse ★",
                rep.winning_sparse_params,
                rep.dev_winning_eval,
                rep.test_winning_eval,
                f"{rep.test_winning_vs_default_sig.delta:+.4f} (p={rep.test_winning_vs_default_sig.p_value:.3f})",
            ),
        ]
        for idx, (label, params, dev_ev, test_ev, sig_str) in enumerate(rows):
            ds_col = ds_name.upper() if idx == 0 else ""
            lat_str = f"{rep.test_latency.p95_ms:.2f}" if idx == len(rows) - 1 else ""
            table.add_row(
                ds_col,
                label,
                f"{params.preset} ({params.k1:.1f}, {params.b:.2f}, {params.delta:.2f})",
                f"{dev_ev.ndcg_at_k:.4f}",
                f"{test_ev.ndcg_at_k:.4f}",
                f"{test_ev.mrr_at_k:.4f}",
                f"{test_ev.recall_at_k.get(100, 0.0):.4f}",
                sig_str,
                lat_str,
            )
        table.add_section()

    console.print(table)


def display_dense_summary_table(reports: Mapping[str, DenseDatasetReport]) -> None:
    """Render Stage 3 spectral regimes and Matryoshka dimensionality tables."""
    t_spec = Table(
        title="[bold cyan]RetTune Stage 3A: 768-d Spectral Geometry & Closed-Form Regimes (Test Split)[/bold cyan]",
        title_justify="left",
    )
    t_spec.add_column("Dataset", style="bold white")
    t_spec.add_column("erank / κ₇₆₇ / ||μ_Q-μ_D||", justify="left")
    t_spec.add_column("Regime", style="cyan")
    t_spec.add_column("Dev NDCG@10", justify="right")
    t_spec.add_column("Test NDCG@10", justify="right", style="bold green")
    t_spec.add_column("Test Recall@100", justify="right", style="bold yellow")
    t_spec.add_column("Δ vs Raw (p-val)", justify="right")

    for ds_name, rep in reports.items():
        diag = rep.spectral_diagnostics
        diag_summary = (
            f"erank={diag.effective_rank:.1f}, κ={diag.condition_number_active:.1f}, "
            f"shift={diag.centroid_l2_distance:.3f}"
        )
        for idx, (r_name, r_res) in enumerate(rep.regimes.items()):
            ds_col = ds_name.upper() if idx == 0 else ""
            d_col = diag_summary if idx == 0 else ""
            sig = r_res.test_vs_raw_sig
            sig_str = "—" if r_name == "raw_cosine" else f"{sig.delta:+.4f} (p={sig.p_value:.3f})"
            t_spec.add_row(
                ds_col,
                d_col,
                r_res.display_label,
                f"{r_res.dev_eval.ndcg_at_k:.4f}",
                f"{r_res.test_eval.ndcg_at_k:.4f}",
                f"{r_res.test_eval.recall_at_k.get(100, 0.0):.4f}",
                sig_str,
            )
        t_spec.add_section()

    console.print(t_spec)
    console.print()

    t_mrl = Table(
        title="[bold cyan]RetTune Stage 3B: Matryoshka (MRL) Prefix Slicing & Compute Tradeoffs[/bold cyan]",
        title_justify="left",
    )
    t_mrl.add_column("Dataset", style="bold white")
    t_mrl.add_column("Dim (d)", justify="right", style="cyan")
    t_mrl.add_column("Test NDCG@10", justify="right", style="bold green")
    t_mrl.add_column("NDCG Ret (%)", justify="right")
    t_mrl.add_column("Test Recall@100", justify="right", style="bold yellow")
    t_mrl.add_column("RAM (MB)", justify="right")
    t_mrl.add_column("FLOPs/q", justify="right")
    t_mrl.add_column("Scan p95 (ms)", justify="right")

    for ds_name, rep in reports.items():
        for idx, (dim, m_res) in enumerate(sorted(rep.mrl_slices.items())):
            ds_col = ds_name.upper() if idx == 0 else ""
            t_mrl.add_row(
                ds_col,
                f"{dim}",
                f"{m_res.test_eval.ndcg_at_k:.4f}",
                f"{m_res.ndcg10_retention_pct:.1f}%",
                f"{m_res.test_eval.recall_at_k.get(100, 0.0):.4f}",
                f"{m_res.corpus_ram_mb:.2f}",
                f"{m_res.flops_per_query:,}",
                f"{m_res.scan_latency_test.p95_ms:.3f}",
            )
        t_mrl.add_section()

    console.print(t_mrl)


def display_fusion_summary_table(
    reports: Mapping[str, FusionDatasetReport],
    transfer_matrix: Sequence[TransferCellResult],
) -> None:
    """Render Stage 4 hybrid calibration, imputation, and OOD transfer tables."""
    table = Table(
        title="[bold cyan]RetTune Stage 4: Score Calibration & Hybrid Fusion Progression (Test Split)[/bold cyan]",
        title_justify="left",
    )
    table.add_column("Dataset", style="bold white")
    table.add_column("Fusion Method", style="cyan")
    table.add_column("α* (Dev)", justify="right")
    table.add_column("Dev NDCG@10", justify="right")
    table.add_column("Test NDCG@10", justify="right", style="bold green")
    table.add_column("Test Recall@100", justify="right", style="bold yellow")
    table.add_column("Δ vs RRF (p-val)", justify="right")

    for ds_name, rep in reports.items():
        rrf_dev = rep.rrf_dev_eval_by_k0[rep.default_rrf_k0]
        rrf_test = rep.rrf_test_eval_by_k0[rep.default_rrf_k0]

        table.add_row(
            ds_name.upper(),
            "Pure Sparse (Stage 2 Winner)",
            "1.00",
            f"{rep.pure_sparse_dev_eval.ndcg_at_k:.4f}",
            f"{rep.pure_sparse_test_eval.ndcg_at_k:.4f}",
            f"{rep.pure_sparse_test_eval.recall_at_k.get(100, 0.0):.4f}",
            f"{rep.pure_sparse_test_eval.ndcg_at_k - rrf_test.ndcg_at_k:+.4f}",
        )
        table.add_row(
            "",
            "Pure Dense (Stage 3 Winner)",
            "0.00",
            f"{rep.pure_dense_dev_eval.ndcg_at_k:.4f}",
            f"{rep.pure_dense_test_eval.ndcg_at_k:.4f}",
            f"{rep.pure_dense_test_eval.recall_at_k.get(100, 0.0):.4f}",
            f"{rep.pure_dense_test_eval.ndcg_at_k - rrf_test.ndcg_at_k:+.4f}",
        )
        table.add_row(
            "",
            f"RRF Baseline (k₀={rep.default_rrf_k0})",
            "—",
            f"{rrf_dev.ndcg_at_k:.4f}",
            f"{rrf_test.ndcg_at_k:.4f}",
            f"{rrf_test.recall_at_k.get(100, 0.0):.4f}",
            "—",
        )
        for cal_name, cal_rep in rep.calibration_progression_4b.items():
            sig = cal_rep.test_vs_rrf_sig
            table.add_row(
                "",
                f"Convex ({cal_name})",
                f"{cal_rep.optimal_alpha_dev:.2f}",
                f"{cal_rep.dev_eval_at_opt.ndcg_at_k:.4f}",
                f"{cal_rep.test_eval_at_opt.ndcg_at_k:.4f}",
                f"{cal_rep.test_eval_at_opt.recall_at_k.get(100, 0.0):.4f}",
                f"{sig.delta:+.4f} (p={sig.p_value:.3f})",
            )
        table.add_section()

    console.print(table)

    if transfer_matrix:
        cal_lbl = transfer_matrix[0].calibration_method
        console.print()
        t_ood = Table(
            title=f"[bold cyan]RetTune Stage 4C: Out-of-Domain Zero-Shot α* Transfer vs. RRF ({cal_lbl})[/bold cyan]",
            title_justify="left",
        )
        t_ood.add_column("Source (α* Tuned)", style="bold white")
        t_ood.add_column("Target (Test Eval)", style="cyan")
        t_ood.add_column("Source α*", justify="right")
        t_ood.add_column("Target α*", justify="right")
        t_ood.add_column("Transfer NDCG@10", justify="right")
        t_ood.add_column("In-Domain NDCG@10", justify="right", style="bold green")
        t_ood.add_column("RRF NDCG@10", justify="right")
        t_ood.add_column("OOD Gap (In - Trans)", justify="right", style="bold yellow")
        t_ood.add_column("Trans - RRF", justify="right")

        for cell in transfer_matrix:
            t_ood.add_row(
                cell.source_dataset.upper(),
                cell.target_dataset.upper(),
                f"{cell.source_alpha_star:.2f}",
                f"{cell.target_in_domain_alpha_star:.2f}",
                f"{cell.target_test_ndcg10_transfer:.4f}",
                f"{cell.target_test_ndcg10_in_domain:.4f}",
                f"{cell.target_test_ndcg10_rrf:.4f}",
                f"{cell.transfer_gap_vs_in_domain:+.4f}",
                f"{cell.transfer_delta_vs_rrf:+.4f}",
            )
        console.print(t_ood)


def display_reranker_summary_table(reports: Mapping[str, RerankerDatasetReport]) -> None:
    """Render Stage 5 Candidate Pool Ceiling decomposition and latency/compute tradeoff tables."""
    table = Table(
        title="[bold cyan]RetTune Stage 5: Candidate Pool Ceiling & Reranker Error Decomposition[/bold cyan]",
        title_justify="left",
    )
    table.add_column("Dataset", style="bold white")
    table.add_column("Stage-1 Pipeline", style="cyan")
    table.add_column("Pool K", justify="right")
    table.add_column("Stage-1 NDCG@10", justify="right")
    table.add_column("Reranked NDCG@10", justify="right", style="bold green")
    table.add_column("Ceiling@K", justify="right", style="bold yellow")
    table.add_column("Reranker Err", justify="right")
    table.add_column("Pool Miss Err", justify="right", style="bold red")
    table.add_column("MaxCorpusNDCG", justify="right")

    for ds_name, rep in reports.items():
        first_row = True
        for pipe_name, k_map in rep.ceiling_decompositions.items():
            for k_val, pt in sorted(k_map.items()):
                table.add_row(
                    ds_name.upper() if first_row else "",
                    pt.pipeline_label,
                    f"{k_val}",
                    f"{pt.first_stage_ndcg10:.4f}",
                    f"{pt.reranked_ndcg10:.4f}",
                    f"{pt.ceiling_ndcg10:.4f}",
                    f"{pt.reranker_scoring_error:.4f}",
                    f"{pt.candidate_pool_miss_error:.4f}",
                    f"{pt.max_corpus_ndcg10:.4f}",
                )
                first_row = False
        table.add_section()

    console.print(table)
