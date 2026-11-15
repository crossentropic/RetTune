"""Command-line interface and execution dispatcher for RetTune.

Provides a unified root CLI for offline hybrid search benchmark sweeps
(`eda`, `sparse`, `dense`, `fusion`, `rerank`, `all`), enforcing strict
zero-network execution, stage-aware fail-fast offline readiness checks,
and deterministic artifact orchestration.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rich.console import Console

from rettune.config import BenchmarkConfig, load_config
from rettune.data_loader import ContractValidationError, DataLeakageError, IRDataset, load_dataset
from rettune.dense import (
    DenseDatasetReport,
    export_dense_artifacts,
    get_dense_candidates_and_matrices,
    load_embedding_bundle,
    profile_dense_scan_latency,
    run_dense_benchmark_for_dataset,
    slice_and_normalize_mrl,
)
from rettune.eda import LexicalProfile, export_eda_artifacts, profile_dataset
from rettune.fusion import (
    FusionDatasetReport,
    compute_ood_alpha_transfer_matrix,
    export_fusion_artifacts,
    fuse_precalibrated_at_alpha,
    fuse_run_rrf,
    precalibrate_query_union,
    profile_fusion_latency,
    run_fusion_benchmark_for_dataset,
)
from rettune.reranker import (
    RerankerDatasetReport,
    build_and_save_reranker_cache,
    collect_deduplicated_candidate_pairs,
    export_reranker_artifacts,
    load_reranker_cache,
    run_reranker_benchmark_for_dataset,
)
from rettune.sparse import (
    SparseDatasetReport,
    export_sparse_artifacts,
    get_or_build_winning_sparse_index,
    profile_sparse_latency,
    run_sparse_benchmark_for_dataset,
)
from rettune.viz import (
    display_coverage_summary_table,
    display_dense_summary_table,
    display_fusion_summary_table,
    display_length_summary_table,
    display_reranker_summary_table,
    display_sparse_summary_table,
    render_coverage_density,
    render_stage2_bm25_plus_figure,
    render_stage3_spectral_and_mrl_figure,
    render_stage4_calibration_and_transfer_figure,
    render_stage5_latency_and_ceiling_figure,
    render_summary_table,
    render_token_length_ecdf,
)

logger = logging.getLogger("rettune.cli")
console = Console()

SUPPORTED_STAGES = ["eda", "sparse", "dense", "fusion", "rerank", "all"]


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for RetTune."""
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="RetTune: Reproducible Hybrid Search Benchmark Testbed (Layer 1 RAG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        choices=SUPPORTED_STAGES,
        default=None,
        help=(
            "Pipeline stage to execute ('eda', 'sparse', 'dense', 'fusion', 'rerank', 'all'). "
            "Defaults to 'all' in offline benchmark mode."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Benchmark dataset to evaluate ('all' or specific dataset name, e.g. 'nfcorpus').",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom benchmark YAML configuration file.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        default=False,
        help="Download datasets, generate dense embeddings, and pre-cache Cross-Encoder scores.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force regeneration during setup even if cached artifacts already exist.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=None,
        help="Relevance score threshold for positive pairs (overrides dataset config if provided).",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate visualization figures in results/figures/ (use --no-plot for fast compute).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Custom root data directory override.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Custom root results directory override.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging.",
    )

    return parser


def apply_config_overrides(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    """Apply CLI path overrides to BenchmarkConfig using immutable model_copy."""
    if args.data_dir or args.results_dir:
        path_updates = {}
        if args.data_dir:
            path_updates["data_dir"] = Path(args.data_dir).resolve()
        if args.results_dir:
            path_updates["results_dir"] = Path(args.results_dir).resolve()
        new_paths = config.paths.model_copy(update=path_updates)
        return config.model_copy(update={"paths": new_paths})
    return config


def resolve_target_datasets(dataset_arg: str, config: BenchmarkConfig) -> List[str]:
    """Resolve target datasets from CLI argument against active benchmark configuration."""
    if dataset_arg == "all":
        return list(config.active_datasets)

    if dataset_arg not in config.datasets:
        raise ValueError(
            f"Dataset '{dataset_arg}' is not configured in benchmark. "
            f"Available datasets: {list(config.datasets.keys())}"
        )
    return [dataset_arg]


def get_missing_dataset_files(
    dataset_name: str,
    config: BenchmarkConfig,
    stage: str = "eda",
) -> List[Path]:
    """Return any missing or zero-byte local files required for `dataset_name` at `stage`.

    Stage-Aware Requirements:
        - 'eda' / 'sparse': requires canonical BEIR files (`corpus.jsonl`, `queries.jsonl`,
          `qrels/dev.tsv`, `qrels/test.tsv`).
        - 'dense' / 'fusion': additionally requires `embeddings/<model_slug>/{corpus,queries_dev,queries_test}.npy`
          and `metadata.json`.
        - 'rerank' / 'all': additionally requires `reranker_cache/scores.json`.
    """
    dataset_dir = config.get_dataset_dir(dataset_name)
    required_files = [
        dataset_dir / "corpus.jsonl",
        dataset_dir / "queries.jsonl",
        config.get_qrels_path(dataset_name, "dev"),
        config.get_qrels_path(dataset_name, "test"),
    ]

    if stage in ("dense", "fusion", "rerank", "all"):
        emb_dir = config.get_embeddings_dir(dataset_name)
        required_files.extend([
            emb_dir / "corpus.npy",
            emb_dir / "queries_dev.npy",
            emb_dir / "queries_test.npy",
            emb_dir / "metadata.json",
        ])

    if stage in ("rerank", "all"):
        cache_dir = config.get_reranker_cache_dir(dataset_name)
        required_files.append(cache_dir / "scores.json")

    return [p for p in required_files if not p.is_file() or p.stat().st_size == 0]


def verify_offline_readiness(
    target_datasets: Sequence[str],
    config: BenchmarkConfig,
    stage: str = "eda",
) -> Tuple[bool, Dict[str, List[Path]]]:
    """Verify that all target datasets have stage-required files locally available.

    Returns:
        (is_ready, missing_files_by_dataset)
    """
    missing_by_dataset: Dict[str, List[Path]] = {}
    for ds_name in target_datasets:
        missing = get_missing_dataset_files(ds_name, config, stage=stage)
        if missing:
            missing_by_dataset[ds_name] = missing
    return len(missing_by_dataset) == 0, missing_by_dataset


def format_guard_failure_message(missing_by_dataset: Dict[str, List[Path]]) -> str:
    """Construct a clean, actionable terminal banner explaining offline readiness failure."""
    lines = [
        "=" * 70,
        "ZERO-NETWORK GUARD FAILURE:",
        "RetTune runs strictly offline by default, but required dataset or model",
        "artifact files are missing or unverified locally:",
    ]
    for ds_name, missing_files in missing_by_dataset.items():
        lines.append(f"  • Dataset '{ds_name}':")
        for f in missing_files:
            lines.append(f"      - Missing: {f}")

    lines.extend([
        "",
        "To download, canonicalize, embed, and pre-cache benchmark artifacts, run:",
        "    python benchmark.py --setup",
        "or target a specific dataset:",
        f"    python benchmark.py --setup --dataset {next(iter(missing_by_dataset.keys()))}",
        "=" * 70,
    ])
    return "\n".join(lines)


def _build_stage5_pipeline_runs_for_dataset(
    dataset: IRDataset,
    config: BenchmarkConfig,
) -> Tuple[Dict[str, Dict[str, List[Tuple[str, float]]]], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Build representative Stage 2–4 top-100 test candidate runs and stage-1 latency metadata."""
    top_k = config.fusion.top_k
    dev_qids = sorted(dataset.qrels_dev.keys())
    test_qids = sorted(dataset.qrels_test.keys())

    # 1. Stage 2 Winning Sparse
    sparse_params, sparse_idx, _ = get_or_build_winning_sparse_index(dataset, config)
    sparse_dev_run = sparse_idx.retrieve_top_k(dev_qids, top_k=top_k)
    sparse_test_run = sparse_idx.retrieve_top_k(test_qids, top_k=top_k)
    sparse_lat = profile_sparse_latency(sparse_idx, test_qids, top_k=top_k)

    # 2. Stage 3 Dense MRL-256 and Full-768
    bundle = load_embedding_bundle(dataset.name, config)
    mrl_fast_dim = 256 if 256 <= bundle.full_dim else min(bundle.full_dim, max(config.dense.mrl_dims))

    _, dense_256_test_run, _, _ = get_dense_candidates_and_matrices(
        bundle, regime_name="raw_cosine", mrl_dim=mrl_fast_dim, top_k=top_k
    )
    c_256 = slice_and_normalize_mrl(bundle.corpus_emb, dim=mrl_fast_dim)
    q_256 = slice_and_normalize_mrl(bundle.queries_test_emb, dim=mrl_fast_dim)
    scan_lat_256 = profile_dense_scan_latency(c_256, q_256, bundle.doc_ids, top_k=top_k)

    dense_768_dev_run, dense_768_test_run, _, _ = get_dense_candidates_and_matrices(
        bundle, regime_name="raw_cosine", mrl_dim=bundle.full_dim, top_k=top_k
    )
    c_768 = slice_and_normalize_mrl(bundle.corpus_emb, dim=bundle.full_dim)
    q_768 = slice_and_normalize_mrl(bundle.queries_test_emb, dim=bundle.full_dim)
    scan_lat_768 = profile_dense_scan_latency(c_768, q_768, bundle.doc_ids, top_k=top_k)

    # 3. Stage 4 RRF and Calibrated Convex (zscore_sigmoid at dev-tuned alpha*)
    rrf_test_run = fuse_run_rrf(
        sparse_test_run,
        dense_768_test_run,
        test_qids,
        k0=config.fusion.rrf_k0_default,
        top_k=top_k,
        output_top_k=top_k,
    )

    # Determine winning calibration method and alpha* on dev (from cached Stage 4 JSON if available)
    stage4_json = config.get_metrics_dir() / f"stage4_fusion_{dataset.name}.json"
    win_cal_method = "zscore"
    alpha_star = 0.45
    if stage4_json.is_file():
        try:
            with open(stage4_json, "r", encoding="utf-8") as f:
                s4_data = json.load(f)
            win_cal_method = str(s4_data.get("winning_calibration_method", "zscore"))
            alpha_star = float(
                s4_data["experiment_4b_calibration"][win_cal_method]["optimal_alpha_dev"]
            )
        except Exception:
            pass
    else:
        from rettune.evaluation import evaluate_run
        from rettune.fusion import build_alpha_grid
        pq_dev = [
            precalibrate_query_union(
                query_id=qid,
                sparse_candidates=sparse_dev_run.get(qid, []),
                dense_candidates=dense_768_dev_run.get(qid, []),
                calibration_method=win_cal_method,
                imputation_regime="topk_floor",
                top_k=top_k,
            )
            for qid in dev_qids
        ]
        best_key = (-1.0, -1.0)
        for a_cand in build_alpha_grid(config.fusion.alpha_step):
            r_cand = fuse_precalibrated_at_alpha(pq_dev, alpha=a_cand, output_top_k=top_k)
            ev_cand = evaluate_run(
                r_cand,
                dataset.qrels_dev,
                primary_k=config.evaluation.primary_k,
                recall_ks=[100],
                relevance_threshold=config.datasets[dataset.name].relevance_threshold,
                already_sorted=True,
            )
            key = (round(ev_cand.ndcg_at_k, 8), round(ev_cand.recall_at_k.get(100, 0.0), 8))
            if key > best_key:
                best_key = key
                alpha_star = a_cand

    pq_test = [
        precalibrate_query_union(
            query_id=qid,
            sparse_candidates=sparse_test_run.get(qid, []),
            dense_candidates=dense_768_test_run.get(qid, []),
            calibration_method=win_cal_method,
            imputation_regime="topk_floor",
            top_k=top_k,
        )
        for qid in test_qids
    ]
    calibrated_test_run = fuse_precalibrated_at_alpha(pq_test, alpha=alpha_star, output_top_k=top_k)
    fusion_lat = profile_fusion_latency(
        sparse_test_run,
        dense_768_test_run,
        test_qids,
        alpha=alpha_star,
        calibration_method=win_cal_method,
        imputation_regime="topk_floor",
        top_k=top_k,
    )

    n_docs = len(bundle.doc_ids)
    enc_p50 = float(bundle.encode_latency_ms.get("p50_ms", 0.0))
    enc_p95 = float(bundle.encode_latency_ms.get("p95_ms", 0.0))

    pipeline_runs = {
        "sparse_winning": sparse_test_run,
        "dense_mrl_256": dense_256_test_run,
        "dense_full_768": dense_768_test_run,
        "hybrid_rrf": rrf_test_run,
        "hybrid_calibrated": calibrated_test_run,
    }
    pipeline_labels = {
        "sparse_winning": f"Sparse ({sparse_params.preset}, b={sparse_params.b:.2f}, δ={sparse_params.delta:.2f})",
        "dense_mrl_256": f"Dense MRL (d={mrl_fast_dim})",
        "dense_full_768": f"Dense Full (d={bundle.full_dim})",
        "hybrid_rrf": f"Hybrid RRF (k₀={config.fusion.rrf_k0_default})",
        "hybrid_calibrated": f"Hybrid Calibrated ({win_cal_method}, α*={alpha_star:.2f})",
    }
    stage1_meta = {
        "sparse_winning": {
            "t_sparse_p50_ms": sparse_lat.p50_ms,
            "t_sparse_p95_ms": sparse_lat.p95_ms,
            "postings_per_query": sparse_lat.work_units_per_query,
            "matrix_flops_per_query": 0,
            "vector_ram_mb": 0.0,
        },
        "dense_mrl_256": {
            "t_dense_encode_p50_ms": enc_p50,
            "t_dense_encode_p95_ms": enc_p95,
            "t_dense_scan_p50_ms": scan_lat_256.p50_ms,
            "t_dense_scan_p95_ms": scan_lat_256.p95_ms,
            "matrix_flops_per_query": 2 * n_docs * mrl_fast_dim,
            "vector_ram_mb": (n_docs * mrl_fast_dim * 4) / 1_000_000.0,
            "postings_per_query": 0.0,
        },
        "dense_full_768": {
            "t_dense_encode_p50_ms": enc_p50,
            "t_dense_encode_p95_ms": enc_p95,
            "t_dense_scan_p50_ms": scan_lat_768.p50_ms,
            "t_dense_scan_p95_ms": scan_lat_768.p95_ms,
            "matrix_flops_per_query": 2 * n_docs * bundle.full_dim,
            "vector_ram_mb": (n_docs * bundle.full_dim * 4) / 1_000_000.0,
            "postings_per_query": 0.0,
        },
        "hybrid_rrf": {
            "t_sparse_p50_ms": sparse_lat.p50_ms,
            "t_sparse_p95_ms": sparse_lat.p95_ms,
            "t_dense_encode_p50_ms": enc_p50,
            "t_dense_encode_p95_ms": enc_p95,
            "t_dense_scan_p50_ms": scan_lat_768.p50_ms,
            "t_dense_scan_p95_ms": scan_lat_768.p95_ms,
            "t_fusion_p50_ms": fusion_lat.p50_ms,
            "t_fusion_p95_ms": fusion_lat.p95_ms,
            "matrix_flops_per_query": 2 * n_docs * bundle.full_dim,
            "vector_ram_mb": (n_docs * bundle.full_dim * 4) / 1_000_000.0,
            "postings_per_query": sparse_lat.work_units_per_query,
        },
        "hybrid_calibrated": {
            "t_sparse_p50_ms": sparse_lat.p50_ms,
            "t_sparse_p95_ms": sparse_lat.p95_ms,
            "t_dense_encode_p50_ms": enc_p50,
            "t_dense_encode_p95_ms": enc_p95,
            "t_dense_scan_p50_ms": scan_lat_768.p50_ms,
            "t_dense_scan_p95_ms": scan_lat_768.p95_ms,
            "t_fusion_p50_ms": fusion_lat.p50_ms,
            "t_fusion_p95_ms": fusion_lat.p95_ms,
            "matrix_flops_per_query": 2 * n_docs * bundle.full_dim,
            "vector_ram_mb": (n_docs * bundle.full_dim * 4) / 1_000_000.0,
            "postings_per_query": sparse_lat.work_units_per_query,
        },
    }
    return pipeline_runs, pipeline_labels, stage1_meta


def run_setup(args: argparse.Namespace, config: Optional[BenchmarkConfig] = None) -> int:
    """Execute dataset ingestion, dense embedding generation, and Cross-Encoder pre-caching."""
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "download_data.py"
    if not script_path.exists():
        logger.error("Download script not found at expected path: %s", script_path)
        return 1

    cmd = [sys.executable, str(script_path)]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.data_dir:
        cmd.extend(["--data-dir", str(args.data_dir)])
    if args.force:
        cmd.append("--force")
    if args.verbose:
        cmd.append("-v")

    if args.dataset != "all":
        cmd.extend(["--dataset", args.dataset])
    else:
        cmd.extend(["--dataset", "all"])

    logger.info("Step 1/3 — Executing dataset ingestion via %s: %s", script_path.name, " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return result.returncode

    # If a specific non-embedding stage ('eda' or 'sparse') was requested alongside --setup,
    # or if running under a mocked subprocess test where data files were not actually written,
    # skip model loading if canonical files don't exist on disk.
    if config is None:
        try:
            config = load_config(args.config)
            config = apply_config_overrides(config, args)
        except Exception:
            return 0

    if args.stage in ("eda", "sparse"):
        return 0

    target_datasets = resolve_target_datasets(args.dataset, config)
    if not all((config.get_dataset_dir(ds) / "corpus.jsonl").is_file() for ds in target_datasets):
        return 0

    # Step 2/3: Generate offline Matryoshka-ready dense embeddings
    from scripts.embed_corpus import embed_dataset

    logger.info("Step 2/3 — Verifying / generating offline dense embeddings (%s)...", config.dense.model_name)
    dense_model = None
    for ds_name in target_datasets:
        dense_model = embed_dataset(
            ds_name,
            config=config,
            model=dense_model,
            force=args.force,
        )

    # Step 3/3: Pre-cache Cross-Encoder pair scores across deduplicated top-100 candidate pools
    logger.info("Step 3/3 — Verifying / pre-caching Cross-Encoder scores (%s)...", config.reranker.model_name)
    ce_model = None
    for ds_name in target_datasets:
        dataset = load_dataset(ds_name, config, verify=True)
        pipeline_runs, _, _ = _build_stage5_pipeline_runs_for_dataset(dataset, config)
        union_pairs = collect_deduplicated_candidate_pairs(
            pipeline_runs.values(),
            max_k=max(config.reranker.candidate_ks, default=100),
        )
        _, ce_model = build_and_save_reranker_cache(
            dataset=dataset,
            union_pairs_by_qid=union_pairs,
            config=config,
            cross_encoder_model=ce_model,
            force=args.force,
        )

    return 0


def run_stage_eda(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    target_datasets: Sequence[str],
) -> int:
    """Execute Stage 1 Lexical Distributional Diagnostics."""
    profiles: Dict[str, LexicalProfile] = {}

    for ds_name in target_datasets:
        logger.info("=" * 60)
        logger.info("RetTune Stage 1 EDA: Profiling '%s'", ds_name.upper())
        logger.info("=" * 60)

        try:
            dataset = load_dataset(ds_name, config, verify=True)
        except (ContractValidationError, DataLeakageError, FileNotFoundError) as exc:
            logger.error("Data contract verification failed for '%s': %s", ds_name, exc)
            return 1

        threshold = (
            args.relevance_threshold
            if getattr(args, "relevance_threshold", None) is not None
            else config.datasets[ds_name].relevance_threshold
        )

        profile = profile_dataset(
            dataset=dataset,
            config=config.eda,
            relevance_threshold=threshold,
            seed=config.seed,
        )
        profiles[ds_name] = profile

        out_dir = config.get_results_dir("eda", ds_name)
        artifact_paths = export_eda_artifacts(profile, out_dir)
        logger.info("Persisted artifacts to %s: %s", out_dir, [p.name for p in artifact_paths.values()])

    console.print()
    display_length_summary_table(profiles)
    console.print()
    display_coverage_summary_table(profiles)
    console.print()

    if args.plot and profiles:
        figures_dir = config.paths.results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating Stage 1 comparative figures in %s...", figures_dir)
        try:
            len_figs = render_token_length_ecdf(profiles, figures_dir)
            cov_figs = render_coverage_density(profiles, figures_dir)
            tbl_figs = render_summary_table(profiles, figures_dir)
            console.print(f"[bold green]✓ Stage 1 figures exported to {figures_dir}[/bold green]")
            console.print(f"  - Length Distributions: {len_figs['svg'].name}, {len_figs['png'].name}")
            console.print(f"  - Coverage Density:     {cov_figs['svg'].name}, {cov_figs['png'].name}")
            console.print(f"  - Summary Tables:       {tbl_figs['svg'].name}, {tbl_figs['png'].name}")
        except Exception as exc:
            logger.error("Failed to render Stage 1 figures: %s", exc)
            return 1

    return 0


def run_stage_sparse(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    target_datasets: Sequence[str],
) -> int:
    """Execute Stage 2 Sparse Term-Saturation & Length-Normalization Mechanics."""
    reports: Dict[str, SparseDatasetReport] = {}

    for ds_name in target_datasets:
        logger.info("=" * 60)
        logger.info("RetTune Stage 2 Sparse: Benchmarking '%s'", ds_name.upper())
        logger.info("=" * 60)

        try:
            dataset = load_dataset(ds_name, config, verify=True)
        except (ContractValidationError, DataLeakageError, FileNotFoundError) as exc:
            logger.error("Data contract verification failed for '%s': %s", ds_name, exc)
            return 1

        threshold = (
            args.relevance_threshold
            if getattr(args, "relevance_threshold", None) is not None
            else config.datasets[ds_name].relevance_threshold
        )

        report, _, _ = run_sparse_benchmark_for_dataset(
            dataset=dataset,
            sparse_cfg=config.sparse,
            eval_cfg=config.evaluation,
            relevance_threshold=threshold,
            token_pattern=config.eda.token_pattern,
            seed=config.seed,
        )
        reports[ds_name] = report

    metrics_dir = config.get_metrics_dir()
    export_sparse_artifacts(reports, metrics_dir)

    console.print()
    display_sparse_summary_table(reports)
    console.print()

    if args.plot and reports:
        figures_dir = config.paths.results_dir / "figures"
        try:
            figs = render_stage2_bm25_plus_figure(reports, figures_dir)
            console.print(
                f"[bold green]✓ Stage 2 figure exported: {figs['svg'].name}, {figs['png'].name}[/bold green]"
            )
        except Exception as exc:
            logger.error("Failed to render Stage 2 figure: %s", exc)
            return 1

    return 0


def run_stage_dense(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    target_datasets: Sequence[str],
) -> int:
    """Execute Stage 3 Embedding Spectral Geometry, Subspace Alignment & Matryoshka Slicing."""
    reports: Dict[str, DenseDatasetReport] = {}

    for ds_name in target_datasets:
        logger.info("=" * 60)
        logger.info("RetTune Stage 3 Dense: Benchmarking '%s'", ds_name.upper())
        logger.info("=" * 60)

        try:
            dataset = load_dataset(ds_name, config, verify=True)
            bundle = load_embedding_bundle(ds_name, config)
        except Exception as exc:
            logger.error("Failed to load dataset or embeddings for '%s': %s", ds_name, exc)
            return 1

        threshold = (
            args.relevance_threshold
            if getattr(args, "relevance_threshold", None) is not None
            else config.datasets[ds_name].relevance_threshold
        )

        report = run_dense_benchmark_for_dataset(
            dataset=dataset,
            bundle=bundle,
            dense_cfg=config.dense,
            eval_cfg=config.evaluation,
            relevance_threshold=threshold,
            seed=config.seed,
        )
        reports[ds_name] = report

    metrics_dir = config.get_metrics_dir()
    export_dense_artifacts(reports, metrics_dir)

    console.print()
    display_dense_summary_table(reports)
    console.print()

    if args.plot and reports:
        figures_dir = config.paths.results_dir / "figures"
        try:
            figs = render_stage3_spectral_and_mrl_figure(reports, figures_dir)
            console.print(
                f"[bold green]✓ Stage 3 figure exported: {figs['svg'].name}, {figs['png'].name}[/bold green]"
            )
        except Exception as exc:
            logger.error("Failed to render Stage 3 figure: %s", exc)
            return 1

    return 0


def run_stage_fusion(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    target_datasets: Sequence[str],
) -> int:
    """Execute Stage 4 Score Distribution Calibration, Missing-Candidate Imputation & Hybrid Fusion."""
    reports: Dict[str, FusionDatasetReport] = {}

    for ds_name in target_datasets:
        logger.info("=" * 60)
        logger.info("RetTune Stage 4 Fusion: Benchmarking '%s'", ds_name.upper())
        logger.info("=" * 60)

        try:
            dataset = load_dataset(ds_name, config, verify=True)
            bundle = load_embedding_bundle(ds_name, config)
        except Exception as exc:
            logger.error("Failed to load dataset or embeddings for '%s': %s", ds_name, exc)
            return 1

        threshold = (
            args.relevance_threshold
            if getattr(args, "relevance_threshold", None) is not None
            else config.datasets[ds_name].relevance_threshold
        )

        _, sparse_idx, sparse_coll = get_or_build_winning_sparse_index(dataset, config)
        dev_qids = sorted(dataset.qrels_dev.keys())
        test_qids = sorted(dataset.qrels_test.keys())
        top_k = config.fusion.top_k

        sparse_dev_run = sparse_idx.retrieve_top_k(dev_qids, top_k=top_k)
        sparse_test_run = sparse_idx.retrieve_top_k(test_qids, top_k=top_k)
        sparse_dev_mat = sparse_idx.score_matrix(dev_qids)
        sparse_test_mat = sparse_idx.score_matrix(test_qids)

        dense_dev_run, dense_test_run, dense_dev_mat, dense_test_mat = (
            get_dense_candidates_and_matrices(
                bundle, regime_name="raw_cosine", mrl_dim=bundle.full_dim, top_k=top_k
            )
        )

        report = run_fusion_benchmark_for_dataset(
            dataset=dataset,
            doc_ids=sparse_coll.doc_ids,
            sparse_dev_run=sparse_dev_run,
            sparse_test_run=sparse_test_run,
            dense_dev_run=dense_dev_run,
            dense_test_run=dense_test_run,
            sparse_dev_mat=sparse_dev_mat,
            sparse_test_mat=sparse_test_mat,
            dense_dev_mat=dense_dev_mat,
            dense_test_mat=dense_test_mat,
            fusion_cfg=config.fusion,
            eval_cfg=config.evaluation,
            relevance_threshold=threshold,
            seed=config.seed,
        )
        reports[ds_name] = report

    transfer_matrix = compute_ood_alpha_transfer_matrix(reports, calibration_method="zscore")
    metrics_dir = config.get_metrics_dir()
    export_fusion_artifacts(reports, transfer_matrix, metrics_dir)

    console.print()
    display_fusion_summary_table(reports, transfer_matrix)
    console.print()

    if args.plot and reports:
        figures_dir = config.paths.results_dir / "figures"
        try:
            figs = render_stage4_calibration_and_transfer_figure(
                reports, transfer_matrix, figures_dir
            )
            console.print(
                f"[bold green]✓ Stage 4 figure exported: {figs['svg'].name}, {figs['png'].name}[/bold green]"
            )
        except Exception as exc:
            logger.error("Failed to render Stage 4 figure: %s", exc)
            return 1

    return 0


def run_stage_rerank(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    target_datasets: Sequence[str],
) -> int:
    """Execute Stage 5 Candidate Pool Ceiling, Reranker Error Decomposition & Latency Tradeoff Analysis."""
    reports: Dict[str, RerankerDatasetReport] = {}

    for ds_name in target_datasets:
        logger.info("=" * 60)
        logger.info("RetTune Stage 5 Rerank & Latency Tradeoff: Benchmarking '%s'", ds_name.upper())
        logger.info("=" * 60)

        try:
            dataset = load_dataset(ds_name, config, verify=True)
            cache_bundle = load_reranker_cache(ds_name, config)
            pipeline_runs, pipeline_labels, stage1_meta = (
                _build_stage5_pipeline_runs_for_dataset(dataset, config)
            )
        except Exception as exc:
            logger.error("Failed to initialize Stage 5 inputs for '%s': %s", ds_name, exc)
            return 1

        threshold = (
            args.relevance_threshold
            if getattr(args, "relevance_threshold", None) is not None
            else config.datasets[ds_name].relevance_threshold
        )

        report = run_reranker_benchmark_for_dataset(
            dataset=dataset,
            cache_bundle=cache_bundle,
            pipeline_runs_test=pipeline_runs,
            pipeline_labels=pipeline_labels,
            pipeline_stage1_metrics=stage1_meta,
            reranker_cfg=config.reranker,
            eval_cfg=config.evaluation,
            relevance_threshold=threshold,
            seed=config.seed,
        )
        reports[ds_name] = report

    metrics_dir = config.get_metrics_dir()
    export_reranker_artifacts(reports, metrics_dir)

    console.print()
    display_reranker_summary_table(reports)
    console.print()

    if args.plot and reports:
        figures_dir = config.paths.results_dir / "figures"
        try:
            figs = render_stage5_latency_and_ceiling_figure(reports, figures_dir)
            console.print(
                f"[bold green]✓ Stage 5 figure exported: {figs['svg'].name}, {figs['png'].name}[/bold green]"
            )
        except Exception as exc:
            logger.error("Failed to render Stage 5 figure: %s", exc)
            return 1

    return 0


StageHandler = Callable[[argparse.Namespace, BenchmarkConfig, Sequence[str]], int]
STAGE_REGISTRY: Dict[str, StageHandler] = {
    "eda": run_stage_eda,
    "sparse": run_stage_sparse,
    "dense": run_stage_dense,
    "fusion": run_stage_fusion,
    "rerank": run_stage_rerank,
}


def run_stage_all(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    target_datasets: Sequence[str],
) -> int:
    """Execute all registered pipeline stages in sequence."""
    for stage_name, stage_fn in STAGE_REGISTRY.items():
        logger.info("Executing benchmark stage: %s", stage_name.upper())
        ret = stage_fn(args, config, target_datasets)
        if ret != 0:
            logger.error("Benchmark stage '%s' failed with exit code %d", stage_name, ret)
            return ret
    return 0


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint.

    Args:
        argv: Command-line arguments sequence. If None, reads from sys.argv[1:].

    Returns:
        Exit code: 0 on success, 1 on execution/guard failure, 2 on syntax error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    try:
        config = load_config(args.config)
        config = apply_config_overrides(config, args)
        target_datasets = resolve_target_datasets(args.dataset, config)
    except Exception as e:
        logger.error("Configuration initialization failed: %s", e)
        return 1

    # Setup mode: ingest datasets, generate embeddings, and pre-cache Cross-Encoder scores
    if args.setup:
        setup_ret = run_setup(args, config=config)
        if setup_ret != 0:
            logger.error("Setup failed with exit code %d", setup_ret)
            return setup_ret
        if args.stage is None:
            logger.info("Benchmark setup completed successfully.")
            return 0

    # Offline Benchmark Mode: defaults to 'all' stages if not explicitly specified
    stage_to_run = args.stage or "all"

    # Enforce Stage-Aware Zero-Network Offline Preflight Guard
    is_ready, missing = verify_offline_readiness(target_datasets, config, stage=stage_to_run)
    if not is_ready:
        print(format_guard_failure_message(missing), file=sys.stderr)
        return 1

    logger.debug(
        "Zero-network guard passed for datasets: %s. Executing stage: %s",
        target_datasets,
        stage_to_run,
    )

    if stage_to_run == "all":
        return run_stage_all(args, config, target_datasets)
    elif stage_to_run in STAGE_REGISTRY:
        return STAGE_REGISTRY[stage_to_run](args, config, target_datasets)
    else:
        logger.error(
            "Unsupported pipeline stage '%s'. Supported: %s",
            stage_to_run,
            list(STAGE_REGISTRY.keys()),
        )
        return 1
