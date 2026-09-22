"""Command-line interface and execution dispatcher for RetTune.

Provides a unified root CLI for offline hybrid search benchmark sweeps,
enforcing strict zero-network execution, fail-fast offline readiness checks,
and clean stage orchestration.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rich.console import Console

from rettune.config import BenchmarkConfig, load_config
from rettune.data_loader import ContractValidationError, DataLeakageError, load_dataset
from rettune.eda import LexicalProfile, export_eda_artifacts, profile_dataset
from rettune.viz import (
    display_coverage_summary_table,
    display_length_summary_table,
    render_coverage_density,
    render_summary_table,
    render_token_length_ecdf,
)

logger = logging.getLogger("rettune.cli")
console = Console()

SUPPORTED_STAGES = ["eda", "all"]


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
        help="Pipeline stage to execute ('eda', 'all'). Defaults to 'all' in offline benchmark mode.",
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
        help="Download, canonicalize, and verify benchmark datasets before running.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download during setup even if dataset files are already cached.",
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


def get_missing_dataset_files(dataset_name: str, config: BenchmarkConfig) -> List[Path]:
    """Return any missing or zero-byte canonical dataset files for a given dataset."""
    dataset_dir = config.get_dataset_dir(dataset_name)
    required_files = [
        dataset_dir / "corpus.jsonl",
        dataset_dir / "queries.jsonl",
        config.get_qrels_path(dataset_name, "dev"),
        config.get_qrels_path(dataset_name, "test"),
    ]
    return [p for p in required_files if not p.is_file() or p.stat().st_size == 0]


def verify_offline_readiness(
    target_datasets: Sequence[str],
    config: BenchmarkConfig,
) -> Tuple[bool, Dict[str, List[Path]]]:
    """Verify that all target datasets have canonical files locally available.

    Returns:
        (is_ready, missing_files_by_dataset)
    """
    missing_by_dataset: Dict[str, List[Path]] = {}
    for ds_name in target_datasets:
        missing = get_missing_dataset_files(ds_name, config)
        if missing:
            missing_by_dataset[ds_name] = missing
    return len(missing_by_dataset) == 0, missing_by_dataset


def format_guard_failure_message(missing_by_dataset: Dict[str, List[Path]]) -> str:
    """Construct a clean, actionable terminal banner explaining offline readiness failure."""
    lines = [
        "=" * 70,
        "ZERO-NETWORK GUARD FAILURE:",
        "RetTune runs strictly offline by default, but required dataset files",
        "are missing or unverified locally:",
    ]
    for ds_name, missing_files in missing_by_dataset.items():
        lines.append(f"  • Dataset '{ds_name}':")
        for f in missing_files:
            lines.append(f"      - Missing: {f}")

    lines.extend([
        "",
        "To download, canonicalize, and verify the required benchmark datasets, run:",
        "    python benchmark.py --setup",
        "or target a specific dataset:",
        f"    python benchmark.py --setup --dataset {next(iter(missing_by_dataset.keys()))}",
        "=" * 70,
    ])
    return "\n".join(lines)


def run_setup(args: argparse.Namespace) -> int:
    """Execute dataset download/setup by delegating to scripts/download_data.py via subprocess."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "download_data.py"
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

    logger.info("Executing dataset setup via %s: %s", script_path.name, " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode


def run_stage_eda(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    target_datasets: Sequence[str],
) -> int:
    """Execute Stage 1.3 Lexical Dynamics & Vocabulary Overlap Profiling."""
    profiles: Dict[str, LexicalProfile] = {}

    for ds_name in target_datasets:
        logger.info("=" * 60)
        logger.info("RetTune Stage 1.3 EDA: Profiling '%s'", ds_name.upper())
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

    # Display Rich console tables
    console.print()
    display_length_summary_table(profiles)
    console.print()
    display_coverage_summary_table(profiles)
    console.print()

    # Generate publication figures if requested
    if args.plot and profiles:
        figures_dir = config.paths.results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating comparative figures in %s...", figures_dir)
        try:
            len_figs = render_token_length_ecdf(profiles, figures_dir)
            cov_figs = render_coverage_density(profiles, figures_dir)
            tbl_figs = render_summary_table(profiles, figures_dir)
            console.print(f"[bold green]✓ Figures successfully exported to {figures_dir}[/bold green]")
            console.print(f"  - Length Distributions: {len_figs['svg'].name}, {len_figs['png'].name}")
            console.print(f"  - Coverage Density:     {cov_figs['svg'].name}, {cov_figs['png'].name}")
            console.print(f"  - Summary Tables:       {tbl_figs['svg'].name}, {tbl_figs['png'].name}")
        except Exception as exc:
            logger.error("Failed to render graphical figures: %s", exc)
            return 1

    return 0


# Registry of benchmark pipeline stage handlers
StageHandler = Callable[[argparse.Namespace, BenchmarkConfig, Sequence[str]], int]
STAGE_REGISTRY: Dict[str, StageHandler] = {
    "eda": run_stage_eda,
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

    # Setup mode: delegate to scripts/download_data.py
    if args.setup:
        setup_ret = run_setup(args)
        if setup_ret != 0:
            logger.error("Setup failed with exit code %d", setup_ret)
            return setup_ret
        if args.stage is None:
            logger.info("Dataset setup completed successfully.")
            return 0

    # Offline Benchmark Mode: defaults to 'all' stages if not explicitly specified
    stage_to_run = args.stage or "all"

    # Enforce Zero-Network Offline Preflight Guard
    is_ready, missing = verify_offline_readiness(target_datasets, config)
    if not is_ready:
        print(format_guard_failure_message(missing), file=sys.stderr)
        return 1

    logger.debug("Zero-network guard passed for datasets: %s. Executing stage: %s", target_datasets, stage_to_run)

    # Dispatch to target stage
    if stage_to_run == "all":
        return run_stage_all(args, config, target_datasets)
    elif stage_to_run in STAGE_REGISTRY:
        return STAGE_REGISTRY[stage_to_run](args, config, target_datasets)
    else:
        logger.error("Unsupported pipeline stage '%s'. Supported: %s", stage_to_run, list(STAGE_REGISTRY.keys()))
        return 1
