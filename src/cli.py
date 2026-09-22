"""Command-line interface and execution dispatcher for RetTune.

Provides a unified root CLI for offline hybrid search benchmark sweeps,
enforcing strict zero-network execution and fail-fast offline readiness checks.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rettune.config import BenchmarkConfig, load_config

logger = logging.getLogger("rettune.cli")

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
        default="all",
        help="Pipeline stage to execute ('eda', 'all').",
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

    # Zero-Network Offline Preflight Check (bypassed only if --setup is explicitly passed)
    if not args.setup:
        is_ready, missing = verify_offline_readiness(target_datasets, config)
        if not is_ready:
            print(format_guard_failure_message(missing), file=sys.stderr)
            return 1

    logger.debug("Offline readiness verified for target datasets: %s", target_datasets)
    return 0
