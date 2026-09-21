#!/usr/bin/env python3
"""CLI runner for Stage 1.3 Lexical Dynamics & Vocabulary Overlap Profiling (EDA).

Computes document/query length distributions, Fisher-Pearson skewness, smoothed BM25 IDF,
and IDF-weighted coverage distributions contrasting ground-truth relevant pairs against
random background noise. Exports CSV summaries to results/eda/<dataset>/ and publication-grade
SVG/PNG comparison figures to results/figures/.
"""

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.table import Table

from rettune.config import BenchmarkConfig, load_config
from rettune.data_loader import ContractValidationError, DataLeakageError, load_dataset
from rettune.eda import LexicalProfile, export_eda_artifacts, profile_dataset
from rettune.viz import render_coverage_density, render_token_length_ecdf

logger = logging.getLogger("rettune.profile_lexical")
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure console logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=format_str, datefmt="%H:%M:%S")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Profile lexical dynamics and vocabulary overlap for RetTune corpora."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Target dataset ('all', 'nfcorpus', 'scifact', 'arguana'). Default: 'all'.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom benchmark YAML configuration file.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate comparative figures in results/figures/ (default: True). Use --no-plot for headless/fast compute.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=None,
        help="Relevance score threshold for positive pairs (overrides dataset config if provided).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    return parser


def display_length_summary_table(profiles: Dict[str, LexicalProfile]) -> None:
    """Render formatted console table summarizing document and query length dynamics."""
    table = Table(
        title="[bold cyan]RetTune Stage 1.3: Corpus & Query Length Dynamics[/bold cyan]",
        title_justify="left",
    )
    table.add_column("Dataset", style="bold white")
    table.add_column("Unit", style="cyan")
    table.add_column("Count (N)", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("IQR", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("p99", justify="right")
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
                f"{summary.percentiles.get('p99', 0.0):.1f}",
                f"[{skew_style}]{summary.skewness:.2f}[/{skew_style}]",
            )
        table.add_section()

    console.print(table)


def display_coverage_summary_table(profiles: Dict[str, LexicalProfile]) -> None:
    """Render formatted console table contrasting relevant vs. background coverage."""
    table = Table(
        title="[bold cyan]RetTune Stage 1.3: IDF-Weighted Query Coverage & Separation[/bold cyan]",
        title_justify="left",
    )
    table.add_column("Dataset", style="bold white")
    table.add_column("Rel Pairs", justify="right")
    table.add_column("Rel Mean", justify="right")
    table.add_column("Rel Med", justify="right")
    table.add_column("Noise Med", justify="right")
    table.add_column("Δ Median", justify="right", style="bold yellow")
    table.add_column("Cohen's d", justify="right", style="bold green")
    table.add_column("Wasserstein (W₁)", justify="right", style="bold magenta")

    for ds_name, prof in profiles.items():
        rel = prof.coverage_summary_relevant_all
        bg = prof.coverage_summary_random
        sep = prof.separation_all

        table.add_row(
            ds_name.upper(),
            f"{rel.count:,}",
            f"{rel.mean:.3f}",
            f"{rel.median:.3f}",
            f"{bg.median:.3f}",
            f"{sep.delta_median:+.3f}",
            f"{sep.cohens_d:.2f}",
            f"{sep.wasserstein_distance:.3f}",
        )

    console.print(table)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        config: BenchmarkConfig = load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 1

    # Determine target datasets
    if args.dataset == "all":
        target_datasets = list(config.active_datasets)
    else:
        if args.dataset not in config.datasets:
            logger.error("Dataset '%s' is not configured in benchmark_config.yaml", args.dataset)
            return 1
        target_datasets = [args.dataset]

    profiles: Dict[str, LexicalProfile] = {}

    for ds_name in target_datasets:
        logger.info("=" * 60)
        logger.info("Processing lexical profiling for: %s", ds_name.upper())
        logger.info("=" * 60)

        # 1. Load dataset with full contract verification
        try:
            dataset = load_dataset(ds_name, config, verify=True)
        except (ContractValidationError, DataLeakageError, FileNotFoundError) as exc:
            logger.error("Data contract verification failed for '%s': %s", ds_name, exc)
            return 1

        # 2. Determine relevance threshold
        threshold = (
            args.relevance_threshold
            if args.relevance_threshold is not None
            else config.datasets[ds_name].relevance_threshold
        )

        # 3. Compute lexical profile
        profile = profile_dataset(
            dataset=dataset,
            config=config.eda,
            relevance_threshold=threshold,
            seed=config.seed,
        )
        profiles[ds_name] = profile

        # 4. Export CSV and JSON artifacts
        out_dir = config.get_results_dir("eda", ds_name)
        artifact_paths = export_eda_artifacts(profile, out_dir)
        logger.info("Persisted artifacts to %s: %s", out_dir, [p.name for p in artifact_paths.values()])

    # Display console tables
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
            console.print(
                f"[bold green]✓ Figures successfully exported to {figures_dir}[/bold green]"
            )
            console.print(f"  - Length Distributions: {len_figs['svg'].name}, {len_figs['png'].name}")
            console.print(f"  - Coverage Density:     {cov_figs['svg'].name}, {cov_figs['png'].name}")
        except Exception as exc:
            logger.error("Failed to render graphical figures: %s", exc)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
