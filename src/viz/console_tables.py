"""Rich console table rendering for RetTune benchmark diagnostics."""

from typing import Dict
from rich.console import Console
from rich.table import Table

from ..eda import LexicalProfile

console = Console()


def display_length_summary_table(profiles: Dict[str, LexicalProfile]) -> None:
    """Render formatted console table summarizing document and query length dynamics."""
    table = Table(
        title="[bold cyan]RetTune Stage 1.3: Corpus & Query Length Dynamics[/bold cyan]",
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
        title="[bold cyan]RetTune Stage 1.3: IDF-Weighted Query Coverage & Separation[/bold cyan]",
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
