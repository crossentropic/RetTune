"""Unit and integration tests for Stage 1.3 Lexical Dynamics & EDA engine."""

import json
import math
from pathlib import Path
import pytest
import numpy as np
from scipy.stats import skew, wasserstein_distance

from rettune.config import BenchmarkConfig, EDAConfig
from rettune.data_loader import Document, IRDataset, load_dataset
from rettune.eda import (
    DistributionSummary,
    LexicalProfile,
    SeparationMetrics,
    compute_distribution_summary,
    compute_idf_overlap,
    compute_separation_metrics,
    compute_smoothed_bm25_idf,
    export_eda_artifacts,
    profile_dataset,
    sample_random_pairs,
    tokenize,
)
from rettune.viz import render_coverage_density, render_token_length_ecdf
import scripts.profile_lexical as profile_lexical_cli


def test_tokenize():
    """Verify alphanumeric regex tokenization and punctuation stripping."""
    assert tokenize("Hello World!") == ["hello", "world"]
    assert tokenize("COVID-19 in patients with MED-4618.") == ["covid", "19", "in", "patients", "with", "med", "4618"]
    assert tokenize("") == []
    assert tokenize("   !@#$%^&*()   ") == []
    assert tokenize("T-cell receptor α/β") == ["t", "cell", "receptor", "α", "β"]


def test_distribution_summary_matches_scipy():
    """Verify sample moments and Fisher-Pearson skewness match SciPy reference."""
    arr = np.array([2.5, 4.0, 7.2, 11.0, 15.5, 22.1, 45.0, 98.3], dtype=np.float64)
    summary = compute_distribution_summary(arr, percentiles=[25, 50, 75, 90, 95, 99])

    assert summary.count == len(arr)
    assert pytest.approx(summary.mean, rel=1e-5) == np.mean(arr)
    assert pytest.approx(summary.std, rel=1e-5) == np.std(arr, ddof=1)
    assert pytest.approx(summary.median, rel=1e-5) == np.median(arr)
    assert pytest.approx(summary.min, rel=1e-5) == np.min(arr)
    assert pytest.approx(summary.max, rel=1e-5) == np.max(arr)

    expected_skew = float(skew(arr, bias=False))
    assert pytest.approx(summary.skewness, rel=1e-5) == expected_skew
    assert summary.skewness > 1.0  # Positively skewed right tail


def test_distribution_summary_edge_cases():
    """Verify sample size guards for n=0, n=1, n=2, and zero variance."""
    # n = 0
    s0 = compute_distribution_summary([])
    assert s0.count == 0
    assert s0.mean == 0.0
    assert s0.skewness == 0.0

    # n = 1
    s1 = compute_distribution_summary([42.0])
    assert s1.count == 1
    assert s1.mean == 42.0
    assert s1.std == 0.0
    assert s1.skewness == 0.0

    # n = 2
    s2 = compute_distribution_summary([10.0, 20.0])
    assert s2.count == 2
    assert s2.mean == 15.0
    assert s2.skewness == 0.0  # Sample skewness undefined for n < 3, guarded to 0.0

    # Zero variance (identical numbers)
    s_const = compute_distribution_summary([5.0, 5.0, 5.0, 5.0, 5.0])
    assert s_const.count == 5
    assert s_const.std == 0.0
    assert s_const.skewness == 0.0


def test_smoothed_bm25_idf_math():
    """Verify hand-computed smoothed BM25 IDF matches ln((N + 1) / (df + 0.5))."""
    # 3-doc synthetic collection
    corpus = [
        ["apple", "banana"],            # Doc 1
        ["banana", "orange"],           # Doc 2
        ["banana", "orange", "pear"],   # Doc 3
    ]
    idf_map = compute_smoothed_bm25_idf(corpus)

    # N = 3
    # apple: df = 1 => ln((3 + 1) / (1 + 0.5)) = ln(4 / 1.5) = ln(8/3)
    assert pytest.approx(idf_map["apple"], rel=1e-5) == math.log(8.0 / 3.0)

    # orange: df = 2 => ln((3 + 1) / (2 + 0.5)) = ln(4 / 2.5) = ln(8/5)
    assert pytest.approx(idf_map["orange"], rel=1e-5) == math.log(8.0 / 5.0)

    # banana: df = 3 => ln((3 + 1) / (3 + 0.5)) = ln(4 / 3.5) = ln(8/7)
    assert pytest.approx(idf_map["banana"], rel=1e-5) == math.log(8.0 / 7.0)

    # pear: df = 1 => ln(8/3)
    assert pytest.approx(idf_map["pear"], rel=1e-5) == math.log(8.0 / 3.0)

    # Strict non-negativity: all terms must have IDF > 0
    for term, val in idf_map.items():
        assert val > 0.0


def test_idf_overlap_bounds_and_math():
    """Verify IDF-weighted coverage bounds [0.0, 1.0], complete matches, and OOV handling."""
    idf_map = {
        "cancer": 2.5,
        "therapy": 1.5,
        "trial": 0.5,
    }

    # 1. Complete containment: query completely inside document => 1.0 exactly
    q = ["cancer", "therapy"]
    d_complete = ["cancer", "therapy", "clinical", "trial"]
    ov_complete = compute_idf_overlap(q, d_complete, idf_map)
    assert ov_complete == 1.0

    # 2. Complete disjointness: 0.0
    d_disjoint = ["unrelated", "biology", "cell"]
    ov_disjoint = compute_idf_overlap(q, d_disjoint, idf_map)
    assert ov_disjoint == 0.0

    # 3. Partial match: exactly cancer (2.5) out of total (2.5 + 1.5 = 4.0) => 2.5 / 4.0 = 0.625
    d_partial = ["cancer", "cell"]
    ov_partial = compute_idf_overlap(q, d_partial, idf_map)
    assert pytest.approx(ov_partial, rel=1e-5) == 2.5 / 4.0

    # 4. Out-of-vocabulary query: returns 0.0 safely without zero-division error
    q_oov = ["unknown_a", "unknown_b"]
    assert compute_idf_overlap(q_oov, d_complete, idf_map) == 0.0

    # 5. Empty query or empty doc: returns 0.0 safely
    assert compute_idf_overlap([], d_complete, idf_map) == 0.0
    assert compute_idf_overlap(q, [], idf_map) == 0.0


def test_random_pair_sampling_reproducibility():
    """Verify background pair sampling is deterministic and reproducible under seed."""
    qids = [f"q_{i}" for i in range(10)]
    dids = [f"d_{j}" for j in range(20)]

    pairs1 = sample_random_pairs(qids, dids, sample_size=50, seed=42)
    pairs2 = sample_random_pairs(qids, dids, sample_size=50, seed=42)
    pairs3 = sample_random_pairs(qids, dids, sample_size=50, seed=999)

    assert len(pairs1) == 50
    assert pairs1 == pairs2
    assert pairs1 != pairs3


def test_separation_metrics_edge_cases():
    """Verify separation contrast handles small N, empty arrays, and identical distributions."""
    # Empty inputs
    sep_empty = compute_separation_metrics([], [])
    assert sep_empty.delta_mean == 0.0
    assert sep_empty.cohens_d == 0.0
    assert sep_empty.wasserstein_distance == 0.0

    # Identical distributions
    arr = [0.2, 0.4, 0.6, 0.8]
    sep_ident = compute_separation_metrics(arr, arr)
    assert sep_ident.delta_mean == 0.0
    assert sep_ident.delta_median == 0.0
    assert sep_ident.cohens_d == 0.0
    assert sep_ident.wasserstein_distance == 0.0

    # Single element sample (n_r=1, n_b=1 => df=0 => guarded to 0.0 without ZeroDivisionError)
    sep_single = compute_separation_metrics([0.8], [0.2])
    assert sep_single.cohens_d == 0.0
    assert pytest.approx(sep_single.delta_mean, rel=1e-5) == 0.6
    assert pytest.approx(sep_single.wasserstein_distance, rel=1e-5) == 0.6

    # Large separation
    rel = [0.8, 0.85, 0.9, 0.95]
    bg = [0.0, 0.05, 0.1, 0.15]
    sep_large = compute_separation_metrics(rel, bg)
    assert sep_large.delta_mean > 0.7
    assert sep_large.delta_median > 0.7
    assert sep_large.cohens_d > 5.0
    assert sep_large.wasserstein_distance > 0.7


def test_profile_dataset_and_export(synthetic_benchmark_config: BenchmarkConfig, tmp_path: Path):
    """End-to-end integration test of dataset profiling and artifact export."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)
    profile = profile_dataset(
        dataset=dataset,
        config=synthetic_benchmark_config.eda,
        relevance_threshold=1.0,
        seed=synthetic_benchmark_config.seed,
    )

    assert isinstance(profile, LexicalProfile)
    assert profile.dataset_name == "synthetic"
    assert profile.doc_summary.count == len(dataset.corpus)
    assert profile.query_summary_all.count == len(dataset.queries)
    assert len(profile.idf_map) > 0
    assert profile.coverage_relevant_all.size > 0
    assert profile.coverage_random_background.size == synthetic_benchmark_config.eda.random_pairs_sample_size

    # Export artifacts
    out_dir = tmp_path / "eda_test_export"
    artifacts = export_eda_artifacts(profile, out_dir)

    assert artifacts["length_summary_csv"].exists()
    assert artifacts["idf_coverage_csv"].exists()
    assert artifacts["separation_metrics_csv"].exists()
    assert artifacts["lexical_stats_json"].exists()

    # Verify JSON structure
    with open(artifacts["lexical_stats_json"], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["dataset_name"] == "synthetic"
    assert "lengths" in data
    assert "coverage" in data
    assert "separation" in data


def test_viz_rendering(synthetic_benchmark_config: BenchmarkConfig, tmp_path: Path):
    """Verify headless rendering of ECDF and coverage step histograms."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)
    profile = profile_dataset(
        dataset=dataset,
        config=synthetic_benchmark_config.eda,
        seed=42,
    )

    profiles = {"synthetic": profile}
    figures_dir = tmp_path / "figures"

    len_figs = render_token_length_ecdf(profiles, figures_dir)
    assert len_figs["svg"].exists()
    assert len_figs["png"].exists()
    assert len_figs["svg"].stat().st_size > 1000
    assert len_figs["png"].stat().st_size > 1000

    cov_figs = render_coverage_density(profiles, figures_dir)
    assert cov_figs["svg"].exists()
    assert cov_figs["png"].exists()
    assert cov_figs["svg"].stat().st_size > 1000
    assert cov_figs["png"].stat().st_size > 1000

    # Empty profiles guard
    assert render_token_length_ecdf({}, figures_dir) == {}
    assert render_coverage_density({}, figures_dir) == {}


def test_profile_lexical_cli_in_process(synthetic_benchmark_config: BenchmarkConfig, tmp_path: Path):
    """Verify profile_lexical CLI entrypoint in-process with --no-plot and custom config."""
    # Write synthetic config to temporary yaml file
    config_dict = synthetic_benchmark_config.model_dump(mode="json")
    # Update results dir to tmp_path
    config_dict["paths"]["results_dir"] = str(tmp_path / "results")
    import yaml
    cfg_file = tmp_path / "test_benchmark_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f)

    argv = [
        "--dataset", "synthetic",
        "--config", str(cfg_file),
        "--no-plot",
    ]
    ret_code = profile_lexical_cli.main(argv)
    assert ret_code == 0

    # Verify CSV files created in results dir
    synth_eda_dir = tmp_path / "results" / "eda" / "synthetic"
    assert (synth_eda_dir / "length_summary.csv").exists()
    assert (synth_eda_dir / "idf_coverage.csv").exists()
    assert (synth_eda_dir / "separation_metrics.csv").exists()
    assert (synth_eda_dir / "lexical_stats.json").exists()
