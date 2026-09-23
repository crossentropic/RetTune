"""Pure mathematical and statistical engine for Stage 1.3 Lexical Dynamics & EDA.

This module is 100% side-effect-free, headless-safe, and independent of any
visualization or GUI libraries. It implements:
1. Alphanumeric tokenization regex.
2. Token length distribution statistics and sample Fisher-Pearson skewness (g1).
3. Corpus-wide smoothed BM25 IDF ensuring strictly non-negative weights.
4. Strictly bounded [0.0, 1.0] IDF-weighted query-passage coverage.
5. Seeded background pair sampling and distribution separation metrics (Cohen's d,
   delta median, and Wasserstein distance).
6. Structured persistence helpers for CSV and JSON summary artifacts.
"""

from collections import Counter
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.stats import wasserstein_distance

from .config import EDAConfig
from .data_loader import Document, IRDataset

logger = logging.getLogger("rettune.eda")


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Statistical summary of a 1D numerical distribution (lengths or coverage ratios)."""
    count: int
    mean: float
    std: float
    median: float
    iqr: float
    skewness: float
    min: float
    max: float
    percentiles: Dict[str, float]

    def to_dict(self, prefix: str = "") -> Dict[str, Any]:
        """Convert to a flat dictionary with full floating-point precision."""
        res: Dict[str, Any] = {
            f"{prefix}count": self.count,
            f"{prefix}mean": self.mean,
            f"{prefix}std": self.std,
            f"{prefix}median": self.median,
            f"{prefix}iqr": self.iqr,
            f"{prefix}skewness": self.skewness,
            f"{prefix}min": self.min,
            f"{prefix}max": self.max,
        }
        for k, v in self.percentiles.items():
            res[f"{prefix}{k}"] = v
        return res


@dataclass(frozen=True, slots=True)
class SeparationMetrics:
    """Contrast metrics evaluating divergence between relevant and background coverage."""
    delta_mean: float
    delta_median: float
    cohens_d: float
    wasserstein_distance: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "delta_mean": self.delta_mean,
            "delta_median": self.delta_median,
            "cohens_d": self.cohens_d,
            "wasserstein_distance": self.wasserstein_distance,
        }


@dataclass(frozen=True, slots=True)
class LexicalProfile:
    """Complete diagnostic lexical profile for an IR dataset."""
    dataset_name: str
    doc_lengths: np.ndarray
    query_lengths_all: np.ndarray
    query_lengths_dev: np.ndarray
    query_lengths_test: np.ndarray
    doc_summary: DistributionSummary
    query_summary_all: DistributionSummary
    query_summary_dev: DistributionSummary
    query_summary_test: DistributionSummary
    idf_map: Dict[str, float]
    coverage_relevant_all: np.ndarray
    coverage_relevant_dev: np.ndarray
    coverage_relevant_test: np.ndarray
    coverage_random_background: np.ndarray
    coverage_summary_relevant_all: DistributionSummary
    coverage_summary_relevant_dev: DistributionSummary
    coverage_summary_relevant_test: DistributionSummary
    coverage_summary_random: DistributionSummary
    separation_all: SeparationMetrics
    separation_dev: SeparationMetrics
    separation_test: SeparationMetrics


def tokenize(text: str, pattern: str | re.Pattern[str] = r"\w+") -> List[str]:
    """Tokenize lowercased text into alphanumeric tokens, stripping punctuation."""
    if not text:
        return []
    if isinstance(pattern, str):
        regex = re.compile(pattern)
    else:
        regex = pattern
    return regex.findall(text.lower())


def compute_distribution_summary(
    values: Sequence[float | int] | np.ndarray,
    percentiles: Sequence[int] = (25, 50, 75, 90, 95, 99),
) -> DistributionSummary:
    """Compute summary statistics, percentiles, and sample Fisher-Pearson skewness.

    For skewness, uses Bessel's correction ddof=1 for std, and guards samples with
    n < 3 or near-zero variance (std < 1e-12) by returning 0.0. Matches
    scipy.stats.skew(x, bias=False).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)

    pct_keys = [f"p{p}" for p in percentiles]

    if n == 0:
        return DistributionSummary(
            count=0,
            mean=0.0,
            std=0.0,
            median=0.0,
            iqr=0.0,
            skewness=0.0,
            min=0.0,
            max=0.0,
            percentiles={k: 0.0 for k in pct_keys},
        )

    if n == 1:
        val = float(arr[0])
        return DistributionSummary(
            count=1,
            mean=val,
            std=0.0,
            median=val,
            iqr=0.0,
            skewness=0.0,
            min=val,
            max=val,
            percentiles={k: val for k in pct_keys},
        )

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    median_val = float(np.median(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Percentiles using linear interpolation
    pct_values = np.percentile(arr, percentiles)
    pct_dict = {f"p{p}": float(v) for p, v in zip(percentiles, pct_values)}

    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    iqr_val = float(p75 - p25)

    # Fisher-Pearson coefficient of skewness g1 (unbiased sample estimator)
    if n < 3 or std_val < 1e-12:
        skew_val = 0.0
    else:
        # Standardized third central moment
        factor = n / ((n - 1) * (n - 2))
        skew_val = float(factor * np.sum(((arr - mean_val) / std_val) ** 3))

    return DistributionSummary(
        count=n,
        mean=mean_val,
        std=std_val,
        median=median_val,
        iqr=iqr_val,
        skewness=skew_val,
        min=min_val,
        max=max_val,
        percentiles=pct_dict,
    )


def compute_smoothed_bm25_idf(corpus_tokens: Iterable[Sequence[str]]) -> Dict[str, float]:
    """Compute corpus-level smoothed BM25 IDF across all vocabulary terms.

    Formula:
        IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)) = ln((N + 1) / (n_t + 0.5))

    Because N >= n_t >= 1 for terms present in the corpus, (N + 1) / (n_t + 0.5) > 1,
    guaranteeing strictly positive weights (IDF > 0) and avoiding negative term scores.
    """
    doc_freqs: Counter[str] = Counter()
    total_docs = 0

    for tokens in corpus_tokens:
        total_docs += 1
        unique_terms = set(tokens)
        doc_freqs.update(unique_terms)

    if total_docs == 0:
        return {}

    idf_map: Dict[str, float] = {}
    n_plus_one = total_docs + 1.0

    for term, df in doc_freqs.items():
        # ln((N + 1) / (df + 0.5))
        idf_val = math.log(n_plus_one / (df + 0.5))
        idf_map[term] = float(idf_val)

    return idf_map


def compute_idf_overlap(
    query_tokens: Sequence[str] | Set[str],
    doc_tokens: Sequence[str] | Set[str],
    idf_map: Dict[str, float],
) -> float:
    """Compute IDF-weighted token overlap of query in passage, bounded in [0.0, 1.0].

    Guards:
    1. If query has no recognized tokens in idf_map (or denom <= 0), returns 0.0.
    2. Uses set intersection for exact term containment.
    3. Fully matching queries return 1.0 exactly without denominator epsilon distortion.
    """
    q_unique = set(query_tokens)
    denom = sum(idf_map.get(t, 0.0) for t in q_unique)
    if denom <= 0.0:
        return 0.0

    d_unique = set(doc_tokens)
    overlap_terms = q_unique & d_unique
    numer = sum(idf_map.get(t, 0.0) for t in overlap_terms)

    ratio = numer / denom
    return float(min(1.0, max(0.0, ratio)))


def sample_random_pairs(
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    sample_size: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Sample M random (query_id, doc_id) background pairs uniformly with replacement."""
    if not query_ids or not doc_ids or sample_size <= 0:
        return []

    rng = np.random.default_rng(seed)
    q_idx = rng.integers(0, len(query_ids), size=sample_size)
    d_idx = rng.integers(0, len(doc_ids), size=sample_size)

    return [(query_ids[qi], doc_ids[di]) for qi, di in zip(q_idx, d_idx)]


def compute_separation_metrics(
    relevant_scores: Sequence[float] | np.ndarray,
    background_scores: Sequence[float] | np.ndarray,
) -> SeparationMetrics:
    """Calculate contrast metrics between relevant and background overlap distributions."""
    r_arr = np.asarray(relevant_scores, dtype=np.float64)
    b_arr = np.asarray(background_scores, dtype=np.float64)

    n_r = int(r_arr.size)
    n_b = int(b_arr.size)

    if n_r == 0 or n_b == 0:
        return SeparationMetrics(
            delta_mean=0.0,
            delta_median=0.0,
            cohens_d=0.0,
            wasserstein_distance=0.0,
        )

    mean_r = float(np.mean(r_arr))
    mean_b = float(np.mean(b_arr))
    delta_mean = mean_r - mean_b
    delta_median = float(np.median(r_arr) - np.median(b_arr))

    # Guarded Cohen's d (requires at least 1 degree of freedom: n_r + n_b - 2 >= 1 => n_r + n_b >= 3)
    if n_r + n_b <= 2:
        cohens_d = 0.0
    else:
        var_r = float(np.var(r_arr, ddof=1)) if n_r > 1 else 0.0
        var_b = float(np.var(b_arr, ddof=1)) if n_b > 1 else 0.0
        pooled_var = ((n_r - 1) * var_r + (n_b - 1) * var_b) / (n_r + n_b - 2)
        s_pooled = math.sqrt(max(0.0, pooled_var))
        cohens_d = (delta_mean / s_pooled) if s_pooled > 1e-12 else 0.0

    w_dist = float(wasserstein_distance(r_arr, b_arr))

    return SeparationMetrics(
        delta_mean=delta_mean,
        delta_median=delta_median,
        cohens_d=cohens_d,
        wasserstein_distance=w_dist,
    )


def profile_dataset(
    dataset: IRDataset,
    config: EDAConfig,
    relevance_threshold: float = 1.0,
    seed: int = 42,
) -> LexicalProfile:
    """Execute complete lexical dynamics profiling for a dataset."""
    logger.info("Profiling dataset '%s' (corpus=%d, queries=%d)...",
                dataset.name, len(dataset.corpus), len(dataset.queries))

    token_regex = re.compile(config.token_pattern)

    # 1. Tokenize corpus passages
    doc_id_list = sorted(dataset.corpus.keys())
    doc_tokens_map: Dict[str, List[str]] = {}
    doc_lengths_list: List[int] = []

    for did in doc_id_list:
        doc = dataset.corpus[did]
        toks = tokenize(doc.get_text(), token_regex)
        doc_tokens_map[did] = toks
        doc_lengths_list.append(len(toks))

    doc_lengths = np.array(doc_lengths_list, dtype=np.int32)

    # 2. Tokenize queries
    query_id_list = sorted(dataset.queries.keys())
    query_tokens_map: Dict[str, List[str]] = {}
    query_lengths_all_list: List[int] = []

    for qid in query_id_list:
        qtext = dataset.queries[qid]
        toks = tokenize(qtext, token_regex)
        query_tokens_map[qid] = toks
        query_lengths_all_list.append(len(toks))

    query_lengths_all = np.array(query_lengths_all_list, dtype=np.int32)

    # Split-specific query lengths
    dev_qids = sorted(dataset.qrels_dev.keys())
    test_qids = sorted(dataset.qrels_test.keys())

    query_lengths_dev = np.array(
        [len(query_tokens_map[qid]) for qid in dev_qids if qid in query_tokens_map],
        dtype=np.int32,
    )
    query_lengths_test = np.array(
        [len(query_tokens_map[qid]) for qid in test_qids if qid in query_tokens_map],
        dtype=np.int32,
    )

    # 3. Compute length summaries
    doc_summary = compute_distribution_summary(doc_lengths, config.percentiles)
    query_summary_all = compute_distribution_summary(query_lengths_all, config.percentiles)
    query_summary_dev = compute_distribution_summary(query_lengths_dev, config.percentiles)
    query_summary_test = compute_distribution_summary(query_lengths_test, config.percentiles)

    # 4. Compute corpus smoothed BM25 IDF
    idf_map = compute_smoothed_bm25_idf(doc_tokens_map.values())

    # 5. Compute IDF coverage on relevant pairs
    def _compute_pairs_overlap(qrels_dict: Dict[str, Dict[str, float]]) -> List[float]:
        overlaps: List[float] = []
        for qid, judgments in qrels_dict.items():
            if qid not in query_tokens_map:
                continue
            q_toks = query_tokens_map[qid]
            for did, score in judgments.items():
                if score >= relevance_threshold:
                    d_toks = doc_tokens_map.get(did)
                    if d_toks is None:
                        # Known anomaly in some benchmarks (e.g., ArguAna #101)
                        continue
                    ov = compute_idf_overlap(q_toks, d_toks, idf_map)
                    overlaps.append(ov)
        return overlaps

    dev_relevant_overlaps = _compute_pairs_overlap(dataset.qrels_dev)
    test_relevant_overlaps = _compute_pairs_overlap(dataset.qrels_test)
    all_relevant_overlaps = dev_relevant_overlaps + test_relevant_overlaps

    cov_dev_arr = np.array(dev_relevant_overlaps, dtype=np.float64)
    cov_test_arr = np.array(test_relevant_overlaps, dtype=np.float64)
    cov_all_arr = np.array(all_relevant_overlaps, dtype=np.float64)

    # 6. Sample background random pairs
    random_pairs = sample_random_pairs(
        query_ids=query_id_list,
        doc_ids=doc_id_list,
        sample_size=config.random_pairs_sample_size,
        seed=seed,
    )

    random_overlaps: List[float] = []
    for qid, did in random_pairs:
        q_toks = query_tokens_map[qid]
        d_toks = doc_tokens_map[did]
        random_overlaps.append(compute_idf_overlap(q_toks, d_toks, idf_map))

    cov_random_arr = np.array(random_overlaps, dtype=np.float64)

    # 7. Compute coverage distribution summaries
    cov_summary_dev = compute_distribution_summary(cov_dev_arr, config.percentiles)
    cov_summary_test = compute_distribution_summary(cov_test_arr, config.percentiles)
    cov_summary_all = compute_distribution_summary(cov_all_arr, config.percentiles)
    cov_summary_random = compute_distribution_summary(cov_random_arr, config.percentiles)

    # 8. Compute separation metrics
    sep_dev = compute_separation_metrics(cov_dev_arr, cov_random_arr)
    sep_test = compute_separation_metrics(cov_test_arr, cov_random_arr)
    sep_all = compute_separation_metrics(cov_all_arr, cov_random_arr)

    return LexicalProfile(
        dataset_name=dataset.name,
        doc_lengths=doc_lengths,
        query_lengths_all=query_lengths_all,
        query_lengths_dev=query_lengths_dev,
        query_lengths_test=query_lengths_test,
        doc_summary=doc_summary,
        query_summary_all=query_summary_all,
        query_summary_dev=query_summary_dev,
        query_summary_test=query_summary_test,
        idf_map=idf_map,
        coverage_relevant_all=cov_all_arr,
        coverage_relevant_dev=cov_dev_arr,
        coverage_relevant_test=cov_test_arr,
        coverage_random_background=cov_random_arr,
        coverage_summary_relevant_all=cov_summary_all,
        coverage_summary_relevant_dev=cov_summary_dev,
        coverage_summary_relevant_test=cov_summary_test,
        coverage_summary_random=cov_summary_random,
        separation_all=sep_all,
        separation_dev=sep_dev,
        separation_test=sep_test,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically to disk using a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = path.parent
    prefix = f".tmp_{path.name}_"
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=temp_dir, prefix=prefix, delete=False, encoding="utf-8") as tf:
            temp_path = Path(tf.name)
            tf.write(content)
            tf.flush()
            os.fsync(tf.fileno())

        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def export_eda_artifacts(profile: LexicalProfile, output_dir: Path | str) -> Dict[str, Path]:
    """Export standardized JSON telemetry snapshot to output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_data = {
        "dataset_name": profile.dataset_name,
        "lengths": {
            "passages": profile.doc_summary.to_dict(),
            "queries_all": profile.query_summary_all.to_dict(),
            "queries_dev": profile.query_summary_dev.to_dict(),
            "queries_test": profile.query_summary_test.to_dict(),
        },
        "coverage": {
            "relevant_all": profile.coverage_summary_relevant_all.to_dict(),
            "relevant_dev": profile.coverage_summary_relevant_dev.to_dict(),
            "relevant_test": profile.coverage_summary_relevant_test.to_dict(),
            "random_background": profile.coverage_summary_random.to_dict(),
        },
        "separation": {
            "all": profile.separation_all.to_dict(),
            "dev": profile.separation_dev.to_dict(),
            "test": profile.separation_test.to_dict(),
        },
        "vocabulary_size": len(profile.idf_map),
    }

    json_path = out_path / "lexical_stats.json"
    _atomic_write_text(json_path, json.dumps(json_data, indent=2) + "\n")

    return {
        "lexical_stats_json": json_path,
    }
