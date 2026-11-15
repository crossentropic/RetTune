"""Handwritten Information Retrieval Evaluation & Statistical Significance Engine.

Why Handwritten Instead of `ranx`:
1. Zero Numba/LLVM Fragility: `ranx` depends on `numba` and `llvmlite`, which frequently
   introduce build and ABI breakages on Python 3.12+ and Apple Silicon.
2. Deterministic Tie-Breaking & Multi-Graded Transparency: Naive `np.argsort` breaks ties
   arbitrarily by index order, which silently corrupts evaluation when multiple documents
   share an imputed floor score or tied rank (e.g., zero-score tail padding in BM25 or
   cutoff floor imputation in Stage 4 hybrid fusion). This module enforces a single
   deterministic tie-breaking convention across all stages—sorting candidates by
   `(-score, doc_id)` (descending score, ascending lexicographical `doc_id`)—and implements
   exact TREC exponential gain (`2^r - 1`) with full-qrels `IDCG@K` normalization in
   dependency-free NumPy.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """Warm-cache query latency distribution and hardware-invariant work accounting."""
    num_queries: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    total_ms: float
    work_units_per_query: float = 0.0
    work_unit_name: str = "ops"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_queries": self.num_queries,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "total_ms": self.total_ms,
            "work_units_per_query": self.work_units_per_query,
            "work_unit_name": self.work_unit_name,
        }


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """Paired statistical comparison between system A (treatment) and system B (baseline)."""
    mean_a: float
    mean_b: float
    delta: float
    ci_lower: float
    ci_upper: float
    ci_half_width: float
    p_value: float
    num_queries: int
    confidence_level: float = 0.95

    @property
    def significant_at_05(self) -> bool:
        return self.p_value < 0.05

    @property
    def significant_at_01(self) -> bool:
        return self.p_value < 0.01

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "delta": self.delta,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_half_width": self.ci_half_width,
            "p_value": self.p_value,
            "num_queries": self.num_queries,
            "confidence_level": self.confidence_level,
            "significant_at_05": self.significant_at_05,
            "significant_at_01": self.significant_at_01,
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete retrieval evaluation result across a query split."""
    ndcg_at_k: float
    mrr_at_k: float
    recall_at_k: Dict[int, float]
    per_query_ndcg: Dict[str, float]
    per_query_mrr: Dict[str, float]
    per_query_recall: Dict[int, Dict[str, float]]
    num_queries: int
    primary_k: int = 10

    def to_summary_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            f"ndcg@{self.primary_k}": self.ndcg_at_k,
            f"mrr@{self.primary_k}": self.mrr_at_k,
            "num_queries": self.num_queries,
        }
        for k, val in sorted(self.recall_at_k.items()):
            res[f"recall@{k}"] = val
        return res


def rank_candidates(
    candidates: Mapping[str, float] | Iterable[Tuple[str, float]],
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Deterministically sort candidate documents by (-score, doc_id).

    Normalizes -0.0 to 0.0 prior to sorting so IEEE-754 signed zeros never affect ordering.
    """
    items = candidates.items() if isinstance(candidates, Mapping) else candidates
    cleaned: List[Tuple[str, float]] = [
        (str(doc_id), 0.0 if float(score) == 0.0 else float(score))
        for doc_id, score in items
    ]
    cleaned.sort(key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        return cleaned[:top_k]
    return cleaned


def topk_stable_indices(scores: np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
    """Return top-K indices sorted by (-score, index) in O(N + K log K) time.

    When `scores` is aligned with ascending `sorted_doc_ids`, breaking ties by ascending
    integer index is mathematically identical to breaking ties by ascending `doc_id`.
    """
    n = int(scores.size)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    s = np.asarray(scores, dtype=np.float64)
    s = np.where(s == 0.0, 0.0, s)

    if top_k is None or top_k >= n:
        return np.argsort(-s, kind="stable")

    k = int(top_k)
    if k <= 0:
        return np.empty(0, dtype=np.int64)

    kth_val = float(np.partition(s, n - k)[n - k])
    gt_idx = np.flatnonzero(s > kth_val)
    gt_sorted = gt_idx[np.argsort(-s[gt_idx], kind="stable")]
    rem = k - int(gt_sorted.size)
    if rem > 0:
        eq_idx = np.flatnonzero(s == kth_val)[:rem]
        return np.concatenate((gt_sorted, eq_idx))
    return gt_sorted


def rank_score_vector(
    scores: np.ndarray,
    sorted_doc_ids: Sequence[str],
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Deterministically rank a 1D score array aligned with ascending sorted_doc_ids.

    Because `sorted_doc_ids` is already in ascending lexicographical order, breaking ties
    by ascending index is mathematically identical to sorting by `(-score, doc_id)`.
    """
    if len(scores) != len(sorted_doc_ids):
        raise ValueError(
            f"Length mismatch: scores ({len(scores)}) != sorted_doc_ids ({len(sorted_doc_ids)})"
        )
    if len(scores) == 0:
        return []

    scores_f64 = np.asarray(scores, dtype=np.float64)
    scores_clean = np.where(scores_f64 == 0.0, 0.0, scores_f64)
    order = topk_stable_indices(scores_clean, top_k=top_k)
    return [(sorted_doc_ids[int(idx)], float(scores_clean[int(idx)])) for idx in order]


def compute_dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at cutoff K using exponential gain (2^r - 1).

    For 1-indexed rank i in [1, K]:
        DCG@K = sum_{i=1}^{min(K, len(relevances))} (2^{r_i} - 1) / log2(i + 1)
    """
    if k <= 0 or not relevances:
        return 0.0
    upto = min(k, len(relevances))
    rels = np.asarray(relevances[:upto], dtype=np.float64)
    rels = np.maximum(0.0, rels)
    gains = np.exp2(rels) - 1.0
    discounts = np.log2(np.arange(2, upto + 2, dtype=np.float64))
    return float(np.sum(gains / discounts))


def compute_idcg_at_k(
    qrel_judgments: Mapping[str, float],
    k: int,
    relevance_threshold: float = 0.0,
) -> float:
    """Compute Ideal DCG at cutoff K over all positive judgments in full qrels for a query."""
    if k <= 0 or not qrel_judgments:
        return 0.0
    pos_rels = sorted(
        (float(r) for r in qrel_judgments.values() if float(r) > 0.0 and float(r) >= relevance_threshold),
        reverse=True,
    )
    if not pos_rels:
        return 0.0
    return compute_dcg_at_k(pos_rels, k)


def compute_ndcg_at_k(
    ranked_doc_ids: Sequence[str],
    qrel_judgments: Mapping[str, float],
    k: int = 10,
    relevance_threshold: float = 0.0,
) -> float:
    """Compute NDCG@K for a deterministically ordered sequence of candidate doc_ids.

    Divides actual DCG@K by the full-qrels IDCG@K (computed across all positive judgments
    for the query in qrel_judgments, regardless of whether they were retrieved).
    """
    if k <= 0 or not qrel_judgments:
        return 0.0
    idcg = compute_idcg_at_k(qrel_judgments, k=k, relevance_threshold=relevance_threshold)
    if idcg <= 0.0:
        return 0.0
    upto = min(k, len(ranked_doc_ids))
    if upto == 0:
        return 0.0
    retrieved_rels = [
        float(qrel_judgments.get(did, 0.0))
        if float(qrel_judgments.get(did, 0.0)) >= relevance_threshold
        else 0.0
        for did in ranked_doc_ids[:upto]
    ]
    dcg = compute_dcg_at_k(retrieved_rels, k=k)
    return float(min(1.0, dcg / idcg))


def compute_recall_at_k(
    ranked_doc_ids: Sequence[str],
    qrel_judgments: Mapping[str, float],
    k: int = 100,
    relevance_threshold: float = 1.0,
) -> float:
    """Compute Recall@K for a deterministically ordered sequence of candidate doc_ids."""
    if k <= 0 or not qrel_judgments:
        return 0.0
    relevant_docs = {
        did for did, score in qrel_judgments.items() if float(score) >= relevance_threshold
    }
    if not relevant_docs:
        return 0.0
    retrieved_set = set(ranked_doc_ids[:k])
    hits = len(relevant_docs & retrieved_set)
    return float(hits / len(relevant_docs))


def compute_mrr_at_k(
    ranked_doc_ids: Sequence[str],
    qrel_judgments: Mapping[str, float],
    k: int = 10,
    relevance_threshold: float = 1.0,
) -> float:
    """Compute Reciprocal Rank at cutoff K using the given relevance threshold."""
    if k <= 0 or not qrel_judgments or not ranked_doc_ids:
        return 0.0
    upto = min(k, len(ranked_doc_ids))
    for rank_idx in range(upto):
        did = ranked_doc_ids[rank_idx]
        if float(qrel_judgments.get(did, 0.0)) >= relevance_threshold:
            return float(1.0 / (rank_idx + 1))
    return 0.0


def evaluate_run(
    run: Mapping[str, Mapping[str, float] | Sequence[Tuple[str, float]] | Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    primary_k: int = 10,
    recall_ks: Sequence[int] = (10, 20, 25, 50, 100, 200),
    relevance_threshold: float = 1.0,
    already_sorted: bool = False,
) -> EvaluationResult:
    """Evaluate a retrieval run across all queries in `qrels`.

    Enforces deterministic `(-score, doc_id)` tie-breaking on all candidate mappings or
    `(doc_id, score)` sequences unless `already_sorted=True` is explicitly provided for
    pre-sorted doc_id sequences.
    """
    query_ids = sorted(qrels.keys())
    unique_recall_ks = sorted(set(int(k) for k in recall_ks))
    max_needed_k = max([primary_k] + unique_recall_ks) if unique_recall_ks else primary_k

    per_query_ndcg: Dict[str, float] = {}
    per_query_mrr: Dict[str, float] = {}
    per_query_recall: Dict[int, Dict[str, float]] = {k: {} for k in unique_recall_ks}

    for qid in query_ids:
        q_judgments = qrels[qid]
        raw_candidates = run.get(qid)

        if raw_candidates is None:
            ranked_doc_ids: List[str] = []
        elif isinstance(raw_candidates, Mapping):
            ranked_pairs = rank_candidates(raw_candidates, top_k=max_needed_k)
            ranked_doc_ids = [did for did, _ in ranked_pairs]
        else:
            seq = list(raw_candidates)
            if not seq:
                ranked_doc_ids = []
            elif isinstance(seq[0], tuple):
                if already_sorted:
                    ranked_doc_ids = [str(item[0]) for item in seq[:max_needed_k]]  # type: ignore[index]
                else:
                    ranked_pairs = rank_candidates(seq, top_k=max_needed_k)  # type: ignore[arg-type]
                    ranked_doc_ids = [did for did, _ in ranked_pairs]
            else:
                ranked_doc_ids = [str(did) for did in seq[:max_needed_k]]

        # NDCG@K uses exponential gain on all positive judgments (>= 0, or threshold if > 0)
        # Note: in graded datasets like NFCorpus (grades 1, 2), threshold is 1.0 so all
        # positive judgments in qrels are included in both DCG and IDCG.
        per_query_ndcg[qid] = compute_ndcg_at_k(
            ranked_doc_ids,
            q_judgments,
            k=primary_k,
            relevance_threshold=min(relevance_threshold, 1e-9) if relevance_threshold <= 1.0 else relevance_threshold,
        )
        per_query_mrr[qid] = compute_mrr_at_k(
            ranked_doc_ids,
            q_judgments,
            k=primary_k,
            relevance_threshold=relevance_threshold,
        )
        for rk in unique_recall_ks:
            per_query_recall[rk][qid] = compute_recall_at_k(
                ranked_doc_ids,
                q_judgments,
                k=rk,
                relevance_threshold=relevance_threshold,
            )

    n_q = len(query_ids)
    if n_q == 0:
        return EvaluationResult(
            ndcg_at_k=0.0,
            mrr_at_k=0.0,
            recall_at_k={rk: 0.0 for rk in unique_recall_ks},
            per_query_ndcg={},
            per_query_mrr={},
            per_query_recall={rk: {} for rk in unique_recall_ks},
            num_queries=0,
            primary_k=primary_k,
        )

    mean_ndcg = float(np.mean(list(per_query_ndcg.values())))
    mean_mrr = float(np.mean(list(per_query_mrr.values())))
    mean_recall = {
        rk: float(np.mean(list(per_query_recall[rk].values())))
        for rk in unique_recall_ks
    }

    return EvaluationResult(
        ndcg_at_k=mean_ndcg,
        mrr_at_k=mean_mrr,
        recall_at_k=mean_recall,
        per_query_ndcg=per_query_ndcg,
        per_query_mrr=per_query_mrr,
        per_query_recall=per_query_recall,
        num_queries=n_q,
        primary_k=primary_k,
    )


def _align_paired_arrays(
    scores_a: Mapping[str, float] | Sequence[float] | np.ndarray,
    scores_b: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract aligned 1D float64 arrays from per-query mappings or sequences."""
    if isinstance(scores_a, Mapping) and isinstance(scores_b, Mapping):
        common_qids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
        if len(common_qids) != len(scores_a) or len(common_qids) != len(scores_b):
            raise ValueError(
                "Paired mappings must share the exact same query IDs "
                f"(got {len(scores_a)} vs {len(scores_b)}, common={len(common_qids)})."
            )
        arr_a = np.array([float(scores_a[q]) for q in common_qids], dtype=np.float64)
        arr_b = np.array([float(scores_b[q]) for q in common_qids], dtype=np.float64)
    else:
        arr_a = np.asarray(scores_a, dtype=np.float64)
        arr_b = np.asarray(scores_b, dtype=np.float64)
        if arr_a.shape != arr_b.shape:
            raise ValueError(
                f"Paired score sequences must have identical shape: {arr_a.shape} != {arr_b.shape}"
            )
    return arr_a, arr_b


def paired_permutation_test(
    scores_a: Mapping[str, float] | Sequence[float] | np.ndarray,
    scores_b: Mapping[str, float] | Sequence[float] | np.ndarray,
    n_resamples: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation (randomization) test via deterministic sign flips.

    Under the null hypothesis H0 that systems A and B are exchangeable on each query q,
    the per-query difference d_q = m_A(q) - m_B(q) has symmetric sign (+d_q or -d_q
    with probability 0.5). Generates `n_resamples` random sign vectors using `seed`
    and computes the fraction of permutations where |mean(s * d)| >= |mean(d)|.
    """
    arr_a, arr_b = _align_paired_arrays(scores_a, scores_b)
    n_q = int(arr_a.size)
    if n_q == 0:
        return 1.0

    diffs = arr_a - arr_b
    obs_abs_diff = float( abs(np.mean(diffs)) )
    if obs_abs_diff <= 1e-15 or np.allclose(diffs, 0.0, atol=1e-15):
        return 1.0

    rng = np.random.default_rng(seed)
    # Process in chunks of 2,500 resamples to keep peak memory tiny
    chunk_size = 2500
    exceed_count = 0
    remaining = n_resamples
    threshold = obs_abs_diff - 1e-12

    while remaining > 0:
        cur = min(chunk_size, remaining)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(cur, n_q))
        perm_abs_means = np.abs((signs @ diffs) / n_q)
        exceed_count += int(np.sum(perm_abs_means >= threshold))
        remaining -= cur

    return float(exceed_count / n_resamples)


def paired_bootstrap_ci(
    scores_a: Mapping[str, float] | Sequence[float] | np.ndarray,
    scores_b: Mapping[str, float] | Sequence[float] | np.ndarray,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute paired bootstrap confidence interval for mean(A - B).

    Returns:
        (mean_diff, ci_lower, ci_upper)
    """
    arr_a, arr_b = _align_paired_arrays(scores_a, scores_b)
    n_q = int(arr_a.size)
    if n_q == 0:
        return 0.0, 0.0, 0.0

    diffs = arr_a - arr_b
    mean_diff = float(np.mean(diffs))
    if n_q == 1 or np.allclose(diffs, diffs[0], atol=1e-15):
        return mean_diff, mean_diff, mean_diff

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_q, size=(n_resamples, n_q))
    boot_means = np.mean(diffs[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower, ci_upper = np.percentile(boot_means, [lower_pct, upper_pct])
    return mean_diff, float(ci_lower), float(ci_upper)


def compare_paired_systems(
    scores_a: Mapping[str, float] | Sequence[float] | np.ndarray,
    scores_b: Mapping[str, float] | Sequence[float] | np.ndarray,
    n_permutations: int = 10000,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> SignificanceResult:
    """Run both paired permutation test and paired bootstrap CI on per-query scores."""
    arr_a, arr_b = _align_paired_arrays(scores_a, scores_b)
    n_q = int(arr_a.size)
    if n_q == 0:
        return SignificanceResult(
            mean_a=0.0,
            mean_b=0.0,
            delta=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            ci_half_width=0.0,
            p_value=1.0,
            num_queries=0,
            confidence_level=confidence_level,
        )

    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    delta, ci_lower, ci_upper = paired_bootstrap_ci(
        arr_a,
        arr_b,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        seed=seed,
    )
    p_val = paired_permutation_test(
        arr_a,
        arr_b,
        n_resamples=n_permutations,
        seed=seed,
    )
    half_width = float((ci_upper - ci_lower) / 2.0)

    return SignificanceResult(
        mean_a=mean_a,
        mean_b=mean_b,
        delta=delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_half_width=half_width,
        p_value=p_val,
        num_queries=n_q,
        confidence_level=confidence_level,
    )


def summarize_latencies_ms(
    latencies_ms: Sequence[float] | np.ndarray,
    work_units_per_query: float = 0.0,
    work_unit_name: str = "ops",
) -> LatencyProfile:
    """Summarize per-query latencies (in milliseconds) into p50, p95, p99 and mean."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return LatencyProfile(
            num_queries=0,
            mean_ms=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            total_ms=0.0,
            work_units_per_query=work_units_per_query,
            work_unit_name=work_unit_name,
        )

    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return LatencyProfile(
        num_queries=int(arr.size),
        mean_ms=float(np.mean(arr)),
        p50_ms=float(p50),
        p95_ms=float(p95),
        p99_ms=float(p99),
        total_ms=float(np.sum(arr)),
        work_units_per_query=float(work_units_per_query),
        work_unit_name=work_unit_name,
    )


def profile_query_execution(
    queries: Sequence[T],
    fn: Callable[[T], Any],
    work_units_per_query: float = 0.0,
    work_unit_name: str = "ops",
    warmup_count: int = 3,
) -> Tuple[List[Any], LatencyProfile]:
    """Execute `fn(q)` across `queries` with high-precision `perf_counter_ns` timing.

    Performs optional warm-up iterations on the first few queries so reported latencies
    reflect warm-cache steady-state execution.
    """
    if not queries:
        return [], summarize_latencies_ms([], work_units_per_query, work_unit_name)

    n_warm = min(warmup_count, len(queries))
    for i in range(n_warm):
        _ = fn(queries[i])

    results: List[Any] = []
    latencies_ms: List[float] = []
    for q in queries:
        t0 = time.perf_counter_ns()
        res = fn(q)
        t1 = time.perf_counter_ns()
        results.append(res)
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    profile = summarize_latencies_ms(
        latencies_ms,
        work_units_per_query=work_units_per_query,
        work_unit_name=work_unit_name,
    )
    return results, profile
