"""Unit tests for Stage 2.1 Handwritten Evaluation Harness & Statistical Engine."""

import math
import numpy as np
import pytest

from rettune.evaluation import (
    EvaluationResult,
    LatencyProfile,
    SignificanceResult,
    compare_paired_systems,
    compute_dcg_at_k,
    compute_idcg_at_k,
    compute_mrr_at_k,
    compute_ndcg_at_k,
    compute_recall_at_k,
    evaluate_run,
    paired_bootstrap_ci,
    paired_permutation_test,
    profile_query_execution,
    rank_candidates,
    rank_score_vector,
    summarize_latencies_ms,
)


def test_deterministic_tie_breaking_mapping_and_vector():
    """Verify (-score, doc_id) tie-breaking across mapping and score vector rankers."""
    candidates = {
        "doc_10": 1.5,
        "doc_2": 0.0,
        "doc_1": 1.5,
        "0042": -0.0,
        "99": 2.0,
        "doc_3": 0.0,
    }
    ranked_map = rank_candidates(candidates)
    expected_order = [
        ("99", 2.0),
        ("doc_1", 1.5),
        ("doc_10", 1.5),
        ("0042", 0.0),
        ("doc_2", 0.0),
        ("doc_3", 0.0),
    ]
    assert ranked_map == expected_order

    # Verify top_k slicing
    assert rank_candidates(candidates, top_k=3) == expected_order[:3]

    # Verify rank_score_vector produces the exact same result when aligned with sorted_doc_ids
    sorted_doc_ids = sorted(candidates.keys())
    scores = np.array([candidates[d] for d in sorted_doc_ids], dtype=np.float64)
    ranked_vec = rank_score_vector(scores, sorted_doc_ids)
    assert ranked_vec == expected_order
    assert rank_score_vector(scores, sorted_doc_ids, top_k=3) == expected_order[:3]


def test_rank_score_vector_validation():
    """Verify rank_score_vector handles empty input and rejects length mismatches."""
    assert rank_score_vector(np.array([]), []) == []
    with pytest.raises(ValueError, match="Length mismatch"):
        rank_score_vector(np.array([1.0, 2.0]), ["doc_1"])


def test_multi_graded_ndcg_and_full_idcg():
    """Verify TREC exponential gain (2^r - 1) and full-qrels IDCG normalization."""
    # Query has 3 relevant docs in qrels with grades 2, 1, 1
    qrels_q = {"d_best": 2.0, "d_mid1": 1.0, "d_mid2": 1.0, "d_zero": 0.0}

    # Ideal gains at K=2: grades [2.0, 1.0] -> (2^2 - 1)/log2(2) + (2^1 - 1)/log2(3)
    expected_idcg_2 = 3.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
    assert pytest.approx(compute_idcg_at_k(qrels_q, k=2), rel=1e-7) == expected_idcg_2

    # Perfect ranking at K=2 -> NDCG@2 == 1.0
    assert pytest.approx(compute_ndcg_at_k(["d_best", "d_mid1"], qrels_q, k=2), rel=1e-7) == 1.0

    # Suboptimal ranking at K=2: ["d_mid1", "d_best"] -> gains [1.0, 3.0]
    expected_dcg_sub = 1.0 / math.log2(2.0) + 3.0 / math.log2(3.0)
    expected_ndcg_sub = expected_dcg_sub / expected_idcg_2
    assert (
        pytest.approx(compute_ndcg_at_k(["d_mid1", "d_best"], qrels_q, k=2), rel=1e-7)
        == expected_ndcg_sub
    )
    assert expected_ndcg_sub < 1.0

    # Candidate list missing d_best entirely still normalizes by FULL qrels IDCG (including d_best!)
    ndcg_missing_best = compute_ndcg_at_k(["d_mid1", "d_mid2"], qrels_q, k=2)
    expected_dcg_missing = 1.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
    assert pytest.approx(ndcg_missing_best, rel=1e-7) == expected_dcg_missing / expected_idcg_2


def test_recall_and_mrr_at_k():
    """Verify Recall@K and MRR@K calculations and edge cases."""
    qrels_q = {"d1": 1.0, "d2": 2.0}
    ranked = ["d_irrel", "d2", "d_other", "d1"]

    # At K=1: 0 hits
    assert compute_recall_at_k(ranked, qrels_q, k=1) == 0.0
    assert compute_mrr_at_k(ranked, qrels_q, k=1) == 0.0

    # At K=2: d2 is at rank 2 (1-indexed) -> Recall@2 = 1/2 = 0.5, MRR@2 = 1/2 = 0.5
    assert pytest.approx(compute_recall_at_k(ranked, qrels_q, k=2)) == 0.5
    assert pytest.approx(compute_mrr_at_k(ranked, qrels_q, k=2)) == 0.5

    # At K=4: both d2 and d1 retrieved -> Recall@4 = 1.0, MRR@4 still 0.5 (first hit at rank 2)
    assert pytest.approx(compute_recall_at_k(ranked, qrels_q, k=4)) == 1.0
    assert pytest.approx(compute_mrr_at_k(ranked, qrels_q, k=4)) == 0.5

    # Empty or zero-judgment edge cases
    assert compute_dcg_at_k([], k=10) == 0.0
    assert compute_ndcg_at_k([], qrels_q, k=10) == 0.0
    assert compute_ndcg_at_k(ranked, {}, k=10) == 0.0
    assert compute_recall_at_k(ranked, {}, k=10) == 0.0
    assert compute_mrr_at_k([], qrels_q, k=10) == 0.0


def test_evaluate_run_end_to_end_and_tie_invariance():
    """Verify evaluate_run handles dict runs, tuple runs, missing queries, and tied scores."""
    qrels = {
        "q1": {"doc_a": 2.0, "doc_b": 1.0},
        "q2": {"doc_c": 1.0},
        "q3": {"doc_d": 1.0},  # Missing from run -> should score 0.0
    }
    # In q1, doc_a and doc_b have tied score 1.0 -> lexicographical tie-breaker puts doc_a before doc_b!
    run = {
        "q1": {"doc_b": 1.0, "doc_a": 1.0, "doc_z": 0.5},
        "q2": [("doc_x", 2.0), ("doc_c", 1.0)],
    }

    res = evaluate_run(run, qrels, primary_k=10, recall_ks=[1, 2, 10])
    assert isinstance(res, EvaluationResult)
    assert res.num_queries == 3
    # q1: doc_a (grade 2) ranked 1st, doc_b (grade 1) ranked 2nd -> NDCG@10 = 1.0, MRR@10 = 1.0, Recall@1 = 0.5
    assert pytest.approx(res.per_query_ndcg["q1"]) == 1.0
    assert pytest.approx(res.per_query_mrr["q1"]) == 1.0
    assert pytest.approx(res.per_query_recall[1]["q1"]) == 0.5
    assert pytest.approx(res.per_query_recall[2]["q1"]) == 1.0

    # q2: doc_c at rank 2 -> MRR@10 = 0.5, NDCG@10 = (1/log2(3)) / (1/log2(2)) = 1/log2(3)
    assert pytest.approx(res.per_query_mrr["q2"]) == 0.5
    assert pytest.approx(res.per_query_ndcg["q2"]) == 1.0 / math.log2(3.0)

    # q3: missing from run -> 0.0
    assert res.per_query_ndcg["q3"] == 0.0
    assert res.per_query_mrr["q3"] == 0.0
    assert res.per_query_recall[10]["q3"] == 0.0


def test_paired_permutation_and_bootstrap_statistical_engine():
    """Verify paired permutation p-values and bootstrap 95% CIs."""
    # 1. Identical runs -> delta = 0.0, p_value = 1.0
    scores_ident = {f"q{i}": 0.5 for i in range(30)}
    assert paired_permutation_test(scores_ident, scores_ident, n_resamples=1000, seed=42) == 1.0
    delta, lo, hi = paired_bootstrap_ci(scores_ident, scores_ident, n_resamples=500, seed=42)
    assert delta == 0.0 and lo == 0.0 and hi == 0.0

    # 2. Strictly dominant system A over system B across 50 queries -> p_value < 0.001, CI > 0
    rng = np.random.default_rng(123)
    base = rng.uniform(0.2, 0.6, size=50)
    treat = base + rng.uniform(0.05, 0.15, size=50)

    sig = compare_paired_systems(
        treat, base, n_permutations=5000, n_bootstrap=1000, confidence_level=0.95, seed=42
    )
    assert isinstance(sig, SignificanceResult)
    assert sig.delta > 0.05
    assert sig.ci_lower > 0.0
    assert sig.ci_upper > sig.ci_lower
    assert sig.p_value < 0.005
    assert sig.significant_at_05 is True
    assert sig.significant_at_01 is True

    # 3. Mismatched keys raise ValueError
    with pytest.raises(ValueError, match="exact same query IDs"):
        paired_permutation_test({"q1": 1.0}, {"q2": 1.0})


def test_latency_profiler():
    """Verify high-precision query timer and percentile summarizer."""
    items = [1, 2, 3, 4, 5]
    outputs, profile = profile_query_execution(
        items,
        fn=lambda x: x * 2,
        work_units_per_query=1024.0,
        work_unit_name="FLOPs",
        warmup_count=2,
    )
    assert outputs == [2, 4, 6, 8, 10]
    assert isinstance(profile, LatencyProfile)
    assert profile.num_queries == 5
    assert profile.p50_ms >= 0.0
    assert profile.p95_ms >= profile.p50_ms
    assert profile.p99_ms >= profile.p95_ms
    assert profile.work_units_per_query == 1024.0
    assert profile.work_unit_name == "FLOPs"

    empty_prof = summarize_latencies_ms([])
    assert empty_prof.num_queries == 0
    assert empty_prof.p50_ms == 0.0
