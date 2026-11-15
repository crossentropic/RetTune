"""Unit and integration tests for Stage 4 Calibration, Imputation & Hybrid Fusion."""

from pathlib import Path
import numpy as np
import pytest

from rettune.calibration import (
    QueryScoreNormalizer,
    calibrate_top_k_candidates,
    fit_query_normalizer,
)
from rettune.config import BenchmarkConfig, EvaluationConfig, FusionConfig
from rettune.data_loader import load_dataset
from rettune.fusion import (
    FusionDatasetReport,
    compute_ood_alpha_transfer_matrix,
    export_fusion_artifacts,
    fuse_query_convex,
    fuse_query_rrf,
    precalibrate_query_union,
    run_fusion_benchmark_for_dataset,
)


def test_natural_fixed_k_sparse_padding_single_and_zero_matches():
    """Verify natural fixed-K sparse padding handles M=1 and M=0 matches without ad-hoc branches."""
    # Case 1: Only M=1 document matches in sparse retrieval with score 15.0 (window_k=100)
    # The remaining 99 slots carry their true mathematical BM25 score of 0.0 -> s_min = 0.0, s_max = 15.0
    norm_m1 = fit_query_normalizer([15.0], method="minmax", window_k=100, pad_with_zeros=True)
    assert norm_m1.s_min == 0.0
    assert norm_m1.s_max == 15.0
    assert pytest.approx(norm_m1.transform_scalar(15.0), rel=1e-6) == 1.0
    assert pytest.approx(norm_m1.transform_scalar(0.0), abs=1e-7) == 0.0
    assert pytest.approx(norm_m1.cutoff_floor_score, abs=1e-7) == 0.0

    # Verify calibrate_top_k_candidates filters out 0.0-scored sparse slots from active dict
    # while fitting normalizer over all K=100 slots
    sparse_cands = [("doc_hit", 15.0), ("doc_unmatched_1", 0.0), ("doc_unmatched_2", 0.0)]
    s_map, s_norm = calibrate_top_k_candidates(
        sparse_cands, method="minmax", window_k=100, is_sparse=True
    )
    assert list(s_map.keys()) == ["doc_hit"]
    assert pytest.approx(s_map["doc_hit"], rel=1e-6) == 1.0
    assert s_norm.s_min == 0.0

    # Case 2: M=0 matches (all 0.0) -> epsilon guard prevents ZeroDivisionError and returns 0.0
    norm_m0 = fit_query_normalizer([], method="minmax", window_k=100, pad_with_zeros=True)
    assert norm_m0.transform_scalar(0.0) == 0.0


def test_zscore_naive_zero_imputation_bug_proof_vs_topk_floor():
    """Prove that naive 0.0 imputation under Z-score causes rank inversions while topk_floor does not."""
    # Top-4 dense candidates with linearly spaced scores [0.9, 0.7, 0.5, 0.3] -> mean = 0.6
    # Ranks 3 and 4 (scores 0.5 and 0.3) have NEGATIVE Z-scores!
    dense_cands = [("d1", 0.9), ("d2", 0.7), ("d3", 0.5), ("d4", 0.3)]
    # Sparse retrieves 'd_sparse_only' which is missing from dense top-4
    sparse_cands = [("d_sparse_only", 10.0), ("d1", 8.0), ("d2", 6.0), ("d3", 4.0)]

    # 1. Under `naive_zero`, 'd_sparse_only' gets dense score 0.0 (the mean of top-4 dense hits!),
    # which is higher than retrieved dense hits d3 (z < 0) and d4 (z < 0) -> rank inversion!
    pq_naive = precalibrate_query_union(
        query_id="q1",
        sparse_candidates=sparse_cands,
        dense_candidates=dense_cands,
        calibration_method="zscore",
        imputation_regime="naive_zero",
        top_k=4,
    )
    assert pq_naive.dense_rank_inversions > 0
    idx_sparse_only = pq_naive.union_doc_ids.index("d_sparse_only")
    idx_d4 = pq_naive.union_doc_ids.index("d4")
    # Unretrieved doc 'd_sparse_only' falsely receives a higher dense score (0.0) than retrieved 'd4'!
    assert pq_naive.dense_scores[idx_sparse_only] == 0.0
    assert pq_naive.dense_scores[idx_d4] < 0.0
    assert pq_naive.dense_scores[idx_sparse_only] > pq_naive.dense_scores[idx_d4]

    # 2. Under `topk_floor`, 'd_sparse_only' receives the cutoff boundary score s_tilde_(K) = z(d4),
    # guaranteeing zero rank inversions!
    pq_floor = precalibrate_query_union(
        query_id="q1",
        sparse_candidates=sparse_cands,
        dense_candidates=dense_cands,
        calibration_method="zscore",
        imputation_regime="topk_floor",
        top_k=4,
    )
    assert pq_floor.sparse_rank_inversions == 0
    assert pq_floor.dense_rank_inversions == 0
    assert pq_floor.dense_scores[idx_sparse_only] == pytest.approx(pq_floor.dense_scores[idx_d4])

    # 3. Under `exact_dual_lookup`, true score of 'd_sparse_only' (0.1 < 0.3) is transformed by the
    # exact same top-4 dense normalizer -> strictly below d4, zero rank inversions!
    doc_ids = ["d1", "d2", "d3", "d4", "d_sparse_only"]
    doc_to_idx = {d: i for i, d in enumerate(doc_ids)}
    sparse_row = np.array([8.0, 6.0, 4.0, 1.0, 10.0])
    dense_row = np.array([0.9, 0.7, 0.5, 0.3, 0.1])

    pq_exact = precalibrate_query_union(
        query_id="q1",
        sparse_candidates=sparse_cands,
        dense_candidates=dense_cands,
        calibration_method="zscore",
        imputation_regime="exact_dual_lookup",
        top_k=4,
        doc_to_idx=doc_to_idx,
        sparse_full_row=sparse_row,
        dense_full_row=dense_row,
    )
    assert pq_exact.sparse_rank_inversions == 0
    assert pq_exact.dense_rank_inversions == 0
    assert pq_exact.dense_scores[idx_sparse_only] < pq_exact.dense_scores[idx_d4]

    # Verify normalizer invariance: retrieved candidates d1, d2, d3, d4 have the EXACT same
    # normalized dense scores across all three imputation regimes!
    for d_ret in ("d1", "d2", "d3", "d4"):
        j = pq_naive.union_doc_ids.index(d_ret)
        assert pq_naive.dense_scores[j] == pytest.approx(pq_floor.dense_scores[j])
        assert pq_floor.dense_scores[j] == pytest.approx(pq_exact.dense_scores[j])


def test_zscore_sigmoid_bounds_and_outlier_mitigation():
    """Verify Z-Score + Sigmoid bounds scores in (0, 1) and mitigates top-1 BM25 outlier compression."""
    # In a K=100 window, a single top-1 BM25 outlier spike (100.0) dominates (s_max - s_min = 98.0),
    # whereas sigma across all 100 candidates remains anchored by the 99 bulk candidates (12.0..2.0)
    scores = [100.0] + list(np.linspace(12.0, 2.0, 99))
    norm_mm = fit_query_normalizer(scores, method="minmax", window_k=100, pad_with_zeros=False)
    norm_zs = fit_query_normalizer(scores, method="zscore_sigmoid", window_k=100, pad_with_zeros=False)

    mm_vals = norm_mm.transform_array(scores)
    zs_vals = norm_zs.transform_array(scores)

    # All zscore_sigmoid outputs are strictly in (0, 1) and monotonically decreasing
    assert np.all((zs_vals > 0.0) & (zs_vals < 1.0))
    assert np.all(np.diff(zs_vals) < 0.0)

    # Under Min-Max, the single top-1 outlier compresses the spread between Rank 2 (12.0) and Rank 100 (2.0)
    # to just (12 - 2) / (100 - 2) = 0.102, whereas Z-score + Sigmoid preserves a 2.5x wider spread!
    spread_mm = mm_vals[1] - mm_vals[-1]
    spread_zs = zs_vals[1] - zs_vals[-1]
    assert spread_zs > 2.0 * spread_mm


def test_rrf_math_and_zero_score_filtering():
    """Verify RRF formula and ensure 0.0-scored sparse padding slots do not receive RRF rank bonuses."""
    sparse_cands = [("d_both", 5.0), ("d_unmatched_zero", 0.0)]
    dense_cands = [("d_dense_top", 0.9), ("d_both", 0.8)]

    fused = fuse_query_rrf(sparse_cands, dense_cands, k0=60, top_k=10)
    fused_dict = dict(fused)

    # d_both is rank 1 in sparse and rank 2 in dense -> 1/61 + 1/62
    assert pytest.approx(fused_dict["d_both"]) == (1.0 / 61.0 + 1.0 / 62.0)
    # d_dense_top is rank 1 in dense only -> 1/61
    assert pytest.approx(fused_dict["d_dense_top"]) == (1.0 / 61.0)
    # d_unmatched_zero had raw BM25 score 0.0 -> must NOT appear in RRF output
    assert "d_unmatched_zero" not in fused_dict


def test_run_fusion_benchmark_and_ood_transfer_end_to_end(
    synthetic_benchmark_config: BenchmarkConfig,
    tmp_path: Path,
):
    """Verify Stage 4A, 4B, and 4C execution and JSON export on synthetic dataset."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)
    doc_ids = sorted(dataset.corpus.keys())
    dev_qids = sorted(dataset.qrels_dev.keys())
    test_qids = sorted(dataset.qrels_test.keys())

    # Construct synthetic sparse and dense runs & matrices
    s_dev = {q: [(doc_ids[0], 5.0), (doc_ids[1], 2.0)] for q in dev_qids}
    s_test = {q: [(doc_ids[2], 5.0), (doc_ids[0], 2.0)] for q in test_qids}
    d_dev = {q: [(doc_ids[1], 0.9), (doc_ids[0], 0.8), (doc_ids[2], 0.4)] for q in dev_qids}
    d_test = {q: [(doc_ids[0], 0.9), (doc_ids[2], 0.8), (doc_ids[3], 0.4)] for q in test_qids}

    s_dev_mat = np.ones((len(dev_qids), len(doc_ids)), dtype=np.float64)
    s_test_mat = np.ones((len(test_qids), len(doc_ids)), dtype=np.float64)
    d_dev_mat = np.ones((len(dev_qids), len(doc_ids)), dtype=np.float64) * 0.5
    d_test_mat = np.ones((len(test_qids), len(doc_ids)), dtype=np.float64) * 0.5

    fusion_cfg = FusionConfig(top_k=5, rrf_k0_default=60, rrf_k0_grid=[20, 60], alpha_step=0.25)
    eval_cfg = EvaluationConfig(
        primary_k=5,
        recall_ks=[5, 20, 100],
        candidate_pool_k=5,
        permutation_n_resamples=200,
        bootstrap_n_resamples=100,
    )

    rep = run_fusion_benchmark_for_dataset(
        dataset=dataset,
        doc_ids=doc_ids,
        sparse_dev_run=s_dev,
        sparse_test_run=s_test,
        dense_dev_run=d_dev,
        dense_test_run=d_test,
        sparse_dev_mat=s_dev_mat,
        sparse_test_mat=s_test_mat,
        dense_dev_mat=d_dev_mat,
        dense_test_mat=d_test_mat,
        fusion_cfg=fusion_cfg,
        eval_cfg=eval_cfg,
        relevance_threshold=1.0,
        seed=42,
    )
    assert isinstance(rep, FusionDatasetReport)
    assert "zscore_sigmoid" in rep.calibration_progression_4b
    assert "zscore" in rep.imputation_experiment_4a

    transfer = compute_ood_alpha_transfer_matrix({"synthetic": rep})
    assert len(transfer) == 1
    assert transfer[0].transfer_gap_vs_in_domain == pytest.approx(0.0)

    paths = export_fusion_artifacts({"synthetic": rep}, transfer, tmp_path / "metrics")
    assert paths["synthetic"].exists()
    assert paths["summary"].exists()
