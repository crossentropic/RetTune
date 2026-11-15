"""Unit and integration tests for Stage 5 Cross-Encoder Reranking & Candidate Pool Ceiling."""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from rettune.config import BenchmarkConfig, EvaluationConfig, RerankerConfig
from rettune.data_loader import Document, IRDataset, load_dataset
from rettune.reranker import (
    RerankerCacheBundle,
    RerankerDatasetReport,
    SystemTradeoffPoint,
    build_and_save_reranker_cache,
    collect_deduplicated_candidate_pairs,
    compute_candidate_pool_ceiling_and_reranked,
    compute_max_corpus_ndcg_at_k,
    export_reranker_artifacts,
    load_reranker_cache,
    mark_non_dominated_systems,
    run_reranker_benchmark_for_dataset,
)


def test_candidate_pool_ceiling_additive_identity_and_bounds():
    """Verify exact additive error decomposition and Ceiling@K >= Reranked@K on graded & missing-doc queries."""
    corpus = {
        "d_grade2": Document(id="d_grade2", text="highly relevant clinical evidence"),
        "d_grade1": Document(id="d_grade1", text="partially relevant trial"),
        "d_irrel1": Document(id="d_irrel1", text="unrelated noise passage one"),
        "d_irrel2": Document(id="d_irrel2", text="unrelated noise passage two"),
    }
    queries = {"q1": "clinical evidence", "q_beir101": "missing corpus doc query"}
    # q1 has d_grade2 (2.0) and d_grade1 (1.0) in corpus.
    # q_beir101 references d_missing_from_corpus (1.0), simulating ArguAna BEIR #101!
    qrels_test = {
        "q1": {"d_grade2": 2.0, "d_grade1": 1.0},
        "q_beir101": {"d_missing_from_corpus": 1.0},
    }
    ds = IRDataset(
        name="ceiling_test",
        corpus=corpus,
        queries=queries,
        qrels_dev=qrels_test,
        qrels_test=qrels_test,
    )

    # MaxCorpusNDCG@10: q1 has 1.0, q_beir101 has 0.0 -> mean = 0.5
    max_ndcg, per_q_max = compute_max_corpus_ndcg_at_k(ds, qrels_test, k=10)
    assert per_q_max["q1"] == pytest.approx(1.0)
    assert per_q_max["q_beir101"] == pytest.approx(0.0)
    assert max_ndcg == pytest.approx(0.5)

    # Suppose first-stage candidate pool of size K=3 for q1 retrieved d_grade1 (1.0) and two irrelevant docs,
    # MISSING d_grade2 (2.0)! And Cross-Encoder ranks d_irrel1 above d_grade1 (a Stage-2 scoring error).
    first_stage_run = {
        "q1": [("d_irrel2", 0.9), ("d_grade1", 0.8), ("d_irrel1", 0.7)],
        "q_beir101": [("d_irrel1", 0.9), ("d_irrel2", 0.8)],
    }
    reranker_scores = {
        "q1": {"d_irrel1": 5.0, "d_grade1": 2.0, "d_irrel2": -1.0},
        "q_beir101": {"d_irrel1": 1.0, "d_irrel2": 0.5},
    }

    eval_cfg = EvaluationConfig(
        primary_k=10,
        recall_ks=[10, 20, 100],
        candidate_pool_k=10,
        permutation_n_resamples=200,
        bootstrap_n_resamples=100,
    )
    pt = compute_candidate_pool_ceiling_and_reranked(
        pipeline_name="hybrid",
        pipeline_label="Hybrid",
        first_stage_run=first_stage_run,
        reranker_scores=reranker_scores,
        dataset=ds,
        qrels=qrels_test,
        candidate_k=3,
        eval_cfg=eval_cfg,
    )

    # Strict non-negativity and ordering invariants
    assert 0.0 <= pt.reranked_ndcg10 <= pt.ceiling_ndcg10 <= pt.max_corpus_ndcg10
    assert pt.reranker_scoring_error > 0.0
    assert pt.candidate_pool_miss_error > 0.0

    # Exact additive identity: Reranked + ScoringError + PoolMiss == MaxCorpusNDCG@10
    assert pytest.approx(
        pt.reranked_ndcg10 + pt.reranker_scoring_error + pt.candidate_pool_miss_error,
        rel=1e-9,
    ) == pt.max_corpus_ndcg10


def test_non_dominated_system_identification():
    """Verify non-dominated system marking in (latency_p95_ms, ndcg10) space."""
    pts = [
        SystemTradeoffPoint(
            system_id="fast_low", display_label="Fast", pipeline_family="sparse", rerank_k=0,
            ndcg10=0.40, recall100=0.8, mrr10=0.4,
            t_sparse_p50_ms=0.5, t_sparse_p95_ms=1.0,
            t_dense_encode_p50_ms=0, t_dense_encode_p95_ms=0,
            t_dense_scan_p50_ms=0, t_dense_scan_p95_ms=0,
            t_fusion_p50_ms=0, t_fusion_p95_ms=0,
            t_rerank_p50_ms=0, t_rerank_p95_ms=0,
            total_p50_ms=0.5, total_p95_ms=1.0,
            matrix_flops_per_query=0, vector_ram_mb=0, postings_per_query=100, reranker_tokens_per_query=0,
        ),
        SystemTradeoffPoint(
            system_id="dominated_mid", display_label="Dominated", pipeline_family="dense", rerank_k=0,
            ndcg10=0.38, recall100=0.8, mrr10=0.38,  # Slower (5ms > 1ms) AND worse NDCG (0.38 < 0.40)!
            t_sparse_p50_ms=0, t_sparse_p95_ms=0,
            t_dense_encode_p50_ms=4.0, t_dense_encode_p95_ms=4.5,
            t_dense_scan_p50_ms=0.4, t_dense_scan_p95_ms=0.5,
            t_fusion_p50_ms=0, t_fusion_p95_ms=0,
            t_rerank_p50_ms=0, t_rerank_p95_ms=0,
            total_p50_ms=4.4, total_p95_ms=5.0,
            matrix_flops_per_query=1000, vector_ram_mb=1.0, postings_per_query=0, reranker_tokens_per_query=0,
        ),
        SystemTradeoffPoint(
            system_id="accurate_high", display_label="Accurate", pipeline_family="hybrid", rerank_k=20,
            ndcg10=0.55, recall100=0.9, mrr10=0.55,
            t_sparse_p50_ms=0.5, t_sparse_p95_ms=1.0,
            t_dense_encode_p50_ms=4.0, t_dense_encode_p95_ms=4.5,
            t_dense_scan_p50_ms=0.4, t_dense_scan_p95_ms=0.5,
            t_fusion_p50_ms=0.1, t_fusion_p95_ms=0.2,
            t_rerank_p50_ms=10.0, t_rerank_p95_ms=12.0,
            total_p50_ms=15.0, total_p95_ms=18.2,
            matrix_flops_per_query=1000, vector_ram_mb=1.0, postings_per_query=100, reranker_tokens_per_query=2500,
        ),
    ]
    marked = mark_non_dominated_systems(pts)
    by_id = {p.system_id: p.is_non_dominated for p in marked}
    assert by_id["fast_low"] is True
    assert by_id["dominated_mid"] is False
    assert by_id["accurate_high"] is True


def test_deduplicated_cache_build_and_stage5_benchmark(
    synthetic_benchmark_config: BenchmarkConfig,
    tmp_path: Path,
):
    """Verify deduplicated pair caching with mock Cross-Encoder and full Stage 5 execution."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)
    doc_ids = sorted(dataset.corpus.keys())
    test_qids = sorted(dataset.qrels_test.keys())

    run_a = {q: [(doc_ids[0], 2.0), (doc_ids[1], 1.0), (doc_ids[2], 0.5)] for q in test_qids}
    run_b = {q: [(doc_ids[1], 0.9), (doc_ids[2], 0.8), (doc_ids[3], 0.7)] for q in test_qids}

    union_pairs = collect_deduplicated_candidate_pairs([run_a, run_b], max_k=3)
    for q in test_qids:
        # Union of {0,1,2} and {1,2,3} has 4 unique doc_ids instead of 6!
        assert len(union_pairs[q]) == 4

    mock_ce = MagicMock()
    mock_ce.predict.side_effect = lambda pairs, **kwargs: np.linspace(3.0, -1.0, len(pairs))

    rerank_cfg = RerankerConfig(candidate_ks=[2, 3])
    eval_cfg = EvaluationConfig(
        primary_k=5,
        recall_ks=[5, 20, 100],
        candidate_pool_k=5,
        permutation_n_resamples=200,
        bootstrap_n_resamples=100,
    )
    cfg = synthetic_benchmark_config.model_copy(
        update={"reranker": rerank_cfg, "evaluation": eval_cfg}
    )

    bundle, _ = build_and_save_reranker_cache(
        dataset=dataset,
        union_pairs_by_qid=union_pairs,
        config=cfg,
        cross_encoder_model=mock_ce,
    )
    assert isinstance(bundle, RerankerCacheBundle)

    # Verify loading from disk works without network/model
    loaded = load_reranker_cache("synthetic", cfg)
    assert len(loaded.scores) == len(test_qids)

    stage1_meta = {
        "sparse": {"t_sparse_p50_ms": 0.1, "t_sparse_p95_ms": 0.2, "postings_per_query": 50.0},
        "hybrid": {
            "t_sparse_p50_ms": 0.1, "t_sparse_p95_ms": 0.2,
            "t_dense_encode_p50_ms": 2.0, "t_dense_encode_p95_ms": 2.5,
            "t_dense_scan_p50_ms": 0.1, "t_dense_scan_p95_ms": 0.2,
            "t_fusion_p50_ms": 0.05, "t_fusion_p95_ms": 0.1,
            "matrix_flops_per_query": 160, "vector_ram_mb": 0.01, "postings_per_query": 50.0,
        },
    }
    rep = run_reranker_benchmark_for_dataset(
        dataset=dataset,
        cache_bundle=loaded,
        pipeline_runs_test={"sparse": run_a, "hybrid": run_b},
        pipeline_labels={"sparse": "Best Sparse", "hybrid": "Calibrated Hybrid"},
        pipeline_stage1_metrics=stage1_meta,
        reranker_cfg=rerank_cfg,
        eval_cfg=eval_cfg,
        relevance_threshold=1.0,
        seed=42,
    )
    assert isinstance(rep, RerankerDatasetReport)
    assert "sparse" in rep.ceiling_decompositions
    assert "hybrid" in rep.ceiling_decompositions
    assert len(rep.system_tradeoffs) == 2 * (1 + len(rerank_cfg.candidate_ks))

    paths = export_reranker_artifacts({"synthetic": rep}, tmp_path / "metrics")
    assert paths["synthetic"].exists()
    assert paths["summary"].exists()
