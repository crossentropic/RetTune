"""Unit and integration tests for Stage 2.2 Sparse BM25Okapi vs. BM25+ Retrieval."""

import math
from pathlib import Path
import numpy as np
import pytest

from rettune.config import BenchmarkConfig, EvaluationConfig, SparseConfig
from rettune.data_loader import Document, IRDataset, load_dataset
from rettune.sparse import (
    BM25PlusIndex,
    SparseDatasetReport,
    build_tokenized_collection,
    compute_long_passage_stratification,
    compute_single_term_diagnostic,
    export_sparse_artifacts,
    run_sparse_benchmark_for_dataset,
    tokenize_with_preset,
)


def test_tokenization_presets():
    """Verify 'standard' (Snowball stemmed) vs 'technical_nostem' (un-stemmed) presets."""
    text = "Patients treated with Imatinib-400mg and MED4618 in clinical trials."

    std_toks = tokenize_with_preset(text, preset="standard")
    nostem_toks = tokenize_with_preset(text, preset="technical_nostem")

    # Stopwords 'with', 'and', 'in' removed in both
    for sw in ("with", "and", "in"):
        assert sw not in std_toks
        assert sw not in nostem_toks

    # 'technical_nostem' preserves exact suffixes ('patients', 'treated', 'clinical', 'trials')
    assert "patients" in nostem_toks
    assert "treated" in nostem_toks
    assert "med4618" in nostem_toks

    # 'standard' stems 'patients' -> 'patient', 'treated' -> 'treat', 'trials' -> 'trial'
    assert "patient" in std_toks
    assert "treat" in std_toks
    assert "trial" in std_toks

    # Empty string and stopword-only query return []
    assert tokenize_with_preset("", preset="standard") == []
    assert tokenize_with_preset("the and of in to", preset="standard") == []

    with pytest.raises(ValueError, match="Unsupported tokenization preset"):
        tokenize_with_preset("hello world", preset="invalid_preset")


def test_bm25_plus_exact_closed_form_math():
    """Verify BM25PlusIndex matches hand-computed Lv & Zhai (CIKM 2011) BM25+ and Okapi equations."""
    corpus = {
        "d1": Document(id="d1", text="alpha beta gamma delta epsilon zeta eta"),  # len 7
        "d2": Document(id="d2", text="alpha omega"),                              # len 2
        "d3": Document(id="d3", text="unrelated passage"),                        # len 2
    }
    queries = {
        "q_multi": "alpha alpha beta",
        "q_single": "alpha",
        "q_empty": "the and in",
        "q_oov": "nonexistent_term_xyz",
    }
    ds = IRDataset(
        name="toy",
        corpus=corpus,
        queries=queries,
        qrels_dev={"q_multi": {"d1": 1.0}},
        qrels_test={"q_single": {"d2": 1.0}},
    )

    coll = build_tokenized_collection(ds, preset="technical_nostem")
    assert coll.doc_ids == ["d1", "d2", "d3"]

    k1 = 1.5
    b = 0.75
    N = 3
    avgdl = (7.0 + 2.0 + 2.0) / 3.0

    # Lucene IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
    # alpha appears in d1, d2 (df=2); beta appears in d1 (df=1)
    idf_alpha = math.log(1.0 + (N - 2.0 + 0.5) / (2.0 + 0.5))
    idf_beta = math.log(1.0 + (N - 1.0 + 0.5) / (1.0 + 0.5))

    idx = BM25PlusIndex(coll, k1=k1, b=b, delta=0.0)

    for delta in [0.0, 0.5, 1.25]:
        idx.set_delta(delta)
        scores = idx.score_query_tokens(coll.query_tokens["q_multi"])

        # In q_multi = ['alpha', 'alpha', 'beta'], c('alpha', q) = 2, c('beta', q) = 1
        # d1 (len=7): tf(alpha)=1, tf(beta)=1
        tfc_d1_alpha = ((k1 + 1.0) * 1.0) / (k1 * (1.0 - b + b * 7.0 / avgdl) + 1.0) + delta
        tfc_d1_beta = ((k1 + 1.0) * 1.0) / (k1 * (1.0 - b + b * 7.0 / avgdl) + 1.0) + delta
        expected_d1 = 2.0 * idf_alpha * tfc_d1_alpha + 1.0 * idf_beta * tfc_d1_beta

        # d2 (len=2): tf(alpha)=1, tf(beta)=0 (non-occurring term gets 0, NOT +delta!)
        tfc_d2_alpha = ((k1 + 1.0) * 1.0) / (k1 * (1.0 - b + b * 2.0 / avgdl) + 1.0) + delta
        expected_d2 = 2.0 * idf_alpha * tfc_d2_alpha

        # d3 (len=2): no matching terms -> strictly 0.0
        expected_d3 = 0.0

        assert np.allclose(scores, [expected_d1, expected_d2, expected_d3], rtol=1e-7, atol=1e-9)

    # Empty and OOV queries must return all-zero vectors without raising
    assert np.all(idx.score_query_tokens(coll.query_tokens["q_empty"]) == 0.0)
    assert np.all(idx.score_query_tokens(coll.query_tokens["q_oov"]) == 0.0)


def test_bm25_plus_long_passage_rescue_and_single_term_invariance():
    """Demonstrate that delta > 0 rescues long multi-term relevant passages while keeping single-term rank-invariant."""
    # Construct a corpus where d_long contains BOTH query terms ('gene', 'mutation') plus 80 filler tokens,
    # while d_short contains ONLY 'gene' in a 2-word document.
    filler = " ".join(f"filler{i}" for i in range(80))
    corpus = {
        "d_long": Document(id="d_long", text=f"rarebuchanan syndrome {filler}"),
        "d_short": Document(id="d_short", text="rarebuchanan case"),
        "d_bg1": Document(id="d_bg1", text="unrelated clinical study one"),
        "d_bg2": Document(id="d_bg2", text="unrelated clinical study two"),
        "d_bg3": Document(id="d_bg3", text="unrelated clinical study three"),
    }
    queries = {
        "q_two_terms": "rarebuchanan syndrome",
        "q_one_term": "rarebuchanan",
    }
    qrels = {
        "q_two_terms": {"d_long": 1.0},
        "q_one_term": {"d_long": 1.0},
    }
    ds = IRDataset(
        name="rescue_demo",
        corpus=corpus,
        queries=queries,
        qrels_dev=qrels,
        qrels_test=qrels,
    )

    coll = build_tokenized_collection(ds, preset="technical_nostem")
    # Under heavy length normalization (b=0.9) and delta=0.0, d_long is heavily penalized for its length
    idx = BM25PlusIndex(coll, k1=1.5, b=0.9, delta=0.0)
    scores_okapi = idx.score_query_tokens(coll.query_tokens["q_two_terms"])
    # Under BM25+ (delta=1.0), matching the second query term 'syndrome' adds +IDF(syndrome)*delta
    idx.set_delta(1.0)
    scores_plus = idx.score_query_tokens(coll.query_tokens["q_two_terms"])

    # Verify delta > 0 increases d_long's margin over d_short on the 2-term query
    d_long_idx = coll.doc_ids.index("d_long")
    d_short_idx = coll.doc_ids.index("d_short")
    margin_okapi = scores_okapi[d_long_idx] - scores_okapi[d_short_idx]
    margin_plus = scores_plus[d_long_idx] - scores_plus[d_short_idx]
    assert margin_plus > margin_okapi
    # In fact, Okapi ranks d_short > d_long (margin < 0) while BM25+ flips d_long > d_short (margin > 0)!
    assert margin_okapi < 0.0
    assert margin_plus > 0.0

    # Verify Single-Term Rank Invariance: on q_one_term ('rarebuchanan'), rank order is 100% identical
    diag = compute_single_term_diagnostic(
        coll, qrels, k1=1.5, b=0.9, delta_plus=1.0, primary_k=5
    )
    assert diag.single_term_query_count == 1
    assert diag.multi_term_query_count == 1
    assert diag.single_term_max_abs_ndcg_diff == 0.0
    assert diag.single_term_exact_rank_match_rate == 1.0


def test_long_passage_stratification_skips_missing_corpus_docs(synthetic_benchmark_config: BenchmarkConfig):
    """Verify long-passage stratification buckets queries into tertiles and skips missing corpus docs."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)
    coll = build_tokenized_collection(dataset, preset="standard")
    idx = BM25PlusIndex(coll, k1=1.5, b=0.75, delta=0.0)

    # Create synthetic qrels with 3 valid queries of varying relevant doc lengths + 1 missing-doc query
    qrels = {
        "q_dev_1": {"doc_2": 1.0},             # short doc
        "q_dev_2": {"99": 1.0},                # medium doc
        "q_test_1": {"doc_1": 1.0},            # longer doc
        "q_test_2": {"MISSING_BEIR_101": 1.0}, # missing from corpus -> must be skipped cleanly
    }
    qids = sorted(qrels.keys())
    from rettune.evaluation import evaluate_run
    run_okapi = idx.retrieve_top_k(qids, top_k=5)
    idx.set_delta(0.75)
    run_plus = idx.retrieve_top_k(qids, top_k=5)

    ev_ok = evaluate_run(run_okapi, qrels, primary_k=5, recall_ks=[5, 100], already_sorted=True)
    ev_pl = evaluate_run(run_plus, qrels, primary_k=5, recall_ks=[5, 100], already_sorted=True)

    buckets = compute_long_passage_stratification(coll, qrels, ev_ok, ev_pl)
    total_bucketed = sum(b.query_count for b in buckets)
    assert total_bucketed == 3  # q_test_2 with missing doc was skipped


def test_run_sparse_benchmark_and_export(synthetic_benchmark_config: BenchmarkConfig, tmp_path: Path):
    """Verify end-to-end Stage 2 sweep and artifact persistence on synthetic dataset."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)
    small_sparse_cfg = SparseConfig(
        tokenization_presets=["standard", "technical_nostem"],
        k1_grid=[1.2, 1.5],
        b_grid=[0.4, 0.75],
        delta_grid=[0.0, 0.5],
        default_k1=1.5,
        default_b=0.75,
        default_delta=0.0,
        top_k=5,
    )
    small_eval_cfg = EvaluationConfig(
        primary_k=5,
        recall_ks=[5, 20, 100],
        candidate_pool_k=5,
        permutation_n_resamples=200,
        bootstrap_n_resamples=100,
    )

    report, winning_idx, winning_coll = run_sparse_benchmark_for_dataset(
        dataset=dataset,
        sparse_cfg=small_sparse_cfg,
        eval_cfg=small_eval_cfg,
        relevance_threshold=1.0,
        seed=42,
    )
    assert isinstance(report, SparseDatasetReport)
    assert report.dataset_name == "synthetic"
    assert report.sweep_records_count == 2 * 2 * 2 * 2
    assert len(report.response_surface_ndcg_dev) == 2
    assert len(report.response_surface_ndcg_dev[0]) == 2

    out_dir = tmp_path / "metrics"
    paths = export_sparse_artifacts({"synthetic": report}, out_dir)
    assert paths["synthetic"].exists()
    assert paths["summary"].exists()
