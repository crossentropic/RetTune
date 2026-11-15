"""Unit and integration tests for Stage 3 Dense Spectral Geometry & Matryoshka Slicing."""

import json
from pathlib import Path
import numpy as np
import pytest

from rettune.config import BenchmarkConfig, DenseConfig, EvaluationConfig
from rettune.data_loader import load_dataset
from rettune.dense import (
    DenseDatasetReport,
    DenseEmbeddingBundle,
    export_dense_artifacts,
    fit_spectral_transforms_and_diagnostics,
    get_dense_candidates_and_matrices,
    l2_normalize_rows,
    load_embedding_bundle,
    run_dense_benchmark_for_dataset,
    slice_and_normalize_mrl,
)
from scripts.embed_corpus import (
    apply_pre_slice_layer_norm,
    atomic_write_json,
    atomic_write_npy,
    strip_trailing_normalize_modules,
)


def test_pre_slice_layernorm_rank_d_minus_1_hyperplane_property():
    """Verify LayerNorm enforces row sum = 0 and makes the D-th singular value of D_c zero."""
    rng = np.random.default_rng(42)
    # N = 40 docs, D = 16 dimensions (N > D)
    raw_corpus = rng.normal(loc=3.0, scale=2.5, size=(40, 16)).astype(np.float32)
    raw_q_dev = rng.normal(loc=-1.5, scale=1.2, size=(12, 16)).astype(np.float32)
    raw_q_test = rng.normal(loc=-1.4, scale=1.2, size=(12, 16)).astype(np.float32)

    ln_corpus = apply_pre_slice_layer_norm(raw_corpus)
    ln_q_dev = apply_pre_slice_layer_norm(raw_q_dev)
    ln_q_test = apply_pre_slice_layer_norm(raw_q_test)

    # 1. Each row of post-LayerNorm embeddings has zero mean (orthogonal to 1_D) and unit variance
    row_means = np.mean(ln_corpus, axis=1)
    row_vars = np.var(ln_corpus, axis=1, ddof=0)
    assert np.allclose(row_means, 0.0, atol=1e-6)
    assert np.allclose(row_vars, 1.0, atol=1e-4)

    # 2. Spectral diagnostics confirm D-th singular value is ~0 and active_rank == D - 1 = 15
    diag, regimes = fit_spectral_transforms_and_diagnostics(
        ln_corpus, ln_q_dev, ln_q_test, random_pairs_sample_size=200, seed=42
    )
    assert diag.full_dim == 16
    assert diag.active_rank == 15
    assert diag.null_space_singular_value < 1e-6
    assert len(diag.singular_values_active) == 15
    assert diag.effective_rank > 1.0
    assert diag.condition_number_active >= 1.0

    # 3. Verify all 4 regimes (+ control) produce C-contiguous float32 unit vectors
    for r_name, (c_vec, q_dev_vec, q_test_vec) in regimes.items():
        for arr in (c_vec, q_dev_vec, q_test_vec):
            assert arr.dtype == np.float32
            assert arr.flags.c_contiguous
            norms = np.linalg.norm(arr, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-5), f"Non-unit vectors in {r_name}"


def test_split_mean_centering_removes_prefix_cone_offset():
    """Prove that split-mean centering recovers alignment under a synthetic query-doc cone shift."""
    rng = np.random.default_rng(7)
    n_items = 20
    dim = 16

    # True underlying semantic vectors shared by query i and document i
    base_semantic = rng.normal(0.0, 1.0, size=(n_items, dim))
    base_semantic = l2_normalize_rows(base_semantic, dtype=np.float64)

    # Inject large opposing prefix shifts on documents (+shift_d) vs queries (+shift_q)
    shift_d = np.zeros((1, dim), dtype=np.float64)
    shift_d[0, 0] = 3.0
    shift_q = np.zeros((1, dim), dtype=np.float64)
    shift_q[0, 1] = 3.0

    raw_docs = apply_pre_slice_layer_norm(base_semantic + shift_d)
    raw_q_dev = apply_pre_slice_layer_norm(base_semantic + shift_q + rng.normal(0, 0.02, size=(n_items, dim)))
    raw_q_test = apply_pre_slice_layer_norm(base_semantic + shift_q)

    diag, regimes = fit_spectral_transforms_and_diagnostics(raw_docs, raw_q_dev, raw_q_test)
    assert diag.centroid_l2_distance > 0.5

    # Under split_mean_centering, diagonal self-similarity (q_i . d_i) is higher than doc_only_centering
    c_split, _, q_test_split = regimes["split_mean_centering"]
    c_doc, _, q_test_doc = regimes["doc_only_centering"]

    diag_sim_split = float(np.mean(np.sum(c_split * q_test_split, axis=1)))
    diag_sim_doc = float(np.mean(np.sum(c_doc * q_test_doc, axis=1)))
    assert diag_sim_split > diag_sim_doc


def test_contiguous_mrl_slicing_and_validation():
    """Verify MRL prefix slicing produces C-contiguous float32 unit vectors."""
    rng = np.random.default_rng(99)
    raw = apply_pre_slice_layer_norm(rng.normal(size=(10, 64)))

    for d in (8, 16, 32, 64):
        sliced = slice_and_normalize_mrl(raw, dim=d)
        assert sliced.shape == (10, d)
        assert sliced.dtype == np.float32
        assert sliced.flags.c_contiguous is True
        assert np.allclose(np.linalg.norm(sliced, axis=1), 1.0, atol=1e-5)

    with pytest.raises(ValueError, match="Invalid Matryoshka slice"):
        slice_and_normalize_mrl(raw, dim=0)
    with pytest.raises(ValueError, match="Invalid Matryoshka slice"):
        slice_and_normalize_mrl(raw, dim=128)


def test_strip_trailing_normalize_modules():
    """Verify trailing Normalize layers are popped from SentenceTransformer pipeline."""
    class DummyTransformer:
        pass

    class DummyPooling:
        pass

    class DummyNormalize:
        pass

    pipeline = [DummyTransformer(), DummyPooling(), DummyNormalize()]
    removed = strip_trailing_normalize_modules(pipeline)
    assert removed == 1
    assert len(pipeline) == 2
    assert isinstance(pipeline[-1], DummyPooling)


def test_load_and_run_dense_benchmark_end_to_end(
    synthetic_benchmark_config: BenchmarkConfig,
    tmp_path: Path,
):
    """Verify embedding bundle loading, Stage 3A/3B execution, and artifact export on synthetic dataset."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)
    doc_ids = sorted(dataset.corpus.keys())
    dev_qids = sorted(dataset.qrels_dev.keys())
    test_qids = sorted(dataset.qrels_test.keys())

    full_dim = 16
    rng = np.random.default_rng(42)
    c_emb = apply_pre_slice_layer_norm(rng.normal(size=(len(doc_ids), full_dim)))
    d_emb = apply_pre_slice_layer_norm(rng.normal(size=(len(dev_qids), full_dim)))
    t_emb = apply_pre_slice_layer_norm(rng.normal(size=(len(test_qids), full_dim)))

    dense_cfg = DenseConfig(
        model_name="test/nomic-synthetic",
        model_slug="nomic-embed-text-v1.5",
        full_dim=full_dim,
        mrl_dims=[4, 8, 16],
        random_pairs_sample_size=100,
        top_k=5,
    )
    eval_cfg = EvaluationConfig(
        primary_k=5,
        recall_ks=[5, 20, 100],
        candidate_pool_k=5,
        permutation_n_resamples=200,
        bootstrap_n_resamples=100,
    )
    cfg = synthetic_benchmark_config.model_copy(
        update={"dense": dense_cfg, "evaluation": eval_cfg}
    )

    emb_dir = cfg.get_embeddings_dir("synthetic")
    atomic_write_npy(emb_dir / "corpus.npy", c_emb)
    atomic_write_npy(emb_dir / "queries_dev.npy", d_emb)
    atomic_write_npy(emb_dir / "queries_test.npy", t_emb)
    atomic_write_json(
        emb_dir / "metadata.json",
        {
            "dataset_name": "synthetic",
            "model_name": dense_cfg.model_name,
            "full_dim": full_dim,
            "doc_ids": doc_ids,
            "dev_query_ids": dev_qids,
            "test_query_ids": test_qids,
            "encode_latency_ms": {"mean_ms": 1.2, "p50_ms": 1.1, "p95_ms": 1.5, "p99_ms": 1.8},
        },
    )

    bundle = load_embedding_bundle("synthetic", cfg)
    assert isinstance(bundle, DenseEmbeddingBundle)
    assert bundle.full_dim == full_dim

    report = run_dense_benchmark_for_dataset(
        dataset=dataset,
        bundle=bundle,
        dense_cfg=dense_cfg,
        eval_cfg=eval_cfg,
        relevance_threshold=1.0,
        seed=42,
    )
    assert isinstance(report, DenseDatasetReport)
    assert set(report.regimes.keys()) == {
        "raw_cosine",
        "split_mean_centering",
        "doc_only_centering",
        "naive_svd_whitening",
        "split_half_whitening",
    }
    assert set(report.mrl_slices.keys()) == {4, 8, 16}
    assert report.mrl_slices[16].ndcg10_retention_pct == pytest.approx(100.0)

    # Test helper for Stage 4/5 candidate & matrix extraction
    dev_run, test_run, dev_mat, test_mat = get_dense_candidates_and_matrices(
        bundle, regime_name="raw_cosine", mrl_dim=8, top_k=5
    )
    assert len(dev_run) == len(dev_qids)
    assert dev_mat.shape == (len(dev_qids), len(doc_ids))
    assert test_mat.shape == (len(test_qids), len(doc_ids))

    paths = export_dense_artifacts({"synthetic": report}, tmp_path / "metrics")
    assert paths["synthetic"].exists()
    assert paths["summary"].exists()
