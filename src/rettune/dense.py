"""Stage 3: Embedding Spectral Geometry, Subspace Alignment & Matryoshka Slicing.

Implements two core empirical investigations on `nomic-ai/nomic-embed-text-v1.5`
post-`LayerNorm`, pre-L2-normalization embeddings:

1. Experiment 3A — Contrastive Subspace Shift & Closed-Form Spectral Transforms (768-d):
   - Quantifies the geometric displacement (`||mu_Q - mu_D||_2` and `cos(mu_Q, mu_D)`)
     induced by asymmetric contrastive prefixes (`search_query:` vs. `search_document:`).
   - Proves and verifies the `LayerNorm` Rank-(D-1) Hyperplane Property: because
     parameterless `LayerNorm` enforces `x^T 1_D = 0`, all L2-normalized vectors and
     their centered matrix `D_c = D_hat - mu_D` lie strictly in the 767-dimensional
     subspace orthogonal to `1_768`, making `sigma_768 = 0`.
   - Benchmarks 4 closed-form linear algebra regimes (plus `doc_only_centering` control)
     fitted strictly on `(corpus, queries_dev)` and evaluated zero-shot on `queries_test`:
       * Regime 1 (`raw_cosine`): Standard L2-normalized dot product.
       * Regime 2 (`split_mean_centering`): Separate centering by `mu_D` and `mu_{Q,dev}`
         followed by L2 re-normalization.
       * Regime 3 (`naive_svd_whitening`): Full `Sigma_{767}^{-1}` whitening on `D_c`
         (demonstrating noise amplification on well-conditioned contrastive spaces).
       * Regime 4 (`split_half_whitening`): Split-centered damped `Sigma_{767}^{-0.5}`
         spectral scaling.

2. Experiment 3B — Matryoshka (MRL) Dimensionality Slicing & Compute Tradeoffs:
   - Evaluates prefix coordinate slices `d in {64, 128, 256, 512, 768}` directly on the
     unrotated post-`LayerNorm` embeddings (since SVD rotation destroys MRL's learned
     coarse-to-fine coordinate ordering), L2-normalizing after slicing and materializing
     C-contiguous `float32` buffers before timing BLAS inner products.
"""

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import BenchmarkConfig, DenseConfig, EvaluationConfig
from .data_loader import IRDataset
from .evaluation import (
    EvaluationResult,
    LatencyProfile,
    SignificanceResult,
    compare_paired_systems,
    evaluate_run,
    rank_score_vector,
    summarize_latencies_ms,
)

logger = logging.getLogger("rettune.dense")


@dataclass(frozen=True, slots=True)
class DenseEmbeddingBundle:
    """In-memory bundle of post-LayerNorm, pre-L2 embeddings and metadata for a dataset."""
    dataset_name: str
    model_name: str
    full_dim: int
    doc_ids: List[str]
    dev_query_ids: List[str]
    test_query_ids: List[str]
    corpus_emb: np.ndarray        # (N, D) float32, post-LayerNorm
    queries_dev_emb: np.ndarray   # (Q_dev, D) float32, post-LayerNorm
    queries_test_emb: np.ndarray  # (Q_test, D) float32, post-LayerNorm
    encode_latency_ms: Dict[str, float]


@dataclass(frozen=True, slots=True)
class SpectralDiagnostics:
    """Geometric and spectral properties of the 768-d embedding space and 767-d active subspace."""
    full_dim: int
    active_rank: int
    doc_centroid_norm: float
    query_dev_centroid_norm: float
    centroid_l2_distance: float
    centroid_cosine_similarity: float
    layernorm_max_abs_row_sum: float
    null_space_singular_value: float
    sigma_max: float
    sigma_min_active: float
    condition_number_active: float
    effective_rank: float
    mean_pairwise_cosine_raw: float
    mean_pairwise_cosine_centered: float
    mean_pairwise_cosine_whitened: float
    singular_values_active: List[float]
    cumulative_variance_milestones: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_dim": self.full_dim,
            "active_rank": self.active_rank,
            "doc_centroid_norm": self.doc_centroid_norm,
            "query_dev_centroid_norm": self.query_dev_centroid_norm,
            "centroid_l2_distance": self.centroid_l2_distance,
            "centroid_cosine_similarity": self.centroid_cosine_similarity,
            "layernorm_max_abs_row_sum": self.layernorm_max_abs_row_sum,
            "null_space_singular_value": self.null_space_singular_value,
            "sigma_max": self.sigma_max,
            "sigma_min_active": self.sigma_min_active,
            "condition_number_active": self.condition_number_active,
            "effective_rank": self.effective_rank,
            "mean_pairwise_cosine_raw": self.mean_pairwise_cosine_raw,
            "mean_pairwise_cosine_centered": self.mean_pairwise_cosine_centered,
            "mean_pairwise_cosine_whitened": self.mean_pairwise_cosine_whitened,
            "singular_values_active": self.singular_values_active,
            "cumulative_variance_milestones": self.cumulative_variance_milestones,
        }


@dataclass(frozen=True, slots=True)
class SpectralRegimeResult:
    """Evaluation result for one closed-form 768-d spectral transform regime."""
    regime_name: str
    display_label: str
    dev_eval: EvaluationResult
    test_eval: EvaluationResult
    test_vs_raw_sig: SignificanceResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_name": self.regime_name,
            "display_label": self.display_label,
            "dev_metrics": self.dev_eval.to_summary_dict(),
            "test_metrics": self.test_eval.to_summary_dict(),
            "test_vs_raw_significance": self.test_vs_raw_sig.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class MatryoshkaSliceResult:
    """Accuracy, memory, FLOPs, and latency profile for a Matryoshka prefix slice d."""
    dim: int
    dev_eval: EvaluationResult
    test_eval: EvaluationResult
    ndcg10_retention_pct: float
    recall100_retention_pct: float
    corpus_ram_mb: float
    flops_per_query: int
    scan_latency_test: LatencyProfile
    encode_latency_ms: Dict[str, float]
    test_vs_full_sig: SignificanceResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "dev_metrics": self.dev_eval.to_summary_dict(),
            "test_metrics": self.test_eval.to_summary_dict(),
            "ndcg10_retention_pct": self.ndcg10_retention_pct,
            "recall100_retention_pct": self.recall100_retention_pct,
            "corpus_ram_mb": self.corpus_ram_mb,
            "flops_per_query": self.flops_per_query,
            "scan_latency_test": self.scan_latency_test.to_dict(),
            "encode_latency_ms": self.encode_latency_ms,
            "test_vs_full_significance": self.test_vs_full_sig.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DenseDatasetReport:
    """Complete Stage 3 dense benchmark report for a single dataset."""
    dataset_name: str
    model_name: str
    spectral_diagnostics: SpectralDiagnostics
    regimes: Dict[str, SpectralRegimeResult]
    winning_regime_name: str
    mrl_slices: Dict[int, MatryoshkaSliceResult]
    winning_mrl_dim: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "spectral_diagnostics": self.spectral_diagnostics.to_dict(),
            "regimes": {k: v.to_dict() for k, v in self.regimes.items()},
            "winning_regime_name": self.winning_regime_name,
            "mrl_slices": {str(k): v.to_dict() for k, v in sorted(self.mrl_slices.items())},
            "winning_mrl_dim": self.winning_mrl_dim,
        }


def l2_normalize_rows(
    matrix: np.ndarray,
    dtype: np.dtype | type = np.float32,
    eps: float = 1e-12,
) -> np.ndarray:
    """L2-normalize each row of a 2D array and materialize a C-contiguous buffer."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix for l2_normalize_rows, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    normed = arr / np.maximum(norms, eps)
    return np.ascontiguousarray(normed, dtype=dtype)


def slice_and_normalize_mrl(
    post_ln_embeddings: np.ndarray,
    dim: int,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Slice first `dim` Matryoshka coordinates from post-LayerNorm embeddings and L2-normalize.

    Materializes a C-contiguous array so BLAS inner products operate without stride overhead.
    """
    if dim <= 0 or dim > post_ln_embeddings.shape[1]:
        raise ValueError(
            f"Invalid Matryoshka slice dim={dim} for embeddings of shape {post_ln_embeddings.shape}"
        )
    sliced = post_ln_embeddings[:, :dim]
    return l2_normalize_rows(sliced, dtype=dtype)


def load_embedding_bundle(
    dataset_name: str,
    config: BenchmarkConfig,
) -> DenseEmbeddingBundle:
    """Load pre-computed post-LayerNorm embeddings and metadata for a dataset."""
    emb_dir = config.get_embeddings_dir(dataset_name)
    corpus_path = emb_dir / "corpus.npy"
    dev_path = emb_dir / "queries_dev.npy"
    test_path = emb_dir / "queries_test.npy"
    meta_path = emb_dir / "metadata.json"

    for p, desc in [
        (corpus_path, "corpus embeddings"),
        (dev_path, "dev query embeddings"),
        (test_path, "test query embeddings"),
        (meta_path, "embedding metadata"),
    ]:
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {desc} for '{dataset_name}' at {p}. "
                "Run `python benchmark.py --setup` to generate offline embeddings."
            )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    corpus_emb = np.ascontiguousarray(np.load(corpus_path), dtype=np.float32)
    dev_emb = np.ascontiguousarray(np.load(dev_path), dtype=np.float32)
    test_emb = np.ascontiguousarray(np.load(test_path), dtype=np.float32)

    doc_ids = [str(x) for x in meta["doc_ids"]]
    dev_qids = [str(x) for x in meta["dev_query_ids"]]
    test_qids = [str(x) for x in meta["test_query_ids"]]
    full_dim = int(meta.get("full_dim", corpus_emb.shape[1]))

    if corpus_emb.shape != (len(doc_ids), full_dim):
        raise ValueError(
            f"Corpus embedding shape mismatch for '{dataset_name}': "
            f"expected {(len(doc_ids), full_dim)}, got {corpus_emb.shape}"
        )
    if dev_emb.shape != (len(dev_qids), full_dim):
        raise ValueError(
            f"Dev query embedding shape mismatch for '{dataset_name}': "
            f"expected {(len(dev_qids), full_dim)}, got {dev_emb.shape}"
        )
    if test_emb.shape != (len(test_qids), full_dim):
        raise ValueError(
            f"Test query embedding shape mismatch for '{dataset_name}': "
            f"expected {(len(test_qids), full_dim)}, got {test_emb.shape}"
        )

    encode_lat = meta.get(
        "encode_latency_ms",
        {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0},
    )

    return DenseEmbeddingBundle(
        dataset_name=dataset_name,
        model_name=str(meta.get("model_name", config.dense.model_name)),
        full_dim=full_dim,
        doc_ids=doc_ids,
        dev_query_ids=dev_qids,
        test_query_ids=test_qids,
        corpus_emb=corpus_emb,
        queries_dev_emb=dev_emb,
        queries_test_emb=test_emb,
        encode_latency_ms={k: float(v) for k, v in encode_lat.items()},
    )


def sample_mean_pairwise_cosine(
    unit_vectors: np.ndarray,
    sample_size: int = 5000,
    seed: int = 42,
) -> float:
    """Compute mean pairwise cosine similarity across random distinct document pairs."""
    n = int(unit_vectors.shape[0])
    if n < 2 or sample_size <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, n, size=sample_size)
    offsets = rng.integers(1, n, size=sample_size)
    idx_b = (idx_a + offsets) % n
    dots = np.sum(unit_vectors[idx_a] * unit_vectors[idx_b], axis=1)
    return float(np.mean(dots))


def fit_spectral_transforms_and_diagnostics(
    corpus_post_ln: np.ndarray,
    queries_dev_post_ln: np.ndarray,
    queries_test_post_ln: np.ndarray,
    random_pairs_sample_size: int = 5000,
    seed: int = 42,
) -> Tuple[SpectralDiagnostics, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Fit closed-form spectral transforms on (corpus, queries_dev) and transform all splits.

    Returns:
        (diagnostics, regime_matrices)
        where `regime_matrices[regime_name]` is `(corpus_vec, dev_q_vec, test_q_vec)` as
        C-contiguous `float32` unit vectors.
    """
    n_docs, full_dim = corpus_post_ln.shape

    # Verify LayerNorm zero-sum hyperplane property on raw post-LayerNorm inputs
    max_abs_row_mean = float(np.max(np.abs(np.mean(corpus_post_ln, axis=1))))

    # 1. Full-dimension L2-normalized unit vectors (Regime 1: Raw Cosine)
    d_hat = l2_normalize_rows(corpus_post_ln, dtype=np.float64)
    q_dev_hat = l2_normalize_rows(queries_dev_post_ln, dtype=np.float64)
    q_test_hat = l2_normalize_rows(queries_test_post_ln, dtype=np.float64)

    # 2. Centroids & Contrastive Prefix Shift
    mu_d = np.mean(d_hat, axis=0, keepdims=True)        # (1, D)
    mu_q_dev = np.mean(q_dev_hat, axis=0, keepdims=True) # (1, D)

    mu_d_norm = float(np.linalg.norm(mu_d))
    mu_q_norm = float(np.linalg.norm(mu_q_dev))
    centroid_l2_dist = float(np.linalg.norm(mu_q_dev - mu_d))
    centroid_cos = float(
        np.sum(mu_q_dev * mu_d) / max(1e-12, mu_d_norm * mu_q_norm)
    )

    # 3. Centered matrices & SVD of D_c in float64
    d_c = d_hat - mu_d
    q_dev_c_split = q_dev_hat - mu_q_dev
    q_test_c_split = q_test_hat - mu_q_dev

    q_dev_c_doc = q_dev_hat - mu_d
    q_test_c_doc = q_test_hat - mu_d

    # SVD: D_c = U Sigma V^T
    _, s_vals, vt = np.linalg.svd(d_c, full_matrices=False)

    # Because LayerNorm enforces sum_j x_j = 0, the 1_D direction is in the null space of D_c.
    # For N >= D, the D-th singular value is mathematically 0.
    max_theoretical_active = min(max(1, n_docs - 1), max(1, full_dim - 1))
    non_zero_count = int(np.sum(s_vals > 1e-6))
    active_rank = min(max_theoretical_active, max(1, non_zero_count))

    null_sigma = float(s_vals[full_dim - 1]) if s_vals.size >= full_dim else float(s_vals[-1])
    s_active = s_vals[:active_rank]
    v_active = vt[:active_rank, :].T  # (D, active_rank)

    sigma_max = float(s_active[0])
    sigma_min_active = float(s_active[-1])
    cond_active = float(sigma_max / max(1e-12, sigma_min_active))

    # Effective Rank (Roy & Vetterli, 2007) over active non-null spectrum
    energy = s_active ** 2
    total_energy = float(np.sum(energy))
    if total_energy > 0.0:
        p = energy / total_energy
        p_pos = p[p > 0.0]
        entropy = -float(np.sum(p_pos * np.log(p_pos)))
        erank = float(math.exp(entropy))
        cum_var = np.cumsum(p)
    else:
        erank = 1.0
        cum_var = np.ones_like(s_active)

    milestones: Dict[str, float] = {}
    for m in (10, 64, 128, 256, 512, active_rank):
        if m <= cum_var.size:
            milestones[f"top_{m}"] = float(cum_var[m - 1])

    # Regime 2: Split-Mean Centering
    d_split_centered = l2_normalize_rows(d_c, dtype=np.float32)
    q_dev_split_centered = l2_normalize_rows(q_dev_c_split, dtype=np.float32)
    q_test_split_centered = l2_normalize_rows(q_test_c_split, dtype=np.float32)

    # Control Variant: Document-Only Mean Centering (subtracting mu_D from both D and Q)
    d_doc_centered = d_split_centered
    q_dev_doc_centered = l2_normalize_rows(q_dev_c_doc, dtype=np.float32)
    q_test_doc_centered = l2_normalize_rows(q_test_c_doc, dtype=np.float32)

    # Regime 3: Naive Corpus-Only SVD Whitening (Sigma_active^{-1})
    inv_s_full = 1.0 / np.maximum(s_active, 1e-12)
    w_full = v_active * inv_s_full[np.newaxis, :]  # (D, active_rank)
    d_naive_white = l2_normalize_rows(d_c @ w_full, dtype=np.float32)
    q_dev_naive_white = l2_normalize_rows(q_dev_c_doc @ w_full, dtype=np.float32)
    q_test_naive_white = l2_normalize_rows(q_test_c_doc @ w_full, dtype=np.float32)

    # Regime 4: Split-Centered Half-Power Spectral Whitening (Sigma_active^{-0.5})
    inv_s_half = 1.0 / np.sqrt(np.maximum(s_active, 1e-12))
    w_half = v_active * inv_s_half[np.newaxis, :]  # (D, active_rank)
    d_half_white = l2_normalize_rows(d_c @ w_half, dtype=np.float32)
    q_dev_half_white = l2_normalize_rows(q_dev_c_split @ w_half, dtype=np.float32)
    q_test_half_white = l2_normalize_rows(q_test_c_split @ w_half, dtype=np.float32)

    d_raw_f32 = np.ascontiguousarray(d_hat, dtype=np.float32)
    q_dev_raw_f32 = np.ascontiguousarray(q_dev_hat, dtype=np.float32)
    q_test_raw_f32 = np.ascontiguousarray(q_test_hat, dtype=np.float32)

    # Pairwise cosine isotropy diagnostics across 5,000 random pairs
    mean_cos_raw = sample_mean_pairwise_cosine(d_raw_f32, random_pairs_sample_size, seed=seed)
    mean_cos_centered = sample_mean_pairwise_cosine(d_split_centered, random_pairs_sample_size, seed=seed)
    mean_cos_whitened = sample_mean_pairwise_cosine(d_naive_white, random_pairs_sample_size, seed=seed)

    diagnostics = SpectralDiagnostics(
        full_dim=full_dim,
        active_rank=active_rank,
        doc_centroid_norm=mu_d_norm,
        query_dev_centroid_norm=mu_q_norm,
        centroid_l2_distance=centroid_l2_dist,
        centroid_cosine_similarity=centroid_cos,
        layernorm_max_abs_row_sum=max_abs_row_mean,
        null_space_singular_value=null_sigma,
        sigma_max=sigma_max,
        sigma_min_active=sigma_min_active,
        condition_number_active=cond_active,
        effective_rank=erank,
        mean_pairwise_cosine_raw=mean_cos_raw,
        mean_pairwise_cosine_centered=mean_cos_centered,
        mean_pairwise_cosine_whitened=mean_cos_whitened,
        singular_values_active=[float(x) for x in s_active],
        cumulative_variance_milestones=milestones,
    )

    regimes = {
        "raw_cosine": (d_raw_f32, q_dev_raw_f32, q_test_raw_f32),
        "split_mean_centering": (d_split_centered, q_dev_split_centered, q_test_split_centered),
        "doc_only_centering": (d_doc_centered, q_dev_doc_centered, q_test_doc_centered),
        "naive_svd_whitening": (d_naive_white, q_dev_naive_white, q_test_naive_white),
        "split_half_whitening": (d_half_white, q_dev_half_white, q_test_half_white),
    }

    return diagnostics, regimes


def retrieve_dense_top_k(
    corpus_vecs: np.ndarray,
    query_vecs: np.ndarray,
    doc_ids: Sequence[str],
    query_ids: Sequence[str],
    top_k: int = 100,
) -> Tuple[Dict[str, List[Tuple[str, float]]], np.ndarray]:
    """Compute exact inner product matrix `Q @ D^T` and rank top-K deterministically.

    Returns:
        (run_dict, score_matrix)
    """
    score_mat = np.matmul(query_vecs, corpus_vecs.T, dtype=np.float32).astype(np.float64)
    run: Dict[str, List[Tuple[str, float]]] = {}
    for i, qid in enumerate(query_ids):
        run[qid] = rank_score_vector(score_mat[i], doc_ids, top_k=top_k)
    return run, score_mat


def profile_dense_scan_latency(
    corpus_vecs: np.ndarray,
    query_vecs: np.ndarray,
    doc_ids: Sequence[str],
    top_k: int = 100,
    warmup_count: int = 5,
) -> LatencyProfile:
    """Measure warm-cache dense scan + top-K ranking latency per query on contiguous float32 buffers."""
    n_q = int(query_vecs.shape[0])
    n_d, dim = corpus_vecs.shape
    flops_per_query = float(2 * n_d * dim)

    if n_q == 0:
        return summarize_latencies_ms([], flops_per_query, "FLOPs")

    c_contig = np.ascontiguousarray(corpus_vecs, dtype=np.float32)
    q_contig = np.ascontiguousarray(query_vecs, dtype=np.float32)

    for i in range(min(warmup_count, n_q)):
        scores = np.dot(c_contig, q_contig[i])
        _ = rank_score_vector(scores, doc_ids, top_k=top_k)

    latencies_ms: List[float] = []
    for i in range(n_q):
        q_vec = q_contig[i]
        t0 = time.perf_counter_ns()
        scores = np.dot(c_contig, q_vec)
        _ = rank_score_vector(scores, doc_ids, top_k=top_k)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    return summarize_latencies_ms(
        latencies_ms,
        work_units_per_query=flops_per_query,
        work_unit_name="FLOPs",
    )


REGIME_LABELS = {
    "raw_cosine": "Regime 1: Raw Cosine (768-d Baseline)",
    "split_mean_centering": "Regime 2: Split-Mean Centering (μ_D, μ_{Q,dev})",
    "doc_only_centering": "Control: Doc-Only Centering (μ_D only)",
    "naive_svd_whitening": "Regime 3: Naive Corpus SVD Whitening (Σ_{767}^{-1})",
    "split_half_whitening": "Regime 4: Split-Centered Half-Power Whitening (Σ_{767}^{-0.5})",
}


def run_dense_benchmark_for_dataset(
    dataset: IRDataset,
    bundle: DenseEmbeddingBundle,
    dense_cfg: DenseConfig,
    eval_cfg: EvaluationConfig,
    relevance_threshold: float = 1.0,
    seed: int = 42,
) -> DenseDatasetReport:
    """Execute Stage 3A (Spectral Regimes) and Stage 3B (Matryoshka Slicing) for one dataset."""
    logger.info(
        "Stage 3 Dense Spectral & Matryoshka Analysis on '%s' (docs=%d, dim=%d)...",
        dataset.name,
        len(bundle.doc_ids),
        bundle.full_dim,
    )

    max_eval_k = max([eval_cfg.primary_k] + list(eval_cfg.recall_ks))

    # --- Experiment 3A: 768-d Subspace Shift & Spectral Transforms ---
    diagnostics, regime_mats = fit_spectral_transforms_and_diagnostics(
        corpus_post_ln=bundle.corpus_emb,
        queries_dev_post_ln=bundle.queries_dev_emb,
        queries_test_post_ln=bundle.queries_test_emb,
        random_pairs_sample_size=dense_cfg.random_pairs_sample_size,
        seed=seed,
    )

    regime_evals_dev: Dict[str, EvaluationResult] = {}
    regime_evals_test: Dict[str, EvaluationResult] = {}

    for r_name, (c_vec, q_dev_vec, q_test_vec) in regime_mats.items():
        dev_run, _ = retrieve_dense_top_k(
            c_vec, q_dev_vec, bundle.doc_ids, bundle.dev_query_ids, top_k=max_eval_k
        )
        test_run, _ = retrieve_dense_top_k(
            c_vec, q_test_vec, bundle.doc_ids, bundle.test_query_ids, top_k=max_eval_k
        )
        regime_evals_dev[r_name] = evaluate_run(
            dev_run,
            dataset.qrels_dev,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )
        regime_evals_test[r_name] = evaluate_run(
            test_run,
            dataset.qrels_test,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )

    raw_test_ndcg = regime_evals_test["raw_cosine"].per_query_ndcg
    regime_results: Dict[str, SpectralRegimeResult] = {}
    winning_regime_name = "raw_cosine"
    winning_regime_key = (-1.0, -1.0)

    for r_name in regime_mats:
        sig = compare_paired_systems(
            regime_evals_test[r_name].per_query_ndcg,
            raw_test_ndcg,
            n_permutations=eval_cfg.permutation_n_resamples,
            n_bootstrap=eval_cfg.bootstrap_n_resamples,
            confidence_level=eval_cfg.confidence_level,
            seed=seed,
        )
        regime_results[r_name] = SpectralRegimeResult(
            regime_name=r_name,
            display_label=REGIME_LABELS.get(r_name, r_name),
            dev_eval=regime_evals_dev[r_name],
            test_eval=regime_evals_test[r_name],
            test_vs_raw_sig=sig,
        )
        dev_ev = regime_evals_dev[r_name]
        cand_key = (round(dev_ev.ndcg_at_k, 8), round(dev_ev.recall_at_k.get(100, 0.0), 8))
        if cand_key > winning_regime_key:
            winning_regime_key = cand_key
            winning_regime_name = r_name

    # --- Experiment 3B: Matryoshka (MRL) Dimensionality Slicing ---
    valid_dims = sorted(set(d for d in dense_cfg.mrl_dims if 1 <= d <= bundle.full_dim))
    if bundle.full_dim not in valid_dims:
        valid_dims.append(bundle.full_dim)

    mrl_dev_evals: Dict[int, EvaluationResult] = {}
    mrl_test_evals: Dict[int, EvaluationResult] = {}
    mrl_latencies: Dict[int, LatencyProfile] = {}

    for dim in valid_dims:
        c_slice = slice_and_normalize_mrl(bundle.corpus_emb, dim=dim, dtype=np.float32)
        q_dev_slice = slice_and_normalize_mrl(bundle.queries_dev_emb, dim=dim, dtype=np.float32)
        q_test_slice = slice_and_normalize_mrl(bundle.queries_test_emb, dim=dim, dtype=np.float32)

        dev_run, _ = retrieve_dense_top_k(
            c_slice, q_dev_slice, bundle.doc_ids, bundle.dev_query_ids, top_k=max_eval_k
        )
        test_run, _ = retrieve_dense_top_k(
            c_slice, q_test_slice, bundle.doc_ids, bundle.test_query_ids, top_k=max_eval_k
        )
        mrl_dev_evals[dim] = evaluate_run(
            dev_run,
            dataset.qrels_dev,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )
        mrl_test_evals[dim] = evaluate_run(
            test_run,
            dataset.qrels_test,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )
        mrl_latencies[dim] = profile_dense_scan_latency(
            c_slice, q_test_slice, bundle.doc_ids, top_k=dense_cfg.top_k
        )

    full_dim_eval = mrl_test_evals[bundle.full_dim]
    full_ndcg = max(1e-12, full_dim_eval.ndcg_at_k)
    full_rec100 = max(1e-12, full_dim_eval.recall_at_k.get(100, 0.0))

    mrl_results: Dict[int, MatryoshkaSliceResult] = {}
    winning_mrl_dim = bundle.full_dim
    winning_mrl_key = (-1.0, -1.0)
    n_docs = len(bundle.doc_ids)

    for dim in valid_dims:
        test_ev = mrl_test_evals[dim]
        dev_ev = mrl_dev_evals[dim]
        sig = compare_paired_systems(
            test_ev.per_query_ndcg,
            full_dim_eval.per_query_ndcg,
            n_permutations=eval_cfg.permutation_n_resamples,
            n_bootstrap=eval_cfg.bootstrap_n_resamples,
            confidence_level=eval_cfg.confidence_level,
            seed=seed,
        )
        ram_mb = float((n_docs * dim * 4) / 1_000_000.0)
        flops = int(2 * n_docs * dim)
        mrl_results[dim] = MatryoshkaSliceResult(
            dim=dim,
            dev_eval=dev_ev,
            test_eval=test_ev,
            ndcg10_retention_pct=float(100.0 * test_ev.ndcg_at_k / full_ndcg),
            recall100_retention_pct=float(
                100.0 * test_ev.recall_at_k.get(100, 0.0) / full_rec100
            ),
            corpus_ram_mb=ram_mb,
            flops_per_query=flops,
            scan_latency_test=mrl_latencies[dim],
            encode_latency_ms=bundle.encode_latency_ms,
            test_vs_full_sig=sig,
        )
        cand_key = (round(dev_ev.ndcg_at_k, 8), round(dev_ev.recall_at_k.get(100, 0.0), 8))
        if cand_key > winning_mrl_key:
            winning_mrl_key = cand_key
            winning_mrl_dim = dim

    return DenseDatasetReport(
        dataset_name=dataset.name,
        model_name=bundle.model_name,
        spectral_diagnostics=diagnostics,
        regimes=regime_results,
        winning_regime_name=winning_regime_name,
        mrl_slices=mrl_results,
        winning_mrl_dim=winning_mrl_dim,
    )


def get_dense_candidates_and_matrices(
    bundle: DenseEmbeddingBundle,
    regime_name: str = "raw_cosine",
    mrl_dim: Optional[int] = None,
    top_k: int = 100,
) -> Tuple[
    Dict[str, List[Tuple[str, float]]],
    Dict[str, List[Tuple[str, float]]],
    np.ndarray,
    np.ndarray,
]:
    """Return (dev_top_k_run, test_top_k_run, dev_score_matrix, test_score_matrix) for Stage 4/5."""
    if mrl_dim is not None and mrl_dim < bundle.full_dim:
        c_vec = slice_and_normalize_mrl(bundle.corpus_emb, dim=mrl_dim, dtype=np.float32)
        q_dev_vec = slice_and_normalize_mrl(bundle.queries_dev_emb, dim=mrl_dim, dtype=np.float32)
        q_test_vec = slice_and_normalize_mrl(bundle.queries_test_emb, dim=mrl_dim, dtype=np.float32)
    elif regime_name == "raw_cosine":
        c_vec = l2_normalize_rows(bundle.corpus_emb, dtype=np.float32)
        q_dev_vec = l2_normalize_rows(bundle.queries_dev_emb, dtype=np.float32)
        q_test_vec = l2_normalize_rows(bundle.queries_test_emb, dtype=np.float32)
    else:
        _, regimes = fit_spectral_transforms_and_diagnostics(
            bundle.corpus_emb, bundle.queries_dev_emb, bundle.queries_test_emb
        )
        c_vec, q_dev_vec, q_test_vec = regimes[regime_name]

    dev_run, dev_mat = retrieve_dense_top_k(
        c_vec, q_dev_vec, bundle.doc_ids, bundle.dev_query_ids, top_k=top_k
    )
    test_run, test_mat = retrieve_dense_top_k(
        c_vec, q_test_vec, bundle.doc_ids, bundle.test_query_ids, top_k=top_k
    )
    return dev_run, test_run, dev_mat, test_mat


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".tmp_{path.name}_", delete=False, encoding="utf-8"
        ) as tf:
            temp_path = Path(tf.name)
            tf.write(json.dumps(data, indent=2) + "\n")
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(temp_path, path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def export_dense_artifacts(
    reports: Mapping[str, DenseDatasetReport],
    metrics_dir: Path | str,
) -> Dict[str, Path]:
    """Export per-dataset and combined Stage 3 dense benchmark JSON artifacts."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    summary_payload: Dict[str, Any] = {"datasets": {}}
    for ds_name, rep in reports.items():
        ds_path = out_dir / f"stage3_dense_{ds_name}.json"
        rep_dict = rep.to_dict()
        _atomic_write_json(ds_path, rep_dict)
        exported[ds_name] = ds_path
        summary_payload["datasets"][ds_name] = rep_dict

    summary_path = out_dir / "stage3_dense_summary.json"
    _atomic_write_json(summary_path, summary_payload)
    exported["summary"] = summary_path
    return exported
