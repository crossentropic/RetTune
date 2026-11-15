"""Stage 4: Score Distribution Calibration, Missing-Candidate Imputation & Hybrid Fusion.

Investigates three coupled mechanics when fusing top-K Sparse (Stage 2) and Dense (Stage 3)
candidate pools over the candidate union U_q = S_q cup D_q:

1. Experiment 4A — The Missing-Candidate Imputation Paradox:
   Anchors query-local normalizer statistics (mu_K, sigma_K, s_min, s_max) strictly on
   each modality's retrieved top-K window and compares:
   - `naive_zero` (`.get(doc_id, 0.0)`): Demonstrates how post-normalization zero
     imputation under Z-score calibration assigns unretrieved documents the *mean*
     of the top-K hits (0.0), falsely ranking missing candidates above Ranks K/2..K
     (which have negative Z-scores).
   - `topk_floor` (`min_{c in C_q} s_tilde(c)`): ANN-realistic production default that
     imputes the cutoff boundary score s_tilde_(K), guaranteeing unretrieved candidates
     never outrank retrieved top-K hits with zero extra index lookups.
   - `exact_dual_lookup`: In-memory upper bound that looks up the true raw score s(d)
     from the full Q x N score matrix for any d in S_q symmetric_diff D_q and transforms
     it via the frozen top-K normalizer `norm_K(s(d))`.

2. Experiment 4B — In-Domain Calibration Progression (alpha-Sweep on dev -> Frozen test):
   Compares parameter-free Reciprocal Rank Fusion (RRF, k0 in {20, 60, 100}) against
   convex score combinations `alpha * s_sparse + (1 - alpha) * s_dense` across
   `alpha in {0.00, 0.05, ..., 1.00}` under `raw`, `minmax`, `zscore`, and
   `zscore_sigmoid` calibration.

3. Experiment 4C — Out-of-Domain Zero-Shot alpha* Transfer vs. RRF Resilience:
   Evaluates the 3x3 cross-domain transfer matrix (applying alpha* tuned on source_dev
   directly to target_test across NFCorpus, SciFact, and ArguAna), quantifying the
   domain-shift generalization gap Delta_NDCG@10 and contrasting against parameter-free RRF.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .calibration import (
    SUPPORTED_CALIBRATION_METHODS,
    QueryScoreNormalizer,
    calibrate_top_k_candidates,
)
from .config import BenchmarkConfig, EvaluationConfig, FusionConfig
from .data_loader import IRDataset
from .evaluation import (
    EvaluationResult,
    LatencyProfile,
    SignificanceResult,
    compare_paired_systems,
    evaluate_run,
    rank_candidates,
    summarize_latencies_ms,
)

logger = logging.getLogger("rettune.fusion")

SUPPORTED_IMPUTATION_REGIMES = ("naive_zero", "topk_floor", "exact_dual_lookup")


@dataclass(frozen=True, slots=True)
class PrecalibratedQueryCandidates:
    """Precomputed calibrated & imputed modality scores over U_q = S_q cup D_q for fast alpha sweeps."""
    query_id: str
    union_doc_ids: List[str]
    sparse_scores: np.ndarray
    dense_scores: np.ndarray
    sparse_rank_inversions: int
    dense_rank_inversions: int


@dataclass(frozen=True, slots=True)
class ImputationExperimentEntry:
    """Metrics for one (calibration_method, imputation_regime) pair in Experiment 4A."""
    calibration_method: str
    imputation_regime: str
    alpha_fixed_05_dev_ndcg10: float
    alpha_fixed_05_test_ndcg10: float
    optimal_alpha_dev: float
    tuned_dev_eval: EvaluationResult
    tuned_test_eval: EvaluationResult
    mean_rank_inversions_per_query: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_method": self.calibration_method,
            "imputation_regime": self.imputation_regime,
            "alpha_05_dev_ndcg@10": self.alpha_fixed_05_dev_ndcg10,
            "alpha_05_test_ndcg@10": self.alpha_fixed_05_test_ndcg10,
            "optimal_alpha_dev": self.optimal_alpha_dev,
            "tuned_dev_metrics": self.tuned_dev_eval.to_summary_dict(),
            "tuned_test_metrics": self.tuned_test_eval.to_summary_dict(),
            "mean_rank_inversions_per_query": self.mean_rank_inversions_per_query,
        }


@dataclass(frozen=True, slots=True)
class CalibrationMethodReport:
    """Alpha-sweep curve on dev and frozen evaluation on test for one calibration method (Exp 4B)."""
    calibration_method: str
    imputation_regime: str
    alpha_grid: List[float]
    dev_ndcg10_curve: List[float]
    dev_recall100_curve: List[float]
    test_ndcg10_curve: List[float]
    optimal_alpha_dev: float
    dev_eval_at_opt: EvaluationResult
    test_eval_at_opt: EvaluationResult
    test_vs_rrf_sig: SignificanceResult
    test_vs_raw_sig: SignificanceResult
    test_vs_best_single_sig: SignificanceResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calibration_method": self.calibration_method,
            "imputation_regime": self.imputation_regime,
            "optimal_alpha_dev": self.optimal_alpha_dev,
            "alpha_grid": self.alpha_grid,
            "dev_ndcg10_curve": self.dev_ndcg10_curve,
            "dev_recall100_curve": self.dev_recall100_curve,
            "test_ndcg10_curve": self.test_ndcg10_curve,
            "dev_metrics_at_opt": self.dev_eval_at_opt.to_summary_dict(),
            "test_metrics_at_opt": self.test_eval_at_opt.to_summary_dict(),
            "significance_vs_rrf": self.test_vs_rrf_sig.to_dict(),
            "significance_vs_raw": self.test_vs_raw_sig.to_dict(),
            "significance_vs_best_single": self.test_vs_best_single_sig.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TransferCellResult:
    """Single cell (source_dataset -> target_dataset) in Experiment 4C OOD alpha* transfer matrix."""
    source_dataset: str
    target_dataset: str
    calibration_method: str
    source_alpha_star: float
    target_in_domain_alpha_star: float
    target_test_ndcg10_transfer: float
    target_test_ndcg10_in_domain: float
    target_test_ndcg10_rrf: float
    transfer_gap_vs_in_domain: float
    transfer_delta_vs_rrf: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "calibration_method": self.calibration_method,
            "source_alpha_star": self.source_alpha_star,
            "target_in_domain_alpha_star": self.target_in_domain_alpha_star,
            "target_test_ndcg@10_transfer": self.target_test_ndcg10_transfer,
            "target_test_ndcg@10_in_domain": self.target_test_ndcg10_in_domain,
            "target_test_ndcg@10_rrf": self.target_test_ndcg10_rrf,
            "transfer_gap_vs_in_domain": self.transfer_gap_vs_in_domain,
            "transfer_delta_vs_rrf": self.transfer_delta_vs_rrf,
        }


@dataclass(frozen=True, slots=True)
class FusionDatasetReport:
    """Complete Stage 4 hybrid fusion report for a single dataset."""
    dataset_name: str
    pure_sparse_dev_eval: EvaluationResult
    pure_sparse_test_eval: EvaluationResult
    pure_dense_dev_eval: EvaluationResult
    pure_dense_test_eval: EvaluationResult
    rrf_dev_eval_by_k0: Dict[int, EvaluationResult]
    rrf_test_eval_by_k0: Dict[int, EvaluationResult]
    default_rrf_k0: int
    imputation_experiment_4a: Dict[str, Dict[str, ImputationExperimentEntry]]
    calibration_progression_4b: Dict[str, CalibrationMethodReport]
    winning_calibration_method: str
    winning_alpha_star: float
    fusion_latency_test: LatencyProfile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "single_modality_baselines": {
                "pure_sparse_dev": self.pure_sparse_dev_eval.to_summary_dict(),
                "pure_sparse_test": self.pure_sparse_test_eval.to_summary_dict(),
                "pure_dense_dev": self.pure_dense_dev_eval.to_summary_dict(),
                "pure_dense_test": self.pure_dense_test_eval.to_summary_dict(),
            },
            "rrf_baselines": {
                "default_k0": self.default_rrf_k0,
                "dev_by_k0": {str(k0): ev.to_summary_dict() for k0, ev in sorted(self.rrf_dev_eval_by_k0.items())},
                "test_by_k0": {str(k0): ev.to_summary_dict() for k0, ev in sorted(self.rrf_test_eval_by_k0.items())},
            },
            "experiment_4a_imputation": {
                cal: {imp: entry.to_dict() for imp, entry in imp_map.items()}
                for cal, imp_map in self.imputation_experiment_4a.items()
            },
            "experiment_4b_calibration": {
                cal: rep.to_dict() for cal, rep in self.calibration_progression_4b.items()
            },
            "winning_calibration_method": self.winning_calibration_method,
            "winning_alpha_star": self.winning_alpha_star,
            "fusion_latency_test": self.fusion_latency_test.to_dict(),
        }


def fuse_query_rrf(
    sparse_candidates: Sequence[Tuple[str, float]] | Mapping[str, float],
    dense_candidates: Sequence[Tuple[str, float]] | Mapping[str, float],
    k0: int = 60,
    top_k: int = 100,
    output_top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Fuse sparse and dense top-K lists for a single query via Reciprocal Rank Fusion.

    Only sparse candidates with strictly positive raw BM25 scores (`score > 0.0`) receive
    sparse rank credit `1 / (k0 + r_sparse)`.
    """
    if k0 <= 0:
        raise ValueError(f"RRF constant k0 must be positive, got {k0}")

    s_ranked = rank_candidates(sparse_candidates, top_k=top_k)
    s_active = [(did, sc) for did, sc in s_ranked if sc > 0.0]
    d_ranked = rank_candidates(dense_candidates, top_k=top_k)

    rrf_scores: Dict[str, float] = {}
    for rank_idx, (did, _) in enumerate(s_active, start=1):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k0 + rank_idx)

    for rank_idx, (did, _) in enumerate(d_ranked, start=1):
        rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k0 + rank_idx)

    return rank_candidates(rrf_scores, top_k=output_top_k)


def fuse_run_rrf(
    sparse_run: Mapping[str, Sequence[Tuple[str, float]] | Mapping[str, float]],
    dense_run: Mapping[str, Sequence[Tuple[str, float]] | Mapping[str, float]],
    query_ids: Sequence[str],
    k0: int = 60,
    top_k: int = 100,
    output_top_k: Optional[int] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Apply RRF across all queries in `query_ids`."""
    fused: Dict[str, List[Tuple[str, float]]] = {}
    for qid in query_ids:
        s_cands = sparse_run.get(qid, [])
        d_cands = dense_run.get(qid, [])
        fused[qid] = fuse_query_rrf(
            s_cands, d_cands, k0=k0, top_k=top_k, output_top_k=output_top_k
        )
    return fused


def precalibrate_query_union(
    query_id: str,
    sparse_candidates: Sequence[Tuple[str, float]] | Mapping[str, float],
    dense_candidates: Sequence[Tuple[str, float]] | Mapping[str, float],
    calibration_method: str = "zscore_sigmoid",
    imputation_regime: str = "topk_floor",
    top_k: int = 100,
    epsilon: float = 1e-8,
    sigmoid_clip: float = 500.0,
    doc_to_idx: Optional[Mapping[str, int]] = None,
    sparse_full_row: Optional[np.ndarray] = None,
    dense_full_row: Optional[np.ndarray] = None,
) -> PrecalibratedQueryCandidates:
    """Calibrate and impute sparse and dense scores over the candidate union U_q = S_q cup D_q.

    Normalizer parameters (mu_K, sigma_K, s_min, s_max) are anchored strictly on each
    modality's retrieved top-K window so that the normalizer is 100% invariant across
    all three imputation regimes (`naive_zero`, `topk_floor`, `exact_dual_lookup`).
    """
    if calibration_method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(f"Unsupported calibration method: {calibration_method}")
    if imputation_regime not in SUPPORTED_IMPUTATION_REGIMES:
        raise ValueError(f"Unsupported imputation regime: {imputation_regime}")

    s_map, s_norm = calibrate_top_k_candidates(
        sparse_candidates,
        method=calibration_method,
        window_k=top_k,
        is_sparse=True,
        epsilon=epsilon,
        sigmoid_clip=sigmoid_clip,
    )
    d_map, d_norm = calibrate_top_k_candidates(
        dense_candidates,
        method=calibration_method,
        window_k=top_k,
        is_sparse=False,
        epsilon=epsilon,
        sigmoid_clip=sigmoid_clip,
    )

    union_doc_ids = sorted(set(s_map.keys()) | set(d_map.keys()))
    n_u = len(union_doc_ids)
    if n_u == 0:
        return PrecalibratedQueryCandidates(
            query_id=query_id,
            union_doc_ids=[],
            sparse_scores=np.empty(0, dtype=np.float64),
            dense_scores=np.empty(0, dtype=np.float64),
            sparse_rank_inversions=0,
            dense_rank_inversions=0,
        )

    s_arr = np.empty(n_u, dtype=np.float64)
    d_arr = np.empty(n_u, dtype=np.float64)

    s_floor = s_norm.cutoff_floor_score
    d_floor = d_norm.cutoff_floor_score
    s_min_retrieved = min(s_map.values()) if s_map else s_floor
    d_min_retrieved = min(d_map.values()) if d_map else d_floor

    s_inversions = 0
    d_inversions = 0

    for i, did in enumerate(union_doc_ids):
        # 1. Sparse score lookup or imputation
        if did in s_map:
            s_val = s_map[did]
        else:
            if imputation_regime == "naive_zero":
                s_val = 0.0
            elif imputation_regime == "topk_floor":
                s_val = s_floor
            elif imputation_regime == "exact_dual_lookup":
                if doc_to_idx is not None and sparse_full_row is not None and did in doc_to_idx:
                    raw_s = float(sparse_full_row[doc_to_idx[did]])
                    s_val = s_norm.transform_scalar(raw_s)
                else:
                    s_val = s_floor
            else:
                s_val = s_floor
            if s_map and s_val > s_min_retrieved + 1e-12:
                s_inversions += 1

        # 2. Dense score lookup or imputation
        if did in d_map:
            d_val = d_map[did]
        else:
            if imputation_regime == "naive_zero":
                d_val = 0.0
            elif imputation_regime == "topk_floor":
                d_val = d_floor
            elif imputation_regime == "exact_dual_lookup":
                if doc_to_idx is not None and dense_full_row is not None and did in doc_to_idx:
                    raw_d = float(dense_full_row[doc_to_idx[did]])
                    d_val = d_norm.transform_scalar(raw_d)
                else:
                    d_val = d_floor
            else:
                d_val = d_floor
            if d_map and d_val > d_min_retrieved + 1e-12:
                d_inversions += 1

        s_arr[i] = s_val
        d_arr[i] = d_val

    return PrecalibratedQueryCandidates(
        query_id=query_id,
        union_doc_ids=union_doc_ids,
        sparse_scores=s_arr,
        dense_scores=d_arr,
        sparse_rank_inversions=s_inversions,
        dense_rank_inversions=d_inversions,
    )


def fuse_precalibrated_at_alpha(
    precalibrated: Sequence[PrecalibratedQueryCandidates],
    alpha: float,
    output_top_k: Optional[int] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """Compute `alpha * s_sparse + (1 - alpha) * s_dense` and rank deterministically."""
    w_s = float(alpha)
    w_d = 1.0 - w_s
    run: Dict[str, List[Tuple[str, float]]] = {}

    for pq in precalibrated:
        if not pq.union_doc_ids:
            run[pq.query_id] = []
            continue
        combined = w_s * pq.sparse_scores + w_d * pq.dense_scores
        combined_clean = np.where(combined == 0.0, 0.0, combined)
        # Because union_doc_ids is already sorted ascending, stable argsort on -combined_clean
        # breaks ties by ascending doc_id!
        order = np.argsort(-combined_clean, kind="stable")
        if output_top_k is not None:
            order = order[:output_top_k]
        run[pq.query_id] = [
            (pq.union_doc_ids[int(idx)], float(combined_clean[int(idx)]))
            for idx in order
        ]
    return run


def fuse_query_convex(
    sparse_candidates: Sequence[Tuple[str, float]] | Mapping[str, float],
    dense_candidates: Sequence[Tuple[str, float]] | Mapping[str, float],
    alpha: float = 0.5,
    calibration_method: str = "zscore_sigmoid",
    imputation_regime: str = "topk_floor",
    top_k: int = 100,
    output_top_k: Optional[int] = None,
    epsilon: float = 1e-8,
    sigmoid_clip: float = 500.0,
    doc_to_idx: Optional[Mapping[str, int]] = None,
    sparse_full_row: Optional[np.ndarray] = None,
    dense_full_row: Optional[np.ndarray] = None,
) -> List[Tuple[str, float]]:
    """Single-query convenience wrapper for calibrated convex hybrid fusion."""
    pq = precalibrate_query_union(
        query_id="q",
        sparse_candidates=sparse_candidates,
        dense_candidates=dense_candidates,
        calibration_method=calibration_method,
        imputation_regime=imputation_regime,
        top_k=top_k,
        epsilon=epsilon,
        sigmoid_clip=sigmoid_clip,
        doc_to_idx=doc_to_idx,
        sparse_full_row=sparse_full_row,
        dense_full_row=dense_full_row,
    )
    return fuse_precalibrated_at_alpha([pq], alpha=alpha, output_top_k=output_top_k)["q"]


def build_alpha_grid(step: float = 0.05) -> List[float]:
    """Generate deterministic alpha sweep grid in [0.0, 1.0]."""
    n_steps = int(round(1.0 / step))
    grid = [round(i * step, 4) for i in range(n_steps + 1)]
    if grid[-1] != 1.0:
        grid.append(1.0)
    return grid


def profile_fusion_latency(
    sparse_run: Mapping[str, Sequence[Tuple[str, float]]],
    dense_run: Mapping[str, Sequence[Tuple[str, float]]],
    query_ids: Sequence[str],
    alpha: float,
    calibration_method: str = "zscore_sigmoid",
    imputation_regime: str = "topk_floor",
    top_k: int = 100,
    warmup_count: int = 5,
) -> LatencyProfile:
    """Measure warm-cache query-local calibration + convex fusion latency per query (ms)."""
    if not query_ids:
        return summarize_latencies_ms([], 0.0, "candidates")

    for qid in query_ids[: min(warmup_count, len(query_ids))]:
        _ = fuse_query_convex(
            sparse_run.get(qid, []),
            dense_run.get(qid, []),
            alpha=alpha,
            calibration_method=calibration_method,
            imputation_regime=imputation_regime,
            top_k=top_k,
            output_top_k=top_k,
        )

    latencies_ms: List[float] = []
    total_union_candidates = 0
    for qid in query_ids:
        s_cands = sparse_run.get(qid, [])
        d_cands = dense_run.get(qid, [])
        t0 = time.perf_counter_ns()
        pq = precalibrate_query_union(
            query_id=qid,
            sparse_candidates=s_cands,
            dense_candidates=d_cands,
            calibration_method=calibration_method,
            imputation_regime=imputation_regime,
            top_k=top_k,
        )
        _ = fuse_precalibrated_at_alpha([pq], alpha=alpha, output_top_k=top_k)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)
        total_union_candidates += len(pq.union_doc_ids)

    mean_union = float(total_union_candidates / len(query_ids))
    return summarize_latencies_ms(
        latencies_ms,
        work_units_per_query=mean_union,
        work_unit_name="union_candidates",
    )


def run_fusion_benchmark_for_dataset(
    dataset: IRDataset,
    doc_ids: Sequence[str],
    sparse_dev_run: Mapping[str, List[Tuple[str, float]]],
    sparse_test_run: Mapping[str, List[Tuple[str, float]]],
    dense_dev_run: Mapping[str, List[Tuple[str, float]]],
    dense_test_run: Mapping[str, List[Tuple[str, float]]],
    sparse_dev_mat: Optional[np.ndarray],
    sparse_test_mat: Optional[np.ndarray],
    dense_dev_mat: Optional[np.ndarray],
    dense_test_mat: Optional[np.ndarray],
    fusion_cfg: FusionConfig,
    eval_cfg: EvaluationConfig,
    relevance_threshold: float = 1.0,
    seed: int = 42,
) -> FusionDatasetReport:
    """Execute Stage 4A (Imputation Paradox) and Stage 4B (Calibration & Alpha Sweep) for a dataset."""
    logger.info("Stage 4 Hybrid Fusion & Calibration Analysis on '%s'...", dataset.name)

    dev_qids = sorted(dataset.qrels_dev.keys())
    test_qids = sorted(dataset.qrels_test.keys())
    doc_to_idx = {did: idx for idx, did in enumerate(doc_ids)}
    alpha_grid = build_alpha_grid(fusion_cfg.alpha_step)
    top_k = fusion_cfg.top_k

    # 1. Single-modality baselines
    pure_sparse_dev = evaluate_run(
        sparse_dev_run,
        dataset.qrels_dev,
        primary_k=eval_cfg.primary_k,
        recall_ks=eval_cfg.recall_ks,
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )
    pure_sparse_test = evaluate_run(
        sparse_test_run,
        dataset.qrels_test,
        primary_k=eval_cfg.primary_k,
        recall_ks=eval_cfg.recall_ks,
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )
    pure_dense_dev = evaluate_run(
        dense_dev_run,
        dataset.qrels_dev,
        primary_k=eval_cfg.primary_k,
        recall_ks=eval_cfg.recall_ks,
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )
    pure_dense_test = evaluate_run(
        dense_test_run,
        dataset.qrels_test,
        primary_k=eval_cfg.primary_k,
        recall_ks=eval_cfg.recall_ks,
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )

    best_single_test_ndcg = (
        pure_dense_test.per_query_ndcg
        if pure_dense_dev.ndcg_at_k >= pure_sparse_dev.ndcg_at_k
        else pure_sparse_test.per_query_ndcg
    )

    # 2. RRF Baselines across k0 in rrf_k0_grid
    rrf_dev_by_k0: Dict[int, EvaluationResult] = {}
    rrf_test_by_k0: Dict[int, EvaluationResult] = {}
    for k0 in sorted(set([fusion_cfg.rrf_k0_default] + list(fusion_cfg.rrf_k0_grid))):
        r_dev = fuse_run_rrf(sparse_dev_run, dense_dev_run, dev_qids, k0=k0, top_k=top_k)
        r_test = fuse_run_rrf(sparse_test_run, dense_test_run, test_qids, k0=k0, top_k=top_k)
        rrf_dev_by_k0[k0] = evaluate_run(
            r_dev,
            dataset.qrels_dev,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )
        rrf_test_by_k0[k0] = evaluate_run(
            r_test,
            dataset.qrels_test,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )

    default_rrf_test = rrf_test_by_k0[fusion_cfg.rrf_k0_default]

    def _precalibrate_split(
        split: str,
        cal_method: str,
        imp_regime: str,
    ) -> List[PrecalibratedQueryCandidates]:
        qids = dev_qids if split == "dev" else test_qids
        s_run = sparse_dev_run if split == "dev" else sparse_test_run
        d_run = dense_dev_run if split == "dev" else dense_test_run
        s_mat = sparse_dev_mat if split == "dev" else sparse_test_mat
        d_mat = dense_dev_mat if split == "dev" else dense_test_mat

        out: List[PrecalibratedQueryCandidates] = []
        for idx_q, qid in enumerate(qids):
            out.append(
                precalibrate_query_union(
                    query_id=qid,
                    sparse_candidates=s_run.get(qid, []),
                    dense_candidates=d_run.get(qid, []),
                    calibration_method=cal_method,
                    imputation_regime=imp_regime,
                    top_k=top_k,
                    epsilon=fusion_cfg.epsilon,
                    sigmoid_clip=fusion_cfg.sigmoid_clip,
                    doc_to_idx=doc_to_idx,
                    sparse_full_row=s_mat[idx_q] if s_mat is not None else None,
                    dense_full_row=d_mat[idx_q] if d_mat is not None else None,
                )
            )
        return out

    # 3. Experiment 4A: Missing-Candidate Imputation Paradox across calibration methods & regimes
    imputation_4a: Dict[str, Dict[str, ImputationExperimentEntry]] = {}
    for cal_method in ("zscore", "zscore_sigmoid", "minmax"):
        imputation_4a[cal_method] = {}
        for imp_regime in SUPPORTED_IMPUTATION_REGIMES:
            pq_dev = _precalibrate_split("dev", cal_method, imp_regime)
            pq_test = _precalibrate_split("test", cal_method, imp_regime)

            inversions = [
                pq.sparse_rank_inversions + pq.dense_rank_inversions for pq in pq_test
            ]
            mean_inv = float(np.mean(inversions)) if inversions else 0.0

            # Evaluate at fixed alpha = 0.5
            run_05_dev = fuse_precalibrated_at_alpha(pq_dev, alpha=0.5)
            run_05_test = fuse_precalibrated_at_alpha(pq_test, alpha=0.5)
            ev_05_dev = evaluate_run(
                run_05_dev,
                dataset.qrels_dev,
                primary_k=eval_cfg.primary_k,
                recall_ks=[100],
                relevance_threshold=relevance_threshold,
                already_sorted=True,
            )
            ev_05_test = evaluate_run(
                run_05_test,
                dataset.qrels_test,
                primary_k=eval_cfg.primary_k,
                recall_ks=[100],
                relevance_threshold=relevance_threshold,
                already_sorted=True,
            )

            # Sweep alpha on dev
            best_alpha = 0.5
            best_key = (-1.0, -1.0)
            best_dev_eval: Optional[EvaluationResult] = None
            for alpha in alpha_grid:
                r_dev = fuse_precalibrated_at_alpha(pq_dev, alpha=alpha)
                ev_dev = evaluate_run(
                    r_dev,
                    dataset.qrels_dev,
                    primary_k=eval_cfg.primary_k,
                    recall_ks=eval_cfg.recall_ks,
                    relevance_threshold=relevance_threshold,
                    already_sorted=True,
                )
                key = (round(ev_dev.ndcg_at_k, 8), round(ev_dev.recall_at_k.get(100, 0.0), 8))
                if key > best_key:
                    best_key = key
                    best_alpha = alpha
                    best_dev_eval = ev_dev

            assert best_dev_eval is not None
            r_test_opt = fuse_precalibrated_at_alpha(pq_test, alpha=best_alpha)
            ev_test_opt = evaluate_run(
                r_test_opt,
                dataset.qrels_test,
                primary_k=eval_cfg.primary_k,
                recall_ks=eval_cfg.recall_ks,
                relevance_threshold=relevance_threshold,
                already_sorted=True,
            )

            imputation_4a[cal_method][imp_regime] = ImputationExperimentEntry(
                calibration_method=cal_method,
                imputation_regime=imp_regime,
                alpha_fixed_05_dev_ndcg10=ev_05_dev.ndcg_at_k,
                alpha_fixed_05_test_ndcg10=ev_05_test.ndcg_at_k,
                optimal_alpha_dev=best_alpha,
                tuned_dev_eval=best_dev_eval,
                tuned_test_eval=ev_test_opt,
                mean_rank_inversions_per_query=mean_inv,
            )

    # 4. Experiment 4B: In-Domain Calibration Progression (using production default `topk_floor`)
    calibration_4b: Dict[str, CalibrationMethodReport] = {}
    raw_opt_test_eval: Optional[EvaluationResult] = None
    temp_cal_data: Dict[str, Tuple[List[float], List[float], List[float], float, EvaluationResult, EvaluationResult]] = {}

    for cal_method in SUPPORTED_CALIBRATION_METHODS:
        pq_dev = _precalibrate_split("dev", cal_method, "topk_floor")
        pq_test = _precalibrate_split("test", cal_method, "topk_floor")

        dev_ndcg_curve: List[float] = []
        dev_rec100_curve: List[float] = []
        test_ndcg_curve: List[float] = []

        best_alpha = 0.5
        best_key = (-1.0, -1.0)
        best_dev_eval = None

        for alpha in alpha_grid:
            r_dev = fuse_precalibrated_at_alpha(pq_dev, alpha=alpha)
            ev_dev = evaluate_run(
                r_dev,
                dataset.qrels_dev,
                primary_k=eval_cfg.primary_k,
                recall_ks=eval_cfg.recall_ks,
                relevance_threshold=relevance_threshold,
                already_sorted=True,
            )
            r_test = fuse_precalibrated_at_alpha(pq_test, alpha=alpha)
            ev_test = evaluate_run(
                r_test,
                dataset.qrels_test,
                primary_k=eval_cfg.primary_k,
                recall_ks=[100],
                relevance_threshold=relevance_threshold,
                already_sorted=True,
            )
            dev_ndcg_curve.append(ev_dev.ndcg_at_k)
            dev_rec100_curve.append(ev_dev.recall_at_k.get(100, 0.0))
            test_ndcg_curve.append(ev_test.ndcg_at_k)

            key = (round(ev_dev.ndcg_at_k, 8), round(ev_dev.recall_at_k.get(100, 0.0), 8))
            if key > best_key:
                best_key = key
                best_alpha = alpha
                best_dev_eval = ev_dev

        assert best_dev_eval is not None
        r_test_opt = fuse_precalibrated_at_alpha(pq_test, alpha=best_alpha)
        ev_test_opt = evaluate_run(
            r_test_opt,
            dataset.qrels_test,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )
        if cal_method == "raw":
            raw_opt_test_eval = ev_test_opt

        temp_cal_data[cal_method] = (
            dev_ndcg_curve,
            dev_rec100_curve,
            test_ndcg_curve,
            best_alpha,
            best_dev_eval,
            ev_test_opt,
        )

    assert raw_opt_test_eval is not None
    winning_cal_method = "zscore_sigmoid"
    winning_cal_key = (-1.0, -1.0)

    for cal_method, (
        dev_ndcg_curve,
        dev_rec100_curve,
        test_ndcg_curve,
        best_alpha,
        best_dev_eval,
        ev_test_opt,
    ) in temp_cal_data.items():
        sig_vs_rrf = compare_paired_systems(
            ev_test_opt.per_query_ndcg,
            default_rrf_test.per_query_ndcg,
            n_permutations=eval_cfg.permutation_n_resamples,
            n_bootstrap=eval_cfg.bootstrap_n_resamples,
            confidence_level=eval_cfg.confidence_level,
            seed=seed,
        )
        sig_vs_raw = compare_paired_systems(
            ev_test_opt.per_query_ndcg,
            raw_opt_test_eval.per_query_ndcg,
            n_permutations=eval_cfg.permutation_n_resamples,
            n_bootstrap=eval_cfg.bootstrap_n_resamples,
            confidence_level=eval_cfg.confidence_level,
            seed=seed,
        )
        sig_vs_single = compare_paired_systems(
            ev_test_opt.per_query_ndcg,
            best_single_test_ndcg,
            n_permutations=eval_cfg.permutation_n_resamples,
            n_bootstrap=eval_cfg.bootstrap_n_resamples,
            confidence_level=eval_cfg.confidence_level,
            seed=seed,
        )
        calibration_4b[cal_method] = CalibrationMethodReport(
            calibration_method=cal_method,
            imputation_regime="topk_floor",
            alpha_grid=alpha_grid,
            dev_ndcg10_curve=dev_ndcg_curve,
            dev_recall100_curve=dev_rec100_curve,
            test_ndcg10_curve=test_ndcg_curve,
            optimal_alpha_dev=best_alpha,
            dev_eval_at_opt=best_dev_eval,
            test_eval_at_opt=ev_test_opt,
            test_vs_rrf_sig=sig_vs_rrf,
            test_vs_raw_sig=sig_vs_raw,
            test_vs_best_single_sig=sig_vs_single,
        )
        cand_key = (
            round(best_dev_eval.ndcg_at_k, 8),
            round(best_dev_eval.recall_at_k.get(100, 0.0), 8),
        )
        if cand_key > winning_cal_key:
            winning_cal_key = cand_key
            winning_cal_method = cal_method

    winning_alpha = calibration_4b[winning_cal_method].optimal_alpha_dev
    fusion_latency = profile_fusion_latency(
        sparse_test_run,
        dense_test_run,
        test_qids,
        alpha=winning_alpha,
        calibration_method=winning_cal_method,
        imputation_regime="topk_floor",
        top_k=top_k,
    )

    return FusionDatasetReport(
        dataset_name=dataset.name,
        pure_sparse_dev_eval=pure_sparse_dev,
        pure_sparse_test_eval=pure_sparse_test,
        pure_dense_dev_eval=pure_dense_dev,
        pure_dense_test_eval=pure_dense_test,
        rrf_dev_eval_by_k0=rrf_dev_by_k0,
        rrf_test_eval_by_k0=rrf_test_by_k0,
        default_rrf_k0=fusion_cfg.rrf_k0_default,
        imputation_experiment_4a=imputation_4a,
        calibration_progression_4b=calibration_4b,
        winning_calibration_method=winning_cal_method,
        winning_alpha_star=winning_alpha,
        fusion_latency_test=fusion_latency,
    )


def compute_ood_alpha_transfer_matrix(
    reports: Mapping[str, FusionDatasetReport],
    calibration_method: str = "zscore_sigmoid",
) -> List[TransferCellResult]:
    """Compute Experiment 4C Out-of-Domain Zero-Shot alpha* Transfer Matrix across datasets.

    Uses each target dataset's precomputed `test_ndcg10_curve` over `alpha_grid` to
    evaluate how `source_dataset`'s `optimal_alpha_dev` performs zero-shot on
    `target_dataset (test)` compared to in-domain `target_alpha_star` and parameter-free RRF.
    """
    dataset_names = list(reports.keys())
    matrix: List[TransferCellResult] = []

    for src_name in dataset_names:
        src_rep = reports[src_name]
        src_cal = src_rep.calibration_progression_4b[calibration_method]
        src_alpha = src_cal.optimal_alpha_dev

        for tgt_name in dataset_names:
            tgt_rep = reports[tgt_name]
            tgt_cal = tgt_rep.calibration_progression_4b[calibration_method]
            tgt_alpha = tgt_cal.optimal_alpha_dev

            # Look up target test NDCG@10 at src_alpha
            alpha_grid = tgt_cal.alpha_grid
            idx = min(
                range(len(alpha_grid)),
                key=lambda i: abs(alpha_grid[i] - src_alpha),
            )
            transfer_ndcg = float(tgt_cal.test_ndcg10_curve[idx])
            in_domain_ndcg = float(tgt_cal.test_eval_at_opt.ndcg_at_k)
            rrf_ndcg = float(tgt_rep.rrf_test_eval_by_k0[tgt_rep.default_rrf_k0].ndcg_at_k)

            matrix.append(
                TransferCellResult(
                    source_dataset=src_name,
                    target_dataset=tgt_name,
                    calibration_method=calibration_method,
                    source_alpha_star=src_alpha,
                    target_in_domain_alpha_star=tgt_alpha,
                    target_test_ndcg10_transfer=transfer_ndcg,
                    target_test_ndcg10_in_domain=in_domain_ndcg,
                    target_test_ndcg10_rrf=rrf_ndcg,
                    transfer_gap_vs_in_domain=float(in_domain_ndcg - transfer_ndcg),
                    transfer_delta_vs_rrf=float(transfer_ndcg - rrf_ndcg),
                )
            )

    return matrix


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


def export_fusion_artifacts(
    reports: Mapping[str, FusionDatasetReport],
    transfer_matrix: Sequence[TransferCellResult],
    metrics_dir: Path | str,
) -> Dict[str, Path]:
    """Export per-dataset and combined Stage 4 hybrid fusion JSON artifacts."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    transfer_zscore = compute_ood_alpha_transfer_matrix(reports, calibration_method="zscore")
    transfer_sigmoid = compute_ood_alpha_transfer_matrix(reports, calibration_method="zscore_sigmoid")

    summary_payload: Dict[str, Any] = {
        "datasets": {},
        "experiment_4c_ood_transfer_matrix": [cell.to_dict() for cell in transfer_matrix],
        "experiment_4c_by_method": {
            "zscore": [cell.to_dict() for cell in transfer_zscore],
            "zscore_sigmoid": [cell.to_dict() for cell in transfer_sigmoid],
        },
    }

    for ds_name, rep in reports.items():
        ds_path = out_dir / f"stage4_fusion_{ds_name}.json"
        rep_dict = rep.to_dict()
        _atomic_write_json(ds_path, rep_dict)
        exported[ds_name] = ds_path
        summary_payload["datasets"][ds_name] = rep_dict

    summary_path = out_dir / "stage4_fusion_summary.json"
    _atomic_write_json(summary_path, summary_payload)
    exported["summary"] = summary_path
    return exported
