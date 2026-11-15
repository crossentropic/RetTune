"""Stage 5: Candidate Pool Ceiling, Reranker Error Decomposition & Latency/Compute Tradeoff Analysis.

Implements Cross-Encoder rescoring (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with
offline deduplicated pair caching (`data/<dataset>/reranker_cache/scores.json`),
exact additive error decomposition, and multi-stage latency/compute tradeoff profiling.

Exact Per-Query Additive Error Decomposition Identity:
    For any candidate pool C_{q, K} of size K handed from Layer 1 retrieval to a Stage-2
    Cross-Encoder reranker:
        NDCG@10_reranked
        + (Ceiling@K - NDCG@10_reranked)          [Stage-2 Reranker Scoring Error >= 0]
        + (MaxCorpusNDCG@10 - Ceiling@K)          [Stage-1 Candidate Pool Misses  >= 0]
        == MaxCorpusNDCG@10
    where both NDCG@10_reranked and Ceiling@K are normalized by the exact same full-qrels
    IDCG@10, guaranteeing Ceiling@K >= NDCG@10_reranked on every query.
"""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .config import BenchmarkConfig, EvaluationConfig, RerankerConfig
from .data_loader import IRDataset
from .evaluation import (
    EvaluationResult,
    SignificanceResult,
    compare_paired_systems,
    compute_idcg_at_k,
    compute_ndcg_at_k,
    evaluate_run,
    rank_candidates,
)

logger = logging.getLogger("rettune.reranker")


@dataclass(frozen=True, slots=True)
class RerankerCacheBundle:
    """Loaded offline Cross-Encoder pair score and latency cache for a dataset."""
    dataset_name: str
    model_name: str
    max_length: int
    scores: Dict[str, Dict[str, float]]        # qid -> {did: logit}
    token_counts: Dict[str, Dict[str, int]]    # qid -> {did: token_count}
    latency_by_k_ms: Dict[int, Dict[str, float]]  # k -> {mean_ms, p50_ms, p95_ms, p99_ms}


@dataclass(frozen=True, slots=True)
class CeilingDecompositionPoint:
    """Candidate Pool Ceiling and error decomposition for a pipeline at cutoff window K."""
    pipeline_name: str
    pipeline_label: str
    candidate_k: int
    first_stage_ndcg10: float
    reranked_ndcg10: float
    ceiling_ndcg10: float
    max_corpus_ndcg10: float
    reranker_scoring_error: float
    candidate_pool_miss_error: float
    rerank_gain_ndcg10: float
    reranked_eval: EvaluationResult
    vs_first_stage_sig: SignificanceResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_label": self.pipeline_label,
            "candidate_k": self.candidate_k,
            "first_stage_ndcg@10": self.first_stage_ndcg10,
            "reranked_ndcg@10": self.reranked_ndcg10,
            "ceiling_ndcg@10": self.ceiling_ndcg10,
            "max_corpus_ndcg@10": self.max_corpus_ndcg10,
            "reranker_scoring_error": self.reranker_scoring_error,
            "candidate_pool_miss_error": self.candidate_pool_miss_error,
            "rerank_gain_ndcg@10": self.rerank_gain_ndcg10,
            "identity_sum_check": (
                self.reranked_ndcg10
                + self.reranker_scoring_error
                + self.candidate_pool_miss_error
            ),
            "reranked_metrics": self.reranked_eval.to_summary_dict(),
            "vs_first_stage_significance": self.vs_first_stage_sig.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SystemTradeoffPoint:
    """Decomposed latency, hardware-invariant compute, and accuracy for one end-to-end system."""
    system_id: str
    display_label: str
    pipeline_family: str
    rerank_k: int
    ndcg10: float
    recall100: float
    mrr10: float
    t_sparse_p50_ms: float
    t_sparse_p95_ms: float
    t_dense_encode_p50_ms: float
    t_dense_encode_p95_ms: float
    t_dense_scan_p50_ms: float
    t_dense_scan_p95_ms: float
    t_fusion_p50_ms: float
    t_fusion_p95_ms: float
    t_rerank_p50_ms: float
    t_rerank_p95_ms: float
    total_p50_ms: float
    total_p95_ms: float
    matrix_flops_per_query: int
    vector_ram_mb: float
    postings_per_query: float
    reranker_tokens_per_query: float
    is_non_dominated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "display_label": self.display_label,
            "pipeline_family": self.pipeline_family,
            "rerank_k": self.rerank_k,
            "ndcg@10": self.ndcg10,
            "recall@100": self.recall100,
            "mrr@10": self.mrr10,
            "latency_breakdown_ms": {
                "sparse_p50": self.t_sparse_p50_ms,
                "sparse_p95": self.t_sparse_p95_ms,
                "dense_encode_p50": self.t_dense_encode_p50_ms,
                "dense_encode_p95": self.t_dense_encode_p95_ms,
                "dense_scan_p50": self.t_dense_scan_p50_ms,
                "dense_scan_p95": self.t_dense_scan_p95_ms,
                "fusion_p50": self.t_fusion_p50_ms,
                "fusion_p95": self.t_fusion_p95_ms,
                "rerank_p50": self.t_rerank_p50_ms,
                "rerank_p95": self.t_rerank_p95_ms,
                "total_p50": self.total_p50_ms,
                "total_p95": self.total_p95_ms,
            },
            "compute_accounting": {
                "matrix_flops_per_query": self.matrix_flops_per_query,
                "vector_ram_mb": self.vector_ram_mb,
                "postings_per_query": self.postings_per_query,
                "reranker_tokens_per_query": self.reranker_tokens_per_query,
            },
            "is_non_dominated": self.is_non_dominated,
        }


@dataclass(frozen=True, slots=True)
class RerankerDatasetReport:
    """Complete Stage 5 reranker & latency/compute tradeoff report for a single dataset."""
    dataset_name: str
    model_name: str
    max_corpus_ndcg10: float
    ceiling_decompositions: Dict[str, Dict[int, CeilingDecompositionPoint]]
    system_tradeoffs: List[SystemTradeoffPoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "max_corpus_ndcg@10": self.max_corpus_ndcg10,
            "ceiling_decompositions": {
                pipe: {str(k): pt.to_dict() for k, pt in sorted(k_map.items())}
                for pipe, k_map in self.ceiling_decompositions.items()
            },
            "system_tradeoffs": [s.to_dict() for s in self.system_tradeoffs],
        }


def estimate_pair_token_count(query_text: str, doc_text: str, max_length: int = 512) -> int:
    """Estimate WordPiece token count for a (query, passage) pair clamped to `max_length`."""
    q_words = len(re.findall(r"\w+", query_text))
    d_words = len(re.findall(r"\w+", doc_text))
    # WordPiece subword multiplier (~1.3 subwords/word) + 3 special tokens ([CLS], [SEP], [SEP])
    est = int(round(1.3 * (q_words + d_words))) + 3
    return min(max_length, max(3, est))


def compute_max_corpus_ndcg_at_k(
    dataset: IRDataset,
    qrels: Mapping[str, Mapping[str, float]],
    k: int = 10,
    relevance_threshold: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Compute MaxCorpusNDCG@K across all queries in `qrels`.

    For each query q, ranks all positive relevance judgments that actually exist in
    `dataset.corpus` and divides by the full-query `IDCG@K` from `qrels[q]`.
    Equals 1.0000 when all judged relevant docs exist in `dataset.corpus` (NFCorpus, SciFact),
    and < 1.0 when upstream corpus omissions exist (e.g., ArguAna BEIR #101).
    """
    ndcg_thresh = min(relevance_threshold, 1e-9) if relevance_threshold <= 1.0 else relevance_threshold
    per_query: Dict[str, float] = {}
    corpus_keys = set(dataset.corpus.keys())

    for qid in sorted(qrels.keys()):
        judgments = qrels[qid]
        idcg = compute_idcg_at_k(judgments, k=k, relevance_threshold=ndcg_thresh)
        if idcg <= 0.0:
            per_query[qid] = 0.0
            continue
        in_corpus_rel_docs = sorted(
            (did for did, s in judgments.items() if did in corpus_keys and float(s) >= ndcg_thresh),
            key=lambda did: (-float(judgments[did]), did),
        )
        per_query[qid] = compute_ndcg_at_k(
            in_corpus_rel_docs,
            judgments,
            k=k,
            relevance_threshold=ndcg_thresh,
        )

    mean_val = float(np.mean(list(per_query.values()))) if per_query else 0.0
    return mean_val, per_query


def compute_candidate_pool_ceiling_and_reranked(
    pipeline_name: str,
    pipeline_label: str,
    first_stage_run: Mapping[str, Sequence[Tuple[str, float]]],
    reranker_scores: Mapping[str, Mapping[str, float]],
    dataset: IRDataset,
    qrels: Mapping[str, Mapping[str, float]],
    candidate_k: int,
    eval_cfg: EvaluationConfig,
    relevance_threshold: float = 1.0,
    seed: int = 42,
) -> CeilingDecompositionPoint:
    """Compute Actual Reranked NDCG@10, Candidate Pool Ceiling@K, and exact error decomposition."""
    qids = sorted(qrels.keys())
    primary_k = eval_cfg.primary_k
    ndcg_thresh = min(relevance_threshold, 1e-9) if relevance_threshold <= 1.0 else relevance_threshold

    first_stage_truncated: Dict[str, List[Tuple[str, float]]] = {}
    reranked_run: Dict[str, List[Tuple[str, float]]] = {}
    ceiling_run: Dict[str, List[Tuple[str, float]]] = {}

    for qid in qids:
        cands = list(first_stage_run.get(qid, []))[:candidate_k]
        first_stage_truncated[qid] = cands

        q_ce_map = reranker_scores.get(qid, {})
        ce_pairs = [(did, float(q_ce_map.get(did, -1e9))) for did, _ in cands]
        reranked_run[qid] = rank_candidates(ce_pairs)

        q_judgments = qrels[qid]
        oracle_pairs = [
            (
                did,
                float(q_judgments.get(did, 0.0))
                if float(q_judgments.get(did, 0.0)) >= ndcg_thresh
                else 0.0,
            )
            for did, _ in cands
        ]
        ceiling_run[qid] = rank_candidates(oracle_pairs)

    ev_first = evaluate_run(
        first_stage_truncated,
        qrels,
        primary_k=primary_k,
        recall_ks=eval_cfg.recall_ks,
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )
    ev_reranked = evaluate_run(
        reranked_run,
        qrels,
        primary_k=primary_k,
        recall_ks=eval_cfg.recall_ks,
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )
    ev_ceiling = evaluate_run(
        ceiling_run,
        qrels,
        primary_k=primary_k,
        recall_ks=[primary_k],
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )
    max_corpus_ndcg, per_q_max_corpus = compute_max_corpus_ndcg_at_k(
        dataset, qrels, k=primary_k, relevance_threshold=relevance_threshold
    )

    # Verify per-query inequality: Reranked@K <= Ceiling@K <= MaxCorpusNDCG@10
    for qid in qids:
        r_val = ev_reranked.per_query_ndcg[qid]
        c_val = ev_ceiling.per_query_ndcg[qid]
        m_val = per_q_max_corpus[qid]
        if r_val > c_val + 1e-9:
            raise AssertionError(
                f"Ceiling violation on query {qid}: reranked ({r_val}) > ceiling ({c_val})"
            )
        if c_val > m_val + 1e-9:
            raise AssertionError(
                f"MaxCorpus violation on query {qid}: ceiling ({c_val}) > max_corpus ({m_val})"
            )

    sig_vs_first = compare_paired_systems(
        ev_reranked.per_query_ndcg,
        ev_first.per_query_ndcg,
        n_permutations=eval_cfg.permutation_n_resamples,
        n_bootstrap=eval_cfg.bootstrap_n_resamples,
        confidence_level=eval_cfg.confidence_level,
        seed=seed,
    )

    reranked_ndcg = ev_reranked.ndcg_at_k
    ceiling_ndcg = ev_ceiling.ndcg_at_k
    scoring_error = float(max(0.0, ceiling_ndcg - reranked_ndcg))
    pool_miss_error = float(max(0.0, max_corpus_ndcg - ceiling_ndcg))
    rerank_gain = float(reranked_ndcg - ev_first.ndcg_at_k)

    return CeilingDecompositionPoint(
        pipeline_name=pipeline_name,
        pipeline_label=pipeline_label,
        candidate_k=candidate_k,
        first_stage_ndcg10=ev_first.ndcg_at_k,
        reranked_ndcg10=reranked_ndcg,
        ceiling_ndcg10=ceiling_ndcg,
        max_corpus_ndcg10=max_corpus_ndcg,
        reranker_scoring_error=scoring_error,
        candidate_pool_miss_error=pool_miss_error,
        rerank_gain_ndcg10=rerank_gain,
        reranked_eval=ev_reranked,
        vs_first_stage_sig=sig_vs_first,
    )


def mark_non_dominated_systems(systems: Sequence[SystemTradeoffPoint]) -> List[SystemTradeoffPoint]:
    """Identify non-dominated configurations in (total_p95_ms, ndcg10) space.

    A system S is non-dominated if no other system S' has both:
        S'.total_p95_ms <= S.total_p95_ms  AND  S'.ndcg10 >= S.ndcg10
    with at least one strict inequality.
    """
    marked: List[SystemTradeoffPoint] = []
    for i, sys_i in enumerate(systems):
        dominated = False
        for j, sys_j in enumerate(systems):
            if i == j:
                continue
            no_slower = sys_j.total_p95_ms <= sys_i.total_p95_ms + 1e-9
            no_worse = sys_j.ndcg10 >= sys_i.ndcg10 - 1e-9
            strictly_better = (
                (sys_j.total_p95_ms < sys_i.total_p95_ms - 1e-6)
                or (sys_j.ndcg10 > sys_i.ndcg10 + 1e-6)
            )
            if no_slower and no_worse and strictly_better:
                dominated = True
                break
        marked.append(
            SystemTradeoffPoint(
                system_id=sys_i.system_id,
                display_label=sys_i.display_label,
                pipeline_family=sys_i.pipeline_family,
                rerank_k=sys_i.rerank_k,
                ndcg10=sys_i.ndcg10,
                recall100=sys_i.recall100,
                mrr10=sys_i.mrr10,
                t_sparse_p50_ms=sys_i.t_sparse_p50_ms,
                t_sparse_p95_ms=sys_i.t_sparse_p95_ms,
                t_dense_encode_p50_ms=sys_i.t_dense_encode_p50_ms,
                t_dense_encode_p95_ms=sys_i.t_dense_encode_p95_ms,
                t_dense_scan_p50_ms=sys_i.t_dense_scan_p50_ms,
                t_dense_scan_p95_ms=sys_i.t_dense_scan_p95_ms,
                t_fusion_p50_ms=sys_i.t_fusion_p50_ms,
                t_fusion_p95_ms=sys_i.t_fusion_p95_ms,
                t_rerank_p50_ms=sys_i.t_rerank_p50_ms,
                t_rerank_p95_ms=sys_i.t_rerank_p95_ms,
                total_p50_ms=sys_i.total_p50_ms,
                total_p95_ms=sys_i.total_p95_ms,
                matrix_flops_per_query=sys_i.matrix_flops_per_query,
                vector_ram_mb=sys_i.vector_ram_mb,
                postings_per_query=sys_i.postings_per_query,
                reranker_tokens_per_query=sys_i.reranker_tokens_per_query,
                is_non_dominated=not dominated,
            )
        )
    return marked


def collect_deduplicated_candidate_pairs(
    candidate_runs: Iterable[Mapping[str, Sequence[Tuple[str, float]]]],
    max_k: int = 100,
) -> Dict[str, Set[str]]:
    """Collect the deduplicated union of top-`max_k` candidate doc_ids per query across runs."""
    union_by_qid: Dict[str, Set[str]] = {}
    for run in candidate_runs:
        for qid, pairs in run.items():
            bucket = union_by_qid.setdefault(qid, set())
            for did, _ in pairs[:max_k]:
                bucket.add(str(did))
    return union_by_qid


def load_reranker_cache(dataset_name: str, config: BenchmarkConfig) -> RerankerCacheBundle:
    """Load pre-cached Cross-Encoder scores and latency metadata from disk."""
    cache_path = config.get_reranker_cache_dir(dataset_name) / "scores.json"
    if not cache_path.is_file() or cache_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing Cross-Encoder cache for '{dataset_name}' at {cache_path}. "
            "Run `python benchmark.py --setup` to pre-cache reranker scores."
        )
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores: Dict[str, Dict[str, float]] = {
        str(qid): {str(did): float(val) for did, val in d_map.items()}
        for qid, d_map in data.get("scores", {}).items()
    }
    token_counts: Dict[str, Dict[str, int]] = {
        str(qid): {str(did): int(val) for did, val in d_map.items()}
        for qid, d_map in data.get("token_counts", {}).items()
    }
    latency_by_k: Dict[int, Dict[str, float]] = {
        int(k): {m_key: float(m_val) for m_key, m_val in lat_map.items()}
        for k, lat_map in data.get("latency_by_k_ms", {}).items()
    }

    return RerankerCacheBundle(
        dataset_name=dataset_name,
        model_name=str(data.get("model_name", config.reranker.model_name)),
        max_length=int(data.get("max_length", config.reranker.max_length)),
        scores=scores,
        token_counts=token_counts,
        latency_by_k_ms=latency_by_k,
    )


def build_and_save_reranker_cache(
    dataset: IRDataset,
    union_pairs_by_qid: Mapping[str, Iterable[str]],
    config: BenchmarkConfig,
    cross_encoder_model: Optional[Any] = None,
    force: bool = False,
) -> Tuple[RerankerCacheBundle, Any]:
    """Score deduplicated (query_id, doc_id) pairs with Cross-Encoder and save to scores.json.

    If `scores.json` already exists and `not force`, incrementally scores ONLY any new
    `(qid, did)` pairs not yet present in the cache!
    """
    cache_dir = config.get_reranker_cache_dir(dataset.name)
    cache_path = cache_dir / "scores.json"

    existing_scores: Dict[str, Dict[str, float]] = {}
    existing_tokens: Dict[str, Dict[str, int]] = {}
    existing_latency: Dict[int, Dict[str, float]] = {}

    if not force and cache_path.is_file() and cache_path.stat().st_size > 0:
        try:
            loaded = load_reranker_cache(dataset.name, config)
            existing_scores = {q: dict(m) for q, m in loaded.scores.items()}
            existing_tokens = {q: dict(m) for q, m in loaded.token_counts.items()}
            existing_latency = dict(loaded.latency_by_k_ms)
        except Exception as exc:
            logger.warning("Could not read existing reranker cache at %s: %s", cache_path, exc)

    missing_pairs: List[Tuple[str, str]] = []
    for qid in sorted(union_pairs_by_qid.keys()):
        q_cache = existing_scores.setdefault(qid, {})
        t_cache = existing_tokens.setdefault(qid, {})
        q_text = dataset.queries.get(qid, "")
        for did in sorted(set(union_pairs_by_qid[qid])):
            if did not in dataset.corpus:
                continue
            if did not in t_cache:
                t_cache[did] = estimate_pair_token_count(
                    q_text, dataset.corpus[did].get_text(), max_length=config.reranker.max_length
                )
            if did not in q_cache:
                missing_pairs.append((qid, did))

    required_ks = sorted(set(config.reranker.candidate_ks))
    need_latency = not all(k in existing_latency for k in required_ks)

    if not missing_pairs and not need_latency:
        logger.info(
            "[SKIP] Reranker cache for '%s' already covers all %d queries.",
            dataset.name,
            len(existing_scores),
        )
        return (
            RerankerCacheBundle(
                dataset_name=dataset.name,
                model_name=config.reranker.model_name,
                max_length=config.reranker.max_length,
                scores=existing_scores,
                token_counts=existing_tokens,
                latency_by_k_ms=existing_latency,
            ),
            cross_encoder_model,
        )

    import torch

    if cross_encoder_model is None:
        from sentence_transformers import CrossEncoder

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(
            "Loading Cross-Encoder '%s' on device='%s' (max_length=%d)...",
            config.reranker.model_name,
            device,
            config.reranker.max_length,
        )
        cross_encoder_model = CrossEncoder(
            config.reranker.model_name,
            max_length=config.reranker.max_length,
            device=device,
        )

    batch_sz = int(config.reranker.batch_size)
    use_fast_tok = (
        hasattr(cross_encoder_model, "tokenizer")
        and hasattr(cross_encoder_model, "model")
        and not hasattr(cross_encoder_model.model, "_mock_name")
    )
    if use_fast_tok:
        cross_encoder_model.model.eval()
        target_dev = cross_encoder_model.model.device
        act_fn = getattr(cross_encoder_model, "default_activation_function", lambda x: x)

    def _predict_slice(pairs_slice: Sequence[Tuple[str, str]]) -> np.ndarray:
        if not use_fast_tok:
            return cross_encoder_model.predict(
                list(pairs_slice),
                batch_size=batch_sz,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        chunk_preds: List[np.ndarray] = []
        n_sl = len(pairs_slice)
        with torch.inference_mode():
            for b_start in range(0, n_sl, batch_sz):
                b_end = min(b_start + batch_sz, n_sl)
                batch_slice = pairs_slice[b_start:b_end]
                tok = cross_encoder_model.tokenizer(
                    [p[0] for p in batch_slice],
                    [p[1] for p in batch_slice],
                    padding=True,
                    pad_to_multiple_of=64,
                    truncation="longest_first",
                    max_length=config.reranker.max_length,
                    return_tensors="pt",
                )
                tok = {k: v.to(target_dev) for k, v in tok.items()}
                out_logits = act_fn(cross_encoder_model.model(**tok, return_dict=True).logits)
                chunk_preds.append(out_logits.squeeze(-1).detach().float().cpu().numpy())
        return np.concatenate(chunk_preds, axis=0) if chunk_preds else np.empty(0, dtype=np.float32)

    if missing_pairs:
        # Sort deduplicated pairs by estimated token length to minimize dynamic batch padding
        missing_pairs.sort(key=lambda item: (existing_tokens[item[0]][item[1]], item[0], item[1]))
        logger.info(
            "Scoring %d deduplicated (query, passage) pairs on '%s' with Cross-Encoder (sorted by length)...",
            len(missing_pairs),
            dataset.name,
        )
        pair_texts = [
            (dataset.queries[qid].strip(), dataset.corpus[did].get_text().strip())
            for qid, did in missing_pairs
        ]
        chunk_sz = 4096
        n_pairs = len(pair_texts)
        for c_start in range(0, n_pairs, chunk_sz):
            c_end = min(c_start + chunk_sz, n_pairs)
            sub_preds = _predict_slice(pair_texts[c_start:c_end])
            for (qid, did), logit in zip(missing_pairs[c_start:c_end], sub_preds):
                existing_scores.setdefault(qid, {})[did] = float(logit)
            logger.info("  Scored %d / %d pairs on '%s'...", c_end, n_pairs, dataset.name)
            if (c_end % 16384 == 0) or (c_end == n_pairs):
                partial_payload = {
                    "dataset_name": dataset.name,
                    "model_name": config.reranker.model_name,
                    "max_length": config.reranker.max_length,
                    "num_queries": len(existing_scores),
                    "total_cached_pairs": sum(len(m) for m in existing_scores.values()),
                    "latency_by_k_ms": {str(k): v for k, v in sorted(existing_latency.items())},
                    "scores": existing_scores,
                    "token_counts": existing_tokens,
                }
                _atomic_write_json(cache_path, partial_payload)

    if need_latency:
        logger.info("Profiling Cross-Encoder batch latency across K=%s on '%s'...", required_ks, dataset.name)
        sample_qids = [q for q in sorted(existing_scores.keys()) if len(existing_scores[q]) >= max(required_ks, default=20)][:25]
        if not sample_qids:
            sample_qids = sorted(existing_scores.keys())[:25]

        for k_val in required_ks:
            latencies_ms: List[float] = []
            # Warmup
            for qid in sample_qids[:3]:
                dids = sorted(existing_scores[qid].keys())[:k_val]
                if dids:
                    batch_inputs = [(dataset.queries[qid].strip(), dataset.corpus[d].get_text().strip()) for d in dids]
                    _ = _predict_slice(batch_inputs)
            for qid in sample_qids:
                dids = sorted(existing_scores[qid].keys())[:k_val]
                if not dids:
                    continue
                batch_inputs = [(dataset.queries[qid].strip(), dataset.corpus[d].get_text().strip()) for d in dids]
                t0 = time.perf_counter_ns()
                _ = _predict_slice(batch_inputs)
                t1 = time.perf_counter_ns()
                latencies_ms.append((t1 - t0) / 1_000_000.0)

            if latencies_ms:
                arr = np.asarray(latencies_ms, dtype=np.float64)
                p50, p95, p99 = np.percentile(arr, [50, 95, 99])
                existing_latency[k_val] = {
                    "mean_ms": float(np.mean(arr)),
                    "p50_ms": float(p50),
                    "p95_ms": float(p95),
                    "p99_ms": float(p99),
                }
            else:
                existing_latency[k_val] = {
                    "mean_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                }

    payload = {
        "dataset_name": dataset.name,
        "model_name": config.reranker.model_name,
        "max_length": config.reranker.max_length,
        "num_queries": len(existing_scores),
        "total_cached_pairs": sum(len(m) for m in existing_scores.values()),
        "latency_by_k_ms": {str(k): v for k, v in sorted(existing_latency.items())},
        "scores": existing_scores,
        "token_counts": existing_tokens,
    }
    _atomic_write_json(cache_path, payload)
    logger.info(
        "Persisted Cross-Encoder cache for '%s' to %s (%d pairs).",
        dataset.name,
        cache_path,
        payload["total_cached_pairs"],
    )

    bundle = RerankerCacheBundle(
        dataset_name=dataset.name,
        model_name=config.reranker.model_name,
        max_length=config.reranker.max_length,
        scores=existing_scores,
        token_counts=existing_tokens,
        latency_by_k_ms=existing_latency,
    )
    return bundle, cross_encoder_model


def compute_mean_reranker_tokens_per_query(
    run: Mapping[str, Sequence[Tuple[str, float]]],
    cache_bundle: RerankerCacheBundle,
    dataset: IRDataset,
    candidate_k: int,
) -> float:
    """Compute average total Cross-Encoder tokens processed per query at cutoff `candidate_k`."""
    qids = sorted(dataset.qrels_test.keys())
    if not qids or candidate_k <= 0:
        return 0.0

    totals: List[int] = []
    for qid in qids:
        cands = list(run.get(qid, []))[:candidate_k]
        t_map = cache_bundle.token_counts.get(qid, {})
        q_tok_sum = 0
        for did, _ in cands:
            if did in t_map:
                q_tok_sum += int(t_map[did])
            elif did in dataset.corpus and qid in dataset.queries:
                q_tok_sum += estimate_pair_token_count(
                    dataset.queries[qid],
                    dataset.corpus[did].get_text(),
                    max_length=cache_bundle.max_length,
                )
        totals.append(q_tok_sum)
    return float(np.mean(totals))


def run_reranker_benchmark_for_dataset(
    dataset: IRDataset,
    cache_bundle: RerankerCacheBundle,
    pipeline_runs_test: Mapping[str, Mapping[str, List[Tuple[str, float]]]],
    pipeline_labels: Mapping[str, str],
    pipeline_stage1_metrics: Mapping[str, Dict[str, Any]],
    reranker_cfg: RerankerConfig,
    eval_cfg: EvaluationConfig,
    relevance_threshold: float = 1.0,
    seed: int = 42,
) -> RerankerDatasetReport:
    """Execute Stage 5 Candidate Pool Ceiling decomposition and latency/compute tradeoff analysis."""
    logger.info("Stage 5 Reranker Ceiling & Latency Tradeoff Analysis on '%s'...", dataset.name)

    max_corpus_ndcg, _ = compute_max_corpus_ndcg_at_k(
        dataset,
        dataset.qrels_test,
        k=eval_cfg.primary_k,
        relevance_threshold=relevance_threshold,
    )

    decompositions: Dict[str, Dict[int, CeilingDecompositionPoint]] = {}
    tradeoff_candidates: List[SystemTradeoffPoint] = []

    for pipe_name, test_run in pipeline_runs_test.items():
        label = pipeline_labels.get(pipe_name, pipe_name)
        s1_meta = pipeline_stage1_metrics[pipe_name]
        decompositions[pipe_name] = {}

        # 1. Add Stage-1-only (K=0 reranking) system point to tradeoff candidates
        s1_eval = evaluate_run(
            test_run,
            dataset.qrels_test,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )
        t_sp_50 = float(s1_meta.get("t_sparse_p50_ms", 0.0))
        t_sp_95 = float(s1_meta.get("t_sparse_p95_ms", 0.0))
        t_de_50 = float(s1_meta.get("t_dense_encode_p50_ms", 0.0))
        t_de_95 = float(s1_meta.get("t_dense_encode_p95_ms", 0.0))
        t_ds_50 = float(s1_meta.get("t_dense_scan_p50_ms", 0.0))
        t_ds_95 = float(s1_meta.get("t_dense_scan_p95_ms", 0.0))
        t_fu_50 = float(s1_meta.get("t_fusion_p50_ms", 0.0))
        t_fu_95 = float(s1_meta.get("t_fusion_p95_ms", 0.0))

        base_p50 = t_sp_50 + t_de_50 + t_ds_50 + t_fu_50
        base_p95 = t_sp_95 + t_de_95 + t_ds_95 + t_fu_95

        tradeoff_candidates.append(
            SystemTradeoffPoint(
                system_id=f"{pipe_name}_no_rerank",
                display_label=f"{label} (No Rerank)",
                pipeline_family=pipe_name,
                rerank_k=0,
                ndcg10=s1_eval.ndcg_at_k,
                recall100=s1_eval.recall_at_k.get(100, 0.0),
                mrr10=s1_eval.mrr_at_k,
                t_sparse_p50_ms=t_sp_50,
                t_sparse_p95_ms=t_sp_95,
                t_dense_encode_p50_ms=t_de_50,
                t_dense_encode_p95_ms=t_de_95,
                t_dense_scan_p50_ms=t_ds_50,
                t_dense_scan_p95_ms=t_ds_95,
                t_fusion_p50_ms=t_fu_50,
                t_fusion_p95_ms=t_fu_95,
                t_rerank_p50_ms=0.0,
                t_rerank_p95_ms=0.0,
                total_p50_ms=base_p50,
                total_p95_ms=base_p95,
                matrix_flops_per_query=int(s1_meta.get("matrix_flops_per_query", 0)),
                vector_ram_mb=float(s1_meta.get("vector_ram_mb", 0.0)),
                postings_per_query=float(s1_meta.get("postings_per_query", 0.0)),
                reranker_tokens_per_query=0.0,
            )
        )

        # 2. Evaluate Cross-Encoder reranking across candidate windows K in candidate_ks
        for k_val in sorted(set(reranker_cfg.candidate_ks)):
            pt = compute_candidate_pool_ceiling_and_reranked(
                pipeline_name=pipe_name,
                pipeline_label=label,
                first_stage_run=test_run,
                reranker_scores=cache_bundle.scores,
                dataset=dataset,
                qrels=dataset.qrels_test,
                candidate_k=k_val,
                eval_cfg=eval_cfg,
                relevance_threshold=relevance_threshold,
                seed=seed,
            )
            decompositions[pipe_name][k_val] = pt

            ce_lat = cache_bundle.latency_by_k_ms.get(
                k_val, {"p50_ms": float(k_val * 0.5), "p95_ms": float(k_val * 0.65)}
            )
            t_rr_50 = float(ce_lat.get("p50_ms", 0.0))
            t_rr_95 = float(ce_lat.get("p95_ms", 0.0))
            mean_tokens = compute_mean_reranker_tokens_per_query(
                test_run, cache_bundle, dataset, candidate_k=k_val
            )

            tradeoff_candidates.append(
                SystemTradeoffPoint(
                    system_id=f"{pipe_name}_rerank_k{k_val}",
                    display_label=f"{label} + CE@{k_val}",
                    pipeline_family=pipe_name,
                    rerank_k=k_val,
                    ndcg10=pt.reranked_ndcg10,
                    recall100=s1_eval.recall_at_k.get(100, 0.0),
                    mrr10=pt.reranked_eval.mrr_at_k,
                    t_sparse_p50_ms=t_sp_50,
                    t_sparse_p95_ms=t_sp_95,
                    t_dense_encode_p50_ms=t_de_50,
                    t_dense_encode_p95_ms=t_de_95,
                    t_dense_scan_p50_ms=t_ds_50,
                    t_dense_scan_p95_ms=t_ds_95,
                    t_fusion_p50_ms=t_fu_50,
                    t_fusion_p95_ms=t_fu_95,
                    t_rerank_p50_ms=t_rr_50,
                    t_rerank_p95_ms=t_rr_95,
                    total_p50_ms=base_p50 + t_rr_50,
                    total_p95_ms=base_p95 + t_rr_95,
                    matrix_flops_per_query=int(s1_meta.get("matrix_flops_per_query", 0)),
                    vector_ram_mb=float(s1_meta.get("vector_ram_mb", 0.0)),
                    postings_per_query=float(s1_meta.get("postings_per_query", 0.0)),
                    reranker_tokens_per_query=mean_tokens,
                )
            )

    marked_tradeoffs = mark_non_dominated_systems(tradeoff_candidates)

    return RerankerDatasetReport(
        dataset_name=dataset.name,
        model_name=cache_bundle.model_name,
        max_corpus_ndcg10=max_corpus_ndcg,
        ceiling_decompositions=decompositions,
        system_tradeoffs=marked_tradeoffs,
    )


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


def export_reranker_artifacts(
    reports: Mapping[str, RerankerDatasetReport],
    metrics_dir: Path | str,
) -> Dict[str, Path]:
    """Export per-dataset and combined Stage 5 reranker & latency tradeoff JSON artifacts."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    summary_payload: Dict[str, Any] = {"datasets": {}}
    for ds_name, rep in reports.items():
        ds_path = out_dir / f"stage5_rerank_{ds_name}.json"
        rep_dict = rep.to_dict()
        _atomic_write_json(ds_path, rep_dict)
        exported[ds_name] = ds_path
        summary_payload["datasets"][ds_name] = rep_dict

    summary_path = out_dir / "stage5_rerank_summary.json"
    _atomic_write_json(summary_path, summary_payload)
    exported["summary"] = summary_path
    return exported
