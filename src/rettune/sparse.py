"""Stage 2: Sparse Term-Saturation & Length-Normalization Mechanics (BM25Okapi vs. BM25+).

Implements sparse lexical retrieval powered by `bm25s` with `method="bm25+"` and
`idf_method="lucene"` (strictly positive smoothed IDF matching Stage 1 EDA).

Mathematical Formulation (Lv & Zhai, CIKM 2011):
    Score_{BM25+}(d, q) = sum_{t in q cap d} c(t, q) * IDF(t) * [
        ((k1 + 1) * c(t, d)) / (k1 * (1 - b + b * |d| / avgdl) + c(t, d)) + delta
    ]
where IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)). Setting delta = 0.0 recovers
exact Lucene-IDF BM25Okapi.

Implementation Note on `bm25s` CSC Matrix Construction:
    In `bm25s` (<= 0.3.x), `method="bm25+"` internally calls `_build_nonoccurrence_array`
    which evaluates `_score_tfc_bm25plus(tf=0) = delta`, subtracts `IDF(t) * delta` from
    each occurring posting entry in the CSC matrix `data`, and then adds a scalar
    `sum_{t in q} IDF(t) * delta` to all documents (including non-matching documents) during
    `get_scores_from_ids`. Left uncorrected, that turns `delta` into a document-invariant
    constant shift and assigns non-zero scores to unmatched documents.
    We restore `nonoccurrence_array[col]` (`IDF(t) * delta`) directly into the non-zero
    CSC entries `data[indptr[col]:indptr[col+1]]` where `c(t, d) > 0` and clear
    `nonoccurrence_array = None`, yielding the exact Lv & Zhai (2011) BM25+ formulation
    while retaining `bm25s`'s vectorized CSC engine.
"""

from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import bm25s
from bm25s.stopwords import STOPWORDS_EN
import numpy as np
import snowballstemmer

from .config import BenchmarkConfig, EvaluationConfig, SparseConfig
from .data_loader import IRDataset
from .evaluation import (
    EvaluationResult,
    LatencyProfile,
    SignificanceResult,
    compare_paired_systems,
    compute_idcg_at_k,
    evaluate_run,
    rank_score_vector,
    summarize_latencies_ms,
    topk_stable_indices,
)

logger = logging.getLogger("rettune.sparse")

STOPWORDS_SET = frozenset(STOPWORDS_EN)


@dataclass(frozen=True, slots=True)
class SparseParams:
    """Hyperparameter tuple for a BM25 / BM25+ sparse retrieval configuration."""
    preset: str
    k1: float
    b: float
    delta: float

    @property
    def is_bm25_plus(self) -> bool:
        return self.delta > 0.0

    @property
    def label(self) -> str:
        variant = "BM25+" if self.delta > 0.0 else "BM25Okapi"
        return f"{variant} ({self.preset}, k1={self.k1:.2f}, b={self.b:.2f}, δ={self.delta:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "k1": self.k1,
            "b": self.b,
            "delta": self.delta,
            "variant": "bm25+" if self.delta > 0.0 else "bm25_okapi",
        }


@dataclass(frozen=True, slots=True)
class TokenizedCollection:
    """Pre-tokenized corpus and queries for a specific tokenization preset."""
    preset: str
    doc_ids: List[str]
    corpus_tokens: List[List[str]]
    corpus_token_ids: List[List[int]]
    vocab_dict: Dict[str, int]
    doc_lengths: np.ndarray
    avgdl: float
    query_tokens: Dict[str, List[str]]


@dataclass(frozen=True, slots=True)
class SingleTermDiagnostic:
    """Empirical verification of single-term query rank invariance under delta > 0."""
    single_term_query_count: int
    multi_term_query_count: int
    empty_query_count: int
    single_term_ndcg_okapi: float
    single_term_ndcg_bm25_plus: float
    single_term_max_abs_ndcg_diff: float
    single_term_exact_rank_match_rate: float
    multi_term_ndcg_okapi: float
    multi_term_ndcg_bm25_plus: float
    multi_term_ndcg_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "single_term_query_count": self.single_term_query_count,
            "multi_term_query_count": self.multi_term_query_count,
            "empty_query_count": self.empty_query_count,
            "single_term_ndcg_okapi": self.single_term_ndcg_okapi,
            "single_term_ndcg_bm25_plus": self.single_term_ndcg_bm25_plus,
            "single_term_max_abs_ndcg_diff": self.single_term_max_abs_ndcg_diff,
            "single_term_exact_rank_match_rate": self.single_term_exact_rank_match_rate,
            "multi_term_ndcg_okapi": self.multi_term_ndcg_okapi,
            "multi_term_ndcg_bm25_plus": self.multi_term_ndcg_bm25_plus,
            "multi_term_ndcg_delta": self.multi_term_ndcg_delta,
        }


@dataclass(frozen=True, slots=True)
class LengthBucketMetrics:
    """Metrics for a query bucket stratified by mean relevant passage token length."""
    bucket_name: str
    query_count: int
    min_rel_doc_len: float
    max_rel_doc_len: float
    mean_rel_doc_len: float
    okapi_ndcg_at_10: float
    bm25_plus_ndcg_at_10: float
    delta_ndcg_at_10: float
    okapi_recall_at_100: float
    bm25_plus_recall_at_100: float
    delta_recall_at_100: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_name": self.bucket_name,
            "query_count": self.query_count,
            "min_rel_doc_len": self.min_rel_doc_len,
            "max_rel_doc_len": self.max_rel_doc_len,
            "mean_rel_doc_len": self.mean_rel_doc_len,
            "okapi_ndcg@10": self.okapi_ndcg_at_10,
            "bm25_plus_ndcg@10": self.bm25_plus_ndcg_at_10,
            "delta_ndcg@10": self.delta_ndcg_at_10,
            "okapi_recall@100": self.okapi_recall_at_100,
            "bm25_plus_recall@100": self.bm25_plus_recall_at_100,
            "delta_recall@100": self.delta_recall_at_100,
        }


@dataclass(frozen=True, slots=True)
class SparseDatasetReport:
    """Comprehensive Stage 2 sparse benchmark report for a single dataset."""
    dataset_name: str
    default_okapi_params: SparseParams
    default_nostem_params: SparseParams
    best_okapi_params: SparseParams
    best_bm25_plus_params: SparseParams
    winning_sparse_params: SparseParams
    dev_default_okapi_eval: EvaluationResult
    dev_default_nostem_eval: EvaluationResult
    dev_best_okapi_eval: EvaluationResult
    dev_best_bm25_plus_eval: EvaluationResult
    dev_winning_eval: EvaluationResult
    test_default_okapi_eval: EvaluationResult
    test_default_nostem_eval: EvaluationResult
    test_best_okapi_eval: EvaluationResult
    test_best_bm25_plus_eval: EvaluationResult
    test_winning_eval: EvaluationResult
    test_winning_vs_default_sig: SignificanceResult
    test_bm25_plus_vs_default_sig: SignificanceResult
    test_bm25_plus_vs_okapi_sig: SignificanceResult
    test_nostem_vs_standard_sig: SignificanceResult
    test_latency: LatencyProfile
    single_term_diagnostic_dev: SingleTermDiagnostic
    single_term_diagnostic_test: SingleTermDiagnostic
    long_passage_stratification_test: List[LengthBucketMetrics]
    response_surface_b_grid: List[float]
    response_surface_delta_grid: List[float]
    response_surface_ndcg_dev: List[List[float]]
    response_surface_recall100_dev: List[List[float]]
    sweep_records_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "params": {
                "default_okapi": self.default_okapi_params.to_dict(),
                "default_nostem": self.default_nostem_params.to_dict(),
                "best_okapi": self.best_okapi_params.to_dict(),
                "best_bm25_plus": self.best_bm25_plus_params.to_dict(),
                "winning_sparse": self.winning_sparse_params.to_dict(),
            },
            "dev_metrics": {
                "default_okapi": self.dev_default_okapi_eval.to_summary_dict(),
                "default_nostem": self.dev_default_nostem_eval.to_summary_dict(),
                "best_okapi": self.dev_best_okapi_eval.to_summary_dict(),
                "best_bm25_plus": self.dev_best_bm25_plus_eval.to_summary_dict(),
                "winning_sparse": self.dev_winning_eval.to_summary_dict(),
            },
            "test_metrics": {
                "default_okapi": self.test_default_okapi_eval.to_summary_dict(),
                "default_nostem": self.test_default_nostem_eval.to_summary_dict(),
                "best_okapi": self.test_best_okapi_eval.to_summary_dict(),
                "best_bm25_plus": self.test_best_bm25_plus_eval.to_summary_dict(),
                "winning_sparse": self.test_winning_eval.to_summary_dict(),
            },
            "significance_tests": {
                "winning_vs_default_okapi_ndcg10": self.test_winning_vs_default_sig.to_dict(),
                "bm25_plus_vs_default_okapi_ndcg10": self.test_bm25_plus_vs_default_sig.to_dict(),
                "bm25_plus_vs_tuned_okapi_ndcg10": self.test_bm25_plus_vs_okapi_sig.to_dict(),
                "nostem_vs_standard_default_ndcg10": self.test_nostem_vs_standard_sig.to_dict(),
            },
            "test_latency": self.test_latency.to_dict(),
            "diagnostics": {
                "single_term_dev": self.single_term_diagnostic_dev.to_dict(),
                "single_term_test": self.single_term_diagnostic_test.to_dict(),
                "long_passage_stratification_test": [
                    b.to_dict() for b in self.long_passage_stratification_test
                ],
                "response_surface_dev": {
                    "preset": self.winning_sparse_params.preset,
                    "k1": self.winning_sparse_params.k1,
                    "b_grid": self.response_surface_b_grid,
                    "delta_grid": self.response_surface_delta_grid,
                    "ndcg@10": self.response_surface_ndcg_dev,
                    "recall@100": self.response_surface_recall100_dev,
                },
            },
            "sweep_records_count": self.sweep_records_count,
        }


def tokenize_with_preset(
    text: str,
    preset: str = "standard",
    token_pattern: str | re.Pattern[str] = r"\w+",
    stemmer: Optional[Any] = None,
) -> List[str]:
    """Tokenize text according to a named RetTune sparse tokenization preset.

    Supported presets:
        - 'standard': Lowercase, regex `\\w+`, English stopword removal, Snowball stemmer.
        - 'technical_nostem': Lowercase, regex `\\w+`, English stopword removal, no stemming.
    """
    if not text:
        return []
    regex = re.compile(token_pattern) if isinstance(token_pattern, str) else token_pattern
    raw_tokens = regex.findall(text.lower())
    filtered = [t for t in raw_tokens if t not in STOPWORDS_SET]
    if not filtered:
        return []

    if preset == "standard":
        active_stemmer = stemmer if stemmer is not None else snowballstemmer.stemmer("english")
        return list(active_stemmer.stemWords(filtered))
    elif preset == "technical_nostem":
        return filtered
    else:
        raise ValueError(
            f"Unsupported tokenization preset '{preset}'. Expected 'standard' or 'technical_nostem'."
        )


def build_tokenized_collection(
    dataset: IRDataset,
    preset: str = "standard",
    token_pattern: str = r"\w+",
) -> TokenizedCollection:
    """Pre-tokenize corpus and queries once for a preset and assign deterministic vocab IDs."""
    regex = re.compile(token_pattern)
    stemmer = snowballstemmer.stemmer("english") if preset == "standard" else None
    stem_word = stemmer.stemWord if stemmer is not None else None
    stem_cache: Dict[str, str] = {}

    def _fast_tokenize(text: str) -> List[str]:
        if not text:
            return []
        filtered = [t for t in regex.findall(text.lower()) if t not in STOPWORDS_SET]
        if not filtered:
            return []
        if stem_word is not None:
            return [
                stem_cache.get(t) or stem_cache.setdefault(t, stem_word(t))
                for t in filtered
            ]
        elif preset == "technical_nostem":
            return filtered
        else:
            raise ValueError(
                f"Unsupported tokenization preset '{preset}'. Expected 'standard' or 'technical_nostem'."
            )

    doc_ids = sorted(dataset.corpus.keys())
    corpus_tokens: List[List[str]] = []
    doc_lengths_list: List[int] = []
    vocab_set = set()

    for did in doc_ids:
        toks = _fast_tokenize(dataset.corpus[did].get_text())
        corpus_tokens.append(toks)
        doc_lengths_list.append(len(toks))
        vocab_set.update(toks)

    sorted_vocab = sorted(vocab_set)
    vocab_dict = {tok: idx for idx, tok in enumerate(sorted_vocab)}
    corpus_token_ids = [[vocab_dict[tok] for tok in toks] for toks in corpus_tokens]

    doc_lengths = np.array(doc_lengths_list, dtype=np.float64)
    avgdl = float(np.mean(doc_lengths)) if doc_lengths.size > 0 else 0.0

    query_tokens: Dict[str, List[str]] = {}
    for qid in sorted(dataset.queries.keys()):
        query_tokens[qid] = _fast_tokenize(dataset.queries[qid])

    return TokenizedCollection(
        preset=preset,
        doc_ids=doc_ids,
        corpus_tokens=corpus_tokens,
        corpus_token_ids=corpus_token_ids,
        vocab_dict=vocab_dict,
        doc_lengths=doc_lengths,
        avgdl=avgdl,
        query_tokens=query_tokens,
    )


class BM25PlusIndex:
    """Exact Lucene-IDF BM25Okapi / BM25+ index powered by `bm25s` CSC matrices.

    Supports fast zero-rebuild updates across `(k1, b, delta)` on the pre-built `bm25s`
    CSC sparsity structure, since the posting list indices (`indices`, `indptr`), term
    frequencies `c(t, d)`, document length ratios `|d| / avgdl`, and Lucene `IDF(t)` are
    invariant to `(k1, b, delta)`.
    """

    def __init__(
        self,
        collection: TokenizedCollection,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.0,
    ) -> None:
        self.collection = collection
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)

        # Build bm25s index with reference (k1=1.5, b=0.75, delta=1.0) so nonoccurrence_array
        # stores exact 1.0 * IDF(t) and scores["data"] stores reference Okapi scores
        ref_k1 = 1.5
        ref_b = 0.75
        self._retriever = bm25s.BM25(
            k1=ref_k1,
            b=ref_b,
            delta=1.0,
            method="bm25+",
            idf_method="lucene",
            dtype="float64",
        )
        # Index using pre-mapped (corpus_token_ids, vocab_dict) tuple
        self._retriever.index(
            (collection.corpus_token_ids, dict(collection.vocab_dict)),
            create_empty_token=False,
            show_progress=False,
        )

        indptr = self._retriever.scores["indptr"]
        indices = self._retriever.scores["indices"]
        ref_okapi_data: np.ndarray = self._retriever.scores["data"].copy()

        if self._retriever.nonoccurrence_array is not None:
            n_voc = len(self._retriever.nonoccurrence_array)
            col_counts = indptr[1 : n_voc + 1] - indptr[:n_voc]
            self._idf_per_entry: np.ndarray = self._retriever.nonoccurrence_array.repeat(col_counts)
            self._idf_vector: np.ndarray = self._retriever.nonoccurrence_array.copy()
            # Disable bm25s's global non-occurrence addition so unmatched docs stay at 0.0
            self._retriever.nonoccurrence_array = None
        else:
            self._idf_per_entry = np.zeros_like(ref_okapi_data)
            self._idf_vector = np.zeros(len(collection.vocab_dict), dtype=np.float64)

        self._dl_ratio_per_entry: np.ndarray = (
            collection.doc_lengths[indices] / collection.avgdl
            if collection.avgdl > 0.0 and indices.size > 0
            else np.ones_like(ref_okapi_data)
        )
        # Recover exact integer term frequencies c(t, d) for each CSC entry
        if ref_okapi_data.size > 0:
            norm_ref = 1.0 - ref_b + ref_b * self._dl_ratio_per_entry
            ratio = np.where(self._idf_per_entry > 0.0, ref_okapi_data / self._idf_per_entry, 0.0)
            self._tf_per_entry: np.ndarray = np.rint(
                (ratio * ref_k1 * norm_ref) / np.maximum(1e-12, (ref_k1 + 1.0) - ratio)
            )
        else:
            self._tf_per_entry = np.zeros_like(ref_okapi_data)

        self._base_okapi_data: np.ndarray = np.zeros_like(ref_okapi_data)
        self.set_params(self.k1, self.b, self.delta)

    def set_params(self, k1: float, b: float, delta: float = 0.0) -> None:
        """Update (k1, b, delta) in-place on the `bm25s` CSC matrix without rebuilding."""
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        if self._tf_per_entry.size > 0:
            denom = (
                self.k1 * (1.0 - self.b + self.b * self._dl_ratio_per_entry)
                + self._tf_per_entry
            )
            self._base_okapi_data = (
                self._idf_per_entry * ((self.k1 + 1.0) * self._tf_per_entry) / denom
            )
        self.set_delta(self.delta)

    def set_delta(self, delta: float) -> None:
        """Update delta in-place on the CSC matrix without recomputing Okapi base scores."""
        self.delta = float(delta)
        if self.delta == 0.0:
            self._retriever.scores["data"] = self._base_okapi_data.copy()
        else:
            self._retriever.scores["data"] = (
                self._base_okapi_data + self.delta * self._idf_per_entry
            )

    def score_idf_overlap_matrix(self, query_ids: Sequence[str]) -> np.ndarray:
        """Compute (Q, N) matrix of sum_{t in q cap d} c(t, q) * IDF(t) for fast delta sweeps."""
        saved_data = self._retriever.scores["data"]
        try:
            self._retriever.scores["data"] = self._idf_per_entry
            return self.score_matrix(query_ids)
        finally:
            self._retriever.scores["data"] = saved_data

    def score_query_tokens(self, query_tokens: Sequence[str]) -> np.ndarray:
        """Compute exact BM25 / BM25+ scores across all corpus documents for tokenized query."""
        num_docs = len(self.collection.doc_ids)
        if not query_tokens:
            return np.zeros(num_docs, dtype=np.float64)

        vocab = self.collection.vocab_dict
        token_ids = [vocab[t] for t in query_tokens if t in vocab]
        if not token_ids:
            return np.zeros(num_docs, dtype=np.float64)

        return self._retriever.get_scores_from_ids(token_ids)

    def count_postings_traversed(self, query_tokens: Sequence[str]) -> int:
        """Count total posting list entries traversed for hardware-invariant work accounting."""
        if not query_tokens:
            return 0
        vocab = self.collection.vocab_dict
        indptr = self._retriever.scores["indptr"]
        total = 0
        for t in query_tokens:
            tid = vocab.get(t)
            if tid is not None:
                total += int(indptr[tid + 1] - indptr[tid])
        return total

    def retrieve_top_k(
        self,
        query_ids: Sequence[str],
        top_k: int = 100,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve deterministically sorted top-K (doc_id, score) pairs for a list of query IDs."""
        run: Dict[str, List[Tuple[str, float]]] = {}
        doc_ids = self.collection.doc_ids
        q_map = self.collection.query_tokens

        for qid in query_ids:
            q_toks = q_map.get(qid, [])
            scores = self.score_query_tokens(q_toks)
            run[qid] = rank_score_vector(scores, doc_ids, top_k=top_k)
        return run

    def score_matrix(self, query_ids: Sequence[str]) -> np.ndarray:
        """Compute full (Q, N) float64 score matrix aligned with `query_ids` and `collection.doc_ids`."""
        n_q = len(query_ids)
        n_d = len(self.collection.doc_ids)
        mat = np.zeros((n_q, n_d), dtype=np.float64)
        q_map = self.collection.query_tokens
        for i, qid in enumerate(query_ids):
            mat[i] = self.score_query_tokens(q_map.get(qid, []))
        return mat


class _FastGridEvaluator:
    """Zero-allocation integer-index evaluator for fast hyperparameter grid sweeps on `dev`."""

    def __init__(
        self,
        doc_ids: Sequence[str],
        qrels: Mapping[str, Mapping[str, float]],
        primary_k: int = 10,
        recall_k: int = 100,
        relevance_threshold: float = 1.0,
    ) -> None:
        self.query_ids = sorted(qrels.keys())
        self.primary_k = int(primary_k)
        self.recall_k = int(recall_k)
        self.max_k = max(self.primary_k, self.recall_k)
        self.discounts = np.log2(np.arange(2, self.primary_k + 2, dtype=np.float64))

        doc_to_idx = {did: i for i, did in enumerate(doc_ids)}
        ndcg_thresh = min(relevance_threshold, 1e-9) if relevance_threshold <= 1.0 else relevance_threshold

        self.q_idcgs: List[float] = []
        self.q_rel_counts: List[int] = []
        self.q_gain_maps: List[Dict[int, float]] = []
        self.q_rel_sets: List[set[int]] = []

        for qid in self.query_ids:
            judgments = qrels[qid]
            idcg = compute_idcg_at_k(judgments, k=self.primary_k, relevance_threshold=ndcg_thresh)
            rel_total = sum(1 for s in judgments.values() if float(s) >= relevance_threshold)
            gain_map: Dict[int, float] = {}
            rel_set: set[int] = set()
            for did, score in judgments.items():
                idx = doc_to_idx.get(did)
                if idx is not None:
                    fscore = float(score)
                    if fscore >= ndcg_thresh and fscore > 0.0:
                        gain_map[idx] = math.pow(2.0, fscore) - 1.0
                    if fscore >= relevance_threshold:
                        rel_set.add(idx)
            self.q_idcgs.append(idcg)
            self.q_rel_counts.append(rel_total)
            self.q_gain_maps.append(gain_map)
            self.q_rel_sets.append(rel_set)

    def evaluate_score_matrix(self, score_mat: np.ndarray) -> Tuple[float, float]:
        """Compute mean (NDCG@primary_k, Recall@recall_k) on a (Q, N) score matrix."""
        n_q = len(self.query_ids)
        if n_q == 0:
            return 0.0, 0.0

        ndcg_sum = 0.0
        recall_sum = 0.0
        pk = self.primary_k
        rk = self.recall_k
        max_k = self.max_k
        discounts = self.discounts

        for i in range(n_q):
            idcg = self.q_idcgs[i]
            rel_total = self.q_rel_counts[i]
            if idcg <= 0.0 and rel_total == 0:
                continue

            top_indices = topk_stable_indices(score_mat[i], top_k=max_k)

            if idcg > 0.0 and top_indices.size > 0:
                gain_map = self.q_gain_maps[i]
                if gain_map:
                    dcg = 0.0
                    upto = min(pk, int(top_indices.size))
                    for r_idx in range(upto):
                        g = gain_map.get(int(top_indices[r_idx]), 0.0)
                        if g > 0.0:
                            dcg += g / discounts[r_idx]
                    ndcg_sum += min(1.0, dcg / idcg)

            if rel_total > 0 and top_indices.size > 0:
                rel_set = self.q_rel_sets[i]
                if rel_set:
                    hits = sum(1 for d_idx in top_indices[:rk] if int(d_idx) in rel_set)
                    recall_sum += hits / rel_total

        return float(ndcg_sum / n_q), float(recall_sum / n_q)


def compute_single_term_diagnostic(
    collection: TokenizedCollection,
    qrels: Mapping[str, Mapping[str, float]],
    k1: float,
    b: float,
    delta_plus: float,
    primary_k: int = 10,
    relevance_threshold: float = 1.0,
) -> SingleTermDiagnostic:
    """Verify mathematically and empirically that single-term queries are rank-invariant to delta.

    For any query with at most 1 unique in-vocabulary term (`|set(q_tok) cap V| <= 1`),
    every document containing that term receives the identical additive shift
    `+c(t, q) * delta * IDF(t)` while non-matching documents receive `0.0`. Because
    `c(t, d) > 0` already has strictly positive Okapi score and `delta >= 0`, document
    ordering is 100% invariant to `delta`.
    """
    effective_delta = delta_plus if delta_plus > 0.0 else 1.0
    index = BM25PlusIndex(collection, k1=k1, b=b, delta=0.0)
    qids = sorted(qrels.keys())

    run_okapi = index.retrieve_top_k(qids, top_k=primary_k)
    index.set_delta(effective_delta)
    run_plus = index.retrieve_top_k(qids, top_k=primary_k)

    eval_okapi = evaluate_run(
        run_okapi,
        qrels,
        primary_k=primary_k,
        recall_ks=[primary_k],
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )
    eval_plus = evaluate_run(
        run_plus,
        qrels,
        primary_k=primary_k,
        recall_ks=[primary_k],
        relevance_threshold=relevance_threshold,
        already_sorted=True,
    )

    vocab = collection.vocab_dict
    single_qids: List[str] = []
    multi_qids: List[str] = []
    empty_qids: List[str] = []

    for qid in qids:
        in_vocab_unique = {t for t in collection.query_tokens.get(qid, []) if t in vocab}
        if len(in_vocab_unique) == 0:
            empty_qids.append(qid)
        elif len(in_vocab_unique) == 1:
            single_qids.append(qid)
        else:
            multi_qids.append(qid)

    if single_qids:
        s_okapi = np.array([eval_okapi.per_query_ndcg[q] for q in single_qids], dtype=np.float64)
        s_plus = np.array([eval_plus.per_query_ndcg[q] for q in single_qids], dtype=np.float64)
        s_ndcg_okapi = float(np.mean(s_okapi))
        s_ndcg_plus = float(np.mean(s_plus))
        s_max_diff = float(np.max(np.abs(s_plus - s_okapi)))
        rank_matches = sum(
            1
            for q in single_qids
            if [d for d, _ in run_okapi[q]] == [d for d, _ in run_plus[q]]
        )
        exact_match_rate = float(rank_matches / len(single_qids))
    else:
        s_ndcg_okapi = 0.0
        s_ndcg_plus = 0.0
        s_max_diff = 0.0
        exact_match_rate = 1.0

    if multi_qids:
        m_okapi = np.array([eval_okapi.per_query_ndcg[q] for q in multi_qids], dtype=np.float64)
        m_plus = np.array([eval_plus.per_query_ndcg[q] for q in multi_qids], dtype=np.float64)
        m_ndcg_okapi = float(np.mean(m_okapi))
        m_ndcg_plus = float(np.mean(m_plus))
        m_delta = float(m_ndcg_plus - m_ndcg_okapi)
    else:
        m_ndcg_okapi = 0.0
        m_ndcg_plus = 0.0
        m_delta = 0.0

    return SingleTermDiagnostic(
        single_term_query_count=len(single_qids),
        multi_term_query_count=len(multi_qids),
        empty_query_count=len(empty_qids),
        single_term_ndcg_okapi=s_ndcg_okapi,
        single_term_ndcg_bm25_plus=s_ndcg_plus,
        single_term_max_abs_ndcg_diff=s_max_diff,
        single_term_exact_rank_match_rate=exact_match_rate,
        multi_term_ndcg_okapi=m_ndcg_okapi,
        multi_term_ndcg_bm25_plus=m_ndcg_plus,
        multi_term_ndcg_delta=m_delta,
    )


def compute_long_passage_stratification(
    collection: TokenizedCollection,
    qrels: Mapping[str, Mapping[str, float]],
    eval_okapi: EvaluationResult,
    eval_bm25_plus: EvaluationResult,
    relevance_threshold: float = 1.0,
) -> List[LengthBucketMetrics]:
    """Stratify queries into Short / Medium / Long tertiles by mean relevant passage length.

    Skips queries with zero in-corpus relevant passages (e.g., ArguAna's BEIR #101 missing
    corpus docs). Computes NDCG@10 and Recall@100 per tertile for Okapi vs. BM25+.
    """
    doc_len_map = {
        did: float(collection.doc_lengths[idx])
        for idx, did in enumerate(collection.doc_ids)
    }

    query_mean_lengths: List[Tuple[str, float]] = []
    for qid in sorted(qrels.keys()):
        in_corpus_rel_lengths = [
            doc_len_map[did]
            for did, score in qrels[qid].items()
            if float(score) >= relevance_threshold and did in doc_len_map
        ]
        if not in_corpus_rel_lengths:
            continue
        query_mean_lengths.append((qid, float(np.mean(in_corpus_rel_lengths))))

    if not query_mean_lengths:
        return []

    lengths_arr = np.array([val for _, val in query_mean_lengths], dtype=np.float64)
    p33, p67 = np.percentile(lengths_arr, [100.0 / 3.0, 200.0 / 3.0])

    buckets: Dict[str, List[Tuple[str, float]]] = {
        "Short (T1)": [],
        "Medium (T2)": [],
        "Long (T3)": [],
    }
    for qid, mlen in query_mean_lengths:
        if mlen <= p33:
            buckets["Short (T1)"].append((qid, mlen))
        elif mlen <= p67:
            buckets["Medium (T2)"].append((qid, mlen))
        else:
            buckets["Long (T3)"].append((qid, mlen))

    results: List[LengthBucketMetrics] = []
    recall_100_okapi = eval_okapi.per_query_recall.get(100, {})
    recall_100_plus = eval_bm25_plus.per_query_recall.get(100, {})

    for b_name in ["Short (T1)", "Medium (T2)", "Long (T3)"]:
        items = buckets[b_name]
        if not items:
            continue
        b_qids = [q for q, _ in items]
        b_lens = np.array([l for _, l in items], dtype=np.float64)

        ok_ndcg = float(np.mean([eval_okapi.per_query_ndcg[q] for q in b_qids]))
        pl_ndcg = float(np.mean([eval_bm25_plus.per_query_ndcg[q] for q in b_qids]))
        ok_rec = float(np.mean([recall_100_okapi.get(q, 0.0) for q in b_qids]))
        pl_rec = float(np.mean([recall_100_plus.get(q, 0.0) for q in b_qids]))

        results.append(
            LengthBucketMetrics(
                bucket_name=b_name,
                query_count=len(b_qids),
                min_rel_doc_len=float(np.min(b_lens)),
                max_rel_doc_len=float(np.max(b_lens)),
                mean_rel_doc_len=float(np.mean(b_lens)),
                okapi_ndcg_at_10=ok_ndcg,
                bm25_plus_ndcg_at_10=pl_ndcg,
                delta_ndcg_at_10=float(pl_ndcg - ok_ndcg),
                okapi_recall_at_100=ok_rec,
                bm25_plus_recall_at_100=pl_rec,
                delta_recall_at_100=float(pl_rec - ok_rec),
            )
        )

    return results


def profile_sparse_latency(
    index: BM25PlusIndex,
    query_ids: Sequence[str],
    top_k: int = 100,
    warmup_count: int = 5,
) -> LatencyProfile:
    """Measure warm-cache sparse retrieval latency (ms) and mean postings traversed."""
    if not query_ids:
        return summarize_latencies_ms([], 0.0, "postings")

    doc_ids = index.collection.doc_ids
    q_map = index.collection.query_tokens

    for qid in query_ids[: min(warmup_count, len(query_ids))]:
        scores = index.score_query_tokens(q_map.get(qid, []))
        _ = rank_score_vector(scores, doc_ids, top_k=top_k)

    latencies_ms: List[float] = []
    total_postings = 0
    for qid in query_ids:
        q_toks = q_map.get(qid, [])
        total_postings += index.count_postings_traversed(q_toks)
        t0 = time.perf_counter_ns()
        scores = index.score_query_tokens(q_toks)
        _ = rank_score_vector(scores, doc_ids, top_k=top_k)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    mean_postings = float(total_postings / len(query_ids))
    return summarize_latencies_ms(
        latencies_ms,
        work_units_per_query=mean_postings,
        work_unit_name="postings",
    )


def run_sparse_benchmark_for_dataset(
    dataset: IRDataset,
    sparse_cfg: SparseConfig,
    eval_cfg: EvaluationConfig,
    relevance_threshold: float = 1.0,
    token_pattern: str = r"\w+",
    seed: int = 42,
) -> Tuple[SparseDatasetReport, BM25PlusIndex, TokenizedCollection]:
    """Run full Stage 2 parameter sweep on `dev` and frozen evaluation on `test` for one dataset.

    Returns:
        (report, winning_sparse_index, winning_tokenized_collection)
    """
    logger.info(
        "Stage 2 Sparse Sweep on '%s' (docs=%d, dev_q=%d, test_q=%d)...",
        dataset.name,
        len(dataset.corpus),
        len(dataset.qrels_dev),
        len(dataset.qrels_test),
    )

    dev_qids = sorted(dataset.qrels_dev.keys())
    test_qids = sorted(dataset.qrels_test.keys())
    max_eval_k = max([eval_cfg.primary_k] + list(eval_cfg.recall_ks))

    collections: Dict[str, TokenizedCollection] = {}
    for preset in sparse_cfg.tokenization_presets:
        collections[preset] = build_tokenized_collection(
            dataset, preset=preset, token_pattern=token_pattern
        )

    fast_dev_eval = _FastGridEvaluator(
        doc_ids=sorted(dataset.corpus.keys()),
        qrels=dataset.qrels_dev,
        primary_k=eval_cfg.primary_k,
        recall_k=100,
        relevance_threshold=relevance_threshold,
    )

    # Track grid evaluations on dev: (preset, k1, b, delta) -> (ndcg@10, recall@100)
    grid_dev_metrics: Dict[Tuple[str, float, float, float], Tuple[float, float]] = {}

    best_okapi_params: Optional[SparseParams] = None
    best_okapi_key: Tuple[float, float] = (-1.0, -1.0)

    best_plus_params: Optional[SparseParams] = None
    best_plus_key: Tuple[float, float] = (-1.0, -1.0)

    winning_params: Optional[SparseParams] = None
    winning_key: Tuple[float, float] = (-1.0, -1.0)

    for preset in sparse_cfg.tokenization_presets:
        coll = collections[preset]
        idx = BM25PlusIndex(coll, k1=sparse_cfg.default_k1, b=sparse_cfg.default_b, delta=0.0)
        idf_overlap_mat = idx.score_idf_overlap_matrix(dev_qids)
        for k1 in sparse_cfg.k1_grid:
            for b in sparse_cfg.b_grid:
                idx.set_params(k1=k1, b=b, delta=0.0)
                okapi_mat = idx.score_matrix(dev_qids)
                for delta in sparse_cfg.delta_grid:
                    score_mat = okapi_mat if delta == 0.0 else (okapi_mat + float(delta) * idf_overlap_mat)
                    ndcg10, rec100 = fast_dev_eval.evaluate_score_matrix(score_mat)
                    grid_dev_metrics[(preset, float(k1), float(b), float(delta))] = (ndcg10, rec100)

                    cand_params = SparseParams(
                        preset=preset, k1=float(k1), b=float(b), delta=float(delta)
                    )
                    score_tuple = (round(ndcg10, 8), round(rec100, 8))

                    if delta == 0.0 and score_tuple > best_okapi_key:
                        best_okapi_key = score_tuple
                        best_okapi_params = cand_params

                    if delta > 0.0 and score_tuple > best_plus_key:
                        best_plus_key = score_tuple
                        best_plus_params = cand_params

                    if score_tuple > winning_key:
                        winning_key = score_tuple
                        winning_params = cand_params

    assert best_okapi_params is not None
    assert winning_params is not None
    if best_plus_params is None:
        best_plus_params = winning_params

    default_okapi_params = SparseParams(
        preset="standard",
        k1=sparse_cfg.default_k1,
        b=sparse_cfg.default_b,
        delta=0.0,
    )
    default_nostem_params = SparseParams(
        preset="technical_nostem" if "technical_nostem" in collections else sparse_cfg.tokenization_presets[0],
        k1=sparse_cfg.default_k1,
        b=sparse_cfg.default_b,
        delta=0.0,
    )

    def _eval_split(params: SparseParams, split: str) -> EvaluationResult:
        coll = collections[params.preset]
        idx = BM25PlusIndex(coll, k1=params.k1, b=params.b, delta=params.delta)
        qids = dev_qids if split == "dev" else test_qids
        qrels = dataset.qrels_dev if split == "dev" else dataset.qrels_test
        run = idx.retrieve_top_k(qids, top_k=max_eval_k)
        return evaluate_run(
            run,
            qrels,
            primary_k=eval_cfg.primary_k,
            recall_ks=eval_cfg.recall_ks,
            relevance_threshold=relevance_threshold,
            already_sorted=True,
        )

    dev_default_okapi_eval = _eval_split(default_okapi_params, "dev")
    dev_default_nostem_eval = _eval_split(default_nostem_params, "dev")
    dev_best_okapi_eval = _eval_split(best_okapi_params, "dev")
    dev_best_plus_eval = _eval_split(best_plus_params, "dev")
    dev_winning_eval = _eval_split(winning_params, "dev")

    test_default_okapi_eval = _eval_split(default_okapi_params, "test")
    test_default_nostem_eval = _eval_split(default_nostem_params, "test")
    test_best_okapi_eval = _eval_split(best_okapi_params, "test")
    test_best_plus_eval = _eval_split(best_plus_params, "test")
    test_winning_eval = _eval_split(winning_params, "test")

    # Paired statistical significance tests on test split
    sig_winning_vs_default = compare_paired_systems(
        test_winning_eval.per_query_ndcg,
        test_default_okapi_eval.per_query_ndcg,
        n_permutations=eval_cfg.permutation_n_resamples,
        n_bootstrap=eval_cfg.bootstrap_n_resamples,
        confidence_level=eval_cfg.confidence_level,
        seed=seed,
    )
    sig_plus_vs_default = compare_paired_systems(
        test_best_plus_eval.per_query_ndcg,
        test_default_okapi_eval.per_query_ndcg,
        n_permutations=eval_cfg.permutation_n_resamples,
        n_bootstrap=eval_cfg.bootstrap_n_resamples,
        confidence_level=eval_cfg.confidence_level,
        seed=seed,
    )
    sig_plus_vs_okapi = compare_paired_systems(
        test_best_plus_eval.per_query_ndcg,
        test_best_okapi_eval.per_query_ndcg,
        n_permutations=eval_cfg.permutation_n_resamples,
        n_bootstrap=eval_cfg.bootstrap_n_resamples,
        confidence_level=eval_cfg.confidence_level,
        seed=seed,
    )
    sig_nostem_vs_std = compare_paired_systems(
        test_default_nostem_eval.per_query_ndcg,
        test_default_okapi_eval.per_query_ndcg,
        n_permutations=eval_cfg.permutation_n_resamples,
        n_bootstrap=eval_cfg.bootstrap_n_resamples,
        confidence_level=eval_cfg.confidence_level,
        seed=seed,
    )

    # Winning index and latency profile on test
    winning_coll = collections[winning_params.preset]
    winning_index = BM25PlusIndex(
        winning_coll,
        k1=winning_params.k1,
        b=winning_params.b,
        delta=winning_params.delta,
    )
    test_latency = profile_sparse_latency(
        winning_index,
        test_qids,
        top_k=sparse_cfg.top_k,
    )

    # Diagnostic 1: Single-term rank invariance (at best_plus_params delta > 0)
    diag_delta = best_plus_params.delta if best_plus_params.delta > 0.0 else 0.5
    single_term_dev = compute_single_term_diagnostic(
        winning_coll,
        dataset.qrels_dev,
        k1=winning_params.k1,
        b=winning_params.b,
        delta_plus=diag_delta,
        primary_k=eval_cfg.primary_k,
        relevance_threshold=relevance_threshold,
    )
    single_term_test = compute_single_term_diagnostic(
        winning_coll,
        dataset.qrels_test,
        k1=winning_params.k1,
        b=winning_params.b,
        delta_plus=diag_delta,
        primary_k=eval_cfg.primary_k,
        relevance_threshold=relevance_threshold,
    )

    # Diagnostic 2: Long-passage rescue stratification on test
    # Compare BM25+ (best_plus_params) vs Default BM25Okapi (standard length penalty b=0.75, delta=0.0)
    # and/or same-base Okapi so the length-normalization rescue effect is directly visible.
    long_passage_buckets = compute_long_passage_stratification(
        winning_coll,
        dataset.qrels_test,
        eval_okapi=test_default_okapi_eval,
        eval_bm25_plus=test_winning_eval if winning_params.delta > 0.0 else test_best_plus_eval,
        relevance_threshold=relevance_threshold,
    )

    # Diagnostic 3: 2D response surface across (b, delta) on dev at (winning_preset, winning_k1)
    surface_ndcg: List[List[float]] = []
    surface_rec100: List[List[float]] = []
    for b_val in sparse_cfg.b_grid:
        row_ndcg: List[float] = []
        row_rec: List[float] = []
        for d_val in sparse_cfg.delta_grid:
            m_ndcg, m_rec = grid_dev_metrics[
                (winning_params.preset, float(winning_params.k1), float(b_val), float(d_val))
            ]
            row_ndcg.append(m_ndcg)
            row_rec.append(m_rec)
        surface_ndcg.append(row_ndcg)
        surface_rec100.append(row_rec)

    report = SparseDatasetReport(
        dataset_name=dataset.name,
        default_okapi_params=default_okapi_params,
        default_nostem_params=default_nostem_params,
        best_okapi_params=best_okapi_params,
        best_bm25_plus_params=best_plus_params,
        winning_sparse_params=winning_params,
        dev_default_okapi_eval=dev_default_okapi_eval,
        dev_default_nostem_eval=dev_default_nostem_eval,
        dev_best_okapi_eval=dev_best_okapi_eval,
        dev_best_bm25_plus_eval=dev_best_plus_eval,
        dev_winning_eval=dev_winning_eval,
        test_default_okapi_eval=test_default_okapi_eval,
        test_default_nostem_eval=test_default_nostem_eval,
        test_best_okapi_eval=test_best_okapi_eval,
        test_best_bm25_plus_eval=test_best_plus_eval,
        test_winning_eval=test_winning_eval,
        test_winning_vs_default_sig=sig_winning_vs_default,
        test_bm25_plus_vs_default_sig=sig_plus_vs_default,
        test_bm25_plus_vs_okapi_sig=sig_plus_vs_okapi,
        test_nostem_vs_standard_sig=sig_nostem_vs_std,
        test_latency=test_latency,
        single_term_diagnostic_dev=single_term_dev,
        single_term_diagnostic_test=single_term_test,
        long_passage_stratification_test=long_passage_buckets,
        response_surface_b_grid=[float(x) for x in sparse_cfg.b_grid],
        response_surface_delta_grid=[float(x) for x in sparse_cfg.delta_grid],
        response_surface_ndcg_dev=surface_ndcg,
        response_surface_recall100_dev=surface_rec100,
        sweep_records_count=len(grid_dev_metrics),
    )

    return report, winning_index, winning_coll


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write JSON artifact to disk."""
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


def get_or_build_winning_sparse_index(
    dataset: IRDataset,
    config: BenchmarkConfig,
) -> Tuple[SparseParams, BM25PlusIndex, TokenizedCollection]:
    """Load winning sparse hyperparameters from cached Stage 2 metrics if available, else run sweep."""
    metrics_file = config.get_metrics_dir() / f"stage2_sparse_{dataset.name}.json"
    if metrics_file.is_file():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            win = data["params"]["winning_sparse"]
            params = SparseParams(
                preset=str(win["preset"]),
                k1=float(win["k1"]),
                b=float(win["b"]),
                delta=float(win["delta"]),
            )
            coll = build_tokenized_collection(
                dataset,
                preset=params.preset,
                token_pattern=config.eda.token_pattern,
            )
            idx = BM25PlusIndex(coll, k1=params.k1, b=params.b, delta=params.delta)
            return params, idx, coll
        except Exception as exc:
            logger.warning(
                "Could not load cached Stage 2 sparse params from %s (%s); running sweep.",
                metrics_file,
                exc,
            )

    threshold = config.datasets[dataset.name].relevance_threshold
    report, idx, coll = run_sparse_benchmark_for_dataset(
        dataset=dataset,
        sparse_cfg=config.sparse,
        eval_cfg=config.evaluation,
        relevance_threshold=threshold,
        token_pattern=config.eda.token_pattern,
        seed=config.seed,
    )
    export_sparse_artifacts({dataset.name: report}, config.get_metrics_dir())
    return report.winning_sparse_params, idx, coll


def export_sparse_artifacts(
    reports: Mapping[str, SparseDatasetReport],
    metrics_dir: Path | str,
) -> Dict[str, Path]:
    """Export per-dataset and combined Stage 2 sparse benchmark JSON artifacts."""
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    summary_payload: Dict[str, Any] = {"datasets": {}}
    for ds_name, rep in reports.items():
        ds_path = out_dir / f"stage2_sparse_{ds_name}.json"
        rep_dict = rep.to_dict()
        _atomic_write_json(ds_path, rep_dict)
        exported[ds_name] = ds_path
        summary_payload["datasets"][ds_name] = rep_dict

    summary_path = out_dir / "stage2_sparse_summary.json"
    _atomic_write_json(summary_path, summary_payload)
    exported["summary"] = summary_path
    return exported
