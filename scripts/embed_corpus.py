#!/usr/bin/env python3
"""Offline dense embedding generator for RetTune (Stage 3.1).

Encodes benchmark corpora and dev/test query splits using `nomic-ai/nomic-embed-text-v1.5`
with asymmetric task prefixes (`"search_document: "` and `"search_query: "`).

Matryoshka Pre-Slice Contract:
    `nomic-ai/nomic-embed-text-v1.5`'s default SentenceTransformer pipeline includes a
    trailing `Normalize` module that L2-normalizes vectors across all 768 dimensions even
    when `normalize_embeddings=False` is passed to `model.encode()`.
    To honor the Matryoshka Representation Learning (MRL) contract—parameterless `LayerNorm`
    across all 768 dimensions *before* slicing to `d in {64, 128, 256, 512, 768}`, followed
    by L2 normalization *after* slicing—this script removes any trailing `Normalize` module
    from the SentenceTransformer sequential pipeline, extracts the pooled 768-d vectors,
    applies `torch.nn.functional.layer_norm(x, (768,))`, and atomically saves the
    post-`LayerNorm`, pre-L2-normalization `float32` arrays to:
        `data/<dataset>/embeddings/nomic-embed-text-v1.5/{corpus,queries_dev,queries_test}.npy`
    alongside `metadata.json` containing sorted ID mappings and query encoding latency.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from rettune.config import BenchmarkConfig, PathsConfig, load_config
    from rettune.data_loader import IRDataset, load_dataset
except ModuleNotFoundError:
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from rettune.config import BenchmarkConfig, PathsConfig, load_config
    from rettune.data_loader import IRDataset, load_dataset

logger = logging.getLogger("rettune.embed_corpus")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate offline Matryoshka-ready dense embeddings for RetTune."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["nfcorpus", "scifact", "arguana", "all"],
        help="Dataset identifier to embed ('nfcorpus', 'scifact', 'arguana', or 'all').",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Custom root data directory (overrides paths.data_dir in config).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom benchmark_config.yaml.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override encoding batch size.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if valid embeddings already exist.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser


def apply_pre_slice_layer_norm(raw_embeddings: np.ndarray | torch.Tensor) -> np.ndarray:
    """Apply parameterless LayerNorm across the full embedding dimension D.

    Enforces zero mean (sum_j x_j = 0) and unit variance across the D coordinates of
    each row vector before Matryoshka prefix slicing or L2 normalization.
    """
    tensor = (
        raw_embeddings.detach().to(dtype=torch.float32, device="cpu")
        if isinstance(raw_embeddings, torch.Tensor)
        else torch.as_tensor(raw_embeddings, dtype=torch.float32)
    )
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {tuple(tensor.shape)}")
    dim = tensor.shape[-1]
    normed = F.layer_norm(tensor, normalized_shape=(dim,))
    return np.ascontiguousarray(normed.numpy(), dtype=np.float32)


def strip_trailing_normalize_modules(model: Any) -> int:
    """Remove trailing Normalize modules from a SentenceTransformer sequential pipeline.

    Returns the number of Normalize modules removed.
    """
    removed = 0
    while len(model) > 0:
        last_mod = model[-1]
        cls_name = type(last_mod).__name__.lower()
        if "normalize" in cls_name:
            del model[-1]
            removed += 1
        else:
            break
    return removed


def atomic_write_npy(target_path: Path, arr: np.ndarray) -> None:
    """Atomically write a NumPy array to a .npy file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".tmp_{target_path.name}_{os.getpid()}.npy")
    try:
        with open(temp_path, "wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(target_path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write a JSON dictionary to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=target_path.parent,
            prefix=f".tmp_{target_path.name}_",
            delete=False,
            encoding="utf-8",
        ) as tf:
            temp_path = Path(tf.name)
            tf.write(json.dumps(payload, indent=2) + "\n")
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def are_embeddings_cached_and_valid(dataset_name: str, config: BenchmarkConfig) -> bool:
    """Verify whether all required dense embedding artifacts exist and match dataset counts."""
    emb_dir = config.get_embeddings_dir(dataset_name)
    corpus_npy = emb_dir / "corpus.npy"
    dev_npy = emb_dir / "queries_dev.npy"
    test_npy = emb_dir / "queries_test.npy"
    meta_json = emb_dir / "metadata.json"

    for p in (corpus_npy, dev_npy, test_npy, meta_json):
        if not p.is_file() or p.stat().st_size == 0:
            return False

    try:
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ds_cfg = config.datasets[dataset_name]
        c_arr = np.load(corpus_npy, mmap_mode="r")
        d_arr = np.load(dev_npy, mmap_mode="r")
        t_arr = np.load(test_npy, mmap_mode="r")

        full_dim = config.dense.full_dim
        if c_arr.shape != (len(meta["doc_ids"]), full_dim):
            return False
        if d_arr.shape != (len(meta["dev_query_ids"]), full_dim):
            return False
        if t_arr.shape != (len(meta["test_query_ids"]), full_dim):
            return False
        if ds_cfg.expected_docs and c_arr.shape[0] != ds_cfg.expected_docs:
            return False
        return True
    except Exception as exc:
        logger.debug("Cached embeddings validation failed for '%s': %s", dataset_name, exc)
        return False


def measure_query_encode_latency(
    model: Any,
    prefixed_queries: Sequence[str],
    sample_count: int = 50,
    warmup_count: int = 5,
) -> Dict[str, float]:
    """Measure single-query encoding wall-clock latency (p50, p95, p99 ms) for metadata.json."""
    if not prefixed_queries:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    sample = list(prefixed_queries[: min(sample_count, len(prefixed_queries))])
    # Warm up across sample shapes so Metal/CUDA compute graphs are compiled (true warm-cache latency)
    for q in sample[: max(warmup_count, len(sample))]:
        raw = model.encode([q], convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=False)
        _ = F.layer_norm(raw.float(), normalized_shape=(raw.shape[-1],))
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    latencies_ms: List[float] = []
    for q in sample:
        t0 = time.perf_counter_ns()
        raw = model.encode([q], convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=False)
        _ = F.layer_norm(raw.float(), normalized_shape=(raw.shape[-1],))
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1_000_000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])
    return {
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
    }


def _encode_texts_chunked(
    model: Any,
    texts: Sequence[str],
    batch_size: int = 32,
    chunk_size: int = 512,
) -> np.ndarray:
    """Encode texts in chunks and flush MPS cache between chunks to avoid Metal heap stalls."""
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    chunks: List[np.ndarray] = []
    n_total = len(texts)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        sub_texts = list(texts[start:end])
        with torch.inference_mode():
            sub_raw = model.encode(
                sub_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        chunks.append(apply_pre_slice_layer_norm(sub_raw))
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logger.info("  Encoded %d / %d items...", end, n_total)

    return np.ascontiguousarray(np.vstack(chunks), dtype=np.float32)


def load_nomic_encoder(model_name: str, max_seq_length: int = 512) -> Any:
    """Load SentenceTransformer encoder and strip trailing Normalize layer for MRL pre-slice contract."""
    from sentence_transformers import SentenceTransformer

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Loading dense encoder '%s' on device='%s'...", model_name, device)
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    model.max_seq_length = max_seq_length
    removed = strip_trailing_normalize_modules(model)
    logger.info(
        "Configured '%s' (max_seq_length=%d, stripped_normalize_layers=%d) for LayerNorm MRL contract.",
        model_name,
        max_seq_length,
        removed,
    )
    return model


def embed_dataset(
    dataset_name: str,
    config: BenchmarkConfig,
    model: Optional[Any] = None,
    batch_size: Optional[int] = None,
    force: bool = False,
) -> Any:
    """Encode and persist corpus, dev queries, and test queries for a single dataset.

    Returns the loaded `model` instance so it can be reused across datasets.
    """
    if not force and are_embeddings_cached_and_valid(dataset_name, config):
        logger.info("[SKIP] Dense embeddings for '%s' already exist and pass validation.", dataset_name)
        return model

    dataset: IRDataset = load_dataset(dataset_name, config, verify=True)
    if model is None:
        model = load_nomic_encoder(config.dense.model_name)

    bsize = batch_size or config.dense.batch_size
    doc_prefix = config.dense.doc_prefix
    query_prefix = config.dense.query_prefix

    doc_ids = sorted(dataset.corpus.keys())
    dev_qids = sorted(dataset.qrels_dev.keys())
    test_qids = sorted(dataset.qrels_test.keys())

    prefixed_docs = [f"{doc_prefix}{dataset.corpus[did].get_text()}" for did in doc_ids]
    prefixed_dev_queries = [f"{query_prefix}{dataset.queries[qid]}" for qid in dev_qids]
    prefixed_test_queries = [f"{query_prefix}{dataset.queries[qid]}" for qid in test_qids]

    logger.info(
        "Encoding '%s' corpus (%d passages, batch_size=%d)...",
        dataset_name,
        len(prefixed_docs),
        bsize,
    )
    corpus_emb = _encode_texts_chunked(model, prefixed_docs, batch_size=bsize, chunk_size=512)

    logger.info("Encoding '%s' dev queries (%d) and test queries (%d)...", dataset_name, len(dev_qids), len(test_qids))
    dev_emb = _encode_texts_chunked(model, prefixed_dev_queries, batch_size=bsize, chunk_size=512)
    test_emb = _encode_texts_chunked(model, prefixed_test_queries, batch_size=bsize, chunk_size=512)

    logger.info("Measuring single-query encoding latency for '%s' metadata...", dataset_name)
    encode_latency = measure_query_encode_latency(model, prefixed_test_queries, sample_count=25, warmup_count=3)

    emb_dir = config.get_embeddings_dir(dataset_name)
    atomic_write_npy(emb_dir / "corpus.npy", corpus_emb)
    atomic_write_npy(emb_dir / "queries_dev.npy", dev_emb)
    atomic_write_npy(emb_dir / "queries_test.npy", test_emb)

    metadata = {
        "dataset_name": dataset_name,
        "model_name": config.dense.model_name,
        "model_slug": config.dense.model_slug,
        "doc_prefix": doc_prefix,
        "query_prefix": query_prefix,
        "full_dim": int(corpus_emb.shape[1]),
        "mrl_contract": "post_layernorm_pre_l2",
        "num_docs": len(doc_ids),
        "num_dev_queries": len(dev_qids),
        "num_test_queries": len(test_qids),
        " encode_latency_ms": encode_latency,
        "encode_latency_ms": encode_latency,
        "doc_ids": doc_ids,
        "dev_query_ids": dev_qids,
        "test_query_ids": test_qids,
    }
    metadata.pop(" encode_latency_ms", None)
    atomic_write_json(emb_dir / "metadata.json", metadata)

    logger.info(
        "Saved embeddings for '%s' to %s [corpus=%s, dev=%s, test=%s, encode_p50=%.2fms]",
        dataset_name,
        emb_dir,
        corpus_emb.shape,
        dev_emb.shape,
        test_emb.shape,
        encode_latency["p50_ms"],
    )
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    try:
        config = load_config(args.config)
        if args.data_dir:
            config = config.model_copy(
                update={
                    "paths": PathsConfig(
                        data_dir=Path(args.data_dir).resolve(),
                        results_dir=config.paths.results_dir,
                    )
                }
            )
        target_datasets = (
            list(config.active_datasets) if args.dataset == "all" else [args.dataset]
        )
        model = None
        for ds_name in target_datasets:
            model = embed_dataset(
                ds_name,
                config=config,
                model=model,
                batch_size=args.batch_size,
                force=args.force,
            )
        return 0
    except Exception as exc:
        logger.exception("Embedding generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
