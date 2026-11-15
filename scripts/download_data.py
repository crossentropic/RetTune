#!/usr/bin/env python3
"""Data ingestion script for RetTune benchmark datasets.

Downloads IR benchmark datasets from Hugging Face Hub, transforms them into the canonical
RetTune local BEIR format under data/<dataset>/, and enforces strict schema and leakage
validation contracts before finishing.
"""

import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Ensure src directory is available on sys.path for direct script execution
try:
    from rettune.config import BenchmarkConfig, DatasetConfig, PathsConfig, load_config
    from rettune.data_loader import (
        ContractValidationError,
        DataLeakageError,
        IRDataset,
        load_dataset,
        load_queries,
    )
except ModuleNotFoundError:
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from rettune.config import BenchmarkConfig, DatasetConfig, PathsConfig, load_config
    from rettune.data_loader import (
        ContractValidationError,
        DataLeakageError,
        IRDataset,
        load_dataset,
        load_queries,
    )

from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger("rettune.download")


def setup_logging(verbose: bool = False) -> None:
    """Configure stdout logging level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=format_str, datefmt="%H:%M:%S")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and canonicalize benchmark datasets for RetTune."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nfcorpus",
        choices=["nfcorpus", "scifact", "arguana", "all"],
        help="Dataset identifier to download: 'nfcorpus', 'scifact', 'arguana', or 'all'. (default: %(default)s)",
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
        help="Path to custom benchmark_config.yaml (defaults to cascading lookup).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset files already exist and pass contract validation.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    return parser


def atomic_write_jsonl(target_path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Atomically write records to a JSONL file with crash protection.
    
    Writes to a process-unique temporary file in the target directory, flushes to disk
    with fsync, and performs an atomic replace into the final destination.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    count = 0

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            f.flush()
            os.fsync(f.fileno())

        temp_path.replace(target_path)
        logger.debug("Atomically wrote %d records to %s", count, target_path)
        return count
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def atomic_write_tsv(
    target_path: Path,
    header: Sequence[str],
    rows: Iterable[Tuple[str, str, float]],
) -> int:
    """Atomically write tab-separated values with clean score formatting.
    
    Scores that are whole numbers are formatted as clean integers (e.g., '1', '2')
    while fractional scores retain standard decimal representation, preventing
    unwanted scientific notation.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}")
    count = 0

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("\t".join(header) + "\n")
            for qid, did, score in rows:
                score_str = f"{int(score)}" if float(score).is_integer() else f"{float(score)}"
                f.write(f"{qid}\t{did}\t{score_str}\n")
                count += 1
            f.flush()
            os.fsync(f.fileno())

        temp_path.replace(target_path)
        logger.debug("Atomically wrote %d TSV rows to %s", count, target_path)
        return count
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


def _get_first_value(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    """Safely get the first non-None value among alternate dictionary keys, avoiding falsy 0 traps."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def extract_corpus(hf_repo: str) -> Iterator[Dict[str, str]]:
    """Stream and validate corpus records from Hugging Face BeIR repository."""
    logger.info("Loading corpus from Hugging Face Hub: %s (corpus)...", hf_repo)
    ds = hf_load_dataset(hf_repo, "corpus", split="corpus")
    seen_ids: Set[str] = set()

    for idx, row in enumerate(ds):
        raw_id = _get_first_value(row, ["_id", "id"])
        if raw_id is None:
            raise ValueError(f"Corpus record at index {idx} in '{hf_repo}' missing '_id' field.")

        doc_id = str(raw_id).strip()
        if not doc_id:
            raise ValueError(f"Corpus record at index {idx} in '{hf_repo}' has empty doc_id.")

        if doc_id in seen_ids:
            raise ValueError(f"Duplicate document ID '{doc_id}' encountered at index {idx} in '{hf_repo}'.")
        seen_ids.add(doc_id)

        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()

        if not title and not text:
            raise ValueError(f"Document '{doc_id}' in '{hf_repo}' has both empty title and text.")

        yield {"_id": doc_id, "title": title, "text": text}


def extract_queries(hf_repo: str) -> Iterator[Dict[str, str]]:
    """Stream and validate queries from Hugging Face BeIR repository."""
    logger.info("Loading queries from Hugging Face Hub: %s (queries)...", hf_repo)
    ds = hf_load_dataset(hf_repo, "queries", split="queries")
    seen_ids: Set[str] = set()

    for idx, row in enumerate(ds):
        raw_id = _get_first_value(row, ["_id", "id"])
        if raw_id is None:
            raise ValueError(f"Query record at index {idx} in '{hf_repo}' missing '_id' field.")

        qid = str(raw_id).strip()
        if not qid:
            raise ValueError(f"Query record at index {idx} in '{hf_repo}' has empty query_id.")

        if qid in seen_ids:
            raise ValueError(f"Duplicate query ID '{qid}' encountered at index {idx} in '{hf_repo}'.")
        seen_ids.add(qid)

        text = str(row.get("text") or "").strip()
        if not text:
            raise ValueError(f"Query '{qid}' in '{hf_repo}' has empty text.")

        yield {"_id": qid, "text": text}


def extract_qrels(hf_qrels_repo: str, split_name: str) -> Iterator[Tuple[str, str, float]]:
    """Stream and validate relevance judgments from Hugging Face BeIR qrels repository."""
    logger.info("Loading qrels from Hugging Face Hub: %s (split='%s')...", hf_qrels_repo, split_name)
    ds = hf_load_dataset(hf_qrels_repo, split=split_name)

    for idx, row in enumerate(ds):
        raw_qid = _get_first_value(row, ["query-id", "query_id", "qid"])
        raw_did = _get_first_value(row, ["corpus-id", "corpus_id", "did"])
        raw_score = _get_first_value(row, ["score"])

        if raw_qid is None or raw_did is None or raw_score is None:
            raise ValueError(
                f"Malformed qrel record at index {idx} in '{hf_qrels_repo}' split '{split_name}': {row}"
            )

        qid = str(raw_qid).strip()
        did = str(raw_did).strip()
        if not qid or not did:
            raise ValueError(f"Empty query ID or doc ID at index {idx} in '{hf_qrels_repo}'.")

        try:
            score = float(raw_score)
        except ValueError as e:
            raise ValueError(f"Invalid relevance score '{raw_score}' at index {idx}: {e}") from e

        yield (qid, did, score)


def normalize_text(t: str) -> str:
    """Normalize text for cross-split comparison preserving domain punctuation."""
    return " ".join(t.lower().split())


def prepare_scifact_qrels(
    train_judgments: Iterable[Tuple[str, str, float]],
    test_judgments: Iterable[Tuple[str, str, float]],
    queries: Dict[str, str],
    seed: int,
    dev_query_count: int = 300,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Prepare balanced dev and test qrels for SciFact with 100% zero leakage.

    1. Keeps all upstream test judgments (300 queries, 339 judgments).
    2. Identifies normalized query texts appearing in test.
    3. Filters candidate train queries: excludes any train query sharing an ID or text with test.
    4. Lexicographically sorts candidate train query IDs for platform-independent determinism.
    5. Deterministically samples dev_query_count query IDs using Random(seed).
    6. Retains all judgments for sampled query IDs.
    7. Deduplicates and canonically sorts rows by (query-id, corpus-id).
    8. Asserts zero ID leakage, zero text leakage, and expected query counts.

    Returns:
        (dev_rows, test_rows)
    """
    test_map: Dict[Tuple[str, str], float] = {}
    test_qids: Set[str] = set()
    for qid, did, score in test_judgments:
        pair = (str(qid).strip(), str(did).strip())
        if pair in test_map and test_map[pair] != score:
            raise ValueError(f"Conflicting test score for pair {pair}: {test_map[pair]} vs {score}")
        test_map[pair] = score
        test_qids.add(pair[0])

    test_texts = {normalize_text(queries[qid]) for qid in test_qids if qid in queries}

    train_by_qid: Dict[str, Dict[str, float]] = defaultdict(dict)
    for qid, did, score in train_judgments:
        sqid, sdid = str(qid).strip(), str(did).strip()
        if sdid in train_by_qid[sqid] and train_by_qid[sqid][sdid] != score:
            raise ValueError(f"Conflicting train score for pair ({sqid}, {sdid}): {train_by_qid[sqid][sdid]} vs {score}")
        train_by_qid[sqid][sdid] = score

    # Filter candidates: exclude any train query colliding with test query ID or test query text
    candidate_qids = [
        qid for qid in train_by_qid
        if qid not in test_qids and normalize_text(queries.get(qid, "")) not in test_texts
    ]
    candidate_qids.sort()

    if len(candidate_qids) < dev_query_count:
        raise ValueError(
            f"Insufficient train query candidates for SciFact: needed {dev_query_count}, got {len(candidate_qids)}"
        )

    rng = random.Random(seed)
    sampled_dev_qids = set(rng.sample(candidate_qids, dev_query_count))

    dev_rows = [
        (qid, did, score)
        for qid in sorted(sampled_dev_qids)
        for did, score in sorted(train_by_qid[qid].items())
    ]
    test_rows = [
        (qid, did, score)
        for (qid, did), score in sorted(test_map.items())
    ]

    dev_qids = {r[0] for r in dev_rows}
    if len(dev_qids) != dev_query_count:
        raise ContractValidationError(f"Expected {dev_query_count} dev queries in SciFact, got {len(dev_qids)}")
    if not dev_qids.isdisjoint(test_qids):
        raise DataLeakageError("Query ID leakage detected between dev and test in SciFact")
    dev_texts = {normalize_text(queries[q]) for q in dev_qids if q in queries}
    if not dev_texts.isdisjoint(test_texts):
        raise DataLeakageError("Query text leakage detected between dev and test in SciFact")

    return dev_rows, test_rows


def prepare_arguana_qrels(
    judgments: Iterable[Tuple[str, str, float]],
    queries: Dict[str, str],
    seed: int,
    target_dev_queries: int = 703,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str, float]]]:
    """Prepare 50/50 balanced dev and test qrels for ArguAna via group-aware clustering.

    ArguAna queries contain duplicate argument texts across categories (e.g. cross-posted
    under 'politics' and 'international'). Partitioning clusters of identical text ensures
    that cross-posted arguments are never split across dev and test.

    1. Groups judgments by (query-id, corpus-id) and validates score consistency.
    2. Clusters query IDs by normalized query text.
    3. Lexicographically sorts unique text clusters for platform-independent determinism.
    4. Shuffles text clusters using Random(seed).
    5. Greedily allocates text clusters to dev split up to target_dev_queries, remainder to test.
    6. Deduplicates and canonically sorts rows by (query-id, corpus-id).
    7. Asserts zero ID leakage, zero text leakage, and expected query counts.

    Returns:
        (dev_rows, test_rows)
    """
    judgments_by_pair: Dict[Tuple[str, str], float] = {}
    qids_to_dids: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for qid, did, score in judgments:
        pair = (str(qid).strip(), str(did).strip())
        if pair in judgments_by_pair and judgments_by_pair[pair] != score:
            raise ValueError(f"Conflicting score for pair {pair}: {judgments_by_pair[pair]} vs {score}")
        if pair not in judgments_by_pair:
            judgments_by_pair[pair] = score
            qids_to_dids[pair[0]].append((pair[1], score))

    text_to_qids: Dict[str, List[str]] = defaultdict(list)
    for qid in sorted(qids_to_dids.keys()):
        norm_text = normalize_text(queries[qid])
        text_to_qids[norm_text].append(qid)

    # Separate multi-item clusters (cross-posted duplicates) and singletons
    # Sort deterministically within each group before shuffling
    multi_clusters = [text_to_qids[txt] for txt in sorted(text_to_qids.keys()) if len(text_to_qids[txt]) > 1]
    single_clusters = [text_to_qids[txt] for txt in sorted(text_to_qids.keys()) if len(text_to_qids[txt]) == 1]

    rng = random.Random(seed)
    rng.shuffle(multi_clusters)
    rng.shuffle(single_clusters)

    dev_qids: Set[str] = set()
    test_qids: Set[str] = set()

    # Split multi-item clusters evenly between dev and test
    half_multi = len(multi_clusters) // 2
    for cluster in multi_clusters[:half_multi]:
        dev_qids.update(cluster)
    for cluster in multi_clusters[half_multi:]:
        test_qids.update(cluster)

    # Fill remaining target dev queries from singletons
    for cluster in single_clusters:
        if len(dev_qids) + len(cluster) <= target_dev_queries:
            dev_qids.update(cluster)
        else:
            test_qids.update(cluster)

    dev_rows = [
        (qid, did, score)
        for qid in sorted(dev_qids)
        for did, score in sorted(qids_to_dids[qid])
    ]
    test_rows = [
        (qid, did, score)
        for qid in sorted(test_qids)
        for did, score in sorted(qids_to_dids[qid])
    ]

    dev_qids = {r[0] for r in dev_rows}
    test_qids = {r[0] for r in test_rows}

    if len(dev_qids) != target_dev_queries:
        raise ContractValidationError(f"Expected {target_dev_queries} dev queries in ArguAna, got {len(dev_qids)}")
    if not dev_qids.isdisjoint(test_qids):
        raise DataLeakageError("Query ID leakage detected between dev and test in ArguAna")
    dev_texts = {normalize_text(queries[q]) for q in dev_qids if q in queries}
    test_texts = {normalize_text(queries[q]) for q in test_qids if q in queries}
    if not dev_texts.isdisjoint(test_texts):
        raise DataLeakageError("Query text leakage detected between dev and test in ArguAna")

    return dev_rows, test_rows


def is_dataset_cached_and_valid(dataset_name: str, config: BenchmarkConfig) -> bool:
    """Check if local dataset files exist and satisfy all schema, count, and leakage contracts."""
    dataset_dir = config.get_dataset_dir(dataset_name)
    required_files = [
        dataset_dir / "corpus.jsonl",
        dataset_dir / "queries.jsonl",
        config.get_qrels_path(dataset_name, "dev"),
        config.get_qrels_path(dataset_name, "test"),
    ]

    if not all(p.exists() for p in required_files):
        return False

    try:
        load_dataset(dataset_name, config, verify=True)
        return True
    except (ContractValidationError, DataLeakageError, FileNotFoundError, Exception) as e:
        logger.debug("Cached dataset validation failed for '%s': %s", dataset_name, e)
        return False


def download_dataset(dataset_name: str, config: BenchmarkConfig, force: bool = False) -> None:
    """Download, standardize, and verify an individual dataset."""
    if dataset_name not in config.datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not configured in benchmark config. "
            f"Available datasets: {list(config.datasets.keys())}"
        )

    if not force and is_dataset_cached_and_valid(dataset_name, config):
        logger.info("[SKIP] Dataset '%s' is already downloaded and passes all contract checks.", dataset_name)
        return

    logger.info("Ingesting dataset '%s' from Hugging Face Hub...", dataset_name)
    ds_cfg: DatasetConfig = config.datasets[dataset_name]
    dataset_dir = config.get_dataset_dir(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corpus
    corpus_path = dataset_dir / "corpus.jsonl"
    doc_count = atomic_write_jsonl(corpus_path, extract_corpus(ds_cfg.hf_repo))
    logger.info("Saved %d passages to %s", doc_count, corpus_path)

    # 2. Queries
    queries_path = dataset_dir / "queries.jsonl"
    query_count = atomic_write_jsonl(queries_path, extract_queries(ds_cfg.hf_repo))
    logger.info("Saved %d queries to %s", query_count, queries_path)

    # Load canonical queries for split text validation and group partitioning
    queries_dict = load_queries(queries_path)

    # 3. Extract Qrels and Apply Dataset-Specific Partitioning
    if dataset_name == "nfcorpus":
        dev_rows = list(extract_qrels(ds_cfg.hf_qrels_repo, split_name="validation"))
        test_rows = list(extract_qrels(ds_cfg.hf_qrels_repo, split_name="test"))
    elif dataset_name == "scifact":
        train_raw = list(extract_qrels(ds_cfg.hf_qrels_repo, split_name="train"))
        test_raw = list(extract_qrels(ds_cfg.hf_qrels_repo, split_name="test"))
        dev_rows, test_rows = prepare_scifact_qrels(
            train_judgments=train_raw,
            test_judgments=test_raw,
            queries=queries_dict,
            seed=config.seed,
            dev_query_count=ds_cfg.expected_dev_queries or 300,
        )
    elif dataset_name == "arguana":
        test_raw = list(extract_qrels(ds_cfg.hf_qrels_repo, split_name="test"))
        dev_rows, test_rows = prepare_arguana_qrels(
            judgments=test_raw,
            queries=queries_dict,
            seed=config.seed,
            target_dev_queries=ds_cfg.expected_dev_queries or 703,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dev_qrels_path = config.get_qrels_path(dataset_name, "dev")
    dev_count = atomic_write_tsv(
        dev_qrels_path,
        header=["query-id", "corpus-id", "score"],
        rows=dev_rows,
    )
    logger.info("Saved %d dev qrel judgments to %s", dev_count, dev_qrels_path)

    test_qrels_path = config.get_qrels_path(dataset_name, "test")
    test_count = atomic_write_tsv(
        test_qrels_path,
        header=["query-id", "corpus-id", "score"],
        rows=test_rows,
    )
    logger.info("Saved %d test qrel judgments to %s", test_count, test_qrels_path)

    # 4. Strict Self-Validation Gate
    logger.info("Verifying contract and leakage rules on ingested '%s' collection...", dataset_name)
    verified_dataset: IRDataset = load_dataset(dataset_name, config, verify=True)
    logger.info(
        "Successfully verified '%s' [Docs: %d, Dev Queries: %d, Test Queries: %d]",
        dataset_name,
        len(verified_dataset.corpus),
        len(verified_dataset.qrels_dev),
        len(verified_dataset.qrels_test),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint for download_data script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=args.verbose)

    try:
        config = load_config(args.config)
        if args.data_dir:
            config = config.model_copy(
                update={"paths": PathsConfig(data_dir=Path(args.data_dir).resolve(), results_dir=config.paths.results_dir)}
            )

        if args.dataset == "all":
            for ds_name in config.active_datasets:
                download_dataset(ds_name, config, force=args.force)
        else:
            download_dataset(args.dataset, config, force=args.force)
        return 0
    except Exception as e:
        logger.exception("Data ingestion failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
