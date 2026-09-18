#!/usr/bin/env python3
"""Data ingestion script for RetTune benchmark datasets.

Downloads IR benchmark datasets from Hugging Face Hub, transforms them into the canonical
RetTune local BEIR format under data/<dataset>/, and enforces strict schema and leakage
validation contracts before finishing.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset as hf_load_dataset

from rettune.config import BenchmarkConfig, DatasetConfig, PathsConfig, load_config
from rettune.data_loader import (
    ContractValidationError,
    DataLeakageError,
    IRDataset,
    load_dataset,
)

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
        help="Dataset identifier to download. Task 4 canary supports 'nfcorpus'. (default: %(default)s)",
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


def extract_corpus(hf_repo: str) -> Iterator[Dict[str, str]]:
    """Stream and validate corpus records from Hugging Face BeIR repository."""
    logger.info("Loading corpus from Hugging Face Hub: %s (corpus)...", hf_repo)
    ds = hf_load_dataset(hf_repo, "corpus", split="corpus")
    seen_ids: Set[str] = set()

    for idx, row in enumerate(ds):
        raw_id = row.get("_id") or row.get("id")
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
        raw_id = row.get("_id") or row.get("id")
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
        raw_qid = row.get("query-id") or row.get("query_id") or row.get("qid")
        raw_did = row.get("corpus-id") or row.get("corpus_id") or row.get("did")
        raw_score = row.get("score")

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
    if dataset_name != "nfcorpus":
        raise NotImplementedError(
            f"Ingestion for dataset '{dataset_name}' is scheduled for Task 7 (Tri-Domain Expansion). "
            "Task 4 exclusively implements the canary dataset 'nfcorpus'."
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

    # 3. Dev Qrels (BeIR names the dev split 'validation')
    dev_qrels_path = config.get_qrels_path(dataset_name, "dev")
    dev_count = atomic_write_tsv(
        dev_qrels_path,
        header=["query-id", "corpus-id", "score"],
        rows=extract_qrels(ds_cfg.hf_qrels_repo, split_name="validation"),
    )
    logger.info("Saved %d dev qrel judgments to %s", dev_count, dev_qrels_path)

    # 4. Test Qrels
    test_qrels_path = config.get_qrels_path(dataset_name, "test")
    test_count = atomic_write_tsv(
        test_qrels_path,
        header=["query-id", "corpus-id", "score"],
        rows=extract_qrels(ds_cfg.hf_qrels_repo, split_name="test"),
    )
    logger.info("Saved %d test qrel judgments to %s", test_count, test_qrels_path)

    # 5. Strict Self-Validation Gate
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
                update={"paths": PathsConfig(data_dir=Path(args.data_dir), results_dir=config.paths.results_dir)}
            )

        if args.dataset in ("scifact", "arguana", "all"):
            raise NotImplementedError(
                f"Ingestion for dataset '{args.dataset}' is scheduled for Task 7 (Tri-Domain Expansion). "
                "Task 4 canary supports '--dataset nfcorpus'."
            )

        download_dataset(args.dataset, config, force=args.force)
        return 0
    except NotImplementedError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception("Data ingestion failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
