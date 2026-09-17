"""Pytest fixtures and synthetic data generators for RetTune tests."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from rettune.config import BenchmarkConfig, DatasetConfig, EDAConfig, PathsConfig


DEFAULT_SYNTHETIC_DOCS = [
    {"_id": "doc_1", "title": "Doc One", "text": "This is passage one about machine learning."},
    {"_id": "doc_2", "title": "Title Only Passage", "text": ""},
    {"_id": "doc_3", "title": "", "text": "This passage has text but no title."},
    {"_id": "0042", "title": "Leading Zero ID", "text": "Passage with leading zeroes in ID."},
    {"_id": 99, "title": "Integer ID", "text": "Passage with integer ID."},
]

DEFAULT_SYNTHETIC_QUERIES = [
    {"_id": "q_dev_1", "text": "machine learning passage"},
    {"_id": "q_dev_2", "text": "numeric integer identifiers"},
    {"_id": "q_test_1", "text": "passage without title"},
    {"_id": "q_test_2", "text": "leading zeroes test"},
]

DEFAULT_SYNTHETIC_DEV_QRELS = [
    ("query-id", "corpus-id", "score"),
    ("q_dev_1", "doc_1", 2.0),
    ("q_dev_2", "99", 1.0),
]

DEFAULT_SYNTHETIC_TEST_QRELS = [
    ("query-id", "corpus-id", "score"),
    ("q_test_1", "doc_3", 1.0),
    ("q_test_2", "0042", 2.0),
]


def write_dataset_files(
    target_dir: Path,
    docs: Optional[List[Dict[str, Any]]] = None,
    queries: Optional[List[Dict[str, Any]]] = None,
    qrels_dev: Optional[List[tuple]] = None,
    qrels_test: Optional[List[tuple]] = None,
) -> Path:
    """Helper to write synthetic dataset files to a given directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    qrels_dir = target_dir / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)

    docs_to_write = docs if docs is not None else DEFAULT_SYNTHETIC_DOCS
    with open(target_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in docs_to_write:
            f.write(json.dumps(doc) + "\n")

    queries_to_write = queries if queries is not None else DEFAULT_SYNTHETIC_QUERIES
    with open(target_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for query in queries_to_write:
            f.write(json.dumps(query) + "\n")

    dev_to_write = qrels_dev if qrels_dev is not None else DEFAULT_SYNTHETIC_DEV_QRELS
    with open(qrels_dir / "dev.tsv", "w", encoding="utf-8") as f:
        for row in dev_to_write:
            f.write("\t".join(str(x) for x in row) + "\n")

    test_to_write = qrels_test if qrels_test is not None else DEFAULT_SYNTHETIC_TEST_QRELS
    with open(qrels_dir / "test.tsv", "w", encoding="utf-8") as f:
        for row in test_to_write:
            f.write("\t".join(str(x) for x in row) + "\n")

    return target_dir


@pytest.fixture
def dataset_writer():
    """Fixture returning the synthetic dataset writer function."""
    return write_dataset_files


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Fixture providing a valid, standardized synthetic dataset directory."""
    dataset_dir = tmp_path / "data" / "synthetic"
    return write_dataset_files(dataset_dir)


@pytest.fixture
def synthetic_benchmark_config(tmp_path: Path, synthetic_dataset_dir: Path) -> BenchmarkConfig:
    """Fixture providing a BenchmarkConfig pointing to the synthetic dataset."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"

    dataset_cfg = DatasetConfig(
        name="synthetic",
        hf_repo="test/synthetic",
        hf_qrels_repo="test/synthetic-qrels",
        qrels_dev_file="qrels/dev.tsv",
        qrels_test_file="qrels/test.tsv",
        relevance_threshold=1.0,
        expected_docs=5,
        expected_dev_queries=2,
        expected_test_queries=2,
    )

    eda_cfg = EDAConfig(
        percentiles=[25, 50, 75, 90, 95, 99],
        random_pairs_sample_size=100,
        token_pattern=r"\w+",
    )

    return BenchmarkConfig(
        seed=42,
        paths=PathsConfig(data_dir=data_dir, results_dir=results_dir),
        active_datasets=["synthetic"],
        datasets={"synthetic": dataset_cfg},
        eda=eda_cfg,
    )
