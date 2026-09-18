"""Unit tests for the download_data ingestion script.

All tests in this suite mock external Hugging Face network calls, strictly upholding
RetTune's zero-network testing invariant.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from rettune.config import BenchmarkConfig, DatasetConfig, PathsConfig
from scripts.download_data import (
    atomic_write_jsonl,
    atomic_write_tsv,
    build_arg_parser,
    download_dataset,
    extract_corpus,
    extract_qrels,
    extract_queries,
    is_dataset_cached_and_valid,
    main,
)


def test_build_arg_parser():
    """Verify argument parser configurations and defaults."""
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.dataset == "nfcorpus"
    assert args.data_dir is None
    assert args.config is None
    assert args.force is False
    assert args.verbose is False

    custom_args = parser.parse_args(["--dataset", "nfcorpus", "--force", "-v", "--data-dir", "/tmp/data"])
    assert custom_args.dataset == "nfcorpus"
    assert custom_args.force is True
    assert custom_args.verbose is True
    assert custom_args.data_dir == Path("/tmp/data")


def test_atomic_write_jsonl(tmp_path: Path):
    """Verify atomic JSONL writing and temp file replacement."""
    target = tmp_path / "subdir" / "test.jsonl"
    records = [{"_id": "1", "text": "hello"}, {"_id": "2", "text": "world"}]

    count = atomic_write_jsonl(target, records)
    assert count == 2
    assert target.exists()

    with open(target, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert lines == records

    # Confirm no leftover temp files
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_atomic_write_jsonl_failure_cleans_up(tmp_path: Path):
    """Verify temporary files are cleaned up if writing fails."""
    target = tmp_path / "fail.jsonl"

    def faulty_generator():
        yield {"_id": "1"}
        raise RuntimeError("Disk error simulation")

    with pytest.raises(RuntimeError, match="Disk error simulation"):
        atomic_write_jsonl(target, faulty_generator())

    assert not target.exists()
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_atomic_write_tsv(tmp_path: Path):
    """Verify atomic TSV writing and integer/decimal score formatting."""
    target = tmp_path / "qrels.tsv"
    rows = [
        ("q1", "d1", 2.0),
        ("q2", "d2", 1.5),
        ("q3", "d3", 0.0),
    ]
    header = ["query-id", "corpus-id", "score"]

    count = atomic_write_tsv(target, header, rows)
    assert count == 3
    assert target.exists()

    content = target.read_text(encoding="utf-8").strip().splitlines()
    assert content[0] == "query-id\tcorpus-id\tscore"
    assert content[1] == "q1\td1\t2"
    assert content[2] == "q2\td2\t1.5"
    assert content[3] == "q3\td3\t0"


@patch("scripts.download_data.hf_load_dataset")
def test_extract_corpus(mock_load):
    """Verify corpus extraction and field normalization."""
    mock_load.return_value = [
        {"_id": "doc1", "title": "Title 1", "text": "Body 1"},
        {"id": "doc2", "title": "Title Only", "text": ""},
        {"_id": 42, "title": "", "text": "Numeric ID"},
    ]

    records = list(extract_corpus("dummy/repo"))
    assert len(records) == 3
    assert records[0] == {"_id": "doc1", "title": "Title 1", "text": "Body 1"}
    assert records[1] == {"_id": "doc2", "title": "Title Only", "text": ""}
    assert records[2] == {"_id": "42", "title": "", "text": "Numeric ID"}


@patch("scripts.download_data.hf_load_dataset")
def test_extract_corpus_duplicate_id_raises(mock_load):
    """Verify duplicate document IDs trigger an early ValueError."""
    mock_load.return_value = [
        {"_id": "doc1", "title": "Title 1", "text": "Body 1"},
        {"_id": "doc1", "title": "Duplicate", "text": "Body 2"},
    ]

    with pytest.raises(ValueError, match="Duplicate document ID 'doc1'"):
        list(extract_corpus("dummy/repo"))


@patch("scripts.download_data.hf_load_dataset")
def test_extract_queries(mock_load):
    """Verify queries extraction and validation."""
    mock_load.return_value = [
        {"_id": "q1", "text": "query text 1"},
        {"id": "0099", "text": "query with leading zeros"},
    ]

    records = list(extract_queries("dummy/repo"))
    assert len(records) == 2
    assert records[0] == {"_id": "q1", "text": "query text 1"}
    assert records[1] == {"_id": "0099", "text": "query with leading zeros"}


@patch("scripts.download_data.hf_load_dataset")
def test_extract_queries_empty_text_raises(mock_load):
    """Verify query with empty text triggers an error."""
    mock_load.return_value = [
        {"_id": "q1", "text": "   "},
    ]

    with pytest.raises(ValueError, match="Query 'q1' in 'dummy/repo' has empty text."):
        list(extract_queries("dummy/repo"))


@patch("scripts.download_data.hf_load_dataset")
def test_extract_qrels(mock_load):
    """Verify qrels extraction handles multiple column conventions."""
    mock_load.return_value = [
        {"query-id": "q1", "corpus-id": "d1", "score": 2},
        {"qid": "q2", "did": "d2", "score": "1.0"},
    ]

    records = list(extract_qrels("dummy/qrels", "test"))
    assert len(records) == 2
    assert records[0] == ("q1", "d1", 2.0)
    assert records[1] == ("q2", "d2", 1.0)


def test_main_unsupported_dataset_exits_with_error(capsys):
    """Verify requesting an unsupported dataset exits with error code 1."""
    exit_code = main(["--dataset", "scifact"])
    assert exit_code == 1


@patch("scripts.download_data.load_dataset")
def test_is_dataset_cached_and_valid(mock_load, tmp_path: Path):
    """Verify cached dataset detection."""
    dataset_dir = tmp_path / "test_ds"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "corpus.jsonl").write_text("{}")
    (dataset_dir / "queries.jsonl").write_text("{}")
    qrels_dir = dataset_dir / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "dev.tsv").write_text("qid\tdid\tscore\n")
    (qrels_dir / "test.tsv").write_text("qid\tdid\tscore\n")

    cfg = BenchmarkConfig(
        seed=42,
        paths=PathsConfig(data_dir=tmp_path, results_dir=tmp_path / "results"),
        active_datasets=["test_ds"],
        datasets={
            "test_ds": DatasetConfig(
                name="test_ds",
                hf_repo="dummy/repo",
                hf_qrels_repo="dummy/qrels",
                qrels_dev_file="qrels/dev.tsv",
                qrels_test_file="qrels/test.tsv",
                relevance_threshold=1.0,
            )
        },
        eda={"percentiles": [50], "random_pairs_sample_size": 100, "token_pattern": "\\w+"},
    )

    mock_load.return_value = MagicMock()
    assert is_dataset_cached_and_valid("test_ds", cfg) is True

    # When load_dataset raises, returns False
    mock_load.side_effect = ValueError("Schema mismatch")
    assert is_dataset_cached_and_valid("test_ds", cfg) is False
