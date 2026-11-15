"""Comprehensive tests for data loading, schema verification, and leakage guards."""

import json
from pathlib import Path
import pytest

from rettune.config import BenchmarkConfig, DatasetConfig
from rettune.data_loader import (
    ContractValidationError,
    DataLeakageError,
    Document,
    IRDataset,
    load_corpus,
    load_dataset,
    load_qrels,
    load_queries,
    verify_dataset_contract,
    verify_split_leakage,
)


def test_load_dataset_success(synthetic_benchmark_config: BenchmarkConfig):
    """Verify standard valid dataset loading, types, and split query accessors."""
    dataset = load_dataset("synthetic", synthetic_benchmark_config, verify=True)

    assert isinstance(dataset, IRDataset)
    assert dataset.name == "synthetic"
    assert len(dataset.corpus) == 5
    assert len(dataset.queries) == 4
    assert len(dataset.qrels_dev) == 2
    assert len(dataset.qrels_test) == 2

    # Check property convenience filters
    dev_q = dataset.dev_queries
    assert set(dev_q.keys()) == {"q_dev_1", "q_dev_2"}
    assert dev_q["q_dev_1"] == "machine learning passage"

    test_q = dataset.test_queries
    assert set(test_q.keys()) == {"q_test_1", "q_test_2"}
    assert test_q["q_test_1"] == "passage without title"


def test_document_get_text():
    """Verify Document text formatting with titles, missing titles, and custom separators."""
    # Title + Text
    doc1 = Document(id="1", title="Cancer Study", text="Clinical trial results.")
    assert doc1.get_text() == "Cancer Study Clinical trial results."
    assert doc1.get_text("\n") == "Cancer Study\nClinical trial results."

    # Title only
    doc2 = Document(id="2", title="Title Only", text="")
    assert doc2.get_text() == "Title Only"

    # Text only
    doc3 = Document(id="3", title="", text="Text Only")
    assert doc3.get_text() == "Text Only"

    # Whitespace stripping
    doc4 = Document(id="4", title="  Padded Title  ", text="  Padded Text  ")
    assert doc4.get_text(" | ") == "Padded Title | Padded Text"


def test_numeric_and_leading_zero_ids(tmp_path: Path):
    """Verify that pure digits and strings with leading zeroes are preserved strictly as strings."""
    docs = [
        {"_id": "0042", "title": "Zero", "text": "Leading zero"},
        {"_id": 42, "title": "Integer", "text": "Raw int"},
    ]
    corpus_file = tmp_path / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    corpus = load_corpus(corpus_file)
    assert "0042" in corpus
    assert "42" in corpus
    assert isinstance(corpus["0042"].id, str)
    assert isinstance(corpus["42"].id, str)
    assert corpus["0042"].id == "0042"
    assert corpus["42"].id == "42"


def test_missing_files_raise_not_found(tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig):
    """Verify FileNotFoundError is raised if any required canonical file is absent."""
    dataset_dir = tmp_path / "data" / "synthetic"
    corpus_file = dataset_dir / "corpus.jsonl"
    corpus_file.unlink()

    with pytest.raises(FileNotFoundError, match="Required corpus file missing"):
        load_dataset("synthetic", synthetic_benchmark_config, verify=True)


def test_duplicate_document_id(tmp_path: Path):
    """Verify duplicate document IDs trigger ContractValidationError."""
    docs = [
        {"_id": "dup_1", "title": "T1", "text": "Text 1"},
        {"_id": "dup_1", "title": "T2", "text": "Text 2"},
    ]
    corpus_file = tmp_path / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    with pytest.raises(ContractValidationError, match="Duplicate document ID 'dup_1'"):
        load_corpus(corpus_file)


def test_duplicate_query_id(tmp_path: Path):
    """Verify duplicate query IDs trigger ContractValidationError."""
    queries = [
        {"_id": "q_dup", "text": "Query text 1"},
        {"_id": "q_dup", "text": "Query text 2"},
    ]
    queries_file = tmp_path / "queries.jsonl"
    with open(queries_file, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    with pytest.raises(ContractValidationError, match="Duplicate query ID 'q_dup'"):
        load_queries(queries_file)


def test_empty_content_rejected(tmp_path: Path):
    """Verify empty document content or empty query text triggers ContractValidationError."""
    empty_doc_file = tmp_path / "empty_doc.jsonl"
    with open(empty_doc_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_id": "bad_doc", "title": "   ", "text": ""}) + "\n")

    with pytest.raises(ContractValidationError, match="both empty title and text"):
        load_corpus(empty_doc_file)

    null_id_doc_file = tmp_path / "null_id_doc.jsonl"
    with open(null_id_doc_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_id": None, "title": "Title", "text": "Text"}) + "\n")

    with pytest.raises(ContractValidationError, match="missing or null '_id'"):
        load_corpus(null_id_doc_file)

    empty_query_file = tmp_path / "empty_query.jsonl"
    with open(empty_query_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_id": "bad_q", "text": "   "}) + "\n")

    with pytest.raises(ContractValidationError, match="empty text"):
        load_queries(empty_query_file)

    null_id_query_file = tmp_path / "null_id_query.jsonl"
    with open(null_id_query_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_id": None, "text": "Valid query text"}) + "\n")

    with pytest.raises(ContractValidationError, match="missing or null '_id'"):
        load_queries(null_id_query_file)


def test_referential_integrity_missing_doc(tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig, dataset_writer):
    """Verify ContractValidationError if qrel references a doc missing from corpus."""
    bad_dev_qrels = [
        ("query-id", "corpus-id", "score"),
        ("q_dev_1", "NON_EXISTENT_DOC", 2.0),
    ]
    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_writer(dataset_dir, qrels_dev=bad_dev_qrels)

    with pytest.raises(ContractValidationError, match="Referential integrity failure.*doc IDs in qrels missing"):
        load_dataset("synthetic", synthetic_benchmark_config, verify=True)


def test_referential_integrity_missing_doc_tolerated_when_flag_disabled(
    tmp_path: Path,
    synthetic_benchmark_config: BenchmarkConfig,
    dataset_writer,
    caplog: pytest.LogCaptureFixture,
):
    """Verify missing doc in qrels is permitted with a warning when strict_doc_referential_integrity=False."""
    import logging
    bad_dev_qrels = [
        ("query-id", "corpus-id", "score"),
        ("q_dev_1", "NON_EXISTENT_DOC", 2.0),
        ("q_dev_2", "doc_2", 1.0),
    ]
    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_writer(dataset_dir, qrels_dev=bad_dev_qrels)

    # Disable strict_doc_referential_integrity on config
    relaxed_ds_cfg = synthetic_benchmark_config.datasets["synthetic"].model_copy(
        update={"strict_doc_referential_integrity": False}
    )
    relaxed_cfg = synthetic_benchmark_config.model_copy(
        update={"datasets": {"synthetic": relaxed_ds_cfg}}
    )

    with caplog.at_level(logging.WARNING, logger="rettune.data_loader"):
        ds = load_dataset("synthetic", relaxed_cfg, verify=True)
        assert "synthetic" == ds.name
        assert any("strict_doc_referential_integrity=False" in record.message for record in caplog.records)


def test_referential_integrity_missing_query(tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig, dataset_writer):
    """Verify ContractValidationError if qrel references a query missing from queries.jsonl."""
    bad_dev_qrels = [
        ("query-id", "corpus-id", "score"),
        ("NON_EXISTENT_QUERY", "doc_1", 2.0),
    ]
    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_writer(dataset_dir, qrels_dev=bad_dev_qrels)

    with pytest.raises(ContractValidationError, match="Referential integrity failure.*query IDs in qrels missing"):
        load_dataset("synthetic", synthetic_benchmark_config, verify=True)


def test_data_leakage_query_id(tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig, dataset_writer):
    """Verify DataLeakageError if the same query ID appears in both dev and test."""
    leaked_test_qrels = [
        ("query-id", "corpus-id", "score"),
        ("q_dev_1", "doc_3", 1.0),  # q_dev_1 is also in dev qrels!
    ]
    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_writer(dataset_dir, qrels_test=leaked_test_qrels)

    with pytest.raises(DataLeakageError, match="Query ID leakage detected between dev and test"):
        load_dataset("synthetic", synthetic_benchmark_config, verify=True)


def test_data_leakage_query_text(tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig, dataset_writer):
    """Verify DataLeakageError if different query IDs have identical normalized text in dev and test."""
    queries_with_leak = [
        {"_id": "q_dev_1", "text": "Machine Learning Passage"},
        {"_id": "q_test_1", "text": "machine  learning   passage"},  # Same text, different ID!
    ]
    dev_qrels = [("query-id", "corpus-id", "score"), ("q_dev_1", "doc_1", 1.0)]
    test_qrels = [("query-id", "corpus-id", "score"), ("q_test_1", "doc_3", 1.0)]

    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_writer(
        dataset_dir,
        queries=queries_with_leak,
        qrels_dev=dev_qrels,
        qrels_test=test_qrels,
    )

    with pytest.raises(DataLeakageError, match="Query text leakage detected"):
        load_dataset("synthetic", synthetic_benchmark_config, verify=True)


def test_data_leakage_query_text_tolerated_when_flag_disabled(
    tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig, dataset_writer, caplog
):
    """Verify that when strict_text_disjointness is False, text collisions log a warning and succeed."""
    queries_with_leak = [
        {"_id": "q_dev_1", "text": "Machine Learning Passage"},
        {"_id": "q_test_1", "text": "machine  learning   passage"},
    ]
    dev_qrels = [("query-id", "corpus-id", "score"), ("q_dev_1", "doc_1", 1.0)]
    test_qrels = [("query-id", "corpus-id", "score"), ("q_test_1", "doc_3", 1.0)]

    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_writer(
        dataset_dir,
        queries=queries_with_leak,
        qrels_dev=dev_qrels,
        qrels_test=test_qrels,
    )

    # Disable strict_text_disjointness on the dataset configuration
    relaxed_cfg = synthetic_benchmark_config.datasets["synthetic"].model_copy(
        update={"strict_text_disjointness": False, "expected_dev_queries": 1, "expected_test_queries": 1}
    )
    test_config = synthetic_benchmark_config.model_copy(
        update={"datasets": {"synthetic": relaxed_cfg}}
    )

    # Should succeed without raising DataLeakageError and log a warning
    with caplog.at_level("WARNING"):
        ds = load_dataset("synthetic", test_config, verify=True)

    assert ds is not None
    assert "Query text split collision tolerated (strict_text_disjointness=False)" in caplog.text


def test_id_leakage_still_enforced_when_text_disjointness_disabled(
    tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig, dataset_writer
):
    """Verify that Query ID leakage is strictly enforced even when text disjointness is relaxed."""
    # query ID 'q_dev_1' is present in both dev and test qrels
    leaked_test_qrels = [
        ("query-id", "corpus-id", "score"),
        ("q_dev_1", "doc_3", 1.0),
    ]
    dataset_dir = tmp_path / "data" / "synthetic"
    dataset_writer(dataset_dir, qrels_test=leaked_test_qrels)

    relaxed_cfg = synthetic_benchmark_config.datasets["synthetic"].model_copy(
        update={"strict_text_disjointness": False}
    )
    test_config = synthetic_benchmark_config.model_copy(
        update={"datasets": {"synthetic": relaxed_cfg}}
    )

    with pytest.raises(DataLeakageError, match="Query ID leakage detected between dev and test"):
        load_dataset("synthetic", test_config, verify=True)


def test_expected_counts_mismatch(tmp_path: Path, synthetic_benchmark_config: BenchmarkConfig):
    """Verify ContractValidationError if counts differ from config expectations."""
    # Modify config to expect 99 documents instead of 5
    modified_dataset_cfg = synthetic_benchmark_config.datasets["synthetic"].model_copy(
        update={"expected_docs": 99}
    )
    bad_config = synthetic_benchmark_config.model_copy(
        update={"datasets": {"synthetic": modified_dataset_cfg}}
    )

    with pytest.raises(ContractValidationError, match="Corpus count mismatch.*expected 99, got 5"):
        load_dataset("synthetic", bad_config, verify=True)


def test_malformed_qrel_records(tmp_path: Path):
    """Verify that invalid scores and malformed TSV lines raise ContractValidationError."""
    tsv_file = tmp_path / "malformed.tsv"

    # Negative score
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("q1\td1\t-1.5\n")
    with pytest.raises(ContractValidationError, match="Negative relevance score"):
        load_qrels(tsv_file)

    # Non-numeric score
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("q1\td1\tnot_a_number\n")
    with pytest.raises(ContractValidationError, match="Invalid relevance score"):
        load_qrels(tsv_file)

    # Missing columns
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("q1\tonly_one_doc\n")
    with pytest.raises(ContractValidationError, match="Malformed qrel line"):
        load_qrels(tsv_file)


def test_qrels_relevance_threshold_filtering(tmp_path: Path):
    """Verify that judgments below relevance_threshold are filtered out."""
    tsv_file = tmp_path / "threshold.tsv"
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("q1\td1\t0.0\n")
        f.write("q1\td2\t1.0\n")
        f.write("q1\td3\t2.0\n")

    # Threshold 1.0 filters out 0.0
    qrels_filtered = load_qrels(tsv_file, relevance_threshold=1.0)
    assert "d1" not in qrels_filtered["q1"]
    assert qrels_filtered["q1"]["d2"] == 1.0
    assert qrels_filtered["q1"]["d3"] == 2.0

    # Threshold 0.0 includes all
    qrels_all = load_qrels(tsv_file, relevance_threshold=0.0)
    assert qrels_all["q1"]["d1"] == 0.0
    assert qrels_all["q1"]["d2"] == 1.0


def test_non_finite_qrel_scores(tmp_path: Path):
    """Verify that nan and inf in relevance scores are caught and rejected."""
    nan_file = tmp_path / "nan.tsv"
    with open(nan_file, "w", encoding="utf-8") as f:
        f.write("q1\td1\tnan\n")

    with pytest.raises(ContractValidationError, match="Non-finite relevance score 'nan'"):
        load_qrels(nan_file)

    inf_file = tmp_path / "inf.tsv"
    with open(inf_file, "w", encoding="utf-8") as f:
        f.write("q1\td1\tinf\n")

    with pytest.raises(ContractValidationError, match="Non-finite relevance score 'inf'"):
        load_qrels(inf_file)


def test_tsv_header_variations_and_whitespace(tmp_path: Path):
    """Verify TSV parsing handles leading blank lines, whitespace around tabs, and CRLF endings."""
    tsv_file = tmp_path / "header_test.tsv"
    # Leading blank lines, spaces around tabs, CRLF line endings
    content = "\r\n  \r\nquery-id \t corpus-id \t score\r\nq1\td1\t1.0\r\n"
    with open(tsv_file, "wb") as f:
        f.write(content.encode("utf-8"))

    qrels = load_qrels(tsv_file)
    assert "q1" in qrels
    assert qrels["q1"]["d1"] == 1.0


def test_conflicting_qrel_scores_below_threshold(tmp_path: Path):
    """Verify conflicting scores for the same pair are caught even if one is below threshold."""
    tsv_file = tmp_path / "conflict.tsv"
    with open(tsv_file, "w", encoding="utf-8") as f:
        f.write("q1\td1\t0.0\n")
        f.write("q1\td1\t2.0\n")

    # With threshold 1.0, 0.0 would be skipped, but the conflict must still be detected
    with pytest.raises(ContractValidationError, match="Conflicting score for pair"):
        load_qrels(tsv_file, relevance_threshold=1.0)
