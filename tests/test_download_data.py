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
    _get_first_value,
    atomic_write_jsonl,
    atomic_write_tsv,
    build_arg_parser,
    download_dataset,
    extract_corpus,
    extract_qrels,
    extract_queries,
    is_dataset_cached_and_valid,
    main,
    prepare_arguana_qrels,
    prepare_scifact_qrels,
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


def test_get_first_value_falsy_and_none():
    """Verify _get_first_value properly retains falsy integer 0, 0.0, and empty string without skipping."""
    assert _get_first_value({"_id": 0, "id": "fallback"}, ["_id", "id"]) == 0
    assert _get_first_value({"_id": None, "id": "fallback"}, ["_id", "id"]) == "fallback"
    assert _get_first_value({"score": 0.0}, ["score"]) == 0.0
    assert _get_first_value({"text": ""}, ["text"]) == ""
    assert _get_first_value({"other": 123}, ["_id", "id"]) is None


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


@patch("scripts.download_data.download_dataset")
def test_main_failure_exits_with_error(mock_download):
    """Verify ingestion error causes main to exit with code 1."""
    mock_download.side_effect = RuntimeError("Simulated download failure")
    exit_code = main(["--dataset", "scifact"])
    assert exit_code == 1


@patch("scripts.download_data.download_dataset")
def test_main_all_dispatches_active_datasets(mock_download):
    """Verify '--dataset all' downloads every active dataset in config."""
    exit_code = main(["--dataset", "all"])
    assert exit_code == 0
    invoked_datasets = [call.args[0] for call in mock_download.call_args_list]
    assert invoked_datasets == ["nfcorpus", "scifact", "arguana"]


def test_prepare_scifact_qrels_pure():
    """Verify SciFact split logic: candidate filtering, multi-judgment preservation, determinism."""
    # 3 test queries, t1 has 2 judgments
    test_judgments = [
        ("t1", "doc_a", 1.0),
        ("t1", "doc_b", 2.0),
        ("t2", "doc_c", 1.0),
        ("t3", "doc_d", 1.0),
    ]
    # 6 train queries:
    # tr1 has 2 judgments
    # tr6 has claim text that collides with test query t2
    train_judgments = [
        ("tr1", "doc_1", 1.0),
        ("tr1", "doc_2", 1.0),
        ("tr2", "doc_3", 1.0),
        ("tr3", "doc_4", 1.0),
        ("tr4", "doc_5", 1.0),
        ("tr5", "doc_6", 1.0),
        ("tr6", "doc_7", 1.0),
    ]
    queries = {
        "t1": "claim about gene expression",
        "t2": "obesity decreases life quality",
        "t3": "vitamin d deficiency causes rickets",
        "tr1": "aspirin reduces cardiovascular disease risk",
        "tr2": "metformin activates ampk in hepatocytes",
        "tr3": "exercise improves cognitive function in seniors",
        "tr4": "smoking increases bladder cancer risk",
        "tr5": "calcium supplementation strengthens bones",
        "tr6": "obesity decreases life quality",  # Exact text collision with t2!
    }

    dev_rows, test_rows = prepare_scifact_qrels(
        train_judgments=train_judgments,
        test_judgments=test_judgments,
        queries=queries,
        seed=42,
        dev_query_count=3,
    )

    dev_qids = {r[0] for r in dev_rows}
    test_qids = {r[0] for r in test_rows}

    assert len(dev_qids) == 3
    assert len(test_qids) == 3
    assert "tr6" not in dev_qids  # Colliding query must be filtered out
    assert dev_qids.isdisjoint(test_qids)

    dev_texts = {queries[q].lower() for q in dev_qids}
    test_texts = {queries[q].lower() for q in test_qids}
    assert dev_texts.isdisjoint(test_texts)

    # Multi-judgment preservation: if tr1 was sampled, both judgments must exist
    if "tr1" in dev_qids:
        tr1_docs = {r[1] for r in dev_rows if r[0] == "tr1"}
        assert tr1_docs == {"doc_1", "doc_2"}

    # Test rows preserved exactly
    assert len(test_rows) == 4
    t1_docs = {r[1] for r in test_rows if r[0] == "t1"}
    assert t1_docs == {"doc_a", "doc_b"}


def test_prepare_scifact_seed_invariance():
    """Verify that any random seed maintains zero ID leakage and zero text leakage."""
    test_judgments = [("t1", "d1", 1.0), ("t2", "d2", 1.0)]
    train_judgments = [
        ("tr1", "d3", 1.0),
        ("tr2", "d4", 1.0),
        ("tr3", "d5", 1.0),
        ("tr4", "d6", 1.0),
        ("tr5", "d7", 1.0),
    ]
    queries = {
        "t1": "claim one",
        "t2": "claim two",
        "tr1": "train claim alpha",
        "tr2": "train claim beta",
        "tr3": "claim one",  # Collision with t1!
        "tr4": "train claim gamma",
        "tr5": "train claim delta",
    }

    for seed in [42, 1234, 99999]:
        dev_rows, test_rows = prepare_scifact_qrels(
            train_judgments=train_judgments,
            test_judgments=test_judgments,
            queries=queries,
            seed=seed,
            dev_query_count=2,
        )
        dev_qids = {r[0] for r in dev_rows}
        test_qids = {r[0] for r in test_rows}
        assert dev_qids.isdisjoint(test_qids)
        assert "tr3" not in dev_qids
        dev_texts = {queries[q].lower() for q in dev_qids}
        test_texts = {queries[q].lower() for q in test_qids}
        assert dev_texts.isdisjoint(test_texts)


def test_prepare_arguana_qrels_pure():
    """Verify ArguAna split logic: group-aware clustering prevents cross-posted text leakage."""
    # 8 queries:
    # q1_pol and q1_int are cross-posted under different topics with identical text
    # q2_econ and q2_soc are cross-posted with identical text
    # q3, q4, q5, q6 are unique singletons
    judgments = [
        ("q1_pol", "d1", 1.0),
        ("q1_int", "d1", 1.0),
        ("q2_econ", "d2", 1.0),
        ("q2_soc", "d2", 1.0),
        ("q3", "d3", 1.0),
        ("q4", "d4", 1.0),
        ("q5", "d5", 1.0),
        ("q6", "d6", 1.0),
    ]
    queries = {
        "q1_pol": "microfinance debt cycles benefit banks",
        "q1_int": "microfinance debt cycles benefit banks",  # Cross-posted!
        "q2_econ": "rural urban migration erodes agriculture",
        "q2_soc": "rural urban migration erodes agriculture",  # Cross-posted!
        "q3": "animals feel suffering like humans",
        "q4": "space exploration budget is unjustified",
        "q5": "renewable subsidies distort energy markets",
        "q6": "universal healthcare reduces preventive costs",
    }

    dev_rows, test_rows = prepare_arguana_qrels(
        judgments=judgments,
        queries=queries,
        seed=42,
        target_dev_queries=4,
    )

    dev_qids = {r[0] for r in dev_rows}
    test_qids = {r[0] for r in test_rows}

    assert len(dev_qids) == 4
    assert len(test_qids) == 4
    assert dev_qids.isdisjoint(test_qids)

    dev_texts = {queries[q].lower() for q in dev_qids}
    test_texts = {queries[q].lower() for q in test_qids}
    assert dev_texts.isdisjoint(test_texts)

    # Cross-posted pairs must stay together
    if "q1_pol" in dev_qids:
        assert "q1_int" in dev_qids
    else:
        assert "q1_pol" in test_qids and "q1_int" in test_qids


def test_prepare_arguana_seed_invariance():
    """Verify that any random seed maintains zero ID and zero text leakage in ArguAna."""
    judgments = [
        ("q1_a", "d1", 1.0), ("q1_b", "d1", 1.0),
        ("q2_a", "d2", 1.0), ("q2_b", "d2", 1.0),
        ("q3", "d3", 1.0),
        ("q4", "d4", 1.0),
    ]
    queries = {
        "q1_a": "argument one",
        "q1_b": "argument one",
        "q2_a": "argument two",
        "q2_b": "argument two",
        "q3": "argument three",
        "q4": "argument four",
    }

    for seed in [42, 777, 99999]:
        dev_rows, test_rows = prepare_arguana_qrels(
            judgments=judgments,
            queries=queries,
            seed=seed,
            target_dev_queries=3,
        )
        dev_qids = {r[0] for r in dev_rows}
        test_qids = {r[0] for r in test_rows}
        assert dev_qids.isdisjoint(test_qids)
        dev_texts = {queries[q].lower() for q in dev_qids}
        test_texts = {queries[q].lower() for q in test_qids}
        assert dev_texts.isdisjoint(test_texts)


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
