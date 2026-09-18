"""Unit tests for configuration schemas and loading."""

from pathlib import Path
import pytest
from pydantic import ValidationError
import yaml

from rettune.config import (
    BenchmarkConfig,
    EDAConfig,
    find_default_config,
    load_config,
)


def test_load_default_config():
    """Verify that configs/benchmark_config.yaml loads with valid types and values."""
    config = load_config()
    assert config.seed == 42
    assert config.paths.data_dir == Path("data")
    assert config.paths.results_dir == Path("results")
    assert "nfcorpus" in config.active_datasets
    assert "scifact" in config.active_datasets
    assert "arguana" in config.active_datasets

    # Check NFCorpus specifics
    nfc = config.datasets.get("nfcorpus")
    assert nfc is not None
    assert nfc.hf_repo == "BeIR/nfcorpus"
    assert nfc.qrels_dev_file == "qrels/dev.tsv"
    assert nfc.qrels_test_file == "qrels/test.tsv"
    assert nfc.expected_docs == 3633
    assert nfc.expected_dev_queries == 324
    assert nfc.expected_test_queries == 323
    assert nfc.strict_text_disjointness is False

    # Check SciFact defaults to strict
    sci = config.datasets.get("scifact")
    assert sci is not None
    assert sci.strict_text_disjointness is True

    # Check ArguAna 1:1 balance
    arg = config.datasets.get("arguana")
    assert arg is not None
    assert arg.expected_dev_queries == 703
    assert arg.expected_test_queries == 703
    assert arg.strict_text_disjointness is True
    assert arg.strict_doc_referential_integrity is False


def test_load_config_nonexistent():
    """Verify FileNotFoundError when explicit path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/nonexistent_file.yaml")


def test_custom_yaml_override(tmp_path: Path):
    """Verify custom YAML can override default parameters."""
    custom_yaml = tmp_path / "custom_config.yaml"
    data = {
        "seed": 999,
        "paths": {
            "data_dir": str(tmp_path / "my_data"),
            "results_dir": str(tmp_path / "my_results"),
        },
        "active_datasets": ["nfcorpus"],
        "datasets": {
            "nfcorpus": {
                "name": "nfcorpus",
                "hf_repo": "BeIR/nfcorpus",
                "hf_qrels_repo": "BeIR/nfcorpus-qrels",
                "qrels_dev_file": "qrels/dev.tsv",
                "qrels_test_file": "qrels/test.tsv",
                "relevance_threshold": 1.0,
            }
        },
        "eda": {
            "percentiles": [50, 95],
            "random_pairs_sample_size": 200,
            "token_pattern": r"\w+",
        },
    }
    with open(custom_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    cfg = load_config(custom_yaml)
    assert cfg.seed == 999
    assert cfg.paths.data_dir == tmp_path / "my_data"
    assert cfg.paths.results_dir == tmp_path / "my_results"
    assert cfg.active_datasets == ["nfcorpus"]
    assert cfg.eda.percentiles == [50, 95]
    assert cfg.eda.random_pairs_sample_size == 200


def test_invalid_percentiles():
    """Verify EDAConfig validation rejects percentiles out of [0, 100]."""
    with pytest.raises(ValueError, match="Percentile 105 must be between 0 and 100"):
        EDAConfig(
            percentiles=[50, 105],
            random_pairs_sample_size=1000,
            token_pattern=r"\w+",
        )

    with pytest.raises(ValueError, match="Percentile -5 must be between 0 and 100"):
        EDAConfig(
            percentiles=[-5, 50],
            random_pairs_sample_size=1000,
            token_pattern=r"\w+",
        )


def test_missing_required_fields(tmp_path: Path):
    """Verify ValidationError when YAML is missing required fields."""
    bad_yaml = tmp_path / "bad_config.yaml"
    with open(bad_yaml, "w", encoding="utf-8") as f:
        yaml.dump({"seed": 42}, f)

    with pytest.raises(ValidationError):
        load_config(bad_yaml)


def test_extra_forbidden_fields(tmp_path: Path):
    """Verify ValidationError when YAML contains unrecognized extra fields (extra='forbid')."""
    bad_yaml = tmp_path / "extra_fields.yaml"
    data = {
        "seed": 42,
        "paths": {"data_dir": "data", "results_dir": "results"},
        "active_datasets": ["nfcorpus"],
        "datasets": {
            "nfcorpus": {
                "name": "nfcorpus",
                "hf_repo": "BeIR/nfcorpus",
                "hf_qrels_repo": "BeIR/nfcorpus-qrels",
                "qrels_dev_file": "qrels/dev.tsv",
                "qrels_test_file": "qrels/test.tsv",
                "relevance_threshold": 1.0,
            }
        },
        "eda": {
            "percentiles": [50],
            "random_pairs_sample_size": 100,
            "token_pattern": r"\w+",
        },
        "unknown_field": "disallowed",
    }
    with open(bad_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_config(bad_yaml)


def test_referential_integrity(tmp_path: Path):
    """Verify ValidationError when active_datasets references an unconfigured dataset."""
    bad_yaml = tmp_path / "integrity_config.yaml"
    data = {
        "seed": 42,
        "paths": {"data_dir": "data", "results_dir": "results"},
        "active_datasets": ["nonexistent_dataset"],
        "datasets": {},
        "eda": {
            "percentiles": [50],
            "random_pairs_sample_size": 100,
            "token_pattern": r"\w+",
        },
    }
    with open(bad_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    with pytest.raises(ValidationError, match="Active datasets not found in 'datasets'"):
        load_config(bad_yaml)


def test_path_helpers():
    """Verify dataset and results path resolution helpers."""
    config = load_config()
    assert config.get_dataset_dir("scifact") == Path("data/scifact")
    assert config.get_results_dir("eda", "scifact") == Path("results/eda/scifact")
    assert config.get_results_dir("sparse") == Path("results/sparse")
    assert config.get_qrels_path("nfcorpus", "dev") == Path("data/nfcorpus/qrels/dev.tsv")
    assert config.get_qrels_path("nfcorpus", "test") == Path("data/nfcorpus/qrels/test.tsv")

    with pytest.raises(KeyError, match="Dataset 'unknown' not configured"):
        config.get_dataset_dir("unknown")

    with pytest.raises(ValueError, match="Unknown split 'val'"):
        config.get_qrels_path("nfcorpus", "val")

    with pytest.raises(KeyError, match="Dataset 'unknown' not configured"):
        config.get_qrels_path("unknown", "dev")


def test_find_default_config():
    """Verify cascading default config locator."""
    path = find_default_config()
    assert path.exists()
    assert path.name == "benchmark_config.yaml"


def test_find_default_config_from_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify default config resolves to repo root when executed from another directory."""
    monkeypatch.chdir(tmp_path)
    path = find_default_config()
    assert path.exists()
    assert path.name == "benchmark_config.yaml"


def test_dataset_config_strict_text_disjointness_default():
    """Verify DatasetConfig schema defaults strict_text_disjointness and doc referential integrity to True."""
    from src.config import DatasetConfig
    cfg = DatasetConfig(
        name="custom_ds",
        hf_repo="test/custom",
        hf_qrels_repo="test/custom-qrels",
        qrels_dev_file="qrels/dev.tsv",
        qrels_test_file="qrels/test.tsv",
        relevance_threshold=1.0,
    )
    assert cfg.strict_text_disjointness is True
    assert cfg.strict_doc_referential_integrity is True
