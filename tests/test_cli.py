"""Unit and integration tests for RetTune CLI orchestration and zero-network guard."""

import argparse
import socket
import subprocess
from pathlib import Path
from typing import Sequence
import pytest

from rettune.cli import (
    apply_config_overrides,
    build_parser,
    get_missing_dataset_files,
    main,
    resolve_target_datasets,
    verify_offline_readiness,
)
from rettune.config import load_config


def test_parser_defaults():
    """Verify default arguments of the CLI parser."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.stage is None
    assert args.dataset == "all"
    assert args.config is None
    assert args.setup is False
    assert args.force is False
    assert args.plot is True
    assert args.data_dir is None
    assert args.results_dir is None
    assert args.verbose is False


def test_parser_custom_flags(tmp_path: Path):
    """Verify custom argument overrides."""
    parser = build_parser()
    custom_cfg = tmp_path / "custom.yaml"
    custom_cfg.touch()

    args = parser.parse_args([
        "--stage", "eda",
        "--dataset", "nfcorpus",
        "--config", str(custom_cfg),
        "--setup",
        "--force",
        "--no-plot",
        "--data-dir", str(tmp_path / "data"),
        "--results-dir", str(tmp_path / "results"),
        "-v",
    ])

    assert args.stage == "eda"
    assert args.dataset == "nfcorpus"
    assert args.config == custom_cfg
    assert args.setup is True
    assert args.force is True
    assert args.plot is False
    assert args.data_dir == tmp_path / "data"
    assert args.results_dir == tmp_path / "results"
    assert args.verbose is True


def test_parser_invalid_stage():
    """Verify that unsupported stages exit with code 2."""
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--stage", "nonexistent_stage"])
    assert exc_info.value.code == 2


def test_apply_config_overrides(tmp_path: Path):
    """Verify immutable override of paths using model_copy."""
    config = load_config()
    custom_data = tmp_path / "custom_data"
    custom_results = tmp_path / "custom_results"

    parser = build_parser()
    args = parser.parse_args([
        "--data-dir", str(custom_data),
        "--results-dir", str(custom_results),
    ])

    updated_config = apply_config_overrides(config, args)

    # Ensure updated config points to new paths
    assert updated_config.paths.data_dir == custom_data.resolve()
    assert updated_config.paths.results_dir == custom_results.resolve()

    # Ensure original config remains unchanged (immutability)
    assert config.paths.data_dir != custom_data.resolve()


def test_resolve_target_datasets():
    """Verify dataset resolution for 'all', specific, and invalid names."""
    config = load_config()

    # "all" resolves to active datasets in config
    all_ds = resolve_target_datasets("all", config)
    assert all_ds == list(config.active_datasets)

    # Specific valid dataset
    nf_ds = resolve_target_datasets("nfcorpus", config)
    assert nf_ds == ["nfcorpus"]

    # Unknown dataset raises informative ValueError
    with pytest.raises(ValueError, match="not configured in benchmark"):
        resolve_target_datasets("unknown_dataset_xyz", config)


def test_verify_offline_readiness_valid():
    """Verify that existing downloaded datasets pass offline readiness."""
    config = load_config()
    is_ready, missing = verify_offline_readiness(["nfcorpus"], config)
    assert is_ready is True
    assert missing == {}


def test_verify_offline_readiness_missing_files(tmp_path: Path):
    """Verify missing files are identified when data directory is empty."""
    config = load_config()
    parser = build_parser()
    args = parser.parse_args(["--data-dir", str(tmp_path)])
    empty_config = apply_config_overrides(config, args)

    is_ready, missing = verify_offline_readiness(["nfcorpus"], empty_config)
    assert is_ready is False
    assert "nfcorpus" in missing
    assert len(missing["nfcorpus"]) == 4  # corpus, queries, dev, test


def test_main_zero_network_guard_fails_fast_on_missing_data(tmp_path: Path, capsys: pytest.CaptureFixture):
    """Verify main() exits with code 1 and helpful instructions when data is missing."""
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()

    exit_code = main([
        "--data-dir", str(empty_data_dir),
        "--dataset", "nfcorpus",
    ])

    assert exit_code == 1

    captured = capsys.readouterr()
    assert "ZERO-NETWORK GUARD FAILURE" in captured.err
    assert "python benchmark.py --setup" in captured.err


def test_main_zero_network_guard_passes_on_valid_data():
    """Verify main() returns 0 on existing local dataset."""
    exit_code = main(["--dataset", "nfcorpus"])
    assert exit_code == 0


def test_main_zero_network_socket_interception(monkeypatch: pytest.MonkeyPatch):
    """Strictly assert zero network connection attempts during offline benchmark run."""
    def guarded_connect(*args, **kwargs):
        raise RuntimeError("UNAUTHORIZED NETWORK ACCESS: Attempted socket connection in offline mode.")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

    # Execution must pass without triggering guarded_connect
    exit_code = main(["--dataset", "nfcorpus", "--no-plot"])
    assert exit_code == 0


def test_main_setup_delegates_to_subprocess(monkeypatch: pytest.MonkeyPatch):
    """Verify --setup delegates to scripts/download_data.py via subprocess."""
    called_cmds = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        called_cmds.append(cmd)
        class FakeResult:
            returncode = 0
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    exit_code = main(["--setup", "--dataset", "nfcorpus", "--force"])
    assert exit_code == 0
    assert len(called_cmds) == 1

    invoked_cmd = called_cmds[0]
    assert "download_data.py" in invoked_cmd[1]
    assert "--dataset" in invoked_cmd
    assert "nfcorpus" in invoked_cmd
    assert "--force" in invoked_cmd


def test_main_setup_failure_returns_error_code(monkeypatch: pytest.MonkeyPatch):
    """Verify --setup failure propagates returncode properly."""
    def fake_failing_run(cmd, *args, **kwargs):
        class FakeResult:
            returncode = 42
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_failing_run)

    exit_code = main(["--setup", "--dataset", "nfcorpus"])
    assert exit_code == 42


def test_main_runs_stage_eda_offline(tmp_path: Path):
    """Verify main() executes Stage 1 EDA end-to-end offline on nfcorpus."""
    results_dir = tmp_path / "results"
    exit_code = main([
        "--stage", "eda",
        "--dataset", "nfcorpus",
        "--results-dir", str(results_dir),
        "--no-plot",
    ])
    assert exit_code == 0

    # Verify artifacts were generated
    eda_json = results_dir / "eda" / "nfcorpus" / "lexical_stats.json"
    assert eda_json.exists()
    assert eda_json.stat().st_size > 0


def test_main_runs_stage_all_offline(tmp_path: Path):
    """Verify main() executes all registered stages (currently EDA) offline."""
    results_dir = tmp_path / "results"
    exit_code = main([
        "--stage", "all",
        "--dataset", "nfcorpus",
        "--results-dir", str(results_dir),
        "--no-plot",
    ])
    assert exit_code == 0

    eda_json = results_dir / "eda" / "nfcorpus" / "lexical_stats.json"
    assert eda_json.exists()


def test_main_setup_then_stage_orchestration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify --setup followed by --stage runs setup then proceeds to stage."""
    setup_called = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        setup_called.append(cmd)
        class FakeResult:
            returncode = 0
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    results_dir = tmp_path / "results"
    exit_code = main([
        "--setup",
        "--stage", "eda",
        "--dataset", "nfcorpus",
        "--results-dir", str(results_dir),
        "--no-plot",
    ])

    assert exit_code == 0
    assert len(setup_called) == 1
    assert (results_dir / "eda" / "nfcorpus" / "lexical_stats.json").exists()

