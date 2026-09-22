"""RetTune: Reproducible Hybrid Search Benchmark Testbed."""

from .cli import build_parser, main, verify_offline_readiness
from .config import BenchmarkConfig, DatasetConfig, EDAConfig, PathsConfig, load_config
from .data_loader import (
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

__all__ = [
    "BenchmarkConfig",
    "DatasetConfig",
    "EDAConfig",
    "PathsConfig",
    "load_config",
    "ContractValidationError",
    "DataLeakageError",
    "Document",
    "IRDataset",
    "load_corpus",
    "load_dataset",
    "load_qrels",
    "load_queries",
    "verify_dataset_contract",
    "verify_split_leakage",
    "build_parser",
    "main",
    "verify_offline_readiness",
]
