"""Configuration schemas and loaders for RetTune."""

from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    data_dir: Path
    results_dir: Path


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    hf_repo: str
    hf_qrels_repo: str
    qrels_dev_file: str
    qrels_test_file: str
    relevance_threshold: float
    expected_docs: Optional[int] = None
    expected_dev_queries: Optional[int] = None
    expected_test_queries: Optional[int] = None
    strict_text_disjointness: bool = True

    def get_qrels_path(self, dataset_dir: Path, split: str) -> Path:
        """Resolve qrels file path for a split ('dev' or 'test')."""
        if split == "dev":
            return dataset_dir / self.qrels_dev_file
        elif split == "test":
            return dataset_dir / self.qrels_test_file
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'dev' or 'test'.")


class EDAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    percentiles: List[int]
    random_pairs_sample_size: int = Field(ge=100)
    token_pattern: str

    @field_validator("percentiles")
    @classmethod
    def validate_percentiles(cls, v: List[int]) -> List[int]:
        for p in v:
            if not (0 <= p <= 100):
                raise ValueError(f"Percentile {p} must be between 0 and 100")
        return sorted(list(set(v)))


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    paths: PathsConfig
    active_datasets: List[str]
    datasets: Dict[str, DatasetConfig]
    eda: EDAConfig

    @model_validator(mode="after")
    def validate_referential_integrity(self) -> "BenchmarkConfig":
        """Verify that every active dataset has an entry in datasets mapping."""
        missing = [d for d in self.active_datasets if d not in self.datasets]
        if missing:
            raise ValueError(f"Active datasets not found in 'datasets' configuration: {missing}")
        return self

    def get_dataset_dir(self, dataset_name: str) -> Path:
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not configured in benchmark.")
        return self.paths.data_dir / dataset_name

    def get_results_dir(self, stage: str, dataset_name: Optional[str] = None) -> Path:
        p = self.paths.results_dir / stage
        if dataset_name:
            p = p / dataset_name
        return p

    def get_qrels_path(self, dataset_name: str, split: str) -> Path:
        dataset_cfg = self.datasets.get(dataset_name)
        if not dataset_cfg:
            raise KeyError(f"Dataset '{dataset_name}' not configured in benchmark.")
        return dataset_cfg.get_qrels_path(self.get_dataset_dir(dataset_name), split)


def find_default_config() -> Path:
    """Find default config path with cascading fallback (CWD first, then repo root)."""
    cwd_candidate = Path("configs/benchmark_config.yaml")
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    repo_candidate = Path(__file__).resolve().parent.parent / "configs" / "benchmark_config.yaml"
    if repo_candidate.exists():
        return repo_candidate.resolve()

    raise FileNotFoundError("Could not locate configs/benchmark_config.yaml in CWD or repo root.")


def load_config(config_path: Optional[Path | str] = None) -> BenchmarkConfig:
    """Load configuration from a YAML file with full Pydantic validation."""
    if config_path is not None:
        target_path = Path(config_path)
        if not target_path.exists():
            raise FileNotFoundError(f"Config file not found: {target_path}")
    else:
        target_path = find_default_config()

    with open(target_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return BenchmarkConfig.model_validate(raw)
