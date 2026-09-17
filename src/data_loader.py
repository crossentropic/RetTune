"""Data loading, representation, validation contracts, and leakage detection for RetTune."""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Optional

from .config import BenchmarkConfig, DatasetConfig


class ContractValidationError(ValueError):
    """Raised when a dataset fails schema, referential integrity, or count validation."""
    pass


class DataLeakageError(ValueError):
    """Raised when train/dev/test splits contaminate each other (e.g., query leakage)."""
    pass


@dataclass(frozen=True, slots=True)
class Document:
    """Lightweight slotted container for a passage or document."""
    id: str
    title: str = ""
    text: str = ""

    def get_text(self, sep: str = " ") -> str:
        """Return combined title and body text with proper whitespace handling."""
        t = self.title.strip()
        c = self.text.strip()
        if t and c:
            return f"{t}{sep}{c}"
        return t or c


@dataclass(frozen=True, slots=True)
class IRDataset:
    """In-memory representation of an IR evaluation collection."""
    name: str
    corpus: Dict[str, Document]             # doc_id -> Document
    queries: Dict[str, str]                # query_id -> text (all queries)
    qrels_dev: Dict[str, Dict[str, float]]  # dev_qid -> {doc_id: score}
    qrels_test: Dict[str, Dict[str, float]] # test_qid -> {doc_id: score}

    @property
    def dev_queries(self) -> Dict[str, str]:
        """Convenience accessor returning queries active in the dev split."""
        return {qid: self.queries[qid] for qid in self.qrels_dev if qid in self.queries}

    @property
    def test_queries(self) -> Dict[str, str]:
        """Convenience accessor returning queries active in the test split."""
        return {qid: self.queries[qid] for qid in self.qrels_test if qid in self.queries}


def load_corpus(path: Path | str) -> Dict[str, Document]:
    """Load passage collection from JSONL format with strict validation."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {target_path}")

    corpus: Dict[str, Document] = {}
    with open(target_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ContractValidationError(
                    f"Corrupted JSON on line {line_num} in {target_path}: {e}"
                ) from e

            if not isinstance(data, dict):
                raise ContractValidationError(
                    f"Line {line_num} in {target_path} is not a valid JSON object."
                )

            if "_id" not in data or data["_id"] is None:
                raise ContractValidationError(
                    f"Line {line_num} in {target_path} missing or null '_id' field."
                )

            doc_id = str(data["_id"]).strip()
            if not doc_id:
                raise ContractValidationError(
                    f"Line {line_num} in {target_path} has empty '_id'."
                )

            if doc_id in corpus:
                raise ContractValidationError(
                    f"Duplicate document ID '{doc_id}' on line {line_num} in {target_path}."
                )

            title = str(data.get("title") or "").strip()
            text = str(data.get("text") or "").strip()

            if not title and not text:
                raise ContractValidationError(
                    f"Document '{doc_id}' on line {line_num} has both empty title and text."
                )

            corpus[doc_id] = Document(id=doc_id, title=title, text=text)

    return corpus


def load_queries(path: Path | str) -> Dict[str, str]:
    """Load queries from JSONL format with strict validation."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Queries file not found: {target_path}")

    queries: Dict[str, str] = {}
    with open(target_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ContractValidationError(
                    f"Corrupted JSON on line {line_num} in {target_path}: {e}"
                ) from e

            if not isinstance(data, dict):
                raise ContractValidationError(
                    f"Line {line_num} in {target_path} is not a valid JSON object."
                )

            if "_id" not in data or data["_id"] is None:
                raise ContractValidationError(
                    f"Line {line_num} in {target_path} missing or null '_id' field."
                )

            query_id = str(data["_id"]).strip()
            if not query_id:
                raise ContractValidationError(
                    f"Line {line_num} in {target_path} has empty '_id'."
                )

            if query_id in queries:
                raise ContractValidationError(
                    f"Duplicate query ID '{query_id}' on line {line_num} in {target_path}."
                )

            text = str(data.get("text") or "").strip()
            if not text:
                raise ContractValidationError(
                    f"Query '{query_id}' on line {line_num} in {target_path} has empty text."
                )

            queries[query_id] = text

    return queries


def load_qrels(path: Path | str, relevance_threshold: float = 0.0) -> Dict[str, Dict[str, float]]:
    """Load relevance judgments from TSV format into nested dictionaries."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Qrels file not found: {target_path}")

    qrels: Dict[str, Dict[str, float]] = defaultdict(dict)
    seen_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    header_checked = False

    with open(target_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.split("\t")
            if len(parts) < 3:
                raise ContractValidationError(
                    f"Malformed qrel line {line_num} in {target_path} (expected at least 3 tab-separated columns): '{line_str}'"
                )

            # Detect and skip standard TSV header row on first non-empty line
            if not header_checked:
                header_checked = True
                if parts[0].strip().lower() in ("query-id", "query_id", "qid"):
                    continue

            qid = str(parts[0]).strip()
            did = str(parts[1]).strip()

            if not qid or not did:
                raise ContractValidationError(
                    f"Empty query ID or doc ID on line {line_num} in {target_path}."
                )

            try:
                score = float(parts[2].strip())
            except ValueError as e:
                raise ContractValidationError(
                    f"Invalid relevance score '{parts[2]}' on line {line_num} in {target_path}."
                ) from e

            if not math.isfinite(score):
                raise ContractValidationError(
                    f"Non-finite relevance score '{parts[2]}' on line {line_num} in {target_path}."
                )

            if score < 0.0:
                raise ContractValidationError(
                    f"Negative relevance score {score} on line {line_num} in {target_path}."
                )

            # Verify no conflicting scores exist for the same pair, regardless of threshold
            if did in seen_scores[qid] and seen_scores[qid][did] != score:
                raise ContractValidationError(
                    f"Conflicting score for pair ({qid}, {did}) on line {line_num} in {target_path}."
                )
            seen_scores[qid][did] = score

            if score >= relevance_threshold:
                qrels[qid][did] = score

    return dict(qrels)


def verify_split_leakage(
    qrels_dev: Dict[str, Dict[str, float]],
    qrels_test: Dict[str, Dict[str, float]],
    queries: Dict[str, str],
) -> None:
    """Verify that dev and test splits are strictly disjoint in IDs and normalized text."""
    dev_qids = set(qrels_dev.keys())
    test_qids = set(qrels_test.keys())

    # 1. Check Query ID disjointness
    id_overlap = dev_qids & test_qids
    if id_overlap:
        sample = sorted(list(id_overlap))[:5]
        raise DataLeakageError(
            f"Query ID leakage detected between dev and test: {len(id_overlap)} overlapping IDs. Sample: {sample}"
        )

    # 2. Check Query Text disjointness (catches identical text under different IDs)
    def normalize_text(t: str) -> str:
        return " ".join(t.lower().split())

    dev_texts = {normalize_text(queries[qid]): qid for qid in dev_qids if qid in queries}
    for test_qid in test_qids:
        if test_qid in queries:
            norm_test = normalize_text(queries[test_qid])
            if norm_test in dev_texts:
                dev_qid = dev_texts[norm_test]
                raise DataLeakageError(
                    f"Query text leakage detected between dev query '{dev_qid}' and test query '{test_qid}': '{norm_test}'"
                )


def verify_dataset_contract(dataset: IRDataset, config: DatasetConfig) -> None:
    """Validate referential integrity and count specifications against configuration."""
    # 1. Referential integrity: qrel docs must exist in corpus
    all_qrel_docs = set()
    for q_dict in dataset.qrels_dev.values():
        all_qrel_docs.update(q_dict.keys())
    for q_dict in dataset.qrels_test.values():
        all_qrel_docs.update(q_dict.keys())

    missing_docs = all_qrel_docs - set(dataset.corpus.keys())
    if missing_docs:
        sample = sorted(list(missing_docs))[:5]
        raise ContractValidationError(
            f"Referential integrity failure in dataset '{dataset.name}': {len(missing_docs)} doc IDs in qrels missing from corpus. Sample: {sample}"
        )

    # 2. Referential integrity: qrel queries must exist in queries
    all_qrel_qids = set(dataset.qrels_dev.keys()) | set(dataset.qrels_test.keys())
    missing_queries = all_qrel_qids - set(dataset.queries.keys())
    if missing_queries:
        sample = sorted(list(missing_queries))[:5]
        raise ContractValidationError(
            f"Referential integrity failure in dataset '{dataset.name}': {len(missing_queries)} query IDs in qrels missing from queries. Sample: {sample}"
        )

    # 3. Expected counts checks if configured
    if config.expected_docs is not None and len(dataset.corpus) != config.expected_docs:
        raise ContractValidationError(
            f"Corpus count mismatch for '{dataset.name}': expected {config.expected_docs}, got {len(dataset.corpus)}"
        )

    if config.expected_dev_queries is not None and len(dataset.qrels_dev) != config.expected_dev_queries:
        raise ContractValidationError(
            f"Dev query count mismatch for '{dataset.name}': expected {config.expected_dev_queries}, got {len(dataset.qrels_dev)}"
        )

    if config.expected_test_queries is not None and len(dataset.qrels_test) != config.expected_test_queries:
        raise ContractValidationError(
            f"Test query count mismatch for '{dataset.name}': expected {config.expected_test_queries}, got {len(dataset.qrels_test)}"
        )


def load_dataset(
    dataset_name: str,
    config: BenchmarkConfig,
    verify: bool = True,
) -> IRDataset:
    """Load and validate an entire IR dataset according to benchmark configuration."""
    if dataset_name not in config.datasets:
        raise KeyError(f"Dataset '{dataset_name}' not configured in benchmark.")

    dataset_cfg = config.datasets[dataset_name]
    dataset_dir = config.get_dataset_dir(dataset_name)

    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_dev_path = config.get_qrels_path(dataset_name, "dev")
    qrels_test_path = config.get_qrels_path(dataset_name, "test")

    for path, desc in [
        (corpus_path, "corpus"),
        (queries_path, "queries"),
        (qrels_dev_path, "dev qrels"),
        (qrels_test_path, "test qrels"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required {desc} file missing for '{dataset_name}': {path}")

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels_dev = load_qrels(qrels_dev_path, relevance_threshold=dataset_cfg.relevance_threshold)
    qrels_test = load_qrels(qrels_test_path, relevance_threshold=dataset_cfg.relevance_threshold)

    dataset = IRDataset(
        name=dataset_name,
        corpus=corpus,
        queries=queries,
        qrels_dev=qrels_dev,
        qrels_test=qrels_test,
    )

    if verify:
        verify_split_leakage(qrels_dev, qrels_test, queries)
        verify_dataset_contract(dataset, dataset_cfg)

    return dataset
