"""Stage 4.1: Query-Local Score Distribution Calibration & Fixed-K Normalizers.

Implements query-local score normalizers anchored strictly on a modality's retrieved
top-K (default K=100) window:
1. `raw`: Identity mapping s(d) -> s(d).
2. `minmax`: Linear scaling to [0, 1] via (s - s_min) / (s_max - s_min + eps).
3. `zscore`: Standardization via (s - mu_K) / (sigma_K + eps).
4. `zscore_sigmoid`: Logistic squash 1 / (1 + exp(-clip(z_q(d), -500, 500))) mapping
   (-inf, +inf) -> (0, 1) and damping extreme top-1 BM25 outlier spikes without
   supervised parameter fitting.

Natural Fixed-K Sparse Padding:
    When a short sparse query matches M < K documents with positive BM25 scores (> 0.0),
    the remaining K - M slots in the fixed-K window carry their true mathematical BM25
    score of 0.0. Computing (mu_K, sigma_K, s_min, s_max) over all K slots guarantees
    s_min = 0.0 whenever M < K, avoiding ad-hoc `if len == 1` branches.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import numpy as np

SUPPORTED_CALIBRATION_METHODS = ("raw", "minmax", "zscore", "zscore_sigmoid")


@dataclass(frozen=True, slots=True)
class QueryScoreNormalizer:
    """Query-local score normalizer anchored on a fixed-K candidate window."""
    method: str
    window_k: int
    s_min: float
    s_max: float
    mean: float
    std: float
    epsilon: float = 1e-8
    sigmoid_clip: float = 500.0

    def transform_scalar(self, raw_score: float) -> float:
        """Transform a single raw score using the frozen top-K window statistics."""
        s = float(raw_score)
        if self.method == "raw":
            return s
        elif self.method == "minmax":
            return float((s - self.s_min) / (self.s_max - self.s_min + self.epsilon))
        elif self.method == "zscore":
            return float((s - self.mean) / (self.std + self.epsilon))
        elif self.method == "zscore_sigmoid":
            z = (s - self.mean) / (self.std + self.epsilon)
            z_clipped = max(-self.sigmoid_clip, min(self.sigmoid_clip, z))
            return float(1.0 / (1.0 + np.exp(-z_clipped)))
        else:
            raise ValueError(f"Unsupported calibration method: '{self.method}'")

    def transform_array(self, raw_scores: np.ndarray | Sequence[float]) -> np.ndarray:
        """Transform a 1D array of raw scores using the frozen top-K window statistics."""
        arr = np.asarray(raw_scores, dtype=np.float64)
        if self.method == "raw":
            return arr.copy()
        elif self.method == "minmax":
            return (arr - self.s_min) / (self.s_max - self.s_min + self.epsilon)
        elif self.method == "zscore":
            return (arr - self.mean) / (self.std + self.epsilon)
        elif self.method == "zscore_sigmoid":
            z = (arr - self.mean) / (self.std + self.epsilon)
            z_clipped = np.clip(z, -self.sigmoid_clip, self.sigmoid_clip)
            return 1.0 / (1.0 + np.exp(-z_clipped))
        else:
            raise ValueError(f"Unsupported calibration method: '{self.method}'")

    @property
    def cutoff_floor_score(self) -> float:
        """Return the minimum normalized score s_tilde_(K) at the top-K cutoff boundary."""
        return self.transform_scalar(self.s_min)


def fit_query_normalizer(
    top_k_scores: Sequence[float] | np.ndarray,
    method: str = "zscore_sigmoid",
    window_k: int = 100,
    pad_with_zeros: bool = True,
    epsilon: float = 1e-8,
    sigmoid_clip: float = 500.0,
) -> QueryScoreNormalizer:
    """Fit query-local normalizer statistics over a fixed-K candidate score window.

    Args:
        top_k_scores: Scores of retrieved candidates (up to K items).
        method: One of 'raw', 'minmax', 'zscore', 'zscore_sigmoid'.
        window_k: Target window depth K (default 100).
        pad_with_zeros: If True and `len(top_k_scores) < window_k`, pads the remaining
            `window_k - len(top_k_scores)` slots with `0.0` (natural fixed-K sparse padding).
        epsilon: Numerical stability guard added to denominators.
        sigmoid_clip: Clipping bound on Z-scores before logistic exp(-z).
    """
    if method not in SUPPORTED_CALIBRATION_METHODS:
        raise ValueError(
            f"Unsupported calibration method '{method}'. Expected one of {SUPPORTED_CALIBRATION_METHODS}."
        )
    if window_k <= 0:
        raise ValueError(f"window_k must be positive, got {window_k}")

    raw_arr = np.asarray(top_k_scores[:window_k], dtype=np.float64)
    n_present = int(raw_arr.size)

    if n_present == 0:
        window_arr = np.zeros(window_k, dtype=np.float64)
    elif pad_with_zeros and n_present < window_k:
        pad = np.zeros(window_k - n_present, dtype=np.float64)
        window_arr = np.concatenate([raw_arr, pad])
    else:
        window_arr = raw_arr

    s_min = float(np.min(window_arr))
    s_max = float(np.max(window_arr))
    mean_val = float(np.mean(window_arr))
    std_val = float(np.std(window_arr, ddof=0))

    return QueryScoreNormalizer(
        method=method,
        window_k=window_k,
        s_min=s_min,
        s_max=s_max,
        mean=mean_val,
        std=std_val,
        epsilon=epsilon,
        sigmoid_clip=sigmoid_clip,
    )


def calibrate_top_k_candidates(
    candidates: Sequence[Tuple[str, float]] | Mapping[str, float],
    method: str = "zscore_sigmoid",
    window_k: int = 100,
    is_sparse: bool = False,
    epsilon: float = 1e-8,
    sigmoid_clip: float = 500.0,
) -> Tuple[Dict[str, float], QueryScoreNormalizer]:
    """Fit a fixed-K normalizer on `candidates` and return normalized `{doc_id: score}`.

    For sparse candidates (`is_sparse=True`), only documents with strictly positive raw
    BM25 score (`score > 0.0`) are retained as active retrieved candidates in the returned
    dictionary, while the normalizer itself is fitted over the full `window_k` (padded
    with `0.0` for unmatched slots).
    """
    items = list(candidates.items()) if isinstance(candidates, Mapping) else list(candidates)
    items.sort(key=lambda item: (-0.0 if float(item[1]) == 0.0 else -float(item[1]), str(item[0])))
    top_items = items[:window_k]

    if is_sparse:
        active_items = [( str(did), float(s) ) for did, s in top_items if float(s) > 0.0]
        raw_scores_for_fit = [s for _, s in active_items]
        normalizer = fit_query_normalizer(
            raw_scores_for_fit,
            method=method,
            window_k=window_k,
            pad_with_zeros=True,
            epsilon=epsilon,
            sigmoid_clip=sigmoid_clip,
        )
    else:
        active_items = [(str(did), float(s)) for did, s in top_items]
        raw_scores_for_fit = [s for _, s in active_items]
        normalizer = fit_query_normalizer(
            raw_scores_for_fit,
            method=method,
            window_k=len(raw_scores_for_fit) if raw_scores_for_fit else window_k,
            pad_with_zeros=False,
            epsilon=epsilon,
            sigmoid_clip=sigmoid_clip,
        )

    if not active_items:
        return {}, normalizer

    active_scores = np.array([s for _, s in active_items], dtype=np.float64)
    normed_scores = normalizer.transform_array(active_scores)
    normed_map = {
        did: float(normed_scores[i]) for i, (did, _) in enumerate(active_items)
    }
    return normed_map, normalizer
