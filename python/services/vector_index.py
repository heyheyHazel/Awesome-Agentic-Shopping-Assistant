"""Catalog embeddings: pre-computed vectors, cosine search, and index building.

The index is built by `scripts/build_index.py` into `data/generated/` (gitignored).
Search is a full matrix multiply: at a few thousand products this costs a couple of
milliseconds, and it lets the hard filters (budget, category, brand) be applied as a
mask so the semantic ranking never proposes something the shopper ruled out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GENERATED_DIR = Path(__file__).resolve().parents[1] / "data" / "generated"
VECTORS_FILE = GENERATED_DIR / "embeddings.npy"
IDS_FILE = GENERATED_DIR / "embedding_ids.json"

# Search depth for the semantic ranking before fusion.
VECTOR_DEPTH = 50


def product_text(product) -> str:
    """The text a product is embedded and keyword-matched on.

    Both retrievers see the same string so a product cannot be findable by one
    signal and invisible to the other for a reason that is only about formatting.
    """
    return f"{product.name} {product.category} {product.brand} {' '.join(product.tags)}".strip()


class VectorIndex:
    """Product vectors in a fixed row order, searched by cosine similarity."""

    def __init__(self, ids: list[str], vectors: np.ndarray):
        self.ids = ids
        self.vectors = vectors
        self.dimension = int(vectors.shape[1]) if vectors.size else 0
        self._row = {pid: index for index, pid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        allowed_ids: set[str] | None = None,
        top_k: int = VECTOR_DEPTH,
        min_similarity: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Top matches as (product_id, similarity), best first.

        When `allowed_ids` is given, both the similarity computation and the ranking
        are restricted to those rows, so a narrow budget filter also makes the search
        proportionally cheaper. `min_similarity` drops the tail — see the note in
        README about why it defaults to zero for this model and catalog.
        """
        if not self.ids:
            return []

        if allowed_ids is None:
            scores = self.vectors @ query_vector
            rows = None
        else:
            picked = np.fromiter(
                (self._row[pid] for pid in allowed_ids if pid in self._row),
                dtype=np.int64,
                count=len(allowed_ids),
            )
            if picked.size == 0:
                return []
            scores = self.vectors[picked] @ query_vector
            rows = picked

        depth = min(top_k, scores.size)
        best = np.argpartition(-scores, depth - 1)[:depth]
        best = best[np.argsort(-scores[best], kind="stable")]

        matches: list[tuple[str, float]] = []
        for local in best:
            score = float(scores[int(local)])
            if not np.isfinite(score) or score < min_similarity:
                break
            row = int(local) if rows is None else int(rows[int(local)])
            matches.append((self.ids[row], score))
        return matches

    def save(self, vectors_file: Path | None = None, ids_file: Path | None = None) -> None:
        vectors_file = vectors_file or VECTORS_FILE
        ids_file = ids_file or IDS_FILE
        vectors_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_file, self.vectors)
        ids_file.write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")


def load(
    expected_ids: list[str] | None = None,
    *,
    vectors_file: Path | None = None,
    ids_file: Path | None = None,
) -> VectorIndex | None:
    """Load the saved index, or None when it is missing or built from other data.

    A stale index is ignored rather than used: product ids are the join key, so a
    catalog that changed shape would otherwise silently rank the wrong rows.
    """
    vectors_file = vectors_file or VECTORS_FILE
    ids_file = ids_file or IDS_FILE
    if not (vectors_file.exists() and ids_file.exists()):
        return None
    try:
        vectors = np.load(vectors_file)
        ids = json.loads(ids_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if len(ids) != vectors.shape[0]:
        return None
    if expected_ids is not None and set(ids) != set(expected_ids):
        return None
    return VectorIndex(ids, vectors.astype(np.float32))


_index: VectorIndex | None = None
_missing = False


def get_index() -> VectorIndex | None:
    """Cached index for the loaded catalog, or None when it has not been built."""
    global _index, _missing
    if _index is None and not _missing:
        from data import PRODUCTS

        _index = load([product.product_id for product in PRODUCTS])
        _missing = _index is None
    return _index


def reset_cache() -> None:
    """Drop the cached index (used by tests after rebuilding)."""
    global _index, _missing
    _index, _missing = None, False
