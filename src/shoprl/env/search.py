"""Deterministic BM25 search over the catalogue.

Upstream serves results from a Lucene index that is not distributed and needs a
JVM; this is a self-contained replacement with the same contract: given keywords,
return a ranked list of catalogue positions. Chinese has no whitespace word
boundaries, so text is indexed as latin words plus CJK character bigrams, which
keeps matching recall close to a word segmentation without a segmenter.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from shoprl.env.catalog import Catalog, Product

_LATIN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# Field weights, applied by repeating a field's tokens. Title and attributes carry
# the discriminating signal; descriptions are long and noisy.
FIELD_WEIGHTS = {"title": 3, "attributes": 3, "category": 2, "shop": 1, "description": 1}


def tokenize(text: str) -> list[str]:
    """Latin words plus CJK bigrams.

    Unigrams are deliberately excluded: a single character matches almost any
    Chinese title, so indexing them turns BM25 into noise. A one-character run
    is kept because it is the whole term.
    """
    text = text.lower()
    tokens = _LATIN.findall(text)
    for run in _CJK.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


def document_text(product: Product) -> str:
    parts = [product.title] * FIELD_WEIGHTS["title"]
    parts += product.attributes * FIELD_WEIGHTS["attributes"]
    parts += [product.category] * FIELD_WEIGHTS["category"]
    parts += [product.shop_name] * FIELD_WEIGHTS["shop"]
    parts += [product.description] * FIELD_WEIGHTS["description"]
    return " ".join(part for part in parts if part)


@dataclass(slots=True)
class Bm25Index:
    """Inverted index over ``Catalog.products`` positions."""

    k1: float = 1.5
    b: float = 0.75
    postings: dict[str, dict[int, int]] = field(default_factory=dict)
    lengths: list[int] = field(default_factory=list)
    avg_length: float = 0.0

    @classmethod
    def build(cls, catalog: Catalog, **kwargs) -> Bm25Index:
        index = cls(**kwargs)
        for position, product in enumerate(catalog.products):
            counts = Counter(tokenize(document_text(product)))
            index.lengths.append(sum(counts.values()))
            for token, count in counts.items():
                index.postings.setdefault(token, {})[position] = count
        index.avg_length = (sum(index.lengths) / len(index.lengths)) if index.lengths else 0.0
        return index

    def search(self, keywords: list[str], *, limit: int = 150) -> list[int]:
        """Rank documents by BM25, ties broken by catalogue order for determinism."""
        terms = [token for keyword in keywords for token in tokenize(keyword)]
        if not terms:
            return []
        total = len(self.lengths)
        scores: dict[int, float] = {}
        for term in set(terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (total - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, frequency in postings.items():
                length = self.lengths[position] or 1
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                scores[position] = scores.get(position, 0.0) + idf * frequency * (self.k1 + 1) / denominator
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [position for position, _ in ranked[:limit]]
