"""Result diversification: Maximal Marginal Relevance (MMR).

Pure relevance ranking tends to return near-duplicate chunks (e.g. three
chunks from the same troubleshooting section). MMR re-ranks to keep results
relevant *and* diverse, which matters for citations: each cited chunk should
add new information.
"""

from __future__ import annotations

from .models import RetrievalResult
from .retrieval import tokenize


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr(results: list[RetrievalResult], k: int, lambda_: float = 0.5) -> list[RetrievalResult]:
    """Re-rank with Maximal Marginal Relevance.

    score = lambda_ * relevance - (1 - lambda_) * max_similarity_to_selected
    Chunk similarity is token-set Jaccard on the chunk text (cheap and
    dependency-free). lambda_=1.0 degenerates to pure relevance order.
    """
    if k <= 0 or not results:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    token_sets = [set(tokenize(r.chunk.text)) for r in results]
    remaining = list(range(len(results)))
    selected: list[int] = []
    while remaining and len(selected) < k:
        best_idx = -1
        best_score = float("-inf")
        for i in remaining:
            relevance = results[i].score
            if selected:
                redundancy = max(_jaccard(token_sets[i], token_sets[s]) for s in selected)
            else:
                redundancy = 0.0
            mmr_score = lambda_ * relevance - (1.0 - lambda_) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [results[i] for i in selected]
