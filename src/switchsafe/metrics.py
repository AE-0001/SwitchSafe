from __future__ import annotations

import re
from collections.abc import Sequence


def normalize(text: str) -> list[str]:
    """Normalize a transcript into comparable lowercase word tokens."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> tuple[int, int, int]:
    """Return substitutions, deletions and insertions for a minimum edit alignment."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    cost = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = (i, 0, i, 0)
    for j in range(1, cols):
        cost[0][j] = (j, 0, 0, j)

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
                continue
            candidates = (
                tuple(a + b for a, b in zip(cost[i - 1][j - 1], (1, 1, 0, 0))),
                tuple(a + b for a, b in zip(cost[i - 1][j], (1, 0, 1, 0))),
                tuple(a + b for a, b in zip(cost[i][j - 1], (1, 0, 0, 1))),
            )
            cost[i][j] = min(candidates, key=lambda item: item[0])
    _, substitutions, deletions, insertions = cost[-1][-1]
    return substitutions, deletions, insertions


def word_error_rate(reference: str, hypothesis: str) -> dict[str, float | int]:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    substitutions, deletions, insertions = edit_counts(ref, hyp)
    errors = substitutions + deletions + insertions
    return {
        "wer": errors / len(ref) if ref else float(bool(hyp)),
        "reference_words": len(ref),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = list(" ".join(normalize(reference)))
    hyp = list(" ".join(normalize(hypothesis)))
    substitutions, deletions, insertions = edit_counts(ref, hyp)
    return (substitutions + deletions + insertions) / len(ref) if ref else float(bool(hyp))

