"""Word deletion sensitivity for a scorer returning calibrated probabilities."""

from collections.abc import Callable
import math
from numbers import Real
import re
from time import perf_counter


_WORDS = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")


def explain_words(
    text: str,
    scorer: Callable[[list[str]], list[list[float]]],
    categories: list[str],
    batch_size: int = 32,
) -> dict:
    """Delete each word span and measure change for the original top class.

    Positive deltas indicate that deletion lowered that class's probability.
    Probabilities are validated but never recalibrated or renormalized. Ties
    select the first category. Malformed inputs or scorer outputs raise
    ``ValueError``; scorer exceptions propagate without a partial result.
    """
    started = perf_counter()
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("text must be a nonempty string of at most 2000 characters")
    if (not isinstance(categories, list) or not categories
            or any(not isinstance(label, str) or not label.strip() for label in categories)
            or len(set(categories)) != len(categories)):
        raise ValueError("categories must be a nonempty list of unique nonempty strings")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not callable(scorer):
        raise ValueError("scorer must be callable")
    words = list(_WORDS.finditer(text))
    if not words:
        raise ValueError("text must contain at least one word")
    labels = list(categories)
    scoring_seconds = 0.0

    def score(texts: list[str]) -> list[list[float]]:
        nonlocal scoring_seconds
        scoring_started = perf_counter()
        rows = scorer(texts)
        scoring_seconds += perf_counter() - scoring_started
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ValueError("scorer must return one probability row per text")
        validated = []
        for row in rows:
            if not isinstance(row, list) or len(row) != len(labels):
                raise ValueError("scorer probability rows must match categories")
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(value) or not 0 <= value <= 1 for value in row):
                raise ValueError("scorer probabilities must be finite numbers between zero and one")
            values = [float(value) for value in row]
            if abs(math.fsum(values) - 1.0) > 1e-4:
                raise ValueError("scorer probability rows must sum to one")
            validated.append(values)
        return validated

    original = score([text])[0]
    target = max(range(len(labels)), key=original.__getitem__)
    tokens = []
    for offset in range(0, len(words), batch_size):
        batch = words[offset:offset + batch_size]
        variants = [text[:word.start()] + text[word.end():] for word in batch]
        for word, row in zip(batch, score(variants)):
            prediction = max(range(len(labels)), key=row.__getitem__)
            tokens.append({
                "word": word.group(), "start": word.start(), "end": word.end(),
                "score_after": row[target], "delta": original[target] - row[target],
                "prediction_after": labels[prediction], "top_score_after": row[prediction],
            })
    return {
        "target_label": labels[target], "original_score": original[target],
        "tokens": tokens, "n_variants": len(words) + 1,
        "scoring_seconds": scoring_seconds, "elapsed_seconds": perf_counter() - started,
    }
