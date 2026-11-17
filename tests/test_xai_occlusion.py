"""Offline contracts for position-specific word deletion explanations."""

from copy import deepcopy

import pytest


def explain(*args, **kwargs):
    from lab10.occlusion import explain_words

    return explain_words(*args, **kwargs)


def test_deltas_keep_original_target_when_top_prediction_changes():
    probabilities = {
        "missing refund": [0.8, 0.2],
        " refund": [0.1, 0.9],
        "missing ": [0.9, 0.1],
    }
    categories = ["refund", "card"]
    before = deepcopy(probabilities)
    result = explain("missing refund", lambda texts: [probabilities[t] for t in texts], categories)
    assert result["target_label"] == "refund"
    assert result["original_score"] == 0.8
    first, second = result["tokens"]
    assert first == {"word": "missing", "start": 0, "end": 7, "score_after": 0.1,
                     "delta": pytest.approx(0.7), "prediction_after": "card", "top_score_after": 0.9}
    assert second["delta"] == pytest.approx(-0.1)
    assert result["n_variants"] == 3
    assert 0 <= result["scoring_seconds"] <= result["elapsed_seconds"]
    assert probabilities == before
    assert categories == ["refund", "card"]


def test_repeated_words_are_deleted_by_span_preserving_all_other_characters():
    observed = []

    def scorer(texts):
        observed.extend(texts)
        return [[0.6, 0.4] for _ in texts]

    result = explain("go,  go! can't A’2", scorer, ["a", "b"])
    assert observed == ["go,  go! can't A’2", ",  go! can't A’2", "go,  ! can't A’2",
                        "go,  go!  A’2", "go,  go! can't "]
    assert [(t["word"], t["start"], t["end"]) for t in result["tokens"]] == [
        ("go", 0, 2), ("go", 5, 7), ("can't", 9, 14), ("A’2", 15, 18)]


def test_original_is_scored_separately_then_variants_respect_batch_boundary():
    calls = []

    def scorer(texts):
        calls.append(list(texts))
        return [[0.5, 0.5] for _ in texts]

    result = explain("a b c d e", scorer, ["first", "second"], batch_size=2)
    assert calls == [["a b c d e"], [" b c d e", "a  c d e"],
                     ["a b  d e", "a b c  e"], ["a b c d "]]
    assert result["target_label"] == "first"
    assert result["n_variants"] == 6


@pytest.mark.parametrize("text,categories,batch_size", [
    ("", ["a"], 32), ("   ", ["a"], 32), ("!!!", ["a"], 32),
    ("é中", ["a"], 32), (None, ["a"], 32), ("a" * 2001, ["a"], 32),
    ("a", [], 32), ("a", ["x", "x"], 32), ("a", [""], 32),
    ("a", [None], 32), ("a", ["  "], 32), ("a", "a", 32),
    ("a", ["x"], 0), ("a", ["x"], -1), ("a", ["x"], True),
    ("a", ["x"], 1.5),
])
def test_invalid_arguments_fail_before_scoring(text, categories, batch_size):
    def scorer(texts):
        pytest.fail("invalid input reached scorer")

    with pytest.raises(ValueError):
        explain(text, scorer, categories, batch_size=batch_size)


@pytest.mark.parametrize("invalid", [
    None, [], [[0.5, 0.5], [0.5, 0.5]], [[1.0]], [[0.2, 0.2]],
    [[float("nan"), 0.5]], [[float("inf"), 0.0]], [[-0.1, 1.1]],
    [["0.5", 0.5]], [[True, False]], [None],
])
@pytest.mark.parametrize("invalid_original", [True, False])
def test_malformed_probabilities_fail_closed_in_original_and_variant_batches(invalid, invalid_original):
    calls = 0

    def scorer(texts):
        nonlocal calls
        calls += 1
        if invalid_original or calls > 1:
            return invalid
        return [[0.7, 0.3]]

    with pytest.raises(ValueError):
        explain("word", scorer, ["a", "b"])


def test_probability_sum_tolerance_accepts_rounding_without_recalibration():
    result = explain("word", lambda texts: [[0.60001, 0.4] for _ in texts], ["a", "b"])
    assert result["original_score"] == 0.60001
    assert result["tokens"][0]["score_after"] == 0.60001


def test_scorer_failures_propagate_without_partial_explanations():
    def scorer(texts):
        if texts == ["word"]:
            return [[0.7, 0.3]]
        raise RuntimeError("unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        explain("word", scorer, ["a", "b"])


def test_scoring_time_includes_original_and_each_variant_batch(monkeypatch):
    from lab10 import occlusion

    ticks = iter([0.0, 1.0, 3.0, 4.0, 7.0, 8.0, 12.0, 15.0])
    monkeypatch.setattr(occlusion, "perf_counter", lambda: next(ticks))
    result = explain("a b", lambda texts: [[1.0] for _ in texts], ["only"], batch_size=1)
    assert result["scoring_seconds"] == 9.0
    assert result["elapsed_seconds"] == 15.0
