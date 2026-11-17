"""Selection, aggregation and provenance checks without weights or network."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from lab10.run import (
    aggregate_words, load_verified_test, run, select_local_cases,
    stratified_sample, validate_config, verify_frozen_inputs,
)


CONFIG = {"schema_version": 1, "seed": 42, "global_per_class": 2,
          "global_min_word_occurrences": 3, "batch_size": 32, "threads": 2,
          "high_confidence_min": .95, "near_threshold_window": .05}


def record(i, score, correct=True, text="My card has arrived", wrong_route=False):
    return {"id": str(i), "group_id": f"g{i}", "score": score,
            "correct": correct, "text": text, "true_route": "cards",
            "predicted_route": "security" if wrong_route else "cards"}


def test_local_cases_are_distinct_and_straddle_frozen_threshold():
    rows = [record(0, .99), record(1, .97, False, wrong_route=True),
            record(2, .8499), record(3, .8501), record(4, .6, text="Not delivered")]
    result = select_local_cases(rows, .85, CONFIG)
    assert len({r["group_id"] for r in result}) == 5
    assert result[0]["correct"] and result[0]["score"] >= .95
    assert result[1]["predicted_route"] != result[1]["true_route"]
    assert result[2]["score"] < .85 <= result[3]["score"]
    assert result[4]["text"] == "Not delivered"
    assert all("reason" not in r for r in rows)
    assert result == select_local_cases(list(reversed(rows)), .85, CONFIG)


def test_missing_diagnostic_role_fails_instead_of_inventing_a_case():
    with pytest.raises(ValueError, match="No existe un caso"):
        select_local_cases([record(0, .99)], .85, CONFIG)


def test_sample_is_balanced_repeatable_and_independent_of_input_order():
    rows = [{"id": f"{c}{i}", "category": c} for c in ["a", "b", "c"] for i in range(8)]
    first = stratified_sample(rows, 2, 42)
    assert len(first) == 6
    assert {c: sum(r["category"] == c for r in first) for c in ["a", "b", "c"]} == {"a": 2, "b": 2, "c": 2}
    assert first == stratified_sample(list(reversed(rows)), 2, 42)
    assert first != stratified_sample(rows, 2, 43)
    with pytest.raises(ValueError, match="demasiado grande"):
        stratified_sample(rows, 9, 42)


def test_global_magnitude_does_not_cancel_signed_effects_or_count_documents():
    explanations = [{"tokens": [{"word": "Card", "delta": .6}, {"word": "card", "delta": -.4}]},
                    {"tokens": [{"word": "card", "delta": .2}, {"word": "once", "delta": 1}]}]
    words = aggregate_words(explanations, 3)
    assert len(words) == 1
    assert words[0]["word"] == "card" and words[0]["count"] == 3
    assert words[0]["mean_abs_delta"] == pytest.approx(.4)
    assert words[0]["mean_signed_delta"] == pytest.approx(.4 / 3)


def make_verified_data(tmp_path):
    repo = tmp_path / "repo"
    data = repo / "data"
    data.mkdir(parents=True)
    (repo / "reports").mkdir()
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["id", "text", "category", "group_id", "split"])
    writer.writeheader()
    writer.writerow(dict(id="test.csv:0", text="Where is my card?", category="arrival", group_id="abc", split="test_unique"))
    (data / "test_unique.csv").write_bytes(stream.getvalue().encode("utf-8"))
    (data / "label_to_id.json").write_text('{"arrival": 0}', encoding="utf-8")
    manifest = {"output_sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                                   for name in ("test_unique.csv", "label_to_id.json")},
                "source": {"revision": "fixed"}, "counts": {"test_unique": 1}, "policy": {"version": 1}}
    encoded = json.dumps(manifest)
    (data / "manifest.json").write_text(encoded, encoding="utf-8")
    (repo / "reports/BANKING77_MANIFEST.json").write_text(encoded, encoding="utf-8")
    return repo, data


def test_test_bytes_are_verified_against_published_manifest(tmp_path):
    repo, data = make_verified_data(tmp_path)
    rows, hashes = load_verified_test(repo, data)
    assert rows[0]["category"] == "arrival"
    assert hashes["test_unique.csv"] == hashlib.sha256((data / "test_unique.csv").read_bytes()).hexdigest()
    with (data / "test_unique.csv").open("ab") as output:
        output.write(b"changed")
    with pytest.raises(ValueError, match="Hash"):
        load_verified_test(repo, data)


def test_altering_local_manifest_cannot_authorize_different_data(tmp_path):
    repo, data = make_verified_data(tmp_path)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    manifest["output_sha256"]["test_unique.csv"] = "0" * 64
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifiesto"):
        load_verified_test(repo, data)


@pytest.mark.parametrize("key,value", [("threads", 0), ("batch_size", True),
                                      ("global_per_class", -1), ("seed", "42"),
                                      ("near_threshold_window", float("nan")),
                                      ("high_confidence_min", 1.1)])
def test_invalid_config_rejected(key, value):
    with pytest.raises(ValueError):
        validate_config({**CONFIG, key: value})


def test_existing_output_rejected_before_importing_model_or_reading_data(tmp_path):
    with pytest.raises(FileExistsError):
        run(Path("missing-repo"), Path("missing-data"), tmp_path, Path("missing-config"))


def test_policy_changed_before_start_cannot_pass_as_frozen(tmp_path):
    paths = ["artifacts/distilbert-v2/manifest.json", "reports/evaluation_scenario_v1/distilbert-v2/decision.json",
             "reports/evaluation_scenario_v1/distilbert-v2/calibration.json", "configs/routes.json", "src/reclamoruta/predictor.py"]
    hashes = []
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("ascii"))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    freeze = {"models": {"distilbert-v2": {"manifest_sha256": hashes[0], "decision_sha256": hashes[1],
                                            "calibration_sha256": hashes[2]}},
              "routes_sha256": hashes[3], "source": {"python_sha256": {"predictor.py": hashes[4]}}}
    (tmp_path / "reports/evaluation_scenario_v1/freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    verify_frozen_inputs(tmp_path)
    (tmp_path / "configs/routes.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="congelamiento"):
        verify_frozen_inputs(tmp_path)
