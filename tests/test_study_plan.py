"""Study sample plans (pilot on validation, final on test_unique) and result analysis."""

import csv
import hashlib
import json
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [f"intent_{i:02d}" for i in range(77)]


def sp():
    from reclamoruta import study_plan as module

    return module


def write_split(path, split, n_per_class=3, classes=CLASSES):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "text", "category", "group_id", "split"])
        row = 0
        for c in classes:
            for k in range(n_per_class):
                words = " ".join(["word"] * (3 + (row * 7) % 25))
                writer.writerow([f"{split}.csv:{row}", f"SECRET {words}", c, f"g{row}", split])
                row += 1
    return path


@pytest.fixture
def test_unique(tmp_path):
    return write_split(tmp_path / "test_unique.csv", "test_unique")


@pytest.fixture
def decision(tmp_path):
    path = tmp_path / "decision.json"
    path.write_text('{"status": "validated"}', encoding="utf-8")
    return path


def final(path, decision, **kwargs):
    return sp().build_plan(path, phase="final", decision_path=decision, **kwargs)


@pytest.fixture
def validation(tmp_path):
    return write_split(tmp_path / "validation_model.csv", "validation_model")


def word_counts(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return {r["id"]: len(r["text"].split()) for r in csv.DictReader(stream)}


def test_final_plan_follows_protocol(test_unique, decision):
    plan = final(test_unique, decision)
    assert plan["phase"] == "final" and plan["seed"] == 73
    assert plan["source_sha256"] == hashlib.sha256(test_unique.read_bytes()).hexdigest()
    slots = plan["participants"]
    assert [s["slot"] for s in slots] == ["S1", "S2", "S3"]
    all_ids = [item["message_id"] for s in slots for item in s["order"]]
    assert len(all_ids) == 120 and len(set(all_ids)) == 120  # no message repeated anywhere
    for slot in slots:
        conditions = [item["condition"] for item in slot["order"]]
        assert len(conditions) == 40
        assert conditions.count("manual") == 20 and conditions.count("asistida") == 20
        assert [item["position"] for item in slot["order"]] == list(range(1, 41))


def test_condition_order_is_counterbalanced_abba(test_unique, decision):
    slots = final(test_unique, decision)["participants"]

    def blocks(slot):
        seq = [item["condition"] for item in slot["order"]]
        return [seq[i] for i in range(0, 40, 10)], all(len(set(seq[i:i + 10])) == 1 for i in range(0, 40, 10))

    (b1, ok1), (b2, ok2), (b3, ok3) = (blocks(s) for s in slots)
    assert ok1 and ok2 and ok3
    assert b1 == ["manual", "asistida", "asistida", "manual"]
    assert b2 == ["asistida", "manual", "manual", "asistida"]
    assert b3 == b1


def test_plan_balances_length_and_classes_between_conditions(test_unique, decision):
    plan = final(test_unique, decision)
    lengths = word_counts(test_unique)
    by_condition = {"manual": [], "asistida": []}
    for slot in plan["participants"]:
        for item in slot["order"]:
            by_condition[item["condition"]].append(lengths[item["message_id"]])
    assert abs(statistics.fmean(by_condition["manual"]) - statistics.fmean(by_condition["asistida"])) < 1.0
    with test_unique.open(encoding="utf-8", newline="") as stream:
        category = {r["id"]: r["category"] for r in csv.DictReader(stream)}
    chosen = [category[i["message_id"]] for s in plan["participants"] for i in s["order"]]
    assert len(set(chosen)) == 77  # every intention appears before any repeats


def test_plan_is_deterministic_and_seed_sensitive(test_unique, decision):
    first = final(test_unique, decision)
    assert first == final(test_unique, decision)
    assert first != final(test_unique, decision, seed=74)


def test_plan_contains_no_message_text_or_labels(test_unique, decision):
    dumped = json.dumps(final(test_unique, decision))
    assert "SECRET" not in dumped
    assert "intent_" not in dumped


def test_pilot_uses_validation_and_final_uses_only_test_unique(test_unique, validation, tmp_path):
    pilot = sp().build_plan(validation, phase="piloto")
    assert pilot["seed"] == 71
    assert sum(len(s["order"]) for s in pilot["participants"]) == 30
    with pytest.raises(sp().StudyPlanError, match="test_unique"):
        sp().build_plan(validation, phase="final", decision_path=validation)
    with pytest.raises(sp().StudyPlanError, match="validación"):
        sp().build_plan(test_unique, phase="piloto")
    official = write_split(tmp_path / "test_official.csv", "test_official")
    with pytest.raises(sp().StudyPlanError):
        sp().build_plan(official, phase="final", decision_path=official)


def test_not_enough_messages_is_rejected(tmp_path):
    small = write_split(tmp_path / "test_unique.csv", "test_unique", n_per_class=1, classes=CLASSES[:50])
    with pytest.raises(sp().StudyPlanError, match="suficientes"):
        sp().build_plan(small, phase="final", decision_path=small)


def test_final_plan_requires_frozen_decision(test_unique):
    with pytest.raises(sp().StudyPlanError, match="decisión"):
        sp().build_plan(test_unique, phase="final")


def test_final_plan_records_frozen_decision_and_write_never_overwrites(test_unique, decision, tmp_path):
    plan = sp().build_plan(test_unique, phase="final", decision_path=decision)
    assert plan["decision_sha256"] == hashlib.sha256(decision.read_bytes()).hexdigest()
    out = tmp_path / "plans" / "final.json"
    sp().write_plan(plan, out)
    assert json.loads(out.read_text(encoding="utf-8")) == plan
    with pytest.raises(sp().StudyPlanError):
        sp().write_plan(plan, out)


# --- analysis -------------------------------------------------------------------

ROUTES = {"a": "r1", "b": "r1", "c": "r2"}


def trials():
    from reclamoruta.study import Trial

    def t(mid, cond, sugg, final, secs, review=False):
        return Trial("P-0000abcd", mid, cond, secs, sugg, ROUTES.get(sugg) if sugg else None,
                     final, ROUTES[final], sugg is not None and final != sugg,
                     review if cond == "asistida" else None, "v2" if sugg else None)

    return [
        t("m1", "manual", None, "a", 10.0), t("m2", "manual", None, "c", 14.0),   # m2 human error
        t("m3", "asistida", "a", "a", 4.0),                                      # correct accepted
        t("m4", "asistida", "b", "c", 9.0),                                      # wrong, detected, fixed
        t("m5", "asistida", "b", "b", 5.0),                                      # wrong, undetected
        t("m6", "asistida", "c", "a", 8.0, review=True),                         # right sugg, human broke it
    ]


GOLD = {"m1": "a", "m2": "b", "m3": "a", "m4": "c", "m5": "c", "m6": "c"}


def test_analysis_reports_detection_and_human_accuracy():
    result = sp().analyze(trials(), GOLD, ROUTES)
    assert result["manual"]["human_accuracy"] == pytest.approx(0.5)
    assisted = result["asistida"]
    assert assisted["wrong_suggestions"] == 2
    assert assisted["detected_wrong_suggestions"] == 1
    assert result["value_inputs"]["q_detectado"] == pytest.approx(0.5)
    assert assisted["human_introduced_errors"] == 1
    assert assisted["human_accuracy"] == pytest.approx(0.5)
    assert result["value_inputs"]["t_manual"] == pytest.approx(12.0)
    assert result["estimated"] is True


def test_analysis_without_wrong_suggestions_leaves_q_undefined():
    result = sp().analyze(trials()[:3], GOLD, ROUTES)
    assert result["value_inputs"]["q_detectado"] is None


def test_analysis_rejects_unknown_message_ids():
    with pytest.raises(sp().StudyPlanError):
        sp().analyze(trials(), {"m1": "a"}, ROUTES)


def test_exported_trials_round_trip(tmp_path):
    from reclamoruta.study import export_trials

    out = export_trials(trials(), tmp_path / "res.csv")
    assert sp().load_trials(out) == trials()
