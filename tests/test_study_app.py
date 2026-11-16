"""Study app: timed manual/assisted trials, per-trial journal, CSV export (TEST-ONLY fake model)."""

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_inputs(tmp_path):
    data = tmp_path / "validation_model.csv"
    with data.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "text", "category", "group_id", "split"])
        writer.writerow(["v:1", "SECRET where is my card", "card_arrival", "g1", "validation_model"])
        writer.writerow(["v:2", "SECRET card stolen", "lost_or_stolen_card", "g2", "validation_model"])
        writer.writerow(["v:3", "SECRET top up failed", "top_up_failed", "g3", "validation_model"])
        writer.writerow(["v:4", "SECRET transfer timing", "transfer_timing", "g4", "validation_model"])
    import hashlib

    plan = {
        "phase": "piloto", "seed": 71, "source_file": data.name,
        "source_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "n_messages": 4,
        "per_participant": 4,
        "participants": [{"slot": "S1", "first_condition": "manual", "order": [
            {"position": 1, "message_id": "v:1", "condition": "manual"},
            {"position": 2, "message_id": "v:2", "condition": "asistida"},
            {"position": 3, "message_id": "v:3", "condition": "asistida"},
            {"position": 4, "message_id": "v:4", "condition": "manual"},
        ]}],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    return data, plan_path


def study_script():
    """DOBLE DE PRUEBA: predictor falso; nunca usar en un estudio real."""
    import json as _json
    import os
    from pathlib import Path as _Path

    import streamlit as st

    from reclamoruta import app as _app
    from reclamoruta import contracts as _c
    from reclamoruta import study_app as _sa

    root = _Path(_app.__file__).resolve().parents[2]
    catalog = list(_json.loads((root / "reports" / "BANKING77_LABEL_TO_ID.json").read_text(encoding="utf-8")))
    routes = _c.load_routes(root / "configs" / "routes.json", catalog)

    class FakePredictor:
        model_version = "fake-v1"

        def predict(self, texts):
            st.session_state["fake_calls"] = st.session_state.get("fake_calls", 0) + 1
            return [_c.Prediction("card_arrival", 0.9, [("card_arrival", 0.9)], "fake-v1", 1.0) for _ in texts]

    decision = _c.BoundDecision("fake-v1", "a" * 64, 0.6, {"method": "none", "fitted": False}, "s-v1")
    runtime = _app.Runtime(FakePredictor(), decision, routes, test_double=True)
    base = _Path(os.environ["STUDY_TEST_DIR"])
    study = _sa.load_study(base / "plan.json", base / "validation_model.csv")

    ticks = st.session_state.setdefault("fake_clock", [0.0])

    def clock():
        ticks[0] += 2.5
        return ticks[0]

    _sa.render_study(runtime, study, base / "results", clock=clock)


@pytest.fixture
def at(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    write_inputs(tmp_path)
    monkeypatch.setenv("STUDY_TEST_DIR", str(tmp_path))
    test = AppTest.from_function(study_script, default_timeout=10)
    test.run()
    assert not test.exception
    return test


def start(at, skip_practice=True):
    at.selectbox(key="study_slot").select("S1")
    at.checkbox(key="study_consent").check()
    at.button(key="study_start").click()
    at.run()
    assert not at.exception
    if skip_practice:
        for i in range(2):
            at.selectbox(key=f"practice_choice_{i}").select("card_arrival")
            at.button(key=f"practice_confirm_{i}").click()
            at.run()
            assert not at.exception
        at.button(key="practice_done").click()
        at.run()
        assert not at.exception
    return at


def answer(at, position, intent):
    at.selectbox(key=f"study_choice_{position}").select(intent)
    at.button(key=f"study_confirm_{position}").click()
    at.run()
    assert not at.exception


def continue_block(at):
    at.button(key="study_continue").click()
    at.run()
    assert not at.exception


def texts(at):
    return "\n".join(str(getattr(e, "value", "")) for kind in ("markdown", "info", "warning", "success", "error", "caption", "subheader")
                     for e in getattr(at, kind))


def test_cannot_start_without_consent(at):
    at.selectbox(key="study_slot").select("S1")
    at.button(key="study_start").click()
    at.run()
    assert any("consentimiento" in w.value for w in at.warning)
    assert "SECRET" not in texts(at)


def test_manual_trial_shows_text_without_suggestion(at):
    start(at)
    assert "SECRET where is my card" in texts(at)
    assert at.session_state["fake_calls"] == 1  # only the assisted practice message
    assert "Sugerencia" not in texts(at)


def test_manual_trial_requires_a_choice(at):
    start(at)
    at.button(key="study_confirm_1").click()
    at.run()
    assert any("Elija" in w.value for w in at.warning)


def test_full_session_journals_without_text_and_exports(at, tmp_path):
    start(at)
    answer(at, 1, "card_arrival")
    assert "Pausa" in texts(at)                # condition changes: untimed pause
    continue_block(at)
    assert "Sugerencia" in texts(at) and at.session_state["fake_calls"] == 2  # practice + trial
    answer(at, 2, "lost_or_stolen_card")       # corrects the suggestion
    answer(at, 3, "card_arrival")              # accepts it
    continue_block(at)
    answer(at, 4, "transfer_timing")
    assert "Gracias" in texts(at)

    results = tmp_path / "results"
    journal = list(results.glob("piloto_S1_P-*.jsonl"))
    exported = list(results.glob("piloto_S1_P-*.csv"))
    assert len(journal) == 1 and len(exported) == 1
    assert "SECRET" not in journal[0].read_text(encoding="utf-8")
    assert "SECRET" not in exported[0].read_text(encoding="utf-8")
    rows = list(csv.DictReader(exported[0].open(encoding="utf-8", newline="")))
    assert [r["message_id"] for r in rows] == ["v:1", "v:2", "v:3", "v:4"]
    assert [r["condition"] for r in rows] == ["manual", "asistida", "asistida", "manual"]
    assert [r["corrected"] for r in rows] == ["False", "True", "False", "False"]
    assert all(float(r["elapsed_s"]) > 0 for r in rows)
    assert len(journal[0].read_text(encoding="utf-8").splitlines()) == 4


def test_slot_already_used_is_refused(at, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "piloto_S1_P-0000abcd.jsonl").write_text("", encoding="utf-8")
    at.selectbox(key="study_slot").select("S1")
    at.checkbox(key="study_consent").check()
    at.button(key="study_start").click()
    at.run()
    assert any("ya tiene resultados" in e.value for e in at.error)


def test_load_study_rejects_modified_source(tmp_path):
    from reclamoruta import study_app

    data, plan = write_inputs(tmp_path)
    data.write_text(data.read_text(encoding="utf-8") + "v:5,extra,card_arrival,g5,validation_model\n",
                    encoding="utf-8")
    with pytest.raises(study_app.StudyPlanError, match="hash"):
        study_app.load_study(plan, data)


def test_load_study_checks_frozen_decision(tmp_path):
    from reclamoruta import study_app

    data, plan_path = write_inputs(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["decision_sha256"] = "b" * 64
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    decision = tmp_path / "decision.json"
    decision.write_text("{}", encoding="utf-8")
    with pytest.raises(study_app.StudyPlanError, match="decisión"):
        study_app.load_study(plan_path, data, decision_path=decision)


def test_untimed_practice_precedes_measurement_and_is_not_recorded(at, tmp_path):
    start(at, skip_practice=False)
    text = texts(at)
    assert "Práctica" in text and "no se registra" in text
    assert "SECRET" not in text                       # practice never uses study messages
    at.selectbox(key="practice_choice_0").select("card_arrival")
    at.button(key="practice_confirm_0").click()
    at.run()
    assert "Sugerencia" in texts(at)                  # second practice is assisted
    assert at.session_state["fake_calls"] == 1
    at.selectbox(key="practice_choice_1").select("card_arrival")
    at.button(key="practice_confirm_1").click()
    at.run()
    journal = list((tmp_path / "results").glob("*.jsonl"))
    assert journal == [] or journal[0].read_text(encoding="utf-8") == ""
    assert at.session_state["study_session"].trials == []
    at.button(key="practice_done").click()
    at.run()
    assert "SECRET where is my card" in texts(at)     # measurement starts after practice
