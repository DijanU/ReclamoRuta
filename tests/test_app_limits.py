"""Per-session usage limit for the public app (TEST-ONLY fake predictor)."""

import pytest


def limited_script():
    """DOBLE DE PRUEBA: predictor falso."""
    import json as _json
    from pathlib import Path as _Path

    import streamlit as st

    from reclamoruta import app as _app
    from reclamoruta import contracts as _c

    root = _Path(_app.__file__).resolve().parents[2]
    catalog = list(_json.loads((root / "reports" / "BANKING77_LABEL_TO_ID.json").read_text(encoding="utf-8")))
    routes = _c.load_routes(root / "configs" / "routes.json", catalog)

    class FakePredictor:
        model_version = "fake-v1"

        def predict(self, texts):
            st.session_state["fake_calls"] = st.session_state.get("fake_calls", 0) + 1
            return [_c.Prediction("card_arrival", 0.9, [("card_arrival", 0.9)], "fake-v1", 1.0)]

    now = st.session_state.setdefault("fake_now", [1000.0])
    decision = _c.BoundDecision("fake-v1", "a" * 64, 0.6, {"method": "none", "fitted": False}, "s-v1")
    _app.render(_app.Runtime(FakePredictor(), decision, routes, test_double=True),
                max_per_minute=2, clock=lambda: now[0])


@pytest.fixture
def at():
    from streamlit.testing.v1 import AppTest

    test = AppTest.from_function(limited_script, default_timeout=10)
    test.run()
    return test


def classify(at, text="Where is my card?"):
    at.text_area(key="message").input(text)
    at.button(key="classify").click()
    at.run()
    assert not at.exception


def test_requests_over_the_limit_are_refused_without_calling_model(at):
    classify(at)
    classify(at, "card still missing")
    classify(at, "third one")
    assert at.session_state["fake_calls"] == 2
    assert any("Límite de uso" in w.value for w in at.warning)


def test_limit_window_slides_after_a_minute(at):
    classify(at)
    classify(at, "second")
    at.session_state["fake_now"] = [1061.0]
    at.run()
    classify(at, "third after a minute")
    assert at.session_state["fake_calls"] == 3


def test_invalid_input_does_not_consume_quota(at):
    classify(at, "   ")
    classify(at)
    classify(at, "second valid")
    assert at.session_state["fake_calls"] == 2
    assert not any("Límite de uso" in w.value for w in at.warning)


def test_limit_setting_from_environment(monkeypatch):
    from reclamoruta import app

    monkeypatch.setenv("RECLAMORUTA_MAX_PER_MINUTE", "7")
    assert app._max_per_minute() == 7
    monkeypatch.setenv("RECLAMORUTA_MAX_PER_MINUTE", "nope")
    with pytest.raises(app.AppConfigError):
        app._max_per_minute()
    monkeypatch.delenv("RECLAMORUTA_MAX_PER_MINUTE")
    assert app._max_per_minute() == 20
