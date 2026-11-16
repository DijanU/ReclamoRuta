"""Public deployment entry point with the real V2 artifact (skipped without LFS weights)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "artifacts" / "distilbert-v2" / "model.safetensors"


@pytest.mark.skipif(not WEIGHTS.exists() or WEIGHTS.stat().st_size < 1_000_000,
                    reason="requiere los pesos reales de Git LFS")
def test_entry_point_serves_frozen_v2_from_any_working_directory(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    for name in ("RECLAMORUTA_ARTIFACT_DIR", "RECLAMORUTA_DECISION", "RECLAMORUTA_CALIBRATION"):
        monkeypatch.delenv(name, raising=False)
    at = AppTest.from_file(str(ROOT / "deploy" / "streamlit_app.py"), default_timeout=180)
    at.run()
    assert not at.exception and not at.error
    at.text_area(key="message").input("My new card still hasn't arrived, it's been two weeks.")
    at.button(key="classify").click()
    at.run()
    assert not at.exception
    intents = [m.value for m in at.metric if m.label == "Intención"]
    assert intents == ["card_arrival"]
    assert any("umbral 0.85" in c.value for c in at.caption)


def test_deploy_requirements_pin_cpu_torch():
    text = (ROOT / "deploy" / "requirements.txt").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" in text
    assert "+cpu" in text
