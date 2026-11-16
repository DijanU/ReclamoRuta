"""Streamlit app for the human timing study (manual vs. assisted routing).

Run one session per participant, operated by a team member::

    RECLAMORUTA_ARTIFACT_DIR=artifacts/distilbert-v2 \
    RECLAMORUTA_DECISION=reports/evaluation_scenario_v1/distilbert-v2/decision.json \
    RECLAMORUTA_CALIBRATION=reports/evaluation_scenario_v1/distilbert-v2/calibration.json \
    RECLAMORUTA_STUDY_PLAN=data/study/plan_piloto.json \
    RECLAMORUTA_STUDY_DATA=data/processed/banking77/validation_model.csv \
    HF_HUB_OFFLINE=1 streamlit run src/reclamoruta/study_app.py

``RECLAMORUTA_STUDY_RESULTS`` (default ``data/study/results``) receives, per
session, an append-only ``.jsonl`` journal written after every trial and a
final ``.csv``. Neither contains message text or participant names. The timer
starts when a message is shown and stops at confirmation; pauses between
condition blocks are not timed. Nothing here retrains or changes thresholds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

from reclamoruta.app import AppConfigError, Runtime, _settings, load_runtime
from reclamoruta.contracts import ContractError, route_message
from reclamoruta.study import StudySession, export_trials, new_participant_id
from reclamoruta.study_plan import StudyPlanError, read_rows

# Synthetic, written by the team (not from BANKING77): untimed and never recorded.
PRACTICE_MESSAGES = (
    "I ordered a new card two weeks ago and it still hasn't arrived.",
    "Why was I charged a fee when I took cash out at the ATM?",
)

CONDITION_HELP = {
    "manual": "Lea la consulta y elija la intención correcta. No hay sugerencia del modelo.",
    "asistida": "Lea la consulta y la sugerencia del modelo. Confírmela o corríjala.",
}


@dataclass(frozen=True)
class Study:
    phase: str
    slots: dict[str, list[tuple[str, str]]]
    texts: dict[str, str]
    plan_sha256: str


def load_study(plan_path: Path, data_path: Path, decision_path: Path | None = None) -> Study:
    """Load a plan and its message texts, checking source and frozen-decision hashes."""
    raw = Path(plan_path).read_bytes()
    plan = json.loads(raw)
    if hashlib.sha256(Path(data_path).read_bytes()).hexdigest() != plan.get("source_sha256"):
        raise StudyPlanError("El archivo de mensajes no coincide con el hash registrado en el plan")
    expected = plan.get("decision_sha256")
    if expected is not None:
        if decision_path is None or hashlib.sha256(Path(decision_path).read_bytes()).hexdigest() != expected:
            raise StudyPlanError("La decisión cargada no es la decisión congelada registrada en el plan")
    texts = {row["id"]: row["text"] for row in read_rows(data_path)}
    slots = {}
    for slot in plan["participants"]:
        order = [(item["message_id"], item["condition"]) for item in slot["order"]]
        if any(mid not in texts for mid, _ in order):
            raise StudyPlanError(f"El plan del puesto {slot['slot']} referencia mensajes ausentes")
        slots[slot["slot"]] = order
    return Study(plan["phase"], slots, texts, hashlib.sha256(raw).hexdigest())


def _journal(path: Path, trial) -> None:
    record = {k: v for k, v in asdict(trial).items() if k != "text"}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _setup(study: Study, results_dir: Path, runtime: Runtime, clock: Callable[[], float]) -> None:
    import streamlit as st

    st.subheader(f"Estudio de tiempos — fase {study.phase}")
    st.markdown(
        "- Participación **voluntaria**; puede detenerse cuando quiera.\n"
        "- Solo se registran un identificador anónimo, el mensaje mostrado (por su ID), "
        "la condición, el tiempo y la intención elegida. **No** se guardan nombres ni el texto.\n"
        "- Los mensajes son consultas públicas de BANKING77 en inglés, no datos de clientes reales.\n"
        "- Primero hay **dos mensajes de práctica** sin cronómetro.\n"
        "- Habrá bloques **manuales** y **asistidos**; entre bloques hay una pausa sin cronómetro."
    )
    slot = st.selectbox("Puesto del plan (lo asigna el equipo)", list(study.slots), index=None,
                        key="study_slot", placeholder="Elija el puesto")
    consent = st.checkbox("La persona participante dio su consentimiento verbal", key="study_consent")
    if not st.button("Comenzar", key="study_start"):
        return
    if slot is None:
        st.warning("Elija el puesto del plan.")
        return
    if not consent:
        st.warning("Se requiere el consentimiento antes de comenzar.")
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    if any(results_dir.glob(f"{study.phase}_{slot}_*")):
        st.error(f"El puesto {slot} ya tiene resultados en esta fase; use otro puesto.")
        return
    pid = new_participant_id()
    st.session_state.update({
        "study_stage": "practice", "practice_index": 0, "study_index": 0, "study_slot_id": slot,
        "study_session": StudySession(participant_id=pid, intent_to_route=runtime.routes.intent_to_route,
                                      clock=clock),
        "study_journal": results_dir / f"{study.phase}_{slot}_{pid}.jsonl",
        "study_current": None,
    })
    st.rerun()


def _trial(study: Study, runtime: Runtime, results_dir: Path) -> None:
    import streamlit as st

    slot = st.session_state["study_slot_id"]
    order = study.slots[slot]
    index = st.session_state["study_index"]
    session: StudySession = st.session_state["study_session"]
    message_id, condition = order[index]
    position = index + 1

    current = st.session_state.get("study_current")
    if current is None or current["position"] != position:
        result = None
        text = study.texts[message_id]
        if condition == "asistida":
            try:
                result = route_message(text, runtime.predictor, runtime.decision, runtime.routes)
            except Exception:  # noqa: BLE001 - stop rather than invent a suggestion
                st.error("No se pudo obtener la sugerencia del modelo. Detenga la sesión y avise al equipo.")
                st.stop()
        session.start(
            message_id, condition,
            suggested_intent=result.intent if result else None,
            needs_review=result.needs_review if result else None,
            model_version=result.prediction.model_version if result else None,
        )
        current = {"position": position, "result": result}
        st.session_state["study_current"] = current

    st.progress(index / len(order), text=f"Mensaje {position} de {len(order)} · condición {condition}")
    st.caption(CONDITION_HELP[condition])
    st.info(study.texts[message_id])
    intents = runtime.routes.intents
    result = current["result"]
    if result is not None:
        st.markdown(f"**Sugerencia del modelo:** `{result.intent}` → {result.route_name} "
                    f"(puntuación {result.score:.2f})")
        if result.needs_review:
            st.warning("Revisión requerida: la puntuación está por debajo del umbral.")
        default = intents.index(result.intent)
    else:
        default = None
    choice = st.selectbox("Intención final", intents, index=default, key=f"study_choice_{position}",
                          placeholder="Elija la intención")
    if choice is not None:
        route = runtime.routes.intent_to_route[choice]
        st.caption(f"Ruta: {runtime.routes.route_names[route]}")
    if not st.button("Confirmar", key=f"study_confirm_{position}"):
        return
    if choice is None:
        st.warning("Elija una intención antes de confirmar.")
        return
    trial = session.finish(choice)
    _journal(st.session_state["study_journal"], trial)
    st.session_state["study_index"] = index + 1
    if index + 1 == len(order):
        journal: Path = st.session_state["study_journal"]
        export_trials(session.trials, journal.with_suffix(".csv"))
        st.session_state["study_stage"] = "done"
    elif order[index + 1][1] != condition:
        st.session_state["study_stage"] = "pause"
    st.rerun()


def _practice(study: Study, runtime: Runtime) -> None:
    """Two untimed warm-up messages, one per condition, in the slot's order."""
    import streamlit as st

    slot = st.session_state["study_slot_id"]
    first = study.slots[slot][0][1]
    conditions = [first, "asistida" if first == "manual" else "manual"]
    index = st.session_state["practice_index"]
    if index >= len(PRACTICE_MESSAGES):
        st.success("Práctica terminada. Desde ahora se mide el tiempo de cada mensaje.")
        if st.button("Empezar la prueba", key="practice_done"):
            st.session_state["study_stage"] = "running"
            st.rerun()
        return
    condition = conditions[index]
    text = PRACTICE_MESSAGES[index]
    st.subheader(f"Práctica {index + 1} de {len(PRACTICE_MESSAGES)} · condición {condition}")
    st.caption("Mensaje de práctica: no se cronometra y no se registra. " + CONDITION_HELP[condition])
    st.info(text)
    intents = runtime.routes.intents
    default = None
    if condition == "asistida":
        cached = st.session_state.get("practice_result")
        if cached is None or cached[0] != index:
            try:
                cached = (index, route_message(text, runtime.predictor, runtime.decision, runtime.routes))
            except Exception:  # noqa: BLE001
                st.error("No se pudo obtener la sugerencia del modelo. Avise al equipo.")
                st.stop()
            st.session_state["practice_result"] = cached
        result = cached[1]
        st.markdown(f"**Sugerencia del modelo:** `{result.intent}` → {result.route_name} "
                    f"(puntuación {result.score:.2f})")
        default = intents.index(result.intent)
    choice = st.selectbox("Intención final", intents, index=default, key=f"practice_choice_{index}",
                          placeholder="Elija la intención")
    if st.button("Confirmar", key=f"practice_confirm_{index}"):
        if choice is None:
            st.warning("Elija una intención antes de confirmar.")
            return
        st.session_state["practice_index"] = index + 1
        st.rerun()


def render_study(runtime: Runtime, study: Study, results_dir: Path,
                 clock: Callable[[], float] = time.perf_counter) -> None:
    import streamlit as st

    st.title("ReclamoRuta — estudio humano")
    if runtime.test_double:
        st.warning("DOBLE DE PRUEBA: predictor falso. No usar para un estudio real.")
    stage = st.session_state.get("study_stage", "setup")
    if stage == "setup":
        _setup(study, Path(results_dir), runtime, clock)
    elif stage == "practice":
        _practice(study, runtime)
    elif stage == "running":
        _trial(study, runtime, Path(results_dir))
    elif stage == "pause":
        slot = st.session_state["study_slot_id"]
        upcoming = study.slots[slot][st.session_state["study_index"]][1]
        st.success(f"Pausa: bloque terminado. Siguiente condición: **{upcoming}**.")
        st.caption(CONDITION_HELP[upcoming])
        if st.button("Continuar", key="study_continue"):
            st.session_state["study_stage"] = "running"
            st.rerun()
    else:
        session: StudySession = st.session_state["study_session"]
        st.success(f"Gracias. Sesión terminada: {len(session.trials)} mensajes registrados.")
        st.caption(f"Resultados: {st.session_state['study_journal'].with_suffix('.csv')}")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ReclamoRuta — estudio", page_icon="⏱️")

    @st.cache_resource(show_spinner="Cargando modelo…")
    def cached(settings) -> Runtime:
        return load_runtime(
            artifact_dir=Path(settings.artifact_dir), decision_path=Path(settings.decision_path),
            routes_path=Path(settings.routes_path), catalog_path=Path(settings.catalog_path),
            device=settings.device, calibration_path=Path(settings.calibration_path),
        )

    try:
        settings = _settings()
        plan = os.environ.get("RECLAMORUTA_STUDY_PLAN", "").strip()
        data = os.environ.get("RECLAMORUTA_STUDY_DATA", "").strip()
        if not plan or not data:
            raise AppConfigError("Faltan RECLAMORUTA_STUDY_PLAN y/o RECLAMORUTA_STUDY_DATA")
        runtime = cached(settings)
        study = load_study(Path(plan), Path(data), decision_path=Path(settings.decision_path))
    except (AppConfigError, StudyPlanError, ContractError, OSError, ValueError) as exc:
        st.title("ReclamoRuta — estudio humano")
        st.error(f"El estudio no puede iniciar: {exc}")
        st.stop()
        return
    render_study(runtime, study, Path(os.environ.get("RECLAMORUTA_STUDY_RESULTS", "data/study/results")))


if __name__ == "__main__":
    main()
