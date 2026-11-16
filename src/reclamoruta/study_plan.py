"""Human study plans and result analysis (L4).

Two phases, with disjoint sources:

- ``piloto``: validation messages (default ``validation_model.csv``), seed 71,
  3 participants x 10. Only to rehearse the protocol and estimate times.
- ``final``: ``test_unique.csv`` only, seed 73, 3 participants x 40 (20 manual,
  20 assisted). Run only after models, routes, calibration, thresholds and
  protocol are frozen; never use its feedback to retrain or move thresholds.

Selection covers every intention before repeating one; messages are assigned
to (participant, condition) cells in length-sorted blocks so cells have
similar lengths. Order is ABBA per participant (S1/S3 start manual, S2
assisted). Plans hold message IDs only: no texts, no labels.

CLI::

    python -m reclamoruta.study_plan plan --phase final \
        --data data/processed/banking77/test_unique.csv \
        --decision reports/evaluation_scenario_v1/distilbert-v2/decision.json \
        --out data/study/plan_final.json
    python -m reclamoruta.study_plan analyze --results data/study/results/final_*.csv \
        --data data/processed/banking77/test_unique.csv --out reports/ESTUDIO_HUMANO_final.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import csv
import hashlib
import json
from pathlib import Path
import random

from reclamoruta.study import CONDITIONS, Trial, summarize

PHASES = {
    "piloto": {"seed": 71, "participants": 3, "per_participant": 10},
    "final": {"seed": 73, "participants": 3, "per_participant": 40},
}


class StudyPlanError(ValueError):
    """Invalid study source, plan or results."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"id", "text", "category", "split"} <= set(rows[0]):
        raise StudyPlanError(f"{path} no es un split procesado (id,text,category,split)")
    return rows


def _check_source(rows: list[dict[str, str]], phase: str) -> None:
    splits = {row["split"] for row in rows}
    if phase == "final" and splits != {"test_unique"}:
        raise StudyPlanError("El ensayo final usa solo test_unique.csv (vista deduplicada del test)")
    if phase == "piloto" and not all(s.startswith("validation") for s in splits):
        raise StudyPlanError("El piloto usa solo mensajes de validación, nunca test")


def _select(rows: list[dict[str, str]], n: int, rng: random.Random) -> list[dict[str, str]]:
    """Round-robin over shuffled classes so every class appears before repeats."""
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["id"]):
        by_class[row["category"]].append(row)
    classes = sorted(by_class)
    rng.shuffle(classes)
    for name in classes:
        rng.shuffle(by_class[name])
    chosen: list[dict[str, str]] = []
    depth = 0
    while len(chosen) < n:
        progressed = False
        for name in classes:
            if depth < len(by_class[name]):
                chosen.append(by_class[name][depth])
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            raise StudyPlanError(f"No hay suficientes mensajes: se necesitan {n}, hay {len(chosen)}")
        depth += 1
    return chosen


def _order(items: list[str], first: str, rng: random.Random) -> list[tuple[str, str]]:
    """ABBA blocks when each condition splits in two halves, else AB."""
    second = "asistida" if first == "manual" else "manual"
    half = len(items) // 2
    pools = {first: items[:half], second: items[half:]}
    for pool in pools.values():
        rng.shuffle(pool)
    if half % 2 == 0:
        q = half // 2
        pattern = [(first, 0, q), (second, 0, q), (second, q, half), (first, q, half)]
    else:
        pattern = [(first, 0, half), (second, 0, half)]
    return [(mid, cond) for cond, lo, hi in pattern for mid in pools[cond][lo:hi]]


def build_plan(
    data_path: Path, *, phase: str, seed: int | None = None,
    participants: int | None = None, per_participant: int | None = None,
    decision_path: Path | None = None,
) -> dict:
    """Return a JSON-safe plan of message IDs per participant slot and position."""
    if phase not in PHASES:
        raise StudyPlanError(f"phase debe ser una de {sorted(PHASES)}")
    defaults = PHASES[phase]
    seed = defaults["seed"] if seed is None else seed
    n_people = participants or defaults["participants"]
    per_person = per_participant or defaults["per_participant"]
    if n_people < 1 or per_person < 2 or per_person % 2:
        raise StudyPlanError("Se requieren participantes >= 1 y un número par de mensajes por persona")
    rows = read_rows(data_path)
    _check_source(rows, phase)
    rng = random.Random(seed)
    chosen = _select(rows, n_people * per_person, rng)

    # Length-sorted blocks of 2*P: each block gives one message to every
    # (participant, condition) cell, so cells end up with similar lengths.
    cells = [(p, c) for p in range(n_people) for c in CONDITIONS]
    chosen.sort(key=lambda r: (len(r["text"].split()), r["id"]))
    assigned: dict[tuple[int, str], list[str]] = defaultdict(list)
    for start in range(0, len(chosen), len(cells)):
        block_cells = cells[:]
        rng.shuffle(block_cells)
        for row, cell in zip(chosen[start:start + len(cells)], block_cells):
            assigned[cell].append(row["id"])

    slots = []
    for p in range(n_people):
        first = "manual" if p % 2 == 0 else "asistida"
        second = "asistida" if first == "manual" else "manual"
        items = assigned[(p, first)] + assigned[(p, second)]
        ordered = _order(items, first, rng)
        slots.append({
            "slot": f"S{p + 1}",
            "first_condition": first,
            "order": [
                {"position": i + 1, "message_id": mid, "condition": cond}
                for i, (mid, cond) in enumerate(ordered)
            ],
        })
    plan = {
        "phase": phase, "seed": seed, "source_file": Path(data_path).name,
        "source_sha256": _sha256(data_path), "n_messages": len(chosen),
        "per_participant": per_person, "participants": slots,
        "generator": "reclamoruta.study_plan",
        "notes": "Solo IDs de mensaje: sin textos ni etiquetas. No usar el feedback para reentrenar ni mover umbrales.",
    }
    if decision_path is not None:
        plan["decision_sha256"] = _sha256(decision_path)
    elif phase == "final":
        raise StudyPlanError("El plan final debe registrar la decisión congelada (--decision)")
    return plan


def write_plan(plan: dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError as exc:
        raise StudyPlanError(f"{out} ya existe; los planes nunca se sobrescriben") from exc
    return out


# --- results ---------------------------------------------------------------------

def _optional(value: str) -> str | None:
    return value if value != "" else None


def _bool(value: str) -> bool | None:
    return {"True": True, "False": False, "": None}[value]


def load_trials(path: Path) -> list[Trial]:
    """Read a CSV written by ``study.export_trials`` back into ``Trial`` objects."""
    trials = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            text = row.get("text")
            if text and text.startswith("'") and text[1:2] in ("=", "+", "-", "@", "\t", "\r"):
                text = text[1:]
            trials.append(Trial(
                participant_id=row["participant_id"], message_id=row["message_id"],
                condition=row["condition"], elapsed_s=float(row["elapsed_s"]),
                suggested_intent=_optional(row["suggested_intent"]),
                suggested_route=_optional(row["suggested_route"]),
                final_intent=row["final_intent"], final_route=row["final_route"],
                corrected=bool(_bool(row["corrected"])), needs_review=_bool(row["needs_review"]),
                model_version=_optional(row["model_version"]), text=text or None,
            ))
    return trials


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze(trials: Iterable[Trial], gold: Mapping[str, str], routes: Mapping[str, str]) -> dict:
    """Join trials with gold labels: human accuracy, detection rate and value inputs.

    ``q_detectado`` = wrong suggestions the person changed / wrong suggestions.
    Undefined (None) when no suggestion was wrong; never forced to 0 or 1.
    """
    trials = list(trials)
    missing = {t.message_id for t in trials} - set(gold)
    if missing:
        raise StudyPlanError(f"{len(missing)} mensajes sin etiqueta de referencia")
    result = summarize(trials)
    for condition in CONDITIONS:
        group = [t for t in trials if t.condition == condition]
        correct = sum(t.final_intent == gold[t.message_id] for t in group)
        route_ok = sum(t.final_route == routes[gold[t.message_id]] for t in group)
        result[condition].update({
            "human_accuracy": _rate(correct, len(group)),
            "human_route_accuracy": _rate(route_ok, len(group)),
        })
    assisted = [t for t in trials if t.condition == "asistida"]
    wrong = [t for t in assisted if t.suggested_intent != gold[t.message_id]]
    detected = [t for t in wrong if t.final_intent != t.suggested_intent]
    fixed = [t for t in wrong if t.final_intent == gold[t.message_id]]
    broken = [t for t in assisted
              if t.suggested_intent == gold[t.message_id] and t.final_intent != t.suggested_intent]
    result["asistida"].update({
        "wrong_suggestions": len(wrong),
        "detected_wrong_suggestions": len(detected),
        "fixed_wrong_suggestions": len(fixed),
        "human_introduced_errors": len(broken),
        "suggestion_accuracy": _rate(len(assisted) - len(wrong), len(assisted)),
    })
    result["value_inputs"]["q_detectado"] = _rate(len(detected), len(wrong))
    result["notes"] = (
        "Estudio pequeño con participantes no profesionales; tiempos observados, "
        "no evidencia de un centro de atención real. q_detectado se basa en "
        f"{len(wrong)} sugerencias incorrectas."
    )
    return result


def _gold(path: Path) -> dict[str, str]:
    return {row["id"]: row["category"] for row in read_rows(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Planes y análisis del estudio humano")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--phase", choices=sorted(PHASES), required=True)
    plan.add_argument("--data", type=Path, required=True)
    plan.add_argument("--decision", type=Path)
    plan.add_argument("--seed", type=int)
    plan.add_argument("--out", type=Path, required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--results", type=Path, nargs="+", required=True)
    analysis.add_argument("--data", type=Path, required=True)
    analysis.add_argument("--routes", type=Path, default=Path("configs/routes.json"))
    analysis.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "plan":
        built = build_plan(args.data, phase=args.phase, seed=args.seed, decision_path=args.decision)
        write_plan(built, args.out)
        print(f"Plan {args.phase}: {built['n_messages']} mensajes, {len(built['participants'])} participantes -> {args.out}")
        return 0
    routes = json.loads(args.routes.read_text(encoding="utf-8"))["intents"]
    trials = [t for path in args.results for t in load_trials(path)]
    result = analyze(trials, _gold(args.data), routes)
    result["results_files"] = {p.name: _sha256(p) for p in args.results}
    write_plan(result, args.out)
    print(json.dumps(result["value_inputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
