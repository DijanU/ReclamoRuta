"""Auditoría XAI offline del modelo final, con test verificado y sin entrenamiento.

Desde la raíz: python -m lab10.run --out lab10/results
Una salida existente se rechaza. El test solo se describe: no selecciona pesos,
temperatura, umbral ni una nueva configuración del producto.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time

from lab10.occlusion import explain_words


NEGATION = re.compile(r"\b(?:no|not|never|cannot|can't|don't|didn't|haven't|hasn't|wasn't|isn't|won't|without)\b", re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_inputs(repo: Path) -> None:
    """Reject pre-existing changes to the evaluated model, policy or scorer."""
    folder = repo / "reports/evaluation_scenario_v1"
    freeze = json.loads((folder / "freeze.json").read_text(encoding="utf-8"))
    model = freeze["models"]["distilbert-v2"]
    expected = {
        repo / "artifacts/distilbert-v2/manifest.json": model["manifest_sha256"],
        folder / "distilbert-v2/decision.json": model["decision_sha256"],
        folder / "distilbert-v2/calibration.json": model["calibration_sha256"],
        repo / "configs/routes.json": freeze["routes_sha256"],
        repo / "src/reclamoruta/predictor.py": freeze["source"]["python_sha256"]["predictor.py"],
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f"Archivo distinto del congelamiento: {path.relative_to(repo)}")


def load_verified_test(repo: Path, data: Path) -> tuple[list[dict], dict[str, str]]:
    """Bind data bytes to the project's published preparation manifest."""
    expected = json.loads((repo / "reports/BANKING77_MANIFEST.json").read_text(encoding="utf-8"))
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for key in ("output_sha256", "source", "counts", "policy"):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"El manifiesto de datos no coincide: {key}")
    hashes = {}
    for name in ("test_unique.csv", "label_to_id.json"):
        content = (data / name).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != expected["output_sha256"][name]:
            raise ValueError(f"Hash inválido: {name}")
        hashes[name] = digest
        if name == "test_unique.csv":
            # Parse the same bytes that were verified, avoiding a second file read.
            import io
            rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"), newline="")))
        else:
            catalog = json.loads(content)
    if len(rows) != expected["counts"]["test_unique"]:
        raise ValueError("Cantidad de prueba inesperada")
    if len({r["group_id"] for r in rows}) != len(rows):
        raise ValueError("La prueba contiene grupos repetidos")
    for row in rows:
        if row["split"] != "test_unique" or row["category"] not in catalog:
            raise ValueError("Partición o etiqueta inválida")
        if not row["text"].strip() or len(row["text"].strip()) > 2000:
            raise ValueError("Texto fuera del contrato de entrada")
    return rows, hashes


def select_local_cases(records: list[dict], threshold: float, config: dict) -> list[dict]:
    """Five distinct, deterministic test cases with predeclared diagnostic roles."""
    selected = []
    used = set()

    def choose(reason, candidates, key):
        available = [r for r in candidates if r["group_id"] not in used]
        if not available:
            raise ValueError(f"No existe un caso para {reason}; no se inventan ejemplos")
        chosen = dict(min(available, key=lambda r: (*key(r), r["id"])))
        used.add(chosen["group_id"])
        chosen.update(case_id=f"L{len(selected) + 1}", reason=reason)
        selected.append(chosen)

    choose("acierto_alta_confianza",
           [r for r in records if r["correct"] and r["score"] >= config["high_confidence_min"] and len(r["text"]) <= 250],
           lambda r: (-r["score"],))
    choose("error_de_ruta",
           [r for r in records if r["predicted_route"] != r["true_route"]],
           lambda r: (-r["score"],))
    choose("cercano_umbral_inferior",
           [r for r in records if 0 < threshold - r["score"] <= config["near_threshold_window"]],
           lambda r: (threshold - r["score"],))
    choose("cercano_umbral_superior",
           [r for r in records if 0 <= r["score"] - threshold <= config["near_threshold_window"]],
           lambda r: (r["score"] - threshold,))
    choose("subgrupo_negacion",
           [r for r in records if NEGATION.search(r["text"])],
           lambda r: (len(r["text"]),))
    return selected


def stratified_sample(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    if isinstance(per_class, bool) or not isinstance(per_class, int) or per_class < 1:
        raise ValueError("global_per_class debe ser entero positivo")
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    rng = random.Random(seed)
    selected = []
    for category in sorted(groups):
        pool = sorted(groups[category], key=lambda r: r["id"])
        if len(pool) < per_class:
            raise ValueError(f"Muestra demasiado grande para {category}")
        selected.extend(rng.sample(pool, per_class))
    return selected


def aggregate_words(explanations: list[dict], min_count: int) -> list[dict]:
    """Occurrence-weighted magnitude, with signed means and support counts."""
    grouped = defaultdict(list)
    for explanation in explanations:
        for token in explanation["tokens"]:
            grouped[token["word"].casefold()].append(token["delta"])
    return sorted([
        {"word": word, "count": len(deltas),
         "mean_abs_delta": math.fsum(abs(d) for d in deltas) / len(deltas),
         "mean_signed_delta": math.fsum(deltas) / len(deltas)}
        for word, deltas in grouped.items() if len(deltas) >= min_count
    ], key=lambda row: (-row["mean_abs_delta"], row["word"]))


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Versión de configuración inválida")
    for name in ("global_per_class", "global_min_word_occurrences", "batch_size", "threads"):
        val = config.get(name)
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError(f"{name} debe ser entero positivo")
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise ValueError("seed debe ser entero")
    for name in ("high_confidence_min", "near_threshold_window"):
        val = config.get(name)
        if isinstance(val, bool) or not isinstance(val, (float, int)) or not math.isfinite(val) or not 0 < val <= 1:
            raise ValueError(f"{name} debe estar en (0, 1]")


def run(repo: Path, data: Path, out: Path, config_path: Path) -> dict:
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError("Elegir un directorio de salida nuevo; no se sobrescribe una ejecución")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    verify_frozen_inputs(repo)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0", OMP_NUM_THREADS=str(config["threads"]), MKL_NUM_THREADS=str(config["threads"]))
    sys.path.insert(0, str(repo / "src"))
    import torch
    torch.set_num_threads(config["threads"])
    from reclamoruta.app import load_runtime

    rows, data_hashes = load_verified_test(repo, data)
    frozen = repo / "reports/evaluation_scenario_v1/distilbert-v2"
    artifact = repo / "artifacts/distilbert-v2"
    # Snapshot all model files and frozen inputs, then check again after analysis.
    bound_files = sorted(p for p in artifact.iterdir() if p.is_file()) + [
        frozen / "decision.json", frozen / "calibration.json", repo / "configs/routes.json",
        repo / "reports/evaluation_scenario_v1/freeze.json", repo / "reports/BANKING77_MANIFEST.json",
        repo / "reports/BANKING77_LABEL_TO_ID.json", config_path,
        repo / "src/reclamoruta/predictor.py", Path(__file__).with_name("occlusion.py"), Path(__file__),
    ]
    before = {str(p.relative_to(repo)): sha256(p) for p in bound_files}
    load_start = time.perf_counter()
    runtime = load_runtime(artifact_dir=artifact, decision_path=frozen / "decision.json",
                           calibration_path=frozen / "calibration.json", routes_path=repo / "configs/routes.json",
                           catalog_path=repo / "reports/BANKING77_LABEL_TO_ID.json", device="cpu")
    load_seconds = time.perf_counter() - load_start
    predictor = runtime.predictor
    threshold = runtime.decision.threshold
    routes = json.loads((repo / "configs/routes.json").read_text(encoding="utf-8"))["intents"]
    inference_start = time.perf_counter()
    predictions = predictor.predict([row["text"] for row in rows])
    predict_seconds = time.perf_counter() - inference_start
    records = []
    for row, pred in zip(rows, predictions, strict=True):
        records.append({"id": row["id"], "group_id": row["group_id"], "text": row["text"],
                        "true_label": row["category"], "predicted_label": pred.label,
                        "predicted_route": routes[pred.label], "true_route": routes[row["category"]],
                        "score": pred.score, "threshold": threshold,
                        "correct": pred.label == row["category"], "needs_review": pred.score < threshold})
    locals_ = select_local_cases(records, threshold, config)

    def calibrated_scorer(texts):
        # Use the exact deployed probability path, including its frozen temperature.
        return [predictor._scale(row) for row in predictor._probabilities(texts)]

    local_start = time.perf_counter()
    for case in locals_:
        case["occlusion"] = explain_words(case["text"], calibrated_scorer, predictor.categories, config["batch_size"])
        if case["occlusion"]["target_label"] != case["predicted_label"] or not math.isclose(case["occlusion"]["original_score"], case["score"], abs_tol=2e-5):
            raise ValueError("La explicación no coincide con la predicción original")
    local_seconds = time.perf_counter() - local_start
    sampled = stratified_sample(rows, config["global_per_class"], config["seed"])
    global_start = time.perf_counter()
    explanations = []
    global_records = []
    records_by_group = {r["group_id"]: r for r in records}
    for i, row in enumerate(sampled):
        explanation = explain_words(row["text"], calibrated_scorer, predictor.categories, config["batch_size"])
        original = records_by_group[row["group_id"]]
        if explanation["target_label"] != original["predicted_label"] or not math.isclose(explanation["original_score"], original["score"], abs_tol=2e-5):
            raise ValueError("La explicación global no coincide con la predicción original")
        explanations.append(explanation)
        # Full texts are only exported for the five illustrated, attributed cases.
        global_records.append({"id": row["id"], "group_id": row["group_id"], "true_label": row["category"],
                               "target_label": explanation["target_label"], "score": explanation["original_score"],
                               "n_variants": explanation["n_variants"], "elapsed_seconds": explanation["elapsed_seconds"]})
        if (i + 1) % 25 == 0:
            print(f"Oclusión global: {i + 1}/{len(sampled)}", flush=True)
    global_seconds = time.perf_counter() - global_start
    after = {str(p.relative_to(repo)): sha256(p) for p in bound_files}
    if before != after:
        raise ValueError("Cambió un archivo vinculado durante la auditoría")
    truth_counts = Counter(r["true_label"] for r in records)
    pred_counts = Counter(r["predicted_label"] for r in records)
    correct_counts = Counter(r["true_label"] for r in records if r["correct"])
    f1 = [2 * correct_counts[c] / (truth_counts[c] + pred_counts[c]) for c in predictor.categories]
    result = {
        "metadata": {"schema_version": 1, "seed": config["seed"], "model_version": predictor.model_version,
                     "threshold": threshold, "temperature": predictor.temperature,
                     "global_per_class": config["global_per_class"], "global_n": len(sampled),
                     "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                     "hashes": before, "data_hashes": data_hashes, "config": config,
                     "environment": {"python": platform.python_version(), "system": platform.system(),
                                     **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")}},
                     "attribution": "Five test excerpts: BANKING77, PolyAI, CC BY 4.0, https://github.com/PolyAI-LDN/task-specific-datasets",
                     "scope": "Post-hoc test audit only. No training, model selection, calibration refit or threshold retuning."},
        "test": {"n": len(rows), "accuracy": sum(r["correct"] for r in records) / len(rows),
                 "macro_f1": math.fsum(f1) / len(f1),
                 "wrong_routes": sum(r["predicted_route"] != r["true_route"] for r in records)},
        "local_cases": locals_,
        "global": {"n": len(sampled), "per_class": config["global_per_class"],
                   "min_word_occurrences": config["global_min_word_occurrences"],
                   "words": aggregate_words(explanations, config["global_min_word_occurrences"]),
                   "n_variants": sum(e["n_variants"] for e in explanations),
                   "elapsed_seconds": global_seconds, "scoring_seconds": math.fsum(e["scoring_seconds"] for e in explanations),
                   "sample": global_records,
                   "interpretation": "Mean absolute original-predicted-class score change by word occurrence. Two messages per true class; not natural production frequencies. Signed means and occurrence counts retained."},
        "cost": {"model_load_seconds": load_seconds, "predict_test_seconds": predict_seconds,
                 "local_elapsed_seconds": local_seconds, "global_elapsed_seconds": global_seconds,
                 "total_elapsed_seconds": time.perf_counter() - started,
                 "global_mean_explanation_seconds": global_seconds / len(sampled),
                 "local_mean_explanation_seconds": local_seconds / len(locals_), "device": "cpu", "threads": config["threads"],
                 "measurement_scope": "Wall-clock explanations with batched variants and a warm shared model. Model load, full-test predictions and rendering are separate."},
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"test": result["test"], "cost": result["cost"]}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--data", type=Path, default=repo / "data/processed/banking77")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--out", type=Path, default=repo / "lab10/results")
    args = parser.parse_args()
    run(repo, args.data.resolve(), args.out.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
