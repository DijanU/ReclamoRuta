"""Render figuras de oclusión sin cargar el clasificador.

Usage: python -m lab10.render --results lab10/results/results.json
       --out lab10/results
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#21618c"
RED = "#b74b40"

def _number(value, digits=3):
    if value is None:
        return "no registrado"
    return f"{float(value):.{digits}f}"


def _tokens(case):
    return case.get("occlusion", {}).get("tokens", [])


def _strongest(case, limit=6):
    return sorted(_tokens(case), key=lambda t: (-abs(t["delta"]), t.get("start", 0)))[:limit]


def _local_figure(cases, out_dir):
    """Keep the entire complaint visible; long texts increase panel height."""
    rows = list(cases[:5])
    wrapped = [textwrap.fill(str(c.get("text", "")), 55, break_long_words=True) for c in rows]
    heights = [max(1.45, 0.17 * (s.count("\n") + 1) + 1.05) for s in wrapped]
    if not rows:
        rows, wrapped, heights = [{}], ["No hay casos locales registrados."], [2.2]
    fig = plt.figure(figsize=(8, sum(heights) + 0.65), facecolor="white")
    grid = fig.add_gridspec(len(rows), 2, width_ratios=[1.5, 1], height_ratios=heights,
                            left=0.025, right=0.97, top=0.94, bottom=0.055, hspace=0.32, wspace=0.42)
    fig.suptitle("Oclusión local · clase objetivo fija", fontsize=13, x=0.025, ha="left")
    reasons = {"acierto_alta_confianza": "Acierto confiado", "error_de_ruta": "Error de ruta",
               "cercano_umbral_inferior": "Bajo el umbral", "cercano_umbral_superior": "Sobre el umbral",
               "subgrupo_negacion": "Negación"}
    for i, (case, content) in enumerate(zip(rows, wrapped)):
        text_ax, bar_ax = fig.add_subplot(grid[i, 0]), fig.add_subplot(grid[i, 1])
        text_ax.axis("off")
        reason = reasons.get(case.get("reason"), case.get("reason", ""))
        title = f"{case.get('case_id', case.get('id', i + 1))} · {reason}"
        status = "sí" if case.get("needs_review") else "no"
        info = (f"Real: {case.get('true_label', '—')}\nPredicha: {case.get('predicted_label', '—')}\n"
                f"Score {_number(case.get('score'))} · Umbral {_number(case.get('threshold'))} · Revisión {status}")
        text_ax.text(0, 1, title, va="top", fontsize=11, fontweight="bold", color=BLUE)
        text_ax.text(0, .81, info, va="top", fontsize=9.3, linespacing=1.15)
        text_ax.text(0, .36, content, va="top", fontsize=10.5, linespacing=1.15)
        chosen = list(reversed(_strongest(case)))
        if chosen:
            values = [t["delta"] for t in chosen]
            labels = [f"{t['word']} [{t.get('start', '?')}:{t.get('end', '?')}]" for t in chosen]
            bar_ax.barh(range(len(chosen)), values, color=[BLUE if x >= 0 else RED for x in values], height=0.62)
            bar_ax.set_yticks(range(len(chosen)), labels, fontsize=8.5)
            bound = max(max(abs(x) for x in values) * 1.4, 0.01)
            bar_ax.set_xlim(-bound, bound)
            for j, delta in enumerate(values):
                bar_ax.text(delta + (bound * .025 if delta >= 0 else -bound * .025), j,
                            f"{delta:+.3f}", va="center", ha="left" if delta >= 0 else "right", fontsize=8.5)
        else:
            bar_ax.text(.5, .5, "Sin tokens evaluados", transform=bar_ax.transAxes, ha="center")
        bar_ax.axvline(0, color="#667788", lw=.7)
        bar_ax.set_xlabel("Delta de score al retirar", fontsize=8.5, labelpad=1)
        bar_ax.tick_params(axis="x", labelsize=7.5, pad=1)
        bar_ax.spines[["top", "right", "left"]].set_visible(False)
    paths = [out_dir / "locales.png", out_dir / "locales.svg"]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return paths


def _global_figure(data, out_dir):
    words = sorted((w for w in data.get("words", []) if w["count"] >= 3),
                   key=lambda w: (-w["mean_abs_delta"], w["word"]))[:12]
    fig, ax = plt.subplots(figsize=(10, 5.1), facecolor="white")
    ordered = list(reversed(words))
    ax.barh([f"{w['word']} (n={w['count']})" for w in ordered],
            [w["mean_abs_delta"] for w in ordered], color=BLUE)
    ax.set_xlabel("Media de |delta| por aparición evaluada")
    ax.set_title(f"Agregado global · {data.get('n', '—')} textos · mínimo 3 apariciones", loc="left", pad=16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    if not words:
        ax.text(.5, .5, "Ninguna palabra cumple count ≥ 3", transform=ax.transAxes, ha="center")
    fig.tight_layout(pad=1.8)
    paths = [out_dir / "global.png", out_dir / "global.svg"]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return paths


def render_results(results_path: Path, out_dir: Path) -> list[Path]:
    """Genera las cuatro figuras y devuelve sus rutas."""
    results_path, out_dir = Path(results_path), Path(out_dir)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = _local_figure(data.get("local_cases", []), out_dir)
    paths += _global_figure(data.get("global", {}), out_dir)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in render_results(args.results, args.out):
        print(path)


if __name__ == "__main__":
    main()
