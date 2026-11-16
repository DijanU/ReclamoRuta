# Protocolo congelado del ensayo humano final

Congelado el 15 de noviembre de 2026, después del piloto (`ESTUDIO_HUMANO_piloto.md`) y antes de mostrar a cualquier persona un mensaje del ensayo final. No se cambia nada de esta lista mientras dure el ensayo. Si algo cambia, se trata como un experimento nuevo y se declara.

## Decisiones tomadas tras el piloto

1. **Práctica sin cronómetro:** cada sesión empieza con dos consultas sintéticas, una por condición, que no se registran.
2. **Sugerencia preseleccionada:** en la condición asistida, la sugerencia del modelo aparece seleccionada, también en los casos con revisión obligatoria, y la persona **siempre puede cambiar la categoría**. Es una decisión del grupo, con el sesgo de anclaje observado en el piloto como límite conocido que se reportará.
3. **Métricas principales del flujo humano:** exactitud de ruta y tiempos. La exactitud de intención se reporta como secundaria, porque elegir entre 77 intenciones resultó difícil para personas no expertas.

## Diseño

- Plan `data/study/plan_final.json`: 120 mensajes de `test_unique.csv`, semilla 73, que cubren las 77 intenciones. 3 puestos (S1–S3) × 40 mensajes, 20 manuales y 20 asistidos por persona, en bloques ABBA de 10 (S1 y S3 empiezan manual, S2 asistida). Ningún mensaje se repite.
- Modelo DistilBERT V2 con su calibración y decisión congeladas (umbral 0.85).
- Pueden participar las mismas personas del piloto, porque sus mensajes eran de validación y no se repiten. Se debe declarar en el informe.
- El feedback de este ensayo **no** se usa para reentrenar ni para mover umbrales. Es un subconjunto del test, no una validación externa.

## Hashes SHA-256 congelados

| Archivo | SHA-256 |
|---|---|
| `data/study/plan_final.json` | `2d23ef6c8301bb3da3ab966299182b5200feb6c89d56a9ae9ab8620ef6a48fec` |
| `data/processed/banking77/test_unique.csv` (no versionado) | `12a685f4e15d86719a7caa3e1502808134997b0101a7db674faac61d821b0d06` |
| `reports/evaluation_scenario_v1/distilbert-v2/decision.json` | `2f254e9fdb76b1eb9ec0915589914512a130c020ac943348358208d334bf8af5` |
| `reports/evaluation_scenario_v1/distilbert-v2/calibration.json` | `f9727ee3a56eb43132b1771425609abecc62a0256fc80da7b5c613198341673e` |
| `artifacts/distilbert-v2/manifest.json` | `56b7d3565d69a7f30d51a4b1a822bce50a81c3b3770be6061b0be68242c2c910` |
| `configs/routes.json` | `1033658df9db207ff34ff1a137abe57536977cc2d37b42f411c477b26fdbe647` |
| `src/reclamoruta/study_app.py` | `01e90c5e54f0eea3a059c90b57fbcabcad831132f1be27fa5a5bc314c12ee4b4` |
| `src/reclamoruta/study.py` | `10da36b3cf23132505f7f089af55b794140a95407933c257de019244acf28c10` |
| `src/reclamoruta/study_plan.py` | `4c82aece4e0414b711b5a28a117f185a3a39a14956c673dc8538cfb496b9fde5` |
| `src/reclamoruta/contracts.py` | `abf75e0f11b8607d508fc26acfd17d7875fe516c796a6a61cdf3801d70e9c441` |

La app del estudio comprueba por sí sola el hash del CSV de mensajes y el de la decisión contra el plan. Para comprobar el resto: `shasum -a 256 <archivo>`.

## Cómo correrlo

Es igual al piloto, con dos cambios:

```bash
export RECLAMORUTA_STUDY_PLAN=data/study/plan_final.json
export RECLAMORUTA_STUDY_DATA=data/processed/banking77/test_unique.csv
```

Cada sesión dura unos 20–25 minutos, más la práctica. Después:

```bash
.venv/bin/python -m reclamoruta.study_plan analyze \
  --results data/study/results/final_*.csv \
  --data data/processed/banking77/test_unique.csv \
  --out reports/ESTUDIO_HUMANO_final.json
```
