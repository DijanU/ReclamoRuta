# Estudio humano — piloto

Fecha: 15 de noviembre de 2026. Responsable: integrante a cargo de producto. Datos agregados: `ESTUDIO_HUMANO_piloto.json`. Ensayos por participante: `data/study/results/piloto_*.csv`. Los ensayos son anónimos y no incluyen textos.

## Protocolo ejecutado

- Plan `data/study/plan_piloto.json`: 30 mensajes de `validation_model.csv`, semilla 71 y 3 puestos (S1–S3), con 10 mensajes por persona (5 manuales y 5 asistidos). S1 y S3 empezaron por la condición manual y S2 por la asistida. No se usó el test.
- Modelo: DistilBERT V2 con la calibración y la decisión congeladas de `evaluation_scenario_v1` (umbral 0.85).
- App: `study_app.py` en una computadora local. El cronómetro corre desde que aparece el mensaje hasta la confirmación, y las pausas entre bloques no se miden.
- Participaron 3 personas voluntarias con consentimiento verbal, sin nombres registrados. *Completar: perfil general (por ejemplo, estudiantes universitarios) y nivel de inglés. No eran agentes bancarios.*

## Resultados

| | Manual | Asistida |
|---|---:|---:|
| Ensayos | 15 | 15 |
| Mediana de tiempo | 38.4 s | 8.8 s |
| Exactitud de intención final | 40.0 % (6/15) | 93.3 % (14/15) |
| Exactitud de ruta final | 93.3 % (14/15) | 100 % (15/15) |

- En la condición asistida, el modelo se equivocó en 1 de 15 sugerencias. Ese caso requería revisión (puntuación menor que el umbral) y la persona aceptó la sugerencia errónea. `q_detectado` = 0/1. **Con una sola observación no estima nada.**
- Nadie corrigió ninguna sugerencia (0/15), así que `t_corregir` no se puede estimar. No se rellena con cero.
- Medianas para `value.py`: `t_manual` 38.4 s, `t_aceptar` 7.9 s (12 casos aceptados sin revisión) y `t_revision` 8.8 s (3 casos con revisión obligatoria).

## Hallazgos para ajustar el protocolo

1. **Efecto de aprendizaje.** El primer mensaje de cada persona fue el más lento: de 38 a 124 s en manual y 40 s en el primer asistido de S2. Con 5 mensajes por condición, eso pesa mucho en las medianas. **Ajuste aplicado:** dos mensajes de práctica sin cronómetro y sin registro, uno por condición, antes de medir (`study_app.py`). Son consultas sintéticas escritas por el equipo, no de BANKING77.
2. **Elegir entre 77 intenciones es difícil para personas no expertas.** La intención manual acertó solo el 40 %, pero la ruta el 93 %: confunden intenciones vecinas que van al mismo equipo. En el ensayo final conviene reportar la ruta como métrica principal del flujo humano y la intención como secundaria.
3. **Aceptación sin revisar en los casos con revisión obligatoria.** En los 3 casos bajo el umbral, las personas aceptaron la sugerencia preseleccionada en una mediana de 8.8 s. Eso contradice el supuesto de que una revisión equivale a clasificar desde cero (35 s en `value_scenario.json`) y sugiere sesgo de anclaje. **Decisión pendiente del equipo antes del ensayo final:** en los casos con revisión obligatoria, ¿se debe mostrar la sugerencia sin preseleccionarla? Cambiaría lo que se mide; si se adopta, debe quedar fijada antes de abrir el ensayo final.
4. **Los tiempos difieren de los supuestos del escenario** (30 / 5 / 20 / 35 s). Si Fernando crea una nueva versión de supuestos con estos datos, debe usar solo validación y declarar que provienen de un piloto de 3 personas y 30 ensayos.

## Límites

Es un piloto: 3 personas, 30 ensayos y participantes no profesionales. Sirve para ajustar el protocolo, no para medir ahorro. No se usa para reentrenar ni para mover umbrales. Los mensajes del piloto son de validación y no se repiten en el ensayo final, que usa test único.
