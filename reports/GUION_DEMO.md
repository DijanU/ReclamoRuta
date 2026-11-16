# Guion de la demo (3–5 minutos)

Modelo: DistilBERT V2 con su calibración y decisión congeladas (umbral 0.85). Los resultados de abajo se comprobaron el 15 de noviembre de 2026 con la app real; si se cambia algún artefacto, volver a comprobarlos. Nada está programado a mano: la app clasifica en vivo.

## Preparación

```bash
cd ReclamoRuta
export RECLAMORUTA_ARTIFACT_DIR=artifacts/distilbert-v2
export RECLAMORUTA_DECISION=reports/evaluation_scenario_v1/distilbert-v2/decision.json
export RECLAMORUTA_CALIBRATION=reports/evaluation_scenario_v1/distilbert-v2/calibration.json
export HF_HUB_OFFLINE=1
.venv/bin/streamlit run src/reclamoruta/app.py
```

Abrir la app antes de empezar para que el modelo ya esté cargado. Tener la URL pública como opción principal y esta copia local como respaldo.

## Recorrido

| # | Consulta (copiar tal cual) | Resultado comprobado | Qué decir |
|---|---|---|---|
| 1 | `My new card still hasn't arrived, it's been two weeks.` | `card_arrival` → Tarjetas, 0.980, sugerencia aceptable | Flujo normal: intención, ruta y probabilidad calibrada. Confirmar. |
| 2 | `There's a payment on my statement I never made.` | `card_payment_not_recognised` → Seguridad, 0.986 | La ruta la define el mapa del grupo: los movimientos no reconocidos van a Seguridad. |
| 3 | `I lost my card yesterday and I think someone is using it` | `lost_or_stolen_phone` → Seguridad, 0.431, **revisión requerida** | La intención es errónea (teléfono en lugar de tarjeta), pero la ruta es correcta y la puntuación baja obliga a revisar. **Corregir** a `lost_or_stolen_card` y mostrar que la ruta se recalcula sin reentrenar. |
| 4 | `What's the weather like tomorrow?` | `card_delivery_estimate`, 0.707, revisión requerida | Fuera del dominio: el modelo no sabe decir "no es bancario". Aquí lo frena el umbral, pero no hay garantía de que siempre lo haga. |
| 5 | `I want to close my account and also change my PIN` | `change_pin`, **0.985**, aceptable | Límite: con dos intenciones elige una con mucha confianza. Una puntuación alta no garantiza que esté completo; por eso la revisión humana siempre está disponible. |
| 6 (opcional) | `Perdí mi tarjeta y necesito bloquearla` | `receiving_money`, 0.085, revisión | Solo funciona en inglés: en español no tiene sentido y la puntuación lo delata. |

Para la comparación de versiones, en lugar de cambiar de modelo en vivo, mostrar la tabla de `VERIFICACION_REAL.md`: en el test, el clásico obtuvo 87.86 %, V1 86.30 % y V2 91.98 %, con su valor estimado por lote. Si el catedrático propone una consulta, usarla en vivo, siempre en inglés y dentro del dominio bancario.

## Mensajes clave

- Sugiere y una persona confirma; no actúa sobre cuentas.
- V2 es mejor que V1 y que el clásico, pero V1 no superó al clásico: más deep learning no mejora automáticamente.
- El valor depende del costo de los errores y de los tiempos humanos. Los tiempos del piloto son preliminares (3 personas).

## Video (2–4 min, MP4)

Grabar la pantalla con QuickTime (Archivo → Nueva grabación de pantalla) recorriendo los casos 1, 3, 4 y 5 con narración. Exportar en 1080p. No mostrar terminales con rutas personales ni datos de participantes.
