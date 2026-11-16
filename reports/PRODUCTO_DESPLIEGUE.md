# Producto, experiencia de uso y despliegue (borrador para el informe)

Estado al 15 de noviembre de 2026. Lo marcado como **pendiente** no se ha hecho y no debe presentarse como hecho.

## Flujo del producto

Un agente pega la consulta del cliente, en inglés. El sistema valida que tenga entre 1 y 2,000 caracteres y la clasifica con DistilBERT V2 calibrado. Luego muestra la intención, la ruta de atención (uno de siete equipos definidos por el grupo) y la probabilidad calibrada. El agente confirma o corrige la sugerencia; si la puntuación está por debajo de 0.85, el sistema exige revisión. La ruta sale de `configs/routes.json`, no del clasificador. Corregir no reentrena el modelo.

## Decisiones de experiencia de uso

- **Revisión humana siempre disponible.** Incluso con una puntuación alta, se puede cambiar la intención. El umbral solo decide cuándo la revisión es **obligatoria**.
- **Puntuación honesta.** La interfaz dice "probabilidad calibrada (estimada en validación)", nunca "certeza", y advierte que una puntuación alta no garantiza acierto.
- **Sin respuestas inventadas.** Si faltan los pesos o el modelo falla, la app muestra un error y se detiene; no ofrece rutas por defecto.
- **Aviso de alcance permanente.** Solo inglés, no ejecuta operaciones, no detecta fraude ni da asesoría financiera, y pide no escribir datos personales.
- **Hallazgo del piloto:** cuando la sugerencia aparece preseleccionada, las personas tienden a aceptarla aunque se exija revisión. Ver `ESTUDIO_HUMANO_piloto.md`, hallazgo 3.

## Privacidad y seguridad

- La app no guarda el texto de las consultas. El estudio humano solo registra IDs anónimos (`P-xxxxxxxx`), IDs de mensaje, tiempos y elecciones.
- Antes de cargar los pesos se verifican el catálogo, las rutas, la decisión y el hash del manifiesto. `Predictor.load` verifica cada archivo. El modelo clásico usa pickle: solo se cargan artefactos propios del repositorio.
- Los errores no muestran trazas (`showErrorDetails = "none"`), no hay subida de archivos, el contenedor corre sin root, los pesos se montan en solo lectura y Hugging Face funciona en modo offline.
- `pip-audit` (15 de noviembre de 2026, sobre el entorno de desarrollo completo): **sin vulnerabilidades conocidas**. Corresponde a ese día y a esas versiones; repetirlo antes de entregar.
- **Pendiente:** HTTPS y límites de uso en el despliegue público. Los debe proveer la plataforma o un proxy inverso; no se agrega autenticación casera.

## Rendimiento medido

| Medición | Valor | Entorno |
|---|---|---|
| Latencia V2, batch 1 (200 consultas de validación) | p50 20.5 ms, p95 30.0 ms, 46.4 consultas/s | CPU, equipo del entrenamiento (`BENCHMARK_distilbert-v2_CPU.json`) |
| Arranque de la app con V2 (importaciones + carga + verificación) | 1.6–2.0 s (3 corridas, archivos en caché del sistema) | Apple M5 Pro, 24 GB, macOS 26.6.2, CPU |
| Memoria máxima del proceso con V2 cargado | ≈ 860 MiB | Ídem |

Implicación: un servidor con 1 GB de RAM queda justo; se recomiendan 2 GB o más. El modelo funciona en CPU, sin GPU. Las mediciones son locales, sin tráfico concurrente.

## Despliegue

- Ya preparados: `Dockerfile` (PyTorch CPU, usuario sin privilegios, healthcheck en `/_stcore/health`), `.streamlit/config.toml` de producción y respaldo local offline verificado: `streamlit run src/reclamoruta/app.py` con los pesos del repositorio y sin descargas.
- **Plataforma elegida: Streamlit Community Cloud**, que es gratuita, publica directamente desde el repositorio público de GitHub e incluye HTTPS. Entrada: `deploy/streamlit_app.py`, que sirve V2 con su calibración y decisión congeladas. Dependencias: `deploy/requirements.txt`, con PyTorch solo-CPU y versiones fijas; se comprobó que se resuelven en Linux con Python 3.12. Límite de uso en la app: 20 clasificaciones por minuto por sesión (`RECLAMORUTA_MAX_PER_MINUTE`). Es una protección por sesión, no un límite de red. La plataforma no publica una cifra oficial y estable de memoria; los ≈ 860 MiB medidos deben verificarse en el despliegue real.
- Alternativa si Community Cloud falla, por ejemplo por memoria o porque no descarga los pesos de Git LFS: Hugging Face Spaces con el `Dockerfile`. Antes de usarla hay que verificar en la cuenta si los Spaces Docker siguen siendo gratuitos, porque la documentación y el foro de Hugging Face se contradicen al respecto.
- **Pendiente:** desplegar y registrar la URL real, el arranque en frío, el comportamiento tras la inactividad y el costo (Q0 si se mantiene en el plan gratuito).

## Monitoreo propuesto (no implementado)

Se propone llevar latencia p50/p95 por día, fracción de casos bajo el umbral, tasa de correcciones por intención y ruta, y errores de carga. No se registrarían textos. Un aumento de correcciones o de casos bajo el umbral indicaría un cambio en el tipo de consultas (deriva) y la necesidad de reevaluar con datos nuevos.
