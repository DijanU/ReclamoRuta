# Laboratorio 10 — Explainable AI

Auditoría del **DistilBERT V2** de ReclamoRuta. Se conservan los pesos, temperatura y umbral de la evaluación del proyecto; no se reentrena ni se seleccionan cambios usando este test.

## Entrega

- `Laboratorio_10_XAI.ipynb`: notebook comentado con investigación, implementación, tablas, figuras e interpretación.
- `results/locales.png` y `.svg`: cinco explicaciones locales.
- `results/global.png` y `.svg`: sensibilidad agregada por palabra.
- `results/results.json`: resultados reales, casos, posiciones, métricas, tiempos, entorno y hashes.

## Ejecución

Desde la raíz del repositorio, con Python 3.11 o superior. Primero instalar las dependencias del proyecto como indica el README principal y descargar los pesos con Git LFS. Para reproducir el entorno registrado, la ejecución publicada usó Python 3.13.2, PyTorch 2.14.0+cu126 en CPU y Transformers 4.56.2; las versiones exactas se guardan en el JSON. Las herramientas gráficas se instalan con:

```powershell
python -m pip install -r lab10/requirements.txt
```

Si faltan datos preparados, obtener los archivos oficiales fijados y ejecutar la preparación explicada en el README principal. Luego generar una ejecución nueva:

```powershell
python -m lab10.run --out data/cache/xai-nueva-ejecucion
python -m lab10.render --results data/cache/xai-nueva-ejecucion/results.json --out data/cache/xai-nueva-ejecucion
```

El directorio de `run` debe ser nuevo. Un error no se reemplaza por datos ficticios. El script verifica SHA-256 del CSV de prueba y catálogo contra el manifiesto de preparación publicado, y contrasta manifiesto del modelo, calibración, decisión, rutas y fuente del predictor con el congelamiento. También verifica que los archivos vinculados permanezcan intactos al terminar. No descarga datos o modelos durante la inferencia.

Para regenerar las figuras sin repetir la inferencia:

```powershell
python -m lab10.render --results lab10/results/results.json --out data/cache/xai-render
```

Los tiempos mostrados pertenecen al análisis guardado, no a la regeneración gráfica. El renderer reemplaza únicamente sus figuras de salida; no modifica el JSON. Las versiones de bibliotecas y las marcas internas de SVG pueden afectar la igualdad de bytes entre ejecuciones; la semilla fija la selección y las atribuciones se comparan con tolerancia numérica.

## Notebook

Abrir `Laboratorio_10_XAI.ipynb` en VS Code o Jupyter y seleccionar el kernel del entorno donde se instalaron las dependencias. Ejecutar las celdas en orden. Con `RUN_MODEL = False` se verifican los resultados reales guardados y se regeneran tablas y figuras. Con `RUN_MODEL = True` se repite la inferencia y oclusión sobre los datos preparados y pesos descargados; requiere esos archivos y no vuelve a entrenar.

La investigación de métodos y las referencias están en las celdas Markdown del notebook. Sus salidas se generaron ejecutando las celdas, no se completaron manualmente.

## Método y parámetros

La clase objetivo es `argmax p_cal(texto)` y permanece fija aunque cambie la clase ganadora. Para cada aparición de palabra se calcula:

```text
delta = puntuación calibrada original de la clase objetivo
        − puntuación de esa misma clase tras eliminar una aparición
```

Positivo indica que retirar la palabra reduce el puntaje; negativo, que lo aumenta. No se suman las barras como si reconstruyeran una predicción. Las palabras se localizan con `[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*`. Se conserva puntuación, espacios y posiciones; las apariciones repetidas se eliminan por separado. Un término con guion puede dividirse y estas unidades no son los subwords de DistilBERT. El texto original y los intervalos permiten reconstruir cada variante.

`config.json` fija semilla 42, lotes de 32 variantes, dos hilos CPU, dos consultas por clase, mínimo de tres apariciones para el ranking global, alta confianza desde 0.95 y cercanía al umbral de ±0.05. La temperatura es la ya ajustada en validación (aproximadamente 1.22365) y el umbral es 0.85. No se usa un conjunto de fondo, máscara o embedding de referencia: se compara con la eliminación literal de una palabra. Esto facilita inspección directa y tiene el costo de introducir frases artificiales.

La semilla 42 permite repetir el muestreo. Dos consultas por clase limitan el cómputo sin excluir ninguna intención; no bastan para medir estabilidad por clase. Tres apariciones evitan ordenar palabras vistas una sola vez, aunque tampoco garantizan significancia estadística. Lote 32 y dos hilos reutilizan la configuración de inferencia CPU para acotar memoria y costo. El mínimo 0.95 identifica un acierto claramente confiado; la ventana ±0.05 permite inspeccionar el límite operativo y luego se toma el ejemplo más cercano de cada lado. El límite local de 250 caracteres facilita mostrar completo el primer mensaje. Son criterios de diagnóstico, no hiperparámetros optimizados con este test.

## Selección local y global

Los cinco casos son distintos, seleccionados de forma determinista:

1. Acierto de mayor confianza, al menos 0.95 y texto de hasta 250 caracteres para facilitar lectura.
2. Error de ruta de mayor confianza según las etiquetas del dataset.
3. Puntaje más cercano a 0.85 por debajo, dentro de una ventana de 0.05.
4. Puntaje más cercano a 0.85 por encima o igual, dentro de una ventana de 0.05.
5. Mensaje más corto con una negación del patrón documentado en `run.py`.

Se excluyen los casos ya seleccionados y los empates se resuelven por identificador. Esta selección es diagnóstica: no permite calcular prevalencia de errores a partir de los cinco ejemplos. El error de ruta se define por el mapa del proyecto, no por departamentos de una institución real. Las etiquetas de referencia pueden contener ambigüedad y deben revisarse junto con el texto.

La muestra global toma dos consultas por cada una de las 77 intenciones, ordenando grupos e identificadores antes de aplicar semilla 42. Son 154 consultas; puede compartir casos con la inspección local. La magnitud global es la media de `abs(delta)` por aparición de cada palabra en minúsculas, exigiendo al menos tres apariciones. Se conserva también media con signo y conteo. No es una media por documento, no reparte interacciones y no representa la frecuencia natural de tráfico. El ranking explica sensibilidad de la clase predicha en contextos distintos, no una única clase global.

## Interpretación y límites

La prueba completa vuelve a producir 91.98 % de exactitud de intención, macro-F1 de 0.91996 y 82 rutas equivocadas sobre 3,079 consultas. Las explicaciones describen ese modelo existente y no se utilizan para mejorar esas cifras sobre el mismo test.

El caso de error de alta confianza dice “whats your exchange rate”: la referencia es `exchange_charge` y V2 predice `exchange_rate`. La frase admite una lectura compatible con la predicción; se debe revisar la etiqueta y el contexto antes de tratar el caso como un fallo semántico indiscutible. Una puntuación alta no resuelve esa ambigüedad.

El caso inferior al umbral muestra alta sensibilidad a “traveling”; el caso superior, a “verify”; y el mensaje con negación, a “can't”. Estos resultados distinguen señales temáticas esperables de asociaciones que conviene investigar con paráfrasis que conserven intención. No demuestran por sí solos un atajo, contaminación o causalidad. El ranking global incluye palabras con apenas tres apariciones; su efecto puede depender de una clase o redacción específica.

La utilidad en atención es facilitar revisión y detectar fragilidad. Un error de ruta puede producir transferencias o retrabajo y afectar la experiencia y confianza; aquí no se midió satisfacción ni daño reputacional. No se recomienda ejecutar explicaciones en todas las consultas por defecto: los tiempos guardados permiten evaluar auditorías por lotes o explicaciones opcionales.

## Pruebas

```powershell
python -m pytest tests/test_xai_occlusion.py tests/test_xai_pipeline.py -q
```

Las pruebas verifican clase fija cuando cambia el ganador, posición de palabras repetidas, batching, salidas inválidas, selección determinista, agregación sin cancelación de signos y rechazo de archivos o políticas alteradas. Las pruebas unitarias usan probabilidades controladas y no sustituyen los resultados reales publicados.

## Fuentes y privacidad

Los cinco extractos del test proceden de BANKING77, PolyAI, CC BY 4.0. Las eliminaciones son modificaciones analíticas de esos extractos. La muestra global conserva identificadores y conteos, sin distribuir sus textos completos. No se incluyen CSV de participantes ni consultas privadas de usuarios.

https://github.com/PolyAI-LDN/task-specific-datasets

https://huggingface.co/datasets/PolyAI/banking77

Los artículos y documentación de LIME, SHAP e Integrated Gradients están citados en el notebook. El análisis es externo a la interfaz publicada y no modifica su flujo de clasificación.
