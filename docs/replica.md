# Código, resultados y pruebas

El repositorio incluye el [pasaje del artículo](../pasajes/pedro-y-mateo.txt) con su puntuación y la configuración del método. Cada audio se procesa por separado. Los modelos se reutilizan entre grabaciones y las consultas al juez por palabra pueden ejecutarse en paralelo.

| Etapa del método | Archivos de referencia |
|---|---|
| Parámetros, modelo de OpenAI y presupuestos | [`config.py`](../lectura/config.py) |
| Secuencia de etapas e identidad de ejecución | [`pipeline.py`](../lectura/pipeline.py) |
| Carga de modelos y extracción de evidencia | [`modelos.py`](../lectura/modelos.py), [`inferencia.py`](../lectura/inferencia.py), [`acustica.py`](../lectura/acustica.py), [`evidencia.py`](../lectura/evidencia.py) |
| Tratamiento del texto, hipótesis y repeticiones | [`texto.py`](../lectura/texto.py), [`alineacion.py`](../lectura/alineacion.py) |
| Prompts, decisiones y recorrido final | [`arbitraje.py`](../lectura/arbitraje.py), [`reconciliacion.py`](../lectura/reconciliacion.py), [`evaluacion.py`](../lectura/evaluacion.py) |
| Cálculo y exportación | [`metricas.py`](../lectura/metricas.py), [`excel.py`](../lectura/excel.py) |

**Pesos y entorno.** Las revisiones de los seis modelos están fijadas en [`modelos.lock.json`](../modelos.lock.json). Las versiones de las dependencias están fijadas en [`uv.lock`](../uv.lock). La versión de eSpeak NG, el controlador y el dispositivo también pueden influir en los resultados. `modelos --check` comprueba la carga de los pesos sin acceder a Hugging Face.

**Identidad y cachés.** `meta__code_sha256` identifica la versión del código utilizada. El [manifiesto del protocolo](../lectura/protocolo.json) registra las huellas de los módulos que intervienen en la evaluación. Al modificar un módulo de la lista, se calcula una identidad nueva. Las cachés `_cache/` (inferencia) y `_llm_cache/` (árbitro) están bajo la carpeta de salida. Para una ejecución independiente, indicar otra carpeta con `--salida`. Para el control opcional de una grabación concreta, `--audio ... --control texto.txt` añade comparaciones WER al resumen. El control no interviene en la calificación.

**Reproducibilidad.** El resultado depende del audio, del pasaje, de la configuración y de las versiones de los modelos y las dependencias. Una nueva llamada a un servicio remoto puede devolver otra decisión incluso con las mismas entradas. Las grabaciones y calificaciones del estudio no se distribuyen.

**Cómo comprobar la instalación**, sin pesos ni credenciales:

```sh
uv run --locked python -m unittest discover -s tests -v
```

Las pruebas usan un pasaje y producciones inventadas con repetición, desacuerdo entre transcriptores y una segunda voz. Comparan las hojas generadas con firmas de referencia en tres condiciones: reglas locales, arbitraje simulado y fallo del servicio. Se fijan el reloj y los metadatos de entorno, y se redondean los números a diez decimales solo para esta prueba. La prueba R2 verifica por separado el transporte con un cliente simulado.
