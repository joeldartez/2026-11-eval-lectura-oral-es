Bogotá, 20 de noviembre de 2026
### Evaluación automática de la precisión, el ritmo y la velocidad de la lectura oral en español con modelos preentrenados

**Evaluación de lectura oral en español**

A partir de una grabación y del texto que debía leerse, este código estima **qué palabras se intentaron, cuáles se leyeron correctamente y cuánto tiempo transcurrió entre ellas**. Entrega un Excel por audio para seguir cada resultado hasta las palabras, los eventos y las decisiones que lo explican.

El repositorio implementa el segundo de los dos métodos descritos en el artículo *¿Qué pulso tiene la lectura en tercero? Evaluación automática de la precisión, el ritmo y la velocidad de la lectura oral en español con modelos preentrenados*. Incluye el [pasaje de Pedro y Mateo](pasajes/pedro-y-mateo.txt). Las grabaciones y calificaciones del estudio son privadas.

[Instalar](#1--preparar-la-máquina) · [Evaluar](#3--evaluar-una-lectura) · [Leer el Excel](#4--leer-el-excel) · [R2 y RunPod](docs/nube.md) · [Reproducibilidad](docs/replica.md)

## 1 · Preparar la máquina

Se requieren **Python 3.12.10**, [uv](https://docs.astral.sh/uv/getting-started/installation/) y **eSpeak NG**. Para reducir el tiempo de inferencia se recomienda una GPU NVIDIA con un controlador compatible con **CUDA 13.2**, la versión que utiliza PyTorch en este entorno. Si CUDA no está disponible, el código selecciona la CPU. Los seis modelos ocupan alrededor de **9,3 GB**. Se necesita espacio adicional para el entorno y la caché.

- **Ubuntu/Debian:** instalar `espeak-ng` y `libsndfile1` con `sudo apt install espeak-ng libsndfile1`.
- **Windows:** instalar [eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases). La biblioteca se detecta en `C:\Program Files\eSpeak NG\libespeak-ng.dll`. Si se instala en otra ubicación, indicar su ruta con la variable de entorno `PHONEMIZER_ESPEAK_LIBRARY`.

Desde la raíz del repositorio:

```sh
uv python install 3.12.10
uv sync --locked
```

## 2 · Configurar OpenAI y descargar los modelos

Para ejecutar el método completo se requiere una **cuenta de la API de OpenAI** con facturación habilitada, acceso al modelo `gpt-6-luna` y una [clave de API](https://platform.openai.com/api-keys).

1. Copiar [`.env.example`](.env.example) a `.env`: `cp .env.example .env` en Linux o `Copy-Item .env.example .env` en PowerShell.
2. Asignar la clave a `OPENAI_API_KEY`. Para uso local, dejar vacíos los campos R2. Las variables del sistema tienen prioridad.

```sh
uv run --locked python -m lectura modelos          # Descargar y abrir los seis modelos
uv run --locked python -m lectura modelos --check  # Comprobar la carga desde disco
```

Los pesos quedan en `modelos/`. Las [revisiones fijadas](modelos.lock.json) permiten descargar las mismas versiones de los modelos. Silero VAD se incluye en su paquete de Python, mientras que eSpeak NG se instala en el sistema. La preparación no consume llamadas a OpenAI.

## 3 · Evaluar una lectura

Colocar en **`audios/`** una grabación por lectura, ya recortada al segmento del pasaje. WAV, FLAC y OGG son opciones adecuadas. No se necesita un CSV. Se utiliza SoundFile/libsndfile para abrir los archivos. Si la grabación está en otro formato (por ejemplo, WebM), convertirla antes de la evaluación. Para comparar ejecuciones, utilizar siempre el mismo archivo de entrada.

```sh
uv run --locked python -m lectura evaluar --audio audios/lectura-01.ogg
uv run --locked python -m lectura evaluar                  # toda la carpeta, por nombre
uv run --locked python -m lectura evaluar --carpeta mis-audios --limite 2
```

- **Otro pasaje:** guardar el texto plano en UTF-8, con su puntuación y sin título, y añadir `--pasaje pasajes/otro.txt`.
- **Otra carpeta de salida:** añadir `--salida resultados/otro-estudio`.
- **Solo reglas locales:** añadir `--sin-llm` para ejecutar la ablación. Para reproducir el método del artículo, utilizar la configuración predeterminada con OpenAI.
- **Todas las opciones:** ejecutar `python -m lectura evaluar --help` dentro del entorno.

## 4 · Leer el Excel

Para `lectura-01.ogg`, el libro se guarda en **`resultados/short/lectura-01/<id>/lectura-01-lectura-short.xlsx`**. El identificador depende del contenido del audio, el pasaje, el protocolo y la configuración. Al repetir la misma ejecución se reutilizan las cachés y se vuelve a escribir ese libro.

| Hoja | Qué se encuentra |
|---|---|
| `lectura` | Resumen por audio: precisión, cobertura, palabras correctas por minuto, ritmo y metadatos. |
| `palabras` | Una fila por **aparición** en el pasaje, incluso si no se intentó. Hipótesis, fonemas, tiempos y créditos. |
| `eventos` | Producciones del recorrido final, incluidas repeticiones y retornos. |
| `fronteras` | Intervalos entre palabras consecutivas y su puntuación de referencia. |
| `segmentos` | Medidas de ritmo y acústica por tramo de la lectura. |
| `ejecucion` | Etapas, tiempos, uso de caché, llamadas al juez y posibles fallos. |
| `variables` | Definiciones, unidades e interpretación de los campos del resumen. |

En `lectura` se pueden comparar tres etapas: `local__` (reglas), `path_llm__` (recorrido elegido) y `final__` (después del arbitraje por palabra). Si `llm__fallback_calls` es mayor que cero, alguna decisión recurrió a las reglas locales. El motivo figura en `ejecucion`. Una celda acústica vacía significa que no fue posible medir esa variable.

## Cómo funciona

1. **Reconocer sin borrar los errores.** Qwen3-ASR y Parakeet transcriben independientemente. Parakeet aporta los tiempos de sus palabras y Qwen3-ForcedAligner sitúa en el audio las palabras que reconoce Qwen. Dos modelos wav2vec2 estiman los sonidos producidos, que se contrastan con los esperados en español latinoamericano mediante eSpeak NG.
2. **Seguir al lector.** Nemotron separa turnos de voz. La alineación conserva varios recorridos posibles: en «Pedro… Pedro y Mateo», la repetición cuenta como producción, pero no como una palabra adicional de avance.
3. **Resolver discrepancias.** El juez de OpenAI recibe evidencia textual estructurada, no la grabación. Escoge entre recorridos existentes y revisa casos dudosos. Cada decisión debe usar los identificadores suministrados.
4. **Medir precisión, velocidad y ritmo.** Se distinguen primer intento y crédito final. Un intervalo largo puede contener habla: para medir pausas se cruza la ausencia de voz de Silero VAD con caídas de intensidad (Praat/Parselmouth) o de energía (librosa).

Los resultados permiten estudiar la precisión, el ritmo y la velocidad de la lectura en voz alta. Por sí solos, no indican si la lectura corresponde al nivel esperado para el grado escolar ni si se comprendió el pasaje. El [mapa del código y las pruebas](docs/replica.md) explica dónde adaptar cada etapa y cómo ejecutar las pruebas incluidas en el repositorio.

**Licencia:** [MIT](LICENSE) para el código propio. [Licencias de modelos, dependencias y servicios](TERCEROS.md).
