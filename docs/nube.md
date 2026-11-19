# Audios en R2 · ejecución en RunPod

**Dónde están los audios** y **dónde corre el programa** son decisiones independientes. Una máquina local puede leer R2 y un pod puede leer archivos de su volumen. Ambos pasan los mismos bytes al evaluador.

### Leer un bucket privado

1. Subir las grabaciones a `audios/` en el bucket de Cloudflare R2, con un nombre distinto para cada grabación. No se requiere un CSV, acceso público ni CORS.
2. Completar `.env` a partir de [`.env.example`](../.env.example): `R2_ENDPOINT_URL` corresponde al endpoint S3 de la cuenta (habitualmente `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`), `R2_BUCKET` al nombre del bucket y las otras dos variables a las credenciales S3. Se requieren permisos para listar y leer objetos. Para subir libros también se necesita permiso de escritura bajo el prefijo de resultados.
3. Ejecutar desde la raíz del repositorio:

```sh
uv run --locked python -m lectura evaluar --r2 --prefijo audios/ --limite 1
uv run --locked python -m lectura evaluar --r2 --prefijo audios/ --publicar-en resultados/lectura-oral/
```

El primer comando guarda el Excel solo en el disco local. El segundo evalúa el prefijo completo y, después de cada exportación, sube el libro a `s3://<bucket>/resultados/lectura-oral/<audio>/<id>/<audio>-lectura-short.xlsx`. Descarga una grabación a la vez a un directorio temporal dentro de `--salida` y lo elimina al salir del contexto. El libro local se conserva si la subida falla. Las cachés y las respuestas del juez permanecen en el disco de trabajo. No se guarda un registro remoto para reanudar la ejecución.

### Ejecutar en RunPod

1. Crear un pod Linux con una GPU NVIDIA, un controlador compatible con CUDA 13.2 y un volumen persistente montado en `/workspace`. Reservar espacio para Python, PyTorch, los pesos y los resultados. Como referencia inicial se sugieren 100 GB. Definir `OPENAI_API_KEY` como secreto del pod. Si se utiliza R2, definir también sus cuatro variables como secretos.
2. En la terminal del pod:

```bash
nvidia-smi
apt-get update
apt-get install -y git curl ca-certificates espeak-ng libsndfile1
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
cd /workspace
# Clonar el repositorio público en /workspace y entrar en su carpeta.
export UV_CACHE_DIR=/workspace/.cache/uv
uv python install 3.12.10
uv sync --locked
uv run --locked python -c "from torch.cuda import is_available
assert is_available(), 'CUDA no disponible'"
uv run --locked python -m lectura modelos
```

3. Ejecutar uno de los comandos R2 anteriores. Para trabajar con archivos locales, copiar las grabaciones a `audios/` y ejecutar `uv run --locked python -m lectura evaluar`. Si el repositorio está en `/workspace`, los libros y las cachés quedan en el volumen. Antes de retirar el pod, conservar el volumen o descargar los libros.
