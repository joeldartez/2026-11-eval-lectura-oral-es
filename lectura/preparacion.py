"""Carga anticipada de los procesadores y pesos utilizados en la evaluación."""

import os

from .config import CFG, PINS
from .modelos import ModelPool


def preparar(check=False):
    if check:
        # Debe fijarse antes de importar transformers o huggingface_hub.
        os.environ["HF_HUB_OFFLINE"] = "1"
    pool = ModelPool(CFG)
    try:
        for name, revision in PINS.items():
            print(f"{name} · {revision}", flush=True)
            pool.get(name)
    finally:
        pool.close()
