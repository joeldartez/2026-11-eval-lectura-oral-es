"""Identidad del protocolo: compatible con las cachés de la versión publicada."""

import json
from pathlib import Path

from .utilidades import digest


def protocol_hash(inference=False):
    base = Path(__file__).parent
    record = json.loads((base / "protocolo.json").read_text(encoding="utf-8"))[
        "inference" if inference else "code"
    ]
    current = digest([(name, (base / name).read_text(encoding="utf-8"))
                      for name in record["files"]])
    return record["identity"] if current == record["release"] else current
