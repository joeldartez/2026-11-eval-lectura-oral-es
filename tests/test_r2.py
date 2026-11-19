"""Prueba del transporte de entrada y salida sin conexión a un bucket."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from lectura.config import ROOT
from lectura.entradas import R2


class R2Test(unittest.TestCase):
    def test_transporte(self):
        remote = R2.__new__(R2)
        remote.bucket, remote.client = "prueba", Mock()
        remote.client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "voz/notas.txt"}]}, {"Contents": [{"Key": "voz/uno.ogg"}]}]
        keys = list(remote.claves("voz/", None))
        self.assertEqual(keys, ["voz/uno.ogg"])
        remote.client.download_file.side_effect = lambda bucket, key, dest: Path(dest).write_bytes(b"audio")
        scratch = ROOT / ".pruebas"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            root = Path(tmp)
            with remote.audio(keys[0], root) as path:
                self.assertEqual((path.name, path.read_bytes()), ("uno.ogg", b"audio"))
            self.assertFalse(path.exists())
            path = root / "uno" / "run" / "informe.xlsx"
            remote.publicar(path, root, "resultados/")
            remote.client.upload_file.assert_called_once_with(str(path), "prueba", "resultados/uno/run/informe.xlsx")
