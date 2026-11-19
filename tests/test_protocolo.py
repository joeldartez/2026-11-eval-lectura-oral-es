"""Regresión de celdas XLSX con evidencia inventada."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openpyxl import load_workbook

from lectura.config import ROOT
from lectura.evaluacion import evaluate_evidence
from lectura.excel import write_short_excel
from tests.escenario import RecordedArbiter, sample


def workbook_signature(evaluate, export, mode, directory):
    ref, evidence, cfg = sample()
    cfg.update(use_llm=mode != "local", llm_words_per_call=24)
    arbiter = RecordedArbiter(fail=mode == "fallback")
    with patch("time.perf_counter", return_value=1.0):
        result = evaluate(evidence, ref, cfg, arbiter)
    # La versión del entorno es metadato, no una característica de la grabación.
    result["summary"]["meta__environment_json"] = "{}"
    result["stage_logs"] = []
    path = export(result, directory / f"{mode}.xlsx")
    book = load_workbook(path)
    signature = []
    try:
        for sheet in book:
            cells = [[(cell.data_type, cell.value) for cell in row] for row in sheet]
            # Cuantizar solo los floats: distintas libm pueden diferir en el último bit.
            values = [[(kind, round(val, 10) if isinstance(val, float) else val)
                       for kind, val in row] for row in cells]
            payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
            signature.append([sheet.title, sheet.dimensions, sheet.freeze_panes,
                              sheet.auto_filter.ref, hashlib.sha256(payload.encode()).hexdigest()])
    finally:
        book.close()
    return signature


class ReplicationTest(unittest.TestCase):
    def test_workbooks(self):
        expected = json.loads(Path(__file__).with_name("firmas.json").read_text())
        scratch = ROOT / ".pruebas"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            for mode, sheets in expected.items():
                with self.subTest(mode=mode):
                    got = workbook_signature(evaluate_evidence, write_short_excel, mode, Path(tmp))
                    self.assertEqual(sheets, got)
