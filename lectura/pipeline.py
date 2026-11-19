"""Evaluación completa de una grabación y exportación."""

import copy
import json
import time
from pathlib import Path
from .arbitraje import LLMArbiter
from .config import CFG, OUTPUT_ROOT, REFERENCE_PATH, ROOT
from .evaluacion import evaluate_evidence
from .evidencia import collect_evidence
from .excel import write_short_excel
from .modelos import ModelPool
from .procedencia import protocol_hash
from .texto import build_reference, norm, tokens, wer
from .utilidades import digest


def evaluar_audio(audio_path, reference_path=REFERENCE_PATH, config=None, pool=None, output_root=OUTPUT_ROOT,
                  evidence=None, control_path=None, export=True):
    cfg = copy.deepcopy(CFG if config is None else config)
    if not cfg['asr_models'] or cfg['models_in_memory'] < 1: raise ValueError('Configuración de modelos inválida.')
    if cfg['llm_workers'] < 1 or cfg['llm_words_per_call'] < 1 or cfg['llm_max_word_calls'] < 0: raise ValueError('Presupuesto LLM inválido.')
    audio_path, reference_path, output_root = Path(audio_path), Path(reference_path), Path(output_root)
    started = time.perf_counter(); logs = []
    code_sha = protocol_hash()
    reference_text = reference_path.read_text(encoding='utf-8'); reference = build_reference(reference_text)
    pool = pool or ModelPool(cfg)
    if evidence is None: evidence = collect_evidence(audio_path, cfg, pool, output_root / '_cache', logs)
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    identity = {'audio': evidence['audio_sha'], 'reference': digest(reference_text), 'code': code_sha, 'protocol': cfg['protocol']}
    arbiter = LLMArbiter(cfg, output_root / '_llm_cache', identity)
    print('Reconciliación y arbitraje' + (' LLM paralelo' if cfg['use_llm'] else ' local'))
    result = evaluate_evidence(evidence, reference, cfg, arbiter)
    result['stage_logs'] = logs
    summary = result['summary']; run_id = digest([identity, cfg])[:16]
    summary.update({'meta__audio': audio_path.name, 'meta__protocol': cfg['protocol'], 'meta__run_id': run_id,
                    'meta__audio_sha256': evidence['audio_sha'], 'meta__reference_sha256': identity['reference'],
                    'meta__code_sha256': code_sha, 'meta__config_json': json.dumps(cfg, ensure_ascii=False, sort_keys=True),
                    'meta__model_revisions_json': json.dumps(pool.pins, sort_keys=True),
                    'meta__legacy_replay': int(evidence.get('legacy_replay', False)),
                    'meta__use_llm': int(cfg['use_llm']), 'execution__wall_s': time.perf_counter()-started,
                    'execution__inference_s': sum(x['seconds'] for x in logs),
                    'execution__cache_hits': sum(x['cache_hit'] for x in logs)})
    # El control nunca interviene en inferencia, selección o calificación.
    if control_path is not None:
        control_text = Path(control_path).read_text(encoding='utf-8')
        control = [{'norm': norm(w)} for w in tokens(control_text)]
        for name, source in evidence['asr'].items(): summary.update({'control_asr_'+name+'__'+k: v for k, v in wer(control, source['text']).items()})
        summary['control__reference_sha256'] = digest(control_text)
    for row in result['words']:
        if row['final_correct'] not in (0, 1) or row['first_final_correct'] not in (0, 1): raise AssertionError('Crédito debe ser 0/1.')
        if not row['attempt_present'] and row['final_correct']: raise AssertionError('No se admite crédito sin intento.')
    if summary['final__correct_estimated']+summary['final__incorrect_estimated'] != len(reference): raise AssertionError('Totales incoherentes.')
    if export:
        path = output_root / audio_path.stem / run_id / (audio_path.stem+'-lectura-short.xlsx')
        export_start = time.perf_counter(); result['excel_path'] = str(write_short_excel(result, path))
        summary['execution__export_s'] = time.perf_counter()-export_start
        print(f'Exportado: {path} | {time.perf_counter()-started:.1f}s')
    return result
