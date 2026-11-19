"""Decisiones locales y finales sobre la misma evidencia."""

import copy
import json
import time
from .alineacion import candidate_paths, word_evidence
from .arbitraje import arbitrate_path, arbitrate_words
from .metricas import reading_metrics
from .reconciliacion import materialize_path, refresh_word_measurements
from .texto import wer
from .utilidades import environment_info, safe


def evaluate_evidence(evidence, reference, cfg, arbiter):
    import numpy as np
    speech_detected = bool(evidence['acoustic']['speech'])
    if not speech_detected:
        evidence = copy.deepcopy(evidence)
        for result in evidence['asr'].values(): result['observations'] = []
    vector_fields = ('f0_times', 'f0', 'intensity_times', 'intensity', 'hnr_times', 'hnr', 'pyin', 'pyin_times', 'formant_times', 'f1', 'f2')
    for key in vector_fields:
        if key in evidence['acoustic']: evidence['acoustic'][key] = np.asarray(evidence['acoustic'][key], dtype=float)
    evidence['acoustic']['pitch_floor'] = cfg['pitch_floor']
    started = time.perf_counter(); paths, local_path = candidate_paths(reference, evidence, cfg)
    for candidate in paths:
        for i, event in enumerate(candidate['events']): event['event_id'] = candidate['path_id']+f':E{i+1:05d}'
    local_rows = word_evidence(reference, local_path, evidence, cfg, paths)
    local_metrics, _, _ = reading_metrics(reference, local_rows, local_path, evidence, cfg)
    chosen_path, logs = arbitrate_path(paths, local_path, reference, arbiter, cfg) if any(p['events'] for p in paths) else (local_path, [])
    rows = word_evidence(reference, chosen_path, evidence, cfg, paths)
    path_metrics, _, _ = reading_metrics(reference, rows, chosen_path, evidence, cfg)
    if cfg['use_llm']:
        try:
            rows, word_logs, final_path = arbitrate_words(rows, reference, chosen_path, paths, arbiter, cfg); logs += word_logs
        except Exception as error:
            final_path = chosen_path
            rows = word_evidence(reference, chosen_path, evidence, cfg, paths)
            logs.append({'task': 'word_arbitration_setup', 'status': 'fallback', 'error_type': type(error).__name__, 'error': str(error)[:300]})
            for row in rows:
                row.update(final_correct=row['local_correct'], first_final_correct=row['first_local_correct'],
                           final_category=row['local_category'], llm_applied=0, llm_changed=0, assignment_changed=0,
                           decision_method='rule_llm_failure_fallback')
    else: final_path = chosen_path
    final_path = materialize_path(chosen_path, rows, paths)
    rows = refresh_word_measurements(rows, reference, final_path, evidence, cfg, paths)
    final_metrics, boundaries, segments = reading_metrics(reference, rows, final_path, evidence, cfg, final=True)
    base = {r['word_id']: r for r in local_rows}
    for row in rows:
        row['baseline_correct'] = base[row['word_id']]['local_correct']
        row['baseline_category'] = base[row['word_id']]['local_category']
        row['baseline_attempt_present'] = base[row['word_id']]['attempt_present']
        row['baseline_event_id'] = base[row['word_id']]['event_id']
        row['path_llm_correct'] = row['local_correct']
        row['total_llm_changed'] = int(row['final_correct'] != row['baseline_correct'])
    ablation_keys = {'correct_estimated', 'correct_first_estimated', 'accuracy_fraction', 'pcpm', 'coverage_fraction',
                     'omissions', 'substitutions', 'partials', 'additions', 'repetitions', 'inversion_positions',
                     'advance_wpm', 'advance_end_start_ratio', 'pause_fraction', 'boundary_interior_400ms_fraction'}
    summary = {**{'local__'+k: v for k, v in local_metrics.items() if k in ablation_keys},
               **{'path_llm__'+k: v for k, v in path_metrics.items() if k in ablation_keys},
               **{'final__'+k: v for k, v in final_metrics.items()}}
    summary.update({'speaker__local': local_path['speaker_id'], 'speaker__final': chosen_path['speaker_id'],
                    'path__local': local_path['path_id'], 'path__final': chosen_path['path_id'],
                    'llm_effect__path_changed': int(local_path['path_id'] != chosen_path['path_id']),
                    'llm_effect__word_arbitrated': sum(r['llm_applied'] for r in rows),
                    'llm_effect__word_credit_changed': sum(r['llm_changed'] for r in rows),
                    'llm_effect__total_credit_changed': sum(r['total_llm_changed'] for r in rows),
                    'llm_effect__credit_recovered': sum(r['final_correct'] > r['baseline_correct'] for r in rows),
                    'llm_effect__credit_removed': sum(r['final_correct'] < r['baseline_correct'] for r in rows),
                    'llm_effect__assignment_changed': sum(r['assignment_changed'] for r in rows),
                    'llm_effect__candidate_id_changed': sum(r['candidate_changed'] for r in rows),
                    'llm__calls': len(logs), 'llm__successes': sum(r['status'] == 'ok' for r in logs),
                    'llm__fallback_calls': sum(r['status'] != 'ok' for r in logs),
                    'llm__summed_call_seconds': sum(r.get('seconds', 0.) for r in logs),
                    'llm__input_tokens': sum(r.get('input_tokens', 0) for r in logs),
                    'llm__output_tokens': sum(r.get('output_tokens', 0) for r in logs),
                    'llm__estimated_usd_new': sum(r.get('usd_new', 0.) for r in logs),
                    'evaluation_wall_s': time.perf_counter()-started})
    summary['meta__environment_json'] = json.dumps(environment_info(), sort_keys=True)
    summary['quality__speech_detected'] = int(speech_detected)
    for name, result in evidence['asr'].items():
        summary.update({'asr_'+name+'__'+k: v for k, v in wer(reference, result['text']).items()})
    return {'summary': safe(summary), 'words': safe(rows), 'events': safe(final_path['events']),
            'boundaries': boundaries, 'segments': segments, 'llm_logs': logs}
