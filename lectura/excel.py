"""Contrato del libro por audio del método de evaluación."""

import json
from pathlib import Path
from .utilidades import safe


def document_variable(name):
    if name.endswith('_s') or 'duration_s' in name: unit = 'seconds'
    elif 'wpm' in name or 'pcpm' in name: unit = 'words/minute'
    elif name.endswith('_hz'): unit = 'Hz'
    elif name.endswith('_db'): unit = 'dB'
    elif name.endswith('_st'): unit = 'semitones'
    else: unit = 'dimensionless/count/category'
    definitions = {
        'wer': '(substitutions + deletions + insertions) / expected_words; normalized words, Levenshtein',
        'accuracy_fraction': 'sum(binary estimated correct credit) / expected_words',
        'pcpm': '60 * estimated_correct / elapsed_reading_seconds',
        'advance_wpm': '60 * first chronological target visits / elapsed seconds; repeats excluded',
        'production_wpm': '60 * target occurrence attempts / elapsed seconds; repeats included',
        'coverage_fraction': 'distinct temporally observed target positions / expected_words',
        'pause_fraction': 'convergent acoustic silence seconds / elapsed seconds',
        'pause_s': '(Praat silence UNION RMS low energy) INTERSECT Silero silence; exclude external voice',
        'phone_mean': 'mean available model token similarities: 1-edit_distance/max(sequence_lengths)',
        'f0_range_st': '12*log2(F0_p90/F0_p10), Praat autocorrelation; clean sounding windows',
        'f0_iqr_st': 'P75-P25 of 12*log2(F0/F0_median)',
        'advance_end_start_ratio': 'advance_wpm last time third / first time third',
    }
    field = name.split('__')[-1]
    definition = definitions.get(field, 'Explicit descriptive field; calculation in reading_metrics / track_summary / evaluate_evidence.')
    if 'boundary_' in field: definition = 'First consecutive target visits; gap = next_start - previous_end; punctuation from reference, not ASR.'
    if 'mixed_median' in field: definition = 'Praat Burg, 6500Hz ceiling; mixed phonetic content, not validated individual vowel.'
    return {'variable': name, 'unit': unit, 'definition': definition,
            'missing': 'Blank = physically unmeasurable numeric descriptor. Word credits are always binary.',
            'level': name.split('__')[0] if '__' in name else 'metadata'}


def flat_word(row):
    excluded = {'phones', 'hypotheses', 'asr_matches', 'asr_sources', 'phone_observed', 'phone_similarity', 'phone_posterior'}
    out = {k: v for k, v in row.items() if k not in excluded}
    for key in ('hypotheses', 'asr_matches', 'phone_observed', 'phone_similarity', 'phone_posterior'):
        for model, value in row[key].items(): out[key+'__'+model] = value
    return out


def write_short_excel(result, path):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook(); wb.remove(wb.active)
    summary = result['summary']; priority = ['meta__audio', 'meta__protocol', 'final__accuracy_fraction', 'final__pcpm',
        'final__coverage_fraction', 'final__omissions', 'final__substitutions', 'final__repetitions', 'final__advance_wpm',
        'final__advance_end_start_ratio', 'final__pause_fraction', 'final__boundary_interior_400ms_fraction',
        'local__accuracy_fraction', 'llm_effect__total_credit_changed', 'llm__calls', 'execution__wall_s']
    ordered = {k: summary[k] for k in priority if k in summary}; ordered.update(summary)
    tables = {'lectura': [ordered], 'palabras': [flat_word(r) for r in result['words']],
              'eventos': result['events'], 'fronteras': result['boundaries'], 'segmentos': result['segments'],
              'ejecucion': result['stage_logs']+result['llm_logs'],
              'variables': [document_variable(k) for k in ordered]}
    for name, records in tables.items():
        ws = wb.create_sheet(name); columns = list(dict.fromkeys(k for r in records for k in r))
        if not columns: columns = ['status']; records = [{'status': 'empty'}]
        if len(columns) > 1000: raise ValueError('Exportación short excede el contrato de columnas compactas.')
        ws.append(columns)
        for row in records:
            values = []
            for key in columns:
                value = safe(row.get(key))
                if isinstance(value, (dict, list)): value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
                if isinstance(value, str) and len(value.encode('utf-16-le'))//2 > 32767: raise ValueError('Texto inesperadamente largo.')
                values.append(value)
            ws.append(values)
            for cell, value in zip(ws[ws.max_row], values):
                if isinstance(value, str): cell.data_type = 's'
        for cell in ws[1]: cell.font = Font(bold=True, color='FFFFFF'); cell.fill = PatternFill('solid', fgColor='24476B')
        ws.freeze_panes = 'C2'; ws.auto_filter.ref = ws.dimensions
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem+'.tmp.xlsx'); wb.save(temporary); temporary.replace(path)
    check = load_workbook(path, read_only=True, data_only=False)
    if check['lectura'].max_row != 2 or check['palabras'].max_row != len(result['words'])+1: raise AssertionError('Contrato Excel incumplido.')
    check.close()
    return path
