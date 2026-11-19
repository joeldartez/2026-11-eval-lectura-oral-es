"""Indicadores de precisión, velocidad y ritmo."""

import statistics
from collections import Counter
from .acustica import track_summary
from .texto import tokens
from .utilidades import clip, finite, overlap, safe, seconds, stats, subtract, union


def reading_metrics(reference, rows, path, evidence, cfg, final=False):
    occurrences = [(i, e) for e in path['events'] for i in e['target_indices']]
    timed = [(i, e) for i, e in occurrences if finite(e.get('start')) and finite(e.get('end')) and e['end'] > e['start']]
    bounds = cfg.get('reading_bounds')
    if bounds is not None:
        if not (len(bounds) == 2 and 0 <= bounds[0] < bounds[1] <= evidence['duration']): raise ValueError('Límites de lectura inválidos.')
        a, b = bounds
    elif timed: a, b = min(e['start'] for _, e in timed), max(e['end'] for _, e in timed)
    else: a, b = 0., evidence['duration']
    elapsed = b-a
    speaker = path['speaker_id']; diar = evidence['diar']
    reader = union([[s['start'], s['end']] for s in diar['segments'] if s['speaker_id'] == speaker])
    external = union([[s['start'], s['end']] for s in diar['segments'] if s['speaker_id'] != speaker])
    clean = subtract(reader, external+diar.get('overlap', []))
    pauses = clip(subtract(evidence['acoustic']['pauses'], external), a, b)
    ordered = sorted(timed, key=lambda pair: (pair[1]['start'], pair[1]['end'], pair[0]))
    primary = {}; occurrence_visits = Counter()
    for i, e in ordered:
        primary.setdefault(i, e); occurrence_visits[i] += 1
    key = 'final_correct' if final else 'local_correct'; first_key = 'first_final_correct' if final else 'first_local_correct'
    correct = sum(r[key] for r in rows); correct_first = sum(r[first_key] for r in rows)
    categories = Counter(r['final_category' if final else 'local_category'] for r in rows)
    phonetics = [r['phone_mean'] for r in rows if finite(r['phone_mean'])]
    repeated = sum(max(0, n-1) for n in occurrence_visits.values())
    additions = sum(max(1, len(tokens(e['text']))) for e in path['events'] if not e['target_indices'])
    inversion_positions = {i for e in path['events'] if e['operation'] == 'inversion' for i in e['target_indices']}
    result = {'expected_words': len(reference), 'attempted_positions': sum(r['attempt_present'] for r in rows),
              'correct_estimated': correct, 'correct_first_estimated': correct_first, 'accuracy_fraction': correct/len(reference),
              'incorrect_estimated': len(reference)-correct, 'no_attempt_positions': sum(not r['attempt_present'] for r in rows),
              'substitutions': categories['substitution'], 'omissions': categories['omission']+categories['no_attempt'],
              'partials': categories['partial'], 'additions': additions, 'repetitions': repeated,
              'inversion_positions': len(inversion_positions), 'coverage_fraction': len(primary)/len(reference),
              'reading_start_s': a, 'reading_end_s': b, 'reading_duration_s': elapsed,
              'pcpm': 60*correct/elapsed if elapsed else 0., 'pcpm_first': 60*correct_first/elapsed if elapsed else 0.,
              'advance_wpm': 60*len(primary)/elapsed if elapsed else 0., 'production_wpm': 60*len(timed)/elapsed if elapsed else 0.,
              'pause_s': seconds(pauses), 'pause_fraction': seconds(pauses)/elapsed if elapsed else 0.,
              'pause_n': len(pauses), 'pause_400ms_n': sum(y-x >= .4 for x, y in pauses),
              'pause_600ms_n': sum(y-x >= .6 for x, y in pauses),
              'pause_400ms_per_min': 60*sum(y-x >= .4 for x, y in pauses)/elapsed if elapsed else 0.,
              'external_voice_s': overlap(a, b, external), 'overlap_s': overlap(a, b, diar.get('overlap', [])),
              'phone_mean': statistics.mean(phonetics) if phonetics else None,
              'phone_evaluable_fraction': len(phonetics)/len(reference),
              'asr_full_agreement_fraction': sum(len(r['asr_matches']) > 1 and all(r['asr_matches'].values()) for r in rows)/len(rows),
              'estimated_not_calibrated': 1}
    boundaries = []
    for i, left in enumerate(reference[:-1]):
        first, second = primary.get(i), primary.get(i+1)
        valid = bool(first and second and first['word_specific_time'] and second['word_specific_time'] and first['event_id'] != second['event_id'])
        gap = second['start']-first['end'] if valid else None
        boundaries.append({'boundary_id': f'B{i+1:04d}', 'left_index': i, 'right_index': i+1, 'type': left['boundary'],
                           'observed': int(valid), 'gap_s': gap,
                           'acoustic_pause_s': overlap(first['end'], second['start'], pauses) if valid and gap > 0 else 0. if valid else None,
                           'repetition_between': int(bool(valid and any(e.get('repeat') and first['end'] <= e['start'] < second['start'] for _, e in timed)))})
    for category in ('interior', 'comma', 'period'):
        group = [r for r in boundaries if r['type'] == category]; gaps = [r['gap_s'] for r in group if r['observed']]
        gs = stats(gaps)
        result.update({f'boundary_{category}_n': len(gaps), f'boundary_{category}_opportunities': len(group),
                       f'boundary_{category}_coverage': len(gaps)/len(group) if group else 0.,
                       f'boundary_{category}_median_s': gs['median'], f'boundary_{category}_p90_s': gs['p90'],
                       f'boundary_{category}_200ms_n': sum(v > .2 for v in gaps),
                       f'boundary_{category}_400ms_n': sum(v > .4 for v in gaps),
                       f'boundary_{category}_400ms_fraction': sum(v > .4 for v in gaps)/len(gaps) if gaps else None})
    for category in ('comma', 'period'):
        x, y = result[f'boundary_{category}_median_s'], result['boundary_interior_median_s']
        result[f'{category}_minus_interior_median_s'] = x-y if x is not None and y is not None else None
    segments = []
    for k in range(3):
        x, y = a+elapsed*k/3, a+elapsed*(k+1)/3
        selected = [(i, e) for i, e in timed if x <= (e['start']+e['end'])/2 < y]
        advance = sum(x <= (e['start']+e['end'])/2 < y for e in primary.values())
        row = {'kind': 'time_third', 'segment_id': k+1, 'start_s': x, 'end_s': y,
               'advance_wpm': 60*advance/(y-x) if y > x else 0., 'production_wpm': 60*len(selected)/(y-x) if y > x else 0.,
               'pause_s': overlap(x, y, pauses), 'positions': advance}
        row.update(track_summary(evidence['acoustic'], x, y, clean)); segments.append(row)
        for field in ('advance_wpm', 'production_wpm', 'pause_s', 'f0_median_hz', 'f0_range_st', 'intensity_relative_median_db'):
            result[f'time_third_{k+1}_{field}'] = row.get(field)
    first, last = result['time_third_1_advance_wpm'], result['time_third_3_advance_wpm']
    result['advance_end_start_ratio'] = last/first if first else None
    result['advance_end_minus_start_wpm'] = last-first
    groups = [('sentence', sid, [w['index'] for w in reference if w['sentence_id'] == sid]) for sid in sorted({w['sentence_id'] for w in reference})]
    for k in range(3):
        lo, hi = round(len(reference)*k/3), round(len(reference)*(k+1)/3)
        groups.append(('text_third', k+1, list(range(lo, hi))))
    for kind, sid, indexes in groups:
        group = [primary[i] for i in indexes if i in primary]
        x, y = (min(e['start'] for e in group), max(e['end'] for e in group)) if group else (None, None)
        row = {'kind': kind, 'segment_id': sid, 'start_s': x, 'end_s': y, 'expected_positions': len(indexes),
               'positions': len(group), 'coverage': len(group)/len(indexes) if indexes else 0.,
               'advance_wpm': 60*len(group)/(y-x) if x is not None and y > x else 0.,
               'pause_s': overlap(x, y, pauses) if x is not None else 0.}
        if x is not None and y > x: row.update(track_summary(evidence['acoustic'], x, y, clean))
        segments.append(row)
        if kind == 'text_third':
            for field in ('advance_wpm', 'coverage', 'pause_s'): result[f'text_third_{sid}_{field}'] = row[field]
    result.update(track_summary(evidence['acoustic'], a, b, clean))
    for name in cfg['phone_models']:
        vals = [r['phone_similarity'].get(name) for r in rows if finite(r['phone_similarity'].get(name))]
        result[name+'_similarity_mean'] = statistics.mean(vals) if vals else None
        result[name+'_evaluable_fraction'] = len(vals)/len(rows)
    # Códigos locales reproducibles, sin prosa del LLM ni puntuación compuesta arbitraria.
    result['feedback_omissions'] = int(result['omissions']/len(rows) >= .05)
    result['feedback_substitutions'] = int(result['substitutions']/len(rows) >= .05)
    result['feedback_repetitions'] = int(repeated/len(rows) >= .03)
    result['feedback_fragmentation'] = int((result['boundary_interior_400ms_fraction'] or 0.) >= .08)
    result['feedback_deceleration'] = int(result['advance_end_start_ratio'] is not None and result['advance_end_start_ratio'] < .85)
    result['feedback_continuity_preserved'] = int(not result['feedback_fragmentation'] and not result['feedback_deceleration'])
    return safe(result), safe(boundaries), safe(segments)
