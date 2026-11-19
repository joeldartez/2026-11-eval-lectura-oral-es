"""Reparación de tiempos y materialización del recorrido final."""

import copy
from collections import Counter, defaultdict
from .alineacion import word_evidence
from .utilidades import finite


def repair_asr_times(asr):
    # Reparar a partir de otra observación acústica sin forzar el pasaje objetivo.
    for name, result in asr.items():
        for word in result['observations']:
            if word.get('timing_valid'): continue
            point = word.get('ownership_time', word.get('start'))
            peers = [p for other, source in asr.items() if other != name for p in source['observations']
                     if p.get('timing_valid') and p['norm'] == word['norm'] and finite(point)
                     and abs((p['start']+p['end'])/2-point) <= .6]
            if peers:
                donor = min(peers, key=lambda p: abs((p['start']+p['end'])/2-point))
                word.update(start=donor['start'], end=donor['end'], timing_valid=True,
                            timing_source=donor['model']+'_observed_donor')


def materialize_path(path, rows, candidates):
    effective = copy.deepcopy(path)
    bank = {e['event_id']: e for p in candidates for e in p['events']}
    original = defaultdict(list)
    for e in effective['events']:
        for index in e['target_indices']: original[index].append(e)
    for row in rows:
        previous = original[row['index']][-1] if original[row['index']] else None
        candidate = bank.get(row['event_id'])
        row['candidate_changed'] = int(bool(candidate) and (not previous or candidate['event_id'] != previous['event_id']))
        if previous and candidate and (previous['model'], previous['obs_ids'], previous['target_indices'], previous['start'], previous['end']) == (candidate['model'], candidate['obs_ids'], candidate['target_indices'], candidate['start'], candidate['end']):
            row['assignment_changed'] = 0; row['event_id'] = previous['event_id']
        if not row['assignment_changed'] or row['event_id'] not in bank: continue
        index, chosen = row['index'], bank[row['event_id']]
        old = original[index][-1] if original[index] else None
        collision = any(set(chosen['obs_ids']) & set(e['obs_ids']) and
                        index not in e['target_indices'] and e['target_indices'] for e in effective['events'])
        if collision:
            row.update(final_correct=row['local_correct'], first_final_correct=row['first_local_correct'],
                       final_category=row['local_category'], assignment_changed=0, decision_method='rule_assignment_collision',
                       event_id=old['event_id'] if old else None, start=old['start'] if old else None,
                       end=old['end'] if old else None, attempt_present=int(bool(old)))
            row['llm_changed'] = 0
            continue
        if old:
            old['target_indices'].remove(index)
            if not old['target_indices']: effective['events'].remove(old)
        existing = next((e for e in effective['events'] if e['event_id'] == chosen['event_id']), None)
        if existing:
            if index not in existing['target_indices']: existing['target_indices'].append(index)
        else:
            new = copy.deepcopy(chosen); new['target_indices'] = [index]; effective['events'].append(new)
    effective['events'].sort(key=lambda e: (e['start'] if finite(e['start']) else float('inf'), e['event_id']))
    seen = Counter()
    for event in effective['events']:
        event['word_specific_time'] = bool(len(event['target_indices']) == 1 and finite(event['start'])
                                          and finite(event['end']) and event['end'] > event['start'])
        event['repeat'] = any(seen[i] > 0 for i in event['target_indices'])
        for i in event['target_indices']: seen[i] += 1
    return effective


def refresh_word_measurements(rows, reference, path, evidence, cfg, candidates):
    refreshed = word_evidence(reference, path, evidence, cfg, candidates)
    fields = ('attempt_present', 'attempts', 'repeats', 'phone_observed', 'phone_similarity', 'phone_posterior', 'phone_mean')
    for row, fresh in zip(rows, refreshed):
        for field in fields: row[field] = fresh[field]
        if not row['attempt_present']:
            row['final_correct'] = row['first_final_correct'] = 0; row['final_category'] = 'no_attempt'
        row['llm_changed'] = int(row['llm_applied'] and row['final_correct'] != row['local_correct'])
    return rows
