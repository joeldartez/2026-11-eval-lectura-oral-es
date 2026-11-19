"""Recorridos candidatos y evidencia por aparición de palabra."""

import statistics
from collections import Counter, defaultdict
from .texto import norm, phone_similarity, phone_units, text_similarity, tokens
from .utilidades import finite, overlap, union


def align_paths(reference, observations, cfg, prefix):
    # Beam por observación: cada arista consume audio. Permite saltos, retornos, parciales, fusiones e inversiones.
    n, m = len(reference), len(observations); rn = [r['norm'] for r in reference]; on = [o['norm'] for o in observations]
    exact = defaultdict(list)
    for i, word in enumerate(rn): exact[word].append(i)
    buckets = [[] for _ in range(m+1)]; buckets[0] = [(0., 0, ())]
    def prune(states):
        out, seen = [], Counter()
        for state in sorted(states, key=lambda s: (s[0], s[1], s[2])):
            key = state[1]
            if seen[key] < cfg['paths_per_model']:
                out.append(state); seen[key] += 1
                if len(out) >= cfg['beam']: break
        return out
    for j in range(m):
        layer = prune(buckets[j]); buckets[j] = []
        for cost, cursor, edges in layer:
            options = [(1, cursor, (), 'addition', .95)]
            starts = set(range(cursor, min(n, cursor+4)))
            anchors = [i for i in exact[on[j]] if max(0, cursor-cfg['backtrack']) <= i <= cursor+24]
            starts.update(sorted(anchors, key=lambda i: (abs(i-cursor), i))[:8])
            for i in starts:
                sim = text_similarity(on[j], rn[i])
                backward = i < cursor
                if backward and sim < .75: continue
                movement = .20 if i == cursor-1 else .50+.015*(cursor-i) if backward else .78*(i-cursor)
                partial = on[j] != rn[i] and len(on[j]) >= 2 and rn[i].startswith(on[j])
                lexical = 'match' if on[j] == rn[i] else 'partial' if partial else 'substitution'
                penalty = 0. if lexical == 'match' else .38 if partial else .55+.65*(1-sim)
                options.append((1, i if partial else i+1, (i,), lexical, penalty+movement))
                if not backward and i+1 < n and on[j] == rn[i]+rn[i+1]:
                    options.append((1, i+2, (i, i+1), 'merge', .16+movement))
                if j+1 < m and not backward:
                    if on[j]+on[j+1] == rn[i]: options.append((2, i+1, (i,), 'split', .16+movement))
                    if i+1 < n and on[j] == rn[i+1] and on[j+1] == rn[i]:
                        options.append((2, i+2, (i+1, i), 'inversion', .35+movement))
            for q, new_cursor, indices, operation, penalty in options:
                edge = (j, q, indices, operation)
                buckets[j+q].append((cost+penalty, new_cursor, edges+(edge,)))
        for future in range(j+1, min(m, j+2)+1):
            if len(buckets[future]) > cfg['beam']*16: buckets[future] = prune(buckets[future])
    finals = prune(buckets[m])[:cfg['paths_per_model']] if m else [(0., 0, ())]
    paths = []
    for rank, (cost, _, edges) in enumerate(finals):
        events, visits = [], Counter()
        for j, q, indices, operation in edges:
            group = observations[j:j+q]
            if operation == 'inversion': parts = [(group[k:k+1], (i,)) for k, i in enumerate(indices)]
            else: parts = [(group, indices)]
            for part, target in parts:
                for i in target: visits[i] += 1
                event = {'event_id': f'{prefix}:E{len(events)+1:05d}', 'target_indices': list(target),
                         'obs_ids': [o['obs_id'] for o in part], 'text': ' '.join(o['text'] for o in part),
                         'operation': operation, 'start': min((o['start'] for o in part if finite(o.get('start'))), default=None),
                         'end': max((o['end'] for o in part if finite(o.get('end'))), default=None),
                         'timing_source': part[0].get('timing_source', prefix), 'model': part[0]['model'],
                         'speaker_id': part[0].get('speaker_id'), 'visit': max((visits[i] for i in target), default=0),
                         'word_specific_time': len(target) == 1}
                if event['visit'] > 1: event['repeat'] = True
                events.append(event)
        if [x for e in events for x in e['obs_ids']] != [o['obs_id'] for o in observations]:
            raise AssertionError('La alineación perdió o duplicó observaciones.')
        paths.append({'path_id': f'{prefix}:P{rank+1}', 'cost': cost, 'events': events, 'model': prefix.split(':')[0],
                      'speaker_id': observations[0].get('speaker_id') if observations else None,
                      'coverage': len(visits)/max(n, 1)})
    return paths


def candidate_paths(reference, evidence, cfg):
    by_speaker = Counter()
    targets = {w['norm'] for w in reference}
    for result in evidence['asr'].values():
        for o in result['observations']:
            if o.get('speaker_id') is not None: by_speaker[o['speaker_id']] += 1+(o['norm'] in targets)*2
    speakers = [cfg['student_speaker']] if cfg.get('student_speaker') else [s for s, _ in by_speaker.most_common(3)]
    if not speakers: speakers = [None]
    paths = []
    for name, result in evidence['asr'].items():
        for speaker in speakers:
            obs = [o for o in result['observations'] if o.get('speaker_id') == speaker]
            paths.extend(align_paths(reference, obs, cfg, f'{name}:{speaker}'))
    primary = cfg['asr_models'][0]
    preferred = [p for p in paths if p['model'] == primary and p['speaker_id'] == speakers[0]]
    local = min(preferred or paths, key=lambda p: p['cost'])
    return paths, local


def path_occurrences(path, reference):
    by_word = defaultdict(list)
    for event in path['events']:
        for i in event['target_indices']: by_word[i].append(event)
    return by_word


def word_evidence(reference, path, evidence, cfg, alternatives):
    rows, visits = [], path_occurrences(path, reference)
    peer_maps = [(p, path_occurrences(p, reference)) for p in alternatives if p['speaker_id'] == path['speaker_id']]
    diar = evidence['diar']; reader = union([[s['start'], s['end']] for s in diar['segments'] if s['speaker_id'] == path['speaker_id']])
    for r in reference:
        events = visits[r['index']]; event = events[-1] if events else None
        row = dict(r, attempt_present=int(bool(events)), attempts=len(events), repeats=max(0, len(events)-1),
                   inversion=int(any(e['operation'] == 'inversion' for e in events)), event_id=event['event_id'] if event else None,
                   start=event['start'] if event else None, end=event['end'] if event else None,
                   word_specific_time=bool(event and event['word_specific_time']), timing_source=event['timing_source'] if event else None)
        hypotheses, matches, asr_sources = {}, {}, {}
        for name in evidence['asr']:
            peers = [(p, v) for p, v in peer_maps if p['model'] == name]
            same = [p for p, _ in peers if p['path_id'] == path['path_id']]
            chosen = same[0] if same else min((p for p, _ in peers), key=lambda p: p['cost'], default=None)
            chosen_map = next((v for p, v in peers if chosen and p['path_id'] == chosen['path_id']), {})
            attempts = chosen_map.get(r['index'], [])
            text = attempts[-1]['text'] if attempts else ''
            matched = bool(attempts and attempts[-1]['operation'] in ('match', 'merge', 'split', 'inversion')
                           and (attempts[-1]['operation'] != 'split' or norm(''.join(tokens(text))) == r['norm']))
            hypotheses[name] = text; matches[name] = int(matched); asr_sources[name] = chosen['path_id'] if chosen else ''
        row['hypotheses'], row['asr_matches'], row['asr_sources'] = hypotheses, matches, asr_sources
        row['asr_match_count'] = sum(matches.values()); row['asr_observed_count'] = sum(bool(v) for v in hypotheses.values())
        row['phone_observed'], row['phone_similarity'], row['phone_posterior'] = {}, {}, {}
        for name, phones in evidence['phones'].items():
            selected = []
            if event and finite(row['start']) and finite(row['end']) and row['end'] > row['start']:
                selected = [p for p in phones if row['start'] <= (p['start']+p['end'])/2 < row['end']
                            and overlap(p['start'], p['end'], reader) >= .5*(p['end']-p['start'])]
            units = phone_units(''.join(p['phone'] for p in selected))
            row['phone_observed'][name] = ''.join(units)
            row['phone_similarity'][name] = phone_similarity(r['phones'], units) if row['word_specific_time'] else None
            row['phone_posterior'][name] = statistics.mean(p['posterior'] for p in selected) if selected else None
        similarities = [v for v in row['phone_similarity'].values() if finite(v)]
        row['phone_mean'] = statistics.mean(similarities) if similarities else None
        all_agree = len(matches) > 1 and all(matches.values())
        some_match = bool(row['asr_match_count']); compatible = bool(similarities and max(similarities) >= cfg['compatible_phone_similarity'])
        local_credit = int(bool(events) and (all_agree or some_match or compatible and row['asr_observed_count'] > 0))
        category = 'correct' if local_credit else 'no_attempt' if not events else 'partial' if event['operation'] == 'partial' else 'substitution'
        row.update(local_correct=local_credit, local_category=category, final_correct=local_credit, final_category=category,
                   decision_method='rule_asr_agreement' if all_agree else 'rule_no_attempt' if not events else 'rule_evidence',
                   support_score=1. if all_agree else .85 if some_match and compatible else .65 if some_match else .45,
                   llm_applied=0, llm_changed=0, assignment_changed=0, llm_reason='')
        row['first_local_correct'] = int(bool(events) and events[0]['operation'] in ('match', 'merge', 'split', 'inversion'))
        row['first_final_correct'] = row['first_local_correct']
        # Incluye ausencia disputada y fonética mixta. Un acuerdo léxico pleno no necesita arbitraje redundante.
        row['mixed'] = bool(not all_agree and (row['asr_observed_count'] > 0 or events))
        rows.append(row)
    return rows
