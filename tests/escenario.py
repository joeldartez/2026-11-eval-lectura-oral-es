"""Escenario sintético: dos voces, repetición y discrepancia entre ASR."""
import copy


def sample():
    """Simular repeticiones, intento parcial, desacuerdo ASR y una segunda voz."""
    from lectura.config import CFG
    words = [('La', 'la'), ('gallina', 'gaʝina'), ('canta', 'kanta'), ('El', 'el'), ('niño', 'niɲo'), ('corre', 'kore')]
    reference = [{'index': i, 'word_id': f'W{i+1:04d}', 'text': text, 'norm': text.lower(),
                  'sentence_id': 1 if i < 3 else 2, 'boundary': 'period' if i in (2, 5) else 'interior',
                  'ipa': ipa, 'phones': list(ipa)} for i, (text, ipa) in enumerate(words)]
    productions = {
        'qwen': [('La', .2, .5), ('gallina', .6, 1.1), ('gallina', 1.5, 1.9), ('can', 2., 2.2),
                 ('canta', 2.3, 2.6), ('El', 3., 3.2), ('niño', 3.3, 3.6), ('camina', 4., 4.4), ('extra', 4.5, 4.8)],
        'parakeet': [('La', .21, .51), ('gallina', .61, 1.11), ('canta', 2.31, 2.61),
                     ('El', 3.01, 3.21), ('niño', 3.31, 3.61), ('corre', 4.01, 4.41)],
    }
    asr = {}
    for name, events in productions.items():
        observations = [{'text': text, 'norm': text.lower(), 'start': a, 'end': b, 'timing_valid': True,
                         'timing_source': name, 'model': name, 'obs_id': f'{name}:O{i+1:05d}',
                         'speaker_id': 'SPEAKER_00', 'speaker_fraction': 1.} for i, (text, a, b) in enumerate(events)]
        observations.append({'text': 'Lee', 'norm': 'lee', 'start': 6.2, 'end': 6.6, 'timing_valid': True,
                             'timing_source': name, 'model': name, 'obs_id': f'{name}:O99999',
                             'speaker_id': 'SPEAKER_01', 'speaker_fraction': 1.})
        asr[name] = {'text': ' '.join(o['text'] for o in observations), 'observations': observations}
    phones = {}
    for name in ('cnam', 'multilingual'):
        events = []
        for word, (_, a, b) in zip(reference, productions['parakeet']):
            for i, phone in enumerate(word['phones']):
                events.append({'phone': phone, 'start': a+(b-a)*i/len(word['phones']),
                               'end': a+(b-a)*(i+1)/len(word['phones']), 'posterior': .8, 'model': name})
        phones[name] = events
    times = [(i+.5)*.01 for i in range(900)]
    acoustic = {'duration': 9., 'f0_times': times, 'f0': [180.+i % 9 for i in range(900)],
                'pitch_window_s': .04, 'intensity_times': times, 'intensity': [60.+i % 5 for i in range(900)],
                'intensity_window_s': .032, 'hnr_times': times, 'hnr': [15.]*900,
                'sounding': [[.1, 6.], [6.1, 7.]], 'speech': [[.1, 6.], [6.1, 7.]], 'pauses': [[1.2, 1.5], [2.7, 3.]],
                'speech_available': True, 'pauses_available': True,
                'pyin_times': times, 'pyin': [185.+i % 9 for i in range(900)],
                'formant_times': times, 'f1': [500.]*900, 'f2': [1500.]*900}
    evidence = {'duration': 9., 'sr': 16000, 'audio_sha': 'synthetic', 'asr': asr, 'phones': phones,
                'diar': {'segments': [{'start': .1, 'end': 6., 'speaker_id': 'SPEAKER_00'},
                                      {'start': 6.1, 'end': 7., 'speaker_id': 'SPEAKER_01'}], 'overlap': []},
                'acoustic': acoustic}
    cfg = copy.deepcopy(CFG)
    cfg.update(use_llm=False, llm_words_per_call=2, llm_max_word_calls=3)
    return reference, evidence, cfg


class RecordedArbiter:
    """Producir decisiones reproducibles, incluidas reasignaciones y fallos de servicio."""

    def __init__(self, fail=False):
        """Configurar respuestas locales sin crear un cliente HTTP."""
        self.fail = fail
        self.client = self

    def call(self, task, payload, schema):
        """Seleccionar candidatos existentes y forzar decisiones distintas a las reglas."""
        if self.fail: raise TimeoutError('fallo simulado')
        if task == 'reading_path':
            chosen = next(p for p in payload['candidates'] if p['path_id'].startswith('parakeet:SPEAKER_00:'))
            answer = {'path_id': chosen['path_id'], 'reason': 'CONTEXT_SEQUENCE'}
        else:
            decisions = []
            for row in payload['words']:
                events = row['candidate_events']
                correct = int(not row['local_correct'] and bool(events))
                decisions.append({'word_id': row['word_id'], 'correct_first': correct, 'correct_final': correct,
                                  'category': 'correct' if correct else 'substitution' if events else 'no_attempt',
                                  'event_id': events[-1]['event_id'] if events else 'NONE', 'support': .7,
                                  'reason': 'CONTEXT_RESOLUTION'})
            answer = {'words': decisions}
        return answer, {'task': task, 'status': 'ok', 'seconds': 0., 'cache_hit': 1,
                        'input_tokens': 10, 'output_tokens': 5, 'usd_new': 0.}
