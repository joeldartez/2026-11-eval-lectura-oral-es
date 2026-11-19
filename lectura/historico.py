"""Lectura opcional de evidencia de ejecuciones anteriores."""

import copy
import json
from pathlib import Path
from .acustica import acoustic_tracks
from .config import CFG, ROOT
from .texto import norm
from .utilidades import file_digest, finite, safe


def load_legacy_evidence(run_dir, config=None, audio_dir=ROOT / 'audios'):
    cfg = CFG if config is None else config
    run_dir = Path(run_dir); d = json.loads((run_dir / 'registro-completo.json').read_text(encoding='utf-8'))
    asr = {}
    for name in cfg['asr_models']:
        source = copy.deepcopy(d['asr'][name]); observations = source['observations']
        for i, o in enumerate(observations):
            o.update(norm=norm(o['text']), model=name, obs_id=f'{name}:O{i+1:05d}', timing_source=name,
                     timing_valid=bool(finite(o.get('start')) and finite(o.get('end')) and o['end'] > o['start']))
        asr[name] = {'text': source['text'], 'observations': observations}
    phones = {name: [{'phone': p['phone'], 'start': p['start'], 'end': p['end'],
                      'posterior': p.get('posterior', 0.), 'model': name} for p in d['phones_initial'][name]['events']] for name in cfg['phone_models']}
    turns = d['turns']; diar = {'segments': [{'start': t['start'], 'end': t['end'], 'speaker_id': t['speaker_id']} for t in turns], 'overlap': []}
    name = run_dir.parent.name+'.ogg'; import soundfile as sf
    y, sr = sf.read(Path(audio_dir) / name, always_2d=True); y = y.mean(axis=1)
    return {'duration': len(y)/sr, 'sr': sr, 'audio_sha': file_digest(Path(audio_dir) / name),
            'asr': asr, 'phones': phones, 'diar': diar, 'acoustic': safe(acoustic_tracks(y, sr, cfg)),
            'model_revisions': {k: v.get('revision') for k, v in d['models'].items()}, 'legacy_replay': True}
