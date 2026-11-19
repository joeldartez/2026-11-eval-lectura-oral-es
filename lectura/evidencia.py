"""Extracción y caché de evidencia por audio."""

import json
import time
from pathlib import Path
from .acustica import acoustic_tracks
from .inferencia import attribute, infer_asr, infer_diarization, infer_phones, windows_for
from .procedencia import protocol_hash
from .reconciliacion import repair_asr_times
from .utilidades import digest, environment_info, file_digest, safe


def collect_evidence(audio_path, cfg, pool, cache_root, logs):
    import numpy as np, soundfile as sf, librosa
    from transformers.utils import logging as hf_logging
    from huggingface_hub.utils import disable_progress_bars
    hf_logging.disable_progress_bar(); disable_progress_bars()
    y, sr = sf.read(audio_path, dtype='float64', always_2d=True); y = y.mean(axis=1)
    if not len(y) or not np.isfinite(y).all(): raise ValueError('Audio vacío o no finito.')
    duration = len(y)/sr; y16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype('float32') if sr != 16000 else y.astype('float32')
    windows = windows_for(duration, cfg); audio_sha = file_digest(audio_path)
    inference_sha = protocol_hash(inference=True)
    def stage(name, settings, fn):
        started = time.perf_counter(); key = digest([audio_sha, name, settings, inference_sha, environment_info(), pool.device, 'short-inference-1'])
        path = Path(cache_root) / (name+'-'+key+'.json'); hit = cfg['inference_cache'] and path.is_file()
        if hit: result = json.loads(path.read_text(encoding='utf-8'))
        else:
            result = safe(fn())
            if cfg['inference_cache']:
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        logs.append({'stage': name, 'seconds': time.perf_counter()-started, 'cache_hit': int(hit)})
        print('{}: {:.1f}s{}'.format(name, logs[-1]['seconds'], ' [caché]' if hit else ''))
        return result
    acoustic = stage('acoustic', {k: cfg[k] for k in ('pitch_floor', 'pitch_ceiling', 'pyin', 'formants')}, lambda: acoustic_tracks(y, sr, cfg))
    if cfg['diarization']:
        diar = stage('diarization', pool.pin('diarizer'), lambda: infer_diarization(y16, 16000, pool))
    else:
        diar = {'segments': [{'start': 0., 'end': duration, 'speaker_id': 'SPEAKER_00'}], 'overlap': []}
    asr = {}
    for name in cfg['asr_models']:
        settings = [pool.pin(name), pool.pin('qwen_aligner') if name != 'parakeet' else None, windows]
        result = stage('asr-'+name, settings, lambda name=name: infer_asr(name, y16, 16000, windows, pool))
        asr[name] = result
    repair_asr_times(asr)
    for result in asr.values(): result['observations'] = attribute(result['observations'], diar)
    phones = {}
    for name in cfg['phone_models']:
        result = stage('phones-'+name, [pool.pin(name), windows], lambda name=name: infer_phones(name, y16, 16000, windows, pool))
        phones[name] = result
    return {'duration': duration, 'sr': sr, 'audio_sha': audio_sha, 'asr': asr, 'phones': phones, 'diar': diar, 'acoustic': acoustic}
