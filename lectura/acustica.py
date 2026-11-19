"""Evidencia de habla, intensidad, tono y pausas."""

import math
from .utilidades import intersect, mask_spans, overlap, safe, stats, subtract, union


def acoustic_tracks(y, sr, cfg):
    import numpy as np, librosa, parselmouth
    from parselmouth.praat import call
    from silero_vad import load_silero_vad, get_speech_timestamps
    import torch
    duration = len(y)/sr; sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=cfg['pitch_floor'], pitch_ceiling=cfg['pitch_ceiling'])
    f0 = pitch.selected_array['frequency'].copy(); f0[f0 <= 0] = np.nan
    intensity = sound.to_intensity(minimum_pitch=100., time_step=.01); db = intensity.values[0].copy()
    grid = call(intensity, 'To TextGrid (silences)', -25., .15, .10, 'silent', 'sounding')
    pauses, sounding = [], []
    for k in range(1, int(call(grid, 'Get number of intervals', 1))+1):
        span = [float(call(grid, 'Get start time of interval', 1, k)), float(call(grid, 'Get end time of interval', 1, k))]
        (pauses if call(grid, 'Get label of interval', 1, k) == 'silent' else sounding).append(span)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000).astype('float32') if sr != 16000 else y.astype('float32')
    vad = load_silero_vad(onnx=False).eval()
    with torch.inference_mode():
        stamps = get_speech_timestamps(torch.from_numpy(y16), vad, sampling_rate=16000, threshold=.5,
            min_speech_duration_ms=100, min_silence_duration_ms=120, speech_pad_ms=0, return_seconds=False)
    speech = [[s['start']/16000, s['end']/16000] for s in stamps]
    vad_pauses = [[a[1], b[0]] for a, b in zip(speech[:-1], speech[1:]) if b[0]-a[1] >= .12]
    rms = librosa.feature.rms(y=y16, frame_length=320, hop_length=160, center=False)[0]
    rms_db = 20*np.log10(np.maximum(rms, 1e-12)); rt = (np.arange(len(rms))*160+160)/16000
    low = mask_spans(rt, rms_db < np.percentile(rms_db, 95)-30., .01, duration)
    low = [v for v in low if v[1]-v[0] >= .12]
    convergent = intersect(union(pauses+low), vad_pauses)
    hnr = sound.to_harmonicity_cc(time_step=.01, minimum_pitch=cfg['pitch_floor'])
    h = hnr.values[0].copy(); h[h <= -199.99] = np.nan
    tracks = {'duration': duration, 'f0_times': pitch.xs(), 'f0': f0, 'pitch_window_s': 3/cfg['pitch_floor'],
              'intensity_times': intensity.xs(), 'intensity': db, 'intensity_window_s': 3.2/100.,
              'hnr_times': hnr.xs(), 'hnr': h, 'sounding': sounding, 'speech': speech,
              'pauses': convergent, 'vad_pauses': vad_pauses, 'energy_pauses': union(pauses+low)}
    if cfg['pyin']:
        pf, _, _ = librosa.pyin(y16, sr=16000, fmin=cfg['pitch_floor'], fmax=cfg['pitch_ceiling'],
            frame_length=1024, hop_length=320, center=False, resolution=.2, n_thresholds=32, fill_na=np.nan)
        tracks.update(pyin=pf, pyin_times=(np.arange(len(pf))*320+512)/16000)
    if cfg['formants']:
        f = sound.to_formant_burg(time_step=.01, max_number_of_formants=5, maximum_formant=6500., window_length=.025)
        times = f.xs(); tracks['formant_times'] = times
        tracks['f1'] = np.array([f.get_value_at_time(1, t) for t in times])
        tracks['f2'] = np.array([f.get_value_at_time(2, t) for t in times])
    return tracks


def track_summary(tracks, a, b, clean):
    import numpy as np
    out = {}; supported_speech = intersect(clean, tracks['sounding'])
    def values(key, times, width):
        t = np.asarray(tracks.get(times, [])); v = np.asarray(tracks.get(key, []))
        mask = (t >= a+width/2) & (t <= b-width/2) & np.isfinite(v)
        inside = np.zeros(len(t), dtype=bool)
        for x, y in supported_speech: inside |= (t-width/2 >= x) & (t+width/2 <= y)
        return t, v, mask & inside
    t, f, mask = values('f0', 'f0_times', tracks['pitch_window_s'])
    s = stats(f[mask].tolist()); median = s['median']
    out.update(f0_median_hz=median, f0_range_st=12*math.log2(s['p90']/s['p10']) if median else None,
               f0_iqr_st=stats((12*np.log2(f[mask]/median)).tolist())['iqr'] if median else None,
               f0_support_s=float(mask.sum()*.01),
               f0_coverage_speech=min(1., mask.sum()*.01/overlap(a, b, supported_speech)) if overlap(a, b, supported_speech) else 0.)
    pairs = mask[1:] & mask[:-1]; delta = 12*np.log2(f[1:][pairs]/f[:-1][pairs])
    out['octave_jumps'] = int(((np.abs(np.abs(delta)-12) <= 2) | (np.abs(np.abs(delta)-24) <= 2)).sum())
    _, intensity, im = values('intensity', 'intensity_times', tracks['intensity_window_s'])
    its = stats(intensity[im].tolist()); reference = max(tracks['intensity']) if len(tracks['intensity']) else None
    out.update(intensity_relative_median_db=its['median']-reference if its['median'] is not None else None,
               intensity_range_db=its['p90']-its['p10'] if its['n'] else None)
    _, h, hm = values('hnr', 'hnr_times', 1/tracks.get('pitch_floor', 75.))
    out['hnr_median_db'] = stats(h[hm].tolist())['median']
    if 'pyin' in tracks and len(tracks['pyin_times']):
        yt = np.asarray(tracks['pyin_times']); yf = np.asarray(tracks['pyin']); indexes = np.searchsorted(yt, t).clip(0, len(yt)-1)
        left = np.maximum(0, indexes-1); indexes = np.where(abs(yt[left]-t) < abs(yt[indexes]-t), left, indexes)
        ym = mask & np.isfinite(yf[indexes]) & (yf[indexes] > 0) & (abs(yt[indexes]-t) <= .010001)
        for x, y in subtract([[a, b]], supported_speech): ym &= ~((yt[indexes]+.032 > x) & (yt[indexes]-.032 < y))
        cents = 1200*np.log2(f[ym]/yf[indexes][ym])
        out['pitch_disagreement_cents_median'] = stats(abs(cents).tolist())['median']
        out['pitch_octave_disagreement_fraction'] = float(((abs(abs(cents)-1200) <= 200) | (abs(abs(cents)-2400) <= 200)).mean()) if len(cents) else None
    for key in ('f1', 'f2'):
        if key in tracks:
            _, v, m = values(key, 'formant_times', .05)
            out[key+'_mixed_median_hz'] = stats(v[m & (v > 0)].tolist())['median']
    return safe(out)
