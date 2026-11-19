"""Transcripción, alineamiento temporal, fonemas y voces."""

import math
import re
from .modelos import inputs_to_device
from .texto import norm, tokens
from .utilidades import finite, mask_spans, overlap, union


def windows_for(duration, cfg):
    result, start = [], 0.
    while start < duration:
        end = min(duration, start+cfg['window_s'])
        result.append({'core_start': start, 'core_end': end, 'start': max(0., start-cfg['context_s']),
                       'end': min(duration, end+cfg['context_s'])})
        start = end
    return result


def owned(record, window):
    a, b = record.get('start'), record.get('end')
    if not finite(a) or not finite(b) or b <= a: return False
    return window['core_start'] <= (a+b)/2 < window['core_end']


def timestamp_words(words, offset, duration, source):
    out = []
    for w in words:
        a, b = w.get('start'), w.get('end')
        valid = finite(a) and finite(b) and 0 <= a < b <= duration+0.02
        out.append({'text': w['text'], 'norm': norm(w['text']), 'start': float(a)+offset if finite(a) else None,
                    'end': float(b)+offset if finite(b) else None, 'timing_valid': bool(valid), 'timing_source': source})
    return out


def qwen_align(x, sr, text, pool):
    import torch
    b = pool.get('qwen_aligner'); model, processor = b['model'], b['processor']
    maximum = (model.config.num_labels-1)*processor.timestamp_segment_time/1000
    if len(x)/sr > maximum: raise ValueError('Recorte superior al contexto temporal del alineador Qwen.')
    batch, lists = processor.prepare_forced_aligner_inputs(audio=x, transcript=text, language='Spanish',
        processor_kwargs={'audio_kwargs': {'sampling_rate': sr, 'truncation': False}, 'text_kwargs': {'truncation': False}})
    batch = inputs_to_device(batch, model)
    with torch.inference_mode(): output = model(**batch)
    words = processor.decode_forced_alignment(logits=output.logits, input_ids=batch['input_ids'],
        word_lists=lists, timestamp_token_id=model.config.timestamp_token_id)[0]
    return [{'text': w['text'], 'start': float(w['start_time']), 'end': float(w['end_time'])} for w in words]


def infer_asr(name, y, sr, windows, pool):
    import torch
    records = []
    for window in windows:
        offset = round(window['start']*sr)/sr
        x = y[round(window['start']*sr):round(window['end']*sr)]
        bundle = pool.get(name); model, processor = bundle['model'], bundle['processor']
        with torch.inference_mode():
            if name == 'qwen':
                batch = processor.apply_transcription_request(audio=x, language='Spanish',
                    processor_kwargs={'audio_kwargs': {'sampling_rate': sr, 'truncation': False}, 'text_kwargs': {'truncation': False}})
                batch = inputs_to_device(batch, model); prefix = batch['input_ids'].shape[1]
                bound = min(1024, max(128, math.ceil(len(x)/sr*24)), model.config.text_config.max_position_embeddings-prefix)
                if bound <= 0: raise RuntimeError('Qwen agotó el contexto.')
                output = model.generate(**batch, max_new_tokens=bound, do_sample=False, num_beams=1)
                ids = output[0, prefix:].cpu().tolist()
                raw = processor.decode(ids, return_format='raw', skip_special_tokens=True, clean_up_tokenization_spaces=False)
                text = raw.split('assistant\n', 1)[-1].split('<asr_text>', 1)[-1].strip()
                native_words = None
            elif name == 'parakeet':
                batch = inputs_to_device(processor(x, sampling_rate=sr, return_tensors='pt', truncation=False), model)
                valid = int(model.encoder._get_subsampling_output_length(batch['attention_mask'].sum(-1)).item())
                bound = max(64, valid*int(model.config.max_symbols_per_step))
                output = model.generate(**batch, max_new_tokens=bound, do_sample=False, num_beams=1, return_dict_in_generate=True)
                texts, batches = processor.decode(output.sequences, durations=output.durations, skip_special_tokens=True)
                text, ts = texts[0], batches[0]
                if sum(output.durations[0].cpu().tolist()) < valid: raise RuntimeError('Parakeet no consumió toda la ventana.')
                stream, ranges, cursor = ''.join(t['token'] for t in ts), [], 0
                for t in ts: ranges.append((cursor, cursor+len(t['token']))); cursor += len(t['token'])
                native_words = []
                for match in re.finditer(r'[^\W\d_]+', stream):
                    group = [t for t, (a, b) in zip(ts, ranges) if a < match.end() and b > match.start()]
                    native_words.append({'text': match.group(), 'start': min(t['start'] for t in group), 'end': max(t['end'] for t in group)})
                if [norm(w['text']) for w in native_words] != [norm(w) for w in tokens(text)]:
                    raise RuntimeError('La segmentación nativa Parakeet cambió las palabras.')
            elif name == 'whisper':
                batch = inputs_to_device(processor(x, sampling_rate=sr, return_tensors='pt', return_attention_mask=True), model)
                bound = min(384, max(96, math.ceil(len(x)/sr*16)))
                output = model.generate(**batch, language='spanish', task='transcribe', do_sample=False, num_beams=1, max_new_tokens=bound)
                ids = output[0].cpu().tolist(); text = processor.decode(ids, skip_special_tokens=True).strip(); native_words = None
            else: raise ValueError(name)
        if native_words is None:
            native_words = qwen_align(x, sr, text, pool) if tokens(text) else []
        records.append((window, timestamp_words(native_words, offset, len(x)/sr, name)))
    return reconcile_windows(records, name)


def reconcile_windows(records, name):
    selected = []
    for window, rows in records:
        for i, word in enumerate(rows):
            if not word['timing_valid']:
                neighbours = [w for w in rows[max(0, i-1):i+2] if finite(w.get('start'))]
                word['ownership_time'] = word['start'] if finite(word.get('start')) else (neighbours[0]['start'] if neighbours else window['core_start'])
        current = [dict(w, model=name) for w in rows if owned(w, window) or
                   not w['timing_valid'] and window['core_start'] <= w['ownership_time'] < window['core_end']]
        if selected and current:
            while current and selected and current[0]['norm'] == selected[-1]['norm']:
                a, b = current[0], selected[-1]
                if abs(a['start']-b['start']) <= 0.08 and abs(a['end']-b['end']) <= 0.08: current.pop(0)
                else: break
        selected.extend(current)
    for i, w in enumerate(selected): w['obs_id'] = f'{name}:O{i+1:05d}'
    return {'text': ' '.join(w['text'] for w in selected), 'observations': selected}


def infer_phones(name, y, sr, windows, pool):
    import numpy as np, torch
    bundle = pool.get(name); model, processor, tokenizer = bundle['model'], bundle['processor'], bundle['tokenizer']
    vocab = {v: k for k, v in tokenizer.get_vocab().items()}; blank = model.config.pad_token_id
    special = set(tokenizer.all_special_ids); delimiter = getattr(tokenizer, 'word_delimiter_token_id', None)
    stride, receptive = 1, 1
    for kernel, hop in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (kernel-1)*stride; stride *= hop
    if getattr(model.config, 'add_adapter', False): stride *= model.config.adapter_stride**model.config.num_adapter_layers
    events = []
    for window in windows:
        i, j = round(window['start']*sr), round(window['end']*sr); x = y[i:j]; offset = i/sr
        batch = inputs_to_device(processor(x, sampling_rate=sr, return_tensors='pt', return_attention_mask=True, truncation=False), model)
        with torch.inference_mode():
            logits = model(**batch).logits[0]; valid = int(model._get_feat_extract_output_lengths(len(x)).item())
            probs = logits[:valid].float().softmax(-1).cpu().numpy()
        if not np.isfinite(probs).all(): raise RuntimeError('Emisiones CTC no finitas.')
        ids = probs.argmax(-1); edges = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1, len(ids)]
        for a, b in zip(edges[:-1], edges[1:]):
            token_id = int(ids[a])
            if token_id == blank or token_id in special or token_id == delimiter: continue
            event = {'phone': vocab[token_id], 'start': max(offset, offset+((receptive-1)/2+(a-.5)*stride)/sr),
                     'end': min(j/sr, offset+((receptive-1)/2+(b-.5)*stride)/sr),
                     'posterior': float(probs[a:b, token_id].mean()), 'model': name}
            if owned(event, window): events.append(event)
    return events


def infer_diarization(y, sr, pool):
    import torch, numpy as np
    b = pool.get('diarizer'); model, processor = b['model'], b['processor']
    batch = inputs_to_device(processor(y, sampling_rate=sr, is_streaming=False, return_tensors='pt',
                                      truncation=False, return_attention_mask=True), model)
    with torch.inference_mode(): logits = model(**batch).logits
    raw = processor.extract_speaker_dict(logits, attention_mask=batch['attention_mask'], threshold=.5)[0]
    segments = [{'start': max(0., float(t['Start'])), 'end': min(len(y)/sr, float(t['End'])),
                 'speaker_id': 'SPEAKER_'+format(int(t['Speaker']), '02d')} for t in raw if t['End']-t['Start'] >= .12]
    active = (logits[0].sigmoid().cpu().numpy() > .5).sum(-1) > 1
    active &= batch['attention_mask'][0].bool().cpu().numpy()
    step = processor.feature_extractor.hop_length/sr
    overlaps = mask_spans((np.arange(len(active))+.5)*step, active, step, len(y)/sr)
    return {'segments': segments, 'overlap': overlaps}


def attribute(observations, diar):
    speakers = sorted({s['speaker_id'] for s in diar['segments']})
    spans = {sp: union([[s['start'], s['end']] for s in diar['segments'] if s['speaker_id'] == sp]) for sp in speakers}
    for obs in observations:
        a, b = obs.get('start'), obs.get('end')
        scores = {sp: overlap(a, b, v) for sp, v in spans.items()}
        speaker = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else None
        if speaker is None and len(speakers) == 1: speaker = speakers[0]
        obs['speaker_id'] = speaker
        obs['speaker_fraction'] = scores[speaker]/(b-a) if speaker and finite(a) and finite(b) and b > a else 0.
    return observations
