"""Carga y reutilización de pesos por revisión."""

import gc
import json
from pathlib import Path
from .config import MODEL_REPOS, PINS, ROOT
from .utilidades import digest


class ModelPool:
    # Se conservan modelos entre audios con una caché LRU acotada y sin paralelizar las inferencias sobre la GPU.
    def __init__(self, cfg, root=ROOT):
        self.cfg, self.root, self.loaded, self.pins = cfg, Path(root), {}, {}
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def pin(self, name):
        from huggingface_hub import model_info
        repo = MODEL_REPOS[name]
        requested = self.cfg['model_revisions'].get(name, 'main')
        if requested == 'main' and name in PINS:
            self.pins[name] = PINS[name]
            return self.pins[name]
        key = digest([repo, requested])
        path = self.root / 'modelos' / 'short-pins' / (key+'.json')
        if name not in self.pins:
            if path.exists(): revision = json.loads(path.read_text())['revision']
            else:
                revision = model_info(repo, revision=requested).sha
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'repo': repo, 'revision': revision}), encoding='utf-8')
            self.pins[name] = revision
        return self.pins[name]

    def get(self, name):
        import torch, transformers as tr
        if name in self.loaded:
            bundle = self.loaded.pop(name); self.loaded[name] = bundle; return bundle
        while len(self.loaded) >= self.cfg['models_in_memory']:
            self.loaded.pop(next(iter(self.loaded))); gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        opts = {'revision': self.pin(name), 'cache_dir': str(self.root / 'modelos'), 'trust_remote_code': False}
        classes = {'qwen': 'Qwen3ASRForConditionalGeneration', 'qwen_aligner': 'Qwen3ASRForTokenClassification',
                   'parakeet': 'ParakeetForTDT', 'whisper': 'WhisperForConditionalGeneration',
                   'cnam': 'AutoModelForCTC', 'multilingual': 'AutoModelForCTC', 'diarizer': 'AutoModelForAudioFrameClassification'}
        tokenizer = None
        if name in ('cnam', 'multilingual'):
            processor = tr.AutoFeatureExtractor.from_pretrained(MODEL_REPOS[name], **opts)
            tokenizer = (tr.Wav2Vec2PhonemeCTCTokenizer.from_pretrained(MODEL_REPOS[name], do_phonemize=False, **opts)
                         if name == 'multilingual' else tr.AutoTokenizer.from_pretrained(MODEL_REPOS[name], **opts))
        else: processor = tr.AutoProcessor.from_pretrained(MODEL_REPOS[name], **opts)
        model = getattr(tr, classes[name]).from_pretrained(MODEL_REPOS[name], dtype=torch.float32, attn_implementation='eager', **opts).eval()
        try: model.to(self.device)
        except torch.cuda.OutOfMemoryError:
            self.loaded.clear(); gc.collect(); torch.cuda.empty_cache(); model.to(self.device)
        bundle = {'model': model, 'processor': processor, 'tokenizer': tokenizer}
        self.loaded[name] = bundle
        return bundle

    def close(self):
        self.loaded.clear(); gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def inputs_to_device(batch, model):
    import torch
    return {k: v.to(model.device, dtype=torch.float32 if v.is_floating_point() else v.dtype)
            for k, v in batch.items() if isinstance(v, torch.Tensor)}
