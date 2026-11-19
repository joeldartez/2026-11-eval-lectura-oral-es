"""Configuración del método y rutas relativas al repositorio."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "pasajes" / "pedro-y-mateo.txt"
OUTPUT_ROOT = ROOT / "resultados" / "short"
CFG = {
    'protocol': 'lectura-short-1.0.0', 'use_llm': True,
    'asr_models': ['qwen', 'parakeet'], 'phone_models': ['cnam', 'multilingual'],
    'diarization': True, 'student_speaker': None, 'reading_bounds': None,
    'window_s': 18.0, 'context_s': 1.5, 'beam': 32, 'paths_per_model': 2,
    'backtrack': 16, 'compatible_phone_similarity': 0.65,
    'llm_model': 'gpt-6-luna', 'llm_workers': 4, 'llm_words_per_call': 24,
    'llm_max_word_calls': 6, 'llm_timeout_s': 90, 'llm_max_output_tokens': 4500,
    'llm_input_usd_million': 0.10, 'llm_output_usd_million': 0.50,
    'pitch_floor': 75.0, 'pitch_ceiling': 1000.0, 'pyin': True, 'formants': True,
    'inference_cache': True, 'llm_cache': True, 'models_in_memory': 2,
    'model_revisions': {},
}
MODEL_REPOS = {
    'qwen': 'Qwen/Qwen3-ASR-0.6B-hf',
    'parakeet': 'nvidia/parakeet-tdt-0.6b-v3',
    'whisper': 'openai/whisper-small',
    'qwen_aligner': 'Qwen/Qwen3-ForcedAligner-0.6B-hf',
    'cnam': 'Cnam-LMSSC/wav2vec2-spanish-phonemizer',
    'multilingual': 'facebook/wav2vec2-lv-60-espeak-cv-ft',
    'diarizer': 'nvidia/Nemotron-3-Diarization',
}
ESPEAK_DLL = Path('C:/Program Files/eSpeak NG/libespeak-ng.dll')

PINS = {k: v["revision"] for k, v in json.loads(
    (ROOT / "modelos.lock.json").read_text(encoding="utf-8")
)["models"].items()}
