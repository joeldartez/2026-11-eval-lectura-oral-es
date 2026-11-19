"""Normalización, fonemas esperados y comparación del texto."""

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from .config import ESPEAK_DLL


def norm(text):
    decomposed = unicodedata.normalize('NFD', str(text).lower())
    return unicodedata.normalize('NFC', ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn' or c == chr(771)))


def tokens(text):
    return re.findall(r'[^\W\d_]+', str(text), re.UNICODE)


def edit_alignment(expected, observed):
    n, m = len(expected), len(observed)
    d = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): d[i][0] = i
    for j in range(m+1): d[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(expected[i-1] != observed[j-1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1]+(expected[i-1] != observed[j-1]):
            out.append(('match' if expected[i-1] == observed[j-1] else 'substitution', i-1, j-1)); i -= 1; j -= 1
        elif i and d[i][j] == d[i-1][j]+1: out.append(('deletion', i-1, None)); i -= 1
        else: out.append(('insertion', None, j-1)); j -= 1
    return list(reversed(out))


@lru_cache(maxsize=50000)
def text_similarity(a, b):
    ops = edit_alignment(list(a), list(b))
    return max(0., 1-sum(op != 'match' for op, _, _ in ops)/max(len(a), len(b), 1))


def wer(reference, text):
    a, b = [w['norm'] for w in reference], [norm(w) for w in tokens(text)]
    counts = Counter(op for op, _, _ in edit_alignment(a, b))
    return {'wer': (counts['substitution']+counts['deletion']+counts['insertion'])/max(len(a), 1),
            'substitutions': counts['substitution'], 'deletions': counts['deletion'], 'insertions': counts['insertion']}


def phone_units(text):
    text = unicodedata.normalize('NFD', str(text)).replace('ɡ', 'g').replace('β', 'b').replace('ð', 'd').replace('ɣ', 'g').replace('θ', 's').replace('ʎ', 'ʝ')
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn' and c not in ' ˈˌːˑ|_͜͡')
    return re.findall(r'tʃ|dʒ|.', text)


def phone_similarity(expected, observed):
    if not expected or not observed: return None
    ops = edit_alignment(expected, observed)
    return max(0., 1-sum(op != 'match' for op, _, _ in ops)/max(len(expected), len(observed)))


def build_reference(text):
    from phonemizer import phonemize
    if ESPEAK_DLL.is_file():
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(str(ESPEAK_DLL))
    matches = list(re.finditer(r'[^\W\d_]+', text, re.UNICODE))
    if not matches: raise ValueError('El pasaje no tiene palabras.')
    ipa = phonemize([m.group() for m in matches], language='es-419', backend='espeak', strip=True, with_stress=True, njobs=1)
    if len(ipa) != len(matches): raise ValueError('Phonemizer cambió el número de palabras.')
    result, sid = [], 1
    for i, (m, sound) in enumerate(zip(matches, ipa)):
        after = text[m.end():matches[i+1].start() if i+1 < len(matches) else len(text)]
        boundary = 'period' if any(c in after for c in '.!?') else 'comma' if any(c in after for c in ',;:') else 'interior'
        result.append({'index': i, 'word_id': f'W{i+1:04d}', 'text': m.group(), 'norm': norm(m.group()),
                       'sentence_id': sid, 'boundary': boundary, 'ipa': sound, 'phones': phone_units(sound)})
        if boundary == 'period': sid += 1
    return result
