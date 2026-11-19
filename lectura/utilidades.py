"""Intervalos, estadísticas y huellas reproducibles."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [safe(v) for v in x]
    if hasattr(x, 'tolist'): return safe(x.tolist())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def digest(x):
    return hashlib.sha256(json.dumps(safe(x), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def environment_info():
    import platform
    from importlib.metadata import version
    packages = ('torch', 'transformers', 'numpy', 'librosa', 'praat-parselmouth', 'phonemizer', 'openai', 'openpyxl')
    return {'python': platform.python_version(), 'packages': {name: version(name) for name in packages}}


def union(spans):
    result = []
    for a, b in sorted((float(a), float(b)) for a, b in spans if finite(a) and finite(b) and b > a):
        if result and a <= result[-1][1] + 1e-8: result[-1][1] = max(result[-1][1], b)
        else: result.append([a, b])
    return result


def intersect(a, b):
    a, b, out, i, j = union(a), union(b), [], 0, 0
    while i < len(a) and j < len(b):
        left, right = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if right > left: out.append([left, right])
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return out


def subtract(spans, exclusions):
    result = union(spans)
    for a, b in union(exclusions):
        result = [[x, y] for left, right in result for x, y in [(left, min(a, right)), (max(b, left), right)] if y > x]
    return union(result)


def seconds(spans): return sum(b-a for a, b in union(spans))


def clip(spans, a, b): return intersect(spans, [[a, b]]) if finite(a) and finite(b) and b > a else []


def overlap(a, b, spans): return seconds(clip(spans, a, b))


def mask_spans(times, mask, step, duration):
    return union([[max(0., float(t)-step/2), min(duration, float(t)+step/2)] for t, yes in zip(times, mask) if yes])


def stats(values):
    import numpy as np
    x = np.asarray([v for v in values if finite(v)], dtype=float)
    if not len(x): return {'n': 0, 'median': None, 'mean': None, 'p10': None, 'p90': None, 'iqr': None}
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {'n': len(x), 'median': float(q[2]), 'mean': float(x.mean()), 'p10': float(q[0]), 'p90': float(q[4]), 'iqr': float(q[3]-q[1])}
