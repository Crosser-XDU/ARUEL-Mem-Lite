import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
import numpy as np
import tiktoken


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def ident(*values):
    return hashlib.sha256(dumps(values).encode()).hexdigest()[:32]


def canonical(text):
    return re.sub(r"[^\w\u3400-\u9fff]", "", unicodedata.normalize("NFKC", str(text)).casefold())


@lru_cache(maxsize=1)
def encoding():
    return tiktoken.get_encoding("o200k_base")


def tokens(text):
    return len(encoding().encode(str(text), disallowed_special=()))


def clip(text, limit):
    e = encoding()
    ids = e.encode(str(text), disallowed_special=())
    return text if len(ids) <= limit else e.decode_bytes(ids[:limit]).decode('utf-8',errors='ignore')


def chunks(text, limit=550, overlap=50):
    """Return complete Unicode slices; source remains immutable in messages."""
    if not text.strip():
        return []
    out, start = [], 0
    while start < len(text):
        lo, hi = start + 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if tokens(text[start:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        end = lo
        if end < len(text):
            pivot = max(text.rfind("\n", start, end), text.rfind("。", start, end))
            if pivot > start + (end - start) // 2:
                end = pivot + 1
        out.append((text[start:end], start, end))
        if end == len(text):
            break
        start = max(start + 1, end - min(overlap, (end-start)//4))
    return out


STOP = set("the a an and or is are was were to of in on for with what which when where who how my me I this that it 的 了 是 在 和 与 我 哪 什么 怎么 这个 那个 吗 呢".lower().split())


def terms(text):
    """Deterministic CJK unigrams/bigrams plus identifier-preserving Latin words."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    out = []
    for run in re.findall(r"[\u3400-\u9fff]+|[a-z0-9_]+(?:[-.:/][a-z0-9_]+)*", text):
        if re.match(r"[\u3400-\u9fff]", run):
            out.extend(run)
            out.extend(run[i:i+2] for i in range(len(run)-1))
        elif run not in STOP:
            out.append(run)
    return [v for v in out if v not in STOP]


def normvec(value):
    v = np.asarray(value, dtype=np.float32)
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        raise ValueError("invalid embedding")
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        raise ValueError("zero embedding")
    return v / n


def iso(ts):
    return datetime.fromtimestamp(ts/1000, timezone.utc).isoformat().replace("+00:00", "Z") if ts is not None else None


def strings(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        value = [value]
    return [str(x).strip() for x in value if isinstance(x, (str, int, float)) and str(x).strip()]


def as_list(value):
    return value if isinstance(value, list) else ([] if value is None else [value])


def finite(value, default=0.):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default
