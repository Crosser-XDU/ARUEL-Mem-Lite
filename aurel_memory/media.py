import base64
import hashlib
import re
import os
import tempfile
from io import BytesIO
from pathlib import Path
from PIL import Image, UnidentifiedImageError

MAX_IMAGE = 10 * 1024 * 1024
MAX_IMAGES = 30 * 1024 * 1024
URI = re.compile(r"^data:image/(jpeg|png|webp);base64,([A-Za-z0-9+/]*={0,2})$")
CAPTION = re.compile(r"(?:Image[ _-]*Caption|图片描述|图像描述)\s*[:：]", re.I)
TIMESTAMP_ONLY=re.compile(r'\s*\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\s*')


def decode(uri):
    if len(uri) > MAX_IMAGE * 4 // 3 + 128:
        raise ValueError("image exceeds 10 MiB")
    m = URI.fullmatch(uri)
    if not m:
        raise ValueError("expected inline JPEG/PNG/WebP Base64 Data URI")
    try:
        data = base64.b64decode(m[2], validate=True)
        if not data or len(data) > MAX_IMAGE:
            raise ValueError("image exceeds 10 MiB or is empty")
        with Image.open(BytesIO(data)) as image:
            if image.format != {"jpeg":"JPEG", "png":"PNG", "webp":"WEBP"}[m[1]]:
                raise ValueError("image MIME mismatch")
            if image.width*image.height > 40_000_000 or getattr(image, "n_frames", 1) != 1:
                raise ValueError("image exceeds pixel limit or is animated")
            size = image.size
            image.verify()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ValueError("invalid image") from e
    return "image/"+m[1], data, size


def text_of(content):
    return content if isinstance(content, str) else "\n".join(x["text"] for x in content if x["type"] == "text")


def images_of(content):
    return [] if isinstance(content, str) else [(i, p["image_url"]["url"]) for i, p in enumerate(content) if p["type"] == "image_url"]


def split_captions(content):
    """Conservative per-part mapping; ambiguous descriptions stay in raw text."""
    if isinstance(content, str):
        return {}, content
    images = images_of(content)
    labels = [];ambiguous=False
    for i, part in enumerate(content):
        if part["type"] != "text":
            continue
        matches = list(CAPTION.finditer(part["text"]))
        if len(matches)>1:ambiguous=True
        if len(matches) == 1:
            m = matches[0]
            cap = part["text"][m.end():].strip()
            if cap.strip(" \t\r\n\"'`[](){}<>:：*_-.") and not TIMESTAMP_ONLY.fullmatch(cap):
                labels.append((i, cap, m.start()))
    if len(images)>1 and any(i+1 not in {n for n,_ in images} for i,_,_ in labels):ambiguous=True
    if ambiguous or (len(images)==1 and len(labels)>1):return {},text_of(content)
    mapping, used = {}, {}
    for i, cap, start in labels:
        candidates = [idx for idx, _ in images if idx == i+1]
        if len(images) == 1 and len(labels) == 1:
            candidates = [images[0][0]]
        if len(candidates) == 1 and candidates[0] not in mapping:
            mapping[candidates[0]] = cap
            used[i] = start
    raw = "\n".join(p["text"][:used[i]] if i in used else p["text"]
                    for i,p in enumerate(content) if p["type"] == "text").strip()
    return mapping, raw


def store_image(root, uri):
    mime, data, (w,h) = decode(uri)
    key = hashlib.sha256(data).hexdigest()
    path = Path(root)/"assets"/key[:2]/key
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # Profiles can write the same immutable asset concurrently.
        fd,name=tempfile.mkstemp(prefix=key+'.',suffix='.tmp',dir=path.parent)
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name):os.unlink(name)
    return {"id":key, "mime":mime, "path":str(path), "width":w, "height":h, "bytes":len(data)}


def asset_uri(asset):
    return "data:"+asset["mime"]+";base64,"+base64.b64encode(Path(asset["path"]).read_bytes()).decode()
