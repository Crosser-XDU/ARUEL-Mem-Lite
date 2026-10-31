from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv


@dataclass
class Settings:
    profile: str = "C"
    data_dir: str = "data"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    embedding_key: str = ""
    embedding_base: str = ""
    service_token: str = ""
    worker_token: str = ""
    worker_url: str = "http://127.0.0.1:18220"
    dimensions: int = 1024
    llm_concurrency: int = 4
    search_concurrency: int = 1
    upstream_timeout: int = 150
    add_timeout: int = 1700
    search_timeout: int = 1500
    candidate_limit: int = 80
    response_image_limit: int = 12
    caption_limit: int = 30
    input_budget: int = 117760
    template_reserve: int = 8192
    image_base_tokens: int = 2833
    image_tile_tokens: int = 5667
    max_body_bytes: int = 48 * 1024 * 1024
    cache_bytes: int = 256 * 1024 * 1024

    @classmethod
    def from_env(cls):
        load_dotenv(os.environ.get("ENV_FILE", ".env"), override=False)
        key = os.environ.get("OPENAI_API_KEY", "")
        base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        s = cls(profile=os.environ.get("MEMORY_PROFILE", "C").upper(),
                data_dir=os.environ.get("MEMORY_DATA", "data"), api_key=key, api_base=base,
                embedding_key=os.environ.get("EMBEDDING_API_KEY", "") or key,
                embedding_base=os.environ.get("EMBEDDING_API_BASE", "") or base,
                service_token=os.environ.get("MEMORY_API_TOKEN", ""),
                worker_token=os.environ.get("WORKER_TOKEN", ""),
                worker_url=os.environ.get("MODEL_WORKER_URL", "http://127.0.0.1:18220"))
        for name in ("llm_concurrency", "search_concurrency", "candidate_limit", "template_reserve",
                     "image_base_tokens", "image_tile_tokens", "upstream_timeout"):
            if name.upper() in os.environ:
                setattr(s, name, int(os.environ[name.upper()]))
        if s.profile not in {"A", "B", "C"}:
            raise ValueError("MEMORY_PROFILE must be A/B/C")
        Path(s.data_dir).mkdir(parents=True, exist_ok=True)
        return s
