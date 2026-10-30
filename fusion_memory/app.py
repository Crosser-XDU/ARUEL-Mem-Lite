"""AML-compatible evidence memory service. Run one worker for ordered ingestion."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .config import Settings
from . import __version__
from .provider import Provider, UpstreamError, text_of
from .retrieval import rank_memories
from .schemas import AddRequest, SearchRequest, media_size
from .storage import Storage, ConflictError

log = logging.getLogger(__name__)


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > self.limit:
                return await JSONResponse({"detail": "request body exceeds 48 MiB"}, 413)(scope, receive, send)
            if not event.get("more_body", False):
                break
        sent = False
        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def source_time(record):
    value = record.get("timestamp_ms")
    if value is not None:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return record["created_at"]


def pack_evidence(store, user_id, ranked, settings, multimodal, states):
    """Deliver original evidence with provenance inside content, under shared budgets."""
    results, image_count, image_bytes, text_chars = [], 0, 0, 0
    by_record = {}
    for state in states:
        by_record.setdefault(state["record_id"], []).append(state)
    for record, score in ranked:
        remaining = settings.max_response_text_chars - text_chars
        if remaining < 256:
            break
        when = source_time(record)
        time_label = (f"source_time={when}" if record.get("timestamp_ms") is not None
                      else f"source_time=unknown | ingested_at={record['created_at']}")
        prefix = (f"[Evidence {record['id']} | role={record['role']} | {time_label}"
                  f" | session={record['session_id']} | position={record['position']}]\n")
        for state in by_record.get(record["id"], []):
            prefix += (f"[State history: {state['subject']} / {state['predicate']}; "
                       f"current_in_stored_history={state['is_current']}; "
                       f"time_basis={'ingestion_fallback' if state['timestamp_is_fallback'] else 'source_message'}; "
                       f"valid_from_unix_ms={state['valid_from_ms']}; "
                       f"valid_to_unix_ms={state['valid_to_ms']}]\n")
        raw = store.get_content(user_id, record["id"])
        original = ([{"type": "text", "text": raw}] if isinstance(raw, str) else raw)
        output, used = [], 0
        def add_text(text):
            nonlocal used
            room = remaining - used
            if room <= 0:
                return
            if len(text) > room:
                marker = "\n[Source text truncated at response budget]"
                text = text[:max(0, room-len(marker))] + marker[:room]
            if text:
                output.append({"type": "text", "text": text})
                used += len(text)
        add_text(prefix)
        for part in original:
            if part["type"] == "text":
                add_text(part["text"])
            else:
                size = media_size([part])
                if (image_count < settings.max_response_images and
                        image_bytes + size <= settings.max_response_image_bytes):
                    output.append(part)
                    image_bytes += size
                    image_count += 1
                else:
                    add_text("[Original image omitted because the response image budget is exhausted.]\n")
                    summary = record.get("structured", {}).get("summary", "")
                    if summary:
                        origin = record.get("structured", {}).get("extraction_source")
                        label = {"provided_caption": "[Provided image caption, not original pixels] ",
                                 "mixed": "[Provided caption and model-extracted image index, not original pixels] "}.get(
                                     origin, "[Model-extracted image index, not original pixels] ")
                        add_text(label + summary)
        text_chars += used
        as_parts = multimodal or not isinstance(raw, str)
        content = output if as_parts else "".join(p["text"] for p in output)
        results.append({"id": record["id"], "content": content,
                        "score": float(score), "created_at": when})
    return {"data": results}


def create_app(settings: Settings | None = None, provider=None, storage=None):
    settings = settings or Settings.from_env()
    store = storage or Storage(settings.db_path)
    provider = provider or Provider(settings)
    locks = [asyncio.Lock() for _ in range(128)]
    started = time.monotonic()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await provider.close()
        store.close()

    app = FastAPI(title="AUREL-Mem-Lite", version=__version__, lifespan=lifespan,
                  description="Caption-first multimodal memory; original multiview hybrid retrieval; Add/Search evidence only.")
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_body_bytes)
    app.state.storage, app.state.provider, app.state.settings = store, provider, settings

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if settings.service_token and request.url.path not in ("/healthz", "/"):
            headers = request.headers.getlist("Authorization")
            supplied = headers[0] if len(headers) == 1 else ""
            scheme, _, token = supplied.partition(" ")
            # Schemes are case-insensitive; the credential remains exact and
            # case-sensitive. Bytes also keep non-ASCII bad input from raising
            # a compare_digest TypeError instead of returning an ordinary 401.
            correct_token = hmac.compare_digest(token.encode("utf-8"), settings.service_token.encode("utf-8"))
            if (scheme.lower() not in ("bearer", "token") or not token
                    or any(character.isspace() for character in token) or not correct_token):
                return JSONResponse({"detail": "invalid service authorization"}, 401)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Pydantic input values may contain private text or huge inline images.
        return JSONResponse({"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                                        for e in exc.errors()]}, 422)

    @app.exception_handler(ConflictError)
    async def conflict_error(request, exc):
        return JSONResponse({"detail": str(exc)}, 409)

    @app.exception_handler(UpstreamError)
    async def upstream_error(request, exc):
        return JSONResponse({"detail": str(exc)}, 503)

    @app.get("/")
    async def index():
        return {"service": "AUREL-Mem-Lite", "add": "/add", "search": "/search", "docs": "/docs"}

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "service": "AUREL-Mem-Lite", "version": __version__, "model": settings.model,
                "api_key_configured": bool(settings.api_key),
                "extraction_policy": "caption_first_text_passthrough",
                "retrieval_mode": "hybrid" if settings.embedding_enabled else "lexical_multiview",
                "embedding_model": settings.embedding_model if settings.embedding_enabled else None,
                "embedding_dimensions": settings.embedding_dimensions if settings.embedding_enabled else None,
                "uptime_seconds": int(time.monotonic()-started)}

    @app.post("/add")
    async def add(request: AddRequest):
        payload = request.model_dump(exclude_none=True)
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        reply = {"success": True, "request_id": request.request_id,
                 "user_id": request.user_id, "session_id": request.session_id}
        lock_id = int(hashlib.sha256(request.user_id.encode()).hexdigest()[:8], 16) % len(locks)
        async with locks[lock_id]:
            if await asyncio.to_thread(store.check_request, request.user_id, request.request_id, fingerprint):
                return reply
            previous = await asyncio.to_thread(store.recent_session_context, request.user_id, request.session_id)
            history = "\n".join(f"{r['role']}: {r['raw_text']}" for r in previous)[-6000:]
            frames = await provider.extract(payload["messages"], history)
            # Fast paths already carry their evidence in the original text.
            # Do not embed duplicated captions, empty fields or provenance labels.
            texts = [text_of(m["content"]) if frame.get("extraction_source") in ("raw_text", "provided_caption")
                     else text_of(m["content"]) + "\n" + json.dumps(frame, ensure_ascii=False)
                     for m, frame in zip(payload["messages"], frames, strict=True)]
            embeddings = await provider.embed(texts)
            prepared = [{"structured": frame, "embedding": vector,
                         "embedding_model": settings.embedding_model if vector is not None else None}
                        for frame, vector in zip(frames, embeddings, strict=True)]
            await asyncio.to_thread(store.commit_add, payload, fingerprint, prepared)
        return reply

    @app.post("/search")
    async def search(request: SearchRequest):
        records = await asyncio.to_thread(store.list_records, request.user_id)
        if not records:
            return {"data": []}
        payload = request.model_dump(exclude_none=True)
        query = await provider.query_text(payload["query"], request.options)
        vector = (await provider.embed([query]))[0]
        ranked = await asyncio.to_thread(
            rank_memories, records, query, vector, min(request.top_k, settings.retrieval_limit),
            query_embedding_model=settings.embedding_model if vector is not None else None,
            neighbor_hops=settings.neighbor_hops, graph_expansion=settings.graph_expansion,
            diversity=settings.diversity,
        )
        states = await asyncio.to_thread(store.list_states, request.user_id)
        return await asyncio.to_thread(pack_evidence, store, request.user_id, ranked, settings,
                                       isinstance(payload["query"], list), states)

    return app
