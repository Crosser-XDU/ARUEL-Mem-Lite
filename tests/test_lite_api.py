"""Real Lite API/Provider/SQLite with synthetic SDK HTTP responses only."""

import asyncio
import base64
from contextlib import contextmanager
from io import BytesIO
import json
import sqlite3

from fastapi.testclient import TestClient
import httpx
from openai import AsyncOpenAI
from PIL import Image
import pytest

from fusion_memory.app import create_app
from fusion_memory.config import Settings
from fusion_memory.provider import Frame, Provider, text_of


TEST_TOKEN = "synthetic-lite-service-token"
TABLES = ("requests", "records", "image_blobs", "image_occurrences", "state_versions")


def image_part(color=(40, 120, 200)):
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color).save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": uri, "detail": "high"}}


def message(content, role="user"):
    return {"role": role, "timestamp": 1704067200000, "content": content}


def payload(messages, request_id="lite-integration-request"):
    return {"request_id": request_id, "user_id": "synthetic-user",
            "session_id": "synthetic-session", "messages": messages}


def completion(content):
    return httpx.Response(200, json={
        "id": "chatcmpl-lite-integration", "object": "chat.completion", "created": 1704067200,
        "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(content, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    })


def embeddings(body):
    return httpx.Response(200, json={
        "object": "list", "model": body["model"],
        "data": [{"object": "embedding", "index": index,
                  "embedding": [1.0] + [0.0] * (body["dimensions"] - 1)}
                 for index in range(len(body["input"]))],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    })


def counts(db_path):
    with sqlite3.connect(db_path) as connection:
        return {name: connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                for name in TABLES}


@contextmanager
def service(tmp_path, handler, **overrides):
    settings = Settings(db_path=str(tmp_path / "lite.sqlite3"),
                        api_key="synthetic-chat-key-not-real", api_base="https://chat.example.invalid/v1",
                        embedding_key="synthetic-embedding-key-not-real",
                        embedding_base="https://embedding.example.invalid/v1",
                        embedding_enabled=True, embedding_dimensions=2,
                        service_token=TEST_TOKEN, **overrides)
    provider = Provider(settings)
    # Replace only the external HTTP boundary, preserving actual SDK encoding,
    # Provider routing, API validation, retrieval and storage transactions.
    asyncio.run(provider.close())
    provider.client = AsyncOpenAI(
        api_key=settings.api_key, base_url=settings.api_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider.embedding_client = AsyncOpenAI(
        api_key=settings.embedding_key, base_url=settings.embedding_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app(settings, provider=provider)
    with TestClient(app, headers={"Authorization": "Token " + TEST_TOKEN}) as client:
        yield client, app, provider, settings


def test_raw_and_supplied_caption_embed_exact_source_without_chat_and_return_original_image(tmp_path):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        assert request.url.path == "/v1/embeddings", "Fast paths must not call chat completion"
        return embeddings(body)

    plain = "  Speaker: Ada\nA cobalt telescope is in the observatory.\n2024-01-01 00:00:00  "
    source = [{"type": "text", "text": "Speaker: Ada\nImage Caption: a cobalt telescope"},
              image_part(), {"type": "text", "text": "\n2024-01-01 00:00:00"}]
    request = payload([message(plain), message(source)])
    with service(tmp_path, handler) as (client, app, provider, settings):
        added = client.post("/add", json=request)
        assert added.status_code == 200, added.text
        assert requests[0][1]["input"] == [plain, text_of(source)]
        assert requests[0][1]["model"] == "text-embedding-v4"
        assert provider.calls["extract"] == 0
        rows = app.state.storage.list_records("synthetic-user")
        assert [row["structured"]["extraction_source"] for row in rows] == ["raw_text", "provided_caption"]
        assert rows[0]["raw_text"] == plain
        assert rows[1]["raw_text"] == text_of(source)
        assert rows[1]["structured"]["provenance"]["caption_verified"] is False
        assert all(row["structured"]["state_updates"] == [] for row in rows)
        response = client.post("/search", json={"user_id": "synthetic-user", "query": "cobalt telescope", "top_k": 5})
        assert response.status_code == 200, response.text
        assert set(response.json()) == {"data"}
        returned = {item["id"]: item for item in response.json()["data"]}
        assert plain in returned[rows[0]["id"]]["content"]
        assert returned[rows[1]["id"]]["content"][1:] == source
        assert returned[rows[1]["id"]]["created_at"] == "2024-01-01T00:00:00Z"
        assert provider.calls["query"] == 0
        assert all(path == "/v1/embeddings" for path, _ in requests)
        before = len(requests)
        assert client.post("/add", json=request).status_code == 200
        assert len(requests) == before, "An idempotent retry must not embed again"
        assert counts(settings.db_path)["requests"] == 1


def test_second_embedding_batch_failure_is_atomic_and_same_request_can_retry(tmp_path):
    state = {"fail": True, "embedding_calls": 0}

    def handler(request):
        assert request.url.path == "/v1/embeddings"
        state["embedding_calls"] += 1
        if state["fail"] and state["embedding_calls"] == 2:
            return httpx.Response(503, json={"error": {"message": "synthetic embedding unavailable"}})
        return embeddings(json.loads(request.content))

    captioned = [{"type": "text", "text": "Image Caption: cobalt telescope"}, image_part()]
    messages = [message(captioned)] + [message(f"telescope observation {index}") for index in range(10)]
    request = payload(messages)
    with service(tmp_path, handler) as (client, app, provider, settings):
        failed = client.post("/add", json=request)
        assert failed.status_code == 503, failed.text
        assert state["embedding_calls"] == 2
        assert counts(settings.db_path) == {name: 0 for name in TABLES}
        assert app.state.storage.list_records("synthetic-user") == []
        assert provider.calls["extract"] == 0
        state["fail"] = False
        retry = client.post("/add", json=request)
        assert retry.status_code == 200, retry.text
        assert counts(settings.db_path) == {
            "requests": 1, "records": 11, "image_blobs": 1, "image_occurrences": 1, "state_versions": 0,
        }
        before = state["embedding_calls"]
        assert client.post("/add", json=request).status_code == 200
        assert state["embedding_calls"] == before


def test_missing_caption_calls_gpt_and_embeds_model_index_but_returns_source_evidence(tmp_path):
    requests = []
    source = [{"type": "text", "text": "Unlabeled photograph"}, image_part()]
    model_frame = Frame(summary="a copper teapot", ocr=["QT-201"], objects=["teapot"]).model_dump()

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/embeddings":
            return embeddings(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] == "gpt-4o-mini"
        assert body["response_format"]["type"] == "json_schema"
        sent_images = [part for part in body["messages"][1]["content"] if part["type"] == "image_url"]
        assert sent_images == [source[1]]
        return completion({"records": [{"index": 0, **model_frame}]})

    with service(tmp_path, handler) as (client, app, provider, _):
        response = client.post("/add", json=payload([message("local plain text"), message(source)]))
        assert response.status_code == 200, response.text
        assert [path for path, _ in requests] == ["/v1/chat/completions", "/v1/embeddings"]
        embedded = requests[1][1]["input"]
        assert embedded[0] == "local plain text"
        prefix, model_json = embedded[1].split("\n", 1)
        assert prefix == text_of(source)
        structured = json.loads(model_json)
        assert structured["extraction_source"] == "model"
        assert structured["summary"] == model_frame["summary"]
        assert structured["ocr"] == ["QT-201"]
        assert provider.calls["extract"] == 1
        rows = app.state.storage.list_records("synthetic-user")
        image_id = next(row["id"] for row in rows if row["has_image"])
        found = client.post("/search", json={"user_id": "synthetic-user", "query": "QT-201", "top_k": 5})
        assert found.status_code == 200, found.text
        item = next(item for item in found.json()["data"] if item["id"] == image_id)
        assert item["content"][1:] == source
        assert "answer" not in found.json()
        assert provider.calls["query"] == 0
        assert sum(path == "/v1/chat/completions" for path, _ in requests) == 1


def test_image_budget_omission_labels_supplied_caption_as_provided_not_model(tmp_path):
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        return embeddings(json.loads(request.content))

    caption = "a cobalt telescope beside a copper compass"
    source = [{"type": "text", "text": "Image Caption: " + caption}, image_part()]
    with service(tmp_path, handler, max_response_images=0) as (client, _, provider, _):
        assert client.post("/add", json=payload([message(source)])).status_code == 200
        response = client.post("/search", json={"user_id": "synthetic-user", "query": "compass", "top_k": 1})
        assert response.status_code == 200, response.text
        parts = response.json()["data"][0]["content"]
        assert all(part["type"] == "text" for part in parts)
        text = text_of(parts)
        assert "[Original image omitted because the response image budget is exhausted.]" in text
        assert "[Provided image caption, not original pixels] " + caption in text
        assert "Model-extracted image index" not in text
        assert provider.calls["extract"] == 0


def test_default_retrieval_cap_remains_32_and_respects_smaller_top_k(tmp_path):
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        return embeddings(json.loads(request.content))

    request = payload([message(f"observatory telescope observation {index}") for index in range(40)])
    with service(tmp_path, handler) as (client, _, provider, settings):
        assert settings.retrieval_limit == 32
        assert client.post("/add", json=request).status_code == 200
        for requested, expected in ((100, 32), (5, 5)):
            response = client.post("/search", json={"user_id": "synthetic-user", "query": "telescope", "top_k": requested})
            assert response.status_code == 200, response.text
            evidence = response.json()["data"]
            assert len(evidence) == expected
            assert len({item["id"] for item in evidence}) == expected
            assert [item["score"] for item in evidence] == sorted((item["score"] for item in evidence), reverse=True)
        assert provider.calls["extract"] == provider.calls["query"] == 0


def test_image_query_still_indexes_with_gpt_and_returns_evidence_without_generating_answer(tmp_path):
    requests = []
    source = [{"type": "text", "text": "Image Caption: a cobalt telescope"}, image_part()]
    query = [{"type": "text", "text": "Which telescope was shown?"}, image_part((200, 60, 30))]

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/embeddings":
            return embeddings(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"][1]["content"] == query
        assert "Do not answer the question" in body["messages"][0]["content"]
        return completion({"description": "a cobalt telescope", "terms": ["cobalt", "telescope"]})

    with service(tmp_path, handler) as (client, _, provider, _):
        assert client.post("/add", json=payload([message(source)])).status_code == 200
        response = client.post("/search", json={"user_id": "synthetic-user", "query": query, "top_k": 1})
        assert response.status_code == 200, response.text
        assert set(response.json()) == {"data"}
        assert response.json()["data"][0]["content"][1:] == source
        assert provider.calls["extract"] == 0
        assert provider.calls["query"] == 1
        assert sum(path == "/v1/chat/completions" for path, _ in requests) == 1


@pytest.mark.parametrize("image_budget", [0, 12])
def test_partial_caption_only_sends_missing_image_and_preserves_mixed_evidence(tmp_path, image_budget):
    requests = []
    known = image_part((20, 180, 80))
    missing = image_part((220, 40, 70))
    source = [{"type": "text", "text": "Image Caption: a cobalt telescope"}, known,
              {"type": "text", "text": "Image Caption:   "}, missing]
    model_frame = Frame(summary="a copper compass", ocr=["CP-772"]).model_dump()

    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/embeddings":
            return embeddings(body)
        assert request.url.path == "/v1/chat/completions"
        sent = [part for part in body["messages"][1]["content"] if part["type"] == "image_url"]
        assert sent == [missing], "The image with a supplied caption must not be re-described"
        return completion({"records": [{"index": 0, **model_frame}]})

    with service(tmp_path, handler, max_response_images=image_budget) as (client, app, provider, settings):
        response = client.post("/add", json=payload([message(source)]))
        assert response.status_code == 200, response.text
        row, = app.state.storage.list_records("synthetic-user")
        structured = row["structured"]
        assert structured["extraction_source"] == "mixed"
        assert structured["caption"] == "a cobalt telescope"
        assert structured["summary"] == "a cobalt telescope\na copper compass"
        assert structured["ocr"] == ["CP-772"]
        assert structured["provenance"]["model_image_part_indices"] == [3]
        assert structured["provenance"]["image_captions"][0]["image_part_index"] == 1
        assert structured["provenance"]["caption_verified"] is False
        embedding_body = next(body for path, body in requests if path == "/v1/embeddings")
        embedded, = embedding_body["input"]
        raw = text_of(source)
        assert embedded.startswith(raw + "\n")
        assert json.loads(embedded[len(raw) + 1:]) == structured
        assert counts(settings.db_path)["image_occurrences"] == 2
        found = client.post("/search", json={"user_id": "synthetic-user", "query": "CP-772", "top_k": 1})
        assert found.status_code == 200, found.text
        content = found.json()["data"][0]["content"]
        if image_budget:
            assert content[1:] == source
        else:
            assert all(part["type"] == "text" for part in content)
            assert "[Provided caption and model-extracted image index, not original pixels] " in text_of(content)
        assert provider.calls["extract"] == 1
        assert provider.calls["query"] == 0
