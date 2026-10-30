"""Real Provider/SDK -> mocked HTTP -> API -> SQLite integration boundaries.

All messages, images, credentials and upstream responses are synthetic. No
Provider extraction method is replaced; only the external HTTP transport is.
"""

import asyncio
import base64
from contextlib import contextmanager
import copy
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
from fusion_memory.provider import Provider, text_of


CHAT_KEY = "synthetic-api-integration-key-not-real"
PRIVATE_SOURCE = "SYNTHETIC_PRIVATE_SOURCE_83592"
PRIVATE_MODEL_OUTPUT = "SYNTHETIC_PRIVATE_MODEL_OUTPUT_51237"
TABLES = ("requests", "records", "image_blobs", "image_occurrences", "state_versions")


def png_uri():
    output = BytesIO()
    Image.new("RGB", (3, 2), (20, 100, 220)).save(output, format="PNG")
    data = output.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode(), data


def missing_caption_content(text, uri=None):
    """Only uncaptained images should reach Lite's model/format boundary."""
    if uri is None:
        uri, _pixels = png_uri()
    return [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": uri}}]


def request_payload(messages, request_id="synthetic-add-1"):
    return {"request_id": request_id, "user_id": "synthetic-user",
            "session_id": "synthetic-session", "messages": messages}


def completion(payload):
    return httpx.Response(200, json={
        "id": "chatcmpl-synthetic-api-integration", "object": "chat.completion",
        "created": 1704067200, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant",
                     "content": json.dumps(payload, ensure_ascii=False)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    })


def source_groups(body):
    groups = []
    for part in body["messages"][1]["content"]:
        if part["type"] == "text" and part["text"].startswith("SOURCE "):
            groups.append({"index": json.loads(part["text"][7:])["index"], "parts": []})
        elif groups:
            groups[-1]["parts"].append(part)
    return groups


def valid_rows(body, *, compatible=False):
    rows = []
    for group in source_groups(body):
        evidence = text_of(group["parts"])
        row = {"index": group["index"], "summary": evidence,
               "facts": evidence if compatible else [evidence],
               "entities": [], "objects": [], "ocr": [], "relations": [],
               "event_time": [], "uncertainties": [], "state_updates": []}
        if "Citylab" in evidence:
            row["state_updates"] = [{"subject": "synthetic-user", "predicate": "lives_in", "value": "Citylab"}]
        rows.append(row)
    # The persistence order must follow SOURCE indices, never response order.
    return list(reversed(rows))


@contextmanager
def real_provider_client(tmp_path, handler, **overrides):
    settings = Settings(db_path=str(tmp_path / "memory.sqlite3"),
                        api_key=CHAT_KEY, api_base="https://chat.example.invalid/v1",
                        embedding_key="synthetic-embedding-key-not-real",
                        embedding_base="https://embedding.example.invalid/v1",
                        embedding_enabled=False, llm_concurrency=1, **overrides)
    assert settings.extraction_output_mode == "auto"
    provider = Provider(settings)
    # The constructor owns two normal SDK clients. Close them before replacing
    # their HTTP transports, so no real client is available during the test.
    asyncio.run(provider.close())
    provider.client = AsyncOpenAI(
        api_key=settings.api_key, base_url=settings.api_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    def reject_embedding_network(request):
        raise AssertionError("Embedding is disabled; no external request is expected")

    provider.embedding_client = AsyncOpenAI(
        api_key=settings.embedding_key, base_url=settings.embedding_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(reject_embedding_network)),
    )
    app = create_app(settings, provider=provider)
    with TestClient(app) as client:
        yield client, app.state.storage, provider, settings


def table_counts(path):
    with sqlite3.connect(path) as connection:
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in TABLES}


def test_compatible_shape_is_normalized_once_and_preserves_source_mapping_and_original_image(tmp_path, caplog):
    requests = []
    uri, pixels = png_uri()
    content = [{"type": "text", "text": "synthetic-alpha telescope before image"},
               {"type": "image_url", "image_url": {"url": uri, "detail": "high"}},
               {"type": "text", "text": "after image " + PRIVATE_SOURCE}]
    payload = request_payload([
        {"role": "user", "timestamp": 1704067200000, "content": content},
        {"role": "assistant", "timestamp": 1704067201000, "content": "synthetic-beta response"},
    ])

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        # A compatibility gateway can return a scalar despite strict schema.
        return completion({"records": valid_rows(body, compatible=True)})

    with real_provider_client(tmp_path, handler) as (client, store, provider, settings):
        result = client.post("/add", json=payload)
        assert result.status_code == 200, result.text
        assert result.json() == {"success": True, **{key: payload[key] for key in ("request_id", "user_id", "session_id")}}
        assert len(requests) == 1
        assert provider.calls["extract"] == 1
        assert [len(source_groups(body)) for body in requests] == [1]
        assert source_groups(requests[0])[0]["parts"] == content
        records = store.list_records(payload["user_id"])
        assert len(records) == 2
        for record, message in zip(records, payload["messages"], strict=True):
            evidence = text_of(message["content"])
            assert record["structured"]["summary"] == evidence
            is_image = not isinstance(message["content"], str)
            assert record["structured"]["facts"] == ([evidence] if is_image else [])
            assert record["structured"]["extraction_source"] == ("model" if is_image else "raw_text")
            if not is_image:
                for field in ("entities", "objects", "ocr", "relations", "event_time",
                              "uncertainties", "state_updates"):
                    assert record["structured"][field] == []
            assert store.get_content(payload["user_id"], record["id"]) == message["content"]
        assert records[1]["structured"]["summary"] not in records[0]["structured"]["facts"]
        restored = store.get_content(payload["user_id"], records[0]["id"])
        assert base64.b64decode(restored[1]["image_url"]["url"].split(",", 1)[1]) == pixels
        search = client.post("/search", json={"user_id": payload["user_id"], "query": "synthetic-alpha telescope", "top_k": 1})
        assert search.status_code == 200, search.text
        assert search.json()["data"][0]["content"][1:] == content
        assert len(requests) == 1  # Plain-text search and disabled embeddings make no upstream call.
    assert CHAT_KEY not in caplog.text
    assert PRIVATE_SOURCE not in caplog.text


def test_later_image_batch_failure_rolls_back_fast_paths_recovers_and_is_idempotent(tmp_path, caplog):
    requests = []
    fail_last_batch = True
    uri, _pixels = png_uri()
    first_content = [{"type": "text", "text": "synthetic-source-0 I live in Citylab " + PRIVATE_SOURCE},
                     {"type": "image_url", "image_url": {"url": uri}}]
    payload = request_payload([
        {"role": "user", "timestamp": 1704067200000 + i,
         "content": (first_content if i == 0 else
                     missing_caption_content(f"synthetic-source-{i} telescope evidence", uri)
                     if i in (2, 3, 4, 5) else f"synthetic-source-{i} telescope evidence")}
        for i in range(7)
    ])

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        groups = source_groups(body)
        if fail_last_batch and len(groups) == 1:
            return completion({"records": [{"index": {"private": PRIVATE_MODEL_OUTPUT + CHAT_KEY},
                                             "summary": "invalid source index"}]})
        return completion({"records": valid_rows(body)})

    with real_provider_client(tmp_path, handler) as (client, store, provider, settings):
        failed = client.post("/add", json=payload)
        assert failed.status_code == 503, failed.text
        assert failed.json() == {"detail": "extraction returned invalid schema; Add has not been committed"}
        assert [len(source_groups(body)) for body in requests] == [4, 1, 1]
        assert "FORMAT CORRECTION" in requests[-1]["messages"][0]["content"]
        assert requests[-1]["messages"][1] == requests[-2]["messages"][1]
        assert PRIVATE_MODEL_OUTPUT not in json.dumps(requests)
        assert table_counts(settings.db_path) == dict.fromkeys(TABLES, 0)
        assert store.list_records(payload["user_id"]) == []
        assert PRIVATE_MODEL_OUTPUT not in failed.text
        assert PRIVATE_SOURCE not in failed.text
        assert CHAT_KEY not in failed.text

        fail_last_batch = False
        recovered = client.post("/add", json=payload)
        assert recovered.status_code == 200, recovered.text
        assert len(requests) == 5  # Both image batches re-extracted; raw-text fast paths use no call.
        records = store.list_records(payload["user_id"])
        assert len(records) == 7
        for record, message in zip(records, payload["messages"], strict=True):
            evidence = text_of(message["content"])
            assert record["structured"]["summary"] == evidence
            is_image = not isinstance(message["content"], str)
            assert record["structured"]["facts"] == ([evidence] if is_image else [])
            assert record["structured"]["extraction_source"] == ("model" if is_image else "raw_text")
            assert store.get_content(payload["user_id"], record["id"]) == message["content"]
        assert table_counts(settings.db_path) == {
            "requests": 1, "records": 7, "image_blobs": 1, "image_occurrences": 5, "state_versions": 1,
        }
        ids = [record["id"] for record in records]
        before_calls = dict(provider.calls)
        retry = client.post("/add", json=payload)
        assert retry.status_code == 200, retry.text
        assert retry.json() == recovered.json()
        assert provider.calls == before_calls
        assert len(requests) == 5
        assert [record["id"] for record in store.list_records(payload["user_id"])] == ids
        conflicting = copy.deepcopy(payload)
        conflicting["messages"][-1]["content"] = "changed source under the same request identifier"
        assert client.post("/add", json=conflicting).status_code == 409
        assert provider.calls == before_calls
        assert [record["id"] for record in store.list_records(payload["user_id"])] == ids
    assert PRIVATE_MODEL_OUTPUT not in caplog.text
    assert PRIVATE_SOURCE not in caplog.text
    assert CHAT_KEY not in caplog.text


@pytest.mark.parametrize("empty_row", [
    {"index": 0},
    {"index": 0, "summary": "   ", "facts": [], "entities": [], "objects": [], "ocr": [],
     "relations": [], "event_time": [], "uncertainties": [], "state_updates": []},
    {"index": 0, "unknown_field": PRIVATE_MODEL_OUTPUT},
    {"index": 0, "facts": {}},
    {"index": 0, "facts": [{}]},
    {"index": 0, "facts": [[]]},
    {"index": 0, "facts": {"text": ""}},
])
def test_empty_or_unknown_only_frame_is_not_api_success(tmp_path, caplog, empty_row):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion({"records": [empty_row]})

    payload = request_payload([{"role": "user", "content": missing_caption_content("synthetic nonempty source")}])
    with real_provider_client(tmp_path, handler) as (client, _store, _provider, settings):
        result = client.post("/add", json=payload)
        assert result.status_code == 503, result.text
        assert len(requests) == 2
        assert table_counts(settings.db_path) == dict.fromkeys(TABLES, 0)
        assert PRIVATE_MODEL_OUTPUT not in result.text
    assert PRIVATE_MODEL_OUTPUT not in caplog.text


def test_explicit_format_rejection_falls_back_once_and_caches_capability_without_exposing_key(tmp_path, caplog):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {
                "message": "response_format json_schema is not supported " + PRIVATE_MODEL_OUTPUT + CHAT_KEY,
                "param": "response_format", "code": "unsupported_parameter",
            }})
        return completion({"records": valid_rows(body, compatible=True)})

    with real_provider_client(tmp_path, handler) as (client, _store, _provider, settings):
        first = client.post("/add", json=request_payload([{"role": "user", "content": missing_caption_content("synthetic-alpha telescope")}]))
        assert first.status_code == 200, first.text
        assert [body["response_format"]["type"] for body in requests] == ["json_schema", "json_object"]
        assert requests[0]["messages"] == requests[1]["messages"]
        second = client.post("/add", json=request_payload(
            [{"role": "user", "content": missing_caption_content("synthetic-beta telescope")}], request_id="synthetic-add-2"))
        assert second.status_code == 200, second.text
        assert [body["response_format"]["type"] for body in requests] == ["json_schema", "json_object", "json_object"]
        assert table_counts(settings.db_path)["records"] == 2
    assert PRIVATE_MODEL_OUTPUT not in caplog.text
    assert CHAT_KEY not in caplog.text


def test_permission_failure_returns_safe_503_and_does_not_trigger_format_fallback_or_commit(tmp_path, caplog):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(403, json={"error": {
            "message": "response_format unsupported private upstream detail " + PRIVATE_MODEL_OUTPUT + CHAT_KEY,
            "param": "response_format", "code": "unsupported_parameter",
        }})

    payload = request_payload([{"role": "user", "content": missing_caption_content(PRIVATE_SOURCE)}])
    with real_provider_client(tmp_path, handler) as (client, _store, _provider, settings):
        result = client.post("/add", json=payload)
        assert result.status_code == 503, result.text
        assert len(requests) == 1
        assert table_counts(settings.db_path) == dict.fromkeys(TABLES, 0)
        for private in (CHAT_KEY, PRIVATE_MODEL_OUTPUT, PRIVATE_SOURCE):
            assert private not in result.text
    for private in (CHAT_KEY, PRIVATE_MODEL_OUTPUT, PRIVATE_SOURCE):
        assert private not in caplog.text


def test_plain_text_add_preserves_raw_source_without_creating_model_claims_or_states(tmp_path):
    def unexpected_request(request):
        pytest.fail("Plain-text Add and Search must not invoke chat when embeddings are disabled")

    source = "synthetic lighthouse I live in Citylab.\n" * 700 + "RAW-END-MARKER"
    content_parts = [{"type": "text", "text": "synthetic reply"},
                     {"type": "text", "text": "yes"}]
    payload = request_payload([
        {"role": "user", "content": source},
        {"role": "assistant", "content": content_parts},
    ])
    with real_provider_client(tmp_path, unexpected_request) as (client, store, provider, settings):
        response = client.post("/add", json=payload)
        assert response.status_code == 200, response.text
        records = store.list_records(payload["user_id"])
        assert len(records) == 2
        for record, message in zip(records, payload["messages"], strict=True):
            assert store.get_content(payload["user_id"], record["id"]) == message["content"]
            frame = record["structured"]
            assert frame["summary"] == text_of(message["content"])
            assert frame["extraction_source"] == "raw_text"
            assert frame["provenance"]["model_called"] is False
            for field in ("facts", "entities", "objects", "ocr", "relations", "event_time",
                          "uncertainties", "state_updates"):
                assert frame[field] == []
        assert table_counts(settings.db_path) == {
            "requests": 1, "records": 2, "image_blobs": 0, "image_occurrences": 0, "state_versions": 0,
        }
        search = client.post("/search", json={"user_id": payload["user_id"],
                                             "query": "synthetic lighthouse", "top_k": 1})
        assert search.status_code == 200, search.text
        assert set(search.json()) == {"data"}
        evidence = search.json()["data"][0]
        assert set(evidence) == {"id", "content", "score", "created_at"}
        assert source in evidence["content"]
        assert provider.calls == {"extract": 0, "query": 0, "embedding": 0, "failed": 0}
