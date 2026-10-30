"""Exercise the real OpenAI SDK serialization through an in-memory transport.

No network requests, real credentials or model calls are used. These tests cover
the upstream protocol and failure boundary rather than substituting Provider
methods, so an SDK/template integration regression is observable.
"""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json

import httpx
from openai import AsyncOpenAI
import pytest

from fusion_memory.config import Settings
from fusion_memory.provider import Provider, UpstreamError


CHAT_KEY = "unit-test-chat-secret-not-real"
EMBEDDING_KEY = "unit-test-embedding-secret-not-real"
CHAT_BASE = "https://chat.example.invalid/v1"
EMBEDDING_BASE = "https://embedding.example.invalid/v1"
PNG_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
    "7jEAAAAASUVORK5CYII="
)


def missing_caption_content(text):
    """Synthetic image with no supplied caption: Lite must use GPT fallback."""
    return [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": PNG_URI}}]


def completion(payload, *, finish_reason="stop", raw=False):
    return httpx.Response(200, json={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1704067200,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": payload if raw else json.dumps(payload, ensure_ascii=False),
            },
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 17, "completion_tokens": 9, "total_tokens": 26},
    })


def source_groups(body):
    """Decode SOURCE framing from the actual serialized SDK request."""
    groups = []
    current = None
    for part in body["messages"][1]["content"]:
        if part["type"] == "text" and part["text"].startswith("SOURCE "):
            current = {"label": json.loads(part["text"][7:]), "parts": []}
            groups.append(current)
        elif current is not None:
            current["parts"].append(part)
    return groups


@asynccontextmanager
async def mocked_provider(handler, **overrides):
    settings = Settings(
        api_key=CHAT_KEY, api_base=CHAT_BASE,
        embedding_key=EMBEDDING_KEY, embedding_base=EMBEDDING_BASE,
        **overrides,
    )
    provider = Provider(settings)
    # Close constructor-owned clients before replacing their transport.
    await provider.close()
    provider.client = AsyncOpenAI(
        api_key=settings.api_key, base_url=settings.api_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider.embedding_client = AsyncOpenAI(
        api_key=settings.embedding_key, base_url=settings.embedding_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        yield provider
    finally:
        await provider.close()


def test_extract_uses_official_multimodal_chat_template_and_maps_source_indices():
    requests = []
    parts = [
        {"type": "text", "text": "这是我的杯子。"},
        {"type": "image_url", "image_url": {"url": PNG_URI, "detail": "high"}},
        {"type": "text", "text": "请记住标签上的编号。"},
    ]
    second_parts = missing_caption_content("另一张图片。")
    messages = [
        {"role": "user", "content": "first raw source"},
        {"role": "user", "timestamp": 1704067200000, "content": parts},
        {"role": "assistant", "content": "我会记住。"},
        {"role": "assistant", "content": second_parts},
    ]

    def handler(request):
        body = json.loads(request.content)
        requests.append((request, body))
        # Return valid records out of order: response order is not source order.
        return completion({"records": [
            {"index": 1, "summary": "另一张图片", "facts": ["second image fact"]},
            {"index": 0, "summary": "用户的杯子", "ocr": ["AB-42"], "entities": ["杯子"]},
        ]})

    async def exercise():
        async with mocked_provider(handler) as provider:
            frames = await provider.extract(messages, history="user: earlier context")
            assert len(frames) == 4
            assert frames[1]["summary"] == "用户的杯子"
            assert frames[1]["ocr"] == ["AB-42"]
            assert frames[3]["summary"] == "另一张图片"
            assert frames[3]["facts"] == ["second image fact"]
            assert frames[1]["extraction_source"] == frames[3]["extraction_source"] == "model"
            for index in (0, 2):
                assert frames[index]["summary"] == messages[index]["content"]
                assert frames[index]["extraction_source"] == "raw_text"
                for field in ("facts", "entities", "objects", "ocr", "relations", "event_time",
                              "uncertainties", "state_updates"):
                    assert frames[index][field] == []
            assert provider.usage == {"prompt_tokens": 17, "completion_tokens": 9}

    asyncio.run(exercise())
    assert len(requests) == 1
    request, body = requests[0]
    assert request.method == "POST"
    assert str(request.url) == CHAT_BASE + "/chat/completions"
    assert request.headers["authorization"] == "Bearer " + CHAT_KEY
    assert body["model"] == "gpt-4o-mini"
    response_format = body["response_format"]
    assert response_format["type"] == "json_schema"
    structured = response_format["json_schema"]
    assert structured["strict"] is True
    assert structured["name"] == "memory_extraction"
    schema = structured["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["records"]
    assert schema["additionalProperties"] is False
    records_schema = schema["properties"]["records"]
    assert records_schema["type"] == "array"
    assert records_schema["minItems"] == records_schema["maxItems"] == 2
    row = records_schema["items"]
    assert row["type"] == "object"
    assert row["additionalProperties"] is False
    expected_fields = {"index", "summary", "facts", "entities", "objects", "ocr", "relations",
                       "event_time", "uncertainties", "state_updates"}
    assert set(row["required"]) == set(row["properties"]) == expected_fields
    assert row["properties"]["index"] == {"type": "integer", "enum": [0, 1]}
    assert row["properties"]["summary"] == {"type": "string"}
    for name in expected_fields - {"index", "summary", "state_updates"}:
        assert row["properties"][name] == {"type": "array", "items": {"type": "string"}}
    states = row["properties"]["state_updates"]
    assert states["type"] == "array"
    state = states["items"]
    assert state["type"] == "object"
    assert state["additionalProperties"] is False
    assert set(state["required"]) == {"subject", "predicate", "value"}
    assert state["properties"] == {key: {"type": "string"} for key in state["required"]}
    assert body["temperature"] == 0
    assert body["messages"][0]["role"] == "system"
    assert "untrusted source data" in body["messages"][0]["content"]
    assert "Do not generate an answer" in body["messages"][0]["content"]
    assert body["messages"][1]["content"][0]["text"].startswith("PRIOR CONTEXT (not new memory):")
    groups = source_groups(body)
    assert groups[0]["label"] == {"index": 0, "role": "user", "timestamp_unix_ms": 1704067200000}
    assert groups[0]["parts"] == parts  # exact image URI, detail and interleaving survive
    assert groups[1]["label"]["role"] == "assistant"
    assert groups[1]["parts"] == second_parts
    assert len(groups) == 2  # Fast-path source messages have no extraction SOURCE index.
    assert messages[1]["content"] == parts


def test_long_source_and_multiple_images_are_all_sent_and_merged_without_source_mixup():
    long_text = "A" * 9999 + "甲" + "B" * 9999 + "乙" + "C" * 8501
    image_parts = [
        {"type": "image_url", "image_url": {"url": PNG_URI, "detail": "low"}}
        for _ in range(5)
    ]
    original_parts = [{"type": "text", "text": long_text}, *image_parts,
                      {"type": "text", "text": "END-OF-LONG-SOURCE"}]
    seen = []

    def handler(request):
        body = json.loads(request.content)
        groups = source_groups(body)
        seen.extend(groups)
        rows = []
        for group in groups:
            content = "".join(p["text"] for p in group["parts"] if p["type"] == "text")
            digest = hashlib.sha256(content.encode()).hexdigest()
            rows.append({"index": group["label"]["index"], "summary": digest, "facts": [digest]})
        return completion({"records": list(reversed(rows))})

    async def exercise():
        async with mocked_provider(handler, llm_concurrency=1) as provider:
            frames = await provider.extract([
                {"role": "user", "timestamp": 1000, "content": original_parts},
                {"role": "assistant", "timestamp": 2000, "content": "SEPARATE-SOURCE"},
            ])
            long_groups = [g for g in seen if g["label"]["timestamp_unix_ms"] == 1000]
            assert len(long_groups) > 1
            received_parts = [p for g in long_groups for p in g["parts"]]
            assert "".join(p["text"] for p in received_parts if p["type"] == "text") == long_text + "END-OF-LONG-SOURCE"
            assert [p for p in received_parts if p["type"] == "image_url"] == image_parts
            expected_digests = [hashlib.sha256("".join(
                p["text"] for p in g["parts"] if p["type"] == "text"
            ).encode()).hexdigest() for g in long_groups]
            assert frames[0]["summary"].splitlines() == expected_digests
            assert frames[0]["facts"] == list(dict.fromkeys(expected_digests))
            assert frames[1]["summary"] == "SEPARATE-SOURCE"
            assert frames[1]["facts"] == []
            assert frames[1]["extraction_source"] == "raw_text"
            assert all(g["label"]["timestamp_unix_ms"] == 1000 for g in seen)
            assert provider.calls["extract"] > 1

    asyncio.run(exercise())


@pytest.mark.parametrize("payload", [
    {},
    {"records": []},
    {"records": [{"index": 5, "summary": "wrong source"}]},
    {"records": [{"index": 0, "state_updates": [{"subject": "owner", "predicate": "location"}]}]},
    {"records": [{"index": 0, "summary": "a"}, {"index": 0, "summary": "b"}]},
    {"records": [{"index": False, "summary": "boolean is not a source index"}]},
    {"records": [{"index": 0.5, "summary": "fractional source index"}]},
    {"records": [{"index": "not-an-index", "summary": "invalid source index"}]},
    {"records": [{"index": -1, "summary": "negative index"}]},
    {"records": [{"index": [], "summary": "unhashable index"}]},
    {"records": ["not a record object"]},
])
def test_invalid_extraction_schema_never_returns_success(payload):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion(payload)

    async def exercise():
        async with mocked_provider(handler) as provider:
            count = 2 if len(payload.get("records", [])) == 2 else 1
            with pytest.raises(UpstreamError, match="^extraction returned invalid schema; Add has not been committed$"):
                await provider.extract([{"role": "user", "content": missing_caption_content("test")} for _ in range(count)])
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 2
    asyncio.run(exercise())
    assert len(requests) == 2
    assert "FORMAT CORRECTION" not in requests[0]["messages"][0]["content"]
    assert "FORMAT CORRECTION" in requests[1]["messages"][0]["content"]


def test_invalid_schema_retries_once_with_original_evidence_and_safe_format_hint(caplog):
    requests = []
    invalid_output_marker = "MALFORMED-MODEL-PRIVATE-OUTPUT-DO-NOT-ECHO"
    parts = [
        {"type": "text", "text": "标签上的编号是 AB-42。"},
        {"type": "image_url", "image_url": {"url": PNG_URI, "detail": "high"}},
    ]

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return completion({"records": [{"index": 0, "state_updates": [
                {"subject": invalid_output_marker, "predicate": "location"}]}]})
        return completion({"records": [{"index": 0, "summary": "标签编号", "facts": ["编号是 AB-42"], "ocr": ["AB-42"]}]})

    async def exercise():
        async with mocked_provider(handler) as provider:
            result = await provider.extract(
                [{"role": "user", "timestamp": 1704067200000, "content": parts}],
                history="Earlier context remains source context only.",
            )
            assert result[0]["facts"] == ["编号是 AB-42"]
            assert result[0]["ocr"] == ["AB-42"]
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 1
            assert provider.usage == {"prompt_tokens": 34, "completion_tokens": 18}

    asyncio.run(exercise())
    assert len(requests) == 2
    assert requests[1]["messages"][1] == requests[0]["messages"][1]
    assert source_groups(requests[1])[0]["parts"] == parts
    assert len(requests[1]["messages"]) == 2  # no malformed assistant response replay
    hint = requests[1]["messages"][0]["content"]
    assert "FORMAT CORRECTION" in hint
    assert "arrays of strings" in hint
    assert "do not change source facts" in hint
    assert invalid_output_marker not in json.dumps(requests)
    assert invalid_output_marker not in caplog.text
    assert CHAT_KEY not in caplog.text
    assert EMBEDDING_KEY not in caplog.text
    assert parts[0]["text"] not in caplog.text
    assert "attempt=1 max_attempts=2" in caplog.text


@pytest.mark.parametrize("response", [
    lambda: completion({"records": [{"index": 0, "summary": "partial"}]}, finish_reason="length"),
    lambda: completion("not JSON", raw=True),
    lambda: completion(["not an object"]),
])
def test_truncated_or_invalid_json_completion_is_an_upstream_failure(response):
    async def exercise():
        async with mocked_provider(lambda request: response()) as provider:
            with pytest.raises(UpstreamError):
                await provider.extract([{"role": "user", "content": missing_caption_content("source")}])
            assert provider.calls["extract"] == 1
            assert provider.calls["failed"] == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["connection", "timeout", "rate_limit", "server_error"])
def test_upstream_failures_do_not_gain_a_schema_retry(failure):
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "connection":
            raise httpx.ConnectError("synthetic connection failure", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        return httpx.Response(429 if failure == "rate_limit" else 503, json={"error": {"message": "unavailable"}})

    async def exercise():
        # mocked_provider disables SDK retries, exposing any new application
        # retry. Production's existing SDK max_retries setting is unchanged.
        async with mocked_provider(handler) as provider:
            with pytest.raises(UpstreamError, match="extract model call failed"):
                await provider.extract([{"role": "user", "content": missing_caption_content("source")}])
            assert provider.calls["extract"] == 1
            assert provider.calls["failed"] == 1

    asyncio.run(exercise())
    assert len(requests) == 1


def test_network_failure_during_format_retry_is_not_retried_again():
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return completion({"records": [{"index": 0, "state_updates": [{"subject": "owner"}]}]})
        raise httpx.ConnectError("synthetic connection failure", request=request)

    async def exercise():
        async with mocked_provider(handler) as provider:
            with pytest.raises(UpstreamError, match="extract model call failed"):
                await provider.extract([{"role": "user", "content": missing_caption_content("source")}])
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 2

    asyncio.run(exercise())
    assert len(requests) == 2


def test_upstream_403_does_not_expose_credentials_or_response_body(caplog):
    private_text = "PRIVATE-UPSTREAM-BODY: " + CHAT_KEY

    def handler(request):
        return httpx.Response(403, json={"error": {"message": private_text, "type": "permission_error"}})

    async def exercise():
        async with mocked_provider(handler) as provider:
            with pytest.raises(UpstreamError) as caught:
                await provider.extract([{"role": "user", "content": missing_caption_content("private user evidence")}])
            assert CHAT_KEY not in str(caught.value)
            assert private_text not in str(caught.value)
            assert "private user evidence" not in str(caught.value)
            assert provider.calls["failed"] == 1

    asyncio.run(exercise())
    assert "error_type=PermissionDeniedError" in caplog.text
    assert CHAT_KEY not in caplog.text
    assert private_text not in caplog.text
    assert "private user evidence" not in caplog.text


def test_image_query_indexes_features_preserves_question_and_does_not_consume_answer_field():
    requests = []
    parts = [
        {"type": "text", "text": "Which earlier photo has this mug?"},
        {"type": "image_url", "image_url": {"url": PNG_URI}},
    ]

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return completion({
            "description": "Red mug with white stars",
            "terms": ["mug", "stars", "红色"],
            "answer": "DO-NOT-RETURN-THIS-FINAL-ANSWER",
        })

    async def exercise():
        async with mocked_provider(handler) as provider:
            query = await provider.query_text(parts, options=["pic-A", "pic-B"])
            assert query.startswith(parts[0]["text"])
            assert "Red mug with white stars" in query
            assert "mug stars 红色" in query
            assert "pic-A\npic-B" in query
            assert "DO-NOT-RETURN-THIS-FINAL-ANSWER" not in query
            assert provider.calls["query"] == 1
            assert provider.calls["extract"] == 0

    asyncio.run(exercise())
    assert requests[0]["messages"][1]["content"] == parts
    assert "Do not answer the question or select any option" in requests[0]["messages"][0]["content"]
    assert requests[0]["model"] == "gpt-4o-mini"


def test_text_only_add_query_and_disabled_embeddings_need_no_upstream_call():
    def unexpected_request(request):
        pytest.fail("text-only Add/query and disabled embeddings should not call the network")

    async def exercise():
        async with mocked_provider(unexpected_request) as provider:
            source = "原始消息 A\n" * 7000 + "TAIL-MUST-BE-PRESERVED"
            parts = [{"type": "text", "text": "short source"},
                     {"type": "text", "text": "yes"}]
            frames = await provider.extract([
                {"role": "user", "content": source},
                {"role": "assistant", "content": parts},
            ])
            assert [frame["summary"] for frame in frames] == [source, "short source\nyes"]
            for frame in frames:
                assert frame["extraction_source"] == "raw_text"
                assert frame["provenance"]["model_called"] is False
                for field in ("facts", "entities", "objects", "ocr", "relations", "event_time",
                              "uncertainties", "state_updates"):
                    assert frame[field] == []
            assert await provider.query_text("What is my dog's name?", ["Bobo", "Kiki"]) == "What is my dog's name?\nBobo\nKiki"
            assert await provider.embed(["a", "b"]) == [None, None]
            assert provider.calls == {"extract": 0, "query": 0, "embedding": 0, "failed": 0}
    asyncio.run(exercise())


@pytest.mark.parametrize("dimensions", [2, 1024])
def test_embedding_v4_uses_separate_configuration_and_restores_response_order_across_batches(dimensions):
    requests = []
    inputs = [f"document-{index}" for index in range(13)]

    def handler(request):
        body = json.loads(request.content)
        requests.append((request, body))
        rows = [{"object": "embedding", "index": index,
                 "embedding": [float(text.rsplit("-", 1)[1])] + [1.0] * (dimensions - 1)}
                for index, text in enumerate(body["input"])]
        return httpx.Response(200, json={
            "object": "list", "data": list(reversed(rows)), "model": "text-embedding-v4",
            "usage": {"prompt_tokens": 13, "total_tokens": 13},
        })

    async def exercise():
        async with mocked_provider(handler, embedding_enabled=True,
                                   embedding_dimensions=dimensions) as provider:
            assert await provider.embed(inputs) == [
                [float(index)] + [1.0] * (dimensions - 1) for index in range(13)
            ]
            assert provider.calls["embedding"] == 2
    asyncio.run(exercise())
    assert [text for _, body in requests for text in body["input"]] == inputs
    for request, body in requests:
        assert str(request.url) == EMBEDDING_BASE + "/embeddings"
        assert request.headers["authorization"] == "Bearer " + EMBEDDING_KEY
        assert body["model"] == "text-embedding-v4"
        assert body["encoding_format"] == "float"
        assert body["dimensions"] == dimensions


def test_embedding_count_mismatch_fails_instead_of_silently_misaligning_records():
    def handler(request):
        return httpx.Response(200, json={
            "object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "text-embedding-v4", "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    async def exercise():
        async with mocked_provider(handler, embedding_enabled=True,
                                   embedding_dimensions=2) as provider:
            with pytest.raises(UpstreamError, match="no silent model substitution"):
                await provider.embed(["first source", "second source"])
            assert provider.calls["failed"] == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("returned_dimensions", [1023, 1025])
def test_embedding_dimension_mismatch_rejects_entire_batch(returned_dimensions):
    def handler(request):
        body = json.loads(request.content)
        assert body["dimensions"] == 1024
        # Count and indices are correct; only the second vector is malformed.
        return httpx.Response(200, json={
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1] * 1024},
                {"object": "embedding", "index": 1, "embedding": [0.2] * returned_dimensions},
            ],
            "model": "text-embedding-v4",
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })

    async def exercise():
        async with mocked_provider(handler, embedding_enabled=True,
                                   embedding_dimensions=1024) as provider:
            with pytest.raises(UpstreamError):
                await provider.embed(["first source", "second source"])
            assert provider.calls["embedding"] == 1
            assert provider.calls["failed"] == 1
    asyncio.run(exercise())


@pytest.mark.parametrize("mode,expected_format", [("auto", "json_schema"), ("strict", "json_schema"), ("json", "json_object")])
def test_explicit_output_modes_send_the_selected_format(mode, expected_format):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        return completion({"records": [{"index": 0, "facts": ["source fact"]}]})

    async def exercise():
        async with mocked_provider(handler, extraction_output_mode=mode) as provider:
            for _ in range(2):
                frames = await provider.extract([{"role": "user", "content": missing_caption_content("source fact")}])
                assert frames[0]["facts"] == ["source fact"]
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 0

    asyncio.run(exercise())
    assert [request["response_format"]["type"] for request in requests] == [expected_format] * 2
    if mode == "json":
        assert all(request["response_format"] == {"type": "json_object"} for request in requests)
    else:
        for request in requests:
            rows = request["response_format"]["json_schema"]["schema"]["properties"]["records"]
            assert rows["minItems"] == rows["maxItems"] == 1
            assert rows["items"]["properties"]["index"]["enum"] == [0]


@pytest.mark.parametrize("status", [400, 422])
def test_explicit_unsupported_auto_fallback_is_once_and_cached(status, caplog):
    requests = []
    private_error = "PRIVATE-UPSTREAM-RESPONSE " + CHAT_KEY

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(status, json={"error": {
                "message": "response_format json_schema is not supported. " + private_error,
                "param": "response_format", "code": "unsupported_value",
            }})
        return completion({"records": [{"index": 0, "facts": ["retained source fact"]}]})

    async def exercise():
        async with mocked_provider(handler) as provider:
            for _ in range(2):
                frames = await provider.extract([{"role": "user", "content": missing_caption_content("PRIVATE-USER-SOURCE")}])
                assert frames[0]["facts"] == ["retained source fact"]
            assert provider.calls["extract"] == 3
            assert provider.calls["failed"] == 0

    asyncio.run(exercise())
    assert [request["response_format"]["type"] for request in requests] == ["json_schema", "json_object", "json_object"]
    assert requests[0]["messages"] == requests[1]["messages"]
    assert caplog.text.count("extraction_output_format_fallback reason=explicit_unsupported") == 1
    assert private_error not in caplog.text
    assert "PRIVATE-USER-SOURCE" not in caplog.text
    assert CHAT_KEY not in caplog.text
    assert EMBEDDING_KEY not in caplog.text


def test_concurrent_adds_share_one_unsupported_probe_and_cached_fallback():
    requests = []

    async def exercise():
        probe_entered = asyncio.Event()
        release_probe = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if body["response_format"]["type"] == "json_schema":
                probe_entered.set()
                await release_probe.wait()
                return httpx.Response(400, json={"error": {
                    "message": "This model does not support response_format json_schema",
                    "param": "response_format", "code": "unsupported_parameter",
                }})
            await asyncio.sleep(0)
            groups = source_groups(body)
            return completion({"records": [{"index": group["label"]["index"], "summary": group["parts"][0]["text"]} for group in groups]})

        async with mocked_provider(handler, llm_concurrency=4) as provider:
            tasks = [asyncio.create_task(provider.extract([{"role": "user", "content": missing_caption_content(f"source-{index}")}])) for index in range(8)]
            try:
                await asyncio.wait_for(probe_entered.wait(), timeout=2)
                await asyncio.sleep(0)
                release_probe.set()
                results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
                assert [frames[0]["summary"] for frames in results] == [f"source-{index}" for index in range(8)]
                # A subsequent extraction must also use the remembered capability.
                await provider.extract([{"role": "user", "content": missing_caption_content("later source")}])
                assert provider.calls["extract"] == 10  # 9 successful responses + 1 capability rejection
            finally:
                release_probe.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(exercise())
    formats = [request["response_format"]["type"] for request in requests]
    assert formats.count("json_schema") == 1
    assert formats.count("json_object") == 9


@pytest.mark.parametrize("status", [400, 422])
def test_strict_mode_never_downgrades_even_for_explicit_unsupported(status, caplog):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": "json_schema response_format is not supported", "code": "unsupported_parameter", "param": "response_format"}})

    async def exercise():
        async with mocked_provider(handler, extraction_output_mode="strict") as provider:
            with pytest.raises(UpstreamError, match="extract model call failed"):
                await provider.extract([{"role": "user", "content": missing_caption_content("source")}])
            assert provider.calls["extract"] == provider.calls["failed"] == 1

    asyncio.run(exercise())
    assert len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "extraction_output_format_fallback" not in caplog.text


@pytest.mark.parametrize("status,error", [
    (400, {"message": "context length exceeded", "param": "messages", "code": "context_length_exceeded"}),
    (400, {"message": "unsupported image format", "param": "messages", "code": "unsupported_value"}),
    (400, {"message": "json_schema is not supported because the schema must define additionalProperties", "param": "response_format", "code": "invalid_json_schema"}),
    (422, {"message": "invalid schema: unsupported field in response_format", "param": "response_format", "code": "invalid_schema"}),
    (400, {"message": "response_format validation failed", "param": "response_format", "code": "invalid_parameter"}),
    (401, {"message": "response_format json_schema not supported", "param": "response_format", "code": "unsupported_parameter"}),
    (403, {"message": "response_format json_schema not supported", "param": "response_format", "code": "unsupported_parameter"}),
    (429, {"message": "response_format json_schema not supported", "param": "response_format", "code": "unsupported_parameter"}),
    (500, {"message": "response_format json_schema not supported", "param": "response_format", "code": "unsupported_parameter"}),
    (503, {"message": "response_format json_schema not supported", "param": "response_format", "code": "unsupported_parameter"}),
])
def test_unrelated_invalid_schema_or_operational_errors_never_downgrade(status, error, caplog):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status, json={"error": error})

    async def exercise():
        async with mocked_provider(handler) as provider:
            with pytest.raises(UpstreamError):
                await provider.extract([{"role": "user", "content": missing_caption_content("PRIVATE-USER-SOURCE")}])
            assert provider.calls["extract"] == provider.calls["failed"] == 1

    asyncio.run(exercise())
    assert len(requests) == 1
    assert requests[0]["response_format"]["type"] == "json_schema"
    assert "extraction_output_format_fallback" not in caplog.text
    assert "PRIVATE-USER-SOURCE" not in caplog.text
    assert CHAT_KEY not in caplog.text


@pytest.mark.parametrize("index", [0, "0", 0.0])
@pytest.mark.parametrize("facts", ["source statement kept verbatim", {"subject": "杯子", "color": "red", "count": 2}])
def test_compatible_shapes_preserve_information_without_a_second_model_request(index, facts, caplog):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion({"records": [{"index": index, "summary": "PRIVATE-SOURCE-SUMMARY", "facts": facts}]})

    async def exercise():
        async with mocked_provider(handler) as provider:
            frames = await provider.extract([{"role": "user", "content": missing_caption_content("PRIVATE-USER-SOURCE")}])
            assert frames[0]["summary"] == "PRIVATE-SOURCE-SUMMARY"
            assert len(frames[0]["facts"]) == 1
            if isinstance(facts, str):
                assert frames[0]["facts"] == [facts]
            else:
                assert json.loads(frames[0]["facts"][0]) == facts
            assert provider.calls["extract"] == 1
            assert provider.calls["failed"] == 0

    asyncio.run(exercise())
    assert len(requests) == 1
    assert "FORMAT CORRECTION" not in requests[0]["messages"][0]["content"]
    assert "PRIVATE-SOURCE-SUMMARY" not in caplog.text
    assert "PRIVATE-USER-SOURCE" not in caplog.text
    assert CHAT_KEY not in caplog.text
    assert EMBEDDING_KEY not in caplog.text


def test_failed_batch_cancels_and_awaits_siblings_before_releasing_add():
    requests = []

    async def exercise():
        blocked_entered = asyncio.Event()
        blocked_response = asyncio.Event()
        cancellation_received = asyncio.Event()
        allow_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()
        network_tasks = set()

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            network_tasks.add(asyncio.current_task())
            groups = source_groups(body)
            if len(groups) == 4:
                blocked_entered.set()
                try:
                    await blocked_response.wait()
                except asyncio.CancelledError:
                    cancellation_received.set()
                    # Cancellation is deliberately not instantaneous: the Add
                    # must await sibling cleanup before exposing its failure.
                    await allow_cleanup.wait()
                    cleanup_finished.set()
                    raise
                return completion({"records": [
                    {"index": index, "summary": "blocked source"} for index in range(4)
                ]})
            assert len(groups) == 1
            await blocked_entered.wait()
            return completion({"records": [{"index": 0, "state_updates": [
                {"subject": "owner", "predicate": "location"}]}]})

        # Strict mode avoids the unrelated first-capability-probe lock; both
        # SOURCE batches should actually occupy the two available slots.
        async with mocked_provider(handler, llm_concurrency=2, extraction_output_mode="strict") as provider:
            extraction = asyncio.create_task(provider.extract([
                {"role": "user", "content": missing_caption_content(f"source-{index}")} for index in range(5)
            ]))
            try:
                await asyncio.wait_for(cancellation_received.wait(), timeout=2)
                assert not extraction.done(), "Add returned before the cancelled sibling finished cleanup"
                assert not cleanup_finished.is_set()
                assert [len(source_groups(request)) for request in requests] == [4, 1, 1]
                allow_cleanup.set()
                with pytest.raises(UpstreamError, match="invalid schema; Add has not been committed"):
                    await asyncio.wait_for(extraction, timeout=2)
                assert cleanup_finished.is_set()
                assert len(network_tasks) == 2
                assert all(task.done() for task in network_tasks)
                assert provider.calls["extract"] == 3
                assert provider.calls["failed"] == 2
                # Acquiring every slot proves no cancelled work retains a
                # permit. This uses bounded event waits, not timing sleeps.
                acquired = 0
                try:
                    for _ in range(2):
                        await asyncio.wait_for(provider.semaphore.acquire(), timeout=1)
                        acquired += 1
                    assert provider.semaphore.locked()
                finally:
                    for _ in range(acquired):
                        provider.semaphore.release()
                assert len(requests) == 3
                assert provider.calls["extract"] == 3
            finally:
                blocked_response.set()
                allow_cleanup.set()
                if not extraction.done():
                    extraction.cancel()
                for task in network_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(extraction, *network_tasks, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["auto", "strict", "json"])
def test_nonempty_model_refusal_rejects_even_parseable_valid_json(mode, caplog):
    requests = []
    private_refusal = "PRIVATE-MODEL-REFUSAL " + CHAT_KEY

    def handler(request):
        requests.append(json.loads(request.content))
        payload = completion({"records": [{"index": 0, "facts": ["syntactically valid source fact"]}]}).json()
        payload["choices"][0]["message"]["refusal"] = private_refusal
        return httpx.Response(200, json=payload)

    async def exercise():
        async with mocked_provider(handler, extraction_output_mode=mode) as provider:
            with pytest.raises(UpstreamError, match="extract model call failed") as caught:
                await provider.extract([{"role": "user", "content": missing_caption_content("PRIVATE-USER-SOURCE")}])
            assert private_refusal not in str(caught.value)
            assert provider.calls["extract"] == 1
            assert provider.calls["failed"] == 1

    asyncio.run(exercise())
    assert len(requests) == 1
    assert "extraction_output_format_fallback" not in caplog.text
    assert private_refusal not in caplog.text
    assert "PRIVATE-USER-SOURCE" not in caplog.text
    assert CHAT_KEY not in caplog.text
