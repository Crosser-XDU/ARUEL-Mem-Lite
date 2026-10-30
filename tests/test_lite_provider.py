"""Lite routing through the real SDK with local, synthetic HTTP responses only."""
import asyncio
from contextlib import asynccontextmanager
import copy
import json

import httpx
from openai import AsyncOpenAI
import pytest

from fusion_memory.config import Settings
from fusion_memory.provider import (
    Frame, LITE_CONTEXT_MAX_CHARS, Provider, UpstreamError,
    fallback_context, provided_captions, text_of,
)


def image(name="missing"):
    # Provider routing does not decode images; API media validation is separate.
    return {"type": "image_url", "image_url": {
        "url": f"data:image/png;base64,{name}", "detail": "low"}}


def text(value):
    return {"type": "text", "text": value}


def message(content, role="user", timestamp=1704067200000):
    return {"role": role, "content": content, "timestamp": timestamp}


def completion(rows):
    return httpx.Response(200, json={
        "id": "chatcmpl-lite-test", "object": "chat.completion", "created": 1704067200,
        "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps({"records": rows})}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    })


def source_groups(body):
    groups = []
    for part in body["messages"][1]["content"]:
        if part["type"] == "text" and part["text"].startswith("SOURCE "):
            groups.append({"label": json.loads(part["text"][7:]), "parts": []})
        elif groups:
            groups[-1]["parts"].append(part)
    return groups


def ordinary_completion(request):
    groups = source_groups(json.loads(request.content))
    return completion([{"index": group["label"]["index"],
                        "summary": f"vision-{group['label']['index']}"} for group in reversed(groups)])


@asynccontextmanager
async def mocked_provider(handler, *, api_key="unit-test-not-real", **kwargs):
    settings = Settings(api_key=api_key, api_base="https://lite.example.invalid/v1", **kwargs)
    provider = Provider(settings)
    await provider.close()
    provider.client = AsyncOpenAI(
        api_key=api_key or "unconfigured", base_url=settings.api_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    # Keep embeddings local too, even if a future routing regression invokes it.
    provider.embedding_client = AsyncOpenAI(
        api_key="unit-test-not-real", base_url=settings.api_base, max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        yield provider
    finally:
        await provider.close()


@pytest.mark.parametrize("content", ["", " ", "yes", "事实。\n2025-01-01 09:00:00",
                                      "long" * 10000, [], [text("first"), text("second")]])
def test_plain_text_always_skips_model_and_preserves_full_index_text(content):
    def forbidden(request):
        pytest.fail("Pure text must not invoke a model, even without a key")

    async def exercise():
        original = copy.deepcopy(content)
        async with mocked_provider(forbidden, api_key="") as provider:
            frames = await provider.extract([message(content)], history="earlier context")
            assert frames[0]["summary"] == text_of(content)
            assert frames[0]["extraction_source"] == "raw_text"
            assert "caption" not in frames[0]
            assert frames[0]["provenance"]["model_called"] is False
            assert all(frames[0][field] == [] for field in Frame.model_fields if field != "summary")
            assert provider.calls == {"extract": 0, "query": 0, "embedding": 0, "failed": 0}
            assert content == original
    asyncio.run(exercise())


@pytest.mark.parametrize("label", ["Image Caption:", "IMAGE_CAPTION:", "Image-Caption：", "ImageCaption:"])
def test_supplied_caption_skips_model_and_marks_unverified_source_provenance(label):
    parts = [text(f"speaker: {label} two red mugs, number AB-42"), image("captioned"),
             text("\n2025-09-09 08:10:00")]
    original = copy.deepcopy(parts)

    def forbidden(request):
        pytest.fail("A supplied caption must not be re-described by a model")

    async def exercise():
        async with mocked_provider(forbidden, api_key="") as provider:
            frame, = await provider.extract([message(parts)])
            assert frame["summary"] == frame["caption"] == "two red mugs, number AB-42"
            assert frame["extraction_source"] == "provided_caption"
            assert frame["provenance"] == {"model_called": False, "caption_verified": False,
                "image_captions": [{"image_part_index": 1, "text_part_index": 0,
                                    "caption": "two red mugs, number AB-42"}]}
            assert all(frame[key] == [] for key in Frame.model_fields if key != "summary")
            assert provider.calls["extract"] == 0
            assert parts == original
    asyncio.run(exercise())


@pytest.mark.parametrize("caption", ["", "speaker: ", "Image Caption:", "Image Caption:  \n",
    "Image Caption: \n2025-09-09 08:10:00", "Image Caption: 2025-09-09 08:10:00",
    'Image Caption: ""', "Image Caption: ]", "Image Caption without a colon"])
def test_missing_blank_or_date_only_caption_falls_back(caption):
    requests = []
    parts = [text(caption), image(), text("\n2025-09-09 08:10:00")]
    original = copy.deepcopy(parts)

    def handler(request):
        requests.append(json.loads(request.content))
        return ordinary_completion(request)

    async def exercise():
        async with mocked_provider(handler) as provider:
            frame, = await provider.extract([message(parts)])
            assert frame["extraction_source"] == "model"
            assert frame["summary"] == "vision-0"
            assert "caption" not in frame
            assert provider.calls["extract"] == 1
            assert parts == original
    asyncio.run(exercise())
    # Existing model chunking elides an empty text part; storage input stays exact.
    assert source_groups(requests[0])[0]["parts"] == [
        part for part in parts if part["type"] != "text" or part["text"]]


def test_multiple_images_require_individual_unambiguous_caption_mapping():
    clear = [text("Image Caption: first scene"), image("first"),
             text("Image Caption: second scene"), image("second"), text("2025-09-09 08:10:00")]
    mapped = provided_captions(clear)
    assert [item["image_part_index"] for item in mapped] == [1, 3]
    assert [item["caption"] for item in mapped] == ["first scene", "second scene"]

    async def exercise():
        async with mocked_provider(lambda request: pytest.fail("mapped captions should skip vision")) as provider:
            frame, = await provider.extract([message(clear)])
            assert frame["caption"] == "first scene\nsecond scene"
            assert frame["provenance"]["image_captions"] == mapped
    asyncio.run(exercise())


@pytest.mark.parametrize("parts", [
    [text("Image Caption: first\nImage Caption: second"), image("first"), image("second")],
    [image("first"), text("Image Caption: first"), image("second"), text("Image Caption: second")],
])
def test_ambiguous_multiimage_caption_falls_back_without_dropping_any_image(parts):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return ordinary_completion(request)

    async def exercise():
        async with mocked_provider(handler) as provider:
            frame, = await provider.extract([message(parts)])
            assert frame["extraction_source"] == "model"
            assert provider.calls["extract"] == 1
    asyncio.run(exercise())
    sent_images = [part for group in source_groups(requests[0]) for part in group["parts"] if part["type"] == "image_url"]
    assert sent_images == [part for part in parts if part["type"] == "image_url"]


@pytest.mark.parametrize("separator", [[], [text("Image Caption: ")], [text("extra source context")]])
def test_partial_caption_mapping_sends_only_uncovered_image_and_keeps_both_origins(separator):
    parts = [text("Image Caption: supplied first scene"), image("covered"), *separator, image("uncovered")]
    original = copy.deepcopy(parts)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion([{"index": 0, "summary": "observed second scene", "facts": ["second image fact"]}])

    async def exercise():
        async with mocked_provider(handler) as provider:
            frame, = await provider.extract([message(parts)])
            assert frame["extraction_source"] == "mixed"
            assert frame["caption"] == "supplied first scene"
            assert frame["summary"] == "supplied first scene\nobserved second scene"
            assert frame["facts"] == ["second image fact"]
            assert frame["provenance"]["image_captions"] == [
                {"image_part_index": 1, "text_part_index": 0, "caption": "supplied first scene"}]
            assert frame["provenance"]["model_image_part_indices"] == [len(parts) - 1]
            assert provider.calls["extract"] == 1
            assert parts == original
    asyncio.run(exercise())
    sent_images = [part for group in source_groups(requests[0]) for part in group["parts"] if part["type"] == "image_url"]
    assert sent_images == [image("uncovered")]
    assert "data:image/png;base64,covered" not in json.dumps(requests[0])


def test_mixed_fastpaths_preserve_global_order_and_send_only_missing_caption_images():
    messages = [message("FIRST TEXT"),
                message([text("speaker: Image Caption: KNOWN CAPTION"), image("must-not-send")]),
                message([text("MISSING-A"), image("A")], timestamp=1111),
                message("SHORT ACK", role="assistant"),
                message([text("array plain text")]),
                message([image("B")], timestamp=2222)]
    original = copy.deepcopy(messages)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        groups = source_groups(body)
        return completion([{"index": group["label"]["index"],
                            "summary": f"time-{group['label']['timestamp_unix_ms']}",
                            "facts": [f"fact-{group['label']['timestamp_unix_ms']}"]}
                           for group in reversed(groups)])

    async def exercise():
        async with mocked_provider(handler) as provider:
            frames = await provider.extract(messages, history="prior session")
            assert [frame["summary"] for frame in frames] == [
                "FIRST TEXT", "KNOWN CAPTION", "time-1111", "SHORT ACK", "array plain text", "time-2222"]
            assert [frame["extraction_source"] for frame in frames] == [
                "raw_text", "provided_caption", "model", "raw_text", "raw_text", "model"]
            assert frames[2]["facts"] == ["fact-1111"]
            assert frames[5]["facts"] == ["fact-2222"]
            assert provider.calls["extract"] == 1
            assert messages == original
    asyncio.run(exercise())
    groups = source_groups(requests[0])
    assert [group["label"]["index"] for group in groups] == [0, 1]
    sent_images = [part for group in groups for part in group["parts"] if part["type"] == "image_url"]
    assert sent_images == [image("A"), image("B")]
    context = requests[0]["messages"][1]["content"][0]["text"]
    assert "KNOWN CAPTION" in context and "SHORT ACK" in context and "prior session" in context
    assert "must-not-send" not in json.dumps(requests[0])
    assert "not a SOURCE" in context


def test_fallback_context_is_bounded_and_does_not_truncate_fastpath_index_text():
    long_text = "NEARBY-" + "X" * 50000
    messages = [message(long_text), message([image()]), message("NEXT-" + "Y" * 50000)]
    history = "OLD-CONTEXT-" + "Z" * 50000
    context = fallback_context(messages, {1}, history)
    assert len(context) <= LITE_CONTEXT_MAX_CHARS
    assert "NEARBY-" in context and "NEXT-" in context
    assert "OLD-CONTEXT-" not in context
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return ordinary_completion(request)

    async def exercise():
        async with mocked_provider(handler) as provider:
            frames = await provider.extract(messages, history)
            assert frames[0]["summary"] == long_text
            assert frames[2]["summary"] == messages[2]["content"]
            assert provider.calls["extract"] == 1
    asyncio.run(exercise())
    assert len(requests[0]["messages"][1]["content"][0]["text"]) <= LITE_CONTEXT_MAX_CHARS + 100


def test_fallback_normalizes_compatible_output_and_retries_empty_frames():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return completion([{"index": 0, "facts": [{}]}])
        return completion([{"index": 0, "summary": "vision evidence", "facts": "compatible fact"}])

    async def exercise():
        async with mocked_provider(handler) as provider:
            frames = await provider.extract([message("local text"), message([image()])])
            assert frames[0]["summary"] == "local text"
            assert frames[1]["facts"] == ["compatible fact"]
            assert provider.calls["extract"] == 2
            assert provider.calls["failed"] == 1
    asyncio.run(exercise())
    assert "FORMAT CORRECTION" in requests[1]["messages"][0]["content"]


def test_fallback_failure_never_returns_partial_fastpath_or_previous_batch_results():
    messages = [message("raw text"), message([text("Image Caption: existing caption"), image("caption")])]
    messages.extend(message([image(str(i))]) for i in range(5))
    original = copy.deepcopy(messages)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 2:
            return httpx.Response(503, json={"error": {"message": "synthetic unavailable"}})
        return ordinary_completion(request)

    async def exercise():
        async with mocked_provider(handler, llm_concurrency=1) as provider:
            with pytest.raises(UpstreamError, match="extract model call failed"):
                await provider.extract(messages)
            assert provider.calls["extract"] == 2
            assert messages == original
    asyncio.run(exercise())
    assert [len(source_groups(body)) for body in calls] == [4, 1]


def test_fallback_repeated_invalid_schema_is_not_hidden_by_fastpath_success():
    async def exercise():
        async with mocked_provider(lambda request: completion([{"index": 0, "summary": ""}])) as provider:
            with pytest.raises(UpstreamError, match="Add has not been committed"):
                await provider.extract([message("valid local text"), message([image()])])
            assert provider.calls["extract"] == 2
    asyncio.run(exercise())
