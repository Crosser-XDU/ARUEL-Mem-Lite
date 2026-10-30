"""HTTP contract tests with real SQLite and image decoding, without model calls."""

import base64
import copy
from dataclasses import replace
import hmac
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from fusion_memory.app import create_app
from fusion_memory.config import Settings
from fusion_memory.provider import Frame, UpstreamError, text_of
from fusion_memory.schemas import MAX_IMAGE_BYTES


def png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (3, 2), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = png_bytes()
PNG_URI = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


def image_part(uri=PNG_URI):
    return {"type": "image_url", "image_url": {"url": uri, "detail": "high"}}


def add_payload(content="The crimson telescope is in the observatory.", *,
                request_id="eval:run:text:conv-0:chunk-0", user_id="alice", session_id="session-0"):
    return {
        "request_id": request_id, "user_id": user_id, "session_id": session_id,
        "messages": [{"role": "user", "timestamp": 1704067200000, "content": content}],
    }


class FakeProvider:
    """Only substitute the external boundary; exercise real validation/retrieval."""

    def __init__(self):
        self.calls = {"extract": 0, "embed": 0, "query": 0, "close": 0}
        self.fail_extract = False
        self.fail_embed = False

    async def extract(self, messages, history=""):
        self.calls["extract"] += 1
        if self.fail_extract:
            raise UpstreamError("extraction unavailable; Add has not been committed")
        return [Frame(summary=text_of(message["content"]),
                      facts=[text_of(message["content"])]).model_dump()
                for message in messages]

    async def embed(self, texts):
        self.calls["embed"] += 1
        if self.fail_embed:
            raise UpstreamError("embedding unavailable; Add has not been committed")
        return [None for _ in texts]

    async def query_text(self, content, options=None):
        self.calls["query"] += 1
        return text_of(content)

    async def close(self):
        self.calls["close"] += 1


class APIContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(db_path=str(Path(self.temp.name) / "memory.sqlite3"),
                                 graph_expansion=False, neighbor_hops=0)
        self.provider = FakeProvider()
        self.app = create_app(self.settings, provider=self.provider)
        self.client = self.enterContext(TestClient(self.app))

    def search(self, query="crimson telescope", user_id="alice", top_k=10, client=None):
        return (client or self.client).post("/search", json={
            "user_id": user_id, "query": query, "top_k": top_k,
        })

    def test_add_exact_echo_and_immediate_text_search(self):
        request = add_payload()
        response = self.client.post("/add", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {
            "success": True, "request_id": request["request_id"],
            "user_id": request["user_id"], "session_id": request["session_id"],
        })
        response = self.search()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(set(response.json()), {"data"})
        results = response.json()["data"]
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(set(result), {"id", "content", "score", "created_at"})
        self.assertIsInstance(result["id"], str)
        self.assertIsInstance(result["content"], str)
        self.assertIsInstance(result["score"], float)
        self.assertGreater(result["score"], 0)
        self.assertIn(request["messages"][0]["content"], result["content"])
        self.assertEqual(result["created_at"], "2024-01-01T00:00:00Z")
        self.assertIn("source_time=2024-01-01T00:00:00Z", result["content"])

    def test_identical_retry_keeps_id_and_skips_provider_different_payload_conflicts(self):
        request = add_payload()
        first = self.client.post("/add", json=request)
        original = self.search().json()["data"]
        before_retry = dict(self.provider.calls)
        retry = self.client.post("/add", json=request)
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(first.json(), retry.json())
        self.assertEqual(self.provider.calls, before_retry)
        self.assertEqual(self.search().json()["data"], original)
        changed = copy.deepcopy(request)
        changed["messages"][0]["content"] = "A different source under the same request identifier."
        before_conflict = dict(self.provider.calls)
        conflict = self.client.post("/add", json=changed)
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertEqual(self.provider.calls, before_conflict)
        self.assertEqual(self.search().json()["data"], original)

    def test_tenants_are_isolated_and_unmatched_queries_return_empty_data(self):
        self.assertEqual(self.client.post("/add", json=add_payload()).status_code, 200)
        before = dict(self.provider.calls)
        foreign = self.search(user_id="bob")
        self.assertEqual(foreign.status_code, 200)
        self.assertEqual(foreign.json(), {"data": []})
        self.assertEqual(self.provider.calls, before)
        no_match = self.search("zzquasarunrelatedzz")
        self.assertEqual(no_match.status_code, 200)
        self.assertEqual(no_match.json(), {"data": []})
        own = add_payload("Bob's jade microscope.", user_id="bob")
        self.assertEqual(self.client.post("/add", json=own).status_code, 200)
        self.assertEqual(self.search("microscope", user_id="alice").json(), {"data": []})
        self.assertEqual(self.search("telescope", user_id="bob").json(), {"data": []})

    def test_ordered_multimodal_content_preserves_image_bytes_and_source_time(self):
        source = [
            {"type": "text", "text": "crimson telescope before picture"},
            image_part(),
            {"type": "text", "text": "between the photographs"},
            image_part(),
            {"type": "text", "text": "after picture, observatory"},
        ]
        request = add_payload(source)
        response = self.client.post("/add", json=request)
        self.assertEqual(response.status_code, 200, response.text)
        response = self.search([{"type": "text", "text": "crimson telescope"}, image_part()])
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()["data"][0]
        self.assertIsInstance(result["content"], list)
        self.assertEqual(result["content"][1:], source)
        self.assertEqual(result["created_at"], "2024-01-01T00:00:00Z")
        self.assertIn("source_time=2024-01-01T00:00:00Z", result["content"][0]["text"])
        pictures = [part for part in result["content"] if part["type"] == "image_url"]
        self.assertEqual(len(pictures), 2)
        for part in pictures:
            encoded = part["image_url"]["url"].split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded, validate=True), PNG)
        # A plain-text query must still return multimodal evidence in order.
        self.assertEqual(self.search().json()["data"][0]["content"][1:], source)

    def test_multimodal_query_returns_text_evidence_as_content_parts(self):
        self.assertEqual(self.client.post("/add", json=add_payload()).status_code, 200)
        response = self.search([{"type": "text", "text": "crimson telescope"}, image_part()])
        self.assertEqual(response.status_code, 200, response.text)
        content = response.json()["data"][0]["content"]
        self.assertIsInstance(content, list)
        self.assertTrue(all(part["type"] == "text" for part in content))
        self.assertIn("crimson telescope", "\n".join(part["text"] for part in content))

    def test_model_failure_returns_503_without_reserving_request_or_partial_write(self):
        for failure in ("fail_extract", "fail_embed"):
            with self.subTest(failure=failure):
                request = add_payload(request_id=failure, user_id=failure)
                request["messages"].append({"role": "assistant", "content": "Second message."})
                setattr(self.provider, failure, True)
                response = self.client.post("/add", json=request)
                self.assertEqual(response.status_code, 503, response.text)
                self.assertEqual(self.app.state.storage.list_records(failure), [])
                self.assertEqual(self.search(user_id=failure).json(), {"data": []})
                setattr(self.provider, failure, False)
                retry = self.client.post("/add", json=request)
                self.assertEqual(retry.status_code, 200, retry.text)
                self.assertEqual(len(self.app.state.storage.list_records(failure)), 2)

    def test_invalid_image_inputs_reject_without_echoing_private_payload(self):
        private = "PRIVATE_SOURCE_SENTINEL_7194"
        invalid_uris = [
            "https://example.invalid/private.png?token=" + private,
            "data:image/png;base64," + private + "%notbase64",
            "data:image/png;base64," + base64.b64encode(b"not an image " + private.encode()).decode(),
        ]
        for uri in invalid_uris:
            for endpoint in ("/add", "/search"):
                with self.subTest(endpoint=endpoint, uri_kind=uri[:25]):
                    content = [{"type": "text", "text": private}, image_part(uri)]
                    body = (add_payload(content) if endpoint == "/add" else
                            {"user_id": "alice", "query": content, "top_k": 5})
                    before = dict(self.provider.calls)
                    response = self.client.post(endpoint, json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertNotIn(private, response.text)
                    self.assertNotIn(uri, response.text)
                    self.assertNotIn('"input"', response.text)
                    self.assertEqual(self.provider.calls, before)
        self.assertEqual(self.app.state.storage.list_records("alice"), [])

    def test_oversize_image_rejects_before_model_call_and_does_not_echo_payload(self):
        encoded = base64.b64encode(PNG + b"\0" * (MAX_IMAGE_BYTES + 1 - len(PNG))).decode()
        uri = "data:image/png;base64," + encoded
        request = add_payload([{"type": "text", "text": "PRIVATE_OVERSIZE_SOURCE_8491"}, image_part(uri)])
        before = dict(self.provider.calls)
        response = self.client.post("/add", json=request)
        self.assertEqual(response.status_code, 422, response.text[:500])
        self.assertLess(len(response.content), 2000)
        self.assertNotIn("PRIVATE_OVERSIZE_SOURCE_8491", response.text)
        self.assertNotIn(encoded[:100], response.text)
        self.assertEqual(self.provider.calls, before)
        self.assertEqual(self.app.state.storage.list_records("alice"), [])

    def test_response_limits_bound_text_image_count_bytes_and_record_count(self):
        # Each independent bound gets its own service configuration and real
        # HTTP round-trip, with unchanged evidence already persisted on disk.
        long_text = "crimson telescope " * 500
        for index in range(3):
            content = [{"type": "text", "text": f"record {index} " + long_text},
                       image_part(), image_part()]
            request = add_payload(content, request_id=f"budget-{index}")
            self.assertEqual(self.client.post("/add", json=request).status_code, 200)
        bounded_settings = replace(self.settings, max_response_images=1,
                                   max_response_image_bytes=len(PNG),
                                   max_response_text_chars=1100, retrieval_limit=2)
        with TestClient(create_app(bounded_settings, provider=FakeProvider())) as bounded:
            response = self.search(client=bounded, top_k=100)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertGreater(len(data), 0)
        self.assertLessEqual(len(data), 2)
        parts = [part for row in data for part in row["content"]]
        images = [part for part in parts if part["type"] == "image_url"]
        self.assertLessEqual(len(images), 1)
        returned_bytes = sum(len(base64.b64decode(p["image_url"]["url"].split(",", 1)[1])) for p in images)
        self.assertLessEqual(returned_bytes, len(PNG))
        self.assertLessEqual(sum(len(p["text"]) for p in parts if p["type"] == "text"), 1100)
        no_image_settings = replace(self.settings, max_response_image_bytes=len(PNG) - 1)
        with TestClient(create_app(no_image_settings, provider=FakeProvider())) as bounded:
            data = self.search(client=bounded).json()["data"]
        self.assertFalse(any(p["type"] == "image_url" for row in data for p in row["content"]))

    def test_string_result_respects_text_character_budget_including_separators(self):
        request = add_payload("crimson telescope " * 500)
        self.assertEqual(self.client.post("/add", json=request).status_code, 200)
        bounded_settings = replace(self.settings, max_response_text_chars=1000)
        with TestClient(create_app(bounded_settings, provider=FakeProvider())) as bounded:
            response = self.search(client=bounded)
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(len(data), 1)
        self.assertLessEqual(sum(len(row["content"]) for row in data), 1000)

    def test_persisted_evidence_and_idempotency_survive_app_restart(self):
        request = add_payload([{"type": "text", "text": "crimson telescope"}, image_part()])
        self.assertEqual(self.client.post("/add", json=request).status_code, 200)
        original = self.search().json()
        next_provider = FakeProvider()
        with TestClient(create_app(self.settings, provider=next_provider)) as restarted:
            result = self.search(client=restarted)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json(), original)
            before_retry = dict(next_provider.calls)
            retry = restarted.post("/add", json=request)
            self.assertEqual(retry.status_code, 200, retry.text)
            self.assertEqual(next_provider.calls, before_retry)
        self.assertEqual(next_provider.calls["close"], 1)

    def test_request_body_limit_and_service_authorization(self):
        guarded_settings = replace(self.settings, service_token="test-service-token", max_body_bytes=1024)
        guarded_provider = FakeProvider()
        with TestClient(create_app(guarded_settings, provider=guarded_provider)) as guarded:
            self.assertEqual(guarded.get("/healthz").status_code, 200)
            self.assertEqual(guarded.post("/add", json=add_payload()).status_code, 401)
            self.assertEqual(self.search(client=guarded).status_code, 401)
            response = guarded.post("/add", json=add_payload("private body " * 300),
                                    headers={"Authorization": "Bearer test-service-token"})
            self.assertEqual(response.status_code, 413, response.text)
            self.assertNotIn("private body", response.text)
        self.assertEqual(guarded_provider.calls["extract"], 0)
        self.assertEqual(self.app.state.storage.list_records("alice"), [])

    def test_token_and_bearer_schemes_allow_add_and_search_case_insensitively(self):
        guarded_settings = replace(self.settings, service_token="synthetic-service-token-42")
        guarded_provider = FakeProvider()
        with TestClient(create_app(guarded_settings, provider=guarded_provider)) as guarded:
            for index, scheme in enumerate(("Token", "token", "TOKEN", "tOkEn", "Bearer", "bearer", "BEARER", "bEaReR")):
                with self.subTest(scheme=scheme):
                    headers = {"Authorization": scheme + " synthetic-service-token-42"}
                    owner = f"authorized-{index}"
                    response = guarded.post("/add", json=add_payload(user_id=owner), headers=headers)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(response.json()["success"])
                    response = guarded.post("/search", json={"user_id": owner, "query": "crimson telescope", "top_k": 5}, headers=headers)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(len(response.json()["data"]), 1)
                    self.assertIn("The crimson telescope is in the observatory.", response.json()["data"][0]["content"])
        self.assertEqual(guarded_provider.calls["extract"], 8)
        self.assertEqual(guarded_provider.calls["query"], 8)

    def test_invalid_token_authorization_rejects_both_endpoints_without_provider_calls(self):
        credential = "synthetic-service-token-42"
        guarded_settings = replace(self.settings, service_token=credential)
        guarded_provider = FakeProvider()
        invalid_headers = [
            {}, {"Authorization": ""}, {"Authorization": "Token"}, {"Authorization": "Token "},
            {"Authorization": "Token incorrect"}, {"Authorization": "Token " + credential.upper()},
            {"Authorization": "Token " + credential + " extra"},
            {"Authorization": "Token " + credential + ", Bearer " + credential},
            {"Authorization": "Token  " + credential}, {"Authorization": "Token " + credential + " "},
            {"Authorization": "Token\t" + credential}, {"Authorization": "Basic " + credential},
            {"Authorization": "Bearer incorrect"}, {"Authorization": "Bearer " + credential + " extra"},
            [("Authorization", "Token " + credential), ("Authorization", "Bearer " + credential)],
            [(b"Authorization", b"Token \xff")],
        ]
        with TestClient(create_app(guarded_settings, provider=guarded_provider)) as guarded:
            for index, headers in enumerate(invalid_headers):
                for endpoint, payload in (("/add", add_payload()), ("/search", {"user_id": "alice", "query": "telescope", "top_k": 5})):
                    with self.subTest(case=index, endpoint=endpoint):
                        before = dict(guarded_provider.calls)
                        response = guarded.post(endpoint, json=payload, headers=headers)
                        self.assertEqual(response.status_code, 401, response.text)
                        self.assertEqual(response.json(), {"detail": "invalid service authorization"})
                        self.assertNotIn(credential, response.text)
                        self.assertEqual(guarded_provider.calls, before)
            self.assertEqual(guarded.get("/").status_code, 200)
            self.assertEqual(guarded.get("/healthz").status_code, 200)
        self.assertEqual(self.app.state.storage.list_records("alice"), [])

    def test_authentication_compares_only_the_actual_credential_in_constant_time(self):
        guarded_settings = replace(self.settings, service_token="synthetic-service-token-42")
        with TestClient(create_app(guarded_settings, provider=FakeProvider())) as guarded:
            with patch("fusion_memory.app.hmac.compare_digest", wraps=hmac.compare_digest) as compare:
                response = guarded.post("/search", json={"user_id": "alice", "query": "telescope", "top_k": 5},
                                        headers={"Authorization": "tOkEn synthetic-service-token-42"})
                self.assertEqual(response.status_code, 200, response.text)
                compare.assert_called_once_with(b"synthetic-service-token-42", b"synthetic-service-token-42")

    def test_service_name_and_version_are_consistent(self):
        self.assertEqual(self.app.title, "AUREL-Mem-Lite")
        self.assertEqual(self.app.version, "2.0.0-lite")
        self.assertEqual(self.client.get("/").json()["service"], "AUREL-Mem-Lite")
        self.assertEqual(self.client.get("/healthz").json()["version"], "2.0.0-lite")
        info = self.client.get("/openapi.json").json()["info"]
        self.assertEqual(info["title"], "AUREL-Mem-Lite")
        self.assertEqual(info["version"], "2.0.0-lite")


if __name__ == "__main__":
    unittest.main()
