"""Storage guarantees that matter to the Add/Search contract (stdlib unittest)."""

import base64
import copy
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fusion_memory.storage import ConflictError, Storage


def payload(request_id="r1", user_id="alice", session_id="session", messages=None):
    return {
        "request_id": request_id, "user_id": user_id, "session_id": session_id,
        "messages": messages or [{"role": "user", "timestamp": 1704067200000, "content": "remember α"}],
    }


def prepared(count=1):
    return [{"structured": {"summary": "semantic memory"}, "embedding": [0.25, 0.75],
             "embedding_model": "test-embedding"} for _ in range(count)]


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name) / "data" / "memory.sqlite3")
        self.storage = Storage(self.db)

    def tearDown(self):
        self.storage.close()
        self.temp.cleanup()

    def test_commit_survives_reopen_and_preserves_raw_and_semantic_layers(self):
        self.storage.commit_add(payload(), "fp", prepared())
        reopened = Storage(self.db)
        records = reopened.list_records("alice")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["timestamp_ms"], 1704067200000)
        self.assertTrue(record["created_at"].endswith("Z"))
        self.assertEqual(record["position"], 0)
        self.assertEqual(record["structured"], {"summary": "semantic memory"})
        self.assertEqual(record["embedding"], [0.25, 0.75])
        self.assertEqual(reopened.get_content("alice", record["id"]), "remember α")
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], "wal")

    def test_idempotent_retry_conflict_and_tenant_isolation(self):
        request = payload()
        self.assertFalse(self.storage.check_request("alice", "r1", "fp"))
        self.storage.commit_add(request, "fp", prepared())
        self.storage.commit_add(request, "fp", prepared())
        self.assertTrue(self.storage.check_request("alice", "r1", "fp"))
        with self.assertRaises(ConflictError):
            self.storage.check_request("alice", "r1", "different")
        with self.assertRaises(ConflictError):
            self.storage.commit_add(request, "different", prepared())
        record = self.storage.list_records("alice")[0]
        self.assertEqual(len(self.storage.list_records("alice")), 1)
        self.assertEqual(self.storage.list_records("bob"), [])
        with self.assertRaises(KeyError):
            self.storage.get_content("bob", record["id"])
        self.storage.commit_add(payload(user_id="bob"), "different", prepared())
        self.assertEqual(len(self.storage.list_records("bob")), 1)

    def test_failure_in_later_message_rolls_back_every_layer(self):
        uri = "data:image/png;base64," + base64.b64encode(b"image evidence").decode()
        request = payload(messages=[
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]},
            {"role": "invalid", "content": "bad later message"},
        ])
        semantics = prepared(2)
        semantics[0]["structured"]["state_updates"] = [
            {"subject": "person", "predicate": "home", "value": "A"}
        ]
        with self.assertRaises(ValueError):
            self.storage.commit_add(request, "fp", semantics)
        self.assertFalse(self.storage.check_request("alice", "r1", "fp"))
        self.assertEqual(self.storage.list_records("alice"), [])
        self.assertEqual(self.storage.list_states("alice"), [])
        with sqlite3.connect(self.db) as conn:
            for table in ("requests", "records", "image_blobs", "image_occurrences", "state_versions"):
                self.assertEqual(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

    def test_image_bytes_deduplicate_without_losing_occurrences_order_or_details(self):
        uri = "data:image/png;base64," + base64.b64encode(b"same image bytes").decode()
        content = [
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": uri, "detail": "high"}},
            {"type": "text", "text": "between"},
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "after"},
        ]
        request = payload(messages=[{"role": "user", "content": content}])
        original = copy.deepcopy(request)
        self.storage.commit_add(request, "fp", prepared())
        self.assertEqual(request, original)
        record = self.storage.list_records("alice")[0]
        self.assertEqual(record["raw_text"], "before\nbetween\nafter")
        self.assertTrue(record["has_image"])
        self.assertEqual(len(record["image_hashes"]), 2)
        self.assertEqual(record["image_hashes"][0], record["image_hashes"][1])
        self.assertNotIn("raw_content", record)
        self.assertNotIn("data", record)
        self.assertEqual(self.storage.get_content("alice", record["id"]), content)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM image_blobs").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT count(*) FROM image_occurrences").fetchone()[0], 2)
        request["user_id"] = "bob"
        self.storage.commit_add(request, "fp", prepared())
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM image_blobs").fetchone()[0], 2)

    def test_concurrent_identical_add_is_once_and_distinct_chunks_remain_contiguous(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: self.storage.commit_add(payload(), "fp", prepared()), range(16)))
        self.assertEqual(len(self.storage.list_records("alice")), 1)

        def write_chunk(index):
            request = payload(f"chunk-{index}", messages=[
                {"role": "user", "timestamp": 1000 - index, "content": f"first-{index}"},
                {"role": "assistant", "timestamp": 1, "content": f"second-{index}"},
            ])
            self.storage.commit_add(request, f"fp-{index}", prepared(2))

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(write_chunk, range(12)))
        records = self.storage.list_records("alice")
        self.assertEqual([r["position"] for r in records], list(range(25)))
        for index in range(1, len(records), 2):
            self.assertEqual(records[index]["request_id"], records[index + 1]["request_id"])
            self.assertTrue(records[index]["raw_text"].startswith("first-"))
            self.assertTrue(records[index + 1]["raw_text"].startswith("second-"))

    def test_state_versions_use_source_time_even_for_late_arriving_evidence(self):
        for request_id, timestamp, city in [("new", 2000, "Shanghai"), ("old", 1000, "Beijing")]:
            request = payload(request_id, messages=[{"role": "user", "timestamp": timestamp, "content": city}])
            semantic = prepared()
            semantic[0]["structured"]["state_updates"] = [
                {"subject": "user", "predicate": "lives_in", "value": city}
            ]
            self.storage.commit_add(request, request_id, semantic)
        states = self.storage.list_states("alice")
        self.assertEqual([s["value"] for s in states], ["Beijing", "Shanghai"])
        self.assertEqual([s["valid_from_ms"] for s in states], [1000, 2000])
        self.assertEqual([s["valid_to_ms"] for s in states], [2000, None])
        self.assertEqual([s["is_current"] for s in states], [False, True])
        self.assertEqual(self.storage.list_states("bob"), [])
        self.assertEqual([r["raw_text"] for r in self.storage.list_records("alice")], ["Shanghai", "Beijing"])

    def test_nan_embedding_and_prepared_mismatch_do_not_commit(self):
        with self.assertRaises(ValueError):
            self.storage.commit_add(payload(), "fp", [])
        semantic = prepared()
        semantic[0]["embedding"] = [float("nan")]
        with self.assertRaises(ValueError):
            self.storage.commit_add(payload(), "fp", semantic)
        self.assertEqual(self.storage.list_records("alice"), [])
        self.assertFalse(self.storage.check_request("alice", "r1", "fp"))

    def test_assistant_evidence_is_preserved_but_cannot_change_user_state(self):
        request = payload(messages=[
            {"role": "user", "timestamp": 1000, "content": "I live in Beijing."},
            {"role": "assistant", "timestamp": 2000, "content": "You live in Shanghai."},
        ])
        semantics = prepared(2)
        for semantic, city in zip(semantics, ["Beijing", "Shanghai"]):
            semantic["structured"]["state_updates"] = [
                {"subject": "user", "predicate": "lives_in", "value": city}
            ]
        self.storage.commit_add(request, "fp", semantics)
        states = self.storage.list_states("alice")
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["value"], "Beijing")
        self.assertTrue(states[0]["is_current"])
        self.assertIsNone(states[0]["valid_to_ms"])
        records = self.storage.list_records("alice")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["structured"], semantics[1]["structured"])
        self.assertEqual(self.storage.get_content("alice", records[1]["id"]), "You live in Shanghai.")
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM state_versions").fetchone()[0], 1)

    def test_explicit_in_memory_database_survives_short_connections(self):
        memory = Storage(":memory:")
        try:
            memory.commit_add(payload(), "fp", prepared())
            self.assertTrue(memory.check_request("alice", "r1", "fp"))
            self.assertEqual(len(memory.list_records("alice")), 1)
        finally:
            memory.close()


if __name__ == "__main__":
    unittest.main()
