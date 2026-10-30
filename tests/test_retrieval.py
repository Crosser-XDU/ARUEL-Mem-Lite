"""Behavioral retrieval checks; no API key, service, or model is required."""

from copy import deepcopy
import math
import unittest

from fusion_memory.retrieval import rank_memories, tokenize


def memory(identifier, text="", *, entities=None, structured=None, **fields):
    details = dict(structured or {})
    if entities is not None:
        details["entities"] = entities
    record = {
        "id": identifier,
        "user_id": "alice",
        "session_id": None,
        "request_id": None,
        "position": 0,
        "role": "user",
        "timestamp_ms": None,
        "created_at": None,
        "raw_text": text,
        "structured": details,
        "embedding": None,
        "embedding_model": None,
        "has_image": False,
        "image_hashes": [],
    }
    record.update(fields)
    return record


def ids(result):
    return [record["id"] for record, _ in result]


class RetrievalTests(unittest.TestCase):
    def test_empty_library_query_and_nonpositive_limit(self):
        record = memory("one", "device")
        self.assertEqual(rank_memories([], "device", None, 5), [])
        self.assertEqual(rank_memories([record], "", None, 5), [])
        self.assertEqual(rank_memories([record], "device", None, 0), [])
        self.assertEqual(rank_memories([record], "device", None, -2), [])

    def test_no_match_does_not_fill_with_background(self):
        records = [memory(str(i), "今天下雨，花园里的植物需要照顾。") for i in range(100)]
        self.assertEqual(rank_memories(records, "量子纠错芯片", None, 10), [])
        self.assertEqual(rank_memories(records, "previous information", None, 10), [])
        incidental_character = [memory(str(i), "买了一台咖啡机") for i in range(30)]
        self.assertEqual(rank_memories(incidental_character, "量子计算机", None, 10), [])

    def test_chinese_ocr_model_number_and_multiview_ranking(self):
        records = [
            memory("unrelated", "办公室采购了一台电视"),
            memory("ocr", has_image=True, structured={"ocr": [{"text": "索尼 XR-55A80L 序列号 SN-00931"}], "objects": ["电视"]}),
            memory("generic", "索尼电视说明书"),
        ]
        found = rank_memories(records, "图片上 XR-55A80L 的序列号是什么", None, 2)
        self.assertEqual(ids(found)[0], "ocr")
        self.assertIn("xr-55a80l", tokenize("型号 XR-55A80L"))
        self.assertIn("序列", tokenize("序列号"))

    def test_semantic_fact_and_visual_evidence_work_without_raw_text(self):
        records = [
            memory("fact", structured={"facts": [{"owner": "小陈", "purchase": "红色咖啡机"}]}),
            memory("visual", structured={"objects": [{"name": "蓝色背包"}]}),
        ]
        self.assertEqual(ids(rank_memories(records, "小陈的咖啡机", None, 2))[0], "fact")
        self.assertEqual(ids(rank_memories(records, "蓝色背包", None, 2))[0], "visual")

    def test_text_hit_expands_to_neighboring_image(self):
        records = [
            memory("text", "这是上周新采购的显卡。", session_id="hardware", request_id="upload", position=0),
            memory("image", has_image=True, session_id="hardware", request_id="upload", position=1, structured={"ocr": "NVIDIA RTX 4090"}),
            memory("other", "下午喝咖啡", session_id="other", position=0),
        ]
        found = rank_memories(records, "新采购的显卡", None, 3)
        self.assertEqual(ids(found), ["text", "image"])
        self.assertEqual(ids(rank_memories(records, "新采购的显卡", None, 3, graph_expansion=False)), ["text"])

    def test_session_context_does_not_reach_across_days(self):
        records = [
            memory("seed", "server migration", session_id="same", created_at="2026-10-01T00:00:00Z"),
            memory("later", "vacation photo", session_id="same", has_image=True, created_at="2026-10-10T00:00:00Z"),
        ]
        self.assertEqual(ids(rank_memories(records, "migration", None, 3)), ["seed"])

    def test_session_neighbors_follow_position_not_timestamp_or_request_id(self):
        records = [
            memory("seed", "purchase receipt", session_id="same", position=0, request_id="z", timestamp_ms=3000),
            memory("image", has_image=True, session_id="same", position=1, request_id="a", timestamp_ms=1000, structured={"ocr": "QX-17"}),
            memory("noise", "lunch menu", session_id="same", position=2, request_id="m", timestamp_ms=2000),
        ]
        self.assertEqual(ids(rank_memories(records, "purchase", None, 3)), ["seed", "image"])

    def test_dense_similarity_models_dimensions_and_invalid_vectors(self):
        records = [
            memory("valid", "fruit orchard", embedding=[1.0, 0.0], embedding_model="same"),
            memory("opposite", "vehicle repair", embedding=[-1.0, 0.0], embedding_model="same"),
            memory("wrong-model", "private notebook", embedding=[1.0, 0.0], embedding_model="different"),
            memory("wrong-size", "stone bridge", embedding=[1.0, 0.0, 0.0], embedding_model="same"),
            memory("invalid", "snow globe", embedding=[float("nan"), 1], embedding_model="same"),
            memory("zero", "empty basket", embedding=[0, 0], embedding_model="same"),
        ]
        self.assertEqual(ids(rank_memories(records, "apples", [1, 0], 10, query_embedding_model="same")), ["valid"])

    def test_incompatible_embedding_still_allows_lexical_evidence(self):
        record = memory("lexical", "orchard apples", embedding=[1, 0], embedding_model="old")
        self.assertEqual(ids(rank_memories([record], "apples", [1, 0], 2, query_embedding_model="new")), ["lexical"])

    def test_latest_prior_and_comparison_preserve_history(self):
        records = [
            memory("a-old", "Router status is offline", entities=["Router"], created_at="2026-01-01T00:00:00Z"),
            memory("b-prior", "Router status is degraded", entities=["Router"], created_at="2026-02-01T00:00:00Z"),
            memory("z-current", "Router status is online", entities=["Router"], created_at="2026-03-01T00:00:00Z"),
        ]
        self.assertEqual(ids(rank_memories(records, "Router latest status", None, 3))[0], "z-current")
        self.assertEqual(ids(rank_memories(records, "Router previous status", None, 3))[0], "b-prior")
        result = ids(rank_memories(records, "比较 Router 以前和现在的 status", None, 3))
        self.assertEqual(set(result), {"a-old", "b-prior", "z-current"})

    def test_explicit_before_and_after_dates_rank_correct_sides(self):
        records = [
            memory("old", "Atlas office in Paris", entities=["Atlas"], created_at="2026-01-15T00:00:00Z"),
            memory("new", "Atlas office in Tokyo", entities=["Atlas"], created_at="2026-03-15T00:00:00Z"),
        ]
        self.assertEqual(ids(rank_memories(records, "Atlas office before 2026-02-01", None, 2))[0], "old")
        self.assertEqual(ids(rank_memories(records, "Atlas office after 2026-02-01", None, 2))[0], "new")

    def test_event_time_is_preferred_over_ingestion_time(self):
        records = [
            memory("old-event", "Atlas location Paris", entities=["Atlas"], created_at="2026-10-28", structured={"event_time": "2026-01-01"}),
            memory("new-event", "Atlas location Tokyo", entities=["Atlas"], created_at="2026-10-01", structured={"event_time": "2026-09-01"}),
        ]
        self.assertEqual(ids(rank_memories(records, "Atlas current location", None, 2))[0], "new-event")

    def test_list_event_time_uses_first_parseable_value(self):
        records = [
            memory("old-event", "Atlas location Paris", entities=["Atlas"], created_at="2026-10-28", structured={"event_time": ["unclear", "2026-01-01"]}),
            memory("new-event", "Atlas location Tokyo", entities=["Atlas"], created_at="2026-10-01", structured={"event_time": ["2026-09-01"]}),
        ]
        self.assertEqual(ids(rank_memories(records, "Atlas current location", None, 2))[0], "new-event")

    def test_two_hop_entity_chain_and_hop_control(self):
        records = [
            memory("seed", "emergency recovery procedure", entities=["Atlas"]),
            memory("bridge", "handoff contract", entities=["Atlas", "Borealis"]),
            memory("destination", "offsite backup credentials", entities=["Borealis"]),
            memory("noise", "lunch menu", entities=["Canteen"]),
        ]
        self.assertEqual(set(ids(rank_memories(records, "emergency", None, 4, neighbor_hops=1))), {"seed", "bridge"})
        self.assertEqual(set(ids(rank_memories(records, "emergency", None, 4, neighbor_hops=2))), {"seed", "bridge", "destination"})
        self.assertEqual(ids(rank_memories(records, "emergency", None, 4, neighbor_hops=0)), ["seed"])

    def test_graph_expansion_never_crosses_user_boundary(self):
        records = [
            memory("seed", "emergency procedure", entities=["Atlas"], session_id="shared"),
            memory("private", "secret credentials", entities=["Atlas"], session_id="shared", user_id="bob"),
        ]
        self.assertEqual(ids(rank_memories(records, "emergency", None, 5)), ["seed"])

    def test_relation_endpoints_build_graph_without_explicit_entities(self):
        records = [
            memory("seed", "emergency", structured={"relations": [{"subject": "Atlas", "predicate": "owns", "object": "Borealis"}]}),
            memory("linked", "offsite copy", entities=[{"name": "Borealis", "type": "project"}]),
        ]
        self.assertEqual(ids(rank_memories(records, "emergency", None, 2)), ["seed", "linked"])

    def test_multi_entity_coverage_and_duplicate_suppression(self):
        records = [
            memory("a", "Atlas backup runs nightly", entities=["Atlas"]),
            memory("b-duplicate", "Atlas backup runs nightly", entities=["Atlas"]),
            memory("c-near", "Atlas backup runs nightly on weekdays", entities=["Atlas"]),
            memory("z", "Borealis backup runs weekly", entities=["Borealis"]),
        ]
        result = ids(rank_memories(records, "Atlas Borealis backup", None, 2))
        self.assertIn("z", result)
        self.assertIn("a", result)
        all_found = ids(rank_memories(records, "backup", None, 10))
        # Repeated events remain available when the caller asks for history.
        self.assertIn("b-duplicate", all_found)
        self.assertEqual(len(all_found), 4)

    def test_distinct_images_and_repeated_image_occurrences_are_preserved(self):
        records = [
            memory("first", has_image=True, image_hashes=["image-a"], timestamp_ms=1000, structured={"summary": "red umbrella"}),
            memory("different-image", has_image=True, image_hashes=["image-b"], timestamp_ms=2000, structured={"summary": "red umbrella"}),
            memory("repeat-visit", has_image=True, image_hashes=["image-a"], timestamp_ms=3000, structured={"summary": "red umbrella"}),
        ]
        result = rank_memories(records, "red umbrella", None, 5)
        self.assertEqual(set(ids(result)), {"first", "different-image", "repeat-visit"})
        self.assertLess(result[1][1], result[0][1])

    def test_duplicate_record_id_is_returned_only_once(self):
        record = memory("same-id", "orchard apples")
        self.assertEqual(ids(rank_memories([record, deepcopy(record)], "apples", None, 5)), ["same-id"])

    def test_conflicting_states_are_not_deduplicated_by_shared_raw_text(self):
        records = [
            memory("old", "status update", structured={"facts": ["Atlas is offline"]}),
            memory("new", "status update", structured={"facts": ["Atlas is online"]}),
        ]
        self.assertEqual(set(ids(rank_memories(records, "Atlas status", None, 2))), {"old", "new"})

    def test_deterministic_ties_nonincreasing_scores_limit_and_no_mutation(self):
        records = [memory("z", "backup red"), memory("a", "backup blue"), memory("b", "backup green")]
        snapshot = deepcopy(records)
        first = rank_memories(records, "backup", None, 2)
        self.assertEqual(ids(first)[0], "a")
        self.assertEqual(ids(first), ids(rank_memories(list(reversed(records)), "backup", None, 2)))
        self.assertEqual(len(first), 2)
        scores = [score for _, score in first]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(math.isfinite(score) and score > 0 for score in scores))
        self.assertEqual(records, snapshot)


if __name__ == "__main__":
    unittest.main()
