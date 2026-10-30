"""Synthetic shape-only cases: no network calls or benchmark content."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from fusion_memory.output_format import LIST_FIELDS, ShapeError, normalize_extraction


def complete_row(index=0):
    return {"index": index, "summary": "synthetic summary",
            **{field: [] for field in LIST_FIELDS}, "state_updates": []}


def test_valid_rows_keep_content_and_restore_source_order_without_repairs():
    first, second = complete_row(), complete_row(1)
    first["facts"] = ["synthetic fact"]
    first["future_field"] = {"unknown": [1, None]}
    data = {"records": [second, first], "extra_root": "ignored"}
    original = deepcopy(data)
    rows, repairs = normalize_extraction(data, 2)
    assert rows == [first, second]
    assert repairs == Counter()
    assert data == original


def test_optional_defaults_and_single_record_wrapper():
    rows, repairs = normalize_extraction({"records": {"index": 0, "summary": "synthetic summary"}}, 1)
    assert rows == [complete_row()]
    assert repairs == Counter(records_object_wrapped=1, optional_field_defaulted=8)


@pytest.mark.parametrize("index,category", [
    ("0", "index_string_to_int"), ("000", "index_string_to_int"),
    (0.0, "index_float_to_int"), (-0.0, "index_float_to_int"),
])
def test_unambiguous_index_representations(index, category):
    rows, repairs = normalize_extraction({"records": [complete_row(index)]}, 1)
    assert type(rows[0]["index"]) is int and rows[0]["index"] == 0
    assert repairs == Counter({category: 1})


@pytest.mark.parametrize("index", [
    True, False, None, "", "-0", "+0", " 0", "0 ", "0.0", "0e0", "０", "٠",
    0.5, float("nan"), float("inf"), float("-inf"), [], {},
])
def test_ambiguous_or_invalid_index_is_rejected(index):
    with pytest.raises(ShapeError) as error:
        normalize_extraction({"records": [complete_row(index)]}, 1)
    assert (error.value.code, error.value.path) == ("index_type", "records[].index")


@pytest.mark.parametrize("value,result,repair", [
    (None, "", "summary_null_to_empty"),
    ({"text": "synthetic"}, "synthetic", "summary_text_unwrapped"),
])
def test_summary_lossless_compatibility(value, result, repair):
    row = complete_row()
    row["summary"] = value
    row["facts"] = ["synthetic fact"]
    rows, repairs = normalize_extraction({"records": [row]}, 1)
    assert rows[0]["summary"] == result
    assert repairs == Counter({repair: 1})


@pytest.mark.parametrize("value", [3, True, [], {"text": None}, {"other": "text"},
                                        {"text": "synthetic", "lost": "must not discard"}])
def test_summary_rejects_lossy_or_complex_unwrapping(value):
    row = {**complete_row(), "summary": value}
    with pytest.raises(ShapeError, match=r"^summary_type at records\[\].summary$"):
        normalize_extraction({"records": [row]}, 1)


@pytest.mark.parametrize("field", LIST_FIELDS)
@pytest.mark.parametrize("value,expected,category", [
    (None, [], "list_null_to_empty"),
    ("synthetic index", ["synthetic index"], "list_string_wrapped"),
    ("", [], "list_string_wrapped"),
    (" \t", [], "list_string_wrapped"),
])
def test_list_field_representation_compatibility(field, value, expected, category):
    row = {**complete_row(), field: value}
    rows, repairs = normalize_extraction({"records": [row]}, 1)
    assert rows[0][field] == expected
    assert repairs == Counter({category: 1})


def test_nested_index_items_are_json_preserved_without_inventing_states():
    relation = {"subject": "synthetic object", "value": {"count": 2, "valid": True},
                "details": [None, "__import__('os').system('do-not-execute')"]}
    values = ["", "literal", 42, 1.5, True, False, relation, ["nested", {"k": None}], None]
    row = {**complete_row(), "relations": values}
    original = deepcopy(row)
    rows, repairs = normalize_extraction({"records": [row]}, 1)
    result = rows[0]
    assert result["relations"][:6] == ["", "literal", "42", "1.5", "true", "false"]
    assert json.loads(result["relations"][6]) == relation
    assert json.loads(result["relations"][7]) == ["nested", {"k": None}]
    assert result["state_updates"] == []
    assert repairs == Counter(list_item_json_encoded=6, list_null_item_removed=1)
    assert row == original


@pytest.mark.parametrize("value", [0, False, ("synthetic",)])
def test_top_level_list_field_is_not_guessed_from_arbitrary_shape(value):
    with pytest.raises(ShapeError) as error:
        normalize_extraction({"records": [{**complete_row(), "facts": value}]}, 1)
    assert error.value.code == "list_type"


@pytest.mark.parametrize("field", LIST_FIELDS)
def test_single_index_object_is_preserved_in_one_json_string(field):
    value = {"synthetic key": [1, None, {"nested": True}]}
    rows, repairs = normalize_extraction({"records": [{**complete_row(), field: value}]}, 1)
    assert json.loads(rows[0][field][0]) == value
    assert rows[0]["state_updates"] == []
    assert repairs == Counter(list_object_wrapped=1)


@pytest.mark.parametrize("item", [float("nan"), float("inf"), {"x": float("inf")},
                                      [float("nan")], {1: "do not coerce key"}, object()])
def test_index_items_must_be_losslessly_json_representable(item):
    with pytest.raises(ShapeError) as error:
        normalize_extraction({"records": [{**complete_row(), "facts": [item]}]}, 1)
    assert error.value.code in {"non_json_value", "item_type"}
    assert error.value.path == "records[].facts[]"


def test_cyclic_value_is_rejected_without_serializing_data_in_error():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(ShapeError) as error:
        normalize_extraction({"records": [{**complete_row(), "facts": [cycle]}]}, 1)
    assert str(error.value) == "non_json_value at records[].facts[]"


@pytest.mark.parametrize("value,expected", [(12, "12"), (2.5, "2.5"), (True, "true"), (False, "false")])
def test_explicit_state_object_wrap_and_scalar_value(value, expected):
    state = {"subject": "synthetic subject", "predicate": "synthetic attribute", "value": value,
             "extra": {"preserved": True}}
    data = {"records": [{**complete_row(), "state_updates": state}]}
    original = deepcopy(data)
    rows, repairs = normalize_extraction(data, 1)
    assert rows[0]["state_updates"] == [{**state, "value": expected}]
    assert repairs == Counter(state_object_wrapped=1, state_value_json_encoded=1)
    assert data == original


def test_null_states_and_empty_string_state_value():
    rows, repairs = normalize_extraction({"records": [{**complete_row(), "state_updates": None}]}, 1)
    assert rows[0]["state_updates"] == []
    assert repairs == Counter(state_null_to_empty=1)
    state = {"subject": "s", "predicate": "p", "value": ""}
    rows, repairs = normalize_extraction({"records": [{**complete_row(), "state_updates": [state]}]}, 1)
    assert rows[0]["state_updates"] == [state] and repairs == Counter()


@pytest.mark.parametrize("state", [
    {"subject": "s", "predicate": "p"},
    {"subject": "s", "value": "v"},
    {"predicate": "p", "value": "v"},
    {"subject": "", "predicate": "p", "value": "v"},
    {"subject": "s", "predicate": " \t", "value": "v"},
    {"subject": 42, "predicate": "p", "value": "v"},
    {"subject": "s", "predicate": False, "value": "v"},
    {"subject": "s", "predicate": "p", "value": None},
    {"subject": "s", "predicate": "p", "value": {"text": "v"}},
    {"subject": "s", "predicate": "p", "value": ["v"]},
    {"subject": "s", "predicate": "p", "value": float("inf")},
])
@pytest.mark.parametrize("wrap", [False, True])
def test_invalid_states_never_infer_fields_or_erase_values(state, wrap):
    value = [state] if wrap else state
    with pytest.raises(ShapeError):
        normalize_extraction({"records": [{**complete_row(), "state_updates": value}]}, 1)


@pytest.mark.parametrize("states", ["synthetic", 3, [None], [True], ["synthetic"], [[{}]]])
def test_state_container_and_items_are_not_loosened(states):
    with pytest.raises(ShapeError):
        normalize_extraction({"records": [{**complete_row(), "state_updates": states}]}, 1)


@pytest.mark.parametrize("data,count,code", [
    ([], 1, "root_type"), ({}, 1, "records_missing"),
    ({"records": None}, 1, "records_type"), ({"records": {}}, 2, "records_type"),
    ({"records": []}, 1, "record_count"),
    ({"records": [complete_row(), complete_row()]}, 1, "record_count"),
    ({"records": [None]}, 1, "record_type"), ({"records": [{}]}, 1, "index_missing"),
    ({"records": [complete_row(-1)]}, 1, "index_range"),
    ({"records": [complete_row(1)]}, 1, "index_range"),
    ({"records": [complete_row(0), complete_row("00")]}, 2, "index_duplicate"),
    ({"records": [complete_row(0), complete_row(2)]}, 2, "index_range"),
    ({"records": []}, True, "invalid_count"), ({"records": []}, -1, "invalid_count"),
])
def test_source_mapping_is_mandatory_and_never_synthesized(data, count, code):
    with pytest.raises(ShapeError) as error:
        normalize_extraction(data, count)
    assert error.value.code == code


def test_empty_batch_and_safe_diagnostic_surface():
    assert normalize_extraction({"records": []}, 0) == ([], Counter())
    secret = "synthetic-private-marker"
    with pytest.raises(ShapeError) as error:
        normalize_extraction({"records": [{**complete_row(), "summary": {secret: secret}}]}, 1)
    assert secret not in str(error.value)
    assert secret not in repr(error.value)
    assert vars(error.value) == {"code": "summary_type", "path": "records[].summary"}
    assert str(ShapeError(secret, secret)) == "invalid_shape at $"


@pytest.mark.parametrize("row", [
    {"index": 0},
    {**complete_row(), "summary": ""},
    {**complete_row(), "summary": " \t", "facts": ["", " \t"]},
    {"index": 0, "summary": None, **{field: None for field in LIST_FIELDS}, "state_updates": None},
    {"index": 0, "unrecognized_model_field": "must not count as indexed evidence"},
])
def test_empty_records_cannot_turn_into_a_successful_write(row):
    with pytest.raises(ShapeError) as error:
        normalize_extraction({"records": [row]}, 1)
    assert (error.value.code, error.value.path) == ("empty_record", "records[]")


def test_facts_or_explicit_state_can_supply_content_without_summary():
    rows, _ = normalize_extraction({"records": [{"index": 0, "facts": "synthetic fact"}]}, 1)
    assert rows[0]["summary"] == "" and rows[0]["facts"] == ["synthetic fact"]
    state = {"subject": "s", "predicate": "p", "value": ""}
    rows, _ = normalize_extraction({"records": [{"index": 0, "state_updates": [state]}]}, 1)
    assert rows[0]["summary"] == "" and rows[0]["state_updates"] == [state]


@pytest.mark.parametrize("field", LIST_FIELDS)
@pytest.mark.parametrize("value", [
    {}, [{}], [[]], {"text": ""}, {"only a key is not evidence": None},
    {"nested": [{"empty": [None, [], {}, " \t"]}]},
    [None, "", {"outer": {"inner": []}}],
])
def test_empty_containers_and_keys_cannot_supply_evidence(field, value):
    with pytest.raises(ShapeError) as error:
        normalize_extraction({"records": [{"index": 0, field: value}]}, 1)
    assert (error.value.code, error.value.path) == ("empty_record", "records[]")


@pytest.mark.parametrize("leaf", [0, False, 0.0, 2.5, True, "synthetic fact"])
def test_actual_nested_leaf_values_count_even_when_falsey(leaf):
    value = {"nested": [None, {}, {"actual": leaf}]}
    rows, _ = normalize_extraction({"records": [{"index": 0, "facts": value}]}, 1)
    assert json.loads(rows[0]["facts"][0]) == value
    assert rows[0]["summary"] == ""


def test_empty_index_structure_is_retained_when_another_known_field_has_content():
    row = {"index": 0, "summary": "synthetic summary", "facts": {"empty": [[]]}}
    rows, _ = normalize_extraction({"records": [row]}, 1)
    assert rows[0]["facts"] == ['{"empty":[[]]}']
    assert rows[0]["summary"] == "synthetic summary"
