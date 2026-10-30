"""Pure, conservative normalization of model extraction output.

Repairs only change representation. Structured list items are serialized intact
as index text; they never create state updates or infer missing source indices.
Diagnostics contain only fixed schema paths and fixed classification codes.
"""

from collections import Counter
import json
import math
import re
from typing import Any


LIST_FIELDS = (
    "facts", "entities", "objects", "ocr", "relations", "event_time", "uncertainties",
)
STATE_FIELDS = ("subject", "predicate", "value")
_CODES = frozenset({
    "invalid_shape", "invalid_count", "root_type", "records_missing", "records_type",
    "record_count", "record_type", "index_missing", "index_type", "index_range",
    "index_duplicate", "summary_type", "list_type", "item_type", "non_json_value",
    "state_type", "state_item_type", "state_field_missing", "state_name_type",
    "state_value_type", "empty_record",
})
_PATHS = frozenset({
    "$", "count", "records", "records[]", "records[].index", "records[].summary",
    "records[].state_updates", "records[].state_updates[]",
    *(f"records[].{field}" for field in LIST_FIELDS),
    *(f"records[].{field}[]" for field in LIST_FIELDS),
    *(f"records[].state_updates[].{field}" for field in STATE_FIELDS),
})
_DECIMAL_INDEX = re.compile(r"[0-9]+\Z")


class ShapeError(ValueError):
    """Schema failure whose public details never include model-provided data."""

    def __init__(self, code: str, path: str):
        self.code = code if isinstance(code, str) and code in _CODES else "invalid_shape"
        self.path = path if isinstance(path, str) and path in _PATHS else "$"
        super().__init__(f"{self.code} at {self.path}")


def _json_value(value: Any, active: set[int]) -> bool:
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) not in (dict, list):
        return False
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    try:
        if isinstance(value, dict):
            return all(type(key) is str and _json_value(item, active)
                       for key, item in value.items())
        return all(_json_value(item, active) for item in value)
    finally:
        active.remove(identity)


def _json_text(value: Any, path: str) -> str:
    try:
        if not _json_value(value, set()):
            raise ShapeError("non_json_value", path)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                          sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ShapeError("non_json_value", path) from None


def _index(value: Any, count: int, repairs: Counter) -> int:
    path = "records[].index"
    if type(value) is int:
        result = value
    elif type(value) is str and _DECIMAL_INDEX.fullmatch(value):
        try:
            result = int(value)
        except ValueError:
            raise ShapeError("index_type", path) from None
        repairs["index_string_to_int"] += 1
    elif type(value) is float and math.isfinite(value) and value.is_integer():
        result = int(value)
        repairs["index_float_to_int"] += 1
    else:
        raise ShapeError("index_type", path)
    if not 0 <= result < count:
        raise ShapeError("index_range", path)
    return result


def _summary(value: Any, repairs: Counter) -> str:
    if value is None:
        repairs["summary_null_to_empty"] += 1
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and set(value) == {"text"} and isinstance(value["text"], str):
        repairs["summary_text_unwrapped"] += 1
        return value["text"]
    raise ShapeError("summary_type", "records[].summary")


def _list(value: Any, field: str, repairs: Counter) -> list[str]:
    path = f"records[].{field}"
    if value is None:
        repairs["list_null_to_empty"] += 1
        return []
    if isinstance(value, str):
        repairs["list_string_wrapped"] += 1
        return [value] if value.strip() else []
    if isinstance(value, dict):
        result = [_json_text(value, path + "[]")]
        repairs["list_object_wrapped"] += 1
        return result
    if not isinstance(value, list):
        raise ShapeError("list_type", path)
    result = []
    for item in value:
        if item is None:
            repairs["list_null_item_removed"] += 1
        elif isinstance(item, str):
            result.append(item)
        elif type(item) in (int, float, bool, dict, list):
            # Retain every key, value and list boundary as searchable index
            # text. This is not a conversion into authoritative state updates.
            result.append(_json_text(item, path + "[]"))
            repairs["list_item_json_encoded"] += 1
        else:
            raise ShapeError("item_type", path + "[]")
    return result


def _states(value: Any, repairs: Counter) -> list[dict]:
    path = "records[].state_updates"
    if value is None:
        repairs["state_null_to_empty"] += 1
        return []
    if isinstance(value, dict):
        for field in STATE_FIELDS:
            if field not in value:
                raise ShapeError("state_field_missing", f"{path}[].{field}")
        value = [value]
        repairs["state_object_wrapped"] += 1
    if not isinstance(value, list):
        raise ShapeError("state_type", path)
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ShapeError("state_item_type", path + "[]")
        for field in STATE_FIELDS:
            if field not in item:
                raise ShapeError("state_field_missing", f"{path}[].{field}")
        for field in ("subject", "predicate"):
            if not isinstance(item[field], str) or not item[field].strip():
                raise ShapeError("state_name_type", f"{path}[].{field}")
        normalized = dict(item)
        state_value = item["value"]
        if isinstance(state_value, str):
            pass
        elif type(state_value) in (int, float, bool):
            normalized["value"] = _json_text(state_value, path + "[].value")
            repairs["state_value_json_encoded"] += 1
        else:
            raise ShapeError("state_value_type", path + "[].value")
        result.append(normalized)
    return result


def _has_index_content(value: Any) -> bool:
    """Count original values, not JSON punctuation or object keys, as evidence.

    Index values have already passed normalization/JSON validation before this
    check. In particular, 0 and False are meaningful leaves, while nested empty
    containers and nulls are not. Strings are not parsed as another JSON layer.
    """
    if isinstance(value, str):
        return bool(value.strip())
    if type(value) in (int, bool):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_index_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_index_content(item) for item in value)
    return False


def normalize_extraction(data: Any, count: int) -> tuple[list[dict], Counter]:
    """Return source-ordered normalized rows and fixed repair-category counts.

    ``count`` counts this model batch's SOURCE units, not original messages.
    The input is not mutated. Unknown row/state fields are preserved for the
    downstream schema's existing extra-field policy, but never put in diagnostics.
    Missing optional fields retain Frame's original empty defaults.
    """
    if type(count) is not int or count < 0:
        raise ShapeError("invalid_count", "count")
    if not isinstance(data, dict):
        raise ShapeError("root_type", "$")
    if "records" not in data:
        raise ShapeError("records_missing", "records")
    rows = data["records"]
    repairs: Counter = Counter()
    if isinstance(rows, dict) and count == 1:
        rows = [rows]
        repairs["records_object_wrapped"] += 1
    if not isinstance(rows, list):
        raise ShapeError("records_type", "records")
    if len(rows) != count:
        raise ShapeError("record_count", "records")
    by_index = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ShapeError("record_type", "records[]")
        if "index" not in row:
            raise ShapeError("index_missing", "records[].index")
        index = _index(row["index"], count, repairs)
        if index in by_index:
            raise ShapeError("index_duplicate", "records[].index")
        normalized = dict(row)
        normalized["index"] = index
        for field in ("summary", *LIST_FIELDS, "state_updates"):
            if field not in row:
                normalized[field] = "" if field == "summary" else []
                repairs["optional_field_defaulted"] += 1
            elif field == "summary":
                normalized[field] = _summary(row[field], repairs)
            elif field == "state_updates":
                normalized[field] = _states(row[field], repairs)
            else:
                normalized[field] = _list(row[field], field, repairs)
        if not (normalized["summary"].strip() or normalized["state_updates"]
                or any(_has_index_content(row.get(field)) for field in LIST_FIELDS)):
            raise ShapeError("empty_record", "records[]")
        by_index[index] = normalized
    # Exact cardinality, range and uniqueness above imply complete coverage.
    return [by_index[index] for index in range(count)], repairs
