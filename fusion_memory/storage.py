"""Transactional, tenant-scoped raw evidence and semantic memory storage.

Each public operation opens a short-lived connection.  SQLite serializes writers,
so request deduplication, session positions, evidence and semantic records commit
as a single unit even when several workers accept the same request concurrently.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class ConflictError(ValueError):
    """An existing tenant/request identifier was reused for different input."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, request_id)
);
CREATE TABLE IF NOT EXISTS image_blobs (
    user_id TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (user_id, sha256)
);
CREATE TABLE IF NOT EXISTS records (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    timestamp_ms INTEGER,
    created_at TEXT NOT NULL,
    ingested_at_ms INTEGER NOT NULL,
    raw_content TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    structured TEXT NOT NULL,
    embedding TEXT,
    embedding_model TEXT,
    has_image INTEGER NOT NULL CHECK (has_image IN (0, 1)),
    image_hashes TEXT NOT NULL,
    UNIQUE (user_id, session_id, position),
    UNIQUE (user_id, id),
    FOREIGN KEY (user_id, request_id) REFERENCES requests (user_id, request_id)
);
CREATE INDEX IF NOT EXISTS records_user_seq ON records (user_id, seq);
CREATE INDEX IF NOT EXISTS records_user_session_seq ON records (user_id, session_id, seq);
CREATE TABLE IF NOT EXISTS image_occurrences (
    user_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    part_index INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY (user_id, record_id, part_index),
    FOREIGN KEY (user_id, record_id) REFERENCES records (user_id, id),
    FOREIGN KEY (user_id, sha256) REFERENCES image_blobs (user_id, sha256)
);
CREATE TABLE IF NOT EXISTS state_versions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    update_index INTEGER NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value_json TEXT NOT NULL,
    effective_at_ms INTEGER NOT NULL,
    timestamp_is_fallback INTEGER NOT NULL,
    UNIQUE (user_id, record_id, update_index),
    FOREIGN KEY (user_id, record_id) REFERENCES records (user_id, id)
);
CREATE INDEX IF NOT EXISTS states_user_time
    ON state_versions (user_id, effective_at_ms, seq);
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _pack_content(content: str | list[dict]) -> tuple[Any, str, list[dict]]:
    """Replace image payloads with references while preserving every occurrence."""
    if isinstance(content, str):
        return content, content, []
    if not isinstance(content, list):
        raise ValueError("content must be a string or an ordered list")
    packed = copy.deepcopy(content)
    texts: list[str] = []
    images: list[dict] = []
    for index, part in enumerate(packed):
        if not isinstance(part, dict):
            raise ValueError("content parts must be objects")
        if part.get("type") == "text":
            if not isinstance(part.get("text"), str):
                raise ValueError("text content parts require text")
            texts.append(part["text"])
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str) or "," not in url:
                raise ValueError("image_url requires an inline base64 data URI")
            prefix, encoded = url.split(",", 1)
            if prefix.lower() not in {
                "data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"
            }:
                raise ValueError("only inline JPEG, PNG and WebP images are supported")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid image base64") from exc
            if not data:
                raise ValueError("image bytes cannot be empty")
            digest = hashlib.sha256(data).hexdigest()
            reference = {"blob_sha256": digest, "data_uri_prefix": prefix}
            # Noncanonical padding bits are legal to some decoders. Preserve the
            # supplied spelling as well as the original image bytes in that case.
            if base64.b64encode(data).decode("ascii") != encoded:
                reference["original_base64"] = encoded
            image["url"] = reference
            images.append({"index": index, "sha256": digest, "data": data})
        else:
            raise ValueError("unsupported content part type")
    return packed, "\n".join(texts), images


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._keeper: sqlite3.Connection | None = None
        self._uri = db_path == ":memory:"
        if self._uri:
            # A keeper only anchors an explicitly requested in-memory database;
            # all public operations still use separate connections.
            self._target = f"file:fusion_memory_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._keeper = self._connect()
        else:
            path = Path(db_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._target = str(path)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._target, uri=self._uri, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    @staticmethod
    def _check_existing(conn: sqlite3.Connection, user_id: str, request_id: str, fingerprint: str) -> bool:
        row = conn.execute(
            "SELECT fingerprint FROM requests WHERE user_id=? AND request_id=?",
            (user_id, request_id),
        ).fetchone()
        if row is None:
            return False
        if row["fingerprint"] != fingerprint:
            raise ConflictError("request_id already exists with different content for this user")
        return True

    def check_request(self, user_id: str, request_id: str, fingerprint: str) -> bool:
        """Return whether exactly this request committed; mismatched reuse fails."""
        with closing(self._connect()) as conn:
            return self._check_existing(conn, user_id, request_id, fingerprint)

    def commit_add(self, payload: dict, fingerprint: str, prepared: list[dict]) -> None:
        """Commit a whole Add atomically, including deduplication and state history.

        The caller computes the fingerprint from validated original input. A
        concurrent identical commit is a no-op; a conflicting one always fails.
        """
        user_id = _identifier(payload.get("user_id"), "user_id")
        session_id = _identifier(payload.get("session_id"), "session_id")
        request_id = _identifier(payload.get("request_id"), "request_id")
        _identifier(fingerprint, "fingerprint")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if len(prepared) != len(messages):
            raise ValueError("prepared must contain exactly one entry per message")

        with self._transaction() as conn:
            if self._check_existing(conn, user_id, request_id, fingerprint):
                return
            created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            ingested_at_ms = time.time_ns() // 1_000_000
            conn.execute(
                "INSERT INTO requests VALUES (?, ?, ?, ?, ?)",
                (user_id, request_id, session_id, fingerprint, created_at),
            )
            next_position = conn.execute(
                "SELECT COALESCE(MAX(position), -1)+1 FROM records WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).fetchone()[0]
            for offset, (message, semantic) in enumerate(zip(messages, prepared, strict=True)):
                role = message.get("role")
                if role not in {"user", "assistant"}:
                    raise ValueError("role must be user or assistant")
                timestamp = message.get("timestamp")
                if timestamp is not None and (not isinstance(timestamp, int) or isinstance(timestamp, bool)):
                    raise ValueError("timestamp must be Unix milliseconds as an integer")
                structured = semantic.get("structured", {})
                if not isinstance(structured, dict):
                    raise ValueError("structured memory must be an object")
                embedding = semantic.get("embedding")
                if embedding is not None and (
                    not isinstance(embedding, list) or not embedding
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in embedding)
                ):
                    raise ValueError("embedding must be a nonempty numeric list or null")
                packed, raw_text, images = _pack_content(message["content"])
                record_id = uuid.uuid4().hex
                conn.execute(
                    """INSERT INTO records (
                        id, user_id, session_id, request_id, position, role,
                        timestamp_ms, created_at, ingested_at_ms, raw_content,
                        raw_text, structured, embedding, embedding_model, has_image, image_hashes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record_id, user_id, session_id, request_id, next_position + offset, role,
                        timestamp, created_at, ingested_at_ms, _json(packed), raw_text,
                        _json(structured), _json(embedding) if embedding is not None else None,
                        semantic.get("embedding_model"), int(bool(images)),
                        _json([item["sha256"] for item in images]),
                    ),
                )
                for item in images:
                    conn.execute(
                        "INSERT OR IGNORE INTO image_blobs (user_id, sha256, data) VALUES (?, ?, ?)",
                        (user_id, item["sha256"], item["data"]),
                    )
                    conn.execute(
                        "INSERT INTO image_occurrences VALUES (?, ?, ?, ?)",
                        (user_id, record_id, item["index"], item["sha256"]),
                    )
                # Assistant statements remain searchable evidence, but cannot
                # promote their own assertions into the user's current state.
                updates = structured.get("state_updates", []) if role == "user" else []
                if not isinstance(updates, list):
                    updates = []
                for update_index, update in enumerate(updates):
                    if not isinstance(update, dict) or "value" not in update:
                        continue
                    subject, predicate = update.get("subject"), update.get("predicate")
                    if not isinstance(subject, str) or not subject or not isinstance(predicate, str) or not predicate:
                        continue
                    conn.execute(
                        """INSERT INTO state_versions (
                            id, user_id, record_id, update_index, subject, predicate,
                            value_json, effective_at_ms, timestamp_is_fallback
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            uuid.uuid4().hex, user_id, record_id, update_index, subject,
                            predicate, _json(update["value"]),
                            timestamp if timestamp is not None else ingested_at_ms, int(timestamp is None),
                        ),
                    )

    def recent_session_context(self, user_id: str, session_id: str) -> list[dict]:
        """Same last-two context as before, without loading a tenant's embeddings."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, raw_text FROM records WHERE user_id=? AND session_id=? ORDER BY seq DESC LIMIT 2",
                (user_id, session_id),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_records(self, user_id: str) -> list[dict]:
        """Read searchable records in ingestion order without loading raw images."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT id, user_id, session_id, request_id, position, role, timestamp_ms,
                    created_at, ingested_at_ms, raw_text, structured, embedding,
                    embedding_model, has_image, image_hashes
                    FROM records WHERE user_id=? ORDER BY seq""",
                (user_id,),
            ).fetchall()
        result = []
        for row in rows:
            record = dict(row)
            record["structured"] = json.loads(record["structured"])
            record["embedding"] = json.loads(record["embedding"]) if record["embedding"] is not None else None
            record["image_hashes"] = json.loads(record["image_hashes"])
            record["has_image"] = bool(record["has_image"])
            result.append(record)
        return result

    def get_content(self, user_id: str, record_id: str) -> str | list:
        """Reconstruct raw ordered content, denying missing or foreign records."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT raw_content FROM records WHERE user_id=? AND id=?", (user_id, record_id)
            ).fetchone()
            if row is None:
                raise KeyError("record not found for this user")
            content = json.loads(row["raw_content"])
            if isinstance(content, list):
                for part in content:
                    if part.get("type") != "image_url":
                        continue
                    reference = part["image_url"]["url"]
                    blob = conn.execute(
                        "SELECT data FROM image_blobs WHERE user_id=? AND sha256=?",
                        (user_id, reference["blob_sha256"]),
                    ).fetchone()
                    if blob is None:
                        raise RuntimeError("stored image evidence is missing")
                    encoded = reference.get("original_base64")
                    if encoded is None:
                        encoded = base64.b64encode(blob["data"]).decode("ascii")
                    part["image_url"]["url"] = f"{reference['data_uri_prefix']},{encoded}"
            return content

    def list_states(self, user_id: str) -> list[dict]:
        """Return all state versions and validity intervals derived from source time.

        Intervals are [valid_from_ms, valid_to_ms). Equal source times are ordered
        by ingestion and update index. Missing timestamps fall back to ingestion
        time and are explicitly flagged; no historical evidence is overwritten.
        Only user messages establish state. Version chains use exact (subject,
        predicate) strings; synonymous natural-language keys are not merged.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT s.id, s.user_id, s.record_id, s.subject, s.predicate,
                    s.value_json, s.effective_at_ms, s.timestamp_is_fallback,
                    r.session_id, r.request_id, r.position, r.timestamp_ms, r.created_at
                    FROM state_versions s
                    JOIN records r ON r.user_id=s.user_id AND r.id=s.record_id
                    WHERE s.user_id=? AND r.role='user'
                    ORDER BY s.effective_at_ms, s.seq""",
                (user_id,),
            ).fetchall()
        states: list[dict] = []
        latest: dict[tuple[str, str], dict] = {}
        for row in rows:
            state = dict(row)
            state["value"] = json.loads(state.pop("value_json"))
            state["timestamp_is_fallback"] = bool(state["timestamp_is_fallback"])
            state["valid_from_ms"] = state["effective_at_ms"]
            state["valid_to_ms"] = None
            state["is_current"] = True
            key = (state["subject"], state["predicate"])
            if key in latest:
                latest[key]["valid_to_ms"] = state["effective_at_ms"]
                latest[key]["is_current"] = False
            latest[key] = state
            states.append(state)
        return states

    def close(self) -> None:
        """Release the optional in-memory database keeper (file DBs need no close)."""
        if self._keeper is not None:
            self._keeper.close()
            self._keeper = None
