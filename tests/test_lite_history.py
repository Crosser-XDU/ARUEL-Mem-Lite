from fusion_memory.storage import Storage


def test_bounded_history_matches_original_last_two_with_tenant_and_session_isolation(tmp_path):
    store = Storage(str(tmp_path / "lite.sqlite3"))
    for n, (user, session) in enumerate([
        ("alice", "one"), ("alice", "one"), ("alice", "two"),
        ("bob", "one"), ("alice", "one"), ("alice", "two"),
    ]):
        message = {"role": "user", "content": f"message-{n}"}
        store.commit_add({"request_id": str(n), "user_id": user, "session_id": session,
                          "messages": [message]}, str(n), [{"structured": {}, "embedding": None}])
    for user, session in (("alice", "one"), ("alice", "two"), ("bob", "one"), ("unknown", "one")):
        expected = [{"role": r["role"], "raw_text": r["raw_text"]}
                    for r in store.list_records(user) if r["session_id"] == session][-2:]
        assert store.recent_session_context(user, session) == expected
    store.close()
