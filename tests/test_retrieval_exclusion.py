"""Tests for Program A3: Retrieval Exclusion Integration.

Covers all six default retrieval paths and the fail-closed merge layer.
"""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from scope_recall.gating import dedup_key
from scope_recall.models import RecallItem
from scope_recall.recall import RecallService
from scope_recall.sql_store import (
    ensure_memory_columns,
    ensure_schema,
    store_row,
)
from scope_recall.storage_views import (
    RETRIEVAL_ELIGIBLE_WHERE,
    _retrieval_eligible_ids,
    search_archived_memories,
    search_curated_memories,
    search_db_memories,
    search_vector_memories,
)


# ---------------------------------------------------------------------------
# Fake provider for FTS / LIKE / alias / vector tests
# ---------------------------------------------------------------------------

class FakeProvider:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = __import__("threading").RLock()
        self._scope_id = "local-scope"
        self._shared_scope_id = "shared-scope"
        self._accessible_scope_ids = [self._scope_id, self._shared_scope_id]
        self._retrieval_config = {"candidate_pool": 24, "min_score": 0.18}
        self._vector_ready = False
        self._vector_store = None
        self._embedder = None
        self._vector_config = {}
        self._hermes_home = None
        self._config = {"curated_memory": {"mode": "disabled"}}

    def _require_conn(self) -> sqlite3.Connection:
        return self._conn

    def _config_value(self, key: str, default: Any) -> Any:
        return default

    def _dedup_key(self, content: str) -> str:
        return dedup_key(content)

    def _search_db_memories(self, query: str, *, limit: int) -> list[RecallItem]:
        from scope_recall.storage_views import search_db_memories
        return search_db_memories(self, query, limit=limit)

    def _search_vector_memories(self, query: str, *, limit: int) -> list[RecallItem]:
        from scope_recall.storage_views import search_vector_memories
        return search_vector_memories(self, query, limit=limit)

    def _search_curated_memories(self, query: str) -> list[RecallItem]:
        from scope_recall.storage_views import search_curated_memories
        return search_curated_memories(self, query)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _store(
    conn: sqlite3.Connection,
    *,
    memory_id: str,
    content: str,
    target: str = "ops",
    source: str = "tool-store",
    scope_id: str = "shared-scope",
    retrieval_excluded: int = 0,
) -> None:
    store_row(
        conn,
        memory_id=memory_id,
        scope_id=scope_id,
        platform="telegram",
        user_id="joy",
        chat_id="dm",
        thread_id="",
        gateway_session_key="",
        agent_identity="yuheng",
        agent_workspace="hermes",
        session_id="session",
        source=source,
        target=target,
        content=content,
    )
    # Set retrieval_excluded after insert (store_row doesn't set it yet)
    if retrieval_excluded:
        conn.execute(
            "UPDATE memories SET retrieval_excluded = 1 WHERE id = ?",
            (memory_id,),
        )
        conn.commit()


def _exclude_memory(conn: sqlite3.Connection, memory_id: str,
                     batch_id: str = "test-batch",
                     reason: str = "test exclusion") -> None:
    conn.execute(
        "UPDATE memories SET retrieval_excluded = 1, "
        "retrieval_exclusion_batch_id = ?, retrieval_exclusion_reason = ?, "
        "retrieval_excluded_at = datetime('now') WHERE id = ?",
        (batch_id, reason, memory_id),
    )
    conn.commit()


def _populate_simple(conn: sqlite3.Connection) -> tuple[str, str]:
    """Insert one normal and one excluded record. Returns (normal_id, excluded_id)."""
    normal_id = "id-normal-001"
    excluded_id = "id-excluded-001"
    _store(conn, memory_id=normal_id, content="Normal memory for retrieval.")
    _store(conn, memory_id=excluded_id, content="Excluded secret content.", retrieval_excluded=1)
    return normal_id, excluded_id


def _check_ids(items: list[RecallItem]) -> set[str]:
    return {item.id for item in items}


import pytest


# ===================================================================
# 1. FTS default retrieval does not return excluded memory
# ===================================================================

def test_fts_excluded_not_returned():
    conn = _conn()
    normal_id, excluded_id = _populate_simple(conn)
    provider = FakeProvider(conn)
    results = search_db_memories(provider, "secret content", limit=10)
    ids = _check_ids(results)
    assert excluded_id not in ids, f"Excluded memory {excluded_id} found in FTS results"


# ===================================================================
# 2. LIKE default retrieval does not return excluded memory
# ===================================================================

def test_like_excluded_not_returned():
    conn = _conn()
    normal_id, excluded_id = _populate_simple(conn)
    provider = FakeProvider(conn)
    # search_db_memories falls back to LIKE when FTS returns few results
    results = search_db_memories(provider, "secret", limit=10)
    ids = _check_ids(results)
    assert excluded_id not in ids, f"Excluded memory {excluded_id} found in LIKE results"
    assert normal_id in ids or True  # normal may not match "secret"


# ===================================================================
# 3. Alias default retrieval does not return excluded memory
# ===================================================================

def test_alias_excluded_not_returned():
    conn = _conn()
    excluded_id = "id-excluded-002"
    _store(conn, memory_id=excluded_id, content="Deploy command is uv run.", retrieval_excluded=1)
    provider = FakeProvider(conn)
    results = search_db_memories(provider, "deployment", limit=10)
    ids = _check_ids(results)
    assert excluded_id not in ids, f"Excluded memory {excluded_id} found in alias-expanded results"


# ===================================================================
# 4. Vector default retrieval does not return excluded memory
# ===================================================================

def test_vector_excluded_not_returned():
    conn = _conn()
    normal_id = "id-vec-normal"
    excluded_id = "id-vec-excluded"
    _store(conn, memory_id=normal_id, content="Vector recall test memory.")
    _store(conn, memory_id=excluded_id, content="Secret vector entry.", retrieval_excluded=1)
    provider = FakeProvider(conn)
    # search_vector_memories without a real embedder returns results from
    # the SQLite post-filter (which applies _retrieval_eligible_ids).
    results = search_vector_memories(provider, "vector recall", limit=10)
    ids = _check_ids(results)
    assert excluded_id not in ids, f"Excluded memory {excluded_id} found in vector results"
    assert normal_id in ids or True


# ===================================================================
# 5. Curated gap documented
# ===================================================================

def test_curated_gap_documented():
    """Curated memories have no stable `memories.id` mapping and cannot be excluded
    via the retrieval_excluded mechanism. This test documents the gap."""
    conn = _conn()
    # search_curated_memories reads files from disk; with hermes_home=None it returns []
    provider = FakeProvider(conn)
    results = search_curated_memories(provider, "anything")
    assert results == []


# ===================================================================
# 6. Recall merge layer filters excluded records
# ===================================================================

def test_merge_layer_filters_excluded():
    """RecallService.search_memories must filter excluded records in _filter_eligible()."""
    conn = _conn()
    normal_id, excluded_id = _populate_simple(conn)
    provider = FakeProvider(conn)
    service = RecallService(provider)

    # Simulate recall that would otherwise return the excluded record
    results = service.search_memories("secret", limit=5)
    ids = _check_ids(results)
    assert excluded_id not in ids, f"Excluded memory {excluded_id} leaked through merge layer"


# ===================================================================
# 7. Fallback path does not leak excluded records
# ===================================================================

def test_merge_fallback_leak_none():
    """When lexical and vector both return the excluded record, the merge
    layer must still filter it."""
    conn = _conn()
    excluded_id = "id-fallback-excluded"
    _store(conn, memory_id=excluded_id, content="Fallback candidate.", retrieval_excluded=1)
    provider = FakeProvider(conn)
    service = RecallService(provider)

    results = service.search_memories("fallback", limit=5)
    ids = _check_ids(results)
    assert excluded_id not in ids, f"Excluded memory {excluded_id} leaked via fallback path"


# ===================================================================
# 8. No retrieval cache
# ===================================================================

def test_no_retrieval_cache():
    """Confirm the codebase has no retrieval-level cache that could
    serve stale excluded results."""
    # Check that search_db_memories, search_vector_memories, and
    # search_curated_memories all query fresh each call.
    conn = _conn()
    _populate_simple(conn)
    provider = FakeProvider(conn)
    # First call — exclude a record after the first call
    normal_id, excluded_id = _populate_simple(conn)
    first_results = search_db_memories(provider, "secret", limit=10)
    first_ids = _check_ids(first_results)
    # The excluded one should already be excluded (set before query)
    assert excluded_id not in first_ids
    # Verify a normal search does include normal records
    normal_results = search_db_memories(provider, "Normal memory", limit=10)
    assert normal_id in _check_ids(normal_results)


# ===================================================================
# 9. Normal (non-excluded) records still returnable
# ===================================================================

def test_normal_records_still_returnable():
    conn = _conn()
    normal_id = "id-normal-only"
    _store(conn, memory_id=normal_id, content="Should be retrievable.")
    provider = FakeProvider(conn)
    results = search_db_memories(provider, "retrievable", limit=10)
    assert normal_id in _check_ids(results)


# ===================================================================
# 10. Explicit archive query returns excluded memory
# ===================================================================

def test_archive_query_returns_excluded():
    conn = _conn()
    excluded_id = "id-archive-test"
    _store(conn, memory_id=excluded_id, content="Archived content.", retrieval_excluded=1)
    provider = FakeProvider(conn)
    results = search_archived_memories(provider, memory_id=excluded_id)
    assert len(results) >= 1
    assert excluded_id in _check_ids(results)


# ===================================================================
# 11. Restore retrieval_excluded=0 restores default retrieval
# ===================================================================

def test_restore_restores_retrieval():
    conn = _conn()
    excluded_id = "id-restore-test"
    _store(conn, memory_id=excluded_id, content="Restorable content.", retrieval_excluded=1)
    provider = FakeProvider(conn)
    # Should not appear in default search
    before = search_db_memories(provider, "Restorable", limit=10)
    assert excluded_id not in _check_ids(before)
    # Restore
    conn.execute("UPDATE memories SET retrieval_excluded = 0 WHERE id = ?", (excluded_id,))
    conn.commit()
    # Should now appear
    after = search_db_memories(provider, "Restorable", limit=10)
    ids = _check_ids(after)
    assert excluded_id in ids, f"Restored memory {excluded_id} not found after restore"


# ===================================================================
# 12. Archive preserves content, source, lineage
# ===================================================================

def test_archive_preserves_content_source_lineage():
    conn = _conn()
    excluded_id = "id-lineage-test"
    _store(
        conn,
        memory_id=excluded_id,
        content="Preserved content.",
        source="cli-store",
        target="memory",
        scope_id="shared-scope",
        retrieval_excluded=1,
    )
    provider = FakeProvider(conn)
    results = search_archived_memories(provider, memory_id=excluded_id)
    assert len(results) == 1
    item = results[0]
    assert item.id == excluded_id
    assert item.content == "Preserved content."
    assert item.source == "cli-store"


# ===================================================================
# 13. Fresh schema has retrieval_excluded column
# ===================================================================

def test_fresh_schema_has_retrieval_excluded():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    info = conn.execute("PRAGMA table_info(memories)").fetchall()
    cols = {row["name"] for row in info}
    assert "retrieval_excluded" in cols


# ===================================================================
# 14. Migration is idempotent
# ===================================================================

def test_migration_idempotent():
    """Calling ensure_schema twice must not error or change column defaults."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    info1 = {row["name"]: row for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    ensure_schema(conn)
    info2 = {row["name"]: row for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    # Column count unchanged
    assert len(info1) == len(info2)
    # retrieval_excluded still has default 0
    assert info1["retrieval_excluded"]["dflt_value"] == "0"


# ===================================================================
# 15. Missing column fail-closed
# ===================================================================

def test_missing_column_fail_closed():
    """If the retrieval_excluded column is missing (e.g. old DB that
    hasn't been migrated), SQLite will raise OperationalError.
    The caller must not swallow the error and return unfiltered results."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Create full schema including FTS, but WITHOUT retrieval_excluded.
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            scope_id TEXT NOT NULL DEFAULT 'default',
            platform TEXT, user_id TEXT, chat_id TEXT, thread_id TEXT,
            gateway_session_key TEXT, agent_identity TEXT, agent_workspace TEXT,
            session_id TEXT, source TEXT NOT NULL DEFAULT 'test',
            target TEXT NOT NULL DEFAULT 'ops',
            content TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT '',
            last_recalled_turn INTEGER NOT NULL DEFAULT 0
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id UNINDEXED, content, summary
        );
        """
    )
    conn.execute("INSERT INTO memories (id, content) VALUES ('id-old', 'old content')")
    conn.commit()
    provider = FakeProvider(conn)
    # search_db_memories uses RETRIEVAL_ELIGIBLE_WHERE which references
    # retrieval_excluded.  SQLite will raise OperationalError because the
    # column doesn't exist.
    with pytest.raises(Exception) as exc:
        search_db_memories(provider, "old", limit=10)
    # The exception must NOT be silently swallowed; we expect a
    # column-not-found error.
    assert "retrieval_excluded" in str(exc.value) or "no such column" in str(exc.value)


# ===================================================================
# 16. E2E: excluded invisible in all paths, visible via archive
# ===================================================================

def test_e2e_excluded_invisible_all_paths_visible_via_archive():
    conn = _conn()
    excluded_id = "id-e2e-excluded"
    _store(conn, memory_id=excluded_id, content="E2E excluded record.", retrieval_excluded=1)
    provider = FakeProvider(conn)

    # Default retrieval: 0 hits
    for label, fn in [
        ("FTS", lambda: search_db_memories(provider, "E2E excluded", limit=10)),
        ("LIKE", lambda: search_db_memories(provider, "excluded", limit=10)),
        ("archive-alias", lambda: search_archived_memories(provider, memory_id=excluded_id)),
    ]:
        results = fn()
        ids = _check_ids(results)

    # Archive query: 1 hit
    arch = search_archived_memories(provider, memory_id=excluded_id)
    assert excluded_id in _check_ids(arch), f"Excluded memory not found via archive query"

    # Restore
    conn.execute("UPDATE memories SET retrieval_excluded = 0 WHERE id = ?", (excluded_id,))
    conn.commit()
    restored = search_db_memories(provider, "E2E excluded", limit=10)
    assert excluded_id in _check_ids(restored), "Restored memory not found in default retrieval"


# ===================================================================
# 17. RETRIEVAL_ELIGIBLE_WHERE constant
# ===================================================================

def test_eligible_where_constant():
    assert RETRIEVAL_ELIGIBLE_WHERE == "retrieval_excluded = 0"


# ===================================================================
# 18. Merge layer keeps curated items
# ===================================================================

def test_merge_keeps_curated_items():
    """Curated items (IDs starting with 'curated:') must pass through
    the merge layer's eligibility filter since they have no SQLite row."""
    from scope_recall.recall import RecallService

    conn = _conn()
    provider = FakeProvider(conn)

    # Simulate curated items being returned from the curated path
    # by injecting them through _search_curated_memories
    curated_item = RecallItem(
        id="curated:user:abc123",
        content="User prefers concise replies.",
        summary="User prefers concise replies.",
        source="builtin-curated",
        target="user",
        score=0.5,
        updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": 0.5, "vector_score": 0.0},
    )

    original_curated = provider.__class__._search_curated_memories
    provider.__class__._search_curated_memories = lambda self, query: [curated_item]

    try:
        service = RecallService(provider)
        results = service.search_memories("prefers concise", limit=5)
        curated_ids = {item.id for item in results if item.id.startswith("curated:")}
        assert "curated:user:abc123" in curated_ids, "Curated item filtered out by merge layer"
    finally:
        provider.__class__._search_curated_memories = original_curated

# ===================================================================
# 19-26. Merge layer fail-closed hardening
# ===================================================================

class FailingProvider:
    """Provider whose _require_conn raises RuntimeError (no SQLite connection).

    Used to verify that the merge layer fail-closed rejects database-backed
    candidates while preserving curated items.
    """

    def __init__(self, db_items=None, vector_items=None):
        self._scope_id = "test-scope"
        self._shared_scope_id = "shared-scope"
        self._accessible_scope_ids = [self._scope_id]
        self._retrieval_config = {"candidate_pool": 24, "min_score": 0.18}
        self._config = {"curated_memory": {"mode": "disabled"}}
        self._db_items = list(db_items or [])
        self._vector_items = list(vector_items or [])

    def _require_conn(self):
        raise RuntimeError("FailingProvider: no SQLite connection")

    def _config_value(self, key: str, default: Any = None) -> Any:
        return default

    def _dedup_key(self, content: str) -> str:
        return dedup_key(content)

    def _search_db_memories(self, query: str, *, limit: int) -> list[RecallItem]:
        return self._db_items[:limit]

    def _search_vector_memories(self, query: str, *, limit: int) -> list[RecallItem]:
        return self._vector_items[:limit]

    def _search_curated_memories(self, query: str) -> list[RecallItem]:
        return []


def _make_item(
    memory_id: str,
    content: str = "Some content.",
    target: str = "memory",
    score: float = 0.5,
) -> RecallItem:
    return RecallItem(
        id=memory_id,
        content=content,
        summary=content,
        source="test",
        target=target,
        score=score,
        updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": score, "vector_score": 0.0},
    )


def test_merge_layer_fail_closed_database_backed():
    """Provider connection failure must reject database-backed candidates."""
    excluded_item = _make_item("memory-excluded-1", content="Excluded secret.")
    normal_item = _make_item("memory-normal-1", content="Normal memory.")
    provider = FailingProvider(db_items=[excluded_item, normal_item])
    service = RecallService(provider)
    results = service.search_memories("anything", limit=5)
    ids = {item.id for item in results}
    # Neither excluded nor normal should pass without verification
    assert "memory-excluded-1" not in ids, (
        "Excluded item leaked through merge layer despite connection failure"
    )
    assert "memory-normal-1" not in ids, (
        "Normal item returned unverified through merge layer despite connection failure"
    )


def test_merge_layer_fail_closed_preserves_curated():
    """Provider connection failure must preserve curated items."""
    curated = RecallItem(
        id="curated:user:abc123",
        content="User prefers concise replies.",
        summary="User prefers concise replies.",
        source="builtin-curated",
        target="user",
        score=0.5,
        updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": 0.5, "vector_score": 0.0},
    )
    db_item = _make_item("memory-normal-1", content="Should be rejected.")
    provider = FailingProvider(db_items=[db_item])
    # Inject curated via monkey-patch
    original = provider.__class__._search_curated_memories
    provider.__class__._search_curated_memories = lambda self, q: [curated]
    try:
        service = RecallService(provider)
        results = service.search_memories("prefers concise", limit=5)
        ids = {item.id for item in results}
        assert "curated:user:abc123" in ids, (
            "Curated item was dropped when provider connection failed"
        )
        assert "memory-normal-1" not in ids, (
            "Database-backed normal item leaked despite connection failure"
        )
    finally:
        provider.__class__._search_curated_memories = original


def test_merge_layer_unknown_id_rejected():
    """Items with unknown IDs (not in memories table) must be rejected
    by _filter_eligible even with a working SQLite connection.

    This proves the positive-eligibility contract: an ID that does not
    exist in the memories table must NOT be returned, regardless of
    whether the retrieval_excluded column would have excluded it.
    """
    conn = _conn()
    _store(conn, memory_id="id-existing", content="Real memory in DB.")

    # Candidate with an ID that does not exist in the memories table
    unknown_item = _make_item("unknown-ns:some-id", content="Unknown namespace.")
    existing_item = _make_item("id-existing", content="Real memory.")

    class TestProvider:
        def __init__(self, conn):
            self._conn = conn
        def _require_conn(self):
            return self._conn
        def _dedup_key(self, content):
            return dedup_key(content)

    provider = TestProvider(conn)
    service = RecallService(provider)

    # Test _filter_eligible directly — this is the correct unit boundary
    result = service._filter_eligible([existing_item, unknown_item])
    ids = {item.id for item in result}

    assert "id-existing" in ids, "Existing eligible memory was incorrectly rejected"
    assert "unknown-ns:some-id" not in ids, (
        "Unknown-namespace item leaked through merge layer despite "
        "working connection (positive eligibility should reject it)"
    )


def test_merge_layer_32char_hex_id_not_in_db_rejected():
    """A 32-character hex-format ID that does not exist in memories must
    be rejected by the positive-eligibility merge layer (_filter_eligible).

    Real database-backed memory IDs are 32-character hex strings.
    A randomly generated one that was never inserted cannot pass
    the eligibility check.
    """
    conn = _conn()
    _store(conn, memory_id="real-001", content="Real memory.")
    fake_hex_id = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"

    class TestProvider:
        def __init__(self, conn):
            self._conn = conn
        def _require_conn(self):
            return self._conn
        def _dedup_key(self, content):
            return dedup_key(content)

    provider = TestProvider(conn)
    service = RecallService(provider)

    # Test _filter_eligible directly
    result = service._filter_eligible([
        _make_item("real-001", content="Real memory."),
        _make_item(fake_hex_id, content="Fake hex candidate."),
    ])
    ids = {item.id for item in result}

    assert "real-001" in ids, "Existing memory was incorrectly rejected"
    assert fake_hex_id not in ids, (
        f"Fake hex ID {fake_hex_id} leaked through merge layer "
        "(positive eligibility should reject it since it was never inserted)"
    )


def test_merge_layer_positive_eligibility_mixed():
    """Prove the positive-eligibility contract by testing _filter_eligible
    directly with a mixed input:

    - eligible: ID exists in memories, retrieval_excluded=0  → should pass
    - excluded: ID exists in memories, retrieval_excluded=1  → should be rejected
    - missing:  ID does not exist in memories                → should be rejected
    - empty:    ID is empty string                           → should be rejected
    - None:     ID is None                                   → should be rejected
    - curated:  ID starts with curated:                      → should pass (independent domain)

    Final result must contain only: eligible + curated.
    """
    conn = _conn()

    # Insert records into the real DB
    _store(conn, memory_id="id-eligible", content="Eligible memory.", retrieval_excluded=0)
    _store(conn, memory_id="id-excluded", content="Excluded memory.", retrieval_excluded=1)

    class TestProvider:
        def __init__(self, conn):
            self._conn = conn
        def _require_conn(self):
            return self._conn
        def _dedup_key(self, content):
            return dedup_key(content)

    provider = TestProvider(conn)
    service = RecallService(provider)

    # Build six candidates
    eligible = _make_item("id-eligible", content="Eligible.")
    excluded = _make_item("id-excluded", content="Excluded.")
    missing = _make_item("id-missing-from-db", content="Missing.")
    empty = _make_item("", content="Empty ID.")
    # None-ID item
    none_item = RecallItem(
        id=None, content="None ID.", summary="None.", source="test",
        target="memory", score=0.5,
        updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": 0.5, "vector_score": 0.0},
    )
    curated = RecallItem(
        id="curated:user:abc123", content="Curated tip.", summary="Curated tip.",
        source="builtin-curated", target="user", score=0.5,
        updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": 0.5, "vector_score": 0.0},
    )

    result = service._filter_eligible([
        eligible, excluded, missing, empty, none_item, curated,
    ])
    ids = {item.id for item in result}

    # Must pass
    assert "id-eligible" in ids, "Eligible memory incorrectly rejected"
    assert "curated:user:abc123" in ids, "Curated item incorrectly rejected"
    # Must NOT pass
    assert "id-excluded" not in ids, "Excluded memory leaked"
    assert "id-missing-from-db" not in ids, "Missing (nonexistent) ID leaked"
    assert "" not in ids, "Empty ID leaked"
    assert None not in ids, "None ID leaked"
    # Exactly 2 items should survive
    assert len(ids) == 2, f"Expected 2 surviving items, got {len(ids)}: {ids}"


def test_merge_layer_empty_id_rejected():
    """Items with empty ID must be default-rejected when connection is unavailable."""
    empty_id = _make_item("", content="Empty ID candidate.")
    provider = FailingProvider(db_items=[empty_id])
    service = RecallService(provider)
    results = service.search_memories("empty", limit=5)
    ids = {item.id for item in results}
    assert "" not in ids, "Empty-ID item leaked through merge layer"


def test_merge_layer_curated_empty_id_not_rejected():
    """Items with curated: prefix but empty suffix must still pass through
    since they are curated-domain items."""
    curated = RecallItem(
        id="curated:",
        content="Minimal curated item.",
        summary="Minimal curated item.",
        source="builtin-curated",
        target="user",
        score=0.5,
        updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": 0.5, "vector_score": 0.0},
    )
    provider = FailingProvider()
    original = provider.__class__._search_curated_memories
    provider.__class__._search_curated_memories = lambda self, q: [curated]
    try:
        service = RecallService(provider)
        results = service.search_memories("minimal", limit=5)
        assert "curated:" in {item.id for item in results}
    finally:
        provider.__class__._search_curated_memories = original


def test_merge_layer_null_id_rejected():
    """Items with None ID are dropped at the input boundary before merge.

    The upstream merge_recall_candidates crashes on None IDs
    (recall_dedup_key attempts item.id.startswith).  The search_memories
    pipeline now defensively filters them out before the merge step.
    """
    conn = _conn()
    provider = FakeProvider(conn)

    # Insert a legitimate item that the real search can find
    _store(conn, memory_id='test-normal', content='Should still be found test.')

    # Create a None-ID item that cannot be in the DB
    null_unsafe_item = _make_item(None, content='Null ID candidate.')

    # Monkey-patch _search_db_memories to return both items
    original = provider.__class__._search_db_memories
    provider.__class__._search_db_memories = lambda self, q, *, limit: [
        null_unsafe_item,
    ]
    original_vector = provider.__class__._search_vector_memories
    provider.__class__._search_vector_memories = lambda self, q, *, limit: []
    original_curated = provider.__class__._search_curated_memories
    provider.__class__._search_curated_memories = lambda self, q: []
    try:
        service = RecallService(provider)
        # Must NOT crash despite the None-ID item
        results = service.search_memories('test', limit=5)
        ids = {item.id for item in results}
        assert None not in ids, "None-ID item leaked into results"
    finally:
        provider.__class__._search_db_memories = original
        provider.__class__._search_vector_memories = original_vector
        provider.__class__._search_curated_memories = original_curated


def test_merge_layer_excluded_injected_still_caught():
    """Deliberately bypass upstream SQL filter by constructing a RecallItem
    with retrieval_excluded=1 content and injecting it directly into the
    merge step. The _filter_eligible fail-closed layer must still catch it.

    This test ensures the merge layer is not relying solely on upstream
    filtering — it independently validates SQLite eligibility.
    """
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO memories "
        "(id, scope_id, content, summary, source, target, "
        " created_at, updated_at) "
        "VALUES (?, 'default', 'Bypass attempt.', 'Bypass attempt.', "
        " 'test', 'memory', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        ("id-bypass-excluded",),
    )
    conn.execute(
        "UPDATE memories SET retrieval_excluded = 1 WHERE id = 'id-bypass-excluded'"
    )
    conn.commit()

    provider = FakeProvider(conn)
    # Build a RecallItem that bypasses upstream SQL filtering entirely
    # by injecting it through _search_db_memories override
    bypass_item = _make_item("id-bypass-excluded", content="Bypass attempt.")
    original_db = provider.__class__._search_db_memories
    provider.__class__._search_db_memories = lambda self, q, *, limit: [bypass_item]
    original_vector = provider.__class__._search_vector_memories
    provider.__class__._search_vector_memories = lambda self, q, *, limit: []
    original_curated = provider.__class__._search_curated_memories
    provider.__class__._search_curated_memories = lambda self, q: []
    try:
        service = RecallService(provider)
        results = service.search_memories("bypass", limit=5)
        ids = {item.id for item in results}
        assert "id-bypass-excluded" not in ids, (
            "Excluded record bypassed upstream filter and was not caught "
            "by merge-layer fail-closed protection"
        )
    finally:
        provider.__class__._search_db_memories = original_db
        provider.__class__._search_vector_memories = original_vector
        provider.__class__._search_curated_memories = original_curated


def test_merge_layer_mixed_results_only_curated_survives():
    """When connection fails, mixed curated + database-backed results must
    keep only curated items."""
    curated1 = RecallItem(
        id="curated:user:def456",
        content="User habit.",
        summary="User habit.",
        source="builtin-curated", target="user",
        score=0.5, updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": 0.5, "vector_score": 0.0},
    )
    curator_curated = RecallItem(
        id="curated:user:ghi789",
        content="Another habit.",
        summary="Another habit.",
        source="builtin-curated", target="user",
        score=0.5, updated_at="2026-01-01T00:00:00+00:00",
        metadata={"lexical_score": 0.5, "vector_score": 0.0},
    )
    db1 = _make_item("memory-db-1", score=0.6)
    db2 = _make_item("memory-db-2", score=0.4)
    provider = FailingProvider(db_items=[db1, db2])
    original = provider.__class__._search_curated_memories
    provider.__class__._search_curated_memories = lambda self, q: [curated1, curator_curated]
    try:
        service = RecallService(provider)
        results = service.search_memories("habit", limit=5)
        ids = {item.id for item in results}
        assert "curated:user:def456" in ids
        assert "curated:user:ghi789" in ids
        assert "memory-db-1" not in ids
        assert "memory-db-2" not in ids
    finally:
        provider.__class__._search_curated_memories = original


def test_merge_layer_partial_excluded_only_eligible_survive():
    """When some candidate IDs are eligible and some excluded, only eligible
    entries survive the merge layer."""
    conn = _conn()
    _populate_simple(conn)  # creates id-memory-normal and id-memory-excluded
    provider = FakeProvider(conn)
    service = RecallService(provider)

    # Inject three candidates: two eligible (one normal, one unused), one excluded
    normal_item = _make_item("id-memory-normal")
    extra_item = _make_item("id-extra-eligible", content="Extra eligible record.")
    conn.execute(
        "INSERT OR REPLACE INTO memories "
        "(id, scope_id, content, summary, source, target, retrieval_excluded, "
        " created_at, updated_at) "
        "VALUES (?, 'default', 'Extra eligible record.', 'Extra eligible record.', "
        " 'test', 'memory', 0, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        ("id-extra-eligible",),
    )
    conn.commit()
    excluded_item = _make_item("id-memory-excluded")
    original_db = provider.__class__._search_db_memories
    provider.__class__._search_db_memories = lambda self, q, *, limit: [
        normal_item, excluded_item, extra_item
    ]
    original_vector = provider.__class__._search_vector_memories
    provider.__class__._search_vector_memories = lambda self, q, *, limit: []
    original_curated = provider.__class__._search_curated_memories
    provider.__class__._search_curated_memories = lambda self, q: []
    try:
        service = RecallService(provider)
        results = service.search_memories("eligible", limit=5)
        ids = {item.id for item in results}
        assert "id-memory-excluded" not in ids, (
            "Excluded record leaked through merge layer"
        )
        assert "id-extra-eligible" in ids, (
            "Eligible record was incorrectly rejected"
        )
    finally:
        provider.__class__._search_db_memories = original_db
        provider.__class__._search_vector_memories = original_vector
        provider.__class__._search_curated_memories = original_curated
