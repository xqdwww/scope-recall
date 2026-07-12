"""Tests for Retrieval Usage Telemetry V1.

Covers the retrieval_events schema, candidate/selected/injected event recording,
hit_count / last_recalled_turn / last_retrieved_at semantics, idempotency,
fail-safety, exclusion filtering, and health reporting (requirements 1-20)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any

import pytest

from scope_recall.sql_store import ensure_schema
from scope_recall.telemetry import (
    RETRIEVAL_TELEMETRY_MIGRATION_ID,
    _query_hash,
    ensure_retrieval_telemetry_schema,
    record_candidate_events,
    record_injected_events,
    record_selected_events,
    retrieval_telemetry_health,
    retrieval_telemetry_started_at,
)

# ---------------------------------------------------------------------------
# Fake provider for telemetry tests
# ---------------------------------------------------------------------------


class FakeTelemetryProvider:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._lock = __import__("threading").RLock()
        self._session_id = "test-session"
        self._current_turn = 42


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def _memory_row(conn: sqlite3.Connection, mid: str, content: str = "test content") -> None:
    """Insert a minimal memories row so injected events can update aggregate fields."""
    conn.execute(
        """INSERT OR IGNORE INTO memories
           (id, scope_id, platform, user_id, chat_id, thread_id, gateway_session_key,
            agent_identity, agent_workspace, session_id, source, target,
            content, summary, created_at, updated_at, last_recalled_turn, dedup_key, metadata)
           VALUES (?, 'local', 'test', 'test_u', 'dm', '', '', 'agent', '', 'sess1',
                   'test', 'ops', ?, ?, ?, ?, 0, ?, '{}')""",
        (
            mid,
            content,
            content[:80],
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00",
            hashlib.sha256(content.encode()).hexdigest(),
        ),
    )
    conn.commit()


# ===========================================================================
# 1. FTS candidate events
# ===========================================================================


def test_1_fts_candidate_events():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem1", "the quick brown fox")

    items = [{"id": "mem1", "score": 0.95}]
    n = record_candidate_events(
        prov, request_id="req1", query="fox", items=items, retrieval_path="fts"
    )

    assert n == 1
    row = conn.execute(
        "SELECT * FROM retrieval_events WHERE request_id='req1'"
    ).fetchone()
    assert row is not None
    assert row["stage"] == "candidate"
    assert row["retrieval_path"] == "fts"
    assert row["memory_id"] == "mem1"
    assert _query_hash("fox") == row["query_hash"]


# ===========================================================================
# 2. Vector candidate events
# ===========================================================================


def test_2_vector_candidate_events():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem2")

    n = record_candidate_events(
        prov, request_id="req2", query="vector query", items=[{"id": "mem2", "score": 0.88}],
        retrieval_path="vector"
    )
    assert n == 1
    row = conn.execute(
        "SELECT * FROM retrieval_events WHERE request_id='req2'"
    ).fetchone()
    assert row["retrieval_path"] == "vector"


# ===========================================================================
# 3. Merge → selected events
# ===========================================================================


def test_3_selected_events():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem3")

    n = record_selected_events(
        prov, request_id="req3", query="merge query",
        items=[{"id": "mem3", "score": 0.75, "rank": 1}]
    )
    assert n == 1
    row = conn.execute(
        "SELECT * FROM retrieval_events WHERE request_id='req3'"
    ).fetchone()
    assert row["stage"] == "selected"
    assert row["retrieval_path"] == "merge"


# ===========================================================================
# 4. Context injection events
# ===========================================================================


def test_4_injected_events():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem4")

    n = record_injected_events(
        prov, request_id="req4", query="inject query",
        items=[{"id": "mem4", "score": 0.9, "turn_id": "42"}]
    )
    assert n == 1
    row = conn.execute(
        "SELECT * FROM retrieval_events WHERE request_id='req4'"
    ).fetchone()
    assert row["stage"] == "injected"
    assert row["retrieval_path"] == "context_pack"


# ===========================================================================
# 5. Candidate events do NOT increment hit_count
# ===========================================================================


def test_5_candidate_does_not_increment_hit_count():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem5")

    assert conn.execute("SELECT hit_count FROM memories WHERE id='mem5'").fetchone()[0] == 0
    record_candidate_events(
        prov, request_id="r5", query="t", items=[{"id": "mem5"}],
        retrieval_path="fts"
    )
    assert conn.execute("SELECT hit_count FROM memories WHERE id='mem5'").fetchone()[0] == 0


# ===========================================================================
# 6. Selected events do NOT increment hit_count
# ===========================================================================


def test_6_selected_does_not_increment_hit_count():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem6")

    record_selected_events(
        prov, request_id="r6", query="t", items=[{"id": "mem6"}]
    )
    assert conn.execute("SELECT hit_count FROM memories WHERE id='mem6'").fetchone()[0] == 0


# ===========================================================================
# 7. Injected events increment hit_count exactly once
# ===========================================================================


def test_7_injected_increments_hit_count():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem7")

    record_injected_events(
        prov, request_id="r7", query="t",
        items=[{"id": "mem7", "score": 0.9, "turn_id": "42"}]
    )
    assert conn.execute("SELECT hit_count FROM memories WHERE id='mem7'").fetchone()[0] == 1


# ===========================================================================
# 8. Retry (idempotent) does not double-increment hit_count
# ===========================================================================


def test_8_retry_idempotent_hit_count():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem8")

    # Two identical calls
    for _ in range(2):
        record_injected_events(
            prov, request_id="r8", query="t",
            items=[{"id": "mem8", "score": 0.9, "turn_id": "42"}]
        )

    # INSERT OR IGNORE means only 1 event row and aggregate update is tied to it.
    cnt = conn.execute(
        "SELECT COUNT(*) FROM retrieval_events WHERE request_id='r8' AND stage='injected'"
    ).fetchone()[0]
    assert cnt == 1

    hc = conn.execute("SELECT hit_count FROM memories WHERE id='mem8'").fetchone()[0]
    assert hc == 1


# ===========================================================================
# 9. last_retrieved_at is updated
# ===========================================================================


def test_9_last_retrieved_at_updated():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem9")

    record_injected_events(
        prov, request_id="r9", query="t",
        items=[{"id": "mem9", "score": 0.9, "turn_id": "42"}]
    )
    lra = conn.execute(
        "SELECT last_retrieved_at FROM memories WHERE id='mem9'"
    ).fetchone()[0]
    assert lra is not None and len(str(lra)) > 10  # ISO timestamp


# ===========================================================================
# 10. last_recalled_turn is updated
# ===========================================================================


def test_10_last_recalled_turn_updated():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem10")

    record_injected_events(
        prov, request_id="r10", query="t",
        items=[{"id": "mem10", "score": 0.9, "turn_id": "99"}]
    )
    lrt = conn.execute(
        "SELECT last_recalled_turn FROM memories WHERE id='mem10'"
    ).fetchone()[0]
    assert int(lrt) >= 99


# ===========================================================================
# 11. Excluded memory → no selected/injected event
# ===========================================================================


def test_11_excluded_no_selected_or_injected():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)

    # Excluded memory
    conn.execute("INSERT INTO memories (id, scope_id, platform, user_id, source, target, content, summary, created_at, updated_at, retrieval_excluded, dedup_key, metadata) VALUES ('excl1', 'local', 'test', 'u', 'test', 'ops', 'x', 'x', '2026-01-01', '2026-01-01', 1, 'hash', '{}')")
    conn.commit()

    record_selected_events(
        prov, request_id="r11", query="t",
        items=[{"id": "excl1"}]
    )
    assert conn.execute("SELECT COUNT(*) FROM retrieval_events WHERE request_id='r11' AND stage='selected'").fetchone()[0] == 0

    record_injected_events(
        prov, request_id="r11", query="t",
        items=[{"id": "excl1", "score": 0.5, "turn_id": "1"}]
    )
    assert conn.execute("SELECT COUNT(*) FROM retrieval_events WHERE request_id='r11' AND stage='injected'").fetchone()[0] == 0
    assert conn.execute("SELECT hit_count FROM memories WHERE id='excl1'").fetchone()[0] == 0


# ===========================================================================
# 12. historical_only memory → no default injected event
# ===========================================================================


def test_12_historical_only_no_default_injected():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)

    _memory_row(conn, "hist1")
    conn.execute(
        "UPDATE memories SET metadata = ? WHERE id='hist1'",
        (json.dumps({"retrieval_policy": "historical_only"}),),
    )
    conn.commit()

    # candidate is fine
    record_candidate_events(
        prov, request_id="r12", query="t", items=[{"id": "hist1"}],
        retrieval_path="fts"
    )
    assert conn.execute("SELECT COUNT(*) FROM retrieval_events WHERE request_id='r12' AND stage='candidate'").fetchone()[0] == 1

    record_selected_events(
        prov, request_id="r12", query="t", items=[{"id": "hist1"}]
    )
    record_injected_events(
        prov, request_id="r12", query="t",
        items=[{"id": "hist1", "turn_id": "1"}]
    )
    assert conn.execute("SELECT COUNT(*) FROM retrieval_events WHERE request_id='r12' AND stage IN ('selected','injected')").fetchone()[0] == 0
    assert conn.execute("SELECT hit_count FROM memories WHERE id='hist1'").fetchone()[0] == 0


# ===========================================================================
# 13. Curated event recorded but does NOT update memories fields
# ===========================================================================


def test_13_curated_does_not_update_memories():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "curated:test:abc123")

    record_injected_events(
        prov, request_id="r13", query="t",
        items=[{"id": "curated:test:abc123", "score": 0.8, "turn_id": "1"}]
    )
    # Event is recorded
    assert conn.execute("SELECT COUNT(*) FROM retrieval_events WHERE request_id='r13' AND stage='injected'").fetchone()[0] == 1

    # memories table is NOT updated for curated IDs
    hc = conn.execute("SELECT hit_count FROM memories WHERE id='curated:test:abc123'").fetchone()
    assert hc is not None
    assert hc[0] == 0  # Not incremented for curated


# ===========================================================================
# 14. Telemetry failure does not block retrieval (returns 0, no crash)
# ===========================================================================


def test_14_telemetry_failure_is_safe():
    # No provider connection — should not crash
    class BareProvider:
        pass

    prov = BareProvider()
    n = record_candidate_events(
        prov, request_id="r14", query="safe",
        items=[{"id": "mem1"}], retrieval_path="fts"
    )
    assert n == 0  # Graceful failure


# ===========================================================================
# 15. Telemetry failure does not update hit_count
# ===========================================================================


def test_15_failure_no_hit_count_update():
    # Provide a fake connection that raises on write
    class BrokenConn:
        def execute(self, *a, **kw):
            raise sqlite3.OperationalError("simulated failure")
        def commit(self):
            pass
        def rollback(self):
            pass

    class BrokenProvider:
        _conn = BrokenConn()
        _lock = __import__("threading").RLock()
        _session_id = "broken"
        _current_turn = 1

    prov = BrokenProvider()
    n = record_injected_events(
        prov, request_id="r15", query="t",
        items=[{"id": "mem_fail", "score": 0.9, "turn_id": "1"}]
    )
    assert n == 0  # No events written


# ===========================================================================
# 16. Fresh schema creates tables
# ===========================================================================


def test_16_fresh_schema():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Only create telemetry schema
    first = ensure_retrieval_telemetry_schema(conn)
    assert first is True

    # Table exists
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_events'"
    ).fetchone()
    assert row is not None

    # Config table exists
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_telemetry_config'"
    ).fetchone()
    assert row is not None

    # telemetry_started_at is set
    assert retrieval_telemetry_started_at(conn) is not None


# ===========================================================================
# 17. Idempotent re-run of schema
# ===========================================================================


def test_17_idempotent_schema():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    first = ensure_retrieval_telemetry_schema(conn)
    second = ensure_retrieval_telemetry_schema(conn)
    assert first is True
    assert second is False  # Already existed


# ===========================================================================
# 18. Append-only: no UPDATE/DELETE of events through normal paths
# ===========================================================================


def test_18_append_only_event_contract():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem18")

    # Insert an event
    record_candidate_events(
        prov, request_id="r18", query="t",
        items=[{"id": "mem18"}], retrieval_path="fts"
    )

    # Verify no function in telemetry module modifies events
    # The only write path is batch_insert_events (INSERT OR IGNORE)
    # Verify by checking the module source
    import scope_recall.telemetry as tm
    source = tm.__file__
    with open(source) as f:
        body = f.read()
    # No UPDATE or DELETE on retrieval_events
    assert "UPDATE retrieval_events" not in body
    assert "DELETE FROM retrieval_events" not in body


# ===========================================================================
# 19. Query body does not enter event table
# ===========================================================================


def test_19_query_body_not_in_event_table():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)
    _memory_row(conn, "mem19")

    record_candidate_events(
        prov, request_id="r19", query="my secret api key=sk-abcd1234",
        items=[{"id": "mem19"}], retrieval_path="fts"
    )

    row = conn.execute(
        "SELECT * FROM retrieval_events WHERE request_id='r19'"
    ).fetchone()
    assert row is not None

    # The full query must NOT appear in any column
    all_text = str(dict(row))
    assert "secret api key" not in all_text
    assert "sk-abcd1234" not in all_text

    # Only the hash is stored
    assert row["query_hash"] is not None
    assert len(row["query_hash"]) == 64  # SHA-256 hex


# ===========================================================================
# 20. Health report returns expected fields
# ===========================================================================


def test_20_health_report():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    ensure_retrieval_telemetry_schema(conn)

    # Add some data
    _memory_row(conn, "mem_a")
    _memory_row(conn, "mem_b")
    record_candidate_events(prov, request_id="ha", query="health", items=[{"id": "mem_a"}, {"id": "mem_b"}], retrieval_path="fts")
    record_selected_events(prov, request_id="ha", query="health", items=[{"id": "mem_a"}])
    record_injected_events(prov, request_id="ha", query="health", items=[{"id": "mem_a", "score": 0.9, "turn_id": "1"}])

    report = retrieval_telemetry_health(prov)

    assert "telemetry_started_at" in report
    assert report["total_events"] >= 4
    assert report["candidate_events"] == 2
    assert report["selected_events"] == 1
    assert report["injected_events"] == 1
    assert report["database_memories_ever_injected_since_start"] >= 1
    assert report["records_with_hit_count_gt_0"] >= 1
    assert report["records_with_last_retrieved_at"] >= 1


def test_21_candidate_paths_have_distinct_idempotency_keys():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    _memory_row(conn, "mem21")
    for path in ("fts", "like", "alias"):
        record_candidate_events(
            prov, request_id="r21", query="same", items=[{"id": "mem21"}],
            retrieval_path=path,
        )
    rows = conn.execute(
        "SELECT retrieval_path FROM retrieval_events WHERE request_id='r21' ORDER BY retrieval_path"
    ).fetchall()
    assert [row[0] for row in rows] == ["alias", "fts", "like"]


def test_22_injected_event_and_aggregate_are_atomic_on_failure():
    conn = _conn()
    prov = FakeTelemetryProvider(conn)
    _memory_row(conn, "mem22")
    conn.execute(
        """CREATE TRIGGER fail_mem22_update BEFORE UPDATE OF hit_count ON memories
           WHEN OLD.id='mem22' BEGIN SELECT RAISE(ABORT, 'aggregate failure'); END"""
    )
    conn.commit()
    assert record_injected_events(
        prov, request_id="r22", query="atomic", items=[{"id": "mem22", "turn_id": "2"}]
    ) == 0
    assert conn.execute("SELECT count(*) FROM retrieval_events WHERE request_id='r22'").fetchone()[0] == 0
    assert conn.execute("SELECT hit_count FROM memories WHERE id='mem22'").fetchone()[0] == 0
    health = retrieval_telemetry_health(prov)
    assert health["telemetry_write_failures"] == 1
    assert health["coverage_complete"] is False


def test_23_old_schema_migration_is_idempotent_and_preserves_rows():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE memories(id TEXT PRIMARY KEY, content TEXT NOT NULL)")
    conn.execute("INSERT INTO memories VALUES('old1', 'unchanged body')")
    assert ensure_retrieval_telemetry_schema(conn) is True
    assert ensure_retrieval_telemetry_schema(conn) is False
    assert conn.execute("SELECT content FROM memories WHERE id='old1'").fetchone()[0] == "unchanged body"
    assert conn.execute("SELECT count(*) FROM retrieval_events").fetchone()[0] == 0
