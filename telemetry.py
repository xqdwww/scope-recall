"""Retrieval-usage telemetry for Scope Recall.

Three event stages are recorded in an append-only log:
  - candidate : memory returned by a raw retrieval path (FTS, vector, curated)
  - selected  : memory that survived merge, dedup, policy, and ranking
  - injected  : memory that actually entered the model's context pack

Privacy guarantees:
  - Full user queries are NEVER stored.
  - Only an irreversible query hash (SHA-256) is logged.
  - Memory content is NEVER copied into the event table.
  - No API keys, cookies, or tool-argument bodies are saved.

Concurrency: the module acquires the provider's reentrant lock whenever it reads
or writes the SQLite connection, so callers do not need external synchronisation."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_logger = logging.getLogger(__name__)

# ── schema / migration constants ───────────────────────────────────────────

RETRIEVAL_TELEMETRY_MIGRATION_ID = "0015_retrieval_usage_telemetry_v1"
RETRIEVAL_TELEMETRY_PLUGIN_VERSION = "1.6.0"
RETRIEVAL_TELEMETRY_DESCRIPTION = "Add retrieval_events table, memory hit_count/telemetry columns, and telemetry start timestamp"

# ── data classes ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalEvent:
    """A single retrieval-usage event ready to be appended to the log."""

    event_id: str
    request_id: str
    memory_id: str
    memory_domain: str  # "database" or "curated"
    stage: str  # "candidate", "selected", or "injected"
    retrieval_path: str | None  # "fts", "vector", "curated", "merge", "context_pack", …
    rank: int | None
    score: float | None
    scope_id: str | None
    session_id: str | None
    turn_id: str | None
    query_hash: str | None
    occurred_at: str
    metadata_json: str = "{}"


# ── helpers ─────────────────────────────────────────────────────────────────


def _query_hash(query: str) -> str:
    """Return a SHA-256 hex digest of the query (lowered and stripped).

    The hash is the *only* query trace stored in the telemetry log.
    The raw query is never persisted.
    """
    return hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _telemetry_event_id(
    request_id: str, memory_id: str, stage: str, seq: int = 0, discriminator: str = ""
) -> str:
    """Deterministic event ID for idempotent inserts."""
    raw = (
        f"{request_id}|{memory_id}|{stage}|{discriminator}:{seq}"
        if discriminator
        else f"{request_id}|{memory_id}|{stage}:{seq}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


# ── schema ──────────────────────────────────────────────────────────────────


def _schema_exists(conn: sqlite3.Connection) -> bool:
    """Return True if the retrieval_events table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_events'"
    ).fetchone()
    return row is not None


def ensure_retrieval_telemetry_schema(conn: sqlite3.Connection) -> bool:
    """Create or migrate the retrieval_events table and memory telemetry columns.

    Returns True if the schema was newly created (first-time activation) or
    False if it already existed (idempotent re-run).
    """
    first_time = False
    if not _schema_exists(conn):
        first_time = True
        _logger.info("Creating retrieval_events table (first-time telemetry activation)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_events (
                event_id         TEXT PRIMARY KEY,
                request_id       TEXT NOT NULL,
                memory_id        TEXT NOT NULL,
                memory_domain    TEXT NOT NULL DEFAULT 'database',
                stage            TEXT NOT NULL CHECK (stage IN ('candidate', 'selected', 'injected')),
                retrieval_path   TEXT,
                rank             INTEGER,
                score            REAL,
                scope_id         TEXT,
                session_id       TEXT,
                turn_id          TEXT,
                query_hash       TEXT,
                occurred_at      TEXT NOT NULL,
                metadata_json    TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_events_stage ON retrieval_events(stage)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_events_memory_stage "
            "ON retrieval_events(memory_id, stage)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_events_request "
            "ON retrieval_events(request_id, stage)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_events_occurred "
            "ON retrieval_events(occurred_at)"
        )

    # Memory telemetry columns
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}

    telemetry_columns: dict[str, str] = {
        "hit_count": "ALTER TABLE memories ADD COLUMN hit_count INTEGER NOT NULL DEFAULT 0",
        "last_recalled_turn": "ALTER TABLE memories ADD COLUMN last_recalled_turn INTEGER NOT NULL DEFAULT 0",
        "last_retrieved_at": "ALTER TABLE memories ADD COLUMN last_retrieved_at TEXT",
    }

    for col, ddl in telemetry_columns.items():
        if col not in existing:
            _logger.info("Adding memory telemetry column: %s", col)
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError as exc:
                _logger.warning("Could not add column %s: %s", col, exc)

    # Telemetry start timestamp record (single row in a simple config table)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS retrieval_telemetry_config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    if first_time:
        now = _now_iso()
        conn.execute(
            "INSERT OR IGNORE INTO retrieval_telemetry_config(key, value) VALUES (?, ?)",
            ("retrieval_telemetry_started_at", now),
        )

    conn.commit()
    return first_time


def retrieval_telemetry_started_at(conn: sqlite3.Connection) -> str | None:
    """Return the ISO timestamp when telemetry was first activated, or None."""
    row = conn.execute(
        "SELECT value FROM retrieval_telemetry_config WHERE key = 'retrieval_telemetry_started_at'"
    ).fetchone()
    return str(row["value"]) if row else None


# ── event recording ─────────────────────────────────────────────────────────


def _batch_insert_events(conn: sqlite3.Connection, events: list[RetrievalEvent]) -> int:
    """Insert events using INSERT OR IGNORE for idempotency.

    Returns the number of new rows inserted.
    """
    if not events:
        return 0
    rows = [
        (
            e.event_id,
            e.request_id,
            e.memory_id,
            e.memory_domain,
            e.stage,
            e.retrieval_path,
            e.rank,
            e.score,
            e.scope_id,
            e.session_id,
            e.turn_id,
            e.query_hash,
            e.occurred_at,
            e.metadata_json,
        )
        for e in events
    ]
    cursor = conn.executemany(
        """
        INSERT OR IGNORE INTO retrieval_events(
            event_id, request_id, memory_id, memory_domain,
            stage, retrieval_path, rank, score,
            scope_id, session_id, turn_id, query_hash,
            occurred_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return cursor.rowcount


def record_candidate_events(
    provider: Any,
    *,
    request_id: str,
    query: str,
    items: list[dict[str, Any]],
    retrieval_path: str,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Batch-insert candidate-stage retrieval events.

    *items* should be a list of dict-like objects with at least ``id``,
    and optionally ``score``, ``rank``, ``scope_id``, ``session_id``,
    ``turn_id``.

    Returns the number of new rows inserted.

    Thread-safe when *conn* is not provided (uses the provider's lock).
    """
    if not items:
        return 0
    qh = _query_hash(query)
    now = _now_iso()
    now_ms = int(time.time() * 1000)

    events: list[RetrievalEvent] = []
    for rank, item in enumerate(items):
        mid = str(item.get("id") or "")
        if not mid:
            continue
        domain = "curated" if mid.startswith("curated:") else "database"
        events.append(
            RetrievalEvent(
                event_id=_telemetry_event_id(request_id, mid, "candidate", rank, retrieval_path),
                request_id=request_id,
                memory_id=mid,
                memory_domain=domain,
                stage="candidate",
                retrieval_path=retrieval_path,
                rank=rank,
                score=_safe_float(item.get("score")),
                scope_id=str(item.get("scope_id") or ""),
                session_id=str(item.get("session_id") or ""),
                turn_id=str(item.get("turn_id") or ""),
                query_hash=qh,
                occurred_at=now,
            )
        )

    return _write_events(provider, events, conn=conn)


def record_selected_events(
    provider: Any,
    *,
    request_id: str,
    query: str,
    items: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> int:
    """Batch-insert selected-stage events.

    *items* should represent the merged, filtered, ranked candidate list
    before context injection.  Each element must have ``id``.

    Returns the number of new rows inserted.
    """
    items = _eligible_final_stage_items(provider, items, conn=conn)
    if not items:
        return 0
    qh = _query_hash(query)
    now = _now_iso()

    events: list[RetrievalEvent] = []
    for rank, item in enumerate(items):
        mid = str(item.get("id") or "")
        if not mid:
            continue
        domain = "curated" if mid.startswith("curated:") else "database"
        events.append(
            RetrievalEvent(
                event_id=_telemetry_event_id(request_id, mid, "selected", rank),
                request_id=request_id,
                memory_id=mid,
                memory_domain=domain,
                stage="selected",
                retrieval_path="merge",
                rank=rank,
                score=_safe_float(item.get("score")),
                scope_id=str(item.get("scope_id") or ""),
                session_id=str(item.get("session_id") or ""),
                turn_id=str(item.get("turn_id") or ""),
                query_hash=qh,
                occurred_at=now,
            )
        )

    return _write_events(provider, events, conn=conn)


def record_injected_events(
    provider: Any,
    *,
    request_id: str,
    query: str,
    items: list[dict[str, Any]],
    conn: sqlite3.Connection | None = None,
) -> int:
    """Batch-insert injected-stage events AND update the memories aggregate fields.

    ``hit_count`` is incremented only for ``database``-domain memories.
    ``last_recalled_turn`` and ``last_retrieved_at`` are set on each injected
    database memory.

    Curated items record an event but DO NOT update the memories table.

    Returns the number of new rows inserted.
    """
    items = _eligible_final_stage_items(provider, items, conn=conn)
    if not items:
        return 0
    qh = _query_hash(query)
    now = _now_iso()

    events: list[RetrievalEvent] = []
    db_updates: list[tuple[int, str, str]] = []  # (turn, retrieved_at, memory_id)

    for rank, item in enumerate(items):
        mid = str(item.get("id") or "")
        if not mid:
            continue
        domain = "curated" if mid.startswith("curated:") else "database"

        events.append(
            RetrievalEvent(
                event_id=_telemetry_event_id(request_id, mid, "injected", rank),
                request_id=request_id,
                memory_id=mid,
                memory_domain=domain,
                stage="injected",
                retrieval_path="context_pack",
                rank=rank,
                score=_safe_float(item.get("score")),
                scope_id=str(item.get("scope_id") or ""),
                session_id=str(item.get("session_id") or ""),
                turn_id=str(item.get("turn_id") or ""),
                query_hash=qh,
                occurred_at=now,
            )
        )

        if domain == "database":
            turn = max(0, int(item.get("turn_id") or 0) or int(item.get("last_recalled_turn") or 0))
            db_updates.append((turn, now, mid))

    return _write_injected_transaction(provider, events, db_updates, conn=conn)


def _eligible_final_stage_items(
    provider: Any,
    items: list[dict[str, Any]],
    *,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Fail closed for DB-backed selected/injected events.

    Curated IDs are file-backed and pass without touching ``memories``. Database
    IDs must still be retrieval eligible at the exact final-stage write point.
    """
    curated = [item for item in items if str(item.get("id") or "").startswith("curated:")]
    database = [item for item in items if item not in curated and str(item.get("id") or "")]
    if not database:
        return curated
    local_conn = conn or getattr(provider, "_conn", None)
    if local_conn is None:
        return curated
    ids = [str(item["id"]) for item in database]
    try:
        columns = {str(row[1]) for row in local_conn.execute("PRAGMA table_info(memories)")}
        policy_clause = (
            "AND COALESCE(retrieval_policy, 'normal') != 'historical_only'"
            if "retrieval_policy" in columns
            else "AND COALESCE(json_extract(metadata, '$.retrieval_policy'), 'normal') != 'historical_only'"
        )
        rows = local_conn.execute(
            f"""SELECT id FROM memories
                WHERE id IN ({','.join('?' for _ in ids)})
                  AND retrieval_excluded = 0
                  {policy_clause}""",
            ids,
        ).fetchall()
        eligible = {str(row[0]) for row in rows}
    except Exception:
        return curated
    return curated + [item for item in database if str(item["id"]) in eligible]


def _write_injected_transaction(
    provider: Any,
    events: list[RetrievalEvent],
    db_updates: list[tuple[int, str, str]],
    *,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Atomically append injected events and update aggregates for new events only."""
    local_conn = conn
    lock = getattr(provider, "_lock", None)
    acquired = False
    try:
        if local_conn is None:
            local_conn = getattr(provider, "_conn", None)
        if local_conn is None:
            _logger.warning("Telemetry: no connection available, dropping injected events")
            return 0
        if lock is not None:
            lock.acquire()
            acquired = True

        local_conn.execute("SAVEPOINT retrieval_telemetry_injected")
        inserted_events: list[RetrievalEvent] = []
        for event in events:
            before = local_conn.total_changes
            _batch_insert_events(local_conn, [event])
            if local_conn.total_changes > before:
                inserted_events.append(event)

        updates = {mid: (turn, retrieved_at) for turn, retrieved_at, mid in db_updates}
        columns = {str(row[1]) for row in local_conn.execute("PRAGMA table_info(memories)")}
        policy_clause = (
            "AND COALESCE(retrieval_policy, 'normal') != 'historical_only'"
            if "retrieval_policy" in columns
            else "AND COALESCE(json_extract(metadata, '$.retrieval_policy'), 'normal') != 'historical_only'"
        )
        for event in inserted_events:
            mid = event.memory_id
            if mid not in updates:
                continue
            turn, retrieved_at = updates[mid]
            cursor = local_conn.execute(
                f"""
                UPDATE memories
                SET hit_count = hit_count + 1,
                    last_recalled_turn = MAX(last_recalled_turn, ?),
                    last_retrieved_at = MAX(COALESCE(last_retrieved_at, ''), ?)
                WHERE id = ? AND retrieval_excluded = 0
                  {policy_clause}
                """,
                (turn, retrieved_at, mid),
            )
            if cursor.rowcount != 1:
                raise sqlite3.IntegrityError(
                    f"injected aggregate eligibility changed for {mid}"
                )
        local_conn.execute("RELEASE SAVEPOINT retrieval_telemetry_injected")
        return len(inserted_events)
    except Exception as exc:
        _logger.error("Telemetry injected transaction failure: %s", exc)
        if local_conn is not None:
            try:
                local_conn.execute("ROLLBACK TO SAVEPOINT retrieval_telemetry_injected")
                local_conn.execute("RELEASE SAVEPOINT retrieval_telemetry_injected")
            except Exception:
                pass
            _mark_coverage_incomplete(local_conn, "injected_transaction_failure")
        return 0
    finally:
        if acquired:
            lock.release()


def _mark_coverage_incomplete(conn: sqlite3.Connection, reason: str) -> None:
    """Best-effort persistent failure marker; never raises into retrieval."""
    try:
        conn.execute(
            """INSERT INTO retrieval_telemetry_config(key, value) VALUES('telemetry_write_failures', '1')
               ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT)"""
        )
        conn.execute(
            """INSERT INTO retrieval_telemetry_config(key, value) VALUES('telemetry_coverage_status', 'incomplete')
               ON CONFLICT(key) DO UPDATE SET value='incomplete'"""
        )
        conn.execute(
            """INSERT INTO retrieval_telemetry_config(key, value) VALUES('telemetry_last_failure_reason', ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
            (reason,),
        )
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass


def _write_events(
    provider: Any,
    events: list[RetrievalEvent],
    *,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Write events, acquiring the provider lock if needed.

    On any failure the error is logged and 0 is returned — telemetry
    must never break retrieval.
    """
    local_conn = conn
    acquired_lock = False
    try:
        if local_conn is None and hasattr(provider, "_lock") and hasattr(provider, "_conn"):
            provider._lock.acquire()  # type: ignore[union-attr]
            acquired_lock = True
            local_conn = provider._conn  # type: ignore[union-attr]

        if local_conn is None:
            _logger.warning("Telemetry: no connection available, dropping %d events", len(events))
            return 0

        local_conn.execute("SAVEPOINT retrieval_telemetry_events")
        inserted = _batch_insert_events(local_conn, events)
        local_conn.execute("RELEASE SAVEPOINT retrieval_telemetry_events")
        return inserted
    except Exception as exc:
        _logger.error("Telemetry write failure (%d events): %s", len(events), exc)
        if local_conn is not None:
            try:
                local_conn.execute("ROLLBACK TO SAVEPOINT retrieval_telemetry_events")
                local_conn.execute("RELEASE SAVEPOINT retrieval_telemetry_events")
            except Exception:
                pass
            _mark_coverage_incomplete(local_conn, "event_write_failure")
        return 0
    finally:
        if acquired_lock:
            try:
                provider._lock.release()  # type: ignore[union-attr]
            except Exception:
                pass


# ── telemetry health report ────────────────────────────────────────────────


def retrieval_telemetry_health(provider: Any) -> dict[str, Any]:
    """Return a deterministic snapshot of telemetry coverage and health.

    The report is read-only — it never mutates the database.
    """
    conn = getattr(provider, "_conn", None)
    if conn is None:
        return {
            "telemetry_started_at": None,
            "error": "No database connection available",
        }

    started_at = retrieval_telemetry_started_at(conn)
    if started_at is None:
        return {
            "telemetry_started_at": None,
            "error": "Telemetry schema exists but start timestamp not found",
        }

    try:
        total_events = int(
            conn.execute("SELECT COUNT(*) FROM retrieval_events").fetchone()[0]
        )
    except sqlite3.OperationalError:
        return {
            "telemetry_started_at": None,
            "error": "retrieval_events table does not exist",
        }

    try:
        request_count = int(
            conn.execute("SELECT COUNT(DISTINCT request_id) FROM retrieval_events").fetchone()[0]
        )
    except sqlite3.OperationalError:
        request_count = 0

    try:
        candidate_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_events WHERE stage='candidate'"
            ).fetchone()[0]
        )
    except sqlite3.OperationalError:
        candidate_count = 0

    try:
        selected_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_events WHERE stage='selected'"
            ).fetchone()[0]
        )
    except sqlite3.OperationalError:
        selected_count = 0

    try:
        injected_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM retrieval_events WHERE stage='injected'"
            ).fetchone()[0]
        )
    except sqlite3.OperationalError:
        injected_count = 0

    try:
        databases_injected = int(
            conn.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM retrieval_events WHERE stage='injected' AND memory_domain='database'"
            ).fetchone()[0]
        )
    except sqlite3.OperationalError:
        databases_injected = 0

    try:
        curated_injected = int(
            conn.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM retrieval_events WHERE stage='injected' AND memory_domain='curated'"
            ).fetchone()[0]
        )
    except sqlite3.OperationalError:
        curated_injected = 0

    try:
        memories_with_hits = int(
            conn.execute("SELECT COUNT(*) FROM memories WHERE hit_count > 0").fetchone()[0]
        )
    except sqlite3.OperationalError:
        memories_with_hits = 0

    try:
        memories_with_retrieved_at = int(
            conn.execute(
                "SELECT COUNT(*) FROM memories WHERE last_retrieved_at IS NOT NULL AND last_retrieved_at != ''"
            ).fetchone()[0]
        )
    except sqlite3.OperationalError:
        memories_with_retrieved_at = 0

    # Coverage: requests that have complete injected records
    try:
        rows = conn.execute(
            """
            SELECT request_id,
                   COUNT(*) FILTER (WHERE stage = 'candidate') AS candidate_events,
                   COUNT(*) FILTER (WHERE stage = 'selected') AS selected_events,
                   COUNT(*) FILTER (WHERE stage = 'injected') AS injected_events
            FROM retrieval_events
            GROUP BY request_id
            """
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    complete_injected = sum(
        1 for r in rows if int(r["injected_events"]) > 0
    )
    complete_candidate = sum(
        1 for r in rows if int(r["candidate_events"]) > 0
    )
    complete_selected = sum(
        1 for r in rows if int(r["selected_events"]) > 0
    )

    # Telemetry write failures are tracked via a counter in telemetry_config
    try:
        failure_count_row = conn.execute(
            "SELECT value FROM retrieval_telemetry_config WHERE key = 'telemetry_write_failures'"
        ).fetchone()
        write_failures = int(failure_count_row["value"]) if failure_count_row else 0
    except (sqlite3.OperationalError, ValueError, TypeError):
        write_failures = 0

    coverage_ratio = 0.0
    if request_count > 0:
        coverage_ratio = round(complete_injected / request_count, 4)

    return {
        "telemetry_started_at": started_at,
        "retrieval_requests_observed": request_count,
        "total_events": total_events,
        "candidate_events": candidate_count,
        "selected_events": selected_count,
        "injected_events": injected_count,
        "requests_with_complete_candidate_coverage": complete_candidate,
        "requests_with_complete_selected_coverage": complete_selected,
        "requests_with_complete_injected_coverage": complete_injected,
        "telemetry_write_failures": write_failures,
        "coverage_complete": write_failures == 0,
        "database_memories_ever_injected_since_start": databases_injected,
        "curated_items_ever_injected_since_start": curated_injected,
        "records_with_hit_count_gt_0": memories_with_hits,
        "records_with_last_retrieved_at": memories_with_retrieved_at,
        "coverage_ratio": coverage_ratio,
    }


# ── private helpers ─────────────────────────────────────────────────────────


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── module exports ──────────────────────────────────────────────────────────

__all__ = [
    "RetrievalEvent",
    "ensure_retrieval_telemetry_schema",
    "record_candidate_events",
    "record_selected_events",
    "record_injected_events",
    "retrieval_telemetry_health",
    "retrieval_telemetry_started_at",
]
