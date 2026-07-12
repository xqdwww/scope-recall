from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prospective_governance import (
    ShadowSafetyError,
    generate_shadow_manifest,
    run_shadow,
    tool_log_evidence,
    validate_manifest,
)


NOW = datetime(2026, 7, 13, tzinfo=timezone.utc)


def make_db(path: Path, *, old: bool = False) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE memories(
        id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT, scope_id TEXT,
        session_id TEXT, created_at TEXT, supersedes TEXT, superseded_by TEXT,
        retrieval_excluded INTEGER DEFAULT 0,
        hit_count INTEGER DEFAULT 0, last_recalled_turn INTEGER DEFAULT 0,
        last_retrieved_at TEXT)"""
    )
    if not old:
        conn.executescript(
            """CREATE TABLE retrieval_events(
            event_id TEXT PRIMARY KEY,request_id TEXT,memory_id TEXT,memory_domain TEXT,
            stage TEXT,retrieval_path TEXT,rank INTEGER,occurred_at TEXT);
            CREATE TABLE retrieval_telemetry_config(key TEXT PRIMARY KEY,value TEXT);
            INSERT INTO retrieval_telemetry_config VALUES
            ('retrieval_telemetry_started_at','2026-07-12T00:00:00+00:00');"""
        )
    conn.commit()
    conn.close()


def ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def add(path: Path, memory_id: str, content: str, **values) -> None:
    conn = sqlite3.connect(path)
    cols = ["id", "content", *values]
    conn.execute(
        f"INSERT INTO memories({','.join(cols)}) VALUES({','.join('?' for _ in cols)})",
        [memory_id, content, *values.values()],
    )
    conn.commit()
    conn.close()


def manifest(path: Path, **kwargs):
    conn = ro(path)
    try:
        return generate_shadow_manifest(conn, as_of=NOW, **kwargs)
    finally:
        conn.close()


def types(result):
    return [c["candidate_type"] for c in result[0]["candidates"]]


def test_shadow_mode_rejects_production_and_ledger_writes(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    conn = ro(db)
    with pytest.raises(ShadowSafetyError):
        generate_shadow_manifest(conn, shadow_only=False)
    with pytest.raises(ShadowSafetyError):
        generate_shadow_manifest(conn, apply=True)
    conn.close()
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def test_exact_and_normalized_hash_duplicates(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    add(db, "a", "same text", created_at="2026-01-01T00:00:00+00:00")
    add(db, "b", "same text", created_at="2026-01-02T00:00:00+00:00")
    add(db, "c", "  NORMALIZED\n value ", created_at="2026-01-01T00:00:00+00:00")
    add(db, "d", "normalized value", created_at="2026-01-02T00:00:00+00:00")
    result = manifest(db)[0]
    duplicate = [c for c in result["candidates"] if c["candidate_type"] == "EXACT_DUPLICATE_CANDIDATE"]
    assert {c["memory_id"] for c in duplicate} == {"b", "d"}
    assert {c["deterministic_evidence"][0] for c in duplicate} == {"FULL_CONTENT_HASH", "NORMALIZED_CONTENT_HASH"}


@pytest.mark.parametrize(
    "content,marker",
    [
        ('Traceback (most recent call last):\n  File "x.py", line 3, in f\nValueError: bad', "PYTHON_TRACEBACK_WITH_FRAME"),
        ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y", "GIT_DIFF_WITH_HUNK"),
        ("================ test session starts ================\na.py .\n================ 1 passed ================", "PYTEST_STRUCTURED_OUTPUT"),
        ("PID PPID USER COMMAND\n12 1 me gateway\n13 1 me worker", "PROCESS_TABLE_BLOCK"),
        ('HTTP/1.1 500\n{"error":"provider error","request_id":"r"}', "HTTP_ERROR_RESPONSE"),
        ("browser automation dom snapshot\nrole=button locator=x\nnodeId=4 aria-label=x", "BROWSER_AUTOMATION_RAW_OUTPUT"),
    ],
)
def test_tool_log_high_precision_structures(content, marker):
    assert marker in tool_log_evidence(content)


@pytest.mark.parametrize("content", ["PASS", "pytest_output.txt", "git status", "PID", "error.log", "browser automation"])
def test_filename_or_keyword_alone_is_not_noise(content):
    assert tool_log_evidence(content) == []


def test_protected_content_and_closed_954_do_not_enter_archive_candidates(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    raw = 'Traceback (most recent call last):\n  File "x.py", line 3\nsecurity recovery evidence'
    add(db, "protected", raw)
    add(db, "closed", "same")
    add(db, "other", "same")
    result = manifest(db, closed_954={"closed"})[0]
    archive_types = {"TOOL_LOG_NOISE_CANDIDATE", "EXACT_DUPLICATE_CANDIDATE", "SUPERSEDED_STATE_CANDIDATE", "STALE_EPISODIC_REVIEW_CANDIDATE"}
    assert not [c for c in result["candidates"] if c["memory_id"] in {"protected", "closed"} and c["candidate_type"] in archive_types]
    assert result["counts"]["protected_excluded"] >= 1


def test_durable_candidate_has_strict_gate_and_curated_equivalence(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    add(db, "good", "Engineering principle: verify new automation with a real task before making it permanent.")
    add(db, "status", "Engineering principle: current HEAD abcdef1 and PID 123")
    add(db, "existing", "Stable preference: concise answers")
    result = manifest(db, curated_texts=["Stable preference: concise answers"])[0]
    durable = [c["memory_id"] for c in result["candidates"] if c["candidate_type"] == "DURABLE_REVIEW_CANDIDATE"]
    assert durable == ["good"]


def test_telemetry_under_30_days_never_says_unused(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    add(db, "a", "Engineering principle: validate changes across sessions.")
    result = manifest(db)[0]
    assert result["telemetry"]["observed_days"] < 30
    dumped = json.dumps(result)
    assert "unused" not in dumped.lower()
    assert "no observed injection since telemetry_started_at" in dumped


def test_stale_episdodic_uses_age_not_usage_as_reason(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE memories ADD COLUMN target TEXT")
    conn.commit()
    conn.close()
    add(db, "old", "A completed task observation.", target="general", created_at="2026-01-01T00:00:00+00:00")
    result = manifest(db)[0]
    candidate = next(c for c in result["candidates"] if c["memory_id"] == "old")
    assert candidate["candidate_type"] == "STALE_EPISODIC_REVIEW_CANDIDATE"
    assert candidate["candidate_reason"].startswith("Age and episodic content type")


def test_valid_supersession_requires_scope_project_time_and_no_cycle(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    scope = "platform:cli|workspace:demo|project:alpha"
    add(db, "old", "old state", scope_id=scope, created_at="2026-01-01T00:00:00+00:00", superseded_by="new")
    add(db, "new", "new state", scope_id=scope, created_at="2026-02-01T00:00:00+00:00", supersedes="old")
    assert "SUPERSEDED_STATE_CANDIDATE" in types(manifest(db))


def test_batch_caps_and_deterministic_aggregation(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    for i in range(105):
        add(db, f"a{i:03}", "identical")
    result, reconciliation = manifest(db, high_limit=100)
    assert result["counts"]["by_confidence"]["HIGH_CONFIDENCE"] == 100
    assert len(result["first_suggested_approval_batch"]) == 100
    validation = validate_manifest(result, reconciliation)
    assert validation["status"] == "PASS"


def test_reconciliation_detects_excluded_without_ledger(tmp_path):
    db = tmp_path / "db.sqlite"
    make_db(db)
    add(db, "x", "x", retrieval_excluded=1)
    _, reconciliation = manifest(db)
    assert any(a["type"] == "EXCLUDED_WITHOUT_APPLIED_EVENT" for a in reconciliation["anomalies"])


@pytest.mark.parametrize("old", [False, True])
def test_fresh_and_old_schema_compatibility_and_output_artifacts(tmp_path, old):
    db = tmp_path / "db.sqlite"
    make_db(db, old=old)
    add(db, "x", "ordinary memory")
    out = tmp_path / "out"
    run_shadow(database=db, output_dir=out, as_of=NOW)
    expected = {
        "WEEKLY_MEMORY_GOVERNANCE_SHADOW_MANIFEST.json",
        "WEEKLY_MEMORY_GOVERNANCE_SHADOW_REPORT.md",
        "WEEKLY_MEMORY_GOVERNANCE_SHADOW_VALIDATION.json",
        "WEEKLY_MEMORY_RECONCILIATION_REPORT.json",
    }
    assert {p.name for p in out.iterdir()} == expected
    assert json.loads((out / "WEEKLY_MEMORY_GOVERNANCE_SHADOW_VALIDATION.json").read_text())["status"] == "PASS"
