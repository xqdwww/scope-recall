"""Deterministic, read-only prospective memory governance shadow generator.

This module deliberately has no apply path.  It reads SQLite through query-only
connections and emits review candidates; candidates are never final actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VERSION = "prospective-memory-governance-v1"
CANDIDATE_TYPES = {
    "TOOL_LOG_NOISE_CANDIDATE",
    "EXACT_DUPLICATE_CANDIDATE",
    "SUPERSEDED_STATE_CANDIDATE",
    "STALE_EPISODIC_REVIEW_CANDIDATE",
    "DURABLE_REVIEW_CANDIDATE",
    "PRODUCTION_LEDGER_INCONSISTENCY",
    "TELEMETRY_COVERAGE_ANOMALY",
}
CONFIDENCES = ("HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "OBSERVE_ONLY")
USAGE_BOUNDARY_TEXT = "no observed injection since telemetry_started_at"


class ShadowSafetyError(RuntimeError):
    """Raised whenever a caller requests anything except shadow-only reads."""


def normalize_content(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _open_readonly(path: Path, *, immutable: bool = False) -> sqlite3.Connection:
    suffix = "?mode=ro&immutable=1" if immutable else "?mode=ro"
    conn = sqlite3.connect(f"file:{path}{suffix}", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        conn.close()
        raise ShadowSafetyError(f"query_only could not be enabled: {path}")
    return conn


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _field(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _memory_project(row: dict[str, Any]) -> str:
    explicit = str(_field(row, "project_id", "project", default="") or "").strip()
    if explicit:
        return explicit
    scope = str(_field(row, "scope_id", "scope", default="") or "")
    match = re.search(r"(?:project|workspace):[^|]*", scope, re.I)
    return match.group(0).casefold() if match else ""


def protected_reasons(row: dict[str, Any], closed_954: set[str]) -> list[str]:
    memory_id = str(_field(row, "id", "memory_id", default=""))
    text = normalize_content(str(_field(row, "content", default="")))
    reasons: list[str] = []
    if memory_id in closed_954:
        reasons.append("CLOSED_954_HISTORICAL_GOVERNANCE_EVIDENCE")
    rules = (
        (r"\b(prefer|preference|always wants?|user likes?|用户偏好|习惯)\b", "STABLE_USER_PREFERENCE"),
        (r"\b(health|medical|family|illness|健康|医疗|家人|家庭)\b", "HEALTH_OR_FAMILY_SENSITIVE"),
        (r"\b(governance approval|approved by|approval_id|治理批准|审批通过)\b", "GOVERNANCE_APPROVAL"),
        (r"(?:\bsecurity\b|\bcredential(?:s)?\b|recovery evidence|安全|恢复证据|密钥|凭证|泄漏)", "SECURITY_OR_RECOVERY_EVIDENCE"),
        (r"\b(architecture|system design|架构|系统设计)\b", "CURRENT_PROJECT_ARCHITECTURE"),
        (r"\b(rollback|disaster recovery|backup restore|回滚|灾难恢复|备份恢复)\b", "ROLLBACK_OR_DISASTER_RECOVERY"),
        (r"\b(durable promotion|promote_to_memory|promote_to_user|持久化晋升)\b", "DURABLE_PROMOTION_LINEAGE"),
    )
    for pattern, reason in rules:
        if re.search(pattern, text, re.I):
            reasons.append(reason)
    if str(_field(row, "retrieval_policy", default="")).casefold() == "historical_only" and _field(row, "superseded_by", default=""):
        reasons.append("COMPLETED_SUPERSESSION_LINEAGE")
    return sorted(set(reasons))


def tool_log_evidence(content: str) -> list[str]:
    """Return structural evidence only; single words and filenames never match."""
    text = content.strip()
    lines = text.splitlines()
    evidence: list[str] = []
    if re.search(r"Traceback \(most recent call last\):", text) and len(re.findall(r'^\s*File "[^"]+", line \d+', text, re.M)) >= 1:
        evidence.append("PYTHON_TRACEBACK_WITH_FRAME")
    if re.search(r"^diff --git a/.+ b/.+$", text, re.M) and re.search(r"^@@ .+ @@", text, re.M):
        evidence.append("GIT_DIFF_WITH_HUNK")
    if len(re.findall(r"^[ MADRCU?!]{2}\s+\S+", text, re.M)) >= 3:
        evidence.append("GIT_STATUS_PORCELAIN_BLOCK")
    if len(re.findall(r"^commit [0-9a-f]{7,40}$", text, re.M)) >= 2:
        evidence.append("GIT_LOG_BLOCK")
    if re.search(r"=+ (?:test session starts|short test summary info) =+", text) and re.search(r"\b\d+ (?:passed|failed|skipped)\b", text):
        evidence.append("PYTEST_STRUCTURED_OUTPUT")
    if re.search(r"\bPID\s+(?:PPID\s+)?(?:USER\s+)?(?:COMMAND|CMD)\b", text, re.I) and len(lines) >= 3:
        evidence.append("PROCESS_TABLE_BLOCK")
    if re.search(r"HTTP/\d(?:\.\d)?\s+[45]\d\d", text) and re.search(r"(?:error|message|provider|request[_ -]?id)", text, re.I):
        evidence.append("HTTP_ERROR_RESPONSE")
    if re.search(r"(?:playwright|browser automation|dom snapshot)", text, re.I) and len(re.findall(r"(?:locator|role=|nodeId|aria-|\[\d+\])", text, re.I)) >= 2:
        evidence.append("BROWSER_AUTOMATION_RAW_OUTPUT")
    status_headers = len(re.findall(r"^(?:status|health|service|process|pid|port|head|queue)\s*[:=]", text, re.I | re.M))
    if status_headers >= 4 and len(lines) >= 6:
        evidence.append("REPEATED_MACHINE_STATUS_BLOCK")
    shell_prompts = len(re.findall(r"^(?:\$|%|>>>|\w+@[^ ]+\s+[^#$%]*[$%])\s*\S+", text, re.M))
    output_markers = len(re.findall(r"^(?:exit code|stdout|stderr|return code)\s*[:=]", text, re.I | re.M))
    if shell_prompts >= 1 and output_markers >= 1 and len(lines) >= 4:
        evidence.append("SHELL_COMMAND_AND_OUTPUT_BLOCK")
    return sorted(set(evidence))


def _stable_durable_evidence(content: str) -> list[str]:
    text = normalize_content(content)
    evidence = []
    if re.search(r"(?:nothing to save|no new (?:user )?preference|没有新的用户偏好|无需(?:长期)?保留|没什么需要长期保留)", text):
        return []
    if re.search(r"(?:user always (?:prefers|wants)|stable (?:user )?preference|用户(?:稳定|一贯)偏好|用户总是)", text):
        evidence.append("EXPLICIT_STABLE_USER_PREFERENCE")
    if re.search(r"(?:engineering principle|workflow principle|as a general rule|工程原则|流程原则|一贯原则)", text):
        evidence.append("EXPLICIT_CROSS_SESSION_PRINCIPLE")
    transient = re.search(r"(?:\bHEAD\b|\bPID\b|commit [0-9a-f]{7,40}|/Users/|\\Users\\|model[- ]?v?\d|当前进度|本次任务|temporary|临时路径)", content, re.I)
    if transient:
        return []
    return evidence


def _curated_equivalent(content: str, curated: Iterable[str]) -> bool:
    normalized = normalize_content(content)
    if not normalized:
        return True
    tokens = set(normalized.split())
    for item in curated:
        other = normalize_content(item)
        if normalized == other or normalized in other or other in normalized:
            return True
        other_tokens = set(other.split())
        if tokens and other_tokens and len(tokens & other_tokens) / len(tokens | other_tokens) >= 0.92:
            return True
    return False


@dataclass(frozen=True)
class TelemetryWindow:
    started_at: str | None
    observed_days: float
    complete_requests: int
    total_requests: int
    injected_events: int
    failure_count: int
    failure_rate: float


def telemetry_window(conn: sqlite3.Connection, as_of: datetime) -> TelemetryWindow:
    tables = _tables(conn)
    started = None
    if "retrieval_telemetry_config" in tables:
        row = conn.execute("SELECT value FROM retrieval_telemetry_config WHERE key='retrieval_telemetry_started_at'").fetchone()
        started = row[0] if row else None
    start_dt = _parse_time(started)
    days = max(0.0, (as_of - start_dt).total_seconds() / 86400) if start_dt else 0.0
    if "retrieval_events" not in tables:
        return TelemetryWindow(started, days, 0, 0, 0, 0, 0.0)
    total = conn.execute("SELECT COUNT(DISTINCT request_id) FROM retrieval_events").fetchone()[0]
    complete = conn.execute("SELECT COUNT(*) FROM (SELECT request_id FROM retrieval_events GROUP BY request_id HAVING COUNT(DISTINCT stage)=3)").fetchone()[0]
    injected = conn.execute("SELECT COUNT(*) FROM retrieval_events WHERE stage='injected'").fetchone()[0]
    failures = 0
    if "retrieval_telemetry_config" in tables:
        row = conn.execute("SELECT value FROM retrieval_telemetry_config WHERE key='telemetry_write_failures'").fetchone()
        try:
            failures = int(row[0]) if row else 0
        except (TypeError, ValueError):
            failures = 0
    return TelemetryWindow(started, days, complete, total, injected, failures, failures / total if total else 0.0)


def _candidate(row: dict[str, Any], kind: str, confidence: str, reason: str, evidence: list[str], protected: list[str], window: TelemetryWindow, action: str) -> dict[str, Any]:
    assert kind in CANDIDATE_TYPES and confidence in CONFIDENCES
    content = str(_field(row, "content", default=""))
    return {
        "memory_id": str(_field(row, "id", "memory_id", default="")),
        "content_hash": sha256_text(content),
        "normalized_content_hash": sha256_text(normalize_content(content)),
        "source": _field(row, "source", default=""),
        "scope": _field(row, "scope_id", "scope", default=""),
        "session": _field(row, "session_id", "session", default=""),
        "created_at": _field(row, "created_at", default=None),
        "candidate_type": kind,
        "candidate_reason": reason,
        "deterministic_evidence": evidence,
        "protected_checks": {"matched": bool(protected), "reasons": protected},
        "telemetry_observation_window": {
            "telemetry_started_at": window.started_at,
            "observed_days": round(window.observed_days, 6),
            "observation": USAGE_BOUNDARY_TEXT,
        },
        "recommended_review_action": action,
        "confidence": confidence,
    }


def _lineage_valid(row: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    source_id = str(_field(row, "id", default=""))
    target_id = str(_field(row, "superseded_by", default="") or "")
    target = by_id.get(target_id)
    if not target:
        return False, ["DANGLING_SUCCESSOR"]
    if str(_field(row, "scope_id", default="")) != str(_field(target, "scope_id", default="")):
        return False, ["SCOPE_MISMATCH"]
    project, target_project = _memory_project(row), _memory_project(target)
    if not project or project != target_project:
        return False, ["PROJECT_MISMATCH_OR_MISSING"]
    older, newer = _parse_time(_field(row, "created_at")), _parse_time(_field(target, "created_at"))
    if not older or not newer or newer <= older:
        return False, ["INVALID_TIME_ORDER"]
    seen = {source_id}
    cursor = target
    while cursor:
        current_id = str(_field(cursor, "id", default=""))
        if current_id in seen:
            return False, ["LINEAGE_CYCLE"]
        seen.add(current_id)
        next_id = str(_field(cursor, "superseded_by", default="") or "")
        if not next_id:
            break
        cursor = by_id.get(next_id)
        if cursor is None:
            return False, ["DANGLING_LINEAGE"]
    reverse = str(_field(target, "supersedes", default="") or "")
    if reverse and source_id not in {part.strip() for part in re.split(r"[,\s]+", reverse)}:
        return False, ["REVERSE_LINEAGE_MISMATCH"]
    return True, ["EXPLICIT_SUCCESSOR", "SAME_SCOPE_AND_PROJECT", "STRICT_TIME_ORDER", "ACYCLIC_NON_DANGLING_LINEAGE"]


def reconciliation_report(conn: sqlite3.Connection, ledger: sqlite3.Connection | None, closed_954: set[str], as_of: datetime, curated_texts: Iterable[str] = ()) -> dict[str, Any]:
    anomalies: list[dict[str, Any]] = []
    tables = _tables(conn)
    memories = [_row_dict(r) for r in conn.execute("SELECT * FROM memories")]
    by_id = {str(_field(r, "id")): r for r in memories}
    ledger_rows: list[dict[str, Any]] = []
    if ledger and "decision_ledger" in _tables(ledger):
        ledger_rows = [_row_dict(r) for r in ledger.execute("SELECT * FROM decision_ledger")]
    applied = [r for r in ledger_rows if str(_field(r, "decision_status", default="")).casefold() in {"applied", "correction_applied", "completed"}]
    archive_actions = {"archive", "archive_applied", "retrieval_exclude", "exclude_from_retrieval"}
    applied_archive = [r for r in applied if str(_field(r, "final_action", "proposed_action", default="")).casefold() in archive_actions]
    # Canonical historical closure rows predate the apply-status convention but
    # carry an apply batch and final ARCHIVE action.
    applied_archive += [r for r in ledger_rows if str(_field(r, "decision_status", default="")).casefold() == "approved" and str(_field(r, "final_action", default="")).casefold() == "archive" and bool(_field(r, "apply_batch_id", default=""))]
    applied_ids: set[str] = set()
    for row in applied_archive:
        memory_id = str(_field(row, "memory_id", default=""))
        evidence_raw = str(_field(row, "evidence_json", default="") or "")
        try:
            evidence = json.loads(evidence_raw) if evidence_raw else {}
        except json.JSONDecodeError:
            evidence = {}
        members = evidence.get("all_memory_ids") if isinstance(evidence, dict) else None
        if isinstance(members, list) and members:
            applied_ids.update(str(value) for value in members)
        elif memory_id in by_id:
            applied_ids.add(memory_id)
    for row in ledger_rows:
        if str(_field(row, "decision_status", default="")).casefold() != "repaired":
            continue
        try:
            evidence = json.loads(str(_field(row, "evidence_json", default="") or "{}"))
        except json.JSONDecodeError:
            evidence = {}
        repaired_id = str(_field(row, "memory_id", default=""))
        if evidence.get("production_state") == "retrieval_excluded=1":
            applied_ids.add(repaired_id)
        if evidence.get("production_expected_state") == "retrieval_excluded=0":
            applied_ids.discard(repaired_id)
    excluded_ids = {mid for mid, row in by_id.items() if int(_field(row, "retrieval_excluded", default=0) or 0) == 1}
    for mid in sorted(excluded_ids - applied_ids):
        anomalies.append({"type": "EXCLUDED_WITHOUT_APPLIED_EVENT", "memory_id": mid})
    for mid in sorted(applied_ids - excluded_ids):
        anomalies.append({"type": "APPLIED_ARCHIVE_WITHOUT_PRODUCTION_STATE", "memory_id": mid})
    dup_ledger = Counter((str(_field(r, "memory_id", default="")), str(_field(r, "apply_batch_id", default="")), str(_field(r, "final_action", default=""))) for r in applied)
    for key, count in sorted(dup_ledger.items()):
        if count > 1 and key[0]:
            anomalies.append({"type": "DUPLICATE_APPLIED_EVENT", "memory_id": key[0], "key": list(key), "count": count})
    telemetry_duplicate_count = incomplete = 0
    aggregate_mismatch = 0
    if "retrieval_events" in tables:
        telemetry_duplicate_count = conn.execute("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM retrieval_events GROUP BY request_id,memory_id,stage,retrieval_path,COALESCE(rank,-1) HAVING n>1)").fetchone()[0]
        incomplete = conn.execute("SELECT COUNT(*) FROM (SELECT request_id FROM retrieval_events GROUP BY request_id HAVING COUNT(DISTINCT stage)<3)").fetchone()[0]
        rows = conn.execute("SELECT memory_id,COUNT(DISTINCT request_id) n,MAX(occurred_at) last_at FROM retrieval_events WHERE stage='injected' AND memory_domain='database' GROUP BY memory_id").fetchall()
        for event in rows:
            mem = by_id.get(str(event[0]))
            if not mem or int(_field(mem, "hit_count", default=0) or 0) < int(event[1]) or str(_field(mem, "last_retrieved_at", default="") or "") < str(event[2] or ""):
                aggregate_mismatch += 1
                anomalies.append({"type": "TELEMETRY_AGGREGATE_MISMATCH", "memory_id": str(event[0])})
        if telemetry_duplicate_count:
            anomalies.append({"type": "TELEMETRY_DUPLICATE_EVENTS", "count": telemetry_duplicate_count})
        if incomplete:
            anomalies.append({"type": "INCOMPLETE_TELEMETRY_COVERAGE", "request_count": incomplete})
    promotion_rows = [r for r in applied if str(_field(r, "final_action", default="")).casefold().startswith("promote_to_")]
    promotion_matches = 0
    for promotion in promotion_rows:
        memory = by_id.get(str(_field(promotion, "memory_id", default="")))
        try:
            promotion_evidence = json.loads(str(_field(promotion, "evidence_json", default="") or "{}"))
        except json.JSONDecodeError:
            promotion_evidence = {}
        inserted = str(promotion_evidence.get("exact_inserted_text") or "")
        if (inserted and _curated_equivalent(inserted, curated_texts)) or (memory and _curated_equivalent(str(_field(memory, "content", default="")), curated_texts)):
            promotion_matches += 1
    return {
        "schema_version": VERSION,
        "generated_at": as_of.isoformat(),
        "checks": {
            "retrieval_excluded_with_applied_event": {"excluded": len(excluded_ids), "applied_archive_ids": len(applied_ids)},
            "duplicate_applied_events": sum(max(0, n - 1) for n in dup_ledger.values()),
            "ledger_update_delete": {"status": "APPEND_ONLY_ROWS_INSPECTED", "rows": len(ledger_rows), "note": "SQLite current-state inspection cannot prove absence of historic mutation without a prior row snapshot."},
            "durable_promotion_curated_patch": {"applied_promotions": len(promotion_rows), "content_matches": promotion_matches, "status": "MATCHED" if promotion_matches == len(promotion_rows) else "MISMATCH"},
            "telemetry_aggregate_mismatches": aggregate_mismatch,
            "event_duplicate_count": telemetry_duplicate_count,
            "incomplete_coverage_requests": incomplete,
            "historical_954_set_size": len(closed_954),
        },
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def generate_shadow_manifest(conn: sqlite3.Connection, *, ledger: sqlite3.Connection | None = None, closed_954: set[str] | None = None, curated_texts: Iterable[str] = (), as_of: datetime | None = None, shadow_only: bool = True, apply: bool = False, high_limit: int = 100, durable_limit: int = 20) -> tuple[dict[str, Any], dict[str, Any]]:
    if not shadow_only or apply:
        raise ShadowSafetyError("Prospective governance V1 has no apply entrypoint; shadow_only=true is mandatory")
    if conn.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise ShadowSafetyError("production connection is not query_only")
    if ledger is not None and ledger.execute("PRAGMA query_only").fetchone()[0] != 1:
        raise ShadowSafetyError("ledger connection is not query_only")
    now = as_of or datetime.now(timezone.utc)
    closed = set(closed_954 or ())
    rows = [_row_dict(r) for r in conn.execute("SELECT * FROM memories")]
    by_id = {str(_field(r, "id")): r for r in rows}
    window = telemetry_window(conn, now)
    candidates: list[dict[str, Any]] = []
    protected_excluded_ids: set[str] = set()
    seen_ids: set[str] = set()

    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    norm_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        content = str(_field(row, "content", default=""))
        raw_groups[sha256_text(content)].append(row)
        norm_groups[sha256_text(normalize_content(content))].append(row)
    duplicate_groups = [("FULL_CONTENT_HASH", group) for group in raw_groups.values() if len(group) > 1]
    raw_member_ids = {str(_field(r, "id")) for _, group in duplicate_groups for r in group}
    duplicate_groups += [("NORMALIZED_CONTENT_HASH", group) for group in norm_groups.values() if len(group) > 1 and not all(str(_field(r, "id")) in raw_member_ids for r in group)]
    for basis, group in sorted(duplicate_groups, key=lambda item: min(str(_field(r, "id")) for r in item[1])):
        ordered = sorted(group, key=lambda r: (str(_field(r, "created_at", default="")), str(_field(r, "id"))))
        canonical = str(_field(ordered[0], "id"))
        for row in ordered[1:]:
            mid = str(_field(row, "id"))
            if mid in seen_ids:
                continue
            protected = protected_reasons(row, closed)
            if protected:
                protected_excluded_ids.add(mid)
                continue
            candidates.append(_candidate(row, "EXACT_DUPLICATE_CANDIDATE", "HIGH_CONFIDENCE", "Content is byte-identical or differs only by conservative Unicode/case/whitespace normalization.", [basis, f"CANONICAL_MEMORY_ID={canonical}"], [], window, "User review for duplicate retention; no action is implied."))
            seen_ids.add(mid)

    for row in rows:
        mid = str(_field(row, "id"))
        protected = protected_reasons(row, closed)
        content = str(_field(row, "content", default=""))
        evidence = tool_log_evidence(content)
        if evidence and mid not in seen_ids:
            if protected:
                protected_excluded_ids.add(mid)
            else:
                candidates.append(_candidate(row, "TOOL_LOG_NOISE_CANDIDATE", "HIGH_CONFIDENCE", "Multiple structural markers identify raw machine/tool output.", evidence, [], window, "User review for possible retrieval exclusion; no action is implied."))
                seen_ids.add(mid)
        successor = str(_field(row, "superseded_by", default="") or "")
        if successor and mid not in seen_ids:
            valid, lineage_evidence = _lineage_valid(row, by_id)
            if valid:
                if protected:
                    protected_excluded_ids.add(mid)
                else:
                    candidates.append(_candidate(row, "SUPERSEDED_STATE_CANDIDATE", "HIGH_CONFIDENCE", "Explicit, validated successor lineage exists.", lineage_evidence + [f"SUCCESSOR={successor}"], [], window, "User review of lineage state; no action is implied."))
                    seen_ids.add(mid)
        durable = _stable_durable_evidence(content)
        if durable and mid not in seen_ids and not _curated_equivalent(content, curated_texts):
            candidates.append(_candidate(row, "DURABLE_REVIEW_CANDIDATE", "MEDIUM_CONFIDENCE", "Explicit stable preference or cross-session engineering principle is absent from curated files.", durable, protected, window, "User review for possible durable curation; no promotion is implied."))
            seen_ids.add(mid)
        created = _parse_time(_field(row, "created_at"))
        age_days = (now - created).total_seconds() / 86400 if created else 0
        episodic = str(_field(row, "memory_type", "type", "target", default="")).casefold() in {"episodic", "event", "task", "general"}
        if mid not in seen_ids and episodic and age_days >= 90 and not protected:
            injected = conn.execute("SELECT 1 FROM retrieval_events WHERE memory_id=? AND stage='injected' LIMIT 1", (mid,)).fetchone() if "retrieval_events" in _tables(conn) else None
            usage_evidence = [f"AGE_DAYS={int(age_days)}"]
            if not injected:
                usage_evidence.append(USAGE_BOUNDARY_TEXT)
            candidates.append(_candidate(row, "STALE_EPISODIC_REVIEW_CANDIDATE", "MEDIUM_CONFIDENCE", "Age and episodic content type warrant human freshness review; usage is supplementary only.", usage_evidence, [], window, "Review current relevance; no archive action is implied."))
            seen_ids.add(mid)

    durable_candidates = [c for c in candidates if c["candidate_type"] == "DURABLE_REVIEW_CANDIDATE"]
    allowed_durable = {c["memory_id"] for c in sorted(durable_candidates, key=lambda c: c["memory_id"])[:durable_limit]}
    candidates = [c for c in candidates if c["candidate_type"] != "DURABLE_REVIEW_CANDIDATE" or c["memory_id"] in allowed_durable]
    candidates.sort(key=lambda c: (CONFIDENCES.index(c["confidence"]), c["candidate_type"], c["memory_id"]))
    high = [c for c in candidates if c["confidence"] == "HIGH_CONFIDENCE"][:high_limit]
    candidates = high + [c for c in candidates if c["confidence"] != "HIGH_CONFIDENCE"]
    reconciliation = reconciliation_report(conn, ledger, closed, now, curated_texts)
    for anomaly in reconciliation["anomalies"]:
        kind = "TELEMETRY_COVERAGE_ANOMALY" if anomaly["type"].startswith(("TELEMETRY", "INCOMPLETE")) else "PRODUCTION_LEDGER_INCONSISTENCY"
        pseudo = {"id": anomaly.get("memory_id", f"reconciliation:{anomaly['type']}"), "content": "", "source": "mechanical_reconciliation", "created_at": now.isoformat()}
        candidates.append(_candidate(pseudo, kind, "HIGH_CONFIDENCE" if kind == "PRODUCTION_LEDGER_INCONSISTENCY" else "OBSERVE_ONLY", "Mechanical reconciliation anomaly; report only.", [json.dumps(anomaly, sort_keys=True, separators=(",", ":"))], [], window, "Investigate evidence; no automatic repair is permitted."))
    candidates.sort(key=lambda c: (CONFIDENCES.index(c["confidence"]), c["candidate_type"], c["memory_id"]))
    candidates = [c for c in candidates if c["confidence"] == "HIGH_CONFIDENCE"][:high_limit] + [c for c in candidates if c["confidence"] != "HIGH_CONFIDENCE"]
    # The approval package is mechanically derived and contains HIGH only, capped at 100.
    approval = [c for c in candidates if c["confidence"] == "HIGH_CONFIDENCE"][:high_limit]
    counts_conf = {name: sum(c["confidence"] == name for c in candidates) for name in CONFIDENCES}
    counts_type = {name: sum(c["candidate_type"] == name for c in candidates) for name in sorted(CANDIDATE_TYPES)}
    manifest = {
        "schema_version": VERSION,
        "shadow_only": True,
        "generated_at": now.isoformat(),
        "usage_boundary": "usage observations only cover the period since telemetry_started_at",
        "telemetry": window.__dict__,
        "limits": {"high_confidence": high_limit, "durable_review": durable_limit},
        "counts": {"total": len(candidates), "by_confidence": counts_conf, "by_candidate_type": counts_type, "protected_excluded": len(protected_excluded_ids), "approval_batch": len(approval)},
        "candidates": candidates,
        "first_suggested_approval_batch": approval,
        "safety": {"production_writes": 0, "ledger_writes": 0, "schema_writes": 0, "apply_entrypoint": "REJECTED"},
    }
    return manifest, reconciliation


def validate_manifest(manifest: dict[str, Any], reconciliation: dict[str, Any]) -> dict[str, Any]:
    candidates = manifest["candidates"]
    recomputed_conf = {name: sum(c["confidence"] == name for c in candidates) for name in CONFIDENCES}
    recomputed_type = {name: sum(c["candidate_type"] == name for c in candidates) for name in sorted(CANDIDATE_TYPES)}
    checks = {
        "shadow_only_true": manifest.get("shadow_only") is True,
        "candidate_types_legal": all(c["candidate_type"] in CANDIDATE_TYPES for c in candidates),
        "confidence_legal": all(c["confidence"] in CONFIDENCES for c in candidates),
        "counts_mechanically_aggregate": manifest["counts"]["total"] == len(candidates) and manifest["counts"]["by_confidence"] == recomputed_conf and manifest["counts"]["by_candidate_type"] == recomputed_type,
        "high_cap": recomputed_conf["HIGH_CONFIDENCE"] <= manifest["limits"]["high_confidence"],
        "durable_cap": recomputed_type["DURABLE_REVIEW_CANDIDATE"] <= manifest["limits"]["durable_review"],
        "approval_batch_high_only": all(c["confidence"] == "HIGH_CONFIDENCE" for c in manifest["first_suggested_approval_batch"]),
        "approval_batch_cap": len(manifest["first_suggested_approval_batch"]) <= 100,
        "closed_954_overlap": not any("CLOSED_954" in reason for c in candidates for reason in c["protected_checks"]["reasons"]),
        "safety_writes_zero": all(manifest["safety"][key] == 0 for key in ("production_writes", "ledger_writes", "schema_writes")),
    }
    return {"schema_version": VERSION, "generated_at": manifest["generated_at"], "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "manifest_sha256": sha256_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))), "reconciliation_anomaly_count": reconciliation["anomaly_count"]}


def render_report(manifest: dict[str, Any], validation: dict[str, Any], reconciliation: dict[str, Any]) -> str:
    counts = manifest["counts"]
    telemetry = manifest["telemetry"]
    return f"""# Weekly Memory Governance Shadow Report

- Status: `{validation['status']}`
- Mode: `shadow_only=true`
- Generated at: `{manifest['generated_at']}`
- Telemetry started at: `{telemetry['started_at']}`
- Observed days: `{telemetry['observed_days']:.6f}`
- Complete requests: `{telemetry['complete_requests']}`
- Injected events: `{telemetry['injected_events']}`
- Telemetry failure rate: `{telemetry['failure_rate']:.6f}`
- Usage boundary: `usage observations only cover the period since telemetry_started_at`

## Candidate distribution

- HIGH_CONFIDENCE: {counts['by_confidence']['HIGH_CONFIDENCE']}
- MEDIUM_CONFIDENCE: {counts['by_confidence']['MEDIUM_CONFIDENCE']}
- OBSERVE_ONLY: {counts['by_confidence']['OBSERVE_ONLY']}
- Protected exclusions: {counts['protected_excluded']}
- First suggested approval batch: {counts['approval_batch']} (HIGH_CONFIDENCE only; maximum 100)
- Reconciliation anomalies: {reconciliation['anomaly_count']}

These are review candidates, not final actions. No apply, archive, promotion, lineage mutation, ledger write, schema write, or memory write occurred.
"""


def run_shadow(*, database: Path, output_dir: Path, ledger_path: Path | None = None, closed_954_ids: set[str] | None = None, curated_paths: Iterable[Path] = (), as_of: datetime | None = None, shadow_only: bool = True, apply: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not shadow_only or apply:
        raise ShadowSafetyError("apply is forbidden; pass shadow_only=true")
    output_dir.mkdir(parents=True, exist_ok=True)
    curated = [p.read_text(encoding="utf-8") for p in curated_paths if p.exists()]
    conn = _open_readonly(database)
    ledger = _open_readonly(ledger_path, immutable=True) if ledger_path and ledger_path.exists() else None
    try:
        closed = set(closed_954_ids or ())
        cols = _columns(conn, "memories")
        if not closed and {"target", "superseded_by"}.issubset(cols):
            frozen = {str(r[0]) for r in conn.execute("SELECT id FROM memories WHERE target='general' AND (superseded_by IS NULL OR superseded_by='')")}
            # The canonical freeze contract is exactly 954; never silently treat a
            # changed population as the historical set.
            if len(frozen) == 954:
                closed = frozen
        manifest, reconciliation = generate_shadow_manifest(conn, ledger=ledger, closed_954=closed, curated_texts=curated, as_of=as_of)
        validation = validate_manifest(manifest, reconciliation)
    finally:
        conn.close()
        if ledger:
            ledger.close()
    artifacts = {
        "WEEKLY_MEMORY_GOVERNANCE_SHADOW_MANIFEST.json": json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "WEEKLY_MEMORY_GOVERNANCE_SHADOW_VALIDATION.json": json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "WEEKLY_MEMORY_RECONCILIATION_REPORT.json": json.dumps(reconciliation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "WEEKLY_MEMORY_GOVERNANCE_SHADOW_REPORT.md": render_report(manifest, validation, reconciliation),
    }
    for name, payload in artifacts.items():
        (output_dir / name).write_text(payload, encoding="utf-8")
    return manifest, validation, reconciliation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--curated", type=Path, action="append", default=[])
    parser.add_argument("--closed-954-ids", type=Path)
    parser.add_argument("--shadow-only", action="store_true", required=True)
    parser.add_argument("--apply", action="store_true", help="Always rejected; present only to make the safety boundary explicit.")
    args = parser.parse_args(argv)
    if args.apply:
        raise ShadowSafetyError("--apply is forbidden in Prospective Governance V1")
    closed: set[str] = set()
    if args.closed_954_ids:
        data = json.loads(args.closed_954_ids.read_text(encoding="utf-8"))
        values = data if isinstance(data, list) else data.get("memory_ids", [])
        closed = {str(value) for value in values}
    run_shadow(database=args.database, ledger_path=args.ledger, output_dir=args.output_dir, closed_954_ids=closed, curated_paths=args.curated, shadow_only=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
