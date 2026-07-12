"""Read views over curated files, SQLite truth rows, and vector companion hits.

These views apply lifecycle and visibility filters before recall merges candidates."""

from __future__ import annotations

import sqlite3
from typing import Any

from .gating import build_fts_query, compact_text, like_terms, normalized_token_set, query_tokens
from .governance import classify_memory
from .graph import load_metadata
from .models import RecallItem
from .scoring import bm25_to_score, lexical_score
from .sql_store import curated_recall_item_id, iter_curated_entries
from .vector_runtime import mark_vector_needs_repair

# Defensive retrieval boundary: lifecycle filtering must happen in the candidate
# SQL/vector-adapter layer, not only after merge/dedupe. Fresh archived rows can
# otherwise consume LIMIT budget or suppress active duplicates.
_RECALL_HIDDEN_LIFECYCLE_VALUES = ("superseded", "obsolete", "rejected", "archived", "candidate", "in_progress")
_RECALL_HIDDEN_LIFECYCLE_SET = set(_RECALL_HIDDEN_LIFECYCLE_VALUES)


def _recall_lifecycle_visible_sql(alias: str) -> str:
    lifecycle_expr = f"LOWER(COALESCE(CASE WHEN json_valid({alias}.metadata) THEN json_extract({alias}.metadata, '$.lifecycle') ELSE '' END, ''))"
    hidden_values = ",".join(f"'{value}'" for value in _RECALL_HIDDEN_LIFECYCLE_VALUES)
    return f"{lifecycle_expr} NOT IN ({hidden_values})"


_ACTIVE_MEMORY_SQL = _recall_lifecycle_visible_sql("memories")
_ACTIVE_MEMORY_SQL_M = _recall_lifecycle_visible_sql("m")


# ---------------------------------------------------------------------------
# Retrieval Exclusion Eligibility Contract
# ---------------------------------------------------------------------------

RETRIEVAL_ELIGIBLE_WHERE = "retrieval_excluded = 0"
"""SQLite WHERE fragment: only rows with retrieval_excluded=0 are eligible for default retrieval.

Records stored as `retrieval_excluded=1` (or the column missing → treated as excluded)
MUST NOT be returned by any default retrieval path.

Semantic boundary with superseded_by:
  - retrieval_excluded = 1  →  do not return via default retrieval (archive-only)
  - superseded_by IS NOT NULL  →  this record has been replaced by another record.
    Such records may still appear in default retrieval if retrieval_excluded=0;
    the supersession relationship is orthogonal to retrieval exclusion.
"""


def _retrieval_eligible_ids(conn: sqlite3.Connection, ids: list[str]) -> set[str]:
    """Return the subset of IDs that are eligible for default retrieval
    (retrieval_excluded = 0) — positive proof of existence + eligibility.

    Positive-eligibility contract
    -----------------------------
    * An ID is eligible only if:
        1. It exists in the ``memories`` table, AND
        2. Its ``retrieval_excluded`` column is ``0``.
    * IDs that are absent from the table (unknown, stale, deleted) are
      never eligible.
    * IDs that exist but have ``retrieval_excluded = 1`` are never eligible.
    * On any SQL error the result is the empty set (fail-closed).

    Chunking
    --------
    SQLite has a compile-time limit of ``SQLITE_MAX_VARIABLE_NUMBER``
    (default 999).  We chunk at 900 to stay well below the limit.  Each
    chunk is a separate parameterised query; results are unioned before
    return.  Deduplication is implicit — an ID is either in the result
    set or not.
    """
    if not ids:
        return set()
    chunk_size = 900
    result: set[str] = set()
    try:
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT id FROM memories WHERE id IN ({placeholders}) AND retrieval_excluded = 0",
                chunk,
            ).fetchall()
            result.update(str(row["id"]) for row in rows)
    except Exception:
        # Fail-closed: on any query error, return empty — no results pass through.
        return set()
    return result


def _excluded_ids(conn: sqlite3.Connection) -> set[str]:
    """Return all IDs with retrieval_exclusion=1.

    Legacy helper — superseded by ``_retrieval_eligible_ids()`` which
    uses a positive proof of existence + eligibility contract.  This
    function is retained for backward compatibility but is NOT used
    by the merge layer in recall.py (which now uses positive eligibility).
    """
    try:
        return {
            str(row["id"])
            for row in conn.execute(
                "SELECT id FROM memories WHERE retrieval_excluded = 1"
            ).fetchall()
        }
    except Exception:
        return set()


def _scope_placeholders(provider: Any) -> str:
    return ",".join("?" for _ in provider._accessible_scope_ids)


def _accessible_scope_params(provider: Any) -> list[str]:
    return [str(scope_id) for scope_id in provider._accessible_scope_ids]


def _alias_like_terms(query: str, tokens: list[str]) -> list[str]:
    """Return alias-expanded LIKE terms that are not already in the raw query.

    This preserves lexical-only recall for curated aliases such as response→reply
    after removing the unsafe arbitrary-recency backfill. We include both the
    canonical alias and known surface forms because SQLite LIKE is not aware of
    our stemming/alias map (e.g. response→reply must still discover rows that
    literally contain "replies").
    """
    raw_terms = set(tokens)
    raw_query = (query or "").lower()
    # Importing here keeps this module's SQL discovery policy in sync with
    # lexical scoring without broad recent-row scans.
    from .aliases import _ALIAS_MAP, canonicalize_alias  # type: ignore[attr-defined]

    canonical_to_terms: dict[str, list[str]] = {}
    for raw in normalized_token_set(tokens):
        canonical_to_terms.setdefault(canonicalize_alias(raw), [])
    for surface, canonical in _ALIAS_MAP.items():
        if canonical in canonical_to_terms:
            canonical_to_terms.setdefault(canonical, []).append(surface)
    terms: list[str] = []
    seen: set[str] = set()
    for canonical, surfaces in canonical_to_terms.items():
        for term in [canonical, *surfaces]:
            if not term or term in raw_terms or term in raw_query or term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= 12:
                return terms
    return terms


def _row_metadata(
    row: sqlite3.Row,
    *,
    lexical_score: float = 0.0,
    vector_score: float = 0.0,
    bm25_score: float | None = None,
) -> dict[str, Any]:
    metadata = load_metadata(row["metadata"] if "metadata" in row.keys() else "{}")
    metadata.update(
        {
            "lexical_score": lexical_score,
            "vector_score": vector_score,
            "scope_id": row["scope_id"],
            "created_at": row["created_at"] if "created_at" in row.keys() else row["updated_at"],
        }
    )
    if bm25_score is not None:
        metadata["bm25_score"] = bm25_score
    return metadata


def search_db_memories(provider: Any, query: str, *, limit: int) -> list[RecallItem]:
    """Search SQLite truth rows for accessible recall candidates.

    Lifecycle and scope filters are applied here before ranking so downstream retrieval cannot accidentally surface archived or inaccessible state."""
    conn = provider._require_conn()
    tokens = query_tokens(query)
    fts_query = build_fts_query(tokens)
    rows: list[sqlite3.Row] = []
    try:
        configured_pool = int((provider._retrieval_config or {}).get("candidate_pool") or 0)
    except (TypeError, ValueError):
        configured_pool = 0
    candidate_pool = max(limit * 2, limit, configured_pool)
    with provider._lock:
        if fts_query:
            rows.extend(
                conn.execute(
                    f"""
                    SELECT m.*, bm25(memories_fts) AS bm25_score
                    FROM memories_fts
                    JOIN memories m ON m.id = memories_fts.memory_id
                    WHERE memories_fts MATCH ? AND m.scope_id IN ({_scope_placeholders(provider)})
                      AND {_ACTIVE_MEMORY_SQL_M} AND m.{RETRIEVAL_ELIGIBLE_WHERE}
                    ORDER BY bm25(memories_fts) ASC, m.updated_at DESC
                    LIMIT ?
                    """,
                    [fts_query, *_accessible_scope_params(provider), candidate_pool],
                ).fetchall()
            )
        like_query_terms = like_terms(query, tokens)
        if like_query_terms:
            clause = " OR ".join(["content LIKE ?", "summary LIKE ?"] * len(like_query_terms))
            params: list[Any] = []
            for term in like_query_terms:
                needle = f"%{term}%"
                params.extend([needle, needle])
            params.extend([*_accessible_scope_params(provider), candidate_pool])
            rows.extend(
                conn.execute(
                    f"""
                    SELECT *
                    FROM memories
                    WHERE ({clause}) AND scope_id IN ({_scope_placeholders(provider)})
                      AND {_ACTIVE_MEMORY_SQL} AND {RETRIEVAL_ELIGIBLE_WHERE}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            )
        alias_terms = _alias_like_terms(query, tokens)
        if alias_terms:
            clause = " OR ".join(["content LIKE ?", "summary LIKE ?"] * len(alias_terms))
            params = []
            for term in alias_terms:
                needle = f"%{term}%"
                params.extend([needle, needle])
            params.extend([*_accessible_scope_params(provider), candidate_pool])
            rows.extend(
                conn.execute(
                    f"""
                    SELECT *
                    FROM memories
                    WHERE ({clause}) AND scope_id IN ({_scope_placeholders(provider)})
                      AND {_ACTIVE_MEMORY_SQL} AND {RETRIEVAL_ELIGIBLE_WHERE}
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
            )
        # Do not backfill retrieval with arbitrary recent memories.
        # Earlier versions scanned newest rows when lexical LIKE/FTS returned too
        # few candidates, then accepted durable/tool rows on source/target bonus
        # alone. That made unrelated fresh conversations recall stale ops notes
        # (for example OpenClaw/凌晨 task context) despite zero token overlap.
        # Recency is only a reranking bonus after relevance is established.

    bm25_raw_scores: dict[str, float | None] = {}
    for row in rows:
        if "bm25_score" not in row.keys():
            continue
        try:
            bm25_raw_scores[str(row["id"])] = float(row["bm25_score"])
        except (TypeError, ValueError):
            continue
    bm25_scores = bm25_to_score(bm25_raw_scores)
    dedup_rows: dict[str, sqlite3.Row] = {row["id"]: row for row in rows}
    min_score = float((provider._retrieval_config or {}).get("min_score") or provider._config_value("min_score", 0.18))
    results: list[RecallItem] = []
    for row in dedup_rows.values():
        score = lexical_score(
            query=query,
            content=row["content"],
            summary=row["summary"],
            source=row["source"],
            target=row["target"],
        )
        if score < min_score * 0.5:
            continue
        results.append(
            RecallItem(
                id=row["id"],
                content=row["content"],
                summary=row["summary"],
                source=row["source"],
                target=row["target"],
                score=score,
                updated_at=row["updated_at"],
                metadata=_row_metadata(
                    row,
                    lexical_score=score,
                    vector_score=0.0,
                    bm25_score=bm25_scores.get(str(row["id"])),
                ),
            )
        )
        if results[-1].metadata is not None and str(row["id"]) in bm25_raw_scores:
            results[-1].metadata["bm25_raw"] = bm25_raw_scores[str(row["id"])]
    return results


def search_vector_memories(provider: Any, query: str, *, limit: int) -> list[RecallItem]:
    """Search vector companion state and return recall candidates that still pass SQLite visibility checks.

    Vector hits are suggestions only; final access and lifecycle validation remains anchored to truth rows."""
    if not provider._vector_ready or not provider._vector_store or not provider._embedder:
        return []
    try:
        query_vector = provider._embedder.embed_query(query)
        top_k = max(limit, int((provider._vector_config or {}).get("top_k") or limit))
        rows = []
        for scope_id in provider._accessible_scope_ids:
            rows.extend(provider._vector_store.search(query_vector, scope_id=scope_id, limit=top_k))
    except Exception as exc:
        mark_vector_needs_repair(provider, exc)
        return []
    threshold = float((provider._retrieval_config or {}).get("vector_min_score") or 0.12)
    id_metadata: dict[str, dict[str, Any]] = {}
    row_ids = [str(row.get("id") or "") for row in rows if str(row.get("id") or "")]
    if row_ids:
        placeholders = ",".join("?" for _ in row_ids)
        try:
            with provider._lock:
                meta_rows = provider._require_conn().execute(
                    f"SELECT id, scope_id, created_at, metadata FROM memories WHERE id IN ({placeholders})",
                    row_ids,
                ).fetchall()
            id_metadata = {}
            for row in meta_rows:
                metadata = load_metadata(row["metadata"])
                metadata.setdefault("created_at", str(row["created_at"]))
                id_metadata[str(row["id"])] = metadata
        except Exception:
            id_metadata = {}
    results: list[RecallItem] = []
    for row in rows:
        row_id = str(row.get("id") or "")
        if row_ids and row_id not in id_metadata:
            continue
        distance = float(row.get("_distance") or 0.0)
        vector_score = max(0.0, 1.0 - distance)
        if vector_score < threshold:
            continue
        metadata = dict(id_metadata.get(row_id) or {})
        lifecycle = str(metadata.get("lifecycle") or "").strip().lower()
        if lifecycle in _RECALL_HIDDEN_LIFECYCLE_SET:
            continue
        metadata.update({"lexical_score": 0.0, "vector_score": vector_score, "scope_id": row.get("scope_id")})
        results.append(
            RecallItem(
                id=row["id"],
                content=row["content"],
                summary=row["summary"],
                source=row["source"],
                target=row["target"],
                score=vector_score,
                updated_at=row["updated_at"],
                metadata=metadata,
            )
        )
    # Phase 2: post-filter against SQLite source-of-truth.
    # Even if the vector index still contains a row for an excluded memory,
    # we must not return it. Batch-query SQLite for eligibility.
    conn = provider._require_conn()
    eligible = _retrieval_eligible_ids(conn, [item.id for item in results])
    results = [item for item in results if item.id in eligible]
    return results


def _curated_memory_allowed(provider: Any) -> bool:
    raw_cfg = (provider._config or {}).get("curated_memory", {})
    if raw_cfg is False:
        return False
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    mode = str(cfg.get("mode") or "single-user").strip().lower()
    if mode in {"disabled", "off", "false", "none"}:
        return False

    scope = getattr(provider, "_scope", None)
    user_id = str(getattr(scope, "user_id", "") or "")
    allowed = [str(item).strip() for item in (cfg.get("allowed_user_ids") or []) if str(item).strip()]
    if allowed:
        return bool(user_id and user_id in allowed)
    if mode in {"explicit-users", "allow-list", "allowlist"}:
        return False
    if mode in {"shared"}:
        # Deprecated compatibility alias kept explicit so operator-facing schema
        # can advertise only the canonical runtime modes.
        mode = "profile-global"
    if mode in {"profile-global", "global", "all-users"}:
        return True
    # Safe default: global curated files may be injected only when Hermes is not
    # running with an explicit gateway user id. Provider-owned SQLite rows remain
    # the scoped durable store for multi-user contexts.
    return not bool(user_id)


def search_curated_memories(provider: Any, query: str) -> list[RecallItem]:
    if not _curated_memory_allowed(provider):
        return []
    min_score = float((provider._retrieval_config or {}).get("min_score") or provider._config_value("min_score", 0.18))
    results: list[RecallItem] = []
    for target, content, updated_at in iter_curated_entries(provider._hermes_home):
        summary = compact_text(content, 220)
        score = lexical_score(
            query=query,
            content=content,
            summary=summary,
            source="builtin-curated",
            target=target,
        )
        if score < min_score:
            continue
        metadata = classify_memory(content, target, "builtin-curated")
        metadata.update({"lexical_score": score, "vector_score": 0.0})
        results.append(
            RecallItem(
                id=curated_recall_item_id(target, content),
                content=content,
                summary=summary,
                source="builtin-curated",
                target=target,
                score=score,
                updated_at=updated_at,
                metadata=metadata,
            )
        )
    return results


# NOTE (data model gap — 2026-07-12): Curated memory entries (file-based
# MEMORY.md/USER.md) do NOT have a stable mapping to `memories.id` — their
# IDs are synthetic (curated:<target>:<sha1>) and cannot carry
# retrieval_excluded state by themselves.  This means curated entries are
# always eligible for default retrieval and cannot be individually excluded
# via the retrieval_excluded mechanism.  The RecallService merge layer
# (recall.py) provides the final fail-closed defense, but curated IDs will
# pass through since they don't exist in the memories table.
# To extend exclusion to curated entries, a mapping layer from curated
# entry hash to a governance-controlled exclusion list would be needed.
# For now, exclusion of curated content requires removing the file from
# the filesystem (deleting entries in MEMORY.md/USER.md).
#
# This gap means the A3 Program retrieval exclusion contract covers only
# database-backed memories (SQLite + vector).  Curated file-based memories
# are governed by a separate lifecycle (file deletion / manifest editing).


# ---------------------------------------------------------------------------
# Explicit archive / history queries
# ---------------------------------------------------------------------------

def search_archived_memories(
    provider: Any,
    *,
    memory_id: str | None = None,
    batch_id: str | None = None,
    reason: str | None = None,
    limit: int = 50,
) -> list[RecallItem]:
    """Retrieve retrieval-excluded records directly, bypassing default exclusion.

    Supports filtering by memory ID, exclusion batch ID, and/or exclusion reason.
    Preserves full content, source, lineage, and raw content hash.

    This is the *only* path that should return excluded records.  It must NOT
    be used by default retrieval (search_db_memories etc.).  The WHERE clause
    explicitly targets retrieval_excluded = 1 to avoid accidental inclusion.
    """
    conn = provider._require_conn()
    clauses: list[str] = ["retrieval_excluded = 1"]
    params: list[Any] = []
    if memory_id is not None:
        clauses.append("id = ?")
        params.append(memory_id)
    if batch_id is not None:
        clauses.append("retrieval_exclusion_batch_id = ?")
        params.append(batch_id)
    if reason is not None:
        clauses.append("retrieval_exclusion_reason LIKE ?")
        params.append(f"%{reason}%")
    # Scope constraint: only accessible scopes
    clauses.append("scope_id IN ({})".format(_scope_placeholders(provider)))
    params.extend(_accessible_scope_params(provider))
    params.append(limit)
    where = " AND ".join(clauses)
    rows = conn.execute(
        f"""
        SELECT id, content, summary, source, target, updated_at, scope_id,
               retrieval_excluded_at, retrieval_exclusion_batch_id, retrieval_exclusion_reason
        FROM memories
        WHERE {where}
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        RecallItem(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            source=row["source"],
            target=row["target"],
            score=1.0,
            updated_at=row["updated_at"],
            metadata={
                "scope_id": row["scope_id"],
                "retrieval_excluded_at": row["retrieval_excluded_at"],
                "retrieval_exclusion_batch_id": row["retrieval_exclusion_batch_id"],
                "retrieval_exclusion_reason": row["retrieval_exclusion_reason"],
            },
        )
        for row in rows
    ]
