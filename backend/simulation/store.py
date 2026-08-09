from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from backend.models import ChatMessage, MemorySource, MessageType
from backend.simulation.action_policy import Action


_ACTIVE_STATUSES = ("PENDING", "RETRY", "BLOCKED", "CLAIMED")
_CANCELLABLE_STATUSES = ("PENDING", "RETRY", "BLOCKED")
_DUE_STATUSES = ("PENDING", "RETRY")


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    event_id: str
    twin_person_id: str
    platform: str
    conversation_id: str
    action: Action
    due_at: datetime
    created_at: datetime
    status: str
    generation_attempts: int = 0
    last_error: str | None = None
    next_attempt_at: datetime | None = None
    claim_token: str | None = None
    claimed_at: datetime | None = None

    @property
    def ready_at(self) -> datetime:
        return self.next_attempt_at or self.due_at


class SQLiteSimulationStore:
    """Durable local state with atomic event claiming and completion.

    A CLAIMED event is owned by the exact ``claim_token`` returned from
    ``claim_due_events``. Recovery deliberately destroys that token. Any worker
    that finishes after recovery/re-claim must therefore fail its write instead
    of committing output into a newer generation attempt.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS simulation_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scheduled_events (
                    event_id TEXT PRIMARY KEY,
                    twin_person_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    generation_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    next_attempt_at TEXT,
                    claim_token TEXT,
                    claimed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_due
                    ON scheduled_events(status, due_at);
                CREATE TABLE IF NOT EXISTS simulation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    twin_person_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sender_person_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sim_messages_conversation
                    ON simulation_messages(twin_person_id, platform, conversation_id, timestamp);
                """
            )
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(scheduled_events)").fetchall()
            }
            migrations = {
                "generation_attempts": "INTEGER NOT NULL DEFAULT 0",
                "last_error": "TEXT",
                "next_attempt_at": "TEXT",
                "claim_token": "TEXT",
                "claimed_at": "TEXT",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    db.execute(
                        f"ALTER TABLE scheduled_events ADD COLUMN {name} {declaration}"
                    )

    def set_last_processed(self, timestamp: datetime) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO simulation_state(key,value) VALUES('last_processed',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (timestamp.isoformat(),),
            )

    def get_last_processed(self) -> datetime | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT value FROM simulation_state WHERE key='last_processed'"
            ).fetchone()
        return datetime.fromisoformat(row["value"]) if row else None

    def schedule(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        action: Action,
        due_at: datetime,
        created_at: datetime,
        replace_same_action: bool = True,
    ) -> ScheduledEvent:
        if action == Action.WAIT:
            raise ValueError("WAIT is not persisted as a future event")
        event_id = uuid.uuid4().hex
        with self._connect() as db:
            if replace_same_action:
                placeholders = ",".join("?" for _ in _CANCELLABLE_STATUSES)
                db.execute(
                    "UPDATE scheduled_events SET status='CANCELLED' "
                    "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
                    f"AND action=? AND status IN ({placeholders})",
                    (
                        twin_person_id,
                        platform,
                        conversation_id,
                        action.value,
                        *_CANCELLABLE_STATUSES,
                    ),
                )
            db.execute(
                "INSERT INTO scheduled_events("
                "event_id,twin_person_id,platform,conversation_id,action,due_at,created_at,"
                "status,generation_attempts,last_error,next_attempt_at,claim_token,claimed_at"
                ") VALUES(?,?,?,?,?,?,?,'PENDING',0,NULL,NULL,NULL,NULL)",
                (
                    event_id,
                    twin_person_id,
                    platform,
                    conversation_id,
                    action.value,
                    due_at.isoformat(),
                    created_at.isoformat(),
                ),
            )
        return ScheduledEvent(
            event_id=event_id,
            twin_person_id=twin_person_id,
            platform=platform,
            conversation_id=conversation_id,
            action=action,
            due_at=due_at,
            created_at=created_at,
            status="PENDING",
        )

    def event(self, event_id: str) -> ScheduledEvent | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(row) if row else None

    def active_events(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        action: Action | None = None,
    ) -> list[ScheduledEvent]:
        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        query = (
            "SELECT * FROM scheduled_events WHERE twin_person_id=? AND platform=? "
            f"AND conversation_id=? AND status IN ({placeholders})"
        )
        params: list[object] = [
            twin_person_id,
            platform,
            conversation_id,
            *_ACTIVE_STATUSES,
        ]
        if action is not None:
            query += " AND action=?"
            params.append(action.value)
        query += " ORDER BY COALESCE(next_attempt_at,due_at),event_id"
        with self._connect() as db:
            rows = db.execute(query, tuple(params)).fetchall()
        return [self._scheduled_from_row(row) for row in rows]

    def cancel_pending(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        action: Action | None = None,
        include_claimed: bool = False,
    ) -> int:
        statuses = _ACTIVE_STATUSES if include_claimed else _CANCELLABLE_STATUSES
        placeholders = ",".join("?" for _ in statuses)
        query = (
            "UPDATE scheduled_events SET status='CANCELLED',claim_token=NULL,claimed_at=NULL "
            "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
            f"AND status IN ({placeholders})"
        )
        params: list[object] = [twin_person_id, platform, conversation_id, *statuses]
        if action is not None:
            query += " AND action=?"
            params.append(action.value)
        with self._connect() as db:
            cursor = db.execute(query, tuple(params))
            return int(cursor.rowcount)

    def postpone_pending_event(self, event_id: str, *, not_before: datetime) -> ScheduledEvent:
        """Extend an unclaimed behavior while a user input burst is still forming."""

        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
            if row is None or row["status"] != "PENDING":
                raise KeyError(f"pending event not found: {event_id}")
            current_due = datetime.fromisoformat(row["due_at"])
            if not_before > current_due:
                db.execute(
                    "UPDATE scheduled_events SET due_at=? WHERE event_id=? AND status='PENDING'",
                    (not_before.isoformat(), event_id),
                )
            updated = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(updated)

    def due_events(self, now: datetime) -> list[ScheduledEvent]:
        placeholders = ",".join("?" for _ in _DUE_STATUSES)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scheduled_events "
                f"WHERE status IN ({placeholders}) "
                "AND COALESCE(next_attempt_at,due_at)<=? "
                "ORDER BY COALESCE(next_attempt_at,due_at),event_id",
                (*_DUE_STATUSES, now.isoformat()),
            ).fetchall()
        return [self._scheduled_from_row(row) for row in rows]

    def claim_due_events(
        self,
        *,
        now: datetime,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        limit: int = 8,
    ) -> list[ScheduledEvent]:
        """Atomically claim due work so another window/process cannot duplicate it."""

        claimed: list[ScheduledEvent] = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT * FROM scheduled_events WHERE twin_person_id=? AND platform=? "
                "AND conversation_id=? AND status IN ('PENDING','RETRY') "
                "AND COALESCE(next_attempt_at,due_at)<=? "
                "ORDER BY COALESCE(next_attempt_at,due_at),event_id LIMIT ?",
                (
                    twin_person_id,
                    platform,
                    conversation_id,
                    now.isoformat(),
                    max(1, int(limit)),
                ),
            ).fetchall()
            for row in rows:
                token = uuid.uuid4().hex
                cursor = db.execute(
                    "UPDATE scheduled_events SET status='CLAIMED',claim_token=?,claimed_at=? "
                    "WHERE event_id=? AND status IN ('PENDING','RETRY')",
                    (token, now.isoformat(), row["event_id"]),
                )
                if cursor.rowcount != 1:
                    continue
                updated = db.execute(
                    "SELECT * FROM scheduled_events WHERE event_id=?", (row["event_id"],)
                ).fetchone()
                claimed.append(self._scheduled_from_row(updated))
        return claimed

    def recover_stale_claims(
        self,
        *,
        now: datetime,
        stale_after_seconds: float = 300.0,
    ) -> int:
        cutoff = now - timedelta(seconds=max(1.0, float(stale_after_seconds)))
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='RETRY',next_attempt_at=?,claim_token=NULL,"
                "claimed_at=NULL,last_error=COALESCE(last_error,'stale generation claim recovered') "
                "WHERE status='CLAIMED' AND claimed_at IS NOT NULL AND claimed_at<=?",
                (now.isoformat(), cutoff.isoformat()),
            )
            return int(cursor.rowcount)

    def pending_events(self) -> list[ScheduledEvent]:
        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scheduled_events "
                f"WHERE status IN ({placeholders}) "
                "ORDER BY COALESCE(next_attempt_at,due_at),event_id",
                _ACTIVE_STATUSES,
            ).fetchall()
        return [self._scheduled_from_row(row) for row in rows]

    def defer_event(
        self,
        event_id: str,
        *,
        retry_at: datetime,
        error: str,
        claim_token: str | None = None,
    ) -> ScheduledEvent:
        with self._connect() as db:
            if claim_token is None:
                cursor = db.execute(
                    "UPDATE scheduled_events SET status='RETRY',next_attempt_at=?,"
                    "generation_attempts=generation_attempts+1,last_error=?,claim_token=NULL,"
                    "claimed_at=NULL WHERE event_id=? AND status IN ('PENDING','RETRY')",
                    (retry_at.isoformat(), error[:500], event_id),
                )
            else:
                cursor = db.execute(
                    "UPDATE scheduled_events SET status='RETRY',next_attempt_at=?,"
                    "generation_attempts=generation_attempts+1,last_error=?,claim_token=NULL,"
                    "claimed_at=NULL WHERE event_id=? AND status='CLAIMED' AND claim_token=?",
                    (retry_at.isoformat(), error[:500], event_id, claim_token),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"retryable scheduled event not owned by caller: {event_id}")
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(row)

    def block_event(
        self,
        event_id: str,
        *,
        error: str,
        claim_token: str | None = None,
    ) -> ScheduledEvent:
        with self._connect() as db:
            if claim_token is None:
                cursor = db.execute(
                    "UPDATE scheduled_events SET status='BLOCKED',next_attempt_at=NULL,"
                    "generation_attempts=generation_attempts+1,last_error=?,claim_token=NULL,"
                    "claimed_at=NULL WHERE event_id=? AND status IN ('PENDING','RETRY')",
                    (error[:500], event_id),
                )
            else:
                cursor = db.execute(
                    "UPDATE scheduled_events SET status='BLOCKED',next_attempt_at=NULL,"
                    "generation_attempts=generation_attempts+1,last_error=?,claim_token=NULL,"
                    "claimed_at=NULL WHERE event_id=? AND status='CLAIMED' AND claim_token=?",
                    (error[:500], event_id, claim_token),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"active scheduled event not owned by caller: {event_id}")
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(row)

    def retry_blocked_event(self, event_id: str, *, retry_at: datetime) -> ScheduledEvent:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='RETRY',next_attempt_at=?,last_error=NULL,"
                "claim_token=NULL,claimed_at=NULL WHERE event_id=? AND status='BLOCKED'",
                (retry_at.isoformat(), event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"blocked scheduled event not found: {event_id}")
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(row)

    def mark_processed(self, event_id: str, *, claim_token: str | None = None) -> None:
        with self._connect() as db:
            if claim_token is None:
                cursor = db.execute(
                    "UPDATE scheduled_events SET status='PROCESSED',last_error=NULL,"
                    "next_attempt_at=NULL,claim_token=NULL,claimed_at=NULL "
                    "WHERE event_id=? AND status IN ('PENDING','RETRY')",
                    (event_id,),
                )
            else:
                cursor = db.execute(
                    "UPDATE scheduled_events SET status='PROCESSED',last_error=NULL,"
                    "next_attempt_at=NULL,claim_token=NULL,claimed_at=NULL "
                    "WHERE event_id=? AND status='CLAIMED' AND claim_token=?",
                    (event_id, claim_token),
                )
            if cursor.rowcount != 1:
                raise KeyError(f"active scheduled event not owned by caller: {event_id}")

    def complete_claimed_event(self, event_id: str, *, claim_token: str) -> None:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='PROCESSED',last_error=NULL,"
                "next_attempt_at=NULL,claim_token=NULL,claimed_at=NULL "
                "WHERE event_id=? AND status='CLAIMED' AND claim_token=?",
                (event_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"claimed event not owned by caller: {event_id}")

    def append_simulation_messages(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        sender_person_id: str,
        messages: Iterable[tuple[datetime, str]],
        metadata: dict[str, object] | None = None,
    ) -> None:
        encoded_metadata = json.dumps(metadata or {}, ensure_ascii=False)
        rows = [
            (
                twin_person_id,
                platform,
                conversation_id,
                timestamp.isoformat(),
                sender_person_id,
                text,
                MessageType.TEXT.value,
                encoded_metadata,
            )
            for timestamp, text in messages
        ]
        if not rows:
            return
        with self._connect() as db:
            db.executemany(
                "INSERT INTO simulation_messages("
                "twin_person_id,platform,conversation_id,timestamp,sender_person_id,text,"
                "message_type,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                rows,
            )

    def complete_claimed_event_with_messages(
        self,
        *,
        event_id: str,
        claim_token: str,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        sender_person_id: str,
        messages: Iterable[tuple[datetime, str]],
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Insert generated output and complete the exact owned claim atomically."""

        rows = list(messages)
        encoded_metadata = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT status,claim_token FROM scheduled_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if (
                state is None
                or state["status"] != "CLAIMED"
                or state["claim_token"] != claim_token
            ):
                raise KeyError(f"claimed event not owned by caller: {event_id}")
            if rows:
                db.executemany(
                    "INSERT INTO simulation_messages("
                    "twin_person_id,platform,conversation_id,timestamp,sender_person_id,text,"
                    "message_type,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                    [
                        (
                            twin_person_id,
                            platform,
                            conversation_id,
                            timestamp.isoformat(),
                            sender_person_id,
                            text,
                            MessageType.TEXT.value,
                            encoded_metadata,
                        )
                        for timestamp, text in rows
                    ],
                )
            cursor = db.execute(
                "UPDATE scheduled_events SET status='PROCESSED',last_error=NULL,"
                "next_attempt_at=NULL,claim_token=NULL,claimed_at=NULL "
                "WHERE event_id=? AND status='CLAIMED' AND claim_token=?",
                (event_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("event claim changed during atomic completion")

    @staticmethod
    def _mark_messages_read_in_db(
        db: sqlite3.Connection,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        sender_person_id: str,
        read_at: datetime,
        sent_before_or_at: datetime | None,
        source: str,
    ) -> int:
        cutoff = sent_before_or_at or read_at
        rows = db.execute(
            "SELECT id,metadata_json FROM simulation_messages "
            "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
            "AND sender_person_id=? AND timestamp<=? ORDER BY timestamp,id",
            (
                twin_person_id,
                platform,
                conversation_id,
                sender_person_id,
                cutoff.isoformat(),
            ),
        ).fetchall()
        updates: list[tuple[str, int]] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if metadata.get("read_at"):
                continue
            metadata["read_at"] = read_at.isoformat()
            metadata["read_status"] = "READ"
            metadata["read_receipt_source"] = source
            updates.append((json.dumps(metadata, ensure_ascii=False), int(row["id"])))
        if updates:
            db.executemany(
                "UPDATE simulation_messages SET metadata_json=? WHERE id=?", updates
            )
        return len(updates)

    def complete_claimed_read_event(
        self,
        *,
        event_id: str,
        claim_token: str,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        sender_person_id: str,
        read_at: datetime,
        sent_before_or_at: datetime | None = None,
        source: str = "SIMULATION_LATENT_READ",
    ) -> int:
        """Apply latent READ state and complete the exact claim atomically."""

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute(
                "SELECT status,claim_token FROM scheduled_events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if (
                state is None
                or state["status"] != "CLAIMED"
                or state["claim_token"] != claim_token
            ):
                raise KeyError(f"claimed read event not owned by caller: {event_id}")
            updated = self._mark_messages_read_in_db(
                db,
                twin_person_id=twin_person_id,
                platform=platform,
                conversation_id=conversation_id,
                sender_person_id=sender_person_id,
                read_at=read_at,
                sent_before_or_at=sent_before_or_at,
                source=source,
            )
            cursor = db.execute(
                "UPDATE scheduled_events SET status='PROCESSED',last_error=NULL,"
                "next_attempt_at=NULL,claim_token=NULL,claimed_at=NULL "
                "WHERE event_id=? AND status='CLAIMED' AND claim_token=?",
                (event_id, claim_token),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("read event claim changed during atomic completion")
            return updated

    def mark_messages_read(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        sender_person_id: str,
        read_at: datetime,
        sent_before_or_at: datetime | None = None,
        source: str = "SIMULATION_LATENT_READ",
    ) -> int:
        with self._connect() as db:
            return self._mark_messages_read_in_db(
                db,
                twin_person_id=twin_person_id,
                platform=platform,
                conversation_id=conversation_id,
                sender_person_id=sender_person_id,
                read_at=read_at,
                sent_before_or_at=sent_before_or_at,
                source=source,
            )

    def backfill_read_receipts(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        user_sender_person_id: str = "self",
    ) -> int:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,timestamp,sender_person_id,metadata_json FROM simulation_messages "
                "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
                "ORDER BY timestamp,id",
                (twin_person_id, platform, conversation_id),
            ).fetchall()
            unread_ids: list[int] = []
            updates: list[tuple[str, int]] = []
            metadata_by_id: dict[int, dict[str, object]] = {}
            for row in rows:
                row_id = int(row["id"])
                metadata = json.loads(row["metadata_json"] or "{}")
                metadata_by_id[row_id] = metadata
                if row["sender_person_id"] == user_sender_person_id:
                    if not metadata.get("read_at"):
                        unread_ids.append(row_id)
                    continue
                if row["sender_person_id"] != twin_person_id or not unread_ids:
                    continue
                read_at = datetime.fromisoformat(row["timestamp"])
                for unread_id in unread_ids:
                    pending = metadata_by_id[unread_id]
                    pending["read_at"] = read_at.isoformat()
                    pending["read_status"] = "READ"
                    pending["read_receipt_source"] = "SIMULATION_BACKFILL"
                    updates.append((json.dumps(pending, ensure_ascii=False), unread_id))
                unread_ids.clear()
            if updates:
                db.executemany(
                    "UPDATE simulation_messages SET metadata_json=? WHERE id=?", updates
                )
        return len(updates)

    def simulation_messages(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
    ) -> list[ChatMessage]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM simulation_messages WHERE twin_person_id=? AND platform=? "
                "AND conversation_id=? ORDER BY timestamp,id",
                (twin_person_id, platform, conversation_id),
            ).fetchall()
        return [
            ChatMessage(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                sender=row["sender_person_id"],
                text=row["text"],
                source=MemorySource.SIMULATION,
                message_type=MessageType(row["message_type"]),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def clear_conversation(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
    ) -> None:
        with self._connect() as db:
            placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
            db.execute(
                "UPDATE scheduled_events SET status='CANCELLED',claim_token=NULL,claimed_at=NULL "
                "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
                f"AND status IN ({placeholders})",
                (twin_person_id, platform, conversation_id, *_ACTIVE_STATUSES),
            )
            db.execute(
                "DELETE FROM simulation_messages WHERE twin_person_id=? AND platform=? "
                "AND conversation_id=?",
                (twin_person_id, platform, conversation_id),
            )

    @staticmethod
    def _scheduled_from_row(row: sqlite3.Row) -> ScheduledEvent:
        keys = set(row.keys())
        return ScheduledEvent(
            event_id=row["event_id"],
            twin_person_id=row["twin_person_id"],
            platform=row["platform"],
            conversation_id=row["conversation_id"],
            action=Action(row["action"]),
            due_at=datetime.fromisoformat(row["due_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            status=row["status"],
            generation_attempts=int(row["generation_attempts"] or 0),
            last_error=row["last_error"],
            next_attempt_at=(
                datetime.fromisoformat(row["next_attempt_at"])
                if "next_attempt_at" in keys and row["next_attempt_at"]
                else None
            ),
            claim_token=(row["claim_token"] if "claim_token" in keys else None),
            claimed_at=(
                datetime.fromisoformat(row["claimed_at"])
                if "claimed_at" in keys and row["claimed_at"]
                else None
            ),
        )
