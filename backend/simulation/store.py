from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from backend.models import ChatMessage, MemorySource, MessageType
from backend.simulation.action_policy import Action


_ACTIVE_STATUSES = ("PENDING", "RETRY", "BLOCKED")
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


class SQLiteSimulationStore:
    """Durable local state for live/shadow simulation.

    Scheduled behavior exists independently from provider availability. Temporary
    generation failures move an event to RETRY; permanent/configuration failures
    move it to BLOCKED. Neither state silently erases the already-decided action.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
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
                    last_error TEXT
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
            # v1.1 migration for databases created by <=1.0.5.
            columns = {
                row["name"]
                for row in db.execute("PRAGMA table_info(scheduled_events)").fetchall()
            }
            if "generation_attempts" not in columns:
                db.execute(
                    "ALTER TABLE scheduled_events ADD COLUMN generation_attempts "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            if "last_error" not in columns:
                db.execute("ALTER TABLE scheduled_events ADD COLUMN last_error TEXT")

    def set_last_processed(self, timestamp: datetime) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO simulation_state(key, value) VALUES('last_processed', ?) "
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
                placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
                db.execute(
                    "UPDATE scheduled_events SET status='CANCELLED' "
                    "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
                    f"AND action=? AND status IN ({placeholders})",
                    (
                        twin_person_id,
                        platform,
                        conversation_id,
                        action.value,
                        *_ACTIVE_STATUSES,
                    ),
                )
            db.execute(
                "INSERT INTO scheduled_events("
                "event_id,twin_person_id,platform,conversation_id,action,due_at,created_at,"
                "status,generation_attempts,last_error"
                ") VALUES(?,?,?,?,?,?,?,'PENDING',0,NULL)",
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

    def cancel_pending(
        self,
        *,
        twin_person_id: str,
        platform: str,
        conversation_id: str,
        action: Action | None = None,
    ) -> int:
        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        query = (
            "UPDATE scheduled_events SET status='CANCELLED' "
            "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
            f"AND status IN ({placeholders})"
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
        with self._connect() as db:
            cursor = db.execute(query, tuple(params))
            return int(cursor.rowcount)

    def due_events(self, now: datetime) -> list[ScheduledEvent]:
        placeholders = ",".join("?" for _ in _DUE_STATUSES)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scheduled_events "
                f"WHERE status IN ({placeholders}) AND due_at<=? "
                "ORDER BY due_at,event_id",
                (*_DUE_STATUSES, now.isoformat()),
            ).fetchall()
        return [self._scheduled_from_row(row) for row in rows]

    def pending_events(self) -> list[ScheduledEvent]:
        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scheduled_events "
                f"WHERE status IN ({placeholders}) ORDER BY due_at,event_id",
                _ACTIVE_STATUSES,
            ).fetchall()
        return [self._scheduled_from_row(row) for row in rows]

    def defer_event(
        self,
        event_id: str,
        *,
        retry_at: datetime,
        error: str,
    ) -> ScheduledEvent:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='RETRY', due_at=?, "
                "generation_attempts=generation_attempts+1, last_error=? "
                "WHERE event_id=? AND status IN ('PENDING','RETRY')",
                (retry_at.isoformat(), error[:500], event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"retryable scheduled event not found: {event_id}")
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(row)

    def block_event(self, event_id: str, *, error: str) -> ScheduledEvent:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='BLOCKED', "
                "generation_attempts=generation_attempts+1, last_error=? "
                "WHERE event_id=? AND status IN ('PENDING','RETRY')",
                (error[:500], event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"active scheduled event not found: {event_id}")
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(row)

    def retry_blocked_event(
        self,
        event_id: str,
        *,
        retry_at: datetime,
    ) -> ScheduledEvent:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='RETRY', due_at=?, last_error=NULL "
                "WHERE event_id=? AND status='BLOCKED'",
                (retry_at.isoformat(), event_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"blocked scheduled event not found: {event_id}")
            row = db.execute(
                "SELECT * FROM scheduled_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return self._scheduled_from_row(row)

    def mark_processed(self, event_id: str) -> None:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='PROCESSED', last_error=NULL "
                "WHERE event_id=? AND status IN ('PENDING','RETRY')",
                (event_id,),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"due scheduled event not found: {event_id}")

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
        placeholders = ",".join("?" for _ in _ACTIVE_STATUSES)
        with self._connect() as db:
            db.execute(
                "UPDATE scheduled_events SET status='CANCELLED' "
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
        )
