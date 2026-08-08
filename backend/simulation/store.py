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


class SQLiteSimulationStore:
    """Durable local state for live/shadow simulation.

    Scheduled actions are persisted before generation. Generated messages are
    stored separately with source=SIMULATION and never become REAL evidence.
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
                    status TEXT NOT NULL DEFAULT 'PENDING'
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
                db.execute(
                    "UPDATE scheduled_events SET status='CANCELLED' "
                    "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
                    "AND action=? AND status='PENDING'",
                    (twin_person_id, platform, conversation_id, action.value),
                )
            db.execute(
                "INSERT INTO scheduled_events("
                "event_id,twin_person_id,platform,conversation_id,action,due_at,created_at,status"
                ") VALUES(?,?,?,?,?,?,?,'PENDING')",
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
            platform=self.platform if hasattr(self, "platform") else platform,
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
        query = (
            "UPDATE scheduled_events SET status='CANCELLED' "
            "WHERE twin_person_id=? AND platform=? AND conversation_id=? AND status='PENDING'"
        )
        params: list[object] = [twin_person_id, platform, conversation_id]
        if action is not None:
            query += " AND action=?"
            params.append(action.value)
        with self._connect() as db:
            cursor = db.execute(query, tuple(params))
            return int(cursor.rowcount)

    def due_events(self, now: datetime) -> list[ScheduledEvent]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scheduled_events WHERE status='PENDING' AND due_at<=? "
                "ORDER BY due_at,event_id",
                (now.isoformat(),),
            ).fetchall()
        return [self._scheduled_from_row(row) for row in rows]

    def pending_events(self) -> list[ScheduledEvent]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM scheduled_events WHERE status='PENDING' ORDER BY due_at,event_id"
            ).fetchall()
        return [self._scheduled_from_row(row) for row in rows]

    def mark_processed(self, event_id: str) -> None:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_events SET status='PROCESSED' "
                "WHERE event_id=? AND status='PENDING'",
                (event_id,),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"pending event not found: {event_id}")

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
        """Clear only SIMULATION state for one live conversation.

        Imported REAL evidence is not stored in this database and therefore can
        never be deleted by this operation.
        """

        with self._connect() as db:
            db.execute(
                "UPDATE scheduled_events SET status='CANCELLED' "
                "WHERE twin_person_id=? AND platform=? AND conversation_id=? "
                "AND status='PENDING'",
                (twin_person_id, platform, conversation_id),
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
        )
