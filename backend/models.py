from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MemorySource(StrEnum):
    REAL = "REAL"
    SIMULATION = "SIMULATION"


class MessageType(StrEnum):
    TEXT = "TEXT"
    SYSTEM = "SYSTEM"
    MEDIA = "MEDIA"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    timestamp: datetime
    sender: str
    text: str
    source: MemorySource = MemorySource.REAL
    message_type: MessageType = MessageType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "text": self.text,
            "source": self.source.value,
            "message_type": self.message_type.value,
            "metadata": self.metadata,
        }
