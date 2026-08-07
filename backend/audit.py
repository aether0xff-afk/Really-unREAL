from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from backend.ingest.archive import ConversationExport, load_kakao_archive


@dataclass(frozen=True, slots=True)
class ConversationAudit:
    chat_name: str
    message_count: int
    participant_counts: dict[str, int]
    participant_count: int
    conversation_kind: str
    first_timestamp: str
    last_timestamp: str


def audit_conversation(conversation: ConversationExport) -> ConversationAudit:
    counts = Counter(message.sender for message in conversation.messages)
    first = min(message.timestamp for message in conversation.messages)
    last = max(message.timestamp for message in conversation.messages)
    participant_count = len(counts)
    return ConversationAudit(
        chat_name=conversation.chat_name,
        message_count=len(conversation.messages),
        participant_counts=dict(counts),
        participant_count=participant_count,
        conversation_kind="direct" if participant_count == 2 else "group",
        first_timestamp=first.isoformat(),
        last_timestamp=last.isoformat(),
    )


def audit_archive(path: str | Path) -> dict[str, object]:
    conversations = load_kakao_archive(path)
    audits = [audit_conversation(conversation) for conversation in conversations]
    return {
        "conversation_count": len(audits),
        "message_count": sum(audit.message_count for audit in audits),
        "direct_conversation_count": sum(audit.conversation_kind == "direct" for audit in audits),
        "group_conversation_count": sum(audit.conversation_kind == "group" for audit in audits),
        "conversations": [asdict(audit) for audit in audits],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit KakaoTalk exports without extracting attachments"
    )
    parser.add_argument("archive", help="Single chat ZIP or ZIP bundle of chat ZIPs")
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
