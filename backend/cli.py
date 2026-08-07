from __future__ import annotations

import argparse
import json
from collections import Counter

from backend.ingest.kakao import parse_kakao_file
from backend.persona.language import build_language_profile
from backend.persona.temporal import build_temporal_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a KakaoTalk text export")
    parser.add_argument("path", help="Path to exported KakaoTalk .txt file")
    parser.add_argument("--target", help="Participant whose observable persona to profile")
    parser.add_argument(
        "--session-gap-hours",
        type=float,
        default=6.0,
        help="Gap that starts a new conversation session (default: 6h)",
    )
    args = parser.parse_args()

    messages = parse_kakao_file(args.path)
    if not messages:
        raise SystemExit("No KakaoTalk messages were recognized.")

    participants = Counter(message.sender for message in messages)
    if args.target is None:
        print(
            json.dumps(
                {
                    "message_count": len(messages),
                    "participants": participants,
                    "first_timestamp": messages[0].timestamp.isoformat(),
                    "last_timestamp": messages[-1].timestamp.isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output = {
        "language": build_language_profile(messages, args.target).to_dict(),
        "temporal": build_temporal_profile(
            messages,
            args.target,
            session_gap_hours=args.session_gap_hours,
        ).to_dict(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
