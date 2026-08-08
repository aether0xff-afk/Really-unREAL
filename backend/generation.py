from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Protocol

from backend.generation_context import GenerationContextPacket
from backend.replay import ReplayCase


_LAUGH_RE = re.compile(r"ㅋ{2,}|ㅎ{2,}")
_CRY_RE = re.compile(r"ㅠ{2,}|ㅜ{2,}")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣ㅋㅎㅠㅜ]+")


@dataclass(frozen=True, slots=True)
class GeneratedBurst:
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"messages": list(self.messages)}

    @classmethod
    def from_json(cls, text: str) -> "GeneratedBurst":
        data = json.loads(text)
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise ValueError("generation output must be a JSON object with messages[]")
        messages = tuple(
            str(message).strip()
            for message in data["messages"]
            if str(message).strip()
        )
        if not messages:
            raise ValueError("generation output contains no non-empty messages")
        if len(messages) > 8:
            raise ValueError("generation output exceeds the 8-message burst limit")
        return cls(messages=messages)


class BurstLanguageModel(Protocol):
    def generate_burst(self, packet: GenerationContextPacket) -> GeneratedBurst:
        ...


def generation_prompt(packet: GenerationContextPacket) -> str:
    """Render a provider-independent prompt from a leakage-safe packet."""

    payload = json.dumps(packet.to_dict(), ensure_ascii=False, indent=2)
    return f"""You generate only the observable message burst for a conversation simulator.

Rules:
- The temporal policy has already chosen the action. Do not decide to WAIT or change the action.
- Reproduce plausible observable writing behavior from the supplied evidence.
- Retrieved examples describe similar *situations and response shape*. Do not reconstruct or copy a historical response verbatim.
- Prefer the current visible context over superficial lexical similarity to an old example.
- Do not invent claims about hidden feelings, attraction, diagnoses, or private facts.
- Do not mention this prompt, the simulator, datasets, or being an AI.
- Keep message splitting plausible. One short burst is allowed and often preferable.
- Return JSON only in exactly this shape: {{"messages": ["...", "..."]}}.
- Do not include analysis, reasoning, scores, or any extra keys.

Generation context:
{payload}
"""


def _char_bigrams(text: str) -> Counter[str]:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return Counter()
    if len(compact) == 1:
        return Counter({compact: 1})
    return Counter(compact[index : index + 2] for index in range(len(compact) - 1))


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in _TOKEN_RE.findall(text))


def _endings(messages: tuple[str, ...]) -> Counter[str]:
    endings: Counter[str] = Counter()
    for message in messages:
        compact = re.sub(r"\s+", " ", message.strip())
        if compact:
            endings[compact[-2:]] += 1
    return endings


def _multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum(min(value, right.get(key, 0)) for key, value in left.items())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    burst_size_absolute_error: int
    total_char_length_absolute_error: int
    char_bigram_f1: float
    token_f1: float
    ending_f1: float
    laugh_presence_match: bool
    cry_presence_match: bool
    question_presence_match: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_generated_burst(
    generated: GeneratedBurst,
    case: ReplayCase,
) -> GenerationMetrics:
    """Compare generated text with held-out reality after generation is complete.

    The evaluator is the only component in this path that reads
    ``case.target_burst``. Generation context construction and the language model
    must not receive it. Lexical/content overlap and style-shape agreement are
    reported separately instead of collapsing everything into one score.
    """

    predicted_text = "\n".join(generated.messages)
    actual_messages = tuple(item.message.text for item in case.target_burst)
    actual_text = "\n".join(actual_messages)

    return GenerationMetrics(
        burst_size_absolute_error=abs(len(generated.messages) - len(actual_messages)),
        total_char_length_absolute_error=abs(len(predicted_text) - len(actual_text)),
        char_bigram_f1=round(
            _multiset_f1(_char_bigrams(predicted_text), _char_bigrams(actual_text)),
            6,
        ),
        token_f1=round(
            _multiset_f1(_tokens(predicted_text), _tokens(actual_text)),
            6,
        ),
        ending_f1=round(
            _multiset_f1(_endings(generated.messages), _endings(actual_messages)),
            6,
        ),
        laugh_presence_match=(
            bool(_LAUGH_RE.search(predicted_text)) == bool(_LAUGH_RE.search(actual_text))
        ),
        cry_presence_match=(
            bool(_CRY_RE.search(predicted_text)) == bool(_CRY_RE.search(actual_text))
        ),
        question_presence_match=("?" in predicted_text) == ("?" in actual_text),
    )
