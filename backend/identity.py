from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


_ALIAS_PUNCT = re.compile(r"[^0-9a-zA-Z가-힣]+")


def normalize_alias(value: str) -> str:
    """Normalize a display name/handle for conservative identity comparison.

    This deliberately does not transliterate Korean names or automatically infer
    that two fuzzy aliases are the same person. It only removes presentation
    differences that are safe to ignore: Unicode normalization, case, leading
    ``@``, whitespace, and punctuation.
    """

    value = unicodedata.normalize("NFC", value).strip().lower()
    if value.startswith("@"):
        value = value[1:]
    return _ALIAS_PUNCT.sub("", value)


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    kakao_alias: str
    instagram_alias: str
    score: float
    reasons: tuple[str, ...]
    safe_auto_match: bool


@dataclass(frozen=True, slots=True)
class PersonEntity:
    person_id: str
    aliases: dict[str, tuple[str, ...]]
    is_self: bool = False

    def aliases_for(self, platform: str) -> tuple[str, ...]:
        return self.aliases.get(platform, ())


class IdentityMap:
    """Explicit mapping from platform aliases to stable local person IDs.

    The map is intentionally local configuration. Fuzzy candidate generation is
    separate from resolution so a plausible-looking match never silently mixes
    two real people into one persona.
    """

    def __init__(self, people: Iterable[PersonEntity] = ()) -> None:
        self.people = tuple(people)
        self._index: dict[tuple[str, str], str] = {}
        self._by_id: dict[str, PersonEntity] = {}
        for person in self.people:
            if person.person_id in self._by_id:
                raise ValueError(f"duplicate person_id: {person.person_id}")
            self._by_id[person.person_id] = person
            for platform, aliases in person.aliases.items():
                for alias in aliases:
                    key = (platform, normalize_alias(alias))
                    if not key[1]:
                        continue
                    previous = self._index.get(key)
                    if previous is not None and previous != person.person_id:
                        raise ValueError(
                            f"alias {alias!r} on {platform!r} is mapped to both "
                            f"{previous!r} and {person.person_id!r}"
                        )
                    self._index[key] = person.person_id

    def resolve(self, platform: str, alias: str) -> str | None:
        return self._index.get((platform, normalize_alias(alias)))

    def get(self, person_id: str) -> PersonEntity:
        try:
            return self._by_id[person_id]
        except KeyError as exc:
            raise KeyError(f"unknown person_id: {person_id}") from exc

    @property
    def self_person_id(self) -> str | None:
        self_ids = [person.person_id for person in self.people if person.is_self]
        if len(self_ids) > 1:
            raise ValueError("identity map contains more than one self person")
        return self_ids[0] if self_ids else None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "IdentityMap":
        raw_people = data.get("people", [])
        if not isinstance(raw_people, list):
            raise ValueError("identity map 'people' must be a list")
        people: list[PersonEntity] = []
        for raw in raw_people:
            if not isinstance(raw, dict):
                raise ValueError("each identity map person must be an object")
            raw_aliases = raw.get("aliases", {})
            if not isinstance(raw_aliases, dict):
                raise ValueError("person aliases must be an object")
            aliases = {
                str(platform): tuple(str(value) for value in values)
                for platform, values in raw_aliases.items()
                if isinstance(values, list)
            }
            people.append(
                PersonEntity(
                    person_id=str(raw["person_id"]),
                    aliases=aliases,
                    is_self=bool(raw.get("is_self", False)),
                )
            )
        return cls(people)

    @classmethod
    def from_json(cls, path: str | Path) -> "IdentityMap":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def suggest_identity_matches(
    kakao_aliases: Iterable[str],
    instagram_aliases: Iterable[str],
    *,
    minimum_score: float = 0.55,
) -> list[IdentityCandidate]:
    """Rank cross-platform identity candidates without applying them.

    Exact normalized display-name matches are marked ``safe_auto_match``.
    Everything else is only a suggestion for human review, even when the fuzzy
    score is high.
    """

    candidates: list[IdentityCandidate] = []
    for kakao in sorted(set(kakao_aliases)):
        nk = normalize_alias(kakao)
        if not nk:
            continue
        for instagram in sorted(set(instagram_aliases)):
            ni = normalize_alias(instagram)
            if not ni:
                continue

            reasons: list[str] = []
            if nk == ni:
                score = 1.0
                reasons.append("normalized display names are identical")
                safe = True
            else:
                ratio = SequenceMatcher(None, nk, ni).ratio()
                containment = min(len(nk), len(ni)) / max(len(nk), len(ni)) if (nk in ni or ni in nk) else 0.0
                score = max(ratio, containment * 0.9)
                if ratio >= 0.75:
                    reasons.append("display names are textually similar")
                if containment >= 0.75:
                    reasons.append("one normalized alias mostly contains the other")
                safe = False

            if score >= minimum_score:
                candidates.append(
                    IdentityCandidate(
                        kakao_alias=kakao,
                        instagram_alias=instagram,
                        score=round(score, 4),
                        reasons=tuple(reasons),
                        safe_auto_match=safe,
                    )
                )

    candidates.sort(
        key=lambda item: (
            not item.safe_auto_match,
            -item.score,
            item.kakao_alias,
            item.instagram_alias,
        )
    )
    return candidates
