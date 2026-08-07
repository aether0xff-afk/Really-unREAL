from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


_ALIAS_PUNCT = re.compile(r"[^0-9a-zA-Z가-힣]+")


def normalize_alias(value: str) -> str:
    """Normalize a display name/handle for conservative identity comparison."""

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
    """Explicit mapping from platform aliases to stable local person IDs."""

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

    def to_dict(self) -> dict[str, object]:
        return {
            "people": [
                {
                    "person_id": person.person_id,
                    "is_self": person.is_self,
                    "aliases": {
                        platform: list(aliases)
                        for platform, aliases in person.aliases.items()
                    },
                }
                for person in self.people
            ]
        }

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
    """Rank cross-platform candidates without applying fuzzy matches."""

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
                containment = (
                    min(len(nk), len(ni)) / max(len(nk), len(ni))
                    if (nk in ni or ni in nk)
                    else 0.0
                )
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


def _stable_person_id(*parts: str) -> str:
    key = "|".join(parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"person-{digest}"


def build_identity_skeleton(
    kakao_aliases: Iterable[str],
    instagram_aliases: Iterable[str],
    candidates: Iterable[IdentityCandidate],
    *,
    self_kakao_alias: str | None = None,
    self_instagram_alias: str | None = None,
) -> IdentityMap:
    """Build a lossless local map using only safe merges.

    Exact cross-platform matches are merged. Every unmatched alias still gets a
    standalone person entity, so no data disappears. Fuzzy review candidates
    remain separate until the user explicitly edits/merges the local map.

    When both self aliases are supplied they are explicitly trusted and merged
    into the stable ``self`` entity even if their display strings differ.
    """

    kakao = {alias for alias in kakao_aliases if normalize_alias(alias)}
    instagram = {alias for alias in instagram_aliases if normalize_alias(alias)}
    consumed_kakao: set[str] = set()
    consumed_instagram: set[str] = set()
    people: list[PersonEntity] = []

    self_k = normalize_alias(self_kakao_alias or "")
    self_i = normalize_alias(self_instagram_alias or "")
    if bool(self_k) != bool(self_i):
        raise ValueError("both self Kakao and Instagram aliases are required")
    if self_k and self_i:
        kakao_self = next((alias for alias in kakao if normalize_alias(alias) == self_k), None)
        instagram_self = next(
            (alias for alias in instagram if normalize_alias(alias) == self_i), None
        )
        if kakao_self is None or instagram_self is None:
            raise ValueError("provided self aliases were not found in both exports")
        people.append(
            PersonEntity(
                person_id="self",
                aliases={"kakao": (kakao_self,), "instagram": (instagram_self,)},
                is_self=True,
            )
        )
        consumed_kakao.add(kakao_self)
        consumed_instagram.add(instagram_self)

    for candidate in candidates:
        if not candidate.safe_auto_match:
            continue
        if candidate.kakao_alias in consumed_kakao or candidate.instagram_alias in consumed_instagram:
            continue
        people.append(
            PersonEntity(
                person_id=_stable_person_id(
                    "pair",
                    normalize_alias(candidate.kakao_alias),
                    normalize_alias(candidate.instagram_alias),
                ),
                aliases={
                    "kakao": (candidate.kakao_alias,),
                    "instagram": (candidate.instagram_alias,),
                },
            )
        )
        consumed_kakao.add(candidate.kakao_alias)
        consumed_instagram.add(candidate.instagram_alias)

    for alias in sorted(kakao - consumed_kakao):
        people.append(
            PersonEntity(
                person_id=_stable_person_id("kakao", normalize_alias(alias)),
                aliases={"kakao": (alias,)},
            )
        )
    for alias in sorted(instagram - consumed_instagram):
        people.append(
            PersonEntity(
                person_id=_stable_person_id("instagram", normalize_alias(alias)),
                aliases={"instagram": (alias,)},
            )
        )

    return IdentityMap(people)


def build_safe_identity_map(
    candidates: Iterable[IdentityCandidate],
    *,
    self_kakao_alias: str | None = None,
    self_instagram_alias: str | None = None,
) -> IdentityMap:
    """Backward-compatible exact-pair-only map builder.

    Prefer ``build_identity_skeleton`` for real imports because it preserves
    unmatched platform-only people instead of omitting them.
    """

    safe = [candidate for candidate in candidates if candidate.safe_auto_match]
    kakao_aliases = [candidate.kakao_alias for candidate in safe]
    instagram_aliases = [candidate.instagram_alias for candidate in safe]
    return build_identity_skeleton(
        kakao_aliases,
        instagram_aliases,
        safe,
        self_kakao_alias=self_kakao_alias,
        self_instagram_alias=self_instagram_alias,
    )
