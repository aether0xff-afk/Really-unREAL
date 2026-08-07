import pytest

from backend.identity import (
    IdentityMap,
    PersonEntity,
    normalize_alias,
    suggest_identity_matches,
)


def test_normalize_alias_only_removes_safe_presentation_differences() -> None:
    assert normalize_alias(" @Ch_R1-S_ ") == "chr1s"
    assert normalize_alias("임 명민") == "임명민"


def test_exact_name_is_safe_but_fuzzy_name_requires_review() -> None:
    candidates = suggest_identity_matches(
        ["이은세", "임명민"],
        ["이은세", "명민"],
        minimum_score=0.5,
    )

    exact = next(
        candidate
        for candidate in candidates
        if candidate.kakao_alias == "이은세" and candidate.instagram_alias == "이은세"
    )
    fuzzy = next(
        candidate
        for candidate in candidates
        if candidate.kakao_alias == "임명민" and candidate.instagram_alias == "명민"
    )

    assert exact.safe_auto_match is True
    assert exact.score == 1.0
    assert fuzzy.safe_auto_match is False
    assert fuzzy.score < 1.0


def test_identity_map_rejects_alias_collision() -> None:
    with pytest.raises(ValueError, match="mapped to both"):
        IdentityMap(
            [
                PersonEntity("person-a", {"kakao": ("같은 이름",)}),
                PersonEntity("person-b", {"kakao": ("같은이름",)}),
            ]
        )
