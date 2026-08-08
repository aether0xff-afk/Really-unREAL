import pytest

from backend.identity import IdentityMap, PersonEntity
from backend.twin import TwinMode, resolve_twin_spec


def _identity_map() -> IdentityMap:
    return IdentityMap(
        (
            PersonEntity("self", {"kakao": ("나",)}, is_self=True),
            PersonEntity("friend", {"kakao": ("친구",)}),
        )
    )


def test_resolves_self_twin_from_identity_map() -> None:
    spec = resolve_twin_spec(_identity_map(), mode=TwinMode.SELF)

    assert spec.target_person_id == "self"
    assert spec.self_person_id == "self"
    assert spec.is_self_twin is True


def test_person_twin_requires_another_person() -> None:
    spec = resolve_twin_spec(_identity_map(), mode=TwinMode.PERSON, person_id="friend")

    assert spec.target_person_id == "friend"
    assert spec.is_self_twin is False

    with pytest.raises(ValueError):
        resolve_twin_spec(_identity_map(), mode=TwinMode.PERSON, person_id="self")
