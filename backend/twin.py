from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from backend.identity import IdentityMap


class TwinMode(StrEnum):
    """Who the simulator is trying to reproduce."""

    PERSON = "person"
    SELF = "self"


@dataclass(frozen=True, slots=True)
class TwinSpec:
    """Resolved simulation target.

    SELF mode treats the identity map's single ``is_self=true`` entity as the
    behavioral target. PERSON mode reproduces another explicitly selected local
    person ID. Both modes share the same replay, persona, retrieval, timing, and
    generation machinery; only the target identity changes.
    """

    mode: TwinMode
    target_person_id: str
    self_person_id: str

    @property
    def is_self_twin(self) -> bool:
        return self.mode == TwinMode.SELF


def resolve_twin_spec(
    identity_map: IdentityMap,
    *,
    mode: TwinMode | str = TwinMode.PERSON,
    person_id: str | None = None,
) -> TwinSpec:
    mode = TwinMode(mode)
    self_person_id = identity_map.self_person_id
    if self_person_id is None:
        raise ValueError("identity map must contain exactly one is_self=true person")

    if mode == TwinMode.SELF:
        if person_id is not None and person_id != self_person_id:
            raise ValueError("SELF twin cannot target another person_id")
        target_person_id = self_person_id
    else:
        if not person_id:
            raise ValueError("PERSON twin requires person_id")
        identity_map.get(person_id)
        if person_id == self_person_id:
            raise ValueError("use mode='self' when targeting the user")
        target_person_id = person_id

    return TwinSpec(
        mode=mode,
        target_person_id=target_person_id,
        self_person_id=self_person_id,
    )
