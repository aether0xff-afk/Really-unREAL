from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime

from backend.fusion import EvidenceMessage
from backend.replay import ReplayCase
from backend.simulation.action_policy import Action


def _observable_action(case: ReplayCase) -> Action:
    if not case.context:
        raise ValueError("ReplayCase has no visible context")
    return (
        Action.INITIATE
        if case.context[-1].sender_person_id == case.person_id
        else Action.REPLY
    )


class EmpiricalTimingSampler:
    """Sample live delays from observed replay timing intervals.

    This remains the unconditional fallback. The optional live-context arguments
    are accepted so the sampler satisfies the same protocol as the richer 1.1
    contextual timing model without changing its historical sampling behavior.
    """

    def __init__(
        self,
        cases: Iterable[ReplayCase],
        *,
        seed: int | None = None,
    ) -> None:
        cases = list(cases)
        if not cases:
            raise ValueError("cannot build timing sampler without replay cases")

        self._rng = random.Random(seed)
        self._conversation: dict[tuple[str, Action], list[ReplayCase]] = defaultdict(list)
        self._platform: dict[tuple[str, Action], list[ReplayCase]] = defaultdict(list)
        self._action: dict[Action, list[ReplayCase]] = defaultdict(list)

        for case in cases:
            if case.action_is_ambiguous or not case.context:
                continue
            action = _observable_action(case)
            self._conversation[(case.conversation_id, action)].append(case)
            self._platform[(case.platform, action)].append(case)
            self._action[action].append(case)

    def _choose_case(self, cases: Sequence[ReplayCase]) -> ReplayCase:
        weights = [max(0.0, float(case.evidence_weight)) for case in cases]
        if not any(weights):
            weights = [1.0] * len(cases)
        return self._rng.choices(list(cases), weights=weights, k=1)[0]

    def _sample_interval(self, case: ReplayCase) -> float:
        lower = max(0.0, float(case.delay_lower_seconds))
        upper = max(lower, float(case.delay_upper_seconds))
        if upper == lower:
            return lower
        return self._rng.uniform(lower, upper)

    def sample_delay_seconds(
        self,
        *,
        platform: str,
        conversation_id: str,
        action: Action,
        observed_at: datetime | None = None,
        visible_context: Sequence[EvidenceMessage] = (),
    ) -> float | None:
        _ = observed_at, visible_context
        if action == Action.WAIT:
            return None

        candidates = self._conversation.get((conversation_id, action))
        if not candidates:
            candidates = self._platform.get((platform, action))
        if not candidates:
            candidates = self._action.get(action)
        if not candidates:
            return None

        return self._sample_interval(self._choose_case(candidates))
