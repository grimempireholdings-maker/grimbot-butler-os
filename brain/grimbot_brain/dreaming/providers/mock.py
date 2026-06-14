from __future__ import annotations

from .rule_based import FactCandidate, Episode


class MockProvider:
    def consolidate(self, episodes: list[Episode]) -> list[FactCandidate]:
        if len(episodes) < 2:
            return []
        episode = episodes[0]
        return [
            FactCandidate(
                content=f"Mock candidate: {episode.content.strip()}"[:2000],
                confidence=0.6,
                importance=episode.importance,
                frequency=2,
                tags=("mock",),
                room_id=episode.room_id,
                zone_id=episode.zone_id,
            )
        ]

    def contradictions(self, candidates: list[FactCandidate]) -> int:
        return 0
