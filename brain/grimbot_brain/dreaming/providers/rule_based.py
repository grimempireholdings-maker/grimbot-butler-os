from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Episode:
    episode_id: int
    content: str
    kind: str
    importance: float
    created_at: str
    anchor: bool
    room_id: int | None
    zone_id: int | None
    room_name: str | None
    zone_name: str | None


@dataclass(frozen=True)
class FactCandidate:
    content: str
    confidence: float
    importance: float
    frequency: int
    tags: tuple[str, ...]
    room_id: int | None
    zone_id: int | None


class RuleBasedProvider:
    minimum_cluster_size = 2

    def consolidate(self, episodes: list[Episode]) -> list[FactCandidate]:
        clusters: dict[tuple[object, ...], list[Episode]] = defaultdict(list)
        for episode in episodes:
            clusters[self._cluster_key(episode)].append(episode)

        candidates = []
        for cluster in clusters.values():
            if len(cluster) < self.minimum_cluster_size:
                continue
            representative = cluster[0]
            tags = self._tags(representative)
            location = f" in {representative.room_name}" if representative.room_name else ""
            content = f"Repeated {representative.kind}{location}: {representative.content.strip()}"
            candidates.append(
                FactCandidate(
                    content=content[:2000],
                    confidence=min(0.95, 0.55 + len(cluster) * 0.08),
                    importance=max(item.importance for item in cluster),
                    frequency=len(cluster),
                    tags=tuple(tags),
                    room_id=representative.room_id,
                    zone_id=representative.zone_id,
                )
            )
        return sorted(candidates, key=lambda item: (-item.frequency, item.content.lower()))

    def contradictions(self, candidates: list[FactCandidate]) -> int:
        positive: set[str] = set()
        negative: set[str] = set()
        for candidate in candidates:
            normalized = _normalize(candidate.content)
            stripped = re.sub(r"\b(?:not|never|no)\b", "", normalized).strip()
            if re.search(r"\b(?:not|never|no)\b", normalized):
                negative.add(stripped)
            else:
                positive.add(stripped)
        return len(positive & negative)

    def _cluster_key(self, episode: Episode) -> tuple[object, ...]:
        tags = self._tags(episode)
        return (
            episode.room_id,
            episode.zone_id,
            episode.kind.lower(),
            _normalize(episode.content),
            tuple(tags),
        )

    def _tags(self, episode: Episode) -> list[str]:
        text = episode.content.lower()
        tags = [f"kind:{episode.kind.lower()}"]
        if episode.room_name:
            tags.append(f"room:{_normalize(episode.room_name)}")
        if episode.zone_name:
            tags.append(f"zone:{_normalize(episode.zone_name)}")
        if any(word in text for word in ("hazard", "unsafe", "danger", "spill", "cord", "obstacle")):
            tags.extend(["hazard", "safety"])
        for action in ("clean", "clear", "organize", "scan", "move", "stop"):
            if action in text:
                tags.append(f"action:{action}")
        for outcome in ("success", "completed", "failed", "blocked", "ignored"):
            if outcome in text:
                tags.append(f"outcome:{outcome}")
        for object_name in ("cord", "cable", "notebook", "dish", "drink", "laundry", "table", "desk"):
            if object_name in text:
                tags.append(f"object:{object_name}")
        return sorted(set(tags))


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
