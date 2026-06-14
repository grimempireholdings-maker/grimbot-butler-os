from __future__ import annotations

from difflib import SequenceMatcher

from .procedure_schemas import Procedure, ProcedureMatchRequest, ProcedureMatchResult
from .procedure_store import ProcedureStore, normalize_name

MINIMUM_FUZZY_CONFIDENCE = 0.60


class ProcedureMatcher:
    def __init__(self, store: ProcedureStore) -> None:
        self.store = store

    def match(self, request: ProcedureMatchRequest) -> ProcedureMatchResult:
        if request.procedure_id is not None:
            procedure = self.store.get(request.procedure_id, active_only=True)
            return self._result(procedure, 1.0, "procedure_id") if procedure else self._no_match()

        query = normalize_name(request.query or "")
        exact = self.store.get_by_name(query, active_only=True)
        if exact:
            return self._result(exact, 1.0, "exact_name")
        if len(query) < 3:
            return self._no_match()

        best: tuple[float, Procedure] | None = None
        for procedure in self.store.list_procedures(active_only=True):
            candidates = [procedure.name, *procedure.trigger_phrases]
            score = max(self._score(query, normalize_name(candidate)) for candidate in candidates)
            score *= procedure.procedure_confidence
            if best is None or score > best[0] or (
                score == best[0] and procedure.procedure_id < best[1].procedure_id
            ):
                best = (score, procedure)

        threshold = max(MINIMUM_FUZZY_CONFIDENCE, request.minimum_confidence)
        if best is None or best[0] < threshold:
            return self._no_match()
        return self._result(best[1], best[0], "fuzzy_trigger")

    def _score(self, query: str, candidate: str) -> float:
        if not query or not candidate:
            return 0.0
        sequence_score = SequenceMatcher(None, query, candidate).ratio()
        query_tokens = set(query.split())
        candidate_tokens = set(candidate.split())
        union = query_tokens | candidate_tokens
        token_score = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
        return max(sequence_score, token_score * 0.9)

    def _result(
        self,
        procedure: Procedure,
        confidence: float,
        match_type: str,
    ) -> ProcedureMatchResult:
        return ProcedureMatchResult(
            matched=True,
            procedure_id=procedure.procedure_id,
            name=procedure.name,
            confidence=max(0.0, min(1.0, confidence)),
            match_type=match_type,
            required_permission=procedure.required_permission,
        )

    def _no_match(self) -> ProcedureMatchResult:
        return ProcedureMatchResult(matched=False, confidence=0.0)
