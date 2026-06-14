from __future__ import annotations

import json
import re
import sqlite3

from ..memory import BrainMemory
from .procedure_schemas import (
    ExecutionStatus,
    PendingProcedure,
    PendingProcedureCreate,
    PendingProcedureReview,
    Procedure,
    ProcedureCreate,
    ProcedureExecution,
    ProcedureStats,
    ProcedureUpdate,
)


class ProcedureStore:
    def __init__(self, memory: BrainMemory) -> None:
        self.memory = memory

    def create(self, request: ProcedureCreate) -> Procedure:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            procedure = self._create_in_transaction(connection, request)
            connection.commit()
            return procedure

    def list_procedures(self, active_only: bool = True) -> list[Procedure]:
        with self._connect() as connection:
            where = "WHERE procedures.status = 'active'" if active_only else ""
            rows = connection.execute(
                f"""
                SELECT procedures.*,
                       COUNT(procedure_executions.execution_id) AS execution_count,
                       SUM(CASE WHEN procedure_executions.status = 'completed' THEN 1 ELSE 0 END) AS success_count,
                       SUM(CASE WHEN procedure_executions.status = 'failed' THEN 1 ELSE 0 END) AS failure_count,
                       MAX(procedure_executions.started_at) AS last_executed_at
                FROM procedures
                LEFT JOIN procedure_executions
                    ON procedure_executions.procedure_id = procedures.procedure_id
                {where}
                GROUP BY procedures.procedure_id
                ORDER BY procedures.normalized_name, procedures.version DESC
                """
            ).fetchall()
        return [self._procedure(row) for row in rows]

    def get(self, procedure_id: int, active_only: bool = True) -> Procedure | None:
        with self._connect() as connection:
            condition = "AND procedures.status = 'active'" if active_only else ""
            row = connection.execute(
                f"""
                SELECT procedures.*,
                       COUNT(procedure_executions.execution_id) AS execution_count,
                       SUM(CASE WHEN procedure_executions.status = 'completed' THEN 1 ELSE 0 END) AS success_count,
                       SUM(CASE WHEN procedure_executions.status = 'failed' THEN 1 ELSE 0 END) AS failure_count,
                       MAX(procedure_executions.started_at) AS last_executed_at
                FROM procedures
                LEFT JOIN procedure_executions
                    ON procedure_executions.procedure_id = procedures.procedure_id
                WHERE procedures.procedure_id = ? {condition}
                GROUP BY procedures.procedure_id
                """,
                (procedure_id,),
            ).fetchone()
        return self._procedure(row) if row else None

    def get_by_name(self, name: str, active_only: bool = True) -> Procedure | None:
        normalized = normalize_name(name)
        with self._connect() as connection:
            status_clause = "AND status = 'active'" if active_only else ""
            row = connection.execute(
                f"""
                SELECT procedure_id
                FROM procedures
                WHERE normalized_name = ? {status_clause}
                ORDER BY version DESC
                LIMIT 1
                """,
                (normalized,),
            ).fetchone()
        return self.get(int(row["procedure_id"]), active_only=active_only) if row else None

    def update(self, procedure_id: int, request: ProcedureUpdate) -> Procedure:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM procedures WHERE procedure_id = ?",
                (procedure_id,),
            ).fetchone()
            if not current:
                connection.rollback()
                raise KeyError(f"Unknown procedure: {procedure_id}")
            if current["status"] != "active":
                connection.rollback()
                raise ValueError("Only active procedures can be updated")
            if normalize_name(request.name) != current["normalized_name"]:
                connection.rollback()
                raise ValueError("Procedure updates cannot rename the procedure")

            connection.execute(
                """
                UPDATE procedures
                SET status = 'archived',
                    archived_at = CURRENT_TIMESTAMP
                WHERE procedure_id = ?
                """,
                (procedure_id,),
            )
            procedure = self._create_in_transaction(connection, request)
            connection.commit()
            return procedure

    def archive(self, procedure_id: int) -> Procedure:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE procedures
                SET status = 'archived',
                    archived_at = CURRENT_TIMESTAMP
                WHERE procedure_id = ? AND status = 'active'
                """,
                (procedure_id,),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise KeyError(f"Unknown active procedure: {procedure_id}")
            connection.commit()
        procedure = self.get(procedure_id, active_only=False)
        if not procedure:
            raise KeyError(f"Unknown procedure: {procedure_id}")
        return procedure

    def flag(self, procedure_id: int) -> Procedure:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE procedures
                SET status = 'flagged'
                WHERE procedure_id = ? AND status = 'active'
                """,
                (procedure_id,),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise KeyError(f"Unknown active procedure: {procedure_id}")
            connection.commit()
        procedure = self.get(procedure_id, active_only=False)
        if not procedure:
            raise KeyError(f"Unknown procedure: {procedure_id}")
        return procedure

    def rollback_lookup(self, name: str, version: int) -> Procedure | None:
        if version < 1:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT procedure_id
                FROM procedures
                WHERE normalized_name = ? AND version = ?
                """,
                (normalize_name(name), version),
            ).fetchone()
        return self.get(int(row["procedure_id"]), active_only=False) if row else None

    def record_execution(
        self,
        procedure_id: int,
        status: ExecutionStatus = "recorded",
        outcome: str | None = None,
    ) -> ProcedureExecution:
        if status not in {"recorded", "completed", "failed", "cancelled"}:
            raise ValueError(f"Invalid execution status: {status}")
        if outcome is not None:
            outcome = outcome.strip()[:1000] or None
        procedure = self.get(procedure_id, active_only=False)
        if not procedure:
            raise KeyError(f"Unknown procedure: {procedure_id}")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO procedure_executions (
                    procedure_id, procedure_version, status, completed_at, outcome
                )
                VALUES (?, ?, ?, CASE WHEN ? = 'recorded' THEN NULL ELSE CURRENT_TIMESTAMP END, ?)
                """,
                (procedure_id, procedure.version, status, status, outcome),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM procedure_executions WHERE execution_id = ?",
                (cursor.lastrowid,),
            ).fetchone()
        return ProcedureExecution.model_validate(dict(row))

    def create_pending(self, request: PendingProcedureCreate) -> PendingProcedure:
        proposal = request.proposal
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO pending_procedures (normalized_name, proposal_json)
                VALUES (?, ?)
                """,
                (normalize_name(proposal.name), proposal.model_dump_json()),
            )
            connection.commit()
            return self._pending_by_id(connection, int(cursor.lastrowid))

    def list_pending(self, pending_only: bool = True) -> list[PendingProcedure]:
        with self._connect() as connection:
            where = "WHERE status = 'pending'" if pending_only else ""
            rows = connection.execute(
                f"""
                SELECT *
                FROM pending_procedures
                {where}
                ORDER BY pending_id DESC
                """
            ).fetchall()
        return [self._pending(row) for row in rows]

    def approve_pending(
        self,
        pending_id: int,
        review: PendingProcedureReview,
    ) -> PendingProcedure:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM pending_procedures WHERE pending_id = ?",
                (pending_id,),
            ).fetchone()
            if not row:
                connection.rollback()
                raise KeyError(f"Unknown pending procedure: {pending_id}")
            if row["status"] != "pending":
                connection.rollback()
                raise ValueError("Pending procedure has already been reviewed")

            proposal = ProcedureCreate.model_validate_json(row["proposal_json"])
            procedure = self._create_in_transaction(connection, proposal)
            cursor = connection.execute(
                """
                UPDATE pending_procedures
                SET status = 'approved',
                    reviewed_at = CURRENT_TIMESTAMP,
                    review_note = ?,
                    approved_procedure_id = ?
                WHERE pending_id = ? AND status = 'pending'
                """,
                (review.note, procedure.procedure_id, pending_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ValueError("Pending procedure has already been reviewed")
            connection.commit()
            return self._pending_by_id(connection, pending_id)

    def reject_pending(
        self,
        pending_id: int,
        review: PendingProcedureReview,
    ) -> PendingProcedure:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pending_procedures
                SET status = 'rejected',
                    reviewed_at = CURRENT_TIMESTAMP,
                    review_note = ?
                WHERE pending_id = ? AND status = 'pending'
                """,
                (review.note, pending_id),
            )
            if cursor.rowcount != 1:
                exists = connection.execute(
                    "SELECT status FROM pending_procedures WHERE pending_id = ?",
                    (pending_id,),
                ).fetchone()
                connection.rollback()
                if not exists:
                    raise KeyError(f"Unknown pending procedure: {pending_id}")
                raise ValueError("Pending procedure has already been reviewed")
            connection.commit()
            return self._pending_by_id(connection, pending_id)

    def _create_in_transaction(
        self,
        connection: sqlite3.Connection,
        request: ProcedureCreate | ProcedureUpdate,
    ) -> Procedure:
        normalized = normalize_name(request.name)
        active = connection.execute(
            "SELECT procedure_id FROM procedures WHERE normalized_name = ? AND status = 'active'",
            (normalized,),
        ).fetchone()
        if active:
            raise ValueError(f"Active procedure already exists: {request.name}")
        row = connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS max_version FROM procedures WHERE normalized_name = ?",
            (normalized,),
        ).fetchone()
        version = int(row["max_version"]) + 1
        cursor = connection.execute(
            """
            INSERT INTO procedures (
                normalized_name, name, version, status, source, procedure_confidence,
                required_permission, trigger_phrases, definition_json
            )
            VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?)
            """,
            (
                normalized,
                request.name,
                version,
                request.source,
                request.procedure_confidence,
                request.required_permission,
                json.dumps(request.trigger_phrases),
                request.model_dump_json(),
            ),
        )
        procedure_id = int(cursor.lastrowid)
        row = connection.execute(
            """
            SELECT procedures.*,
                   0 AS execution_count,
                   0 AS success_count,
                   0 AS failure_count,
                   NULL AS last_executed_at
            FROM procedures
            WHERE procedure_id = ?
            """,
            (procedure_id,),
        ).fetchone()
        return self._procedure(row)

    def _procedure(self, row: sqlite3.Row) -> Procedure:
        definition = ProcedureCreate.model_validate_json(row["definition_json"])
        payload = definition.model_dump()
        payload.update(
            {
                "procedure_id": int(row["procedure_id"]),
                "version": int(row["version"]),
                "status": row["status"],
                "created_at": row["created_at"],
                "archived_at": row["archived_at"],
                "stats": ProcedureStats(
                    execution_count=int(row["execution_count"] or 0),
                    success_count=int(row["success_count"] or 0),
                    failure_count=int(row["failure_count"] or 0),
                    last_executed_at=row["last_executed_at"],
                ),
            }
        )
        return Procedure.model_validate(payload)

    def _pending_by_id(self, connection: sqlite3.Connection, pending_id: int) -> PendingProcedure:
        row = connection.execute(
            "SELECT * FROM pending_procedures WHERE pending_id = ?",
            (pending_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Unknown pending procedure: {pending_id}")
        return self._pending(row)

    def _pending(self, row: sqlite3.Row) -> PendingProcedure:
        return PendingProcedure(
            pending_id=int(row["pending_id"]),
            status=row["status"],
            proposal=ProcedureCreate.model_validate_json(row["proposal_json"]),
            submitted_at=row["submitted_at"],
            reviewed_at=row["reviewed_at"],
            review_note=row["review_note"],
            approved_procedure_id=row["approved_procedure_id"],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.memory.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def normalize_name(value: str) -> str:
    normalized = re.sub(r"_+", " ", re.sub(r"[^\w]+", " ", value.casefold())).strip()
    return normalized or "unnamed procedure"
