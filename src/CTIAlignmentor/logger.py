from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AlignmentLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_entry(self, lines: list[str]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def log_merge(
        self,
        *,
        stage: str,
        active_node_id: str,
        active_node_text: str,
        merged_nodes: list[tuple[str, str]],
        reason: dict[str, Any] | str,
    ) -> None:
        reason_text = (
            json.dumps(reason, ensure_ascii=False, sort_keys=True)
            if isinstance(reason, dict)
            else str(reason)
        )
        merged_text = "; ".join(
            f"{node_id} | {node_text}" for node_id, node_text in merged_nodes
        )
        lines = [
            f"Stage: {stage}",
            f"Active: {active_node_id} | {active_node_text}",
            f"Merged: {merged_text}",
            f"Reason: {reason_text}",
            "---",
        ]
        self._write_entry(lines)

    def log_llm_rejection(
        self,
        *,
        stage: str,
        candidate_nodes: list[tuple[str, str]],
        reason: dict[str, Any] | str,
    ) -> None:
        reason_text = (
            json.dumps(reason, ensure_ascii=False, sort_keys=True)
            if isinstance(reason, dict)
            else str(reason)
        )
        candidate_text = "; ".join(
            f"{node_id} | {node_text}" for node_id, node_text in candidate_nodes
        )
        lines = [
            f"Stage: {stage}",
            "Outcome: LLM-rejected",
            f"Candidate: {candidate_text}",
            f"Reason: {reason_text}",
            "---",
        ]
        self._write_entry(lines)
