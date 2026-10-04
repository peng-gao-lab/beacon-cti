from __future__ import annotations

from copy import deepcopy
from typing import Any


def dedupe_edges(
    edges: list[dict[str, Any]],
    inactive_edges: list[dict[str, Any]],
) -> int:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    removed = 0

    for edge in edges:
        key = (str(edge.get("node_u", "")), str(edge.get("node_v", "")))
        if key in seen:
            inactive_edges.append(deepcopy(edge))
            removed += 1
            continue
        seen.add(key)
        deduped.append(edge)

    edges[:] = deduped
    return removed
