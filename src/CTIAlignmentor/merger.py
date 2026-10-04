from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .logger import AlignmentLogger


def _normalize_pair(item: Any) -> list[Any] | None:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return [item[0], item[1]]
    return None


def _dedupe_pairs(pairs: list[list[Any]]) -> list[list[Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[list[Any]] = []
    for pair in pairs:
        key = (str(pair[0]), str(pair[1]))
        if key in seen:
            continue
        seen.add(key)
        result.append(pair)
    return result


def _is_ttp_node(node: dict[str, Any]) -> bool:
    node_type = str(node.get("node_type", "")).strip().upper()
    node_class = str(node.get("node_class", "")).strip().lower()
    return node_type == "TTP" or node_class == "ttp"


def _normalize_pair_source_value(value: Any) -> list[list[Any]] | None:
    if not isinstance(value, list):
        return None
    pairs: list[list[Any]] = []
    for item in value:
        pair = _normalize_pair(item)
        if pair is None:
            continue
        pairs.append(pair)
    return _dedupe_pairs(pairs)


def _empty_ttp_source_value() -> dict[str, list[list[Any]]]:
    return {"best_span_ranges": [], "best_span_sentences_list": []}


def _dedupe_ttp_source_value(
    best_span_ranges: list[list[Any]], best_span_sentences_list: list[list[Any]]
) -> dict[str, list[list[Any]]]:
    seen: set[tuple[int, int, tuple[str, ...]]] = set()
    ranges_out: list[list[Any]] = []
    sentences_out: list[list[Any]] = []
    for span_range, span_sentences in zip(best_span_ranges, best_span_sentences_list):
        if (
            not isinstance(span_range, list)
            or len(span_range) < 2
            or not isinstance(span_sentences, list)
        ):
            continue
        try:
            start = int(span_range[0])
            end = int(span_range[1])
        except (TypeError, ValueError):
            continue
        if start > end:
            continue
        sentences = [str(sentence) for sentence in span_sentences]
        dedupe_key = (start, end, tuple(sentences))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        ranges_out.append([start, end])
        sentences_out.append(sentences)
    return {
        "best_span_ranges": ranges_out,
        "best_span_sentences_list": sentences_out,
    }


def _normalize_ttp_source_value(
    best_span_ranges: Any, best_span_sentences_list: Any
) -> dict[str, list[list[Any]]]:
    if not isinstance(best_span_ranges, list) or not isinstance(best_span_sentences_list, list):
        return _empty_ttp_source_value()
    ranges: list[list[Any]] = []
    sentences_list: list[list[Any]] = []
    for span_range, span_sentences in zip(best_span_ranges, best_span_sentences_list):
        if not isinstance(span_sentences, list):
            continue
        ranges.append(list(span_range) if isinstance(span_range, (list, tuple)) else [span_range])
        sentences_list.append([str(sentence) for sentence in span_sentences])
    return _dedupe_ttp_source_value(ranges, sentences_list)


def _convert_pair_list_to_ttp_source_value(value: Any) -> dict[str, list[list[Any]]]:
    pairs = _normalize_pair_source_value(value)
    if pairs is None:
        return _empty_ttp_source_value()
    indexed_pairs: list[tuple[int, str]] = []
    for pair in pairs:
        try:
            indexed_pairs.append((int(pair[0]), str(pair[1])))
        except (TypeError, ValueError):
            continue
    indexed_pairs.sort(key=lambda item: item[0])

    best_span_ranges: list[list[Any]] = []
    best_span_sentences_list: list[list[Any]] = []
    current_range: list[Any] | None = None
    current_sentences: list[Any] = []
    for idx, sentence in indexed_pairs:
        if current_range is None:
            current_range = [idx, idx]
            current_sentences = [sentence]
            continue
        if idx == current_range[1] + 1:
            current_range[1] = idx
            current_sentences.append(sentence)
            continue
        best_span_ranges.append(current_range)
        best_span_sentences_list.append(current_sentences)
        current_range = [idx, idx]
        current_sentences = [sentence]
    if current_range is not None:
        best_span_ranges.append(current_range)
        best_span_sentences_list.append(current_sentences)
    return _dedupe_ttp_source_value(best_span_ranges, best_span_sentences_list)


def _extract_source_map(node: dict[str, Any]) -> dict[str, Any]:
    source = node.get("source")
    out: dict[str, Any] = {}
    if not isinstance(source, dict):
        return out
    is_ttp = _is_ttp_node(node)
    for k, v in source.items():
        if not isinstance(k, str):
            continue
        if is_ttp:
            if isinstance(v, dict):
                out[k] = _normalize_ttp_source_value(
                    v.get("best_span_ranges"),
                    v.get("best_span_sentences_list"),
                )
            elif isinstance(v, list):
                out[k] = _convert_pair_list_to_ttp_source_value(v)
            continue
        normalized_pairs = _normalize_pair_source_value(v)
        if normalized_pairs is None:
            continue
        out[k] = normalized_pairs
    return out


def _merge_source_maps(
    base: dict[str, Any], incoming: dict[str, Any], *, is_ttp: bool
) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in incoming.items():
        if is_ttp:
            current = result.get(key, _empty_ttp_source_value())
            result[key] = _normalize_ttp_source_value(
                list(current.get("best_span_ranges", [])) + list(value.get("best_span_ranges", [])),
                list(current.get("best_span_sentences_list", []))
                + list(value.get("best_span_sentences_list", [])),
            )
            continue
        current_pairs = list(result.get(key, []))
        current_pairs.extend(value)
        result[key] = _dedupe_pairs(current_pairs)
    return result


def _build_source_key(node_text: str, article_name: str) -> str:
    entity = str(node_text).strip()
    article = str(article_name).strip()
    if not entity:
        return ""
    if not article:
        return entity
    suffix = f"_{article}"
    if entity.endswith(suffix):
        return entity
    return f"{entity}{suffix}"


def _public_node_copy(node: dict[str, Any]) -> dict[str, Any]:
    node_copy = deepcopy(node)
    for key in list(node_copy.keys()):
        if str(key).startswith("__"):
            node_copy.pop(key, None)
    return node_copy


@dataclass
class GraphStore:
    active_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_bucket: dict[str, str] = field(default_factory=dict)
    node_order: dict[str, list[str]] = field(
        default_factory=lambda: {"high_level": [], "ttp": [], "ioc": []}
    )
    edges: list[dict[str, Any]] = field(default_factory=list)
    edge_ids: set[str] = field(default_factory=set)
    inactive_nodes: list[dict[str, Any]] = field(default_factory=list)
    inactive_edges: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def merge_top_level_fields(self, graph: dict[str, Any]) -> None:
        reserved = {"nodes", "edges", "inactive_nodes", "inactive_edges", "metadata"}
        for key, value in graph.items():
            if key in reserved:
                continue
            if key not in self.extra_fields:
                self.extra_fields[key] = deepcopy(value)
        metadata = graph.get("metadata")
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if key not in self.metadata:
                    self.metadata[key] = deepcopy(value)

    def add_node(self, node: dict[str, Any], bucket: str) -> None:
        node_id = str(node.get("node_id") or "")
        if not node_id:
            raise ValueError("Node is missing node_id")
        if node_id in self.active_nodes:
            raise ValueError(f"Duplicated active node_id: {node_id}")
        self.active_nodes[node_id] = node
        self.node_bucket[node_id] = bucket
        self.node_order.setdefault(bucket, []).append(node_id)

    def add_edge(self, edge: dict[str, Any]) -> None:
        edge_id = str(edge.get("edge_id") or "")
        if not edge_id:
            raise ValueError("Edge is missing edge_id")
        if edge_id in self.edge_ids:
            raise ValueError(f"Duplicated edge_id: {edge_id}")
        self.edges.append(edge)
        self.edge_ids.add(edge_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self.active_nodes

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self.edge_ids

    def get_nodes(self, bucket: str) -> list[dict[str, Any]]:
        return [
            self.active_nodes[node_id]
            for node_id in self.node_order.get(bucket, [])
            if node_id in self.active_nodes
        ]

    def iter_all_nodes(self) -> list[dict[str, Any]]:
        return list(self.active_nodes.values())

    def _delete_node(self, node_id: str) -> None:
        bucket = self.node_bucket.get(node_id)
        if bucket:
            try:
                self.node_order[bucket].remove(node_id)
            except ValueError:
                pass
        self.active_nodes.pop(node_id, None)
        self.node_bucket.pop(node_id, None)

    def _node_degrees(self, node_ids: set[str]) -> dict[str, int]:
        degrees = {node_id: 0 for node_id in node_ids}
        for edge in self.edges:
            node_u = str(edge.get("node_u", ""))
            node_v = str(edge.get("node_v", ""))
            if node_u in degrees:
                degrees[node_u] += 1
            if node_v in degrees:
                degrees[node_v] += 1
        return degrees

    def _choose_active_node(self, node_ids: list[str]) -> str:
        active_ids = [node_id for node_id in node_ids if node_id in self.active_nodes]
        if len(active_ids) < 2:
            return active_ids[0]
        position = {node_id: idx for idx, node_id in enumerate(active_ids)}
        degree = self._node_degrees(set(active_ids))
        return max(active_ids, key=lambda node_id: (degree[node_id], -position[node_id]))

    def _collect_node_source(self, node: dict[str, Any]) -> dict[str, Any]:
        return _extract_source_map(node)

    def ensure_node_source(self, node: dict[str, Any]) -> None:
        source_map = self._collect_node_source(node)
        node_text = str(node.get("node_text", "")).strip()
        article_name = str(node.get("__source_article", "")).strip()
        source_key = _build_source_key(node_text, article_name)
        if source_key and source_key not in source_map:
            source_map[source_key] = (
                _empty_ttp_source_value() if _is_ttp_node(node) else []
            )
        node["source"] = source_map

    def ensure_all_sources(self) -> None:
        for node in self.active_nodes.values():
            self.ensure_node_source(node)

    def _redirect_edges(self, src_node_id: str, dst_node_id: str) -> None:
        for edge in self.edges:
            touched = False
            if edge.get("node_u") == src_node_id:
                touched = True
            if edge.get("node_v") == src_node_id:
                touched = True
            if not touched:
                continue
            self.inactive_edges.append(deepcopy(edge))
            if edge.get("node_u") == src_node_id:
                edge["node_u"] = dst_node_id
            if edge.get("node_v") == src_node_id:
                edge["node_v"] = dst_node_id

    def apply_merge_cluster(
        self,
        *,
        node_ids: list[str],
        stage: str,
        reason: dict[str, Any] | str,
        logger: AlignmentLogger,
    ) -> int:
        active_ids = [node_id for node_id in node_ids if node_id in self.active_nodes]
        if len(active_ids) < 2:
            return 0

        active_id = self._choose_active_node(active_ids)
        active_node = self.active_nodes[active_id]

        source_map: dict[str, Any] = {}
        for node_id in active_ids:
            source_map = _merge_source_maps(
                source_map,
                self._collect_node_source(self.active_nodes[node_id]),
                is_ttp=_is_ttp_node(active_node),
            )

        aliases = list(active_node.get("alias", [])) if isinstance(active_node.get("alias"), list) else []
        alias_set = {str(alias) for alias in aliases}
        merged_nodes: list[tuple[str, str]] = []

        for node_id in active_ids:
            if node_id == active_id:
                continue
            merged_node = self.active_nodes[node_id]
            merged_text = str(merged_node.get("node_text", ""))
            if merged_text and merged_text != active_node.get("node_text") and merged_text not in alias_set:
                aliases.append(merged_text)
                alias_set.add(merged_text)
            merged_nodes.append((node_id, merged_text))
            self._redirect_edges(node_id, active_id)
            self.inactive_nodes.append(deepcopy(merged_node))
            self._delete_node(node_id)

        active_node["alias"] = aliases
        active_node["source"] = _merge_source_maps(
            _extract_source_map(active_node),
            source_map,
            is_ttp=_is_ttp_node(active_node),
        )
        self.ensure_node_source(active_node)

        logger.log_merge(
            stage=stage,
            active_node_id=active_id,
            active_node_text=str(active_node.get("node_text", "")),
            merged_nodes=merged_nodes,
            reason=reason,
        )
        return len(merged_nodes)

    def rebuild_node_edge_refs(self) -> None:
        for node in self.active_nodes.values():
            node["edges"] = []
        for edge in self.edges:
            edge_id = edge.get("edge_id")
            node_u = str(edge.get("node_u", ""))
            node_v = str(edge.get("node_v", ""))
            if node_u in self.active_nodes:
                self.active_nodes[node_u].setdefault("edges", []).append(edge_id)
            if node_v in self.active_nodes:
                self.active_nodes[node_v].setdefault("edges", []).append(edge_id)

    def neighbors(self, node_id: str) -> set[str]:
        nbrs: set[str] = set()
        for edge in self.edges:
            node_u = str(edge.get("node_u", ""))
            node_v = str(edge.get("node_v", ""))
            if node_u == node_id and node_v in self.active_nodes:
                nbrs.add(node_v)
            elif node_v == node_id and node_u in self.active_nodes:
                nbrs.add(node_u)
        return nbrs

    def ttp_neighbors(self, node_id: str) -> set[str]:
        return {
            nbr
            for nbr in self.neighbors(node_id)
            if self.node_bucket.get(nbr) == "ttp"
        }

    def to_output(self, merged_from_files: list[str]) -> dict[str, Any]:
        self.ensure_all_sources()
        self.rebuild_node_edge_refs()

        nodes_dict = {
            bucket: [_public_node_copy(node) for node in self.get_nodes(bucket)]
            for bucket in ("high_level", "ttp", "ioc")
        }

        metadata = deepcopy(self.metadata)
        metadata["merged_from_files"] = merged_from_files
        metadata["alignment_pipeline"] = "CTIAlignmentor"

        output: dict[str, Any] = {
            "nodes": nodes_dict,
            "edges": deepcopy(self.edges),
            "inactive_nodes": [_public_node_copy(node) for node in self.inactive_nodes],
            "inactive_edges": deepcopy(self.inactive_edges),
            "merged_from_files": merged_from_files,
            "metadata": metadata,
        }
        for key, value in self.extra_fields.items():
            if key in output:
                continue
            output[key] = deepcopy(value)
        return output
