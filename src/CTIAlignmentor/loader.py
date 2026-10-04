from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .merger import GraphStore
from .path_guard import resolve_existing_path


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


def _extract_source_value_from_node_fields(node: dict[str, Any]) -> Any:
    if _is_ttp_node(node):
        return _normalize_ttp_source_value(
            node.get("best_span_ranges"),
            node.get("best_span_sentences_list"),
        )
    indices = node.get("sentence_indices")
    spans = node.get("span_sentences")
    pairs: list[list[Any]] = []
    if isinstance(indices, list) and isinstance(spans, list):
        for idx, span in zip(indices, spans):
            pairs.append([idx, span])
    return _dedupe_pairs(pairs)


def _drop_legacy_evidence_fields(node: dict[str, Any]) -> None:
    node.pop("sentence_indices", None)
    node.pop("span_sentences", None)


def _extract_source_map(node: dict[str, Any]) -> dict[str, Any]:
    source = node.get("source")
    out: dict[str, Any] = {}
    if not isinstance(source, dict):
        return out
    is_ttp = _is_ttp_node(node)
    for key, value in source.items():
        if not isinstance(key, str):
            continue
        if is_ttp:
            if isinstance(value, dict):
                out[key] = _normalize_ttp_source_value(
                    value.get("best_span_ranges"),
                    value.get("best_span_sentences_list"),
                )
            elif isinstance(value, list):
                out[key] = _convert_pair_list_to_ttp_source_value(value)
            continue
        normalized_pairs = _normalize_pair_source_value(value)
        if normalized_pairs is None:
            continue
        out[key] = normalized_pairs
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


def _build_source_key(entity_name: str, article_name: str) -> str:
    entity = str(entity_name).strip()
    article = str(article_name).strip()
    if not entity:
        return ""
    if not article:
        return entity
    suffix = f"_{article}"
    if entity.endswith(suffix):
        return entity
    return f"{entity}{suffix}"


def _prepare_node_source(node: dict[str, Any], article_name: str) -> None:
    article = str(article_name).strip()
    node_text = str(node.get("node_text", "")).strip()
    is_ttp = _is_ttp_node(node)

    normalized_source: dict[str, Any] = {}
    for raw_key, value in _extract_source_map(node).items():
        normalized_key = str(raw_key).strip()
        if not normalized_key:
            continue
        source_key = (
            _build_source_key(normalized_key, article)
            if normalized_key == node_text
            else normalized_key
        )
        if not source_key:
            continue
        normalized_source = _merge_source_maps(
            normalized_source,
            {source_key: value},
            is_ttp=is_ttp,
        )

    default_key = _build_source_key(node_text, article)
    if default_key:
        normalized_source = _merge_source_maps(
            normalized_source,
            {default_key: _extract_source_value_from_node_fields(node)},
            is_ttp=is_ttp,
        )

    node["source"] = normalized_source
    node["__source_article"] = article
    _drop_legacy_evidence_fields(node)


def _expand_path_tokens(input_paths: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for raw in input_paths:
        raw_value = str(raw).strip()
        if not raw_value:
            continue

        try:
            resolve_existing_path(raw_value)
        except FileNotFoundError:
            pass
        else:
            tokens.append(raw_value)
            continue

        for part in raw_value.split(","):
            part = part.strip()
            if part:
                tokens.append(part)
    return tokens


def resolve_input_files(input_paths: list[str]) -> list[Path]:
    tokens = _expand_path_tokens(input_paths)
    files: list[Path] = []
    seen: set[Path] = set()
    for token in tokens:
        path = resolve_existing_path(token)
        candidates: list[Path] = []
        if path.is_dir():
            candidates = sorted(
                [p for p in path.rglob("*.json") if p.is_file()],
                key=lambda p: str(p),
            )
        elif path.is_file() and path.suffix.lower() == ".json":
            candidates = [path]
        else:
            continue
        for file_path in candidates:
            file_path = resolve_existing_path(file_path)
            if file_path in seen:
                continue
            seen.add(file_path)
            files.append(file_path)
    if not files:
        raise ValueError("No input JSON files resolved from --input_paths")
    return files


def _make_unique_id(base_id: str, existing_ids: set[str], seed: str) -> str:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base_id}__{digest}"
    serial = 1
    while candidate in existing_ids:
        candidate = f"{base_id}__{digest}_{serial}"
        serial += 1
    return candidate


def _same_identity(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    return (
        str(existing.get("node_class", "")) == str(incoming.get("node_class", ""))
        and str(existing.get("node_type", "")) == str(incoming.get("node_type", ""))
        and str(existing.get("node_text", "")) == str(incoming.get("node_text", ""))
    )

def _merge_node_observation(dst: dict[str, Any], src: dict[str, Any]) -> None:
    merged_source = _merge_source_maps(
        _extract_source_map(dst),
        _extract_source_map(src),
        is_ttp=_is_ttp_node(dst),
    )
    if merged_source:
        dst["source"] = merged_source
    if "__source_article" not in dst and "__source_article" in src:
        dst["__source_article"] = str(src["__source_article"])


def _safe_node_id(bucket: str, file_idx: int, node_idx: int, node: dict[str, Any]) -> str:
    node_id = str(node.get("node_id") or "")
    if node_id:
        return node_id
    return f"{bucket.upper()}_AUTO_{file_idx}_{node_idx}"


def load_graphs(input_files: list[Path]) -> tuple[GraphStore, list[str]]:
    store = GraphStore()
    merged_from_files: list[str] = []

    for file_idx, json_path in enumerate(input_files):
        article_name = json_path.stem
        with json_path.open("r", encoding="utf-8") as f:
            graph = json.load(f)
        if not isinstance(graph, dict):
            raise ValueError(f"Invalid graph format: {json_path}")

        merged_from_files.append(json_path.name)
        store.merge_top_level_fields(graph)

        node_map: dict[str, str] = {}
        nodes_section = graph.get("nodes", {})
        if not isinstance(nodes_section, dict):
            raise ValueError(f"Invalid nodes section in {json_path}")

        for bucket in ("high_level", "ttp", "ioc"):
            nodes = nodes_section.get(bucket, [])
            if not isinstance(nodes, list):
                continue
            for node_idx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    continue
                old_id = _safe_node_id(bucket, file_idx, node_idx, node)
                node_copy = deepcopy(node)
                node_copy.setdefault("node_class", bucket)
                _prepare_node_source(node_copy, article_name)
                new_id = old_id
                if store.has_node(new_id):
                    existing = store.active_nodes[new_id]
                    if _same_identity(existing, node_copy):
                        _merge_node_observation(existing, node_copy)
                        node_map[old_id] = new_id
                        continue
                    seed = f"{json_path.name}:{bucket}:{node_idx}:{old_id}"
                    new_id = _make_unique_id(new_id, set(store.active_nodes.keys()), seed)

                node_copy["node_id"] = new_id
                node_map[old_id] = new_id
                store.add_node(node_copy, bucket=bucket)

        edges = graph.get("edges", [])
        if isinstance(edges, list):
            for edge_idx, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    continue
                edge_copy = deepcopy(edge)
                edge_id = str(edge_copy.get("edge_id") or f"EDGE_AUTO_{file_idx}_{edge_idx}")
                if store.has_edge(edge_id):
                    seed = f"{json_path.name}:edge:{edge_idx}:{edge_id}"
                    edge_id = _make_unique_id(edge_id, set(store.edge_ids), seed)
                edge_copy["edge_id"] = edge_id

                node_u = str(edge_copy.get("node_u", ""))
                node_v = str(edge_copy.get("node_v", ""))
                edge_copy["node_u"] = node_map.get(node_u, node_u)
                edge_copy["node_v"] = node_map.get(node_v, node_v)

                if not store.has_node(edge_copy["node_u"]):
                    continue
                if not store.has_node(edge_copy["node_v"]):
                    continue
                store.add_edge(edge_copy)

        for inactive_node in graph.get("inactive_nodes", []) or []:
            if isinstance(inactive_node, dict):
                inactive_copy = deepcopy(inactive_node)
                _prepare_node_source(inactive_copy, article_name)
                store.inactive_nodes.append(inactive_copy)
        for inactive_edge in graph.get("inactive_edges", []) or []:
            if isinstance(inactive_edge, dict):
                store.inactive_edges.append(deepcopy(inactive_edge))

    store.rebuild_node_edge_refs()
    return store, merged_from_files
