#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ1 evaluator:
- Compare predicted graph JSON and GT graph JSON
- Compute precision / recall / f1 for:
  1) Nodes (overall)
  2) Nodes.high_level
  3) Nodes.ttp
  4) Nodes.ioc
  5) Edges

Matching strategy:
- Nodes and edges can only match within the same broad type and subtype
- High-Level and IoC nodes use character-level fuzzy matching without partial-substring shortcuts
- TTP nodes are normalized to ATT&CK IDs and require exact equality
- One-to-one assignment retains the original greedy strategy
- Supports both single-file mode and batch mode
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


NODE_TYPES = ("high_level", "ttp", "ioc")
EDGE_TYPE_TARGETS = ("high_level-ttp", "ioc-ttp")
TTP_PATTERN = re.compile(r"T\d{4}(?:\.\d{3})?", flags=re.IGNORECASE)
HEX_PATTERN = re.compile(r"^[a-fA-F0-9]{16,}$")
WS_PATTERN = re.compile(r"\s+")
SUBTYPE_ALIASES = {
    "ip": "ipaddress",
    "md5": "hash",
    "sha1": "hash",
    "sha256": "hash",
}


@dataclass(frozen=True)
class NodeEntry:
    text: str
    node_type: str
    node_subtype: str
    normalized: str


@dataclass(frozen=True)
class EdgeEntry:
    u_text: str
    v_text: str
    u_type: str
    v_type: str
    u_subtype: str
    v_subtype: str
    relation_type: str
    normalized: Tuple[str, str]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def normalize_whitespace(text: str) -> str:
    return WS_PATTERN.sub(" ", text).strip()


def extract_ttp_id(text: str) -> Optional[str]:
    match = TTP_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).upper()


def normalize_text(text: str, node_type: str) -> str:
    raw = normalize_whitespace(str(text))
    if node_type == "ttp":
        ttp_id = extract_ttp_id(raw)
        if ttp_id:
            return ttp_id
        raw = raw.replace("TTP:", "").replace("ttp:", "")
        return raw.upper()
    return raw.lower()


def normalize_node_subtype(node_subtype: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(node_subtype).lower())
    return SUBTYPE_ALIASES.get(normalized, normalized)


def infer_node_type_from_text(text: str) -> str:
    t = normalize_whitespace(str(text))
    if extract_ttp_id(t):
        return "ttp"
    tl = t.lower()
    if (
        "\\" in tl
        or "/" in tl
        or ":" in tl
        or "http://" in tl
        or "https://" in tl
        or "@" in tl
        or HEX_PATTERN.fullmatch(tl) is not None
    ):
        return "ioc"
    return "high_level"


def ngram_set(text: str, n: int = 3) -> set[str]:
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def token_sort(text: str) -> str:
    tokens = normalize_whitespace(text).split(" ")
    tokens = [tok for tok in tokens if tok]
    return " ".join(sorted(tokens))


def text_similarity(a: str, b: str, node_type: Optional[str] = None) -> float:
    kind = node_type or infer_node_type_from_text(a) or infer_node_type_from_text(b)

    na = normalize_text(a, kind)
    nb = normalize_text(b, kind)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    if kind == "ttp":
        return 0.0

    seq = SequenceMatcher(None, na, nb).ratio()
    tok = SequenceMatcher(None, token_sort(na), token_sort(nb)).ratio()
    grams_a = ngram_set(na, 3)
    grams_b = ngram_set(nb, 3)
    jac = (len(grams_a & grams_b) / len(grams_a | grams_b)) if (grams_a or grams_b) else 0.0

    return max(seq, tok, jac)


def calculate_metrics_from_counts(tp: int, fp: int, fn: int, pred_count: int, gt_count: int, threshold: float) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "threshold": threshold,
    }


def greedy_match(
    predicted_items: Sequence[Any],
    gt_items: Sequence[Any],
    scorer: Callable[[Any, Any], float],
    threshold: float,
) -> Dict[str, Any]:
    candidates: List[Tuple[float, int, int]] = []
    for i, pred in enumerate(predicted_items):
        for j, gt in enumerate(gt_items):
            score = scorer(pred, gt)
            if score >= threshold:
                candidates.append((score, i, j))

    candidates.sort(key=lambda x: x[0], reverse=True)
    matched_pred = set()
    matched_gt = set()
    matched_pairs: List[Tuple[int, int, float]] = []

    for score, i, j in candidates:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        matched_pairs.append((i, j, score))

    pred_count = len(predicted_items)
    gt_count = len(gt_items)
    unmatched_pred_indices = [i for i in range(pred_count) if i not in matched_pred]
    unmatched_gt_indices = [j for j in range(gt_count) if j not in matched_gt]

    return {
        "matched_pairs": matched_pairs,
        "unmatched_pred_indices": unmatched_pred_indices,
        "unmatched_gt_indices": unmatched_gt_indices,
        "tp": len(matched_pairs),
        "fp": len(unmatched_pred_indices),
        "fn": len(unmatched_gt_indices),
        "pred_count": pred_count,
        "gt_count": gt_count,
        "threshold": threshold,
    }


def greedy_match_metrics(
    predicted_items: Sequence[Any],
    gt_items: Sequence[Any],
    scorer: Callable[[Any, Any], float],
    threshold: float,
) -> Dict[str, Any]:
    match = greedy_match(predicted_items, gt_items, scorer=scorer, threshold=threshold)
    return calculate_metrics_from_counts(
        match["tp"],
        match["fp"],
        match["fn"],
        match["pred_count"],
        match["gt_count"],
        threshold,
    )


def deduplicate_nodes(nodes: Iterable[Tuple[str, str]], node_type: str) -> List[NodeEntry]:
    seen = set()
    unique_nodes: List[NodeEntry] = []
    for text, node_subtype in nodes:
        t = str(text).strip()
        if not t:
            continue
        norm = normalize_text(t, node_type)
        key = (node_type, node_subtype, norm)
        if key in seen:
            continue
        seen.add(key)
        unique_nodes.append(
            NodeEntry(
                text=t,
                node_type=node_type,
                node_subtype=node_subtype,
                normalized=norm,
            )
        )
    return unique_nodes


def extract_nodes_and_maps(
    data: Dict[str, Any],
) -> Tuple[Dict[str, List[NodeEntry]], Dict[str, str], Dict[str, str], Dict[str, str]]:
    nodes = data.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("'nodes' must be a dict")

    by_type: Dict[str, List[Tuple[str, str]]] = {k: [] for k in NODE_TYPES}
    node_id_to_text: Dict[str, str] = {}
    node_id_to_type: Dict[str, str] = {}
    node_id_to_subtype: Dict[str, str] = {}

    for node_type in NODE_TYPES:
        node_list = nodes.get(node_type, [])
        if not isinstance(node_list, list):
            continue
        for node in node_list:
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("node_id", "")).strip()
            text = str(node.get("node_text", "")).strip()
            if not text:
                text = str(node.get("normalized_value", "")).strip()
            if not text and node_id:
                text = node_id
            if not text:
                continue

            node_subtype = normalize_node_subtype(node.get("node_type", ""))
            by_type[node_type].append((text, node_subtype))
            if node_id:
                node_id_to_text[node_id] = text
                node_id_to_type[node_id] = node_type
                node_id_to_subtype[node_id] = node_subtype

    deduped = {k: deduplicate_nodes(v, k) for k, v in by_type.items()}
    return deduped, node_id_to_text, node_id_to_type, node_id_to_subtype


def edge_normalized_key(edge: EdgeEntry) -> Tuple[Tuple[str, str, str], Tuple[str, str, str]]:
    a = (edge.u_type, edge.u_subtype, normalize_text(edge.u_text, edge.u_type))
    b = (edge.v_type, edge.v_subtype, normalize_text(edge.v_text, edge.v_type))
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def canonical_edge_relation(u_type: str, v_type: str) -> str:
    pair = tuple(sorted((u_type, v_type)))
    if pair == ("high_level", "ttp"):
        return "high_level-ttp"
    if pair == ("ioc", "ttp"):
        return "ioc-ttp"
    return f"{pair[0]}-{pair[1]}"


def extract_edges(
    data: Dict[str, Any],
    node_id_to_text: Dict[str, str],
    node_id_to_type: Dict[str, str],
    node_id_to_subtype: Dict[str, str],
) -> List[EdgeEntry]:
    edges = data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("'edges' must be a list")

    unique: Dict[Tuple[Tuple[str, str, str], Tuple[str, str, str]], EdgeEntry] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        node_u = str(edge.get("node_u", "")).strip()
        node_v = str(edge.get("node_v", "")).strip()
        if not node_u or not node_v:
            continue

        u_text = node_id_to_text.get(node_u, node_u)
        v_text = node_id_to_text.get(node_v, node_v)
        u_type = node_id_to_type.get(node_u, infer_node_type_from_text(u_text))
        v_type = node_id_to_type.get(node_v, infer_node_type_from_text(v_text))
        u_subtype = node_id_to_subtype.get(node_u, "")
        v_subtype = node_id_to_subtype.get(node_v, "")

        entry = EdgeEntry(
            u_text=u_text,
            v_text=v_text,
            u_type=u_type,
            v_type=v_type,
            u_subtype=u_subtype,
            v_subtype=v_subtype,
            relation_type=canonical_edge_relation(u_type, v_type),
            normalized=(normalize_text(u_text, u_type), normalize_text(v_text, v_type)),
        )
        unique[edge_normalized_key(entry)] = entry

    return list(unique.values())


def node_similarity(pred: NodeEntry, gt: NodeEntry) -> float:
    if pred.node_type != gt.node_type:
        return 0.0
    if pred.node_subtype != gt.node_subtype:
        return 0.0
    if pred.node_type == "ttp":
        return 1.0 if pred.normalized == gt.normalized else 0.0
    return text_similarity(pred.text, gt.text, node_type=pred.node_type)


def endpoint_similarity(
    text_a: str,
    type_a: str,
    subtype_a: str,
    text_b: str,
    type_b: str,
    subtype_b: str,
) -> float:
    if type_a != type_b:
        return 0.0
    if subtype_a != subtype_b:
        return 0.0
    if type_a == "ttp":
        normalized_a = normalize_text(text_a, "ttp")
        normalized_b = normalize_text(text_b, "ttp")
        return 1.0 if normalized_a == normalized_b else 0.0
    return text_similarity(text_a, text_b, node_type=type_a)


def edge_endpoint_by_type(edge: EdgeEntry, node_type: str) -> Optional[Tuple[str, str]]:
    if edge.u_type == node_type:
        return edge.u_text, edge.u_subtype
    if edge.v_type == node_type:
        return edge.v_text, edge.v_subtype
    return None


def edge_similarity(pred: EdgeEntry, gt: EdgeEntry) -> float:
    if pred.relation_type != gt.relation_type:
        return 0.0

    if pred.relation_type in EDGE_TYPE_TARGETS:
        entity_type = "high_level" if pred.relation_type == "high_level-ttp" else "ioc"
        pred_ttp = edge_endpoint_by_type(pred, "ttp")
        gt_ttp = edge_endpoint_by_type(gt, "ttp")
        pred_entity = edge_endpoint_by_type(pred, entity_type)
        gt_entity = edge_endpoint_by_type(gt, entity_type)

        if None in (pred_ttp, gt_ttp, pred_entity, gt_entity):
            return 0.0
        pred_ttp_text, pred_ttp_subtype = pred_ttp
        gt_ttp_text, gt_ttp_subtype = gt_ttp
        pred_entity_text, pred_entity_subtype = pred_entity
        gt_entity_text, gt_entity_subtype = gt_entity

        if endpoint_similarity(
            pred_ttp_text,
            "ttp",
            pred_ttp_subtype,
            gt_ttp_text,
            "ttp",
            gt_ttp_subtype,
        ) < 1.0:
            return 0.0

        # The edge threshold applies to the fuzzy entity endpoint. A correct TTP
        # endpoint must not compensate for an entity similarity below threshold.
        return endpoint_similarity(
            pred_entity_text,
            entity_type,
            pred_entity_subtype,
            gt_entity_text,
            entity_type,
            gt_entity_subtype,
        )

    # Retain orientation-independent matching for any non-target edge type, while
    # endpoint_similarity still enforces identical endpoint node types.
    direct = (
        endpoint_similarity(pred.u_text, pred.u_type, pred.u_subtype, gt.u_text, gt.u_type, gt.u_subtype)
        + endpoint_similarity(pred.v_text, pred.v_type, pred.v_subtype, gt.v_text, gt.v_type, gt.v_subtype)
    ) / 2.0
    reverse = (
        endpoint_similarity(pred.u_text, pred.u_type, pred.u_subtype, gt.v_text, gt.v_type, gt.v_subtype)
        + endpoint_similarity(pred.v_text, pred.v_type, pred.v_subtype, gt.u_text, gt.u_type, gt.u_subtype)
    ) / 2.0
    return max(direct, reverse)


def evaluate_edge_type_metrics(
    pred_edges: Sequence[EdgeEntry],
    gt_edges: Sequence[EdgeEntry],
    edge_match: Dict[str, Any],
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    matched_pairs: List[Tuple[int, int, float]] = edge_match["matched_pairs"]

    results: Dict[str, Dict[str, Any]] = {}
    for target in EDGE_TYPE_TARGETS:
        pred_count = sum(edge.relation_type == target for edge in pred_edges)
        gt_count = sum(edge.relation_type == target for edge in gt_edges)
        tp = sum(
            pred_edges[pred_idx].relation_type == target
            and gt_edges[gt_idx].relation_type == target
            for pred_idx, gt_idx, _score in matched_pairs
        )
        fp = pred_count - tp
        fn = gt_count - tp
        results[target] = calculate_metrics_from_counts(
            tp,
            fp,
            fn,
            pred_count,
            gt_count,
            threshold,
        )

    return results


def evaluate_one_file(pred_path: Path, gt_path: Path, node_threshold: float, edge_threshold: float) -> Dict[str, Any]:
    pred_data = read_json(pred_path)
    gt_data = read_json(gt_path)

    pred_nodes_by_type, pred_id_text, pred_id_type, pred_id_subtype = extract_nodes_and_maps(pred_data)
    gt_nodes_by_type, gt_id_text, gt_id_type, gt_id_subtype = extract_nodes_and_maps(gt_data)

    node_metrics: Dict[str, Dict[str, Any]] = {}
    all_pred_nodes = pred_nodes_by_type["high_level"] + pred_nodes_by_type["ttp"] + pred_nodes_by_type["ioc"]
    all_gt_nodes = gt_nodes_by_type["high_level"] + gt_nodes_by_type["ttp"] + gt_nodes_by_type["ioc"]

    node_metrics["overall"] = greedy_match_metrics(
        all_pred_nodes,
        all_gt_nodes,
        scorer=node_similarity,
        threshold=node_threshold,
    )

    for node_type in NODE_TYPES:
        node_metrics[node_type] = greedy_match_metrics(
            pred_nodes_by_type[node_type],
            gt_nodes_by_type[node_type],
            scorer=node_similarity,
            threshold=node_threshold,
        )

    pred_edges = extract_edges(pred_data, pred_id_text, pred_id_type, pred_id_subtype)
    gt_edges = extract_edges(gt_data, gt_id_text, gt_id_type, gt_id_subtype)
    edge_match = greedy_match(
        pred_edges,
        gt_edges,
        scorer=edge_similarity,
        threshold=edge_threshold,
    )
    edge_metrics = calculate_metrics_from_counts(
        edge_match["tp"],
        edge_match["fp"],
        edge_match["fn"],
        edge_match["pred_count"],
        edge_match["gt_count"],
        edge_threshold,
    )
    edge_type_metrics = evaluate_edge_type_metrics(
        pred_edges,
        gt_edges,
        edge_match,
        edge_threshold,
    )

    return {
        "file_name": pred_path.name,
        "pred_file": str(pred_path),
        "gt_file": str(gt_path),
        "node_metrics": node_metrics,
        "edge_metrics": edge_metrics,
        "edge_type_metrics": edge_type_metrics,
    }


def aggregate_metrics(file_results: Sequence[Dict[str, Any]], key_path: Tuple[str, ...], threshold: float) -> Dict[str, Any]:
    tp = fp = fn = pred_count = gt_count = 0
    for result in file_results:
        cur: Any = result
        for key in key_path:
            cur = cur[key]
        tp += int(cur["tp"])
        fp += int(cur["fp"])
        fn += int(cur["fn"])
        pred_count += int(cur["pred_count"])
        gt_count += int(cur["gt_count"])
    return calculate_metrics_from_counts(tp, fp, fn, pred_count, gt_count, threshold)


def aggregate_macro_metrics(file_results: Sequence[Dict[str, Any]], key_path: Tuple[str, ...], threshold: float) -> Dict[str, Any]:
    metrics: List[Dict[str, Any]] = []
    for result in file_results:
        cur: Any = result
        for key in key_path:
            cur = cur[key]
        if int(cur["pred_count"]) == 0 and int(cur["gt_count"]) == 0:
            continue
        metrics.append(cur)

    if not metrics:
        return {
            "file_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "threshold": threshold,
        }

    count = len(metrics)
    return {
        "file_count": count,
        "precision": sum(float(metric["precision"]) for metric in metrics) / count,
        "recall": sum(float(metric["recall"]) for metric in metrics) / count,
        "f1_score": sum(float(metric["f1_score"]) for metric in metrics) / count,
        "threshold": threshold,
    }


def evaluate_batch(pred_dir: Path, gt_dir: Path, node_threshold: float, edge_threshold: float) -> Dict[str, Any]:
    pred_files = {p.name: p for p in pred_dir.glob("*.json")}
    gt_files = {p.name: p for p in gt_dir.glob("*.json")}

    common_names = sorted(set(pred_files.keys()) & set(gt_files.keys()))

    file_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for name in common_names:
        pred_path = pred_files[name]
        gt_path = gt_files[name]
        try:
            file_results.append(
                evaluate_one_file(
                    pred_path=pred_path,
                    gt_path=gt_path,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "file_name": name,
                    "pred_file": str(pred_path),
                    "gt_file": str(gt_path),
                    "error": str(exc),
                }
            )

    valid_results = file_results
    summary = {
        "file_count_total_common": len(common_names),
        "file_count_evaluated": len(valid_results),
        "file_count_errors": len(errors),
        "node_metrics": {
            "macro": {
                "overall": aggregate_macro_metrics(valid_results, ("node_metrics", "overall"), node_threshold),
                "high_level": aggregate_macro_metrics(valid_results, ("node_metrics", "high_level"), node_threshold),
                "ttp": aggregate_macro_metrics(valid_results, ("node_metrics", "ttp"), node_threshold),
                "ioc": aggregate_macro_metrics(valid_results, ("node_metrics", "ioc"), node_threshold),
            },
            "micro": {
                "overall": aggregate_metrics(valid_results, ("node_metrics", "overall"), node_threshold),
                "high_level": aggregate_metrics(valid_results, ("node_metrics", "high_level"), node_threshold),
                "ttp": aggregate_metrics(valid_results, ("node_metrics", "ttp"), node_threshold),
                "ioc": aggregate_metrics(valid_results, ("node_metrics", "ioc"), node_threshold),
            },
        },
        "edge_metrics": {
            "macro": {
                "edges": aggregate_macro_metrics(valid_results, ("edge_metrics",), edge_threshold),
            },
            "micro": {
                "edges": aggregate_metrics(valid_results, ("edge_metrics",), edge_threshold),
            },
        },
        "edge_type_metrics": {
            "macro": {
                "high_level-ttp": aggregate_macro_metrics(
                    valid_results,
                    ("edge_type_metrics", "high_level-ttp"),
                    edge_threshold,
                ),
                "ioc-ttp": aggregate_macro_metrics(
                    valid_results,
                    ("edge_type_metrics", "ioc-ttp"),
                    edge_threshold,
                ),
            },
            "micro": {
                "high_level-ttp": aggregate_metrics(
                    valid_results,
                    ("edge_type_metrics", "high_level-ttp"),
                    edge_threshold,
                ),
                "ioc-ttp": aggregate_metrics(
                    valid_results,
                    ("edge_type_metrics", "ioc-ttp"),
                    edge_threshold,
                ),
            },
        },
    }

    return {
        "mode": "batch",
        "config": {
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir),
            "node_threshold": node_threshold,
            "edge_threshold": edge_threshold,
            "node_type_constrained": True,
            "node_subtype_constrained": True,
            "edge_type_constrained": True,
            "high_level_matching": "same_subtype_fuzzy_without_partial_substring",
            "ttp_matching": "normalized_attack_id_exact",
            "ioc_matching": "same_subtype_fuzzy_without_value_canonicalization_or_partial_substring",
            "ioc_subtype_aliases": SUBTYPE_ALIASES,
            "assignment": "one_to_one_greedy",
            "macro_empty_policy": "exclude_files_where_pred_and_gt_are_both_empty",
        },
        "summary": summary,
        "files": valid_results,
        "errors": errors,
    }


def format_metric_line(name: str, metric: Dict[str, Any]) -> str:
    return (
        f"{name:<18} P={metric['precision']:.4f} "
        f"R={metric['recall']:.4f} "
        f"F1={metric['f1_score']:.4f} "
        f"(TP={metric['tp']}, FP={metric['fp']}, FN={metric['fn']}, "
        f"Pred={metric['pred_count']}, GT={metric['gt_count']})"
    )


def format_macro_metric_line(name: str, metric: Dict[str, Any]) -> str:
    return (
        f"{name:<18} P={metric['precision']:.4f} "
        f"R={metric['recall']:.4f} "
        f"F1={metric['f1_score']:.4f} "
        f"(Files={metric['file_count']})"
    )


def print_single_result(result: Dict[str, Any]) -> None:
    print("=" * 88)
    print(f"File: {result['file_name']}")
    print("-" * 88)
    print("Node metrics")
    print(format_metric_line("overall", result["node_metrics"]["overall"]))
    print(format_metric_line("high_level", result["node_metrics"]["high_level"]))
    print(format_metric_line("ttp", result["node_metrics"]["ttp"]))
    print(format_metric_line("ioc", result["node_metrics"]["ioc"]))
    print("-" * 88)
    print("Edge metrics")
    print(format_metric_line("edges", result["edge_metrics"]))
    print("-" * 88)
    print("Edge type metrics")
    print(format_metric_line("high_level-ttp", result["edge_type_metrics"]["high_level-ttp"]))
    print(format_metric_line("ioc-ttp", result["edge_type_metrics"]["ioc-ttp"]))
    print("=" * 88)


def validate_threshold(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RQ1 prediction against GT with broad-type and subtype constraints, "
            "non-partial fuzzy High-Level/IoC matching, and exact normalized TTP IDs."
        )
    )
    parser.add_argument(
        "pred_path",
        type=str,
        help="Prediction file path (single mode) or prediction directory (batch mode).",
    )
    parser.add_argument(
        "gt_path",
        type=str,
        help="GT file path (single mode) or GT directory (batch mode).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode: compare same-name JSON files in two directories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (required in batch mode).",
    )
    parser.add_argument(
        "--node-threshold",
        type=float,
        default=0.85,
        help="Fuzzy threshold for same-subtype High-Level/IoC node matching, default 0.85.",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.85,
        help="Fuzzy threshold for the same-subtype non-TTP endpoint of same-type edges, default 0.85.",
    )
    parser.add_argument(
        "--summary-average",
        choices=("macro", "micro"),
        default="macro",
        help=(
            "Batch-mode terminal summary to print. The output JSON always stores both "
            "macro and micro metrics; default: macro."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    validate_threshold("node-threshold", args.node_threshold)
    validate_threshold("edge-threshold", args.edge_threshold)

    pred_path = Path(args.pred_path)
    gt_path = Path(args.gt_path)

    if args.batch:
        if not pred_path.is_dir():
            raise FileNotFoundError(f"Prediction directory not found: {pred_path}")
        if not gt_path.is_dir():
            raise FileNotFoundError(f"GT directory not found: {gt_path}")
        if not args.output:
            parser.error("--batch mode requires --output")

        result = evaluate_batch(
            pred_dir=pred_path,
            gt_dir=gt_path,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Batch evaluation done. Output written to: {output_path}")
        print("-" * 88)
        print(f"Evaluated files: {result['summary']['file_count_evaluated']}")
        print(f"Errors: {result['summary']['file_count_errors']}")
        print("-" * 88)
        summary_average = args.summary_average
        summary_formatter = (
            format_macro_metric_line if summary_average == "macro" else format_metric_line
        )
        print(f"Node summary ({summary_average})")
        print(
            summary_formatter(
                "overall",
                result["summary"]["node_metrics"][summary_average]["overall"],
            )
        )
        print(
            summary_formatter(
                "high_level",
                result["summary"]["node_metrics"][summary_average]["high_level"],
            )
        )
        print(
            summary_formatter(
                "ttp",
                result["summary"]["node_metrics"][summary_average]["ttp"],
            )
        )
        print(
            summary_formatter(
                "ioc",
                result["summary"]["node_metrics"][summary_average]["ioc"],
            )
        )
        print("-" * 88)
        print(f"Edge summary ({summary_average})")
        print(
            summary_formatter(
                "edges",
                result["summary"]["edge_metrics"][summary_average]["edges"],
            )
        )
        return

    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    result = evaluate_one_file(
        pred_path=pred_path,
        gt_path=gt_path,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
    )
    print_single_result(result)

    if args.output:
        print("Single mode ignores --output and only prints to terminal as requested.")


if __name__ == "__main__":
    main()
