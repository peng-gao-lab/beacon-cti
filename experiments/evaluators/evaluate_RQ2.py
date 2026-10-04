#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the results of the alignment operation in RQ2.

Recommended inputs:
1. Algorithm output root directory, e.g.:
   output/experiments/rq2_beacon/groups
   Each subdirectory under it represents one group; group/ctialignmentor contains the prediction JSON.
2. Single GT root directory, e.g.:
   dataset/BEACON-Group/single_gt
   This is the base input graph for alignment.
3. Alignment GT root directory, e.g.:
   dataset/BEACON-Group/alignment_gt
   This is the GT aligned graph derived from the Single GT.
4. Output path; the results are written as JSON and a summary is printed to the terminal.

Compatible inputs:
- The legacy 3 positional arguments are also supported:
  prediction_root aligned_groundtruth_root output_path
  In this case the script tries to infer group_report_single_GT from a sibling directory of aligned_groundtruth_root.

Evaluation definition:
- First, build base occurrence components from the Single GT. Each member is
  identified by entity_type + node_text/alias + report file stem, so same-named
  TTP/IOC/high-level nodes from different reports are not merged prematurely at the base stage.
- Then, preferentially read the source field of the algorithm output graph and the
  Alignment GT graph to map each final node back to its base occurrences in the
  Single GT; fall back to node_text+alias only when no parsable source is available.
- The number of new alignments (merge links) of a final cluster is computed as max(component_count - 1, 0).
- For any predicted component cluster P and GT component cluster G, if
  k = |P ∩ G|, the correct-alignment contribution is max(k - 1, 0).
- TP = sum_P sum_G max(|P ∩ G| - 1, 0)
- Ngt = sum_G (|G| - 1)
- Npred = sum_P (|P| - 1)
- FP = Npred - TP
- FN = Ngt - TP
- precision = TP / (TP + FP)
- recall = TP / (TP + FN)
- F1 = 2 * TP / (2 * TP + FP + FN)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Iterable


ENTITY_TYPES = ("high_level", "ttp", "ioc")
IGNORED_GROUP_DIR_NAMES = {"cost", "evaluation_result"}
AGGREGATE_COUNT_KEYS = (
    "tp",
    "fp",
    "fn",
    "predicted_alignment_count",
    "groundtruth_alignment_count",
)
RATE_KEYS = ("precision", "recall", "f1")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        if item not in self.parent:
            self.parent[item] = item

    def find(self, item: str) -> str:
        self.add(item)
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root


def normalize_member(value: object) -> str:
    """Normalize cluster member text to reduce formatting noise as much as possible."""
    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    if text.startswith("TTP:"):
        text = text[4:]

    return " ".join(text.split())


def to_string_list(value: object) -> list[str]:
    """Convert the alias field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def iter_node_lists(nodes: object) -> Iterable[list[dict]]:
    """Handle nodes being either a dict or a list."""
    if isinstance(nodes, dict):
        for node_list in nodes.values():
            if isinstance(node_list, list):
                yield [node for node in node_list if isinstance(node, dict)]
        return

    if isinstance(nodes, list):
        yield [node for node in nodes if isinstance(node, dict)]


def iter_node_lists_for_type(nodes: object, entity_type: str) -> Iterable[list[dict]]:
    """Extract the given type by nodes key; also supports the legacy list form."""
    if isinstance(nodes, dict):
        node_list = nodes.get(entity_type, [])
        if isinstance(node_list, list):
            yield [node for node in node_list if isinstance(node, dict)]
        return

    if isinstance(nodes, list):
        yield [
            node
            for node in nodes
            if isinstance(node, dict) and node.get("node_class") == entity_type
        ]


def iter_typed_nodes(data: dict, entity_type: str | None = None) -> Iterable[tuple[str, dict]]:
    """Iterate over nodes while preserving the entity type."""
    nodes = data.get("nodes", {})
    if isinstance(nodes, dict):
        if entity_type is not None:
            node_list = nodes.get(entity_type, [])
            if isinstance(node_list, list):
                for node in node_list:
                    if isinstance(node, dict):
                        yield entity_type, node
            return

        for node_type, node_list in nodes.items():
            if not isinstance(node_list, list):
                continue
            for node in node_list:
                if isinstance(node, dict):
                    yield str(node_type), node
        return

    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("node_class", "unknown"))
            if entity_type is None or node_type == entity_type:
                yield node_type, node


def make_member_id(entity_type: str, text: str, source_stem: str | None = None) -> str:
    if source_stem is None:
        return f"{entity_type}\t{text}"
    return f"{entity_type}\t{text}\t{source_stem}"


def extract_alias_member_ids(
    node: dict,
    entity_type: str,
    source_stem: str | None = None,
) -> set[str]:
    cluster: set[str] = set()

    node_text = normalize_member(node.get("node_text"))
    if node_text:
        cluster.add(make_member_id(entity_type, node_text, source_stem))

    for alias in to_string_list(node.get("alias", [])):
        alias_text = normalize_member(alias)
        if alias_text:
            cluster.add(make_member_id(entity_type, alias_text, source_stem))

    return cluster


def parse_source_key(source_key: str, base_stems: list[str]) -> tuple[str, str] | None:
    for stem in base_stems:
        suffix = f"_{stem}"
        if source_key.endswith(suffix):
            text = normalize_member(source_key[: -len(suffix)])
            if text:
                return text, stem
    return None


def extract_base_clusters(
    data_by_stem: Iterable[tuple[str, dict]],
    entity_type: str | None = None,
) -> list[set[str]]:
    clusters: list[set[str]] = []
    for source_stem, data in data_by_stem:
        for node_type, node in iter_typed_nodes(data, entity_type):
            cluster = extract_alias_member_ids(node, node_type, source_stem)
            if cluster:
                clusters.append(cluster)
    return clusters


def extract_final_clusters(
    data: dict,
    base_stems: list[str],
    entity_type: str | None = None,
) -> list[set[str]]:
    clusters: list[set[str]] = []
    for node_type, node in iter_typed_nodes(data, entity_type):
        source_cluster: set[str] = set()
        source = node.get("source")
        if isinstance(source, dict):
            for raw_key in source:
                parsed = parse_source_key(str(raw_key), base_stems)
                if parsed is None:
                    continue
                text, source_stem = parsed
                source_cluster.add(make_member_id(node_type, text, source_stem))

        if source_cluster:
            clusters.append(source_cluster)
            continue

        fallback_cluster = extract_alias_member_ids(node, node_type)
        if fallback_cluster:
            clusters.append(fallback_cluster)

    return clusters


def extract_clusters_from_node_lists(node_lists: Iterable[list[dict]]) -> list[set[str]]:
    clusters: list[set[str]] = []
    for node_list in node_lists:
        for node in node_list:
            cluster: set[str] = set()

            node_text = normalize_member(node.get("node_text"))
            if node_text:
                cluster.add(node_text)

            for alias in to_string_list(node.get("alias", [])):
                alias_text = normalize_member(alias)
                if alias_text:
                    cluster.add(alias_text)

            if cluster:
                clusters.append(cluster)

    return clusters


def extract_clusters(data: dict) -> list[set[str]]:
    """Extract the list of clusters from JSON."""
    return extract_clusters_from_node_lists(iter_node_lists(data.get("nodes", {})))


def extract_clusters_by_type(data: dict, entity_type: str) -> list[set[str]]:
    """Extract the list of clusters from JSON by entity type."""
    return extract_clusters_from_node_lists(
        iter_node_lists_for_type(data.get("nodes", {}), entity_type)
    )


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_component_lookup(base_clusters: Iterable[set[str]]) -> dict[str, str]:
    union_find = UnionFind()
    for cluster in base_clusters:
        members = sorted(member for member in cluster if member)
        if not members:
            continue

        first_member = members[0]
        union_find.add(first_member)
        for member in members[1:]:
            union_find.union(first_member, member)

    return {member: union_find.find(member) for member in union_find.parent}


def to_component_clusters(
    clusters: Iterable[set[str]],
    component_lookup: dict[str, str],
) -> list[set[str]]:
    component_clusters: list[set[str]] = []
    for cluster in clusters:
        component_cluster = {
            component_lookup.get(member, member)
            for member in cluster
            if member
        }
        if component_cluster:
            component_clusters.append(component_cluster)
    return component_clusters


def alignment_count(component_clusters: Iterable[set[str]]) -> int:
    return sum(max(len(cluster) - 1, 0) for cluster in component_clusters)


def calculate_tp(pred_clusters: list[set[str]], gt_clusters: list[set[str]]) -> int:
    """
    Compute the number of true positives according to TP = sum_P sum_G max(|P∩G|-1, 0).
    The input clusters have already been mapped to base component clusters.
    """
    member_to_gt_indices: dict[str, list[int]] = {}
    for gt_index, gt_cluster in enumerate(gt_clusters):
        for member in gt_cluster:
            member_to_gt_indices.setdefault(member, []).append(gt_index)

    tp = 0
    for pred_cluster in pred_clusters:
        overlap_sizes: dict[int, int] = {}
        for member in pred_cluster:
            for gt_index in member_to_gt_indices.get(member, []):
                overlap_sizes[gt_index] = overlap_sizes.get(gt_index, 0) + 1

        tp += sum(max(overlap_size - 1, 0) for overlap_size in overlap_sizes.values())

    return tp


def calculate_metrics(pred_clusters: list[set[str]], gt_clusters: list[set[str]]) -> dict:
    tp = calculate_tp(pred_clusters, gt_clusters)
    predicted_alignments = alignment_count(pred_clusters)
    groundtruth_alignments = alignment_count(gt_clusters)
    fp = predicted_alignments - tp
    fn = groundtruth_alignments - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_alignment_count": predicted_alignments,
        "groundtruth_alignment_count": groundtruth_alignments,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_cluster_count": len(pred_clusters),
        "groundtruth_cluster_count": len(gt_clusters),
        "predicted_non_singleton_cluster_count": sum(
            1 for cluster in pred_clusters if len(cluster) > 1
        ),
        "groundtruth_non_singleton_cluster_count": sum(
            1 for cluster in gt_clusters if len(cluster) > 1
        ),
    }


def calculate_macro_average(results: list[dict]) -> dict:
    if not results:
        return {
            **{key: 0.0 for key in AGGREGATE_COUNT_KEYS},
            **{key: 0.0 for key in RATE_KEYS},
        }

    return {
        **{key: mean(result[key] for result in results) for key in AGGREGATE_COUNT_KEYS},
        **{key: mean(result[key] for result in results) for key in RATE_KEYS},
    }


def calculate_micro_average(results: list[dict]) -> dict:
    if not results:
        return {
            **{key: 0 for key in AGGREGATE_COUNT_KEYS},
            **{key: 0.0 for key in RATE_KEYS},
        }

    total_tp = sum(result["tp"] for result in results)
    total_fp = sum(result["fp"] for result in results)
    total_fn = sum(result["fn"] for result in results)
    total_predicted_alignments = sum(
        result["predicted_alignment_count"] for result in results
    )
    total_groundtruth_alignments = sum(
        result["groundtruth_alignment_count"] for result in results
    )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    f1 = (
        2 * total_tp / (2 * total_tp + total_fp + total_fn)
        if 2 * total_tp + total_fp + total_fn > 0
        else 0.0
    )

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "predicted_alignment_count": total_predicted_alignments,
        "groundtruth_alignment_count": total_groundtruth_alignments,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calculate_summary_averages(group_results: list[dict], mode: str) -> dict:
    if mode == "macro":
        aggregate = calculate_macro_average
    elif mode == "micro":
        aggregate = calculate_micro_average
    else:
        raise ValueError(f"Unknown average mode: {mode}")

    summary = {
        "overall": aggregate(group_results),
    }
    for entity_type in ENTITY_TYPES:
        type_results = [result["type_results"][entity_type] for result in group_results]
        if mode == "macro":
            type_results = [
                result
                for result in type_results
                if result["predicted_alignment_count"] > 0
                or result["groundtruth_alignment_count"] > 0
            ]
        summary[entity_type] = aggregate(type_results)

    return summary


def format_metric_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_average_line(label: str, metrics: dict) -> str:
    return (
        f"{label}: "
        f"TP={format_metric_value(metrics['tp'])}, "
        f"FP={format_metric_value(metrics['fp'])}, "
        f"FN={format_metric_value(metrics['fn'])}, "
        f"Pred={format_metric_value(metrics['predicted_alignment_count'])}, "
        f"GT={format_metric_value(metrics['groundtruth_alignment_count'])}, "
        f"P={metrics['precision']:.4f}, "
        f"R={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}"
    )


def select_json_file(directory: Path) -> tuple[Path | None, list[str]]:
    """
    Prefer JSON files located directly in the directory; if there are none, search recursively.
    If multiple candidates are found, sort them by file name, pick the last one, and record a warning.
    """
    warnings: list[str] = []

    direct_candidates = sorted(
        path for path in directory.glob("*.json") if path.is_file()
    )
    if direct_candidates:
        candidates = direct_candidates
    else:
        candidates = sorted(
            path for path in directory.rglob("*.json") if path.is_file()
        )

    if not candidates:
        return None, warnings

    selected = candidates[-1]
    if len(candidates) > 1:
        warnings.append(
            f"Directory {directory} contains {len(candidates)} JSON files, using {selected.name}"
        )

    return selected, warnings


def select_base_json_files(directory: Path) -> tuple[list[Path], list[str]]:
    """
    The Single GT is the base for alignment; usually each group contains multiple report JSONs.
    Therefore, read all JSONs under the group instead of selecting only one.
    """
    warnings: list[str] = []

    direct_candidates = sorted(
        path for path in directory.glob("*.json") if path.is_file()
    )
    if direct_candidates:
        return direct_candidates, warnings

    recursive_candidates = sorted(
        path for path in directory.rglob("*.json") if path.is_file()
    )
    if recursive_candidates:
        warnings.append(
            f"Directory {directory} has no direct JSON files; found {len(recursive_candidates)} JSON files recursively"
        )
    return recursive_candidates, warnings


def resolve_output_path(raw_output_path: str) -> Path:
    output_path = Path(raw_output_path)
    if raw_output_path.endswith(os.sep) or (output_path.exists() and output_path.is_dir()):
        return output_path / "evaluate_RQ2.json"
    return output_path


def infer_single_groundtruth_root(aligned_groundtruth_root: Path) -> Path | None:
    candidates: list[Path] = []

    if aligned_groundtruth_root.name == "group_report_align_GT":
        candidates.append(aligned_groundtruth_root.parent / "group_report_single_GT")

    inferred_name = aligned_groundtruth_root.name.replace("align", "single")
    if inferred_name != aligned_groundtruth_root.name:
        candidates.append(aligned_groundtruth_root.parent / inferred_name)

    candidates.append(aligned_groundtruth_root.parent / "group_report_single_GT")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def list_group_directories(root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and not path.name.startswith("_")
        and path.name not in IGNORED_GROUP_DIR_NAMES
    }


def evaluate_cluster_sets(
    base_clusters: list[set[str]],
    pred_clusters: list[set[str]],
    gt_clusters: list[set[str]],
) -> dict:
    component_lookup = build_component_lookup(base_clusters)
    pred_component_clusters = to_component_clusters(pred_clusters, component_lookup)
    gt_component_clusters = to_component_clusters(gt_clusters, component_lookup)
    metrics = calculate_metrics(pred_component_clusters, gt_component_clusters)
    metrics["base_component_count"] = len(set(component_lookup.values()))
    metrics["base_non_singleton_cluster_count"] = sum(
        1 for cluster in base_clusters if len(cluster) > 1
    )
    return metrics


def evaluate_group(
    group_name: str,
    prediction_group_dir: Path,
    single_groundtruth_group_dir: Path,
    aligned_groundtruth_group_dir: Path,
) -> dict:
    warnings: list[str] = []

    prediction_json, prediction_warnings = select_json_file(
        prediction_group_dir / "ctialignmentor"
    )
    warnings.extend(prediction_warnings)

    single_groundtruth_jsons, single_groundtruth_warnings = select_base_json_files(
        single_groundtruth_group_dir
    )
    warnings.extend(single_groundtruth_warnings)

    aligned_groundtruth_json, aligned_groundtruth_warnings = select_json_file(
        aligned_groundtruth_group_dir
    )
    warnings.extend(aligned_groundtruth_warnings)

    if prediction_json is None:
        raise FileNotFoundError(f"No prediction JSON found in {prediction_group_dir / 'ctialignmentor'}")
    if not single_groundtruth_jsons:
        raise FileNotFoundError(f"No Single GT JSON found in {single_groundtruth_group_dir}")
    if aligned_groundtruth_json is None:
        raise FileNotFoundError(f"No Alignment GT JSON found in {aligned_groundtruth_group_dir}")

    prediction_data = load_json(prediction_json)
    single_groundtruth_data = [
        (path.stem, load_json(path))
        for path in single_groundtruth_jsons
    ]
    single_groundtruth_stems = sorted(
        (path.stem for path in single_groundtruth_jsons),
        key=len,
        reverse=True,
    )
    aligned_groundtruth_data = load_json(aligned_groundtruth_json)

    base_clusters = extract_base_clusters(single_groundtruth_data)
    pred_clusters = extract_final_clusters(prediction_data, single_groundtruth_stems)
    gt_clusters = extract_final_clusters(aligned_groundtruth_data, single_groundtruth_stems)
    metrics = evaluate_cluster_sets(base_clusters, pred_clusters, gt_clusters)

    type_results = {}
    for entity_type in ENTITY_TYPES:
        base_type_clusters = extract_base_clusters(single_groundtruth_data, entity_type)
        pred_type_clusters = extract_final_clusters(
            prediction_data,
            single_groundtruth_stems,
            entity_type,
        )
        gt_type_clusters = extract_final_clusters(
            aligned_groundtruth_data,
            single_groundtruth_stems,
            entity_type,
        )
        type_results[entity_type] = evaluate_cluster_sets(
            base_type_clusters,
            pred_type_clusters,
            gt_type_clusters,
        )

    return {
        "group": group_name,
        "prediction_file": str(prediction_json),
        "single_groundtruth_files": [str(path) for path in single_groundtruth_jsons],
        "groundtruth_file": str(aligned_groundtruth_json),
        "aligned_groundtruth_file": str(aligned_groundtruth_json),
        "warnings": warnings,
        "type_results": type_results,
        **metrics,
    }


def parse_paths(parser: argparse.ArgumentParser, raw_paths: list[str]) -> tuple[Path, Path, Path, Path, str]:
    if len(raw_paths) == 3:
        prediction_root = Path(raw_paths[0])
        aligned_groundtruth_root = Path(raw_paths[1])
        output_path = resolve_output_path(raw_paths[2])
        single_groundtruth_root = infer_single_groundtruth_root(aligned_groundtruth_root)
        if single_groundtruth_root is None:
            parser.error(
                "The 3-argument compatibility mode could not infer the Single GT directory; please use the 4-argument mode: "
                "prediction_root single_groundtruth_root aligned_groundtruth_root output_path"
            )
        return (
            prediction_root,
            single_groundtruth_root,
            aligned_groundtruth_root,
            output_path,
            "inferred_single_groundtruth_root",
        )

    if len(raw_paths) == 4:
        return (
            Path(raw_paths[0]),
            Path(raw_paths[1]),
            Path(raw_paths[2]),
            resolve_output_path(raw_paths[3]),
            "explicit_single_groundtruth_root",
        )

    parser.error(
        "Wrong number of arguments. Usage: evaluate_RQ2.py prediction_root single_groundtruth_root "
        "aligned_groundtruth_root output_path"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the results of the alignment operation in RQ2, computing precision, recall "
            "and F1 based on the new alignments from Single GT -> final graph."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Recommended 4 positional arguments: prediction_root single_groundtruth_root "
            "aligned_groundtruth_root output_path; legacy 3 arguments also supported: "
            "prediction_root aligned_groundtruth_root output_path"
        ),
    )
    args = parser.parse_args()

    (
        prediction_root,
        single_groundtruth_root,
        aligned_groundtruth_root,
        output_path,
        input_mode,
    ) = parse_paths(parser, args.paths)

    if not prediction_root.is_dir():
        raise NotADirectoryError(f"Prediction directory does not exist or is not a directory: {prediction_root}")
    if not single_groundtruth_root.is_dir():
        raise NotADirectoryError(f"Single GT directory does not exist or is not a directory: {single_groundtruth_root}")
    if not aligned_groundtruth_root.is_dir():
        raise NotADirectoryError(f"Alignment GT directory does not exist or is not a directory: {aligned_groundtruth_root}")

    prediction_groups = list_group_directories(prediction_root)
    single_groundtruth_groups = list_group_directories(single_groundtruth_root)
    aligned_groundtruth_groups = list_group_directories(aligned_groundtruth_root)
    all_group_names = sorted(
        set(prediction_groups) | set(single_groundtruth_groups) | set(aligned_groundtruth_groups)
    )

    group_results: list[dict] = []
    skipped_groups: list[dict] = []

    for group_name in all_group_names:
        prediction_group_dir = prediction_groups.get(group_name)
        single_groundtruth_group_dir = single_groundtruth_groups.get(group_name)
        aligned_groundtruth_group_dir = aligned_groundtruth_groups.get(group_name)

        if prediction_group_dir is None:
            skipped_groups.append(
                {
                    "group": group_name,
                    "reason": "Group missing from the prediction directory",
                }
            )
            continue

        if single_groundtruth_group_dir is None:
            skipped_groups.append(
                {
                    "group": group_name,
                    "reason": "Group missing from the Single GT directory",
                }
            )
            continue

        if aligned_groundtruth_group_dir is None:
            skipped_groups.append(
                {
                    "group": group_name,
                    "reason": "Group missing from the Alignment GT directory",
                }
            )
            continue

        try:
            result = evaluate_group(
                group_name,
                prediction_group_dir,
                single_groundtruth_group_dir,
                aligned_groundtruth_group_dir,
            )
        except Exception as exc:  # noqa: BLE001
            skipped_groups.append(
                {
                    "group": group_name,
                    "reason": str(exc),
                }
            )
            continue

        group_results.append(result)

    macro_average = calculate_summary_averages(group_results, "macro")
    micro_average = calculate_summary_averages(group_results, "micro")

    output = {
        "evaluation_definition": "base_aware_alignment_delta",
        "input_mode": input_mode,
        "prediction_root": str(prediction_root),
        "single_groundtruth_root": str(single_groundtruth_root),
        "aligned_groundtruth_root": str(aligned_groundtruth_root),
        "groundtruth_root": str(aligned_groundtruth_root),
        "output_path": str(output_path),
        "prediction_group_count": len(prediction_groups),
        "single_groundtruth_group_count": len(single_groundtruth_groups),
        "groundtruth_group_count": len(aligned_groundtruth_groups),
        "aligned_groundtruth_group_count": len(aligned_groundtruth_groups),
        "processed_group_count": len(group_results),
        "skipped_group_count": len(skipped_groups),
        "macro_average": macro_average,
        "micro_average": micro_average,
        "group_results": group_results,
        "skipped_groups": skipped_groups,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("RQ2 Base-Aware Alignment Evaluation")
    print("=" * 72)
    print(f"Input mode:           {input_mode}")
    print(f"Prediction groups:    {len(prediction_groups)}")
    print(f"Single GT groups:     {len(single_groundtruth_groups)}")
    print(f"Align GT groups:      {len(aligned_groundtruth_groups)}")
    print(f"Evaluated groups:     {len(group_results)}")
    print(f"Skipped groups:       {len(skipped_groups)}")
    print("-" * 72)

    for result in group_results:
        print(
            f"[{result['group']}] "
            f"TP={result['tp']}, "
            f"FP={result['fp']}, "
            f"FN={result['fn']}, "
            f"P={result['precision']:.4f}, "
            f"R={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}"
        )

    if skipped_groups:
        print("-" * 72)
        print("Skipped groups:")
        for skipped in skipped_groups:
            print(f"  - {skipped['group']}: {skipped['reason']}")

    print("-" * 72)
    print("Macro Average:")
    print(f"  {format_average_line('overall', macro_average['overall'])}")
    for entity_type in ENTITY_TYPES:
        print(f"  {format_average_line(entity_type, macro_average[entity_type])}")

    print("Micro Average:")
    print(f"  {format_average_line('overall', micro_average['overall'])}")
    for entity_type in ENTITY_TYPES:
        print(f"  {format_average_line(entity_type, micro_average[entity_type])}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
