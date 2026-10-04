from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .deduper import dedupe_edges
from .embedder import DisabledEmbedder, OpenAIEmbedder
from .llm_client import (
    BaseLLMClient,
    DisabledLLMClient,
    LLMClusterResponse,
    OpenAILLMClient,
    filter_llm_clusters,
)
from .loader import load_graphs, resolve_input_files
from .logger import AlignmentLogger
from .matcher import (
    CandidateGroup,
    DeterministicCluster,
    highlevel_char_phase,
    highlevel_embedding_phase,
    highlevel_neighbor_phase,
    ioc_canonical_clusters,
    ttp_exact_clusters,
)
from .merger import GraphStore
from .normalizer import SUPPORTED_IOC_TYPES, canonicalize_ioc_value, normalize_ioc_type
from .path_guard import PathValidationError, ensure_output_dir, ensure_within_project


HIGHLEVEL_STAGE_ORDER = ("char", "embed", "neighbor")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _resolve_highlevel_stages(
    *,
    enable_llm: bool,
    requested_stages: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Return enabled non-deterministic HighLevel stages in pipeline order."""
    if not enable_llm:
        return ()

    unknown_stages = sorted(set(requested_stages) - set(HIGHLEVEL_STAGE_ORDER))
    if unknown_stages:
        raise ValueError(
            "Unsupported HighLevel stage(s): " + ", ".join(unknown_stages)
        )

    requested = set(requested_stages)
    return tuple(stage for stage in HIGHLEVEL_STAGE_ORDER if stage in requested)


def _resolve_highlevel_char_modes(
    *,
    enabled_highlevel_stages: tuple[str, ...],
    disable_deterministic_char: bool,
) -> tuple[bool, bool]:
    """Return (deterministic_enabled, fuzzy_candidates_enabled)."""
    return (
        not disable_deterministic_char,
        "char" in enabled_highlevel_stages,
    )


def _build_highlevel_char_work(
    high_nodes: list[dict[str, Any]],
    *,
    deterministic_enabled: bool,
    fuzzy_candidates_enabled: bool,
) -> tuple[list[DeterministicCluster], list[CandidateGroup]]:
    """Build only the deterministic/fuzzy char work enabled for this run."""
    if not deterministic_enabled and not fuzzy_candidates_enabled:
        return [], []

    direct_clusters, char_candidates = highlevel_char_phase(
        high_nodes,
        build_candidates=fuzzy_candidates_enabled,
    )
    if not deterministic_enabled:
        direct_clusters = []
    return direct_clusters, char_candidates


def _dedupe_after_stage(store: GraphStore) -> int:
    removed = dedupe_edges(store.edges, store.inactive_edges)
    store.rebuild_node_edge_refs()
    return removed


def _apply_deterministic(
    store: GraphStore,
    clusters: list[DeterministicCluster],
    *,
    stage: str,
    logger: AlignmentLogger,
) -> int:
    merged = 0
    for cluster in clusters:
        merged += store.apply_merge_cluster(
            node_ids=cluster.node_ids,
            stage=stage,
            reason=cluster.reason,
            logger=logger,
        )
    return merged


def _extract_evidence_sentences(node: dict[str, Any]) -> list[str]:
    source = node.get("source", {})
    if isinstance(source, dict):
        out: list[str] = []
        for value in source.values():
            if isinstance(value, dict):
                best_span_sentences_list = value.get("best_span_sentences_list")
                if not isinstance(best_span_sentences_list, list):
                    continue
                for span_sentences in best_span_sentences_list:
                    if not isinstance(span_sentences, list):
                        continue
                    for sentence in span_sentences:
                        out.append(str(sentence))
                        if len(out) >= 8:
                            return out
                continue
            if not isinstance(value, list):
                continue
            for pair in value:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    out.append(str(pair[1]))
                    if len(out) >= 8:
                        return out
        return out
    return []


def _extract_source(node: dict[str, Any]) -> dict[str, Any]:
    source = node.get("source", {})
    if isinstance(source, dict):
        return source
    return {}


def _build_llm_nodes_payload(
    store: GraphStore,
    node_ids: list[str],
    *,
    use_source: bool,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for node_id in node_ids:
        if not store.has_node(node_id):
            continue
        node = store.active_nodes[node_id]
        item: dict[str, Any] = {
            "node_id": node_id,
            "node_text": str(node.get("node_text", "")),
        }
        if use_source:
            item["source"] = _extract_source(node)
        else:
            item["evidence_sentences"] = _extract_evidence_sentences(node)
            if not item["evidence_sentences"]:
                item["source"] = _extract_source(node)
        payload.append(item)
    return payload


def _llm_node_confidence(
    response: LLMClusterResponse,
    *,
    node_id: str,
    cluster_idx: int,
) -> float:
    node_map = response.node_confidence.get(node_id, {})
    return float(node_map.get(str(cluster_idx), node_map.get("0", 0.0)))


def _build_candidate_log_nodes(
    payload: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    return [
        (str(item.get("node_id", "")), str(item.get("node_text", "")))
        for item in payload
    ]


def _build_llm_rejection_summary(
    payload: list[dict[str, Any]],
    response: LLMClusterResponse,
    filtered_clusters: list[Any],
    *,
    node_threshold: float,
    cluster_threshold: float,
) -> dict[str, Any] | None:
    text_by_node_id = {
        str(item.get("node_id", "")): str(item.get("node_text", "")) for item in payload
    }
    clustered_node_ids: set[str] = set()
    rejected_clusters: list[dict[str, Any]] = []

    for idx, cluster in enumerate(response.clusters):
        cluster_conf = (
            response.cluster_confidence[idx]
            if idx < len(response.cluster_confidence)
            else 0.0
        )
        clustered_node_ids.update(cluster)

        kept_nodes: list[dict[str, Any]] = []
        rejected_nodes: list[dict[str, Any]] = []
        for node_id in cluster:
            confidence = _llm_node_confidence(
                response,
                node_id=node_id,
                cluster_idx=idx,
            )
            node_info = {
                "node_id": node_id,
                "node_text": text_by_node_id.get(node_id, ""),
                "confidence": confidence,
            }
            if confidence >= node_threshold:
                kept_nodes.append(node_info)
            else:
                rejected_nodes.append(node_info)

        rejection_reasons: list[str] = []
        if cluster_conf < cluster_threshold:
            rejection_reasons.append("cluster_confidence_below_threshold")
        if rejected_nodes:
            rejection_reasons.append("node_confidence_below_threshold")
        if cluster_conf >= cluster_threshold and len(kept_nodes) < 2:
            rejection_reasons.append("fewer_than_two_nodes_after_node_threshold")

        if rejection_reasons:
            rejected_clusters.append(
                {
                    "cluster_index": idx,
                    "cluster_node_ids": cluster,
                    "cluster_confidence": cluster_conf,
                    "kept_nodes": kept_nodes,
                    "rejected_nodes": rejected_nodes,
                    "rejection_reasons": rejection_reasons,
                }
            )

    unclustered_nodes = [
        {
            "node_id": node_id,
            "node_text": node_text,
            "reason": "not_returned_in_any_llm_cluster",
        }
        for node_id, node_text in text_by_node_id.items()
        if node_id not in clustered_node_ids
    ]

    if not rejected_clusters and not unclustered_nodes:
        return None

    summary = {
        "accepted_clusters": [
            {
                "node_ids": filt.node_ids,
                "cluster_confidence": filt.cluster_confidence,
                "node_confidence": filt.node_confidence,
            }
            for filt in filtered_clusters
        ],
        "rejected_clusters": rejected_clusters,
        "unclustered_nodes": unclustered_nodes,
    }
    if not response.clusters:
        summary["group_rejection_reason"] = "llm_returned_no_clusters"
    return summary


def _apply_llm_candidates(
    store: GraphStore,
    candidate_groups: list[CandidateGroup],
    *,
    stage: str,
    llm_client: BaseLLMClient,
    node_threshold: float,
    cluster_threshold: float,
    logger: AlignmentLogger,
    use_source: bool,
    log_rejected_llm: bool = False,
) -> int:
    merged = 0
    for group in candidate_groups:
        node_ids = [node_id for node_id in group.node_ids if store.has_node(node_id)]
        if len(node_ids) < 2:
            continue
        payload = _build_llm_nodes_payload(store, node_ids, use_source=use_source)
        if len(payload) < 2:
            continue

        response = llm_client.cluster_group(
            stage=stage,
            node_type=group.node_type,
            nodes=payload,
        )
        filtered_clusters = filter_llm_clusters(
            response,
            node_threshold=node_threshold,
            cluster_threshold=cluster_threshold,
        )
        if log_rejected_llm:
            rejection_summary = _build_llm_rejection_summary(
                payload,
                response,
                filtered_clusters,
                node_threshold=node_threshold,
                cluster_threshold=cluster_threshold,
            )
            if rejection_summary is not None:
                reason = {
                    "rule": "llm_candidate_rejected",
                    "candidate_rule": group.reason.get("rule"),
                    "candidate_group": group.reason,
                    "node_type": group.node_type,
                    "node_threshold": node_threshold,
                    "cluster_threshold": cluster_threshold,
                    "source_mode": "source" if use_source else "evidence_sentences",
                    **rejection_summary,
                }
                logger.log_llm_rejection(
                    stage=stage,
                    candidate_nodes=_build_candidate_log_nodes(payload),
                    reason=reason,
                )
        for filt in filtered_clusters:
            reason = {
                "rule": "llm_cluster",
                "candidate_rule": group.reason.get("rule"),
                "node_type": group.node_type,
                "cluster_confidence": filt.cluster_confidence,
                "node_confidence": filt.node_confidence,
                "node_threshold": node_threshold,
                "cluster_threshold": cluster_threshold,
                "source_mode": "source" if use_source else "evidence_sentences",
            }
            merged += store.apply_merge_cluster(
                node_ids=filt.node_ids,
                stage=stage,
                reason=reason,
                logger=logger,
            )
    return merged


def _process_ioc_canonicalization(store: GraphStore) -> None:
    for node in store.get_nodes("ioc"):
        ioc_type = normalize_ioc_type(str(node.get("node_type", "")))
        raw_value = str(node.get("node_text", ""))
        raw, canonical = canonicalize_ioc_value(ioc_type, raw_value)
        node["raw_value"] = raw
        node["canonical_value"] = canonical
        if ioc_type in SUPPORTED_IOC_TYPES:
            node["normalized_value"] = canonical


def _build_llm_client(args: argparse.Namespace) -> BaseLLMClient:
    if not args.enable_llm:
        return DisabledLLMClient()
    return OpenAILLMClient(
        model=args.openai_model,
        api_key_env=args.openai_api_key_env,
    )


def _build_embedder(
    args: argparse.Namespace,
    *,
    enabled_highlevel_stages: tuple[str, ...],
):
    if not args.enable_llm or "embed" not in enabled_highlevel_stages:
        return DisabledEmbedder()
    if args.embed_backend != "openai":
        raise ValueError(f"Unsupported embed backend: {args.embed_backend}")
    return OpenAIEmbedder(
        model=args.embed_model,
        api_key_env=args.openai_api_key_env,
    )


def _run_pipeline(args: argparse.Namespace) -> Path:
    input_files = resolve_input_files(args.input_paths)
    store, merged_from_files = load_graphs(input_files)

    output_dir = ensure_output_dir(args.output_dir)
    log_path = (
        ensure_within_project(args.log_path, must_exist=False)
        if args.log_path
        else ensure_within_project(output_dir / "alignment.log", must_exist=False)
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = AlignmentLogger(log_path)

    enabled_highlevel_stages = _resolve_highlevel_stages(
        enable_llm=args.enable_llm,
        requested_stages=args.highlevel_stages,
    )
    (
        highlevel_deterministic_char_enabled,
        fuzzy_char_enabled,
    ) = _resolve_highlevel_char_modes(
        enabled_highlevel_stages=enabled_highlevel_stages,
        disable_deterministic_char=args.disable_highlevel_deterministic_char,
    )
    llm_client = _build_llm_client(args)
    embedder = _build_embedder(
        args,
        enabled_highlevel_stages=enabled_highlevel_stages,
    )
    node_threshold = _clamp(float(args.node_threshold))
    cluster_threshold = _clamp(float(args.cluster_threshold))
    embed_sim_threshold = _clamp(float(args.embed_sim_threshold))
    neighbor_shared_ttp_threshold = int(args.neighbor_shared_ttp_threshold)
    if "neighbor" in enabled_highlevel_stages and neighbor_shared_ttp_threshold < 1:
        raise ValueError("--neighbor_shared_ttp_threshold must be >= 1")

    # Stage 1: TTP exact alignment
    ttp_clusters = ttp_exact_clusters(store.get_nodes("ttp"))
    _apply_deterministic(store, ttp_clusters, stage="TTP", logger=logger)
    _dedupe_after_stage(store)

    # Stage 2: IoC canonical alignment
    _process_ioc_canonicalization(store)
    ioc_clusters = ioc_canonical_clusters(store.get_nodes("ioc"))
    _apply_deterministic(store, ioc_clusters, stage="IoC", logger=logger)
    _dedupe_after_stage(store)

    # Stage 3.1: HighLevel char phase. Deterministic rules and fuzzy
    # candidates are independently switchable for component ablations.
    direct_clusters, char_candidates = _build_highlevel_char_work(
        store.get_nodes("high_level"),
        deterministic_enabled=highlevel_deterministic_char_enabled,
        fuzzy_candidates_enabled=fuzzy_char_enabled,
    )
    _apply_deterministic(
        store,
        direct_clusters,
        stage="HighLevel-char",
        logger=logger,
    )
    if fuzzy_char_enabled:
        _apply_llm_candidates(
            store,
            char_candidates,
            stage="HighLevel-char",
            llm_client=llm_client,
            node_threshold=node_threshold,
            cluster_threshold=cluster_threshold,
            logger=logger,
            use_source=False,
        )
    _dedupe_after_stage(store)

    # Stage 3.2: HighLevel embedding phase
    if "embed" in enabled_highlevel_stages:
        embed_candidates = highlevel_embedding_phase(
            store.get_nodes("high_level"),
            embedder=embedder,
            embed_sim_threshold=embed_sim_threshold,
        )
        _apply_llm_candidates(
            store,
            embed_candidates,
            stage="HighLevel-embed",
            llm_client=llm_client,
            node_threshold=node_threshold,
            cluster_threshold=cluster_threshold,
            logger=logger,
            use_source=True,
        )
    _dedupe_after_stage(store)

    # Stage 3.3: HighLevel neighbor rounds
    if "neighbor" in enabled_highlevel_stages:
        for round_idx in range(1, int(args.max_neighbor_rounds) + 1):
            ttp_neighbors = {
                str(node.get("node_id")): store.ttp_neighbors(str(node.get("node_id")))
                for node in store.get_nodes("high_level")
            }
            neighbor_candidates = highlevel_neighbor_phase(
                store.get_nodes("high_level"),
                ttp_neighbors_by_node=ttp_neighbors,
                shared_ttp_threshold=neighbor_shared_ttp_threshold,
            )
            merged = _apply_llm_candidates(
                store,
                neighbor_candidates,
                stage=f"HighLevel-neighbor-round{round_idx}",
                llm_client=llm_client,
                node_threshold=node_threshold,
                cluster_threshold=cluster_threshold,
                logger=logger,
                use_source=True,
                log_rejected_llm=True,
            )
            _dedupe_after_stage(store)
            if merged == 0:
                break

    store.ensure_all_sources()
    output_graph = store.to_output(merged_from_files)

    output_name = datetime.now().strftime("%Y%m%d_%H%M%S.json")
    output_path = ensure_within_project(output_dir / output_name, must_exist=False)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_graph, f, ensure_ascii=False, indent=2)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m CTIAlignmentor.align",
        description="Align and merge CTI graph JSON files.",
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="Input file paths and/or directories (supports comma-separated tokens).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory (created if missing).",
    )
    parser.add_argument(
        "--log_path",
        default=None,
        help="Optional log path (default: output_dir/alignment.log).",
    )
    parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.70,
        help="Minimum node confidence for keeping a node in an LLM cluster.",
    )
    parser.add_argument(
        "--cluster_threshold",
        type=float,
        default=0.70,
        help="Minimum cluster confidence for keeping an LLM cluster.",
    )
    parser.add_argument(
        "--enable_llm",
        action="store_true",
        help="Enable LLM-based candidate clustering.",
    )
    parser.add_argument(
        "--highlevel_stages",
        nargs="*",
        choices=HIGHLEVEL_STAGE_ORDER,
        default=list(HIGHLEVEL_STAGE_ORDER),
        help=(
            "Enabled non-deterministic HighLevel candidate stages. The execution "
            "order is always char, embed, neighbor regardless of argument order. "
            "HighLevel deterministic rules run by default. This option only "
            "takes effect with --enable_llm. An explicitly empty list disables all three channels."
        ),
    )
    parser.add_argument(
        "--disable_highlevel_deterministic_char",
        action="store_true",
        help=(
            "Disable type-specific deterministic HighLevel lexical and identifier "
            "canonicalization rules. TTP and IoC deterministic alignment remain "
            "enabled. Intended only for full character-stage ablation."
        ),
    )
    parser.add_argument(
        "--openai_model",
        default="gpt-5-mini",
        choices=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-mini-2025-08-07",
            "gpt-5.1",
            "gpt-5.6-luna",
        ],
        help="OpenAI model for LLM candidate clustering.",
    )
    parser.add_argument(
        "--openai_api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable name holding the OpenAI API key.",
    )
    parser.add_argument(
        "--embed_backend",
        default="openai",
        help="Embedding backend (currently: openai).",
    )
    parser.add_argument(
        "--embed_model",
        default="text-embedding-3-large",
        help="Embedding model name.",
    )
    parser.add_argument(
        "--embed_sim_threshold",
        type=float,
        default=0.84,
        help="Cosine similarity threshold for embedding candidate pairs.",
    )
    parser.add_argument(
        "--max_neighbor_rounds",
        type=int,
        default=3,
        help="Maximum rounds for HighLevel neighbor alignment.",
    )
    parser.add_argument(
        "--neighbor_shared_ttp_threshold",
        type=int,
        default=3,
        help="Minimum shared TTP neighbors required to form a HighLevel neighbor candidate edge.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = _run_pipeline(args)
    except (ValueError, PathValidationError, RuntimeError) as exc:
        print(f"[CTIAlignmentor] ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"[CTIAlignmentor] Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
