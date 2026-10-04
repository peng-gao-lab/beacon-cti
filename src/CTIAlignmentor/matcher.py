from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from . import normalizer
from .normalizer import canonicalize_cve


_ACRONYM_MATCH_TYPES = {
    "ThreatActor",
    "Organization",
    "Campaign",
    "MalwareFamily",
    "Tool",
    "Product",
    "Sector",
}
_ACRONYM_STOPWORDS = {"the", "and", "of"}


@dataclass
class DeterministicCluster:
    node_ids: list[str]
    reason: dict[str, Any]


@dataclass
class CandidateGroup:
    node_type: str
    node_ids: list[str]
    reason: dict[str, Any]


def _node_id(node: dict[str, Any]) -> str:
    return str(node.get("node_id", ""))


def _node_text(node: dict[str, Any]) -> str:
    return str(node.get("node_text", ""))


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _normalize_for_acronym_match(text: str) -> str:
    normalized = re.sub(r"[\W_]+", " ", (text or "").lower())
    parts = [part for part in normalized.split() if part and part not in _ACRONYM_STOPWORDS]
    return " ".join(parts)


def _acronym_candidate_edges(
    nodes: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    short_forms: dict[str, list[str]] = defaultdict(list)
    long_forms: list[tuple[str, str]] = []

    for node in nodes:
        node_id = _node_id(node)
        normalized = _normalize_for_acronym_match(_node_text(node))
        if not normalized:
            continue

        if 2 <= len(normalized) <= 10 and normalized.isalpha() and " " not in normalized:
            short_forms[normalized].append(node_id)

        words = normalized.split(" ")
        if len(words) >= 2:
            acronym = "".join(word[0] for word in words)
            long_forms.append((node_id, acronym))

    edges: set[tuple[str, str]] = set()
    for long_id, acronym in long_forms:
        if not acronym:
            continue
        min_match_len = math.ceil((2 * len(acronym)) / 3)
        for short_text, short_ids in short_forms.items():
            if len(short_text) < min_match_len:
                continue
            if short_text not in acronym:
                continue
            for short_id in short_ids:
                if long_id == short_id:
                    continue
                edges.add(_pair_key(long_id, short_id))
    return edges


def connected_components(
    node_ids: list[str], edges: list[tuple[str, str]]
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    used_nodes: set[str] = set()
    for node_a, node_b in edges:
        if node_a == node_b:
            continue
        if node_a not in adjacency or node_b not in adjacency:
            continue
        adjacency[node_a].add(node_b)
        adjacency[node_b].add(node_a)
        used_nodes.add(node_a)
        used_nodes.add(node_b)

    components: list[list[str]] = []
    visited: set[str] = set()
    for node in used_nodes:
        if node in visited:
            continue
        stack = [node]
        comp: list[str] = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        if len(comp) > 1:
            components.append(sorted(comp))
    return components


def jaro_winkler(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0

    len_l = len(left)
    len_r = len(right)
    match_distance = max(len_l, len_r) // 2 - 1
    left_matches = [False] * len_l
    right_matches = [False] * len_r

    matches = 0
    for i, ch in enumerate(left):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_r)
        for j in range(start, end):
            if right_matches[j] or ch != right[j]:
                continue
            left_matches[i] = True
            right_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    j = 0
    for i in range(len_l):
        if not left_matches[i]:
            continue
        while not right_matches[j]:
            j += 1
        if left[i] != right[j]:
            transpositions += 1
        j += 1
    transpositions /= 2

    jaro = (
        (matches / len_l)
        + (matches / len_r)
        + ((matches - transpositions) / matches)
    ) / 3

    prefix = 0
    for ch_l, ch_r in zip(left, right):
        if ch_l != ch_r:
            break
        prefix += 1
        if prefix == 4:
            break
    return jaro + 0.1 * prefix * (1 - jaro)


def levenshtein_norm(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    prev = list(range(len(right) + 1))
    for i, ch_l in enumerate(left, start=1):
        cur = [i]
        for j, ch_r in enumerate(right, start=1):
            cost = 0 if ch_l == ch_r else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    dist = prev[-1]
    return 1.0 - (dist / max(len(left), len(right)))


def _ngram_counter(text: str, n: int = 3) -> Counter[str]:
    if not text:
        return Counter()
    if len(text) < n:
        return Counter({text: 1})
    return Counter(text[i : i + n] for i in range(0, len(text) - n + 1))


def ngram3_cosine(left: str, right: str) -> float:
    c1 = _ngram_counter(left, n=3)
    c2 = _ngram_counter(right, n=3)
    if not c1 or not c2:
        return 0.0
    dot = sum(c1[k] * c2.get(k, 0) for k in c1)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def ngram3_jaccard(left: str, right: str) -> float:
    s1 = set(_ngram_counter(left, 3).keys())
    s2 = set(_ngram_counter(right, 3).keys())
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        n1 += a * a
        n2 += b * b
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(n1) * math.sqrt(n2))


def _pairs(nodes: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            out.append((nodes[i], nodes[j]))
    return out


def _clusters_by_key(
    nodes: list[dict[str, Any]],
    key_fn,
    reason: dict[str, Any],
) -> list[DeterministicCluster]:
    groups: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        key = key_fn(node)
        if not key:
            continue
        groups[key].append(_node_id(node))
    return [
        DeterministicCluster(node_ids=ids, reason=dict(reason))
        for ids in groups.values()
        if len(ids) > 1
    ]


def ttp_exact_clusters(ttp_nodes: list[dict[str, Any]]) -> list[DeterministicCluster]:
    groups: dict[str, list[str]] = defaultdict(list)
    for node in ttp_nodes:
        groups[_node_text(node)].append(_node_id(node))
    return [
        DeterministicCluster(
            node_ids=ids,
            reason={"rule": "ttp_exact_node_text", "confidence": 1.0},
        )
        for ids in groups.values()
        if len(ids) > 1
    ]


def ioc_canonical_clusters(ioc_nodes: list[dict[str, Any]]) -> list[DeterministicCluster]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for node in ioc_nodes:
        ioc_type = normalizer.normalize_ioc_type(str(node.get("node_type", "")))
        canonical = str(node.get("canonical_value", "")).strip()
        if ioc_type not in normalizer.SUPPORTED_IOC_TYPES:
            continue
        if not canonical:
            continue
        groups[(ioc_type, canonical)].append(_node_id(node))
    clusters: list[DeterministicCluster] = []
    for (ioc_type, canonical), ids in groups.items():
        if len(ids) < 2:
            continue
        clusters.append(
            DeterministicCluster(
                node_ids=ids,
                reason={
                    "rule": "ioc_canonical_exact",
                    "ioc_type": ioc_type,
                    "canonical_value": canonical,
                    "confidence": 1.0,
                },
            )
        )
    return clusters


def highlevel_char_phase(
    high_nodes: list[dict[str, Any]],
    *,
    build_candidates: bool = True,
) -> tuple[list[DeterministicCluster], list[CandidateGroup]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in high_nodes:
        by_type[str(node.get("node_type", "OtherStrict"))].append(node)

    deterministic: list[DeterministicCluster] = []
    candidates: list[CandidateGroup] = []

    for node_type, nodes in by_type.items():
        node_ids = [_node_id(n) for n in nodes]
        candidate_edges: set[tuple[str, str]] = set()

        if node_type == "ThreatActor":
            deterministic.extend(
                _clusters_by_key(
                    nodes,
                    key_fn=lambda n: normalizer.extract_actor_canonical_id(_node_text(n)),
                    reason={"rule": "actor_canonical_id_equal", "confidence": 1.0},
                )
            )
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    text_a = _node_text(node_a)
                    text_b = _node_text(node_b)
                    id_a = normalizer.extract_actor_canonical_id(text_a)
                    id_b = normalizer.extract_actor_canonical_id(text_b)
                    if id_a or id_b:
                        continue

                    a_join = normalizer.normalize_joined(text_a)
                    b_join = normalizer.normalize_joined(text_b)
                    if not a_join or not b_join:
                        continue
                    digits_a = normalizer.extract_digit_sequence(a_join)
                    digits_b = normalizer.extract_digit_sequence(b_join)
                    if digits_a and digits_b and digits_a != digits_b:
                        continue
                    score = 0.5 * jaro_winkler(a_join, b_join) + 0.5 * levenshtein_norm(
                        a_join, b_join
                    )
                    threshold = 0.95 if min(len(a_join), len(b_join)) <= 6 else 0.90
                    if score >= threshold:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        elif node_type == "Campaign":
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    a_space = normalizer.strip_campaign_prefix_once(
                        normalizer.normalize_basic(_node_text(node_a))
                    )
                    b_space = normalizer.strip_campaign_prefix_once(
                        normalizer.normalize_basic(_node_text(node_b))
                    )
                    a_join = a_space.replace(" ", "")
                    b_join = b_space.replace(" ", "")
                    score = (
                        0.50 * ngram3_cosine(a_space, b_space)
                        + 0.25 * jaro_winkler(a_join, b_join)
                        + 0.25 * levenshtein_norm(a_join, b_join)
                    )
                    threshold = 0.95 if min(len(a_join), len(b_join)) < 5 else 0.90
                    if score >= threshold:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        elif node_type == "MalwareFamily":
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    a_norm = normalizer.normalize_basic(_node_text(node_a))
                    b_norm = normalizer.normalize_basic(_node_text(node_b))
                    a_join = a_norm.replace(" ", "")
                    b_join = b_norm.replace(" ", "")
                    score = (
                        0.55 * jaro_winkler(a_join, b_join)
                        + 0.35 * levenshtein_norm(a_join, b_join)
                        + 0.10 * ngram3_jaccard(a_norm, b_norm)
                    )
                    threshold = 0.95 if min(len(a_join), len(b_join)) <= 6 else 0.90
                    if score >= threshold:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        elif node_type == "Tool":
            deterministic.extend(
                _clusters_by_key(
                    nodes,
                    key_fn=lambda n: normalizer.strip_tail_version_platform(
                        normalizer.normalize_basic(_node_text(n))
                    ),
                    reason={"rule": "tool_v1_exact", "confidence": 1.0},
                )
            )
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    a_norm = normalizer.normalize_basic(_node_text(node_a))
                    b_norm = normalizer.normalize_basic(_node_text(node_b))
                    a_join = a_norm.replace(" ", "")
                    b_join = b_norm.replace(" ", "")
                    score = (
                        0.50 * jaro_winkler(a_join, b_join)
                        + 0.35 * levenshtein_norm(a_join, b_join)
                        + 0.15 * ngram3_jaccard(a_norm, b_norm)
                    )
                    threshold = 0.98 if min(len(a_join), len(b_join)) <= 4 else 0.93
                    if score >= threshold:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        elif node_type == "Vulnerability":
            deterministic.extend(
                _clusters_by_key(
                    nodes,
                    key_fn=lambda n: canonicalize_cve(_node_text(n)),
                    reason={"rule": "vulnerability_cve_exact", "confidence": 1.0},
                )
            )

        elif node_type == "Organization":
            deterministic.extend(
                _clusters_by_key(
                    nodes,
                    key_fn=lambda n: normalizer.strip_org_suffix_once(
                        normalizer.normalize_us_abbr(normalizer.normalize_basic(_node_text(n)))
                    ),
                    reason={"rule": "organization_suffix_weaken_exact", "confidence": 1.0},
                )
            )
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    a_norm = normalizer.normalize_us_abbr(
                        normalizer.normalize_basic(_node_text(node_a))
                    )
                    b_norm = normalizer.normalize_us_abbr(
                        normalizer.normalize_basic(_node_text(node_b))
                    )
                    a_join = a_norm.replace(" ", "")
                    b_join = b_norm.replace(" ", "")
                    score = (
                        0.55 * ngram3_cosine(a_norm, b_norm)
                        + 0.30 * levenshtein_norm(a_join, b_join)
                        + 0.15 * jaro_winkler(a_join, b_join)
                    )
                    if score >= 0.90:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        elif node_type == "Product":
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    a_norm = normalizer.strip_tail_version_platform(
                        normalizer.normalize_basic(_node_text(node_a))
                    )
                    b_norm = normalizer.strip_tail_version_platform(
                        normalizer.normalize_basic(_node_text(node_b))
                    )
                    a_join = a_norm.replace(" ", "")
                    b_join = b_norm.replace(" ", "")
                    score = (
                        0.45 * jaro_winkler(a_join, b_join)
                        + 0.30 * levenshtein_norm(a_join, b_join)
                        + 0.25 * ngram3_cosine(a_norm, b_norm)
                    )
                    if score >= 0.92:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        elif node_type == "Location":
            short_nodes = [
                n for n in nodes if len(normalizer.normalize_basic(_node_text(n)).replace(" ", "")) <= 4
            ]
            deterministic.extend(
                _clusters_by_key(
                    short_nodes,
                    key_fn=lambda n: normalizer.normalize_us_abbr(
                        normalizer.normalize_basic(_node_text(n))
                    ),
                    reason={"rule": "location_short_exact", "confidence": 1.0},
                )
            )
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    a_norm = normalizer.normalize_us_abbr(
                        normalizer.normalize_basic(_node_text(node_a))
                    )
                    b_norm = normalizer.normalize_us_abbr(
                        normalizer.normalize_basic(_node_text(node_b))
                    )
                    a_join = a_norm.replace(" ", "")
                    b_join = b_norm.replace(" ", "")
                    if min(len(a_join), len(b_join)) <= 4:
                        continue
                    score = max(jaro_winkler(a_join, b_join), levenshtein_norm(a_join, b_join))
                    if score >= 0.93:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        elif node_type == "Sector":
            deterministic.extend(
                _clusters_by_key(
                    nodes,
                    key_fn=lambda n: normalizer.normalize_basic(_node_text(n)),
                    reason={"rule": "sector_exact", "confidence": 1.0},
                )
            )

        else:
            deterministic.extend(
                _clusters_by_key(
                    nodes,
                    key_fn=lambda n: normalizer.normalize_basic(_node_text(n)),
                    reason={"rule": "other_strict_exact", "confidence": 1.0},
                )
            )
            if build_candidates:
                for node_a, node_b in _pairs(nodes):
                    a_join = normalizer.normalize_joined(_node_text(node_a))
                    b_join = normalizer.normalize_joined(_node_text(node_b))
                    score = 0.5 * jaro_winkler(a_join, b_join) + 0.5 * levenshtein_norm(
                        a_join, b_join
                    )
                    if score >= 0.95:
                        candidate_edges.add(_pair_key(_node_id(node_a), _node_id(node_b)))

        if build_candidates:
            if node_type in _ACRONYM_MATCH_TYPES:
                candidate_edges.update(_acronym_candidate_edges(nodes))
            candidate_components = connected_components(node_ids, sorted(candidate_edges))
            for component in candidate_components:
                candidates.append(
                    CandidateGroup(
                        node_type=node_type,
                        node_ids=component,
                        reason={
                            "rule": "char_candidate_component",
                            "node_type": node_type,
                            "candidate_pairs": len(candidate_edges),
                        },
                    )
                )

    return deterministic, candidates


def _lightweight_char_filter(text_a: str, text_b: str) -> bool:
    a_norm = normalizer.normalize_basic(text_a)
    b_norm = normalizer.normalize_basic(text_b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    if set(a_norm.split()) & set(b_norm.split()):
        return True
    if ngram3_jaccard(a_norm, b_norm) >= 0.08:
        return True
    a_join = a_norm.replace(" ", "")
    b_join = b_norm.replace(" ", "")
    return jaro_winkler(a_join, b_join) >= 0.65


def highlevel_embedding_phase(
    high_nodes: list[dict[str, Any]],
    *,
    embedder: Any,
    embed_sim_threshold: float,
) -> list[CandidateGroup]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in high_nodes:
        by_type[str(node.get("node_type", "OtherStrict"))].append(node)

    candidate_groups: list[CandidateGroup] = []
    for node_type, nodes in by_type.items():
        eligible: list[dict[str, Any]] = []
        for node in nodes:
            text = _node_text(node)
            if len(text.strip()) < 4:
                continue
            if node_type == "Vulnerability":
                continue
            if node_type == "ThreatActor" and normalizer.extract_actor_canonical_id(text):
                continue
            eligible.append(node)

        if len(eligible) < 2:
            continue

        texts = [_node_text(node) for node in eligible]
        vectors = embedder.embed_texts(texts)
        if len(vectors) != len(eligible):
            continue

        candidate_edges: set[tuple[str, str]] = set()
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                left = eligible[i]
                right = eligible[j]
                if not _lightweight_char_filter(_node_text(left), _node_text(right)):
                    continue
                sim = cosine_similarity(vectors[i], vectors[j])
                if sim >= embed_sim_threshold:
                    candidate_edges.add(_pair_key(_node_id(left), _node_id(right)))

        components = connected_components(
            [_node_id(node) for node in eligible], sorted(candidate_edges)
        )
        for comp in components:
            candidate_groups.append(
                CandidateGroup(
                    node_type=node_type,
                    node_ids=comp,
                    reason={
                        "rule": "embedding_candidate_component",
                        "node_type": node_type,
                        "embed_sim_threshold": embed_sim_threshold,
                        "candidate_pairs": len(candidate_edges),
                    },
                )
            )
    return candidate_groups


def highlevel_neighbor_phase(
    high_nodes: list[dict[str, Any]],
    *,
    ttp_neighbors_by_node: dict[str, set[str]],
    shared_ttp_threshold: int = 3,
) -> list[CandidateGroup]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in high_nodes:
        by_type[str(node.get("node_type", "OtherStrict"))].append(node)

    groups: list[CandidateGroup] = []
    for node_type, nodes in by_type.items():
        if len(nodes) < 2:
            continue
        node_ids = [_node_id(node) for node in nodes]
        edges: list[tuple[str, str]] = []
        for node_a, node_b in _pairs(nodes):
            id_a = _node_id(node_a)
            id_b = _node_id(node_b)
            shared = ttp_neighbors_by_node.get(id_a, set()) & ttp_neighbors_by_node.get(
                id_b, set()
            )
            if len(shared) >= shared_ttp_threshold:
                edges.append((id_a, id_b))
        components = connected_components(node_ids, edges)
        for comp in components:
            groups.append(
                CandidateGroup(
                    node_type=node_type,
                    node_ids=comp,
                    reason={
                        "rule": "neighbor_shared_ttp",
                        "node_type": node_type,
                        "shared_ttp_threshold": shared_ttp_threshold,
                    },
                )
            )
    return groups
