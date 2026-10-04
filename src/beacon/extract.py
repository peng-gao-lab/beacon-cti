"""Lightweight "extract" views of component outputs.

These functions are ports of the helper scripts that the original batch
pipelines called after each component:

* ``extract_high_level_from_ctihighlevelextractor.py`` -> :func:`extract_highlevel`
* ``extract_ttp_from_ctitagger.py``                    -> :func:`extract_tagger`
* ``extract_iocs.py``                                  -> :func:`extract_ioc`
* ``extract_graph_from_ctiaugmentor.py``               -> :func:`extract_augmentor`
* ``extract_graph_with_ttp_from_ctiaugmentor.py``      -> :func:`extract_alignmentor`

The transformation logic and the output file naming are kept as in the
scripts. The only differences are that errors raise exceptions instead of
calling ``sys.exit`` and that nothing is printed. The extract files are for
viewing only; downstream stages always consume the full component outputs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CTIHighLevelExtractor -> <stem>.txt, one entity_text per line
# ---------------------------------------------------------------------------
def extract_highlevel(json_file: Path, output_dir: Path) -> Path:
    data = _load_json(json_file)
    if "entities" not in data:
        raise ValueError("JSON file has no 'entities' field")
    entities = data["entities"]
    if not isinstance(entities, list):
        raise ValueError("'entities' is not a list")

    extracted_texts = []
    for entity in entities:
        if isinstance(entity, dict) and "entity_text" in entity:
            entity_text = entity["entity_text"]
            if entity_text:
                extracted_texts.append(entity_text)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{Path(json_file).stem}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for entity_text in extracted_texts:
            f.write(f"{entity_text}\n")
    return output_file


# ---------------------------------------------------------------------------
# CTITagger -> <name>.json, final TTPs merged by ttp_id
# ---------------------------------------------------------------------------
def extract_tagger(json_file: Path, output_dir: Path) -> Path:
    data = _load_json(json_file)
    if not isinstance(data, list):
        raise ValueError("JSON file should be a list")

    ttp_dict: dict[str, set] = defaultdict(set)
    for element in data:
        if not isinstance(element, dict):
            continue
        if "final_ttps" not in element:
            continue
        final_ttps = element["final_ttps"]
        if not isinstance(final_ttps, list):
            continue
        for ttp in final_ttps:
            if not isinstance(ttp, dict):
                continue
            if "ttp_id" not in ttp:
                continue
            ttp_id = ttp["ttp_id"]
            best_span_sentences = ttp.get("best_span_sentences", [])
            if not isinstance(best_span_sentences, list):
                continue
            ttp_dict[ttp_id].update(best_span_sentences)

    output_data = []
    for ttp_id, sentences_set in sorted(ttp_dict.items()):
        output_data.append({
            "ttp_id": ttp_id,
            "best_span_sentences": sorted(list(sentences_set)),
        })

    output_file = Path(output_dir) / Path(json_file).name
    _dump_json(output_data, output_file)
    return output_file


# ---------------------------------------------------------------------------
# CTIIOCExtractor -> <name>.json, list of {ioc_type, value}
# ---------------------------------------------------------------------------
def extract_ioc(json_file: Path, output_dir: Path) -> Path:
    data = _load_json(json_file)
    extracted_iocs = []
    if "iocs" in data and isinstance(data["iocs"], list):
        for ioc in data["iocs"]:
            extracted_iocs.append({
                "ioc_type": ioc.get("ioc_type", ""),
                "value": ioc.get("value", ""),
            })
    output_file = Path(output_dir) / Path(json_file).name
    _dump_json(extracted_iocs, output_file)
    return output_file


# ---------------------------------------------------------------------------
# CTIAugmentor -> <name>.json, nodes / edges / debug_edges by text
# ---------------------------------------------------------------------------
def _normalize_alias(alias_value: Any) -> list[str]:
    if isinstance(alias_value, list):
        return [str(alias) for alias in alias_value if str(alias).strip()]
    if isinstance(alias_value, str) and alias_value.strip():
        return [alias_value]
    return []


def _edge_texts(edge: dict, node_map: dict) -> tuple[str, str]:
    node_u_id = edge.get("node_u")
    node_v_id = edge.get("node_v")
    node_u_text = ""
    node_v_text = ""
    if node_u_id and node_u_id in node_map:
        node_u_text = node_map[node_u_id]["node_text"]
    if node_v_id and node_v_id in node_map:
        node_v_text = node_map[node_v_id]["node_text"]
    return node_u_text, node_v_text


def _edges_info(data: dict, node_map: dict) -> tuple[list, list]:
    edges_info = []
    for edge in data.get("edges", []):
        node_u_text, node_v_text = _edge_texts(edge, node_map)
        edges_info.append({
            "node_u_text": node_u_text,
            "node_v_text": node_v_text,
            "edge_type": edge.get("edge_type", ""),
        })

    debug_edges_info = []
    for edge in data.get("debug_edges", []):
        node_u_text, node_v_text = _edge_texts(edge, node_map)
        confidence = None
        verification = edge.get("verification", {})
        if verification:
            confidence = verification.get("confidence")
        debug_edges_info.append({
            "node_u_text": node_u_text,
            "node_v_text": node_v_text,
            "edge_type": edge.get("edge_type", ""),
            "confidence": confidence,
        })
    return edges_info, debug_edges_info


def extract_augmentor(json_file: Path, output_dir: Path) -> Path:
    data = _load_json(json_file)

    node_map: dict[str, dict] = {}
    for _node_class_key, node_list in data.get("nodes", {}).items():
        for node in node_list:
            node_id = node.get("node_id")
            if node_id:
                node_map[node_id] = {
                    # The original script labels this field with node_type.
                    "node_class": node.get("node_type", ""),
                    "node_text": node.get("node_text", ""),
                    "alias": _normalize_alias(node.get("alias", [])),
                }

    nodes_info = []
    for _node_id, node_info in node_map.items():
        node_entry = {
            "node_class": node_info["node_class"],
            "node_text": node_info["node_text"],
        }
        if node_info["alias"]:
            node_entry["alias"] = node_info["alias"]
        nodes_info.append(node_entry)

    edges_info, debug_edges_info = _edges_info(data, node_map)
    graph_info = {
        "nodes": nodes_info,
        "edges": edges_info,
        "debug_edges": debug_edges_info,
    }
    output_file = Path(output_dir) / Path(json_file).name
    _dump_json(graph_info, output_file)
    return output_file


# ---------------------------------------------------------------------------
# CTIAlignmentor -> <name>.json, as above plus per-node related TTP ids
# ---------------------------------------------------------------------------
def _ordered_unique(items: list) -> list:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _resolve_ttp_id(node_info: dict) -> str | None:
    ttp_id = node_info.get("ttp_id")
    if isinstance(ttp_id, str) and ttp_id.strip():
        return ttp_id
    node_id = node_info.get("node_id", "")
    if isinstance(node_id, str) and node_id.startswith("TTP_"):
        return node_id.removeprefix("TTP_")
    node_text = node_info.get("node_text", "")
    if isinstance(node_text, str) and node_text.startswith("TTP:"):
        return node_text.split("TTP:", 1)[1].strip()
    return None


def _related_ttp_ids(node_id: str, node_info: dict, node_map: dict, edge_map: dict) -> list:
    related_ttp_ids = []
    for edge_id in node_info.get("raw_edges", []):
        edge = edge_map.get(edge_id)
        if not edge:
            continue
        node_u_id = edge.get("node_u")
        node_v_id = edge.get("node_v")
        if node_u_id == node_id:
            other_node_id = node_v_id
        elif node_v_id == node_id:
            other_node_id = node_u_id
        else:
            continue
        other_node_info = node_map.get(other_node_id)
        if not other_node_info:
            continue
        other_ttp_id = _resolve_ttp_id(other_node_info)
        if other_ttp_id:
            related_ttp_ids.append(other_ttp_id)
    return _ordered_unique(related_ttp_ids)


def extract_alignmentor(json_file: Path, output_dir: Path) -> Path:
    data = _load_json(json_file)

    node_map: dict[str, dict] = {}
    for _node_class_key, node_list in data.get("nodes", {}).items():
        for node in node_list:
            node_id = node.get("node_id")
            if node_id:
                node_map[node_id] = {
                    "node_id": node_id,
                    "node_class": node.get("node_class", ""),
                    "node_type": node.get("node_type", ""),
                    "node_text": node.get("node_text", ""),
                    "alias": _normalize_alias(node.get("alias", [])),
                    "raw_edges": list(node.get("edges", [])),
                    "ttp_id": node.get("ttp_id", ""),
                }

    edge_map: dict[str, dict] = {}
    for edge in data.get("edges", []):
        edge_id = edge.get("edge_id")
        if edge_id:
            edge_map[edge_id] = edge

    nodes_info = []
    for node_id, node_info in node_map.items():
        node_entry = {
            "node_class": node_info["node_type"],
            "node_text": node_info["node_text"],
        }
        if node_info["alias"]:
            node_entry["alias"] = node_info["alias"]
        related_ttp_ids = _related_ttp_ids(node_id, node_info, node_map, edge_map)
        if related_ttp_ids:
            node_entry["edges"] = related_ttp_ids
        nodes_info.append(node_entry)

    edges_info, debug_edges_info = _edges_info(data, node_map)
    graph_info = {
        "nodes": nodes_info,
        "edges": edges_info,
        "debug_edges": debug_edges_info,
    }
    output_file = Path(output_dir) / Path(json_file).name
    _dump_json(graph_info, output_file)
    return output_file


EXTRACTORS = {
    "highlevel": extract_highlevel,
    "tagger": extract_tagger,
    "ioc": extract_ioc,
    "augmentor": extract_augmentor,
    "alignmentor": extract_alignmentor,
}


def main(argv: list[str] | None = None) -> int:
    """``python -m beacon.extract <component> <json_file> <output_dir>``"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m beacon.extract",
        description="Write the lightweight extract view of one component output file.",
    )
    parser.add_argument("component", choices=sorted(EXTRACTORS))
    parser.add_argument("json_file")
    parser.add_argument("output_dir")
    args = parser.parse_args(argv)
    out = EXTRACTORS[args.component](Path(args.json_file), Path(args.output_dir))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
