"""Restore pre-review contextual nodes from retained entities and rejection logs."""
import re

from common import node_graph, read_json, run

REJECTED = re.compile(r"^\s*Filtered entity \[(\d+)/(\d+)\]:\s*([^|]+)\|\s?(.*)$", re.MULTILINE)
SUMMARY = re.compile(r"Precision check done: kept (\d+) entities \(filtered (\d+)\)")


def build(cfg, report_id):
    payload = read_json(cfg["full_run_root"] / "ctihighlevelextractor" / f"{report_id}.json")
    entities = payload["entities"]
    log = (cfg["highlevel_log_dir"] / f"{report_id}.log").read_text(encoding="utf-8")
    rejected = REJECTED.findall(log)
    summaries = SUMMARY.findall(log)
    if len(summaries) != 1:
        raise ValueError("Expected one completed precision-review summary in the log")
    retained_count, rejected_count = map(int, summaries[0])
    if retained_count != len(entities) or rejected_count != len(rejected):
        raise ValueError("Review log counts do not match the retained output/rejected records")
    total = retained_count + rejected_count
    if any(int(n) != total for _, n, _, _ in rejected):
        raise ValueError("Inconsistent pre-review candidate count")
    if len({i for i, _, _, _ in rejected}) != len(rejected):
        raise ValueError("Duplicate rejection records in log")
    values = [(e["entity_type"], e["entity_text"]) for e in entities]
    values.extend((t.strip(), text.strip()) for _, _, t, text in rejected)
    graph = node_graph("high_level", values)
    return graph, {"retained_entities": retained_count, "restored_rejections": rejected_count,
                   "evaluation_nodes": len(graph["nodes"]["high_level"]),
                   "limitation": "Rejected candidates have type/text only; no evidence or aliases are fabricated."}


if __name__ == "__main__":
    run("without_entity_review", build)
