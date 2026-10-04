"""Convert the recorded merged pre-second-pass IoCs to node predictions."""
from common import node_graph, read_json, run


def build(cfg, report_id):
    payload = read_json(cfg["ioc_pre_verification_dir"] / f"{report_id}.json")
    metadata = payload["metadata"]
    if metadata["stage"] != "pre_second_pass_verification":
        raise ValueError("Expected pre-second-pass candidates")
    merged, occurrences = payload["merged_candidates"], payload["candidate_occurrences"]
    if (metadata["merged_candidate_count"] != len(merged)
            or metadata["candidate_occurrence_count"] != len(occurrences)):
        raise ValueError("Candidate count metadata mismatch")
    values = []
    for ioc in merged:
        text = str(ioc.get("value", "")).strip() or str(ioc.get("normalized_value", "")).strip()
        subtype = ioc.get("ioc_type", "")
        if text and subtype:
            values.append((subtype, text))
    graph = node_graph("ioc", values)
    return graph, {"candidate_occurrences": len(occurrences), "merged_candidates": len(merged),
                   "evaluation_nodes": len(graph["nodes"]["ioc"]),
                   "reference": "full_run_root, not the recording run's post-filter output"}


if __name__ == "__main__":
    run("without_ioc_filtering", build)
