"""Accept stored TTP proposer candidates without using verification decisions."""
from common import node_graph, read_json, run


def build(cfg, report_id):
    behaviors = read_json(cfg["full_run_root"] / "ctitagger" / f"{report_id}.json")
    values = []
    for behavior in behaviors:
        # Required field, so an extract/ view cannot silently look like zero candidates.
        for candidate in behavior["ttp_candidates"]:
            values.append(("TTP", candidate["ttp_id"]))
    graph = node_graph("ttp", values)
    return graph, {"behaviors": len(behaviors), "candidate_occurrences": len(values),
                   "evaluation_nodes": len(graph["nodes"]["ttp"])}


if __name__ == "__main__":
    run("without_ttp_verification", build)
