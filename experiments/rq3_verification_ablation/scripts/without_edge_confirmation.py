"""Rebuild unverified anchored edges from the three extraction outputs.

No accepted/debug edge lists, LLM client instances or gold labels are used in
candidate construction. Native node loading, candidate generation and center
selection are reused without changing their rules.
"""
from common import read_json, run


def build(cfg, report_id):
    from CTIAugmentor.data_loader import DataLoader
    from CTIAugmentor.candidate_generator import CandidateGenerator
    from CTIAugmentor.pipeline import CTIAugmentorPipeline

    root = cfg["full_run_root"]
    loader = DataLoader()
    highlevel = loader.build_highlevel_nodes(
        read_json(root / "ctihighlevelextractor" / f"{report_id}.json")["entities"])
    ioc = loader.build_ioc_nodes(
        read_json(root / "ctiiocextractor" / f"{report_id}.json")["iocs"])
    ttp = loader.build_ttp_nodes(
        read_json(root / "ctitagger" / f"{report_id}.json"))
    candidates = CandidateGenerator(delta=cfg["delta"]).generate_candidates(highlevel, ioc, ttp)
    grouped = {}
    for c in candidates:
        key = (c.edge_type, c.node_u_id, c.node_v_id)
        grouped.setdefault(key, set()).update(c.evidence_indices_set)
    edges = []

    def add_edge(edge_type, u, v, evidence, stage):
        edges.append({
            "edge_id": f"ABL_EDGE_{len(edges) + 1}",
            "node_u": u, "node_v": v, "edge_type": edge_type,
            "evidence": {"evidence_sentence_indices_global": sorted(evidence)},
            "verification": {"status": "bypassed", "stage": stage},
        })

    for (kind, u, v), evidence in grouped.items():
        add_edge(kind, u, v, evidence, "proximity")
    proximity_count = len(edges)
    center = CTIAugmentorPipeline._select_potential_center_entity(highlevel)
    if center is not None:
        linked = {e["node_v"] for e in edges if e["node_u"] == center.node_id}
        for node in sorted(ttp.values(), key=lambda n: n.ttp_id):
            if node.node_id not in linked:
                evidence = CTIAugmentorPipeline._extract_ttp_evidence_indices(node)
                add_edge("high_level-ttp", center.node_id, node.node_id,
                         evidence, "center_entity_supplement")
    all_nodes = {**highlevel, **ioc, **ttp}
    for edge in edges:
        for endpoint in (edge["node_u"], edge["node_v"]):
            all_nodes[endpoint].edges.append(edge["edge_id"])
    graph = {"nodes": {"high_level": [n.to_dict() for n in highlevel.values()],
                       "ttp": [n.to_dict() for n in ttp.values()],
                       "ioc": [n.to_dict() for n in ioc.values()]}, "edges": edges}
    return graph, {"proximity_candidate_instances": len(candidates),
                   "proximity_candidate_edges": proximity_count,
                   "center_supplement_candidate_edges": len(edges) - proximity_count,
                   "selected_center": center.node_text if center else None}


if __name__ == "__main__":
    run("without_edge_confirmation", build)
