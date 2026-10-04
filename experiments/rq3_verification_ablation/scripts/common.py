"""RQ3 anchoring ablations using the normal RQ1 run config and unchanged metric."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from beacon.config import load_config

# The evaluator is part of this repository, not an external workspace script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluators"))
from evaluator_RQ1 import (  # noqa: E402
    aggregate_metrics, evaluate_one_file, extract_nodes_and_maps,
)

SCOPES = {
    "without_entity_review": [("node_metrics", "high_level")],
    "without_ttp_verification": [("node_metrics", "ttp")],
    "without_ioc_filtering": [("node_metrics", "ioc")],
    "without_edge_confirmation": [
        ("edge_type_metrics", "high_level-ttp"),
        ("edge_type_metrics", "ioc-ttp"),
    ],
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    # New destination only; never silently overwrite an experiment artifact.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def node_graph(layer, values):
    """Preserve input order; use the original evaluator's subtype/text dedup."""
    graph = {"nodes": {"high_level": [], "ttp": [], "ioc": []}, "edges": []}
    graph["nodes"][layer] = [
        {"node_id": f"{layer}_{i}", "node_type": subtype, "node_text": text}
        for i, (subtype, text) in enumerate(values, 1)
    ]
    entries = extract_nodes_and_maps(graph)[0][layer]
    graph["nodes"][layer] = [
        {"node_id": f"{layer}_{i}", "node_class": layer,
         "node_type": n.node_subtype, "node_text": n.text, "edges": []}
        for i, n in enumerate(entries, 1)
    ]
    return graph


def settings_from_args(argv=None):
    parser = argparse.ArgumentParser(description="RQ3 verification ablation from a completed RQ1 run")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "rq1_beacon/config.yaml",
                        help="The same BEACON YAML used for the normal full RQ1 run")
    parser.add_argument("--gt-dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "dataset/BEACON-Single/gt")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Optional new destination; defaults to rq3_verification_ablation beside the full output")
    args = parser.parse_args(argv)
    full = load_config(args.config)
    if full.input.kind != "reports" or full.input.layout != "flat":
        raise ValueError("Use the normal flat report-input RQ1 configuration")
    root = full.output.root
    cfg = {
        "run_config": full.config_path,
        "reports_dir": full.input.path,
        "full_run_root": root,
        "ioc_pre_verification_dir": root / "ctiiocextractor/pre_verification",
        "highlevel_log_dir": root / "ctihighlevelextractor/logs",
        "gt_dir": args.gt_dir.resolve(),
        "output_root": (args.output_root or root.parent / "rq3_verification_ablation").resolve(),
        "node_threshold": 0.85,
        "edge_threshold": 0.85,
        "delta": full.augmentor.delta,
    }
    reports = sorted(p for p in full.input.path.glob(full.input.pattern) if p.is_file())
    if full.input.limit is not None:
        reports = reports[:full.input.limit]
    if not reports:
        raise ValueError("No report TXT files found")
    return cfg, reports


def run(mode, build):
    cfg, reports = settings_from_args()
    dest = cfg["output_root"] / mode
    if dest.exists():
        raise FileExistsError(f"Use --output-root for a new destination; already exists: {dest}")
    # Do not let a configured destination overlap the source trees.
    for key in ("reports_dir", "full_run_root", "ioc_pre_verification_dir", "gt_dir", "highlevel_log_dir"):
        source = cfg[key]
        if dest.is_relative_to(source) or source.is_relative_to(dest):
            raise ValueError(f"Output overlaps {key}: {dest}")
    rows, full_rows, audit_rows = [], [], []
    for report in reports:
        name = report.stem + ".json"
        try:
            graph, audit = build(cfg, report.stem)
            pred = dest / "predictions" / name
            write_json(pred, graph)
            gt = cfg["gt_dir"] / name
            metric = evaluate_one_file(pred, gt, cfg["node_threshold"], cfg["edge_threshold"])
            full = evaluate_one_file(cfg["full_run_root"] / "ctiaugmentor" / name,
                                     gt, cfg["node_threshold"], cfg["edge_threshold"])
        except Exception as exc:
            raise RuntimeError(f"{mode}, report {report.stem}: {exc}") from exc
        # Only report the component's own scope, not spurious overall metrics.
        row, full_row = {}, {}
        for family, scope in SCOPES[mode]:
            row.setdefault(family, {})[scope] = metric[family][scope]
            full_row.setdefault(family, {})[scope] = full[family][scope]
        rows.append(row)
        full_rows.append(full_row)
        audit_rows.append({"report_id": report.stem, "audit": audit,
                           "ablated": row, "full": full_row})
    summary = {}
    for family, scope in SCOPES[mode]:
        threshold = cfg["edge_threshold"] if family == "edge_type_metrics" else cfg["node_threshold"]
        ablated = aggregate_metrics(rows, (family, scope), threshold)
        full = aggregate_metrics(full_rows, (family, scope), threshold)
        summary[scope] = {
            "ablated": ablated, "full": full,
            "delta_percentage_points": {
                k: 100 * (ablated[k] - full[k])
                for k in ("precision", "recall", "f1_score")
            },
        }
        print(f"{scope}: P={ablated['precision']:.4f} R={ablated['recall']:.4f} "
              f"F1={ablated['f1_score']:.4f}")
    write_json(dest / "evaluation.json", {
        "experiment": mode, "report_count": len(reports),
        "scope": "component output only; no downstream rerun",
        "aggregation": "micro",
        "reference": str(cfg["full_run_root"]),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
        "summary": summary, "reports": audit_rows,
    })
    print(f"Saved predictions and evaluation to {dest}")
