# Paper experiments

Run every command from the repository root. Each `config.yaml` is an input to `python -m beacon --config ...` and specifies the experiment's configurable settings. Predictions are not included; generate them first, then evaluate.

| Paper experiment | Configuration |
| --- | --- |
| RQ1 full BEACON | `rq1_beacon/config.yaml` |
| RQ2 full BEACON | `rq2_beacon/config.yaml` |
| RQ3 w/o entity review, w/o TTP verification, w/o IoC filtering, w/o edge confirmation | offline scripts, see [Verification ablations](#rq3-verification-ablations) |
| RQ3 w/o character measures | `rq3_no_fuzzy_char/config.yaml` |
| RQ3 w/o character rules | `rq3_no_char/config.yaml` |
| RQ3 w/o embedding | `rq3_no_embed/config.yaml` |
| RQ3 w/o neighborhood | `rq3_no_neighbor/config.yaml` |

All configurations process the full dataset (`input.limit: null`) and write to their own directory under `output/experiments/`. `run.workers` controls the maximum number of reports or alignment groups processed concurrently.

Sampling temperatures and output token limits are defined in the code, which cannot be changed via YAML.

## RQ1

```bash
python -m beacon --config experiments/rq1_beacon/config.yaml
python experiments/evaluators/evaluator_RQ1.py \
  output/experiments/rq1_beacon/ctiaugmentor \
  dataset/BEACON-Single/gt --batch \
  --node-threshold 0.85 --edge-threshold 0.85 --summary-average micro \
  --output output/experiments/rq1_beacon/evaluation_RQ1.json
```

The evaluator matches predicted and gold nodes one-to-one within the same node type. Contextual entities and IoCs match by string similarity (threshold 0.85), techniques by exact ATT&CK ID, and an edge matches when both its endpoints match. The paper reports the micro averages; the output file also contains macro averages.

## RQ2 and the consolidation ablations

```bash
python -m beacon --config experiments/rq2_beacon/config.yaml
python experiments/evaluators/evaluate_RQ2.py \
  output/experiments/rq2_beacon/groups \
  dataset/BEACON-Group/single_gt \
  dataset/BEACON-Group/alignment_gt \
  output/experiments/rq2_beacon/evaluation_RQ2.json
```

For a consolidation ablation, replace rq2_beacon in both commands with its directory name, such as rq3_no_neighbor.

RQ2 aligns 100 report-level GT graphs within 33 groups and evaluates merge-link precision, recall and F1 against the consolidated GT. Merging k nodes counts as k−1 links. Within a predicted cluster, m nodes from the same gold cluster contribute max(m−1, 0) correct links; these counts are summed over gold clusters.

| Condition | Candidate channels |
| --- | --- |
| Full | char, embed, neighbor |
| no_fuzzy_char | embed, neighbor |
| no_char | embed, neighbor |
| no_embed | char, neighbor |
| no_neighbor | char, embed |

Only `no_char` also disables deterministic character rules for contextual entities. Deterministic alignment of techniques and IoCs stays on in all conditions.

The two scripts in `evaluators/` are the evaluators used for the paper.

## RQ3 verification ablations

These four ablations remove one verification step each and are computed offline from the outputs of the normal RQ1 run. They need no LLM calls and no extra YAML.

After completing the RQ1 run above, run any of:

```bash
python experiments/rq3_verification_ablation/scripts/without_entity_review.py
python experiments/rq3_verification_ablation/scripts/without_ttp_verification.py
python experiments/rq3_verification_ablation/scripts/without_ioc_filtering.py
python experiments/rq3_verification_ablation/scripts/without_edge_confirmation.py
```

The scripts read `experiments/rq1_beacon/config.yaml` by default; pass `--config` if the run used another file. `--gt-dir` and `--output-root` override the ground-truth directory and the destination.

The entity-review ablation reads rejected entities from the contextual extractor’s logs, so components.highlevel.log must be true during the RQ1 run.

Outputs go to `output/experiments/rq3_verification_ablation/<ablation>/` with `predictions/<report_id>.json` and `evaluation.json`. Each ablation is scored with the RQ1 evaluator on the scope it changes (contextual nodes, technique nodes, IoC nodes, or the two edge types) and reported next to the full-run score.
