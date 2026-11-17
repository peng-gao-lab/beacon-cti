# BEACON datasets

Ground-truth (GT) annotations, metadata and the resources used in the BEACON experiments. Report texts are not included. Obtain them from the original publishers and place the TXT files in the corresponding `reports/` directories. Licensing is described in `LICENSE.md`: our annotations and metadata are CC BY 4.0, the report texts and quoted evidence stay with their publishers, and ATT&CK content stays with MITRE.

## Contents

| Path | Contents |
| --- | --- |
| `BEACON-Single/reports/` | Directory for the 150 report texts obtained separately |
| `BEACON-Single/gt/` | 150 GT graphs (4,966 nodes, 3,429 edges) |
| `BEACON-Group/reports/<group>/` | Directories for the 100 report texts obtained separately, in 33 groups |
| `BEACON-Group/single_gt/<group>/` | 100 report-level GT graphs (2,908 nodes, 2,278 edges) |
| `BEACON-Group/alignment_gt/<group>/aligned_gt.json` | 33 consolidated GT graphs (1,950 nodes, 1,537 edges) |
| `resources/mitre.json` | ATT&CK v17.1 corpus, 1,076 entries |
| `metadata/` | publisher summaries, publisher links and dataset statistics |

A report and its GT share the same filename stem (and, in BEACON-Group, the same group directory). RQ1 takes the BEACON-Single report texts as input. RQ2 takes the BEACON-Group report-level GT graphs as input and is scored against the consolidated GT.

## GT format

A GT graph has `nodes` and `edges`. `nodes` holds three lists: `high_level` (contextual entities), `ttp` and `ioc`. Every node has `node_id`, `node_class`, `node_type`, `node_text` and the IDs of its `edges`. An edge has `edge_id`, `node_u` and `node_v`; the node classes of its endpoints tell whether it is an entity-to-technique or an IoC-to-technique edge.

Evidence is quoted from the report text. Contextual and IoC nodes carry `sentence_indices` and `span_sentences`; technique nodes carry `best_span_ranges` and `best_span_sentences_list`. Indices are zero-based and inclusive, under the tool's built-in sentence splitter applied to the report file.

Consolidated GT keeps the aligned nodes and edges together with their aliases, evidence and a `source` mapping back to the individual reports.

A consolidated node that merges k report-level nodes counts as k−1 links.

## Metadata

- `metadata/single_report_source.json` and `metadata/group_report_source.json`: reports per publisher.
- `metadata/source_url.json`: publisher websites。
- `metadata/experiment_dataset_statistics.json`: dataset counts and how they were counted.

## Notes

BEACON-Single and BEACON-Group share 15 source articles, with different cleaned text versions.
