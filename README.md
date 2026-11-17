# BEACON: Behavior-Anchored Cross-Source Knowledge Graph Construction for Cyber Threat Intelligence

<p align="center">
  <a href="#repository-layout">📁 Repository layout</a> | <a href="#installation">📦 Installation</a> | <a href="#datasets-and-resources">📚 Datasets and resources</a> | <a href="#running-the-tool">🚀 Running the tool</a> | <a href="#configuration">⚙️ Configuration</a> | <a href="#output">📂 Output</a> | <a href="#api-cost-recording">💰 API cost recording</a> | <a href="#paper-experiments">🧪 Paper experiments</a> | <a href="#paper-and-citation">📝 Paper and Citation</a> | <a href="#license">⚖️ License</a>
</p>

BEACON constructs a unified knowledge graph from heterogeneous cyber threat intelligence (CTI) reports by anchoring contextual entities and IoCs to ATT&CK techniques and aligning entities across reports.

| Stage | What it does |
| --- | --- |
| Contextual entity extraction | Extract entities |
| Behavior grounding | Extract attack behaviors and map them to MITRE ATT&CK techniques |
| IoC extraction | Extract IoCs |
| Anchored attachment | Construct contextual entity-TTP edges and IoC-TTP edges |
| Consolidation | Align entities across graphs with deterministic rules, then character, embedding and neighborhood candidates verified by an LLM |

<a id="repository-layout"></a>

## 📁 Repository layout

```text
README.md
environment.yml  pyproject.toml  LICENSE
configs/        YAML templates for the four modes of use
dataset/        datasets and resources
experiments/    experiment configurations and evaluators
src/beacon/     YAML parsing and the pipeline runner
src/CTI*/       the five components
src/beacon_cost/ optional API cost recording
```

<a id="installation"></a>

## 📦 Installation

```bash
conda env create -f environment.yml
conda activate beacon
python -m nltk.downloader stopwords # English stopwords for TTP evidence matching
```

This creates the `beacon` environment and installs the package in editable mode. 
If you do not use Conda, install the package with pip install -e . using Python 3.10 or newer. Then run the tool with either beacon or python -m beacon.

Components that call an LLM need an API key:

```bash
export OPENAI_API_KEY="..."
```

`components.alignmentor.api_key_env` can change the variable name that Alignmentor reads.

<a id="datasets-and-resources"></a>

## 📚 Datasets and resources

[dataset/README.md](dataset/README.md) describes the data and file formats.

- BEACON-Single: graph ground truth and metadata for 150 reports.
- BEACON-Group: report-level graphs, consolidated ground truth and metadata for 100 reports in 33 groups.
- `dataset/resources/mitre.json`: the ATT&CK v17.1 corpus (Enterprise, Mobile and ICS) in the JSON format the tool reads. Raw STIX bundles are not accepted.

Report texts are not included. Obtain them from the original publishers and place the TXT files in the corresponding `reports/` directories, using the same filename stems as the GT files.

<a id="running-the-tool"></a>

## 🚀 Running the tool

```bash
python -m beacon --config path/to/config.yaml
```

Paths inside a YAML are relative to that YAML file. Start from one of the templates in `configs/` and set the input and output paths:

| Mode | Template | Input |
| --- | --- | --- |
| Build one graph per report | `configs/extraction.yaml` | TXT files in one directory |
| Build graphs and consolidate them all | `configs/extraction_align_flat.yaml` | TXT files in one directory |
| Build graphs and consolidate within groups | `configs/extraction_align_groups.yaml` | one subdirectory of TXT files per group |
| Consolidate existing graphs | `configs/align_existing_graphs.yaml` | full graph JSONs, flat or grouped |

Flat input is `reports/<report_id>.txt`. Grouped input is `reports/<group_id>/<report_id>.txt`.

<a id="configuration"></a>

## ⚙️ Configuration

| Setting | Meaning |
| --- | --- |
| `input.kind` | `reports` or `graphs` |
| `input.layout` | `flat` or `groups` |
| `input.path`, `input.pattern` | input directory and file glob |
| `input.limit` | process only the first N reports or groups (sorted); the default `null` means all |
| `output.root` | where this run writes |
| `output.existing` | `skip` reuses outputs that already exist; `overwrite` replaces them |
| `run.workers` | reports or groups processed in parallel |
| `run.sentence_splitter` | `regex` or `nltk` |
| `resources.mitre_json` | the ATT&CK corpus |
| `models.default` | chat model for every component unless `components.<name>.model` overrides it |
| `models.embedding` | embedding model for consolidation |
| `components.<name>.enabled` | turn a component on or off |
| `components.<name>.log` | write the component's log |
| `components.<name>.extract` | also write a simplified, human-readable view of the output |

Component names are `highlevel`, `tagger`, `ioc`, `augmentor` and `alignmentor`.

Sentence splitting uses the components' built-in rule-based splitter by default, which is what the paper's runs used. `run.sentence_splitter: nltk` switches to NLTK Punkt. Use one splitter for every component.

For consolidation, `highlevel_stages` selects the candidate channels from `char`, `embed` and `neighbor`. `enable_llm: false` skips LLM verification. `disable_highlevel_deterministic_char: true` turns off the deterministic character rules for contextual entities.

With `output.existing: skip`, an existing output is reused even if its inputs or settings changed. After changing a model, a resource or an upstream output, write to a new output directory or delete the affected outputs first.

<a id="output"></a>

## 📂 Output

For flat input:

```text
<output.root>/
  ctihighlevelextractor/<report_id>.json
  ctitagger/<report_id>.json
  ctiiocextractor/<report_id>.json
  ctiaugmentor/<report_id>.json
  ctialignmentor/<timestamp>.json
  beacon_runs/<run_id>/            config and status of each run
```

Grouped runs put the component directories under `<output.root>/groups/<group_id>/`. Logs and simplified views go into `logs/` and `extract/` inside each component directory. The IoC component also keeps its pre-filtering candidates in `ctiiocextractor/pre_verification/`, which the IoC filtering ablation uses.

<a id="api-cost-recording"></a>

## 💰 API cost recording

Set `cost.enabled: true` in YAML to record API usage. The runner appends call records to `<output.root>/cost/api_calls.jsonl` and updates `<output.root>/cost/summary.json` when the run finishes. Runs sharing an output directory accumulate in the same ledger.

Records include the component and stage, model, token counts, latency, status and estimated USD cost. Estimates use the dated price list in `src/beacon_cost/pricing/`; `cost.pricebook` can specify another price-list file. Failed calls without usage information are recorded with unknown cost.

<a id="paper-experiments"></a>

## 🧪 Paper experiments

[experiments/README.md](experiments/README.md) provides the run and evaluation commands and links to the YAML configurations used for the paper. RQ1 runs on the BEACON-Single report texts. RQ2 starts from the BEACON-Group report-level ground-truth graphs and is scored against the consolidated ground truth.

<a id="paper-and-citation"></a>

## 📝 Paper and Citation

This repository provides the implementation and datasets for [BEACON: Behavior-Anchored Cross-Source Knowledge Graph Construction for Cyber Threat Intelligence](https://arxiv.org/abs/2608.28394), accepted to WSDM 2027.

If you use BEACON or its datasets in your research, please cite our paper.

```bibtex
@inproceedings{li2027beacon,
  title = {{BEACON}: Behavior-Anchored Cross-Source Knowledge Graph Construction for Cyber Threat Intelligence},
  author = {Changze Li and Yutong Cheng and Tsania Camila Finnisa and Qian Cui and Wei Ding and Peng Gao},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search and Data Mining},
  series = {WSDM '27},
  year = {2027},
  publisher = {Association for Computing Machinery},
  location = {Hong Kong, Hong Kong},
  doi = {10.1145/3862322.3863856},
  isbn = {979-8-4007-3117-4},
  url = {https://arxiv.org/abs/2608.28394}
}
```

<a id="license"></a>

## ⚖️ License

The software, prompts and documentation are under the MIT License ([LICENSE](LICENSE)). Our annotations, dataset organization and metadata are under CC BY 4.0. Report texts and quoted evidence remain subject to their publishers' rights, and ATT&CK content is subject to MITRE's terms. See [dataset/LICENSE.md](dataset/LICENSE.md) for the scope of the dataset license before redistributing its contents.
