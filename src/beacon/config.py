"""YAML configuration for BEACON runs.

All relative paths in the YAML are resolved against the directory that
contains the YAML file. Defaults for component parameters equal the explicit
settings of the reference GPT-4o runs (see ``configs/*.yaml`` and the
implementation notes); the component CLIs have partly different defaults, so
every parameter is always passed explicitly to the component subprocess.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

INPUT_KINDS = ("reports", "graphs")
LAYOUTS = ("flat", "groups")
EXISTING_POLICIES = ("skip", "overwrite")
CONSOLE_MODES = ("quiet", "verbose")
COMPONENT_ERROR_POLICIES = ("fail", "warn")
SENTENCE_SPLITTERS = ("regex", "nltk")
HIGHLEVEL_STAGES = ("char", "embed", "neighbor")
COMPONENT_ORDER = ("highlevel", "tagger", "ioc", "augmentor", "alignmentor")

# Output directory names per component (as used by the reference pipelines).
COMPONENT_DIRS = {
    "highlevel": "ctihighlevelextractor",
    "tagger": "ctitagger",
    "ioc": "ctiiocextractor",
    "augmentor": "ctiaugmentor",
    "alignmentor": "ctialignmentor",
}

DEFAULT_MODEL = "gpt-4o"
DEFAULT_EMBED_MODEL = "text-embedding-3-large"
DEFAULT_PRICEBOOK_NAME = "openai_standard_2026-07-21.json"


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration."""


@dataclass
class InputConfig:
    kind: str = "reports"
    layout: str = "flat"
    path: Path = Path(".")
    pattern: str = "*.txt"
    limit: int | None = None


@dataclass
class OutputConfig:
    root: Path = Path("./output")
    existing: str = "skip"


@dataclass
class RunConfig:
    workers: int = 1
    console: str = "quiet"
    # The extraction components catch failed LLM calls internally and still
    # exit 0 with an incomplete (often empty) result. "fail" marks such a
    # stage as failed and stops the report's downstream stages; "warn" keeps
    # going and records the count in status.json.
    component_errors: str = "fail"
    # Sentence splitter used by the four extraction components. "regex" is the
    # components' built-in splitter and what the reference runs used; "nltk"
    # requires the punkt_tab data and is rejected at preflight if unavailable.
    sentence_splitter: str = "regex"


@dataclass
class ResourcesConfig:
    mitre_json: Path | None = None


@dataclass
class ModelsConfig:
    default: str = DEFAULT_MODEL
    embedding: str = DEFAULT_EMBED_MODEL


@dataclass
class ComponentBase:
    enabled: bool = True
    log: bool = True
    extract: bool = True
    model: str | None = None  # None -> models.default


@dataclass
class HighLevelConfig(ComponentBase):
    chunk_size: int = 100
    overlap_size: int = 2
    token_overlap_threshold: float = 0.6


@dataclass
class TaggerConfig(ComponentBase):
    threshold: float = 0.71
    lmax: int = 5
    high_freq_threshold: float = 0.01
    enable_ngram: bool = True


@dataclass
class IocConfig(ComponentBase):
    pass


@dataclass
class AugmentorConfig(ComponentBase):
    threshold: float = 0.7
    delta: int = 0
    check_split: bool = False


@dataclass
class AlignmentorConfig(ComponentBase):
    enabled: bool = False
    node_threshold: float = 0.7
    cluster_threshold: float = 0.7
    enable_llm: bool = True
    highlevel_stages: list[str] = field(default_factory=lambda: list(HIGHLEVEL_STAGES))
    disable_highlevel_deterministic_char: bool = False
    embed_backend: str = "openai"
    embed_model: str | None = None  # None -> models.embedding
    embed_sim_threshold: float = 0.7
    max_neighbor_rounds: int = 3
    neighbor_shared_ttp_threshold: int = 2
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class CostConfig:
    enabled: bool = True
    pricebook: Path | None = None  # None -> bundled DEFAULT_PRICEBOOK_NAME


@dataclass
class BeaconConfig:
    config_path: Path
    input: InputConfig
    output: OutputConfig
    run: RunConfig
    resources: ResourcesConfig
    models: ModelsConfig
    highlevel: HighLevelConfig
    tagger: TaggerConfig
    ioc: IocConfig
    augmentor: AugmentorConfig
    alignmentor: AlignmentorConfig
    cost: CostConfig

    # -- convenience -------------------------------------------------------
    def component(self, name: str) -> ComponentBase:
        return getattr(self, name)

    def model_for(self, name: str) -> str:
        comp = self.component(name)
        return comp.model or self.models.default

    def embed_model(self) -> str:
        return self.alignmentor.embed_model or self.models.embedding

    def enabled_components(self) -> list[str]:
        return [c for c in COMPONENT_ORDER if self.component(c).enabled]

    def pricebook_path(self) -> Path:
        if self.cost.pricebook is not None:
            return self.cost.pricebook
        import beacon_cost  # bundled sibling package

        return Path(beacon_cost.__file__).parent / "pricing" / DEFAULT_PRICEBOOK_NAME

    def to_plain(self) -> dict[str, Any]:
        """JSON/YAML friendly snapshot of the effective configuration."""

        def convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        data = {k: convert(asdict(v)) if hasattr(v, "__dataclass_fields__") else convert(v)
                for k, v in self.__dict__.items()}
        data["effective"] = {
            "models": {c: self.model_for(c) for c in COMPONENT_ORDER},
            "embedding_model": self.embed_model(),
            "pricebook": str(self.pricebook_path()) if self.cost.enabled else None,
        }
        return data


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _section(raw: dict, name: str) -> dict:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{name}' must be a mapping")
    return dict(value)


def _build(cls, data: dict, section: str):
    allowed = set(cls.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in '{section}': {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"Invalid '{section}' section: {exc}") from exc


def _resolve(base: Path, value: Any, section: str, key: str, *, required: bool) -> Path | None:
    if value is None or value == "":
        if required:
            raise ConfigError(f"'{section}.{key}' is required")
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _check_choice(value: Any, choices: tuple, section: str, key: str) -> None:
    if value not in choices:
        raise ConfigError(f"'{section}.{key}' must be one of {list(choices)}, got {value!r}")


def load_config(config_path: str | Path) -> BeaconConfig:
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top level of the YAML must be a mapping")

    known_top = {"input", "output", "run", "resources", "models", "components", "cost"}
    unknown_top = set(raw) - known_top
    if unknown_top:
        raise ConfigError(f"Unknown top-level key(s): {sorted(unknown_top)}")

    base = config_path.parent

    inp = _section(raw, "input")
    inp["path"] = _resolve(base, inp.get("path"), "input", "path", required=True)
    input_cfg = _build(InputConfig, inp, "input")
    _check_choice(input_cfg.kind, INPUT_KINDS, "input", "kind")
    _check_choice(input_cfg.layout, LAYOUTS, "input", "layout")
    if input_cfg.limit is not None and (not isinstance(input_cfg.limit, int) or input_cfg.limit < 0):
        raise ConfigError("'input.limit' must be a non-negative integer or null")
    if input_cfg.kind == "graphs" and "pattern" not in inp:
        input_cfg.pattern = "*.json"

    out = _section(raw, "output")
    out["root"] = _resolve(base, out.get("root"), "output", "root", required=True)
    output_cfg = _build(OutputConfig, out, "output")
    _check_choice(output_cfg.existing, EXISTING_POLICIES, "output", "existing")

    run_cfg = _build(RunConfig, _section(raw, "run"), "run")
    _check_choice(run_cfg.console, CONSOLE_MODES, "run", "console")
    _check_choice(run_cfg.component_errors, COMPONENT_ERROR_POLICIES, "run", "component_errors")
    _check_choice(run_cfg.sentence_splitter, SENTENCE_SPLITTERS, "run", "sentence_splitter")
    if not isinstance(run_cfg.workers, int) or run_cfg.workers < 1:
        raise ConfigError("'run.workers' must be an integer >= 1")

    res = _section(raw, "resources")
    res["mitre_json"] = _resolve(base, res.get("mitre_json"), "resources", "mitre_json", required=False)
    resources_cfg = _build(ResourcesConfig, res, "resources")

    models_cfg = _build(ModelsConfig, _section(raw, "models"), "models")

    comps = _section(raw, "components")
    unknown_comps = set(comps) - set(COMPONENT_ORDER)
    if unknown_comps:
        raise ConfigError(f"Unknown component(s): {sorted(unknown_comps)}; "
                          f"expected {list(COMPONENT_ORDER)}")
    highlevel_cfg = _build(HighLevelConfig, _section(comps, "highlevel"), "components.highlevel")
    tagger_cfg = _build(TaggerConfig, _section(comps, "tagger"), "components.tagger")
    ioc_cfg = _build(IocConfig, _section(comps, "ioc"), "components.ioc")
    augmentor_cfg = _build(AugmentorConfig, _section(comps, "augmentor"), "components.augmentor")
    alignmentor_cfg = _build(AlignmentorConfig, _section(comps, "alignmentor"), "components.alignmentor")

    stages = list(alignmentor_cfg.highlevel_stages or [])
    for st in stages:
        _check_choice(st, HIGHLEVEL_STAGES, "components.alignmentor", "highlevel_stages")
    alignmentor_cfg.highlevel_stages = stages

    cost = _section(raw, "cost")
    cost["pricebook"] = _resolve(base, cost.get("pricebook"), "cost", "pricebook", required=False)
    cost_cfg = _build(CostConfig, cost, "cost")

    cfg = BeaconConfig(
        config_path=config_path,
        input=input_cfg,
        output=output_cfg,
        run=run_cfg,
        resources=resources_cfg,
        models=models_cfg,
        highlevel=highlevel_cfg,
        tagger=tagger_cfg,
        ioc=ioc_cfg,
        augmentor=augmentor_cfg,
        alignmentor=alignmentor_cfg,
        cost=cost_cfg,
    )
    validate(cfg)
    return cfg


def validate(cfg: BeaconConfig) -> None:
    """Cross-field checks: stage dependencies and resource requirements."""
    if not cfg.input.path.exists():
        raise ConfigError(f"'input.path' does not exist: {cfg.input.path}")
    if not cfg.input.path.is_dir():
        raise ConfigError(f"'input.path' must be a directory: {cfg.input.path}")

    if cfg.input.kind == "graphs":
        if not cfg.alignmentor.enabled:
            raise ConfigError("input.kind=graphs requires components.alignmentor.enabled=true")
        for c in ("highlevel", "tagger", "ioc", "augmentor"):
            if cfg.component(c).enabled:
                raise ConfigError(
                    f"input.kind=graphs runs only the alignmentor; set components.{c}.enabled=false"
                )
    else:
        if not cfg.enabled_components():
            raise ConfigError("No component is enabled")
        if cfg.augmentor.enabled:
            missing = [c for c in ("highlevel", "tagger", "ioc") if not cfg.component(c).enabled]
            if missing:
                raise ConfigError(
                    "components.augmentor requires highlevel, tagger and ioc to be enabled; "
                    f"disabled: {missing}"
                )
        if cfg.alignmentor.enabled and not cfg.augmentor.enabled:
            raise ConfigError("components.alignmentor requires components.augmentor.enabled=true")

    needs_mitre = cfg.input.kind == "reports" and (cfg.tagger.enabled or cfg.augmentor.enabled)
    if needs_mitre:
        if cfg.resources.mitre_json is None:
            raise ConfigError("'resources.mitre_json' is required when tagger or augmentor is enabled")
        if not cfg.resources.mitre_json.is_file():
            raise ConfigError(f"'resources.mitre_json' not found: {cfg.resources.mitre_json}")

    if cfg.cost.enabled and cfg.cost.pricebook is not None and not cfg.cost.pricebook.is_file():
        raise ConfigError(f"'cost.pricebook' not found: {cfg.cost.pricebook}")

    if cfg.alignmentor.enabled and cfg.alignmentor.enable_llm:
        if cfg.alignmentor.embed_backend != "openai":
            raise ConfigError("components.alignmentor.embed_backend must be 'openai'")
        if "neighbor" in cfg.alignmentor.highlevel_stages and cfg.alignmentor.neighbor_shared_ttp_threshold < 1:
            raise ConfigError("components.alignmentor.neighbor_shared_ttp_threshold must be >= 1")
