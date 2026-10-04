"""Stage scheduling for BEACON.

Each component is executed as a subprocess of the current interpreter through
its original module entry point (``python -m CTIHighLevelExtractor`` etc.), so
the components keep their own process, imports and printing. The runner only
decides *what* to run, passes the configured parameters, captures the console
output into per-report log files and writes the extract views.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import extract as extract_views
from .config import COMPONENT_DIRS, BeaconConfig, ConfigError

ALIGN_OUTPUT_PREFIX = "[CTIAlignmentor] Output: "

# Console messages the components print when they give up on an LLM call or
# on parsing its response and continue with a partial result (they still exit
# with code 0). Only the final "gave up" messages are listed, not the
# "retrying" ones. CTIAugmentor raises on failed edge verification, so it
# exits non-zero by itself; its center-entity supplement swallows errors
# without printing anything and cannot be detected here.
COMPONENT_FAILURE_MARKERS = {
    "highlevel": ("LLM call failed:", "JSON parsing failed", "Precision check failed"),
    "tagger": ("❌",),
    "ioc": ("Warning: Error",),
    "augmentor": (),
}
TAIL_LINES = 40
EXTRACTION_STAGES = ("highlevel", "tagger", "ioc", "augmentor")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class StageRecord:
    status: str = "not_run"  # ok | skipped | failed | not_run
    output: str | None = None
    extract: str | None = None
    log: str | None = None
    seconds: float = 0.0
    message: str | None = None


@dataclass
class ReportJob:
    report_id: str
    group_id: str | None
    input_txt: Path
    out_base: Path
    stages: dict[str, StageRecord] = field(default_factory=dict)
    status: str = "pending"  # ok | failed | skipped

    @property
    def label(self) -> str:
        return f"{self.group_id}/{self.report_id}" if self.group_id else self.report_id


@dataclass
class AlignJob:
    set_id: str
    group_id: str | None
    out_base: Path
    members: list[Path]
    record: StageRecord = field(default_factory=StageRecord)
    member_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
def nltk_status() -> dict[str, Any]:
    """Report which sentence splitter / stopword list the components will use.

    The components try NLTK first and silently fall back to their built-in
    regex splitter (and a short built-in stopword list) when the NLTK data is
    missing. The reference GPT-4o runs used the regex fallback for sentence
    splitting and the NLTK ``stopwords`` corpus.
    """
    info: dict[str, Any] = {}
    try:
        import nltk  # noqa: F401
    except ImportError:
        info.update(nltk="missing", sentence_splitter="regex-fallback", stopwords="builtin-fallback")
        return info
    import nltk

    info["nltk"] = getattr(nltk, "__version__", "unknown")

    def has(resource: str) -> bool:
        try:
            nltk.data.find(resource)
            return True
        except LookupError:
            return False

    info["punkt"] = has("tokenizers/punkt")
    info["punkt_tab"] = has("tokenizers/punkt_tab")
    try:
        from nltk.tokenize import sent_tokenize

        sent_tokenize("First sentence. Second sentence.")
        info["sentence_splitter"] = "nltk-punkt"
    except Exception:
        info["sentence_splitter"] = "regex-fallback"
    info["stopwords"] = "nltk" if has("corpora/stopwords") else "builtin-fallback"
    return info


def _needs_api_key(cfg: BeaconConfig) -> str | None:
    """Return the env var name that must hold the API key, or None."""
    if cfg.input.kind == "reports" and any(cfg.component(c).enabled for c in EXTRACTION_STAGES):
        return "OPENAI_API_KEY"
    if cfg.alignmentor.enabled and cfg.alignmentor.enable_llm:
        return cfg.alignmentor.api_key_env
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg: BeaconConfig, *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._print_lock = threading.Lock()
        self.report_jobs: list[ReportJob] = []
        self.align_jobs: list[AlignJob] = []
        self.preflight: dict[str, Any] = {}

    # -- console -----------------------------------------------------------
    def say(self, msg: str) -> None:
        with self._print_lock:
            print(msg, flush=True)

    # -- enumeration -------------------------------------------------------
    def _list_files(self, directory: Path) -> list[Path]:
        return sorted(p for p in directory.glob(self.cfg.input.pattern) if p.is_file())

    def enumerate_inputs(self) -> list[tuple[str | None, Path, list[Path]]]:
        """Return [(group_id, group_dir, files)] in stable sorted order.

        ``input.limit`` keeps the first N reports (flat layout) or the first N
        groups (groups layout) of the sorted listing, counted before any
        skip decision.
        """
        cfg = self.cfg
        sets: list[tuple[str | None, Path, list[Path]]] = []
        if cfg.input.layout == "flat":
            files = self._list_files(cfg.input.path)
            if cfg.input.limit is not None:
                files = files[: cfg.input.limit]
            sets.append((None, cfg.input.path, files))
        else:
            group_dirs = sorted(p for p in cfg.input.path.iterdir() if p.is_dir())
            if cfg.input.limit is not None:
                group_dirs = group_dirs[: cfg.input.limit]
            for gdir in group_dirs:
                sets.append((gdir.name, gdir, self._list_files(gdir)))
        return sets

    def _out_base(self, group_id: str | None) -> Path:
        root = self.cfg.output.root
        return root / "groups" / group_id if group_id else root

    def build_jobs(self) -> None:
        cfg = self.cfg
        for group_id, _gdir, files in self.enumerate_inputs():
            out_base = self._out_base(group_id)
            set_id = group_id if group_id else "flat"
            if cfg.input.kind == "reports":
                jobs = [ReportJob(report_id=f.stem, group_id=group_id, input_txt=f, out_base=out_base)
                        for f in files]
                self.report_jobs.extend(jobs)
                if cfg.alignmentor.enabled:
                    members = [out_base / COMPONENT_DIRS["augmentor"] / f"{j.report_id}.json" for j in jobs]
                    self.align_jobs.append(AlignJob(set_id=set_id, group_id=group_id, out_base=out_base,
                                                    members=members, member_ids=[j.report_id for j in jobs]))
            else:
                self.align_jobs.append(AlignJob(set_id=set_id, group_id=group_id, out_base=out_base,
                                                members=list(files), member_ids=[f.stem for f in files]))

    # -- preflight ---------------------------------------------------------
    def run_preflight(self) -> None:
        cfg = self.cfg
        info: dict[str, Any] = {
            "python": sys.executable,
            "python_version": sys.version.split()[0],
        }
        key_env = _needs_api_key(cfg)
        info["api_key_env"] = key_env
        if key_env and not os.environ.get(key_env):
            msg = f"Environment variable {key_env} is not set but an LLM-backed component is enabled"
            if self.dry_run:
                self.say(f"WARNING: {msg}")
            else:
                raise ConfigError(msg)
        if cfg.input.kind == "reports":
            n = nltk_status()
            n["configured"] = cfg.run.sentence_splitter
            if cfg.run.sentence_splitter == "nltk" and n.get("sentence_splitter") != "nltk-punkt":
                raise ConfigError(
                    "run.sentence_splitter=nltk but NLTK sentence tokenization is not available "
                    "(install nltk and run: python -m nltk.downloader punkt_tab)"
                )
            n["sentence_splitter"] = "regex" if cfg.run.sentence_splitter == "regex" else "nltk-punkt"
            info["nltk"] = n
        self.preflight = info

    # -- subprocess execution ----------------------------------------------
    def _cost_env(self, scope_type: str, scope_id: str, group_id: str | None) -> dict[str, str]:
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["PYTHONUNBUFFERED"] = "1"
        env["BEACON_SENTENCE_SPLITTER"] = self.cfg.run.sentence_splitter
        if self.cfg.cost.enabled:
            cost_dir = self.cfg.output.root / "cost"
            cost_dir.mkdir(parents=True, exist_ok=True)
            env["BEACON_COST_LOG"] = str(cost_dir / "api_calls.jsonl")
            env["BEACON_COST_RUN_ID"] = self.run_id
            env["BEACON_COST_SCOPE_TYPE"] = scope_type
            env["BEACON_COST_SCOPE_ID"] = scope_id
            env["BEACON_COST_GROUP_ID"] = group_id or ""
            env["BEACON_COST_PRICEBOOK"] = str(self.cfg.pricebook_path())
        else:
            for k in list(env):
                if k.startswith("BEACON_COST_"):
                    env.pop(k)
        return env

    def _execute(self, cmd: list[str], env: dict[str, str], log_path: Path | None,
                 label: str) -> tuple[int, list[str], list[str]]:
        """Run a component. Returns (returncode, tail lines, all lines)."""
        verbose = self.cfg.run.console == "verbose"
        tail: deque[str] = deque(maxlen=TAIL_LINES)
        all_lines: list[str] = []
        log_file = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w", encoding="utf-8")
            log_file.write("$ " + " ".join(cmd) + "\n\n")
        try:
            proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                tail.append(line)
                all_lines.append(line)
                if log_file is not None:
                    log_file.write(line + "\n")
                if verbose:
                    self.say(f"  [{label}] {line}")
            rc = proc.wait()
        finally:
            if log_file is not None:
                log_file.close()
        return rc, list(tail), all_lines

    def _report_failure(self, label: str, stage: str, rc: int, tail: list[str], log: Path | None) -> None:
        lines = [f"[{label}] {stage}: FAILED (exit code {rc})"]
        if log is not None:
            lines.append(f"    log: {log}")
        if tail and self.cfg.run.console != "verbose":
            lines.append("    last output lines:")
            lines.extend("    | " + t for t in tail[-15:])
        self.say("\n".join(lines))

    # -- command builders ----------------------------------------------------
    def _cmd_highlevel(self, job: ReportJob, out_dir: Path) -> list[str]:
        c = self.cfg.highlevel
        return [sys.executable, "-m", "CTIHighLevelExtractor",
                "--input_txt", str(job.input_txt), "--output_dir", str(out_dir),
                "--model", self.cfg.model_for("highlevel"), "--provider", "openai",
                "--chunk_size", str(c.chunk_size), "--overlap_size", str(c.overlap_size),
                "--token_overlap_threshold", str(c.token_overlap_threshold)]

    def _cmd_tagger(self, job: ReportJob, out_dir: Path) -> list[str]:
        c = self.cfg.tagger
        cmd = [sys.executable, "-m", "CTITagger",
               "--report", str(job.input_txt), "--mitre", str(self.cfg.resources.mitre_json),
               "--output", str(out_dir), "--provider", "openai",
               "--model", self.cfg.model_for("tagger"), "--threshold", str(c.threshold),
               "--lmax", str(c.lmax), "--high-freq-threshold", str(c.high_freq_threshold)]
        if not c.enable_ngram:
            cmd.append("--disable-ngram")
        return cmd

    def _cmd_ioc(self, job: ReportJob, out_dir: Path) -> list[str]:
        return [sys.executable, "-m", "CTIIOCExtractor.main",
                "--input", str(job.input_txt), "--output", str(out_dir),
                "--model", self.cfg.model_for("ioc"),
                "--pre-verification-output-dir", str(out_dir / "pre_verification")]

    def _cmd_augmentor(self, job: ReportJob, out_dir: Path) -> list[str]:
        c = self.cfg.augmentor
        stem = job.report_id
        cmd = [sys.executable, "-m", "CTIAugmentor",
               "--highlevel_json", str(job.out_base / COMPONENT_DIRS["highlevel"] / f"{stem}.json"),
               "--ioc_json", str(job.out_base / COMPONENT_DIRS["ioc"] / f"{stem}.json"),
               "--ttp_json", str(job.out_base / COMPONENT_DIRS["tagger"] / f"{stem}.json"),
               "--input_txt", str(job.input_txt),
               "--mitre_json", str(self.cfg.resources.mitre_json),
               "--output_dir", str(out_dir),
               "--model", self.cfg.model_for("augmentor"),
               "--threshold", str(c.threshold), "--delta", str(c.delta)]
        if c.check_split:
            cmd.append("--check_split")
        return cmd

    def _cmd_alignmentor(self, job: AlignJob, out_dir: Path, log_path: Path) -> list[str]:
        c = self.cfg.alignmentor
        cmd = [sys.executable, "-m", "CTIAlignmentor.align", "--input_paths"]
        cmd.extend(str(p) for p in job.members)
        cmd.extend(["--output_dir", str(out_dir), "--log_path", str(log_path),
                    "--node_threshold", str(c.node_threshold),
                    "--cluster_threshold", str(c.cluster_threshold)])
        if c.enable_llm:
            cmd.append("--enable_llm")
            cmd.append("--highlevel_stages")
            cmd.extend(c.highlevel_stages)
            cmd.extend(["--openai_model", self.cfg.model_for("alignmentor"),
                        "--openai_api_key_env", c.api_key_env,
                        "--embed_backend", c.embed_backend,
                        "--embed_model", self.cfg.embed_model(),
                        "--embed_sim_threshold", str(c.embed_sim_threshold),
                        "--max_neighbor_rounds", str(c.max_neighbor_rounds),
                        "--neighbor_shared_ttp_threshold", str(c.neighbor_shared_ttp_threshold)])
        if c.disable_highlevel_deterministic_char:
            cmd.append("--disable_highlevel_deterministic_char")
        return cmd

    # -- report stages -------------------------------------------------------
    def _write_extract(self, stage: str, output_json: Path, comp_dir: Path, rec: StageRecord) -> None:
        comp = self.cfg.component(stage)
        if not comp.extract:
            return
        extract_dir = comp_dir / "extract"
        try:
            rec.extract = str(extract_views.EXTRACTORS[stage](output_json, extract_dir))
        except Exception as exc:  # extract is a view; do not fail the stage
            rec.message = f"extract failed: {exc}"
            self.say(f"    extract for {stage} failed: {exc}")

    def _set_aside(self, output_json: Path, rec: StageRecord) -> None:
        """Move a failed stage's partial output to <component>/failed/ so that a
        later run with output.existing=skip does not mistake it for a result."""
        if not output_json.is_file():
            return
        failed_dir = output_json.parent / "failed"
        failed_dir.mkdir(parents=True, exist_ok=True)
        target = failed_dir / output_json.name
        output_json.replace(target)
        rec.output = str(target)

    def _run_stage(self, job: ReportJob, stage: str) -> bool:
        cfg = self.cfg
        comp = cfg.component(stage)
        comp_dir = job.out_base / COMPONENT_DIRS[stage]
        output_json = comp_dir / f"{job.report_id}.json"
        rec = StageRecord()
        job.stages[stage] = rec

        if cfg.output.existing == "skip" and output_json.is_file():
            rec.status = "skipped"
            rec.output = str(output_json)
            extract_path = comp_dir / "extract" / (
                f"{job.report_id}.txt" if stage == "highlevel" else f"{job.report_id}.json")
            if comp.extract and not extract_path.is_file():
                self._write_extract(stage, output_json, comp_dir, rec)
            elif comp.extract:
                rec.extract = str(extract_path)
            self.say(f"[{job.label}] {stage}: skipped (output exists)")
            return True

        comp_dir.mkdir(parents=True, exist_ok=True)
        log_path = comp_dir / "logs" / f"{job.report_id}.log" if comp.log else None
        builder = getattr(self, f"_cmd_{stage}")
        cmd = builder(job, comp_dir)
        env = self._cost_env("report", job.report_id, job.group_id)

        t0 = time.monotonic()
        rc, tail, lines = self._execute(cmd, env, log_path, f"{job.label}:{stage}")
        rec.seconds = round(time.monotonic() - t0, 1)
        rec.log = str(log_path) if log_path else None

        if rc != 0 or not output_json.is_file():
            rec.status = "failed"
            rec.message = f"exit code {rc}" if rc != 0 else "component exited 0 but produced no output file"
            self._set_aside(output_json, rec)
            self._report_failure(job.label, stage, rc, tail, log_path)
            return False

        markers = COMPONENT_FAILURE_MARKERS.get(stage, ())
        swallowed = [ln for ln in lines if any(m in ln for m in markers)]
        if swallowed:
            note = (f"component reported {len(swallowed)} internal LLM/parsing failure(s) and "
                    f"continued with a partial result; first: {swallowed[0].strip()[:160]}")
            if cfg.run.component_errors == "fail":
                rec.status = "failed"
                rec.message = note
                self._set_aside(output_json, rec)
                self.say(f"[{job.label}] {stage}: FAILED ({note})" + (f"\n    log: {log_path}" if log_path else ""))
                return False
            rec.message = note
            self.say(f"[{job.label}] {stage}: WARNING {note}")

        rec.status = "ok"
        rec.output = str(output_json)
        self._write_extract(stage, output_json, comp_dir, rec)
        self.say(f"[{job.label}] {stage}: ok ({rec.seconds}s) -> {output_json}")
        return True

    def process_report(self, job: ReportJob) -> None:
        stages = [s for s in EXTRACTION_STAGES if self.cfg.component(s).enabled]
        for stage in stages:
            if not self._run_stage(job, stage):
                job.status = "failed"
                for later in stages[stages.index(stage) + 1:]:
                    job.stages[later] = StageRecord(status="not_run", message="upstream stage failed")
                return
        statuses = {r.status for r in job.stages.values()}
        job.status = "skipped" if statuses == {"skipped"} else "ok"

    # -- alignment -----------------------------------------------------------
    def _existing_alignment_outputs(self, align_dir: Path) -> list[Path]:
        return sorted(p for p in align_dir.glob("*.json") if p.is_file())

    def process_alignment(self, job: AlignJob) -> None:
        cfg = self.cfg
        comp = cfg.alignmentor
        rec = job.record
        align_dir = job.out_base / COMPONENT_DIRS["alignmentor"]
        label = f"align:{job.set_id}"

        if not job.members:
            rec.status = "not_run"
            rec.message = "empty input set"
            self.say(f"[{label}]: not run (empty input set)")
            return

        # Members produced by this run must all be present.
        if cfg.input.kind == "reports":
            failed = [j.report_id for j in self.report_jobs
                      if j.group_id == job.group_id and j.status not in ("ok", "skipped")]
            if failed:
                rec.status = "not_run"
                rec.message = f"incomplete set; failed reports: {failed}"
                self.say(f"[{label}]: not run, set incomplete (failed: {failed})")
                return
        missing = [str(p) for p in job.members if not p.is_file()]
        if missing:
            rec.status = "not_run"
            rec.message = f"missing member graphs: {missing}"
            self.say(f"[{label}]: not run, missing graphs: {missing}")
            return

        existing = self._existing_alignment_outputs(align_dir)
        if existing and cfg.output.existing == "skip":
            rec.status = "skipped"
            rec.output = str(existing[-1]) if len(existing) == 1 else json.dumps([str(p) for p in existing])
            if comp.extract:
                for p in existing:
                    if not (align_dir / "extract" / p.name).is_file():
                        self._write_extract("alignmentor", p, align_dir, rec)
                rec.extract = str(align_dir / "extract")
            self.say(f"[{label}]: skipped (output exists: {[p.name for p in existing]})")
            return

        align_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = align_dir / "logs"
        if existing:  # overwrite policy
            for p in existing:
                p.unlink()
            shutil.rmtree(align_dir / "extract", ignore_errors=True)
            shutil.rmtree(logs_dir, ignore_errors=True)

        tmp_dir: str | None = None
        if comp.log:
            logs_dir.mkdir(parents=True, exist_ok=True)
            decision_log = logs_dir / "alignment.log"
            console_log: Path | None = logs_dir / "console.log"
        else:
            tmp_dir = tempfile.mkdtemp(prefix="beacon_align_")
            decision_log = Path(tmp_dir) / "alignment.log"
            console_log = None

        cmd = self._cmd_alignmentor(job, align_dir, decision_log)
        env = self._cost_env("group" if job.group_id else "set", job.set_id, job.group_id)
        t0 = time.monotonic()
        try:
            rc, tail, lines = self._execute(cmd, env, console_log, label)
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        rec.seconds = round(time.monotonic() - t0, 1)
        rec.log = str(logs_dir) if comp.log else None

        output_path: Path | None = None
        for line in lines:
            if line.startswith(ALIGN_OUTPUT_PREFIX):
                output_path = Path(line[len(ALIGN_OUTPUT_PREFIX):].strip())
        if rc != 0 or output_path is None or not output_path.is_file():
            rec.status = "failed"
            rec.message = f"exit code {rc}" if rc != 0 else "no output path reported by CTIAlignmentor"
            self._report_failure(label, "alignmentor", rc, tail, console_log)
            return

        rec.status = "ok"
        rec.output = str(output_path)
        if comp.log and not decision_log.exists():
            decision_log.touch()  # no merge decisions were logged
        self._write_extract("alignmentor", output_path, align_dir, rec)
        self.say(f"[{label}]: ok ({rec.seconds}s) -> {output_path}")

    # -- bookkeeping -----------------------------------------------------------
    def _run_dir(self) -> Path:
        return self.cfg.output.root / "beacon_runs" / self.run_id

    def write_config_snapshot(self) -> None:
        run_dir = self._run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg.to_plain(), f, sort_keys=False, allow_unicode=True)

    def summarize_cost(self) -> dict[str, Any] | None:
        if not self.cfg.cost.enabled:
            return None
        cost_dir = self.cfg.output.root / "cost"
        log = cost_dir / "api_calls.jsonl"
        summary = cost_dir / "summary.json"
        if not log.is_file():
            return {"log": str(log), "summary": None, "note": "no API calls recorded"}
        try:
            from beacon_cost.summarize import main as summarize_main

            rc = summarize_main(["--input", str(log), "--output", str(summary)])
            if rc != 0:
                self.say(f"WARN: cost summary exited with code {rc}")
        except Exception as exc:
            self.say(f"WARN: failed to summarize cost log: {exc}")
        return {"log": str(log), "summary": str(summary) if summary.is_file() else None}

    def status_payload(self, cost: dict[str, Any] | None) -> dict[str, Any]:
        reports = [{
            "report_id": j.report_id, "group_id": j.group_id, "input": str(j.input_txt),
            "status": j.status, "stages": {k: asdict(v) for k, v in j.stages.items()},
        } for j in self.report_jobs]
        alignments = [{
            "set_id": a.set_id, "group_id": a.group_id, "members": a.member_ids,
            "member_graphs": [str(p) for p in a.members], **asdict(a.record),
        } for a in self.align_jobs]
        counts = {
            "reports_total": len(self.report_jobs),
            "reports_ok": sum(j.status == "ok" for j in self.report_jobs),
            "reports_skipped": sum(j.status == "skipped" for j in self.report_jobs),
            "reports_failed": sum(j.status == "failed" for j in self.report_jobs),
            "alignments_total": len(self.align_jobs),
            "alignments_ok": sum(a.record.status == "ok" for a in self.align_jobs),
            "alignments_skipped": sum(a.record.status == "skipped" for a in self.align_jobs),
            "alignments_failed": sum(a.record.status == "failed" for a in self.align_jobs),
            "alignments_not_run": sum(a.record.status == "not_run" for a in self.align_jobs),
        }
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config_path": str(self.cfg.config_path),
            "output_root": str(self.cfg.output.root),
            "mode": {"kind": self.cfg.input.kind, "layout": self.cfg.input.layout},
            "components": self.cfg.enabled_components(),
            "preflight": self.preflight,
            "summary": counts,
            "cost": cost,
            "reports": reports,
            "alignments": alignments,
        }

    def print_plan(self) -> None:
        cfg = self.cfg
        self.say(f"BEACON run {self.run_id}")
        self.say(f"  config:      {cfg.config_path}")
        self.say(f"  input:       {cfg.input.path} (kind={cfg.input.kind}, layout={cfg.input.layout}, "
                 f"pattern={cfg.input.pattern}, limit={cfg.input.limit})")
        self.say(f"  output root: {cfg.output.root} (existing={cfg.output.existing})")
        self.say(f"  components:  {', '.join(cfg.enabled_components())}")
        models = ", ".join(f"{c}={cfg.model_for(c)}" for c in cfg.enabled_components())
        self.say(f"  models:      {models}" + (f"; embedding={cfg.embed_model()}"
                                                if cfg.alignmentor.enabled and cfg.alignmentor.enable_llm else ""))
        if cfg.input.kind == "reports":
            self.say(f"  reports:     {len(self.report_jobs)}")
        if self.align_jobs:
            self.say(f"  alignment sets: {len(self.align_jobs)} "
                     f"({', '.join(f'{a.set_id}[{len(a.members)}]' for a in self.align_jobs)})")
        if self.preflight.get("nltk"):
            n = self.preflight["nltk"]
            self.say(f"  nltk:        sentence_splitter={n.get('sentence_splitter')}, stopwords={n.get('stopwords')}")
        self.say(f"  workers:     {cfg.run.workers}; cost tracking: {'on' if cfg.cost.enabled else 'off'}")

    # -- main -----------------------------------------------------------------
    def run(self) -> int:
        self.build_jobs()
        self.run_preflight()
        self.print_plan()
        if self.dry_run:
            self.say("Dry run: nothing executed.")
            return 0

        self.cfg.output.root.mkdir(parents=True, exist_ok=True)
        self.write_config_snapshot()

        workers = self.cfg.run.workers
        if self.report_jobs:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self.process_report, self.report_jobs))
        if self.align_jobs:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(self.process_alignment, self.align_jobs))

        cost = self.summarize_cost()
        payload = self.status_payload(cost)
        status_path = self._run_dir() / "status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        s = payload["summary"]
        self.say("")
        self.say("Run finished.")
        if self.report_jobs:
            self.say(f"  reports:    total {s['reports_total']}, ok {s['reports_ok']}, "
                     f"skipped {s['reports_skipped']}, failed {s['reports_failed']}")
        if self.align_jobs:
            self.say(f"  alignments: total {s['alignments_total']}, ok {s['alignments_ok']}, "
                     f"skipped {s['alignments_skipped']}, failed {s['alignments_failed']}, "
                     f"not run {s['alignments_not_run']}")
        if cost and cost.get("summary"):
            self.say(f"  cost:       {cost['summary']}")
        self.say(f"  status:     {status_path}")
        failed = s["reports_failed"] + s["alignments_failed"] + s["alignments_not_run"]
        return 1 if failed else 0
