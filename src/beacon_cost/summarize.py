"""Aggregate BEACON API-call sidecars without reading prediction outputs."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - BEACON experiment hosts are Linux.
    fcntl = None


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
COST_FIELDS = (
    "uncached_input_cost_usd",
    "cached_input_cost_usd",
    "output_cost_usd",
    "total_cost_usd",
)


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _json_decimal(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000000000001")))


def _record_key(record: Dict[str, Any], fields: Tuple[str, ...]) -> str:
    return " / ".join(str(record.get(field) or "unknown") for field in fields)


def _empty_bucket() -> Dict[str, Any]:
    return {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "priced_calls": 0,
        "unpriced_successful_calls": 0,
        "unknown_failed_call_costs": 0,
        "tokens": {field: 0 for field in TOKEN_FIELDS},
        "cost_usd": {field: Decimal(0) for field in COST_FIELDS},
    }


def _add(bucket: Dict[str, Any], record: Dict[str, Any]) -> None:
    bucket["calls"] += 1
    status = record.get("status")
    if status == "success":
        bucket["successful_calls"] += 1
    else:
        bucket["failed_calls"] += 1
        bucket["unknown_failed_call_costs"] += 1

    usage = record.get("usage") or {}
    for field in TOKEN_FIELDS:
        value = usage.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            bucket["tokens"][field] += value

    cost = record.get("cost") or {}
    total_cost = _decimal(cost.get("total_cost_usd"))
    if status == "success":
        if total_cost is None:
            bucket["unpriced_successful_calls"] += 1
        else:
            bucket["priced_calls"] += 1
    for field in COST_FIELDS:
        value = _decimal(cost.get(field))
        if value is not None:
            bucket["cost_usd"][field] += value


def _finalize(bucket: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(bucket)
    result["tokens"] = dict(bucket["tokens"])
    result["cost_usd"] = {
        field: _json_decimal(value) for field, value in bucket["cost_usd"].items()
    }
    result["cost_complete_for_successful_calls"] = (
        bucket["unpriced_successful_calls"] == 0
    )
    return result


def _iter_records(root: Path) -> Iterable[Tuple[Path, int, Dict[str, Any]]]:
    paths: List[Path]
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*.jsonl"))
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError("Invalid JSONL at %s:%d: %s" % (path, line_number, exc))
                    if record.get("record_type") == "api_call":
                        yield path, line_number, record
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def summarize(root: Path) -> Dict[str, Any]:
    total = _empty_bucket()
    dimensions = {
        "by_module": ("module",),
        "by_stage": ("module", "stage"),
        "by_model": ("model_requested",),
        "by_endpoint": ("endpoint",),
        "by_scope": ("scope_type", "group_id", "scope_id"),
        "by_run": ("run_id",),
    }
    buckets: Dict[str, Dict[str, Dict[str, Any]]] = {
        name: defaultdict(_empty_bucket) for name in dimensions
    }
    source_files = set()
    pricebook_versions = set()
    run_ids = set()
    scope_ids = set()

    for path, _line_number, record in _iter_records(root):
        source_files.add(str(path))
        _add(total, record)
        for name, fields in dimensions.items():
            _add(buckets[name][_record_key(record, fields)], record)
        pricebook_version = (record.get("pricing") or {}).get("pricebook_version")
        if pricebook_version:
            pricebook_versions.add(str(pricebook_version))
        if record.get("run_id"):
            run_ids.add(str(record["run_id"]))
        if record.get("scope_id"):
            scope_ids.add(
                (
                    str(record.get("scope_type") or "unknown"),
                    str(record.get("group_id") or "unknown"),
                    str(record["scope_id"]),
                )
            )

    output: Dict[str, Any] = {
        "schema_version": "1.0",
        "record_type": "beacon_cost_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "source_jsonl_files": len(source_files),
        "run_ids": sorted(run_ids),
        "scope_count": len(scope_ids),
        "pricebook_versions": sorted(pricebook_versions),
        "cost_semantics": (
            "Estimated standard-tier API cost computed from provider response usage; "
            "not an invoice. Failed calls without response usage remain unknown."
        ),
        "total": _finalize(total),
    }
    for name in dimensions:
        output[name] = {
            key: _finalize(bucket) for key, bucket in sorted(buckets[name].items())
        }
    return output


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize BEACON cost JSONL sidecars")
    parser.add_argument("--input", required=True, help="JSONL file or directory to scan recursively")
    parser.add_argument("--output", required=True, help="Summary JSON path")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        parser.error("input does not exist: %s" % input_path)
    result = summarize(input_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        ".%s.%d.tmp" % (output_path.name, os.getpid())
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary_path), str(output_path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print("BEACON cost summary: %s" % output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
