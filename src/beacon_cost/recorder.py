"""Record OpenAI usage without changing BEACON model behavior.

Tracking is enabled only when ``BEACON_COST_LOG`` is set.  All recorder
failures are swallowed deliberately: accounting must never change prompts,
responses, exceptions, retry behavior, or prediction outputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - BEACON experiment hosts are Linux.
    fcntl = None


SCHEMA_VERSION = "1.0"
DEFAULT_PRICEBOOK = Path(__file__).parent / "pricing" / "openai_standard_2026-07-21.json"

_counter_lock = threading.Lock()
_call_counter = 0
_warning_lock = threading.Lock()
_write_lock = threading.Lock()
_warning_emitted = False
_pricebook_cache: Dict[str, Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]] = {}


def _next_call_index() -> int:
    global _call_counter
    with _counter_lock:
        _call_counter += 1
        return _call_counter


def _warn_once(message: str) -> None:
    global _warning_emitted
    with _warning_lock:
        if _warning_emitted:
            return
        _warning_emitted = True
    print("[BEACON cost] WARNING: %s" % message, file=sys.stderr)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _decimal_json(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value.quantize(Decimal("0.000000000001")))


def _extract_usage(response: Any, endpoint: str) -> Dict[str, Optional[int]]:
    usage = _get(response, "usage")
    if usage is None:
        return {
            "input_tokens": None,
            "cached_input_tokens": None,
            "uncached_input_tokens": None,
            "output_tokens": None,
            "reasoning_output_tokens": None,
            "total_tokens": None,
        }

    input_tokens = _as_int(
        _first_not_none(_get(usage, "prompt_tokens"), _get(usage, "input_tokens"))
    )
    output_tokens = _as_int(
        _first_not_none(_get(usage, "completion_tokens"), _get(usage, "output_tokens"))
    )
    if endpoint == "embeddings" and output_tokens is None:
        output_tokens = 0

    input_details = _first_not_none(
        _get(usage, "prompt_tokens_details"), _get(usage, "input_tokens_details")
    )
    output_details = _first_not_none(
        _get(usage, "completion_tokens_details"), _get(usage, "output_tokens_details")
    )
    cached_tokens = _as_int(_get(input_details, "cached_tokens"))
    if cached_tokens is None and input_tokens is not None:
        cached_tokens = 0
    reasoning_tokens = _as_int(_get(output_details, "reasoning_tokens"))
    if reasoning_tokens is None and output_tokens is not None:
        reasoning_tokens = 0

    uncached_tokens: Optional[int] = None
    if input_tokens is not None and cached_tokens is not None:
        uncached_tokens = max(input_tokens - cached_tokens, 0)

    total_tokens = _as_int(_get(usage, "total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": uncached_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _load_pricebook() -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    configured_path = os.getenv("BEACON_COST_PRICEBOOK", "").strip()
    path = Path(configured_path) if configured_path else DEFAULT_PRICEBOOK
    cache_key = str(path)
    cached = _pricebook_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        raw = path.read_bytes()
        pricebook = json.loads(raw.decode("utf-8"))
        digest = hashlib.sha256(raw).hexdigest()
        result = (pricebook, digest, None)
    except Exception as exc:  # Fail open by design.
        result = (None, None, "%s: %s" % (type(exc).__name__, str(exc)))
    _pricebook_cache[cache_key] = result
    return result


def _match_price(
    pricebook: Dict[str, Any], model_requested: str, model_resolved: Optional[str]
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    candidates = [candidate for candidate in (model_resolved, model_requested) if candidate]
    for entry in pricebook.get("models", []):
        aliases = entry.get("model_ids", [])
        for candidate in candidates:
            if candidate in aliases:
                return str(entry.get("name", candidate)), entry
    return None, None


def _pricing_and_cost(
    *,
    model_requested: str,
    model_resolved: Optional[str],
    usage: Dict[str, Optional[int]],
) -> Tuple[Dict[str, Any], Dict[str, Optional[float]]]:
    pricebook, digest, pricebook_error = _load_pricebook()
    pricing: Dict[str, Any] = {
        "status": "unavailable",
        "pricebook_version": None,
        "pricebook_sha256": digest,
        "effective_date": None,
        "currency": "USD",
        "service_tier": "standard",
        "matched_model": None,
        "rates_per_million_tokens": None,
        "source_url": None,
    }
    costs: Dict[str, Optional[float]] = {
        "uncached_input_cost_usd": None,
        "cached_input_cost_usd": None,
        "output_cost_usd": None,
        "total_cost_usd": None,
    }

    if pricebook is None:
        pricing["reason"] = "pricebook_load_failed"
        pricing["detail"] = pricebook_error
        return pricing, costs

    pricing.update(
        {
            "pricebook_version": pricebook.get("pricebook_version"),
            "effective_date": pricebook.get("effective_date"),
            "currency": pricebook.get("currency", "USD"),
            "service_tier": pricebook.get("service_tier", "standard"),
        }
    )
    matched_name, entry = _match_price(pricebook, model_requested, model_resolved)
    if entry is None:
        pricing["reason"] = "model_not_in_pricebook"
        return pricing, costs

    rates = entry.get("rates_per_million_tokens", {})
    input_rate = _as_decimal(rates.get("input"))
    cached_rate = _as_decimal(rates.get("cached_input"))
    output_rate = _as_decimal(rates.get("output"))
    pricing.update(
        {
            "status": "matched",
            "matched_model": matched_name,
            "rates_per_million_tokens": {
                "input": float(input_rate) if input_rate is not None else None,
                "cached_input": float(cached_rate) if cached_rate is not None else None,
                "output": float(output_rate) if output_rate is not None else None,
            },
            "source_url": entry.get("source_url"),
        }
    )

    uncached_tokens = usage.get("uncached_input_tokens")
    cached_tokens = usage.get("cached_input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        uncached_tokens is None
        or cached_tokens is None
        or output_tokens is None
        or input_rate is None
        or cached_rate is None
        or output_rate is None
    ):
        pricing["status"] = "usage_unavailable"
        pricing["reason"] = "response_usage_incomplete"
        return pricing, costs

    million = Decimal(1_000_000)
    input_cost = Decimal(uncached_tokens) * input_rate / million
    cached_cost = Decimal(cached_tokens) * cached_rate / million
    output_cost = Decimal(output_tokens) * output_rate / million
    total_cost = input_cost + cached_cost + output_cost
    costs.update(
        {
            "uncached_input_cost_usd": _decimal_json(input_cost),
            "cached_input_cost_usd": _decimal_json(cached_cost),
            "output_cost_usd": _decimal_json(output_cost),
            "total_cost_usd": _decimal_json(total_cost),
        }
    )
    return pricing, costs


def _base_record(
    *,
    module: str,
    stage: str,
    endpoint: str,
    model_requested: str,
    application_attempt: Optional[int],
    latency_seconds: float,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "api_call",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "call_id": str(uuid.uuid4()),
        "call_index_in_process": _next_call_index(),
        "run_id": os.getenv("BEACON_COST_RUN_ID") or None,
        "scope_type": os.getenv("BEACON_COST_SCOPE_TYPE") or None,
        "scope_id": os.getenv("BEACON_COST_SCOPE_ID") or None,
        "group_id": os.getenv("BEACON_COST_GROUP_ID") or None,
        "module": module,
        "stage": stage,
        "provider": "openai",
        "endpoint": endpoint,
        "model_requested": model_requested,
        "application_attempt": application_attempt,
        "sdk_transport_attempts": None,
        "latency_seconds": round(max(float(latency_seconds), 0.0), 6),
    }


def _append_record(record: Dict[str, Any]) -> None:
    log_path = os.getenv("BEACON_COST_LOG", "").strip()
    if not log_path:
        return
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with _write_lock:
            fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                offset = 0
                while offset < len(encoded):
                    written = os.write(fd, encoded[offset:])
                    if written <= 0:
                        raise OSError("short write while appending cost record")
                    offset += written
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(fd)
    except Exception as exc:  # Cost tracking must never change inference.
        _warn_once("could not write %s (%s: %s)" % (log_path, type(exc).__name__, str(exc)))


def record_openai_response(
    *,
    module: str,
    stage: str,
    endpoint: str,
    model_requested: str,
    response: Any,
    latency_seconds: float,
    application_attempt: Optional[int] = None,
    request_id: Optional[str] = None,
    http_status: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a successful response. This function never raises."""
    if not os.getenv("BEACON_COST_LOG", "").strip():
        return
    try:
        usage = _extract_usage(response, endpoint)
        model_resolved = _get(response, "model")
        pricing, costs = _pricing_and_cost(
            model_requested=model_requested,
            model_resolved=str(model_resolved) if model_resolved else None,
            usage=usage,
        )
        record = _base_record(
            module=module,
            stage=stage,
            endpoint=endpoint,
            model_requested=model_requested,
            application_attempt=application_attempt,
            latency_seconds=latency_seconds,
        )
        response_request_id = _first_not_none(request_id, _get(response, "_request_id"))
        record.update(
            {
                "status": "success",
                "model_resolved": str(model_resolved) if model_resolved else None,
                "response_id": _get(response, "id"),
                "request_id": str(response_request_id) if response_request_id else None,
                "http_status": http_status,
                "usage": usage,
                "pricing": pricing,
                "cost": costs,
            }
        )
        if extra:
            record["context"] = dict(extra)
        _append_record(record)
    except Exception as exc:
        _warn_once("could not build cost record (%s: %s)" % (type(exc).__name__, str(exc)))


def record_openai_error(
    *,
    module: str,
    stage: str,
    endpoint: str,
    model_requested: str,
    error: BaseException,
    latency_seconds: float,
    application_attempt: Optional[int] = None,
    http_status: Optional[int] = None,
    request_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an application-level failed call with unknown billable usage."""
    if not os.getenv("BEACON_COST_LOG", "").strip():
        return
    try:
        record = _base_record(
            module=module,
            stage=stage,
            endpoint=endpoint,
            model_requested=model_requested,
            application_attempt=application_attempt,
            latency_seconds=latency_seconds,
        )
        error_digest = hashlib.sha256(str(error).encode("utf-8", errors="replace")).hexdigest()
        record.update(
            {
                "status": "error",
                "model_resolved": None,
                "response_id": None,
                "request_id": request_id,
                "http_status": http_status,
                "usage": {
                    "input_tokens": None,
                    "cached_input_tokens": None,
                    "uncached_input_tokens": None,
                    "output_tokens": None,
                    "reasoning_output_tokens": None,
                    "total_tokens": None,
                },
                "pricing": {
                    "status": "usage_unavailable",
                    "reason": "request_failed_without_response_usage",
                },
                "cost": {
                    "uncached_input_cost_usd": None,
                    "cached_input_cost_usd": None,
                    "output_cost_usd": None,
                    "total_cost_usd": None,
                },
                "error": {
                    "type": type(error).__name__,
                    "message_sha256": error_digest,
                },
            }
        )
        if extra:
            record["context"] = dict(extra)
        _append_record(record)
    except Exception as exc:
        _warn_once("could not build failed-call record (%s: %s)" % (type(exc).__name__, str(exc)))


def monotonic_time() -> float:
    """Small indirection used by instrumented clients and offline tests."""
    return time.monotonic()
