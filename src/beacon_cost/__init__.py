"""Fail-open sidecar cost tracking for BEACON API calls."""

from .recorder import record_openai_error, record_openai_response

__all__ = ["record_openai_error", "record_openai_response"]
