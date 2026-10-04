from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from beacon_cost.recorder import (
        monotonic_time,
        record_openai_error,
        record_openai_response,
    )
except ImportError:  # Cost tracking must never affect alignment.
    from time import monotonic as monotonic_time

    def record_openai_error(**kwargs):
        return None

    def record_openai_response(**kwargs):
        return None


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class DisabledEmbedder(BaseEmbedder):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        raise RuntimeError("Embedding backend is disabled")


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        model: str,
        api_key_env: str,
        timeout_sec: int = 60,
    ) -> None:
        self.model = model
        self.timeout_sec = timeout_sec
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key env: {api_key_env}")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        url = "https://api.openai.com/v1/embeddings"
        body = {"model": self.model, "input": list(texts)}
        req = Request(
            url=url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            data=json.dumps(body).encode("utf-8"),
        )
        started_at = monotonic_time()
        try:
            with urlopen(req, timeout=self.timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                http_status = resp.getcode()
                request_id = resp.headers.get("x-request-id")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            record_openai_error(
                module="CTIAlignmentor",
                stage="highlevel_embedding",
                endpoint="embeddings",
                model_requested=self.model,
                error=exc,
                latency_seconds=monotonic_time() - started_at,
                application_attempt=1,
                http_status=exc.code,
                request_id=exc.headers.get("x-request-id") if exc.headers else None,
                extra={"input_count": len(texts)},
            )
            raise RuntimeError(f"OpenAI embedding request failed: {exc.code} {detail}") from exc
        except URLError as exc:
            record_openai_error(
                module="CTIAlignmentor",
                stage="highlevel_embedding",
                endpoint="embeddings",
                model_requested=self.model,
                error=exc,
                latency_seconds=monotonic_time() - started_at,
                application_attempt=1,
                extra={"input_count": len(texts)},
            )
            raise RuntimeError(f"OpenAI embedding request failed: {exc.reason}") from exc

        record_openai_response(
            module="CTIAlignmentor",
            stage="highlevel_embedding",
            endpoint="embeddings",
            model_requested=self.model,
            response=payload,
            latency_seconds=monotonic_time() - started_at,
            application_attempt=1,
            request_id=request_id,
            http_status=http_status,
            extra={"input_count": len(texts)},
        )

        data = payload.get("data", [])
        if not isinstance(data, list):
            return []
        vectors: list[list[float]] = []
        for item in sorted(data, key=lambda x: int(x.get("index", 0))):
            vec = item.get("embedding", [])
            if isinstance(vec, list):
                try:
                    vectors.append([float(v) for v in vec])
                except (TypeError, ValueError):
                    vectors.append([])
        return vectors
