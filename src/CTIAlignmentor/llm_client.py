from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from jinja2 import Template

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

OPENAI_CHAT_TEMPERATURE = 0.0


@dataclass
class LLMClusterResponse:
    clusters: list[list[str]]
    cluster_confidence: list[float]
    node_confidence: dict[str, dict[str, float]]


@dataclass
class FilteredCluster:
    node_ids: list[str]
    cluster_confidence: float
    node_confidence: dict[str, float]


class BaseLLMClient(ABC):
    @abstractmethod
    def cluster_group(
        self,
        *,
        stage: str,
        node_type: str,
        nodes: list[dict[str, Any]],
    ) -> LLMClusterResponse:
        raise NotImplementedError


class DisabledLLMClient(BaseLLMClient):
    def cluster_group(
        self,
        *,
        stage: str,
        node_type: str,
        nodes: list[dict[str, Any]],
    ) -> LLMClusterResponse:
        return LLMClusterResponse(clusters=[], cluster_confidence=[], node_confidence={})


class OpenAILLMClient(BaseLLMClient):
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
        self._load_prompt_templates()

    def _load_prompt_templates(self) -> None:
        prompt_dir = Path(__file__).parent / "prompts"
        system_path = prompt_dir / "alignor_system_prompt.jinja"
        user_path = prompt_dir / "alignor_user_prompt.jinja"

        with system_path.open("r", encoding="utf-8") as f:
            self.system_prompt_template = Template(f.read())

        with user_path.open("r", encoding="utf-8") as f:
            self.user_prompt_template = Template(f.read())

        self.system_prompt = self.system_prompt_template.render().strip()

    def _user_prompt(self, stage: str, node_type: str, nodes: list[dict[str, Any]]) -> str:
        return self.user_prompt_template.render(
            stage_json=json.dumps(stage, ensure_ascii=False),
            node_type_json=json.dumps(node_type, ensure_ascii=False),
            nodes_json=json.dumps(nodes, ensure_ascii=False),
        ).strip()

    @staticmethod
    def _strip_code_fence(raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        return raw.strip()

    def cluster_group(
        self,
        *,
        stage: str,
        node_type: str,
        nodes: list[dict[str, Any]],
    ) -> LLMClusterResponse:
        url = "https://api.openai.com/v1/chat/completions"
        body = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": self._user_prompt(stage=stage, node_type=node_type, nodes=nodes),
                },
            ],
        }
        # GPT-5-family Chat Completions only accepts its default temperature.
        # Omitting this request parameter is a compatibility adjustment and
        # leaves CTIAlignmentor's prompts, candidates, thresholds, and parsing unchanged.
        if not self.model.startswith("gpt-5"):
            body["temperature"] = OPENAI_CHAT_TEMPERATURE
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
                stage=stage,
                endpoint="chat.completions",
                model_requested=self.model,
                error=exc,
                latency_seconds=monotonic_time() - started_at,
                application_attempt=1,
                http_status=exc.code,
                request_id=exc.headers.get("x-request-id") if exc.headers else None,
                extra={"node_type": node_type, "node_count": len(nodes)},
            )
            raise RuntimeError(f"OpenAI LLM request failed: {exc.code} {detail}") from exc
        except URLError as exc:
            record_openai_error(
                module="CTIAlignmentor",
                stage=stage,
                endpoint="chat.completions",
                model_requested=self.model,
                error=exc,
                latency_seconds=monotonic_time() - started_at,
                application_attempt=1,
                extra={"node_type": node_type, "node_count": len(nodes)},
            )
            raise RuntimeError(f"OpenAI LLM request failed: {exc.reason}") from exc

        record_openai_response(
            module="CTIAlignmentor",
            stage=stage,
            endpoint="chat.completions",
            model_requested=self.model,
            response=payload,
            latency_seconds=monotonic_time() - started_at,
            application_attempt=1,
            request_id=request_id,
            http_status=http_status,
            extra={"node_type": node_type, "node_count": len(nodes)},
        )

        content = (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        content = self._strip_code_fence(str(content))
        parsed = json.loads(content)
        clusters = parsed.get("clusters", [])
        cluster_conf = parsed.get("cluster_confidence", [])
        node_conf = parsed.get("node_confidence", {})

        if not isinstance(clusters, list):
            clusters = []
        clean_clusters: list[list[str]] = []
        for cluster in clusters:
            if not isinstance(cluster, list):
                continue
            clean_clusters.append([str(node_id) for node_id in cluster])

        clean_cluster_conf: list[float] = []
        if isinstance(cluster_conf, list):
            for val in cluster_conf:
                try:
                    clean_cluster_conf.append(float(val))
                except (TypeError, ValueError):
                    clean_cluster_conf.append(0.0)
        while len(clean_cluster_conf) < len(clean_clusters):
            clean_cluster_conf.append(0.0)

        clean_node_conf: dict[str, dict[str, float]] = {}
        if isinstance(node_conf, dict):
            for node_id, cluster_map in node_conf.items():
                if isinstance(cluster_map, dict):
                    clean_node_conf[str(node_id)] = {}
                    for idx, conf in cluster_map.items():
                        try:
                            clean_node_conf[str(node_id)][str(idx)] = float(conf)
                        except (TypeError, ValueError):
                            continue
                elif isinstance(cluster_map, (float, int)):
                    clean_node_conf[str(node_id)] = {"0": float(cluster_map)}

        return LLMClusterResponse(
            clusters=clean_clusters,
            cluster_confidence=clean_cluster_conf,
            node_confidence=clean_node_conf,
        )


def filter_llm_clusters(
    response: LLMClusterResponse,
    *,
    node_threshold: float,
    cluster_threshold: float,
) -> list[FilteredCluster]:
    filtered: list[FilteredCluster] = []
    for idx, cluster in enumerate(response.clusters):
        cluster_conf = (
            response.cluster_confidence[idx]
            if idx < len(response.cluster_confidence)
            else 0.0
        )
        if cluster_conf < cluster_threshold:
            continue

        kept_nodes: list[str] = []
        kept_conf: dict[str, float] = {}
        for node_id in cluster:
            node_map = response.node_confidence.get(node_id, {})
            confidence = node_map.get(str(idx), node_map.get("0", 0.0))
            if confidence >= node_threshold:
                kept_nodes.append(node_id)
                kept_conf[node_id] = confidence
        if len(kept_nodes) < 2:
            continue
        filtered.append(
            FilteredCluster(
                node_ids=kept_nodes,
                cluster_confidence=cluster_conf,
                node_confidence=kept_conf,
            )
        )
    return filtered
