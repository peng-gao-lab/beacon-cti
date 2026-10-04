#!/usr/bin/env python3
"""
LLM Verifier for semantic edge verification
"""

import json
import os
from typing import List, Dict, Any, Optional
from jinja2 import Template
from pathlib import Path
from .models import CandidateEdgeInstance, HighLevelNode, IoCNode, TTPNode

try:
    from beacon_cost.recorder import (
        monotonic_time,
        record_openai_error,
        record_openai_response,
    )
except ImportError:  # Cost tracking must never affect verification.
    from time import monotonic as monotonic_time

    def record_openai_error(**kwargs):
        return None

    def record_openai_response(**kwargs):
        return None


class LLMVerifier:
    """Base class for LLM verifiers"""
    
    def verify_batch(
        self,
        batch_context: List[str],
        candidates: List[CandidateEdgeInstance],
        node_map: Dict[str, Any],
        threshold: float = 0.8,
        evidence_indices: Optional[List[int]] = None,
        ttp_description_map: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Verify a batch of candidate edges
        
        Args:
            batch_context: Context sentences (ordered by evidence_indices)
            candidates: List of candidate edges in this batch
            node_map: Dictionary mapping node_id to node objects
            threshold: Confidence threshold
            evidence_indices: Global sentence indices corresponding to batch_context (optional)
            ttp_description_map: Mapping from TTP ID to MITRE description (optional)
        
        Returns:
            List of verification results, each containing:
            {
                "candidate_index": int,
                "verified": bool,
                "confidence": float,
                "reasoning": str
            }
        """
        raise NotImplementedError

    def verify_center_entity_missing_ttps(
        self,
        sentences: List[str],
        center_entity: Dict[str, Any],
        ttp_candidates: List[Dict[str, Any]],
        threshold: float = 0.8,
        ttp_description_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Verify whether the potential center entity is truly central, then verify
        missing center-entity ↔ TTP links.

        Args:
            sentences: Full original text split into indexed sentences
            center_entity: Potential center entity metadata
            ttp_candidates: Unlinked TTP candidates and their evidence indices
            threshold: Confidence threshold
            ttp_description_map: Mapping from TTP ID to MITRE description (optional)

        Returns:
            {
                "is_center_entity": bool,
                "center_entity_reasoning": str,
                "results": [
                    {
                        "ttp_node_id": str,
                        "ttp_id": str,
                        "verified": bool,
                        "confidence": float,
                        "reasoning": str,
                        "model": str
                    }
                ]
            }
        """
        raise NotImplementedError


class OpenAIVerifier(LLMVerifier):
    """OpenAI-based verifier"""
    
    def __init__(self, model: str = "gpt-5"):
        """
        Initialize OpenAI verifier
        
        Args:
            model: Model name (e.g., "gpt-5", "gpt-4o", "gpt-5-mini")
        """
        self.model = model
        self._load_prompt_template()
    
    def _load_prompt_template(self):
        """Load prompt templates from files"""
        prompt_dir = Path(__file__).parent / "prompts"
        edge_template_path = prompt_dir / "augmentor_verify_edges.jinja"
        center_template_path = prompt_dir / "augmentor_verify_center_entity_ttp.jinja"

        with open(edge_template_path, 'r', encoding='utf-8') as f:
            self.edge_template = Template(f.read())

        with open(center_template_path, 'r', encoding='utf-8') as f:
            self.center_template = Template(f.read())

    @staticmethod
    def _extract_json_text(response_text: str) -> str:
        """Extract JSON text from model response (supports fenced blocks)."""
        normalized = response_text.strip()
        if not normalized.startswith("```"):
            return normalized

        lines = normalized.split('\n')
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip().startswith("```"):
                if not in_json:
                    in_json = True
                else:
                    break
            elif in_json:
                json_lines.append(line)
        return '\n'.join(json_lines).strip()
    
    def _call_openai(
        self,
        prompt: str,
        max_retries: int = 2,
        cost_stage: str = "unspecified",
    ) -> str:
        """
        Call OpenAI API
        
        Args:
            prompt: Prompt text
            max_retries: Maximum number of retries
        
        Returns:
            Response text
        """
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            client = OpenAI(api_key=api_key)
            
            # Prepare API call parameters
            # GPT-5 doesn't support temperature=0.0, only default value (1) is supported
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a semantic verification system for cybersecurity knowledge graphs. You must output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            
            # Only set temperature for models that support it (not gpt-5)
            if not self.model.startswith("gpt-5"):
                api_params["temperature"] = 0.0
            
            for attempt in range(max_retries):
                api_response_received = False
                try:
                    started_at = monotonic_time()
                    response = client.chat.completions.create(**api_params)
                    api_response_received = True
                    record_openai_response(
                        module="CTIAugmentor",
                        stage=cost_stage,
                        endpoint="chat.completions",
                        model_requested=self.model,
                        response=response,
                        latency_seconds=monotonic_time() - started_at,
                        application_attempt=attempt + 1,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if not api_response_received:
                        record_openai_error(
                            module="CTIAugmentor",
                            stage=cost_stage,
                            endpoint="chat.completions",
                            model_requested=self.model,
                            error=e,
                            latency_seconds=monotonic_time() - started_at,
                            application_attempt=attempt + 1,
                        )
                    if attempt < max_retries - 1:
                        continue
                    else:
                        raise e
            
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")
    
    def verify_batch(
        self,
        batch_context: List[str],
        candidates: List[CandidateEdgeInstance],
        node_map: Dict[str, Any],
        threshold: float = 0.8,
        evidence_indices: Optional[List[int]] = None,
        ttp_description_map: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Verify a batch of candidate edges using OpenAI
        
        Args:
            batch_context: Context sentences (ordered by evidence_indices)
            candidates: List of candidate edges
            node_map: Dictionary mapping node_id to node objects
            threshold: Confidence threshold
            evidence_indices: Global sentence indices corresponding to batch_context (optional)
            ttp_description_map: Mapping from TTP ID to MITRE description (optional)
        
        Returns:
            List of verification results
        """
        # Get evidence indices for this batch (should be same for all candidates in batch)
        if evidence_indices is None:
            evidence_indices = candidates[0].evidence_indices_set if candidates else []
        if ttp_description_map is None:
            ttp_description_map = {}
        
        # Create global to local index mapping
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(evidence_indices)}
        
        # Prepare candidate data for prompt
        candidate_data = []
        ttp_definitions = []
        seen_ttp_ids = set()
        for i, candidate in enumerate(candidates):
            node_u = node_map.get(candidate.node_u_id)
            node_v = node_map.get(candidate.node_v_id)
            
            if not node_u or not node_v:
                continue
            
            # Determine which is left (high_level/ioc) and which is right (ttp)
            if candidate.edge_type == "high_level-ttp":
                left_node = node_u
                right_node = node_v
            else:  # ioc-ttp
                left_node = node_u
                right_node = node_v
            
            # Map trigger_sentence_index to local index
            left_trigger_local = global_to_local.get(candidate.trigger_sentence_index, 0)
            
            # Map ttp_span_range to local indices
            ttp_span_local = []
            for global_idx in range(candidate.ttp_span_range[0], candidate.ttp_span_range[1] + 1):
                if global_idx in global_to_local:
                    ttp_span_local.append(global_to_local[global_idx])
            
            # If no overlap, use trigger index as fallback
            if not ttp_span_local:
                ttp_span_local = [left_trigger_local]
            
            candidate_data.append({
                "left_node_text": left_node.node_text if hasattr(left_node, 'node_text') else str(left_node),
                "left_node_class": left_node.node_class if hasattr(left_node, 'node_class') else "unknown",
                "left_node_type": left_node.node_type if hasattr(left_node, 'node_type') else "unknown",
                "right_node_text": right_node.node_text if hasattr(right_node, 'node_text') else str(right_node),
                "right_node_class": right_node.node_class if hasattr(right_node, 'node_class') else "unknown",
                "right_node_type": right_node.node_type if hasattr(right_node, 'node_type') else "unknown",
                "left_trigger_sentence_local_index": left_trigger_local,
                "ttp_span_range_local": ttp_span_local if ttp_span_local else [left_trigger_local],
                "min_dist": candidate.min_dist
            })

            right_ttp_id = ""
            if hasattr(right_node, "ttp_id"):
                right_ttp_id = str(right_node.ttp_id).strip().upper()

            if not right_ttp_id and hasattr(right_node, "node_text"):
                right_text = str(right_node.node_text).strip()
                if right_text.upper().startswith("TTP:"):
                    right_ttp_id = right_text.split(":", 1)[1].strip().upper()

            if right_ttp_id and right_ttp_id not in seen_ttp_ids:
                ttp_description = str(ttp_description_map.get(right_ttp_id, "")).strip()
                if ttp_description:
                    ttp_definitions.append({
                        "ttp_id": right_ttp_id,
                        "description": ttp_description
                    })
                    seen_ttp_ids.add(right_ttp_id)
        
        if not candidate_data:
            return []
        
        # Determine node descriptions
        if candidates and candidates[0].edge_type == "high_level-ttp":
            left_desc = "High Level Entity (e.g., MalwareFamily, Campaign, Organization)"
            right_desc = "TTP (Tactics, Techniques, and Procedures from MITRE ATT&CK)"
        else:
            left_desc = "IoC (Indicator of Compromise, e.g., IP, domain, hash)"
            right_desc = "TTP (Tactics, Techniques, and Procedures from MITRE ATT&CK)"
        
        # Render prompt
        prompt = self.edge_template.render(
            context_sentences=batch_context,
            ttp_definitions=ttp_definitions,
            candidate_edges=candidate_data,
            threshold=threshold,
            left_node_description=left_desc,
            right_node_description=right_desc
        )
        
        # Call OpenAI
        try:
            response_text = self._call_openai(prompt, cost_stage="edge_verification")
            
            # Parse JSON response
            response_text = self._extract_json_text(response_text)
            
            result_data = json.loads(response_text)
            
            # Convert to our format
            results = []
            for result in result_data.get("results", []):
                candidate_idx = result.get("candidate_index", 0)
                confidence = float(result.get("confidence", 0.0))
                reasoning = result.get("reasoning", "")
                
                if candidate_idx < 1 or candidate_idx > len(candidates):
                    continue
                
                actual_candidate_idx = candidate_idx - 1  # Convert from 1-based to 0-based
                
                # Map back to original candidate index in filtered list
                if actual_candidate_idx < len(candidates):
                    results.append({
                        "candidate_index": actual_candidate_idx,
                        "verified": confidence >= threshold,
                        "confidence": confidence,
                        "reasoning": reasoning,
                        "model": self.model
                    })
            
            return results
            
        except json.JSONDecodeError as e:
            # Retry once with error message
            error_prompt = prompt + f"\n\nPrevious error: {str(e)}\nPlease output valid JSON only, no markdown."
            try:
                response_text = self._call_openai(
                    error_prompt,
                    cost_stage="edge_verification_json_retry",
                )
                response_text = self._extract_json_text(response_text)
                result_data = json.loads(response_text)
                # Process same as above
                results = []
                for result in result_data.get("results", []):
                    candidate_idx = result.get("candidate_index", 0)
                    confidence = float(result.get("confidence", 0.0))
                    reasoning = result.get("reasoning", "")
                    
                    if candidate_idx < 1 or candidate_idx > len(candidates):
                        continue
                    
                    actual_candidate_idx = candidate_idx - 1
                    if actual_candidate_idx < len(candidates):
                        results.append({
                            "candidate_index": actual_candidate_idx,
                            "verified": confidence >= threshold,
                            "confidence": confidence,
                            "reasoning": reasoning,
                            "model": self.model
                        })
                return results
            except:
                # If retry fails, return empty results
                return []

    def verify_center_entity_missing_ttps(
        self,
        sentences: List[str],
        center_entity: Dict[str, Any],
        ttp_candidates: List[Dict[str, Any]],
        threshold: float = 0.8,
        ttp_description_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Verify center-entity status and supplementary center-entity ↔ TTP links.
        """
        if not ttp_candidates:
            return {
                "is_center_entity": False,
                "center_entity_reasoning": "No unlinked TTP candidates",
                "results": []
            }

        if ttp_description_map is None:
            ttp_description_map = {}

        prepared_candidates: List[Dict[str, Any]] = []
        seen_ttp_ids = set()
        ttp_definitions: List[Dict[str, str]] = []

        for candidate in ttp_candidates:
            ttp_node_id = str(candidate.get("ttp_node_id", "")).strip()
            ttp_id = str(candidate.get("ttp_id", "")).strip().upper()
            ttp_node_text = str(candidate.get("ttp_node_text", "")).strip()
            evidence_indices = candidate.get("evidence_sentence_indices_global", [])

            if not ttp_node_id:
                continue

            if not isinstance(evidence_indices, list):
                evidence_indices = []
            evidence_indices = [
                int(item) for item in evidence_indices
                if isinstance(item, int) and 0 <= item < len(sentences)
            ]

            prepared_candidates.append({
                "candidate_index": len(prepared_candidates) + 1,
                "ttp_node_id": ttp_node_id,
                "ttp_id": ttp_id,
                "ttp_node_text": ttp_node_text,
                "evidence_sentence_indices_global": evidence_indices
            })

            if ttp_id and ttp_id not in seen_ttp_ids:
                description = str(ttp_description_map.get(ttp_id, "")).strip()
                if description:
                    ttp_definitions.append({
                        "ttp_id": ttp_id,
                        "description": description
                    })
                    seen_ttp_ids.add(ttp_id)

        if not prepared_candidates:
            return {
                "is_center_entity": False,
                "center_entity_reasoning": "No valid unlinked TTP candidates",
                "results": []
            }

        prompt = self.center_template.render(
            context_sentences=sentences,
            center_entity=center_entity,
            ttp_candidates=prepared_candidates,
            ttp_definitions=ttp_definitions,
            threshold=threshold
        )

        try:
            response_text = self._call_openai(
                prompt,
                cost_stage="center_entity_ttp_verification",
            )
            response_text = self._extract_json_text(response_text)
            result_data = json.loads(response_text)
        except Exception:
            return {
                "is_center_entity": False,
                "center_entity_reasoning": "Failed to parse model response",
                "results": []
            }

        is_center_entity = bool(result_data.get("is_center_entity", False))
        center_reasoning = str(result_data.get("center_entity_reasoning", "")).strip()

        if not is_center_entity:
            return {
                "is_center_entity": False,
                "center_entity_reasoning": center_reasoning,
                "results": []
            }

        # Map candidate_index -> prepared candidate metadata
        candidate_index_map = {
            item["candidate_index"]: item for item in prepared_candidates
        }

        results: List[Dict[str, Any]] = []
        seen_candidate_indices = set()
        for raw_result in result_data.get("results", []):
            candidate_index = raw_result.get("candidate_index", 0)
            if not isinstance(candidate_index, int):
                continue
            if candidate_index in seen_candidate_indices:
                continue

            candidate_meta = candidate_index_map.get(candidate_index)
            if candidate_meta is None:
                continue

            confidence = float(raw_result.get("confidence", 0.0))
            reasoning = str(raw_result.get("reasoning", "")).strip()

            results.append({
                "ttp_node_id": candidate_meta["ttp_node_id"],
                "ttp_id": candidate_meta["ttp_id"],
                "verified": confidence >= threshold,
                "confidence": confidence,
                "reasoning": reasoning,
                "model": self.model
            })
            seen_candidate_indices.add(candidate_index)

        return {
            "is_center_entity": True,
            "center_entity_reasoning": center_reasoning,
            "results": results
        }
