#!/usr/bin/env python3
"""
High-level Entity Extractor
Implements the core logic: sliding-window processing, position validation, deduplication and merging
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from jinja2 import Template

from .llm_provider import LLMProvider


# Enumeration of allowed entity types
ALLOWED_ENTITY_TYPES = {
    "ThreatActor",
    "Campaign",
    "MalwareFamily",
    "Tool",
    # "Vulnerability",
    "Organization",
    "Product",
    "Location",
    "Sector",
    "OtherStrict"
}


class EntityExtractor:
    """High-level Entity Extractor"""
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        chunk_size: int = 40,
        overlap_size: int = 2,
        token_overlap_threshold: float = 0.6,
        max_retries: int = 1
    ):
        """
        Initialize the entity extractor
        
        Args:
            llm_provider: LLM provider instance
            chunk_size: Number of sentences per chunk (default 40)
            overlap_size: Number of overlapping sentences between chunks (default 2)
            token_overlap_threshold: Token overlap threshold (default 0.6)
            max_retries: Maximum number of retries on JSON parse errors (default 1)
        """
        self.llm_provider = llm_provider
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.token_overlap_threshold = token_overlap_threshold
        self.max_retries = max_retries
        
        # Load the Jinja template
        template_path = Path(__file__).parent.parent / "prompts" / "highlevel_entities.jinja"
        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = Template(f.read())
    
    def extract(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extract contextual entities from a list of sentences
        
        Args:
            sentences: List of sentences
        
        Returns:
            List of entities (each containing entity_type, entity_text, sentence_indices)
        """
        all_entities = []
        
        # Sliding-window processing
        for chunk_start in range(0, len(sentences), self.chunk_size - self.overlap_size):
            chunk_end = min(chunk_start + self.chunk_size, len(sentences))
            chunk_sentences = sentences[chunk_start:chunk_end]
            
            print(f"Processing chunk [{chunk_start}:{chunk_end}] ({len(chunk_sentences)} sentences)...")
            
            # Call the LLM for extraction
            chunk_entities = self._extract_from_chunk(
                chunk_sentences,
                chunk_start
            )
            
            all_entities.extend(chunk_entities)
            print(f"  Extracted {len(chunk_entities)} entities")
        
        return all_entities
    
    def _extract_from_chunk(
        self,
        chunk_sentences: List[str],
        chunk_start: int
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from a single chunk
        
        Args:
            chunk_sentences: List of sentences in the chunk
            chunk_start: Start index of the chunk in the global sentence list
        
        Returns:
            List of entities (sentence_indices already converted to global indices)
        """
        # Build the prompt
        prompt = self._build_prompt(chunk_sentences)
        
        # Call the LLM (with retries)
        response_text = None
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response_text = self.llm_provider.call(
                    prompt=prompt,
                    max_tokens=8192,
                    cost_module="CTIHighLevelExtractor",
                    cost_stage="entity_extraction",
                    cost_attempt=attempt + 1,
                )
                break
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    print(f"  LLM call failed, retrying {attempt + 1}/{self.max_retries}...")
                    # On retry, the error message can be added to the prompt
                    prompt = self._build_prompt(chunk_sentences, error_hint=str(e))
                else:
                    print(f"  LLM call failed: {e}")
                    return []
        
        if not response_text:
            return []
        
        # Parse the JSON response
        parsed_entities = self._parse_response(
            response_text,
            chunk_sentences,
            chunk_start
        )
        
        # Position validation and correction
        validated_entities = []
        for entity in parsed_entities:
            validated = self._validate_and_fix_entity(
                entity,
                chunk_sentences,
                chunk_start
            )
            if validated:
                validated_entities.append(validated)
        
        return validated_entities
    
    def _build_prompt(
        self,
        chunk_sentences: List[str],
        error_hint: Optional[str] = None
    ) -> str:
        """
        Build the LLM prompt
        
        Args:
            chunk_sentences: List of sentences in the chunk
            error_hint: Error hint (used on retry)
        
        Returns:
            Prompt text
        """
        max_index = len(chunk_sentences) - 1
        
        prompt = self.prompt_template.render(
            sentences=enumerate(chunk_sentences),
            max_index=max_index
        )
        
        if error_hint:
            prompt += f"\n\nPrevious error: {error_hint}\nPlease ensure the output is valid JSON matching the schema."
        
        return prompt
    
    def _parse_response(
        self,
        response_text: str,
        chunk_sentences: List[str],
        chunk_start: int
    ) -> List[Dict[str, Any]]:
        """
        Parse the LLM response
        
        Args:
            response_text: Text returned by the LLM
            chunk_sentences: List of sentences in the chunk
            chunk_start: Global start index of the chunk
        
        Returns:
            List of parsed entities (sentence_indices are local indices)
        """
        entities = []
        
        try:
            # Try to extract JSON (may be wrapped in markdown)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if "entities" in data and isinstance(data["entities"], list):
                    for entity_data in data["entities"]:
                        # Validate required fields
                        if not isinstance(entity_data, dict):
                            continue
                        
                        entity_text = entity_data.get("entity_text", "").strip()
                        entity_type = entity_data.get("entity_type", "").strip()
                        sentence_indices = entity_data.get("sentence_indices", [])
                        
                        # Basic validation
                        if not entity_text or not entity_type:
                            continue
                        
                        if entity_type not in ALLOWED_ENTITY_TYPES:
                            print(f"  Warning: unknown entity type '{entity_type}', skipping")
                            continue
                        
                        if not isinstance(sentence_indices, list):
                            continue
                        
                        # Verify sentence_indices are within the valid range
                        valid_indices = [
                            idx for idx in sentence_indices
                            if isinstance(idx, int) and 0 <= idx < len(chunk_sentences)
                        ]
                        
                        if not valid_indices:
                            continue
                        
                        entities.append({
                            "entity_text": entity_text,
                            "entity_type": entity_type,
                            "sentence_indices": valid_indices  # local indices
                        })
        
        except json.JSONDecodeError as e:
            print(f"  JSON parsing failed: {e}")
            print(f"  Response content (first 500 chars): {response_text[:500]}")
        except Exception as e:
            print(f"  Error while parsing the response: {e}")
        
        return entities
    
    def _validate_and_fix_entity(
        self,
        entity: Dict[str, Any],
        chunk_sentences: List[str],
        chunk_start: int
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and correct the entity's position information
        
        Two-stage validation:
        1. Exact substring check
        2. Token overlap fallback
        
        If validation fails, automatically search within this chunk
        
        Args:
            entity: Entity dict (containing entity_text, entity_type, sentence_indices (local indices))
            chunk_sentences: List of sentences in the chunk
            chunk_start: Global start index of the chunk
        
        Returns:
            Validated entity (sentence_indices converted to global indices), or None if validation fails
        """
        entity_text = entity["entity_text"]
        local_indices = entity["sentence_indices"]
        
        # Stage 1: exact substring check
        validated_local_indices = []
        for local_idx in local_indices:
            if 0 <= local_idx < len(chunk_sentences):
                sentence = chunk_sentences[local_idx]
                if entity_text in sentence:
                    validated_local_indices.append(local_idx)
        
        # Stage 2: token overlap fallback (if every exact substring check fails)
        if not validated_local_indices:
            for local_idx in local_indices:
                if 0 <= local_idx < len(chunk_sentences):
                    sentence = chunk_sentences[local_idx]
                    if self._check_token_overlap(entity_text, sentence):
                        validated_local_indices.append(local_idx)
        
        # If still unresolved, automatically search within this chunk
        if not validated_local_indices:
            # First, a full exact-substring scan
            for local_idx in range(len(chunk_sentences)):
                sentence = chunk_sentences[local_idx]
                if entity_text in sentence:
                    validated_local_indices.append(local_idx)
                    break  # the first match is enough
        
        # If still unresolved, do a full token-overlap scan and take the top-1
        if not validated_local_indices:
            best_idx = None
            best_score = 0.0
            
            for local_idx in range(len(chunk_sentences)):
                sentence = chunk_sentences[local_idx]
                score = self._calculate_token_overlap_score(entity_text, sentence)
                if score >= self.token_overlap_threshold and score > best_score:
                    best_score = score
                    best_idx = local_idx
            
            if best_idx is not None:
                validated_local_indices.append(best_idx)
        
        # If it still fails in the end, discard the entity
        if not validated_local_indices:
            return None
        
        # Convert to global indices
        global_indices = [chunk_start + local_idx for local_idx in validated_local_indices]
        global_indices = sorted(set(global_indices))  # deduplicate and sort
        
        return {
            "entity_text": entity_text,
            "entity_type": entity["entity_type"],
            "sentence_indices": global_indices
        }
    
    def _check_token_overlap(self, entity_text: str, sentence: str) -> bool:
        """
        Check whether the token overlap reaches the threshold
        
        Args:
            entity_text: Entity text
            sentence: Sentence text
        
        Returns:
            Whether the threshold is reached
        """
        score = self._calculate_token_overlap_score(entity_text, sentence)
        return score >= self.token_overlap_threshold
    
    def _calculate_token_overlap_score(self, entity_text: str, sentence: str) -> float:
        """
        Compute the token overlap score (Jaccard similarity)
        
        Args:
            entity_text: Entity text
            sentence: Sentence text
        
        Returns:
            Overlap score (0.0-1.0)
        """
        # Simple tokenization: strip punctuation, lowercase, split
        def tokenize(text: str) -> Set[str]:
            # Remove punctuation, keep alphanumerics and common symbols
            text_clean = re.sub(r'[^\w\s-]', ' ', text.lower())
            tokens = set(text_clean.split())
            return tokens
        
        entity_tokens = tokenize(entity_text)
        sentence_tokens = tokenize(sentence)
        
        if not entity_tokens:
            return 0.0
        
        # Jaccard similarity
        intersection = len(entity_tokens & sentence_tokens)
        union = len(entity_tokens | sentence_tokens)
        
        if union == 0:
            return 0.0
        
        return intersection / union


class EntityMerger:
    """Entity deduplicator and merger"""
    
    @staticmethod
    def merge_entities(
        entities: List[Dict[str, Any]],
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Merge duplicate entities
        
        Args:
            entities: List of entities (each containing entity_type, entity_text, sentence_indices)
            sentences: List of sentences (used to back-fill span_sentences)
        
        Returns:
            List of merged entities (containing normalized_value, span_sentences, aliases)
        """
        # Group by (entity_type, normalized_value)
        entity_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        
        for entity in entities:
            entity_type = entity["entity_type"]
            entity_text = entity["entity_text"]
            normalized_value = EntityMerger._normalize_value(entity_type, entity_text)
            
            key = (entity_type, normalized_value)
            
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Merge each group
        merged_entities = []
        for key, group in entity_groups.items():
            merged = EntityMerger._merge_group(group, sentences)
            if merged:
                merged_entities.append(merged)
        
        return merged_entities
    
    @staticmethod
    def _normalize_value(entity_type: str, entity_text: str) -> str:
        """
        Normalize the entity value (used for deduplication)
        
        Args:
            entity_type: Entity type
            entity_text: Entity text
        
        Returns:
            Normalized value
        """
        # Basic cleanup
        normalized = entity_text.strip()
        
        # Strip surrounding quotes
        if (normalized.startswith('"') and normalized.endswith('"')) or \
           (normalized.startswith("'") and normalized.endswith("'")):
            normalized = normalized[1:-1].strip()
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Type-specific handling
        # if entity_type == "Vulnerability":
        #     # Normalize CVE to uppercase
        #     normalized = normalized.upper()
        #     # Ensure the format is CVE-YYYY-NNNN
        #     cve_match = re.search(r'CVE-(\d{4})-(\d+)', normalized, re.IGNORECASE)
        #     if cve_match:
        #         year = cve_match.group(1)
        #         num = cve_match.group(2)
        #         normalized = f"CVE-{year}-{num}"
        if entity_type in ["Organization", "Product"]:
            # Keep the original casing as entity_text, but use lowercase for normalized_value (easier merging)
            normalized = normalized.lower()
        else:
            # All other types are lowercased
            normalized = normalized.lower()
        
        return normalized
    
    @staticmethod
    def _merge_group(
        group: List[Dict[str, Any]],
        sentences: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Merge the entities within one group
        
        Args:
            group: List of entities sharing the same (entity_type, normalized_value)
            sentences: List of sentences
        
        Returns:
            Merged entity dict
        """
        if not group:
            return None
        
        # Merge all sentence_indices
        all_indices = set()
        all_texts = set()
        
        for entity in group:
            all_indices.update(entity["sentence_indices"])
            all_texts.add(entity["entity_text"])
        
        # Sort
        sorted_indices = sorted(all_indices)
        
        # Pick the most common entity_text as the main text (or the first one)
        main_text = group[0]["entity_text"]
        if len(all_texts) > 1:
            # If there are several different text forms, pick the longest (usually the most complete)
            main_text = max(all_texts, key=len)
        
        # Back-fill span_sentences
        span_sentences = [
            sentences[idx] if 0 <= idx < len(sentences) else ""
            for idx in sorted_indices
        ]
        
        # Build aliases (the other distinct text forms)
        aliases = sorted([text for text in all_texts if text != main_text])
        
        return {
            "entity_type": group[0]["entity_type"],
            "entity_text": main_text,
            "normalized_value": EntityMerger._normalize_value(
                group[0]["entity_type"],
                main_text
            ),
            "sentence_indices": sorted_indices,
            "span_sentences": span_sentences,
            "aliases": aliases if aliases else []
        }


class EntityPrecisionVerifier:
    """Entity precision checker (after merge)"""

    ALLOWED_TYPES = {
        "ThreatActor",
        "Campaign",
        "MalwareFamily",
        "Tool",
        # "Vulnerability",
        "Organization",
        "Product",
        "Location",
        "Sector",
    }

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_retries: int = 1,
        max_span_sentences: int = 3
    ):
        """
        Initialize the precision checker

        Args:
            llm_provider: LLM provider instance
            max_retries: Maximum number of retries on JSON parse errors
            max_span_sentences: Maximum number of context sentences passed to the LLM
        """
        self.llm_provider = llm_provider
        self.max_retries = max_retries
        self.max_span_sentences = max_span_sentences

        template_path = (
            Path(__file__).parent.parent / "prompts" / "highlevel_entity_precision_check.jinja"
        )
        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = Template(f.read())

    def verify_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the precision check on each merged entity

        Args:
            entities: List of merged entities

        Returns:
            List of entities that passed the check
        """
        verified_entities = []

        for idx, entity in enumerate(entities, 1):
            keep = self._verify_single_entity(entity)
            if keep:
                verified_entities.append(entity)
            else:
                print(
                    f"  Filtered entity [{idx}/{len(entities)}]: "
                    f"{entity.get('entity_type', '')} | {entity.get('entity_text', '')}"
                )

        return verified_entities

    def _verify_single_entity(self, entity: Dict[str, Any]) -> bool:
        """Check whether a single entity should be kept"""
        entity_text = str(entity.get("entity_text", "")).strip()
        entity_type = str(entity.get("entity_type", "")).strip()
        if not entity_text:
            return False
        if entity_type not in self.ALLOWED_TYPES:
            return False

        span_sentences = entity.get("span_sentences", [])
        if not isinstance(span_sentences, list):
            span_sentences = []
        span_sentences = [str(s) for s in span_sentences[:self.max_span_sentences]]

        prompt = self._build_prompt(entity, span_sentences)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response_text = self.llm_provider.call(
                    prompt=prompt,
                    max_tokens=512,
                    cost_module="CTIHighLevelExtractor",
                    cost_stage="precision_check",
                    cost_attempt=attempt + 1,
                )
                keep = self._parse_verification_response(response_text)
                if keep is not None:
                    return keep

                last_error = ValueError("Response does not contain a valid keep field")
            except Exception as e:
                last_error = e

            if attempt < self.max_retries:
                prompt = self._build_prompt(entity, span_sentences, error_hint=str(last_error))

        # Conservative fallback when the verification call fails: do not block the main flow, keep by default
        print(f"  Precision check failed, keeping the entity: {last_error}")
        return True

    def _build_prompt(
        self,
        entity: Dict[str, Any],
        span_sentences: List[str],
        error_hint: Optional[str] = None
    ) -> str:
        """Build the precision check prompt"""
        prompt = self.prompt_template.render(
            entity_text=entity.get("entity_text", ""),
            entity_type=entity.get("entity_type", ""),
            span_sentences=span_sentences
        )

        if error_hint:
            prompt += (
                f"\n\nPrevious error: {error_hint}\n"
                "Return strict JSON only."
            )

        return prompt

    def _parse_verification_response(self, response_text: str) -> Optional[bool]:
        """Parse the precision check response; returns keep (True/False) or None"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group(0))
            keep = data.get("keep")
            if keep is None:
                keep = data.get("is_valid")

            if isinstance(keep, bool):
                return keep
            if isinstance(keep, str):
                normalized = keep.strip().lower()
                if normalized in {"true", "yes", "1"}:
                    return True
                if normalized in {"false", "no", "0"}:
                    return False
        except Exception:
            return None

        return None
