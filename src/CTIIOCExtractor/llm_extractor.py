#!/usr/bin/env python3
"""
LLM IOC Extractor
"""

import re
import json
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple

# Import LLM Provider
try:
    from CTITagger.core.llm_provider import LLMProvider
except ImportError:
    # If import fails, define basic interface
    LLMProvider = None

from .ioc_validator import IOCValidator


IOC_TYPE_ENUM = [
    "ip",
    "domain",
    "url",
    "email",
    "md5",
    "sha1",
    "sha256",
    "sha512",
    "cve",
    "file_path",
    "registry_key",
    "mutex",
    "aes_key",
]

IOC_TYPE_DEFINITION_LINES = """- IP addresses (IPv4/IPv6)
- Domains (Refer to the DNS domain, including obfuscated domains using [.], such as example[.]com, mol.com-ho[.]me)
- URLs (excluding source code reference links like GitHub, such as https://github.com/...)
- Email addresses
- File hashes (MD5, SHA1, SHA256, SHA512)
- CVE numbers
- File paths
- Registry keys
- Mutex names (such as mpk1, AB12CD34, etc.)
- AES keys (such as agkrhfpdbvhdhrkj and other encryption key strings)
- Other security-related indicators"""


class LLMExtractor:
    """LLM IOC Extractor"""
    
    def __init__(self, llm_provider: Optional[LLMProvider], model: str = "gpt-5-mini"):
        """
        Initialize LLM Extractor
        
        Args:
            llm_provider: LLM Provider instance
            model: LLM model name (default: gpt-5-mini)
        """
        self.llm_provider = llm_provider
        self.model = model
        self.validator = IOCValidator()
        # Observability only: a snapshot of parsed/validated candidates before
        # the second-pass LLM review.  The extraction and review paths do not
        # read this field.
        self.last_pre_verification_iocs: List[Dict[str, Any]] = []
    
    def extract(
        self,
        sentences: List[str],
        chunk_size: int = 50,
        overlap: int = 2,
        report_text: Optional[str] = None,
        record_pre_verification: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract IOCs using LLM
        
        Args:
            sentences: List of sentences
            chunk_size: Number of sentences per chunk
            overlap: Number of overlapping sentences between chunks
            report_text: Full report text (used for second-pass verification)
            record_pre_verification: Record a detached pre-review snapshot
        
        Returns:
            List of IOCs
        """
        self.last_pre_verification_iocs = []
        if not self.llm_provider:
            return []
        
        all_iocs = []
        
        # Process in chunks with sliding window
        for chunk_start in range(0, len(sentences), chunk_size - overlap):
            chunk_end = min(chunk_start + chunk_size, len(sentences))
            chunk_sentences = sentences[chunk_start:chunk_end]
            chunk_local_indices = list(range(len(chunk_sentences)))
            
            # Build prompt
            prompt = self._build_llm_prompt(chunk_sentences, chunk_local_indices)
            
            # Call LLM
            try:
                # For gpt-5 and gpt-5-mini, llm_provider automatically handles max_completion_tokens and doesn't pass temperature
                # For other models, pass temperature
                kwargs = {"max_tokens": 16000}
                if "gpt-5" not in self.model.lower():
                    kwargs["temperature"] = 0.3
                
                response = self.llm_provider.call(
                    prompt=prompt,
                    system_prompt=self._build_system_prompt(),
                    cost_module="CTIIOCExtractor",
                    cost_stage="chunk_extraction",
                    cost_attempt=1,
                    **kwargs
                )
                
                # Parse response
                chunk_iocs = self._parse_llm_response(response, chunk_sentences, chunk_start)
                all_iocs.extend(chunk_iocs)
            
            except Exception as e:
                print(f"Warning: Error extracting IOCs from chunk [{chunk_start}:{chunk_end}]: {e}")
                continue

        # Record a detached snapshot immediately before second-pass review.
        # This is used only for offline ablation analysis and is never consumed
        # by the extraction pipeline itself.
        if record_pre_verification:
            self.last_pre_verification_iocs = deepcopy(all_iocs)

        # Second-pass review: verify extracted IOC items against full report text.
        if report_text and all_iocs:
            print("🧪 Starting second-pass IOC verification...")
            all_iocs = self._second_pass_verify_and_fix(all_iocs, report_text)
            print(f"✅ Second-pass verification completed, {len(all_iocs)} IOCs kept")
        
        return all_iocs
    
    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        return f"""You are a professional cybersecurity IOC (Indicators of Compromise) extraction expert.
Your task is to extract all IOCs from the given text, including:
{IOC_TYPE_DEFINITION_LINES}

Important requirements:
1. Only extract IOCs that actually appear in the original text, do not fabricate or modify
2. Must preserve the exact string from the original text (including obfuscation formats like [.])
3. Each IOC must be listed separately
4. Must indicate which sentence each IOC appears in (using sentence indices)
5. Sentence indices must only fall within the current chunk range (starting from 0)
6. If the same IOC appears in multiple sentences, list all sentence indices
7. For obfuscated domains (such as mol.com-ho[.]me), must extract the complete domain, not just a portion
8. **Do not extract GitHub and other source code reference links** (such as https://github.com/rapid7/...), these are source code references, not malicious infrastructure
9. Must extract Mutex names and AES keys and other encryption-related indicators

The output format must be valid JSON conforming to the following schema:
{{
  "iocs": [
    {{
      "ioc_type": "ip|domain|url|email|md5|sha1|sha256|sha512|cve|file_path|registry_key|mutex|aes_key",
      "value": "exact string from original text",
      "sentence_indices": [0, 1, 2]
    }}
  ]
}}"""
    
    def _build_llm_prompt(self, sentences: List[str], local_indices: List[int]) -> str:
        """Build LLM prompt"""
        sentences_text = "\n".join([f"[{i}] {s}" for i, s in zip(local_indices, sentences)])
        
        prompt = f"""Please extract all IOCs (Indicators of Compromise) from the following text.

Text (sentence indices start from 0, only use indices within this chunk):
{sentences_text}

Please extract all IOCs and indicate which sentences each IOC appears in (using sentence indices).
Sentence indices must only fall within this chunk range (from 0 to {len(sentences)-1}).

Special attention:
- For obfuscated domains (such as mol.com-ho[.]me, ntg-sa[.]com), must extract the complete domain
- Preserve the exact format from the original text, including obfuscation characters like [.]

Output JSON format:
{{
  "iocs": [
    {{
      "ioc_type": "ip",
      "value": "192.168.1.1",
      "sentence_indices": [0, 2]
    }},
    {{
      "ioc_type": "domain",
      "value": "mol.com-ho[.]me",
      "sentence_indices": [1]
    }}
  ]
}}"""
        
        return prompt
    
    def _parse_llm_response(self, response: str, chunk_sentences: List[str], chunk_start: int) -> List[Dict[str, Any]]:
        """
        Parse LLM response
        
        Args:
            response: LLM response text
            chunk_sentences: List of sentences in the chunk
            chunk_start: Starting index of the chunk in the global sentence list
        
        Returns:
            List of IOCs (converted to global indices)
        """
        iocs = []
        
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                if "iocs" in data:
                    for ioc_data in data["iocs"]:
                        ioc_type = ioc_data.get("ioc_type", "")
                        value = ioc_data.get("value", "")
                        local_indices = ioc_data.get("sentence_indices", [])
                        
                        if not value or not ioc_type:
                            continue
                        
                        # Validate that IOC exists in sentences
                        validated_indices = []
                        for local_idx in local_indices:
                            # Check if local index is valid
                            if 0 <= local_idx < len(chunk_sentences):
                                sentence = chunk_sentences[local_idx]
                                global_idx = chunk_start + local_idx
                                
                                # Exact substring check
                                if value in sentence:
                                    validated_indices.append(global_idx)
                                else:
                                    # Token overlap fallback
                                    if self._check_token_overlap(value, sentence):
                                        validated_indices.append(global_idx)
                        
                        if validated_indices:
                            # Filter obvious false positives
                            if self._should_filter_ioc(ioc_type, value):
                                continue
                            
                            normalized_value = self.validator.normalize_ioc_value(ioc_type, value)
                            
                            iocs.append({
                                "ioc_type": ioc_type,
                                "value": value,  # raw_value, preserve original form
                                "normalized_value": normalized_value,  # normalized value for deduplication
                                "sentence_indices": validated_indices,
                                "source": "llm"
                            })
        
        except Exception as e:
            print(f"Warning: Error parsing LLM response: {e}")
            print(f"Response content: {response[:500]}")
        
        return iocs
    
    def _check_token_overlap(self, value: str, sentence: str) -> bool:
        """
        Check if value exists in sentence using token overlap
        
        Args:
            value: IOC value
            sentence: Sentence text
        
        Returns:
            True if there is sufficient token overlap
        """
        # Simple tokenization
        value_tokens = set(re.findall(r'\b\w+\b', value.lower()))
        sentence_tokens = set(re.findall(r'\b\w+\b', sentence.lower()))
        
        if not value_tokens:
            return False
        
        # Calculate overlap ratio
        overlap = len(value_tokens & sentence_tokens)
        overlap_ratio = overlap / len(value_tokens)
        
        # If more than 80% of tokens overlap, consider it a match
        return overlap_ratio >= 0.8
    
    def _should_filter_ioc(self, ioc_type: str, value: str) -> bool:
        """
        Determine if an IOC should be filtered (obvious false positives)
        
        Args:
            ioc_type: IOC type
            value: IOC value
        
        Returns:
            True if should be filtered
        """
        # Filter source code reference links like GitHub
        if ioc_type == "url":
            value_lower = value.lower()
            # GitHub links are usually source code references, not malicious infrastructure
            if "github.com" in value_lower:
                return True
            # Other common source code hosting platforms
            if any(domain in value_lower for domain in ["gitlab.com", "bitbucket.org", "sourceforge.net"]):
                return True
        
        return False

    def _build_second_pass_system_prompt(self) -> str:
        """Build system prompt for second-pass IOC review."""
        return """You are a cybersecurity IOC quality review expert.
You must review extracted IOCs against the full report text.

Output strict JSON object only.
Each key must be a tuple-like string in this exact format:
"(ioc_type, value)"

Each value must be:
- "refuse" (if this IOC should not be extracted), or
- the corrected ioc_type string (if type is wrong).

Do not output correct IOCs.
Do not output explanation text."""

    def _build_second_pass_prompt(self, report_text: str, compact_iocs_json: str) -> str:
        """Build user prompt for second-pass IOC review."""
        valid_types = "|".join(IOC_TYPE_ENUM)
        return f"""Review the extracted IOC list and report only problematic IOCs.

IOC type definitions (copied from extraction prompt):
{IOC_TYPE_DEFINITION_LINES}

Valid ioc_type enum:
{valid_types}

Full report text:
{report_text}

Extracted IOC list (compact JSON):
{compact_iocs_json}

Rules:
1. If an IOC should not be extracted from the report, output value as "refuse".
2. If IOC exists but type is wrong, output value as the correct ioc_type.
3. If IOC is correct, do not output it.
4. Use exact ioc_type and value strings from the IOC list when writing keys.
5. Key format must be "(ioc_type, value)".
6. Output JSON object only, no markdown and no extra fields.

Example:
{{
  "(url, https://github.com/rapid7/repo)": "refuse",
  "(domain, 10.10.10.10)": "ip"
}}"""

    def _parse_tuple_key(self, key: str) -> Optional[Tuple[str, str]]:
        """Parse tuple-like JSON key in form '(ioc_type, value)'."""
        raw = key.strip()
        if not (raw.startswith("(") and raw.endswith(")")):
            return None
        inner = raw[1:-1]
        parts = inner.split(",", 1)
        if len(parts) != 2:
            return None
        ioc_type = parts[0].strip().strip("\"'")
        value = parts[1].strip().strip("\"'")
        if not ioc_type or not value:
            return None
        return ioc_type, value

    def _parse_second_pass_response(self, response: str) -> Dict[Tuple[str, str], str]:
        """Parse second-pass review response."""
        decisions: Dict[Tuple[str, str], str] = {}
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return decisions

            data = json.loads(json_match.group(0))
            if not isinstance(data, dict):
                return decisions

            issues = data.get("issues", data)
            if not isinstance(issues, dict):
                return decisions

            allowed_types = set(IOC_TYPE_ENUM)
            for raw_key, raw_value in issues.items():
                if not isinstance(raw_key, str):
                    continue
                if not isinstance(raw_value, str):
                    continue

                parsed_key = self._parse_tuple_key(raw_key)
                if not parsed_key:
                    continue

                decision = raw_value.strip()
                if not decision:
                    continue

                if decision.lower() == "refuse":
                    decisions[parsed_key] = "refuse"
                    continue

                fixed_type = decision.lower()
                if fixed_type in allowed_types:
                    decisions[parsed_key] = fixed_type

        except Exception as e:
            print(f"Warning: Error parsing second-pass IOC review response: {e}")

        return decisions

    def _apply_second_pass_decisions(
        self,
        iocs: List[Dict[str, Any]],
        decisions: Dict[Tuple[str, str], str]
    ) -> List[Dict[str, Any]]:
        """Apply second-pass review decisions to IOC list."""
        if not decisions:
            return iocs

        corrected: List[Dict[str, Any]] = []
        for ioc in iocs:
            ioc_type = str(ioc.get("ioc_type", ""))
            value = str(ioc.get("value", ""))

            decision = decisions.get((ioc_type, value))
            if decision is None:
                decision = decisions.get((ioc_type.lower(), value))

            if decision is None:
                corrected.append(ioc)
                continue

            if decision == "refuse":
                continue

            updated = dict(ioc)
            updated["ioc_type"] = decision
            updated["normalized_value"] = self.validator.normalize_ioc_value(decision, value)

            if self._should_filter_ioc(updated["ioc_type"], value):
                continue

            corrected.append(updated)

        return corrected

    def _second_pass_verify_and_fix(
        self,
        iocs: List[Dict[str, Any]],
        report_text: str
    ) -> List[Dict[str, Any]]:
        """Run second-pass LLM review and apply corrections."""
        if not self.llm_provider or not iocs:
            return iocs

        compact_iocs: List[Dict[str, str]] = []
        seen = set()
        for ioc in iocs:
            ioc_type = str(ioc.get("ioc_type", ""))
            value = str(ioc.get("value", ""))
            if not ioc_type or not value:
                continue
            key = (ioc_type, value)
            if key in seen:
                continue
            seen.add(key)
            compact_iocs.append({"ioc_type": ioc_type, "value": value})

        if not compact_iocs:
            return iocs

        compact_json = json.dumps(compact_iocs, ensure_ascii=False, separators=(",", ":"))
        prompt = self._build_second_pass_prompt(report_text, compact_json)

        try:
            kwargs = {"max_tokens": 16000}
            if "gpt-5" not in self.model.lower():
                kwargs["temperature"] = 0.0

            response = self.llm_provider.call(
                prompt=prompt,
                system_prompt=self._build_second_pass_system_prompt(),
                cost_module="CTIIOCExtractor",
                cost_stage="second_pass_verification",
                cost_attempt=1,
                **kwargs
            )
        except Exception as e:
            print(f"Warning: Error in second-pass IOC verification: {e}")
            return iocs

        decisions = self._parse_second_pass_response(response)
        if not decisions:
            return iocs

        return self._apply_second_pass_decisions(iocs, decisions)
