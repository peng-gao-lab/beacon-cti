#!/usr/bin/env python3
"""
Stage 4: TTP Verifier
Verifies candidate techniques, judging support based on the MITRE JSON contents and the evidence sentences
"""

import json
import re
from typing import List, Dict, Any, Optional
from .llm_provider import LLMProvider
from .behavior_extractor import AttackBehavior
from .ttp_proposer import TtpCandidate


class VerifiedTtp:
    """Verified TTP data structure"""
    
    def __init__(
        self,
        ttp_id: str,
        presence_confidence: float,
        best_span_sentences: List[str],
        best_span_range: tuple
    ):
        self.ttp_id = ttp_id
        self.presence_confidence = presence_confidence
        self.best_span_sentences = best_span_sentences
        self.best_span_range = best_span_range
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            "ttp_id": self.ttp_id,
            "presence_confidence": self.presence_confidence,
            "best_span_sentences": self.best_span_sentences,
            "best_span_range": self.best_span_range
        }


class VerifiedResult:
    """Verification result data structure"""
    
    def __init__(self):
        self.final_ttps: List[VerifiedTtp] = []
        self.per_ttp_detail: Dict[str, Dict[str, Any]] = {}
        self.span_sentences: Dict[str, List[str]] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            "final_ttps": [t.to_dict() for t in self.final_ttps],
            "per_ttp_detail": self.per_ttp_detail,
            "span_sentences": self.span_sentences
        }


class TtpVerifier:
    """TTP verifier"""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the verifier
        
        Args:
            llm_provider: LLM provider instance
        """
        self.llm_provider = llm_provider
    
    def load_mitre_json(self, mitre_json_path: str) -> Dict[str, Any]:
        """
        Load the MITRE JSON file
        
        Args:
            mitre_json_path: Path to the MITRE JSON file
        
        Returns:
            MITRE data dictionary
        """
        try:
            with open(mitre_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load MITRE JSON: {e}")
            return {}
    
    def get_ttp_contents(self, mitre_json: Dict[str, Any], ttp_id: str) -> Optional[str]:
        """
        Get the contents field of a TTP
        
        Args:
            mitre_json: MITRE JSON data
            ttp_id: TTP ID
        
        Returns:
            The contents string if present, otherwise None
        """
        ttp_data = mitre_json.get(ttp_id)
        if not ttp_data:
            return None
        
        contents = ttp_data.get("contents", "")
        if not contents:
            return None
        
        # contents may be a JSON string that needs parsing
        if isinstance(contents, str):
            try:
                contents_dict = json.loads(contents)
                # Extract the description field
                if isinstance(contents_dict, dict):
                    description = contents_dict.get("description", "")
                    name = contents_dict.get("name", "")
                    return f"{name}\n\n{description}" if name else description
            except:
                # If it is not JSON, return it as is
                return contents
        
        return contents
    
    def verify_ttps_with_mitre(
        self,
        behavior: AttackBehavior,
        mitre_json: Dict[str, Any],
        threshold: float = 0.71
    ) -> VerifiedResult:
        """
        Verify the candidate techniques of a behavior
        
        Args:
            behavior: AttackBehavior object
            mitre_json: MITRE JSON data
            threshold: Verifier confidence threshold (default 0.71)
        
        Returns:
            VerifiedResult object
        """
        result = VerifiedResult()
        
        if not behavior.ttp_candidates:
            return result
        
        # Prepare the contents of every candidate technique
        ttp_contents_map = {}
        invalid_ttps = []
        
        for candidate in behavior.ttp_candidates:
            # Handle dict-format candidates (converted in ctitagger.py)
            if isinstance(candidate, dict):
                ttp_id = candidate.get("ttp_id", "")
                propose_confidence = candidate.get("propose_confidence", 0.0)
            else:
                # Handle TtpCandidate object format
                ttp_id = candidate.ttp_id
                propose_confidence = candidate.propose_confidence
            
            if not ttp_id:
                continue
                
            contents = self.get_ttp_contents(mitre_json, ttp_id)
            
            if not contents:
                invalid_ttps.append(ttp_id)
                result.per_ttp_detail[ttp_id] = {
                    "propose_confidence": propose_confidence,
                    "presence_confidence": 0.0,
                    "status": "invalid",
                    "reason": "contents does not exist or is empty"
                }
            else:
                ttp_contents_map[ttp_id] = contents
        
        if not ttp_contents_map:
            return result
        
        # Build the LLM prompt to verify all candidates in one call
        evidence_sentences = "\n".join(behavior.best_span_sentences) if behavior.best_span_sentences else ""
        
        # Build the TTP info list
        ttp_info_list = []
        for candidate in behavior.ttp_candidates:
            # Handle dict-format candidates
            if isinstance(candidate, dict):
                ttp_id = candidate.get("ttp_id", "")
                propose_confidence = candidate.get("propose_confidence", 0.0)
            else:
                ttp_id = candidate.ttp_id
                propose_confidence = candidate.propose_confidence
            
            if ttp_id and ttp_id in ttp_contents_map:
                ttp_info_list.append({
                    "ttp_id": ttp_id,
                    "propose_confidence": propose_confidence,
                    "contents": ttp_contents_map[ttp_id]
                })
        
        system_prompt = """You are a cybersecurity expert. Your task is to determine whether each MITRE ATT&CK TTP is present/supported in the evidence based on the given attack behavior and evidence sentences.

Important constraints:
1. You must strictly base your judgment on the evidence sentences (best_span_sentences), and you are not allowed to make inferences based on common knowledge
2. If the key points described in the TTP's contents are not clearly reflected in the evidence sentences, the confidence must be low (close to 0)
3. You must output strict JSON format, no explanatory text allowed
4. Each TTP's presence_confidence must be a float between [0,1]
5. IMPORTANT: Some of the given TTPs may be misleading or not actually present in the evidence. You must carefully and accurately judge whether each TTP is truly supported by the evidence sentences, avoiding false positives.

Confidence calibration rules (required):
- 0.00-0.10: No supporting evidence, or evidence contradicts the TTP.
- 0.11-0.30: Very weak support (only vague/indirect lexical overlap, missing core TTP behavior).
- 0.31-0.50: Partial support (some relevant action is present, but key required elements are missing).
- 0.51-0.70: Moderate support (core behavior is plausible in evidence, but important ambiguity remains).
- 0.71-0.90: Strong support (most key TTP elements are explicitly supported by evidence).
- 0.90-1.00: Very strong support (clear, direct, and specific evidence of the TTP behavior).

Scoring discipline:
- Base confidence on evidence quality and completeness, not on prior likelihood.
- Do not copy propose_confidence as final confidence; use propose_confidence only as weak prior context.
- Prefer conservative scoring when evidence is ambiguous.
- Use two-decimal precision (e.g., 0.73).

Output format (strict JSON, no other text):
{
  "ttp_verifications": [
    {
      "ttp_id": "T1059",
      "presence_confidence": 0.75
    }
  ]
}"""

        user_prompt = f"""Based on the following attack behavior and evidence sentences, determine whether each TTP is supported in the evidence:

Attack Behavior: {behavior.behavior_text}

Evidence sentences (you must strictly base your judgment on these sentences):
{evidence_sentences}

TTPs to verify:
"""
        
        for i, ttp_info in enumerate(ttp_info_list, 1):
            # Limit the contents length to keep the prompt short
            contents_preview = ttp_info['contents'][:2000]
            if len(ttp_info['contents']) > 2000:
                contents_preview += "..."
            user_prompt += f"""
{i}. TTP ID: {ttp_info['ttp_id']}
   Propose Confidence: {ttp_info['propose_confidence']}
   Contents:
   {contents_preview}
"""
        
        user_prompt += "\nPlease output presence_confidence for each TTP, indicating the degree to which the TTP is supported in the evidence sentences."
        
        max_retries = 2
        verification_results = {}
        
        for attempt in range(max_retries):
            try:
                response = self.llm_provider.call(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=3000,
                    cost_module="CTITagger",
                    cost_stage="ttp_verification",
                    cost_attempt=attempt + 1,
                )
                
                # Clean the response
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.strip()
                
                # Extract the JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)
                
                # Parse the JSON
                result_data = json.loads(response)
                
                if "ttp_verifications" not in result_data:
                    raise ValueError("Missing ttp_verifications field")
                
                for item in result_data["ttp_verifications"]:
                    if "ttp_id" not in item or "presence_confidence" not in item:
                        continue
                    
                    ttp_id = item["ttp_id"].strip().upper()
                    confidence = float(item["presence_confidence"])
                    
                    # Validate the confidence range
                    if not (0.0 <= confidence <= 1.0):
                        confidence = max(0.0, min(1.0, confidence))
                    
                    verification_results[ttp_id] = confidence
                
                break  # Parsed successfully; exit the retry loop
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ JSON parsing failed, retrying... ({e})")
                    continue
                else:
                    print(f"❌ JSON parsing failed: {e}")
                    # Set default low confidence for all candidates
                    for ttp_info in ttp_info_list:
                        verification_results[ttp_info["ttp_id"]] = 0.0
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Error verifying TTPs, retrying... ({e})")
                    continue
                else:
                    print(f"❌ Failed to verify TTPs: {e}")
                    # Set default low confidence for all candidates
                    for ttp_info in ttp_info_list:
                        verification_results[ttp_info["ttp_id"]] = 0.0
        
        # Process the verification results
        for candidate in behavior.ttp_candidates:
            # Handle dict-format candidates
            if isinstance(candidate, dict):
                ttp_id = candidate.get("ttp_id", "")
                propose_confidence = candidate.get("propose_confidence", 0.0)
            else:
                ttp_id = candidate.ttp_id
                propose_confidence = candidate.propose_confidence
            
            if not ttp_id:
                continue
            
            # Skip invalid TTPs
            if ttp_id in invalid_ttps:
                continue
            
            presence_confidence = verification_results.get(ttp_id, 0.0)
            
            # Save the details
            result.per_ttp_detail[ttp_id] = {
                "propose_confidence": propose_confidence,
                "presence_confidence": presence_confidence,
                "status": "verified"
            }
            
            # Threshold filtering
            if presence_confidence >= threshold:
                verified_ttp = VerifiedTtp(
                    ttp_id=ttp_id,
                    presence_confidence=presence_confidence,
                    best_span_sentences=behavior.best_span_sentences,
                    best_span_range=behavior.best_span_range
                )
                result.final_ttps.append(verified_ttp)
                result.span_sentences[ttp_id] = behavior.best_span_sentences
        
        return result
