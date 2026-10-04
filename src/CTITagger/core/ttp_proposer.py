#!/usr/bin/env python3
"""
Stage 3: TTP Candidate Proposer
Proposes N=5 MITRE ATT&CK candidate techniques for each behavior
"""

import json
import re
from typing import List, Dict, Any, Set
from .llm_provider import LLMProvider
from .behavior_extractor import AttackBehavior


class TtpCandidate:
    """Candidate technique data structure"""
    
    def __init__(self, ttp_id: str, propose_confidence: float):
        self.ttp_id = ttp_id
        self.propose_confidence = propose_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            "ttp_id": self.ttp_id,
            "propose_confidence": self.propose_confidence
        }


class TtpProposer:
    """Candidate technique proposer"""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the proposer
        
        Args:
            llm_provider: LLM provider instance
        """
        self.llm_provider = llm_provider
    
    def load_allowed_ids(self, mitre_json_path: str) -> Set[str]:
        """
        Load all available TTP IDs from the MITRE JSON (parent techniques only, no sub-techniques)
        
        Args:
            mitre_json_path: Path to the MITRE JSON file
        
        Returns:
            Set of TTP IDs (parent techniques only, e.g. T1059, not T1059.003)
        """
        try:
            with open(mitre_json_path, 'r', encoding='utf-8') as f:
                mitre_data = json.load(f)
            
            allowed_ids = set()
            for key, value in mitre_data.items():
                if isinstance(value, dict) and "mitre_id" in value:
                    mitre_id = value["mitre_id"]
                    # Only add parent techniques (IDs without ".")
                    if mitre_id.startswith("T") and "." not in mitre_id:
                        allowed_ids.add(mitre_id)
                elif key.startswith("T") and "." not in key:  # Only add parent techniques, e.g. T1059, not T1059.003
                    allowed_ids.add(key)
            
            return allowed_ids
        except Exception as e:
            print(f"❌ Failed to load MITRE JSON: {e}")
            return set()
    
    def validate_ttp_id(self, ttp_id: str, allowed_ids: Set[str]) -> bool:
        """
        Check whether a TTP ID is in allowed_ids (parent techniques only, no sub-techniques)
        
        Args:
            ttp_id: TTP ID to validate
            allowed_ids: Set of allowed TTP IDs (parent techniques only)
        
        Returns:
            Whether the ID is valid
        """
        # Normalize the format
        ttp_id = ttp_id.strip().upper()
        if not ttp_id.startswith("T"):
            return False
        
        # Only parent techniques are allowed (no ".")
        if "." in ttp_id:
            return False
        
        return ttp_id in allowed_ids
    
    def propose_ttp_candidates(
        self,
        behavior: AttackBehavior,
        allowed_ids: Set[str],
        n: int = 5
    ) -> List[TtpCandidate]:
        """
        Propose N candidate techniques for a behavior
        
        Args:
            behavior: AttackBehavior object
            allowed_ids: Set of allowed TTP IDs
            n: Number of candidates (default 5)
        
        Returns:
            List of TtpCandidate objects
        """
        system_prompt = """You are a cybersecurity expert. Your task is to recommend the most relevant TTPs (Techniques only, NOT sub-techniques) from the MITRE ATT&CK framework based on the given attack behavior and evidence sentences.

Requirements:
1. You must output strict JSON format, no explanatory text allowed
2. You must output exactly N different TTP candidates (N=5)
3. Each candidate must contain ttp_id (format like "T1059", NOT "T1059.003") and propose_confidence (a float between 0-1)
4. IMPORTANT: You must ONLY use parent techniques (e.g., "T1059"), NOT sub-techniques (e.g., "T1059.003"). Sub-techniques are NOT allowed.
5. You should select TTP IDs that are valid MITRE ATT&CK technique IDs (parent techniques only)

Output format (strict JSON, no other text):
{
  "ttp_candidates": [
    {
      "ttp_id": "T1059",
      "propose_confidence": 0.85
    },
    {
      "ttp_id": "T1071",
      "propose_confidence": 0.75
    }
  ]
}"""

        evidence_sentences = "\n".join(behavior.best_span_sentences) if behavior.best_span_sentences else ""
        
        # Limit the length of evidence_sentences to keep the prompt short
        if len(evidence_sentences) > 500:
            evidence_sentences = evidence_sentences[:500] + "..."
        
        # Simplify the prompt structure to keep it concise
        user_prompt = f"""Recommend 5 MITRE ATT&CK TTP IDs.

Behavior: {behavior.behavior_text}

Evidence: {evidence_sentences}

Output JSON (only parent techniques, NOT sub-techniques):
{{
  "ttp_candidates": [
    {{"ttp_id": "T1059", "propose_confidence": 0.85}},
    {{"ttp_id": "T1071", "propose_confidence": 0.75}}
  ]
}}"""

        max_retries = 4  # Increased retry count, since gpt-5 may occasionally return an empty response
        for attempt in range(max_retries):
            try:
                response = self.llm_provider.call(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=4000,
                    cost_module="CTITagger",
                    cost_stage="ttp_proposal",
                    cost_attempt=attempt + 1,
                )
                
                # Check whether the response is None
                if response is None:
                    raise ValueError("LLM returned None response")
                
                # Clean the response
                response = response.strip()
                
                # Check whether the response is empty (before cleanup)
                if not response:
                    raise ValueError("LLM returned empty response")
                
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.strip()
                
                # Check again whether the response is empty (after cleanup)
                if not response:
                    raise ValueError("LLM returned empty response (after cleanup)")
                
                # Extract the JSON - try several patterns
                json_match = None
                # First try to match a complete JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                # If that fails, try a more lenient match
                if not json_match:
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                
                if json_match:
                    response = json_match.group(0)
                else:
                    raise ValueError("Unable to extract JSON from response")
                
                # Parse the JSON
                result = json.loads(response)
                
                if "ttp_candidates" not in result:
                    raise ValueError("Missing ttp_candidates field")
                
                candidates = []
                invalid_ids = []  # Track invalid TTP IDs
                seen_ids = set()
                
                for item in result["ttp_candidates"]:
                    if "ttp_id" not in item or "propose_confidence" not in item:
                        continue
                    
                    ttp_id = item["ttp_id"].strip().upper()
                    confidence = float(item["propose_confidence"])
                    
                    # Filter out sub-techniques (IDs containing ".")
                    if "." in ttp_id:
                        invalid_ids.append(ttp_id)
                        continue
                    
                    # Validate the ID (must be a parent technique and in allowed_ids)
                    if not self.validate_ttp_id(ttp_id, allowed_ids):
                        invalid_ids.append(ttp_id)
                        continue
                    
                    # Deduplicate
                    if ttp_id in seen_ids:
                        continue
                    seen_ids.add(ttp_id)
                    
                    # Validate the confidence range
                    if not (0.0 <= confidence <= 1.0):
                        confidence = max(0.0, min(1.0, confidence))
                    
                    candidates.append(TtpCandidate(ttp_id, confidence))
                
                # Retry if there are invalid IDs or fewer than N candidates
                if invalid_ids or len(candidates) < n:
                    if attempt < max_retries - 1:
                        # Build the retry prompt
                        retry_messages = []
                        
                        if invalid_ids:
                            retry_messages.append(f"The following TTP IDs are invalid (not in the allowed list or are sub-techniques): {', '.join(invalid_ids)}")
                            retry_messages.append("Please replace them with valid parent technique IDs from MITRE ATT&CK framework.")
                        
                        if len(candidates) < n:
                            missing = n - len(candidates)
                            retry_messages.append(f"Need {missing} more valid TTP candidates to complete to {n} total candidates.")
                        
                        retry_prompt = f"""Previous output had some issues:

{chr(10).join(retry_messages)}

Valid candidates generated so far:
{json.dumps([c.to_dict() for c in candidates], ensure_ascii=False, indent=2)}

Please regenerate all {n} TTP candidates, ensuring:
1. All ttp_ids are valid MITRE ATT&CK parent technique IDs (format: T####, NOT T####.###)
2. All candidates are unique
3. Output complete JSON format containing all {n} candidates

Original request:
{user_prompt}"""
                        
                        user_prompt = retry_prompt
                        continue
                    else:
                        # Last attempt; return the candidates collected so far
                        if invalid_ids:
                            print(f"⚠️ Found {len(invalid_ids)} invalid TTP IDs: {', '.join(invalid_ids)}")
                        print(f"⚠️ Only generated {len(candidates)} valid candidates (need {n})")
                        return candidates[:n]
                
                # Return the first N
                return candidates[:n]
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ JSON parsing failed, retrying... ({e})")
                    # On retry, simplify the prompt or add more explicit instructions
                    # #region Simplified prompt (temporarily commented out)
                    # if attempt == 1:
                    #     user_prompt = f"""Recommend 5 most relevant TTP IDs from MITRE ATT&CK framework.
                    #
                    # Attack Behavior: {behavior.behavior_text}
                    #
                    # Evidence sentences:
                    # {evidence_sentences[:500] if len(evidence_sentences) > 500 else evidence_sentences}
                    #
                    # Available TTP IDs (first 50):
                    # {", ".join(sorted(list(allowed_ids))[:50])}
                    #
                    # Please output strict JSON format:
                    # {{
                    #   "ttp_candidates": [
                    #     {{"ttp_id": "T1059", "propose_confidence": 0.85}},
                    #     {{"ttp_id": "T1059.003", "propose_confidence": 0.75}}
                    #   ]
                    # }}"""
                    # #endregion
                    continue
                else:
                    print(f"❌ JSON parsing failed: {e}")
                    return []
            except ValueError as e:
                # Handle empty-response errors
                if "empty response" in str(e).lower() or "none response" in str(e).lower() or "empty string" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️ {str(e)}, retrying...")
                        # Progressively simplify the prompt on retry
                        # #region Simplified prompt (temporarily commented out)
                        # if attempt == 1:
                        #     # First retry: simplify substantially
                        #     user_prompt = f"""Recommend 5 MITRE ATT&CK TTP IDs.
                        #
                        # Behavior: {behavior.behavior_text[:100]}
                        #
                        # Evidence: {evidence_sentences[:200] if len(evidence_sentences) > 200 else evidence_sentences}
                        #
                        # TTP IDs: {", ".join(sorted(list(allowed_ids))[:30])}
                        #
                        # JSON: {{"ttp_candidates": [{{"ttp_id": "T1059", "propose_confidence": 0.8}}]}}"""
                        # elif attempt == 2:
                        #     # Second retry: minimal prompt
                        #     user_prompt = f"""TTP IDs: {", ".join(sorted(list(allowed_ids))[:20])}
                        #
                        # Behavior: {behavior.behavior_text[:80]}
                        #
                        # Output: {{"ttp_candidates": [{{"ttp_id": "T1059", "propose_confidence": 0.8}}]}}"""
                        # #endregion
                        continue
                    else:
                        print(f"❌ {str(e)} (retried {max_retries-1} times)")
                        print(f"   Tip: gpt-5 model may return empty responses for some prompts, consider using gpt-4o or gpt-5-mini")
                        return []
                else:
                    raise
            except RuntimeError as e:
                # Handle RuntimeError (empty-response errors from llm_provider)
                if "empty content" in str(e).lower() or "empty string" in str(e).lower():
                    if attempt < max_retries - 1:
                        print(f"⚠️ {str(e)}, retrying...")
                        # Progressively simplify the prompt on retry
                        # #region Simplified prompt (temporarily commented out)
                        # if attempt == 1:
                        #     user_prompt = f"""Recommend 5 MITRE ATT&CK TTP IDs.
                        #
                        # Behavior: {behavior.behavior_text[:100]}
                        #
                        # Evidence: {evidence_sentences[:200] if len(evidence_sentences) > 200 else evidence_sentences}
                        #
                        # TTP IDs: {", ".join(sorted(list(allowed_ids))[:30])}
                        #
                        # JSON: {{"ttp_candidates": [{{"ttp_id": "T1059", "propose_confidence": 0.8}}]}}"""
                        # elif attempt == 2:
                        #     user_prompt = f"""TTP IDs: {", ".join(sorted(list(allowed_ids))[:20])}
                        #
                        # Behavior: {behavior.behavior_text[:80]}
                        #
                        # Output: {{"ttp_candidates": [{{"ttp_id": "T1059", "propose_confidence": 0.8}}]}}"""
                        # #endregion
                        continue
                    else:
                        print(f"❌ {str(e)} (retried {max_retries-1} times)")
                        print(f"   Tip: gpt-5 model may return empty responses for some prompts, consider using gpt-4o or gpt-5-mini")
                        return []
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Error generating TTP candidates, retrying... ({e})")
                    continue
                else:
                    print(f"❌ Failed to generate TTP candidates: {e}")
                    return []
        
        return []
