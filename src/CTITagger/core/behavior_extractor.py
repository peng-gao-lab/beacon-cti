#!/usr/bin/env python3
"""
Stage 1: Attack Behavior Extractor
Uses an LLM to precisely extract attack behaviors from a CTI report
"""

import json
import re
from typing import List, Dict, Any
from .llm_provider import LLMProvider


class AttackBehavior:
    """Attack behavior data structure"""
    
    def __init__(self, behavior_text: str):
        self.behavior_text = behavior_text
        self.best_span_sentences = []  # Filled in Stage 2
        self.best_span_range = None  # (start_idx, end_idx)
        self.best_score = 0.0
        self.debug_info = {}
        self.ttp_candidates = []  # Filled in Stage 3
        self.final_ttps = []  # Filled in Stage 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary"""
        return {
            "behavior_text": self.behavior_text,
            "best_span_sentences": self.best_span_sentences,
            "best_span_range": self.best_span_range,
            "best_score": self.best_score,
            "debug_info": self.debug_info,
            "ttp_candidates": self.ttp_candidates,
            "final_ttps": self.final_ttps
        }


class AttackBehaviorExtractor:
    """Attack behavior extractor"""
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the extractor
        
        Args:
            llm_provider: LLM provider instance
        """
        self.llm_provider = llm_provider
    
    def extract(self, raw_text: str) -> List[AttackBehavior]:
        """
        Extract attack behaviors from the raw text
        
        Args:
            raw_text: Raw text of the CTI report
        
        Returns:
            List of AttackBehavior objects
        """
        system_prompt = """You are a cybersecurity expert. Your task is to accurately extract attack behaviors from CTI reports.

Requirements:
1. Each extracted attack behavior must be a very brief sentence, closely matching the original wording
2. Reuse technical terms and verb phrases from the original text as much as possible
3. If the original text contains specific strings (domains, hashes, paths, command lines), prioritize preserving them in behavior_text
4. Each behavior corresponds to one or more sentences in the original text
5. You must output strict JSON format, no explanatory text allowed, no extra fields allowed

Output format (strict JSON, no other text):
{
  "behaviors": [
    {
      "behavior_text": "A brief sentence describing the attack behavior"
    }
  ]
}"""

        user_prompt = f"""Please extract all attack behaviors from the following CTI report:

{raw_text}

Please output strict JSON format according to the requirements, do not add any explanatory text."""

        try:
            response = self.llm_provider.call(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=8000,
                cost_module="CTITagger",
                cost_stage="behavior_extraction",
                cost_attempt=1,
            )
            
            # Clean the response: strip any markdown code fence markers
            response = response.strip()
            if response.startswith("```"):
                # Strip the markdown code fence
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
            response = response.strip()
            
            # Try to extract the JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            # Parse the JSON
            result = json.loads(response)
            behaviors = []
            
            if "behaviors" in result:
                for item in result["behaviors"]:
                    if "behavior_text" in item:
                        behavior = AttackBehavior(item["behavior_text"])
                        behaviors.append(behavior)
            
            return behaviors
            
        except json.JSONDecodeError as e:
            # JSON parsing failed, try retry once
            print(f"⚠️ JSON parsing failed, attempting retry: {e}")
            try:
                # More lenient extraction
                response_clean = re.sub(r'[^\{\}\[\]\w\s:,"\.\-\']', '', response)
                result = json.loads(response_clean)
                behaviors = []
                if "behaviors" in result:
                    for item in result["behaviors"]:
                        if "behavior_text" in item:
                            behavior = AttackBehavior(item["behavior_text"])
                            behaviors.append(behavior)
                return behaviors
            except:
                print(f"❌ Still unable to parse JSON after retry, returning empty list")
                return []
        except Exception as e:
            print(f"❌ Error extracting attack behavior: {e}")
            return []
