#!/usr/bin/env python3
"""
CTITagger Pipeline - the complete CTI report processing flow
Integrates four stages: behavior extraction, span location, TTP candidate proposal, TTP verification
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .llm_provider import LLMProvider, create_llm_provider
from .behavior_extractor import AttackBehavior, AttackBehaviorExtractor
from .span_locator import SpanLocator, SpanLocatorConfig
from .ttp_proposer import TtpProposer
from .ttp_verifier import TtpVerifier, VerifiedResult


class CTITagger:
    """CTITagger main pipeline class"""
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        provider_type: str = "openai",
        model: str = "gpt-4o",
        span_config: Optional[SpanLocatorConfig] = None,
        ttp_threshold: float = 0.71
    ):
        """
        Initialize CTITagger
        
        Args:
            llm_provider: LLM provider instance (if given, provider_type and model are ignored)
            provider_type: LLM provider type (default "openai")
            model: Model name (default "gpt-4o"; gpt-5, gpt-5-mini, etc. are supported)
            span_config: Span Locator configuration (if None, the default configuration is used)
            ttp_threshold: TTP verification threshold (default 0.71)
        """
        # Initialize the LLM provider
        if llm_provider is None:
            self.llm_provider = create_llm_provider(provider_type, model)
        else:
            self.llm_provider = llm_provider
        
        # Initialize each module
        self.behavior_extractor = AttackBehaviorExtractor(self.llm_provider)
        self.span_locator = SpanLocator(span_config)
        self.ttp_proposer = TtpProposer(self.llm_provider)
        self.ttp_verifier = TtpVerifier(self.llm_provider)
        
        self.ttp_threshold = ttp_threshold
    
    def process(
        self,
        report_text: str,
        mitre_json_path: str
    ) -> List[Dict[str, Any]]:
        """
        Process a CTI report by running the full pipeline
        
        Args:
            report_text: Text content of the CTI report
            mitre_json_path: Path to the MITRE ATT&CK JSON file
        
        Returns:
            List of results, one per behavior with its final TTPs
        """
        results = []
        
        # Stage 1: extract attack behaviors
        print("🔍 Stage 1: extracting attack behaviors...")
        behaviors = self.behavior_extractor.extract(report_text)
        print(f"✅ Extracted {len(behaviors)} attack behaviors")
        
        if not behaviors:
            print("⚠️ No attack behavior was extracted")
            return results
        
        # Stage 2: locate the span of each behavior
        print("\n📍 Stage 2: locating behavior spans...")
        for i, behavior in enumerate(behaviors, 1):
            print(f"  Processing behavior {i}/{len(behaviors)}: {behavior.behavior_text[:50]}...")
            
            span_sentences, span_range, score, debug_info = self.span_locator.locate_span(
                behavior.behavior_text,
                report_text
            )
            
            behavior.best_span_sentences = span_sentences
            behavior.best_span_range = span_range
            behavior.best_score = score
            behavior.debug_info = debug_info
            
            print(f"    ✅ Located span: {span_range}, score={score:.3f}")
        
        # Load the MITRE JSON and allowed_ids
        print("\n📚 Loading MITRE JSON...")
        mitre_json = self.ttp_verifier.load_mitre_json(mitre_json_path)
        allowed_ids = self.ttp_proposer.load_allowed_ids(mitre_json_path)
        print(f"✅ Loaded {len(allowed_ids)} available TTP IDs")
        
        # Stage 3: propose candidate techniques for each behavior
        print("\n🎯 Stage 3: proposing candidate techniques...")
        for i, behavior in enumerate(behaviors, 1):
            print(f"  Processing behavior {i}/{len(behaviors)}...")
            
            ttp_candidates = self.ttp_proposer.propose_ttp_candidates(
                behavior,
                allowed_ids,
                n=5
            )
            
            behavior.ttp_candidates = [c.to_dict() for c in ttp_candidates]
            print(f"    ✅ Proposed {len(ttp_candidates)} candidate techniques")
        
        # Stage 4: verify candidate techniques
        print("\n✅ Stage 4: verifying candidate techniques...")
        for i, behavior in enumerate(behaviors, 1):
            print(f"  Processing behavior {i}/{len(behaviors)}...")
            
            verified_result = self.ttp_verifier.verify_ttps_with_mitre(
                behavior,
                mitre_json,
                threshold=self.ttp_threshold
            )
            
            behavior.final_ttps = verified_result.to_dict()["final_ttps"]
            
            # Build the result
            result = {
                "behavior_text": behavior.behavior_text,
                "best_span_sentences": behavior.best_span_sentences,
                "best_span_range": behavior.best_span_range,
                "best_score": behavior.best_score,
                "ttp_candidates": behavior.ttp_candidates,
                "final_ttps": behavior.final_ttps,
                "debug_info": {
                    "span_debug": behavior.debug_info,
                    "ttp_verification_detail": verified_result.per_ttp_detail
                }
            }
            
            results.append(result)
            print(f"    ✅ Verification complete, final TTP count: {len(behavior.final_ttps)}")
        
        return results
    
    def process_file(
        self,
        report_file_path: str,
        mitre_json_path: str,
        output_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Read a report from a file and process it
        
        Args:
            report_file_path: Path to the CTI report file (.txt file)
            mitre_json_path: Path to the MITRE ATT&CK JSON file
            output_path: Output file path (optional; results are saved if given)
        
        Returns:
            List of results
        """
        # Read the report file
        report_path = Path(report_file_path)
        if not report_path.exists():
            raise FileNotFoundError(f"Report file does not exist: {report_file_path}")
        
        with open(report_path, 'r', encoding='utf-8') as f:
            report_text = f.read()
        
        # Process
        results = self.process(report_text, mitre_json_path)
        
        # Save the results (if an output path was given)
        if output_path:
            output_file = Path(output_path)
            # Make sure the output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
