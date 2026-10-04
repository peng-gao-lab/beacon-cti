#!/usr/bin/env python3
"""
CTI IOC Extractor Main Class
Integrates all functional modules
"""

from copy import deepcopy
from typing import List, Dict, Any, Optional

# Import LLM Provider
try:
    from CTITagger.core.llm_provider import LLMProvider
except ImportError:
    LLMProvider = None

from .sentence_splitter import SentenceSplitter
from .regex_extractor import RegexExtractor
from .llm_extractor import LLMExtractor
from .merger import IOCMerger


class CTIIOCExtractor:
    """CTI IOC Extractor"""
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None, model: str = "gpt-5-mini"):
        """
        Initialize IOC Extractor
        
        Args:
            llm_provider: LLM Provider instance (optional)
            model: LLM model name (default: gpt-5-mini)
        """
        self.llm_provider = llm_provider
        self.model = model
        
        # Initialize components
        self.sentence_splitter = SentenceSplitter()
        self.regex_extractor = RegexExtractor()
        self.llm_extractor = LLMExtractor(llm_provider, model)
        self.merger = IOCMerger()
        # Optional observability state for the CLI sidecar recorder.  Neither
        # field is used to produce the normal extraction result.
        self.pre_verification_iocs: List[Dict[str, Any]] = []
        self.pre_verification_merged_iocs: List[Dict[str, Any]] = []
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentence list
        
        Args:
            text: Original text
        
        Returns:
            List of sentences
        """
        return self.sentence_splitter.split_sentences(text)
    
    def extract(
        self,
        text: str,
        record_pre_verification: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract IOCs from text
        
        Args:
            text: Input text
            record_pre_verification: Populate optional pre-review recording state
        
        Returns:
            List of IOCs
        """
        self.pre_verification_iocs = []
        self.pre_verification_merged_iocs = []

        # 1. Split sentences
        sentences = self.split_sentences(text)
        print(f"✅ Sentence splitting completed, {len(sentences)} sentences")
        
        # 2. Regex extraction (disabled, using LLM only)
        # print("🔍 Starting regex IOC extraction...")
        # regex_iocs = self.regex_extractor.extract(sentences)
        # print(f"✅ Regex extraction found {len(regex_iocs)} IOCs")
        regex_iocs = []  # Disable regex, pass empty list
        
        # 3. LLM extraction
        print("🤖 Starting LLM IOC extraction...")
        llm_iocs = self.llm_extractor.extract(
            sentences,
            report_text=text,
            record_pre_verification=record_pre_verification,
        )
        print(f"✅ LLM extraction found {len(llm_iocs)} IOCs")

        # Copy and merge the pre-review snapshot for optional sidecar output.
        # This does not feed back into llm_iocs or the final result.
        if record_pre_verification:
            self.pre_verification_iocs = deepcopy(
                self.llm_extractor.last_pre_verification_iocs
            )
            self.pre_verification_merged_iocs = self.merger.merge_and_deduplicate(
                regex_iocs,
                self.pre_verification_iocs,
                sentences,
            )
        
        # 4. Merge and deduplicate
        print("🔄 Merging and deduplicating IOCs...")
        merged_iocs = self.merger.merge_and_deduplicate(regex_iocs, llm_iocs, sentences)
        print(f"✅ Final extraction found {len(merged_iocs)} unique IOCs")
        
        return merged_iocs
