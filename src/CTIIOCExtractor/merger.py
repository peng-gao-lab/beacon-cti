#!/usr/bin/env python3
"""
IOC Merger and Deduplicator
"""

from typing import List, Dict, Any
from collections import defaultdict


class IOCMerger:
    """IOC Merger and Deduplicator"""
    
    @staticmethod
    def merge_and_deduplicate(
        regex_iocs: List[Dict[str, Any]], 
        llm_iocs: List[Dict[str, Any]], 
        sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate IOCs
        
        Args:
            regex_iocs: IOCs extracted by regex
            llm_iocs: IOCs extracted by LLM
            sentences: List of sentences
        
        Returns:
            Merged IOC list
        """
        # Use (ioc_type, normalized_value) as key for deduplication
        # value field preserves raw_value (original form), normalized_value is used for deduplication
        ioc_dict = defaultdict(lambda: {
            "ioc_type": "",
            "value": "",  # raw_value, preserve the first encountered original form
            "normalized_value": "",
            "sources": set(),
            "sentence_indices": set()
        })
        
        # Process regex IOCs
        for ioc in regex_iocs:
            key = (ioc["ioc_type"], ioc["normalized_value"])
            if not ioc_dict[key]["ioc_type"]:
                ioc_dict[key]["ioc_type"] = ioc["ioc_type"]
                ioc_dict[key]["value"] = ioc["value"]  # Preserve the first encountered raw_value
                ioc_dict[key]["normalized_value"] = ioc["normalized_value"]
            
            ioc_dict[key]["sources"].add("regex")
            ioc_dict[key]["sentence_indices"].update(ioc["sentence_indices"])
        
        # Process LLM IOCs
        for ioc in llm_iocs:
            key = (ioc["ioc_type"], ioc["normalized_value"])
            if not ioc_dict[key]["ioc_type"]:
                ioc_dict[key]["ioc_type"] = ioc["ioc_type"]
                ioc_dict[key]["value"] = ioc["value"]  # Preserve the first encountered raw_value
                ioc_dict[key]["normalized_value"] = ioc["normalized_value"]
            
            ioc_dict[key]["sources"].add("llm")
            ioc_dict[key]["sentence_indices"].update(ioc["sentence_indices"])
        
        # Convert to final format
        merged_iocs = []
        for key, data in ioc_dict.items():
            sentence_indices = sorted(list(data["sentence_indices"]))
            span_sentences = [sentences[idx] if 0 <= idx < len(sentences) else "" for idx in sentence_indices]
            
            merged_iocs.append({
                "ioc_type": data["ioc_type"],
                "value": data["value"],
                "normalized_value": data["normalized_value"],
                "sources": sorted(list(data["sources"])),
                "sentence_indices": sentence_indices,
                "span_sentences": span_sentences
            })
        
        return merged_iocs

