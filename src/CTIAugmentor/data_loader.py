#!/usr/bin/env python3
"""
Data loader for CTIAugmentor
Loads data from CTIHighLevelExtractor, CTIIoCExtractor, and CTITagger outputs
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .models import HighLevelNode, IoCNode, TTPNode
from .sentence_splitter import SentenceSplitter


class DataLoader:
    """Data loader for CTIAugmentor inputs"""
    
    def __init__(self):
        self.sentence_splitter = SentenceSplitter()
    
    def load_highlevel_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CTIHighLevelExtractor output JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "entities" in data:
            return data["entities"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected format in highlevel JSON: {file_path}")
    
    def load_ioc_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CTIIoCExtractor output JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "iocs" in data:
            return data["iocs"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected format in ioc JSON: {file_path}")
    
    def load_ttp_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CTITagger output JSON (array of behaviors)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected format in ttp JSON: {file_path}")
    
    def load_input_txt(self, file_path: str) -> str:
        """Load original CTI report text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def build_highlevel_nodes(
        self,
        entities: List[Dict[str, Any]]
    ) -> Dict[str, HighLevelNode]:
        """
        Build High Level nodes with deduplication
        Deduplication key: (entity_type, normalized_value)
        """
        node_map = {}
        
        for entity in entities:
            entity_type = entity.get("entity_type", "")
            normalized_value = entity.get("normalized_value", "")
            
            # Generate deduplication key
            key = (entity_type, normalized_value)
            key_str = f"{entity_type}|||{normalized_value}"
            
            # Generate stable hash for node_id
            hash_obj = hashlib.sha1(key_str.encode('utf-8'))
            node_id = "HL_" + hash_obj.hexdigest()[:16]
            
            if node_id in node_map:
                # Merge: combine sentence_indices and span_sentences
                existing = node_map[node_id]
                # Merge sentence_indices (avoid duplicates)
                combined_indices = list(set(existing.sentence_indices + entity.get("sentence_indices", [])))
                combined_indices.sort()
                existing.sentence_indices = combined_indices
                # Merge span_sentences (avoid duplicates)
                combined_sentences = list(set(existing.span_sentences + entity.get("span_sentences", [])))
                existing.span_sentences = combined_sentences
            else:
                # Create new node
                node = HighLevelNode(
                    node_id=node_id,
                    node_type=entity_type,
                    node_text=entity.get("entity_text", ""),
                    normalized_value=normalized_value,
                    sentence_indices=entity.get("sentence_indices", []),
                    span_sentences=entity.get("span_sentences", []),
                    aliases=entity.get("aliases")
                )
                node_map[node_id] = node
        
        return node_map
    
    def build_ioc_nodes(
        self,
        iocs: List[Dict[str, Any]]
    ) -> Dict[str, IoCNode]:
        """
        Build IoC nodes with deduplication
        Deduplication key: (ioc_type, normalized_value or value)
        """
        node_map = {}
        
        for ioc in iocs:
            ioc_type = ioc.get("ioc_type", "")
            normalized_value = ioc.get("normalized_value")
            value = ioc.get("value", "")
            
            # Use normalized_value if available, otherwise use value
            dedup_value = normalized_value if normalized_value else value
            
            # Generate deduplication key
            key_str = f"{ioc_type}|||{dedup_value}"
            
            # Generate stable hash for node_id
            hash_obj = hashlib.sha1(key_str.encode('utf-8'))
            node_id = "IOC_" + hash_obj.hexdigest()[:16]
            
            if node_id in node_map:
                # Merge: combine sentence_indices and span_sentences
                existing = node_map[node_id]
                combined_indices = list(set(existing.sentence_indices + ioc.get("sentence_indices", [])))
                combined_indices.sort()
                existing.sentence_indices = combined_indices
                combined_sentences = list(set(existing.span_sentences + ioc.get("span_sentences", [])))
                existing.span_sentences = combined_sentences
            else:
                # Create new node
                node = IoCNode(
                    node_id=node_id,
                    node_type=ioc_type,
                    node_text=value,
                    normalized_value=normalized_value,
                    sentence_indices=ioc.get("sentence_indices", []),
                    span_sentences=ioc.get("span_sentences", []),
                    sources=ioc.get("sources")
                )
                node_map[node_id] = node
        
        return node_map
    
    def build_ttp_nodes(
        self,
        behaviors: List[Dict[str, Any]]
    ) -> Dict[str, TTPNode]:
        """
        Build TTP nodes with deduplication
        Deduplication key: ttp_id (merge ranges and sentences_list)
        """
        node_map = {}
        
        for behavior in behaviors:
            final_ttps = behavior.get("final_ttps", [])
            
            for final_ttp in final_ttps:
                ttp_id = final_ttp.get("ttp_id", "")
                if not ttp_id:
                    continue
                
                best_span_sentences = final_ttp.get("best_span_sentences", [])
                best_span_range = final_ttp.get("best_span_range", [])
                
                if not best_span_range or len(best_span_range) != 2:
                    continue
                
                node_id = "TTP_" + ttp_id
                
                if node_id in node_map:
                    # Merge: add to best_span_sentences_list and best_span_ranges
                    existing = node_map[node_id]
                    existing.best_span_sentences_list.append(best_span_sentences)
                    existing.best_span_ranges.append(best_span_range)
                else:
                    # Create new node
                    node = TTPNode(
                        node_id=node_id,
                        ttp_id=ttp_id,
                        node_text=f"TTP:{ttp_id}",
                        best_span_sentences_list=[best_span_sentences],
                        best_span_ranges=[best_span_range]
                    )
                    node_map[node_id] = node
        
        return node_map
    
    def split_text(self, text: str) -> List[str]:
        """Split text into sentences (using same logic as CTITagger)"""
        return self.sentence_splitter.split_sentences(text)
    
    def check_alignment(
        self,
        sentences: List[str],
        highlevel_nodes: Dict[str, HighLevelNode],
        ioc_nodes: Dict[str, IoCNode],
        ttp_nodes: Dict[str, TTPNode]
    ) -> Tuple[bool, List[str]]:
        """
        Check alignment between sentence indices and actual sentences
        Returns (is_valid, error_messages)
        """
        errors = []
        
        # Check High Level nodes
        for node_id, node in highlevel_nodes.items():
            for i, sent_idx in enumerate(node.sentence_indices):
                if sent_idx < 0 or sent_idx >= len(sentences):
                    errors.append(
                        f"HighLevel node {node_id}: sentence_index {sent_idx} out of range "
                        f"(total sentences: {len(sentences)})"
                    )
                    continue
                
                expected_sentence = node.span_sentences[i] if i < len(node.span_sentences) else None
                actual_sentence = sentences[sent_idx]
                
                if expected_sentence:
                    # Allow trimming whitespace differences
                    if expected_sentence.strip() != actual_sentence.strip():
                        errors.append(
                            f"HighLevel node {node_id}: sentence {sent_idx} mismatch\n"
                            f"  Expected: {expected_sentence[:100]}...\n"
                            f"  Actual: {actual_sentence[:100]}..."
                        )
        
        # Check IoC nodes
        for node_id, node in ioc_nodes.items():
            for i, sent_idx in enumerate(node.sentence_indices):
                if sent_idx < 0 or sent_idx >= len(sentences):
                    errors.append(
                        f"IoC node {node_id}: sentence_index {sent_idx} out of range "
                        f"(total sentences: {len(sentences)})"
                    )
                    continue
                
                expected_sentence = node.span_sentences[i] if i < len(node.span_sentences) else None
                actual_sentence = sentences[sent_idx]
                
                if expected_sentence:
                    if expected_sentence.strip() != actual_sentence.strip():
                        errors.append(
                            f"IoC node {node_id}: sentence {sent_idx} mismatch\n"
                            f"  Expected: {expected_sentence[:100]}...\n"
                            f"  Actual: {actual_sentence[:100]}..."
                        )
        
        # Check TTP nodes
        for node_id, node in ttp_nodes.items():
            for range_idx, span_range in enumerate(node.best_span_ranges):
                if len(span_range) != 2:
                    errors.append(f"TTP node {node_id}: invalid span_range {span_range}")
                    continue
                
                start, end = span_range
                if start < 0 or end >= len(sentences) or start > end:
                    errors.append(
                        f"TTP node {node_id}: span_range [{start}, {end}] out of range "
                        f"(total sentences: {len(sentences)})"
                    )
                    continue
                
                # Get actual sentences from range
                actual_sentences = sentences[start:end+1]
                expected_sentences = node.best_span_sentences_list[range_idx] if range_idx < len(node.best_span_sentences_list) else []
                
                if expected_sentences:
                    if len(actual_sentences) != len(expected_sentences):
                        errors.append(
                            f"TTP node {node_id}: span_range [{start}, {end}] length mismatch\n"
                            f"  Expected {len(expected_sentences)} sentences, got {len(actual_sentences)}"
                        )
                    else:
                        for j, (exp_sent, act_sent) in enumerate(zip(expected_sentences, actual_sentences)):
                            if exp_sent.strip() != act_sent.strip():
                                errors.append(
                                    f"TTP node {node_id}: span_range [{start}, {end}], sentence {j} mismatch\n"
                                    f"  Expected: {exp_sent[:100]}...\n"
                                    f"  Actual: {act_sent[:100]}..."
                                )
        
        return len(errors) == 0, errors
