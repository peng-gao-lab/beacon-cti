#!/usr/bin/env python3
"""
Data models for CTIAugmentor
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class HighLevelNode:
    """High Level entity node"""
    node_id: str
    node_class: str = "high_level"
    node_type: str = ""  # entity_type
    node_text: str = ""  # entity_text
    normalized_value: str = ""
    sentence_indices: List[int] = field(default_factory=list)
    span_sentences: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    aliases: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "node_id": self.node_id,
            "node_class": self.node_class,
            "node_type": self.node_type,
            "node_text": self.node_text,
            "normalized_value": self.normalized_value,
            "sentence_indices": self.sentence_indices,
            "span_sentences": self.span_sentences,
            "edges": self.edges
        }
        if self.aliases:
            result["aliases"] = self.aliases
        return result


@dataclass
class IoCNode:
    """IoC node"""
    node_id: str
    node_class: str = "ioc"
    node_type: str = ""  # ioc_type
    node_text: str = ""  # value
    normalized_value: Optional[str] = None
    sentence_indices: List[int] = field(default_factory=list)
    span_sentences: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    sources: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "node_id": self.node_id,
            "node_class": self.node_class,
            "node_type": self.node_type,
            "node_text": self.node_text,
            "sentence_indices": self.sentence_indices,
            "span_sentences": self.span_sentences,
            "edges": self.edges
        }
        if self.normalized_value:
            result["normalized_value"] = self.normalized_value
        if self.sources:
            result["sources"] = self.sources
        return result


@dataclass
class TTPNode:
    """TTP node"""
    node_id: str
    node_class: str = "ttp"
    node_type: str = "TTP"
    node_text: str = ""  # "TTP:" + ttp_id
    ttp_id: str = ""
    best_span_sentences_list: List[List[str]] = field(default_factory=list)
    best_span_ranges: List[List[int]] = field(default_factory=list)  # [[start, end], ...]
    edges: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "node_class": self.node_class,
            "node_type": self.node_type,
            "node_text": self.node_text,
            "ttp_id": self.ttp_id,
            "best_span_sentences_list": self.best_span_sentences_list,
            "best_span_ranges": self.best_span_ranges,
            "edges": self.edges
        }


@dataclass
class Edge:
    """Knowledge graph edge"""
    edge_id: str
    node_u: str
    node_v: str
    edge_type: str  # "high_level-ttp" or "ioc-ttp"
    evidence: Dict[str, Any] = field(default_factory=dict)
    verification: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "edge_id": self.edge_id,
            "node_u": self.node_u,
            "node_v": self.node_v,
            "edge_type": self.edge_type,
            "evidence": self.evidence,
            "verification": self.verification
        }


@dataclass
class CandidateEdgeInstance:
    """Candidate edge instance (before verification)"""
    node_u_id: str
    node_v_id: str
    edge_type: str  # "high_level-ttp" or "ioc-ttp"
    trigger_sentence_index: int  # For high_level/ioc: the sentence index that triggered this
    ttp_span_range: List[int]  # [start, end] for TTP
    min_dist: int
    evidence_indices_set: List[int]  # Sorted list of all sentence indices involved
    
    def get_evidence_key(self) -> tuple:
        """Get a tuple key for grouping by evidence"""
        return tuple(sorted(self.evidence_indices_set))
