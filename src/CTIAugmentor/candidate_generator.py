#!/usr/bin/env python3
"""
Candidate edge generator
Generates candidate edges based on proximity (distance between spans)
"""

from typing import List, Dict, Set, Tuple
from .models import HighLevelNode, IoCNode, TTPNode, CandidateEdgeInstance


class CandidateGenerator:
    """Generate candidate edges based on proximity"""
    
    def __init__(self, delta: int = 0):
        """
        Initialize candidate generator
        
        Args:
            delta: Distance threshold (0 = same sentence overlap, 1 = adjacent sentences, etc.)
        """
        self.delta = delta
    
    def compute_span_distance(self, span1: List[int], span2: List[int]) -> int:
        """
        Compute distance between two spans
        - dist = 0 if spans overlap
        - Otherwise dist = min(|a_end - b_start|, |b_end - a_start|)
        
        Args:
            span1: [start, end] (inclusive)
            span2: [start, end] (inclusive)
        
        Returns:
            Distance between spans
        """
        a_start, a_end = span1
        b_start, b_end = span2
        
        # Check overlap
        if not (a_end < b_start or b_end < a_start):
            return 0
        
        # No overlap, compute minimum gap
        dist1 = abs(a_end - b_start) if a_end < b_start else abs(b_end - a_start)
        return dist1
    
    def generate_candidates(
        self,
        highlevel_nodes: Dict[str, HighLevelNode],
        ioc_nodes: Dict[str, IoCNode],
        ttp_nodes: Dict[str, TTPNode]
    ) -> List[CandidateEdgeInstance]:
        """
        Generate candidate edges: HighLevel ↔ TTP and IoC ↔ TTP
        
        Args:
            highlevel_nodes: Dictionary of High Level nodes
            ioc_nodes: Dictionary of IoC nodes
            ttp_nodes: Dictionary of TTP nodes
        
        Returns:
            List of candidate edge instances
        """
        candidates = []
        
        # Generate HighLevel ↔ TTP candidates
        for hl_node in highlevel_nodes.values():
            for ttp_node in ttp_nodes.values():
                # For each discrete sentence index of HighLevel
                for hl_sent_idx in hl_node.sentence_indices:
                    hl_span = [hl_sent_idx, hl_sent_idx]
                    
                    # For each span range of TTP
                    for ttp_range in ttp_node.best_span_ranges:
                        dist = self.compute_span_distance(hl_span, ttp_range)
                        
                        if dist <= self.delta:
                            # Create candidate instance
                            # Evidence indices: hl_sent_idx + all sentences in ttp_range
                            evidence_indices = [hl_sent_idx]
                            evidence_indices.extend(range(ttp_range[0], ttp_range[1] + 1))
                            evidence_indices = sorted(list(set(evidence_indices)))
                            
                            candidate = CandidateEdgeInstance(
                                node_u_id=hl_node.node_id,
                                node_v_id=ttp_node.node_id,
                                edge_type="high_level-ttp",
                                trigger_sentence_index=hl_sent_idx,
                                ttp_span_range=ttp_range,
                                min_dist=dist,
                                evidence_indices_set=evidence_indices
                            )
                            candidates.append(candidate)
        
        # Generate IoC ↔ TTP candidates
        for ioc_node in ioc_nodes.values():
            for ttp_node in ttp_nodes.values():
                # For each discrete sentence index of IoC
                for ioc_sent_idx in ioc_node.sentence_indices:
                    ioc_span = [ioc_sent_idx, ioc_sent_idx]
                    
                    # For each span range of TTP
                    for ttp_range in ttp_node.best_span_ranges:
                        dist = self.compute_span_distance(ioc_span, ttp_range)
                        
                        if dist <= self.delta:
                            # Create candidate instance
                            evidence_indices = [ioc_sent_idx]
                            evidence_indices.extend(range(ttp_range[0], ttp_range[1] + 1))
                            evidence_indices = sorted(list(set(evidence_indices)))
                            
                            candidate = CandidateEdgeInstance(
                                node_u_id=ioc_node.node_id,
                                node_v_id=ttp_node.node_id,
                                edge_type="ioc-ttp",
                                trigger_sentence_index=ioc_sent_idx,
                                ttp_span_range=ttp_range,
                                min_dist=dist,
                                evidence_indices_set=evidence_indices
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def group_candidates_by_evidence(
        self,
        candidates: List[CandidateEdgeInstance]
    ) -> Dict[Tuple[int, ...], List[CandidateEdgeInstance]]:
        """
        Group candidates by evidence_indices_set (for batch processing)
        
        Args:
            candidates: List of candidate edge instances
        
        Returns:
            Dictionary mapping evidence key to list of candidates
        """
        groups = {}
        
        for candidate in candidates:
            key = candidate.get_evidence_key()
            if key not in groups:
                groups[key] = []
            groups[key].append(candidate)
        
        return groups
