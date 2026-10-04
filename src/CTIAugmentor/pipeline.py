#!/usr/bin/env python3
"""
CTIAugmentor Pipeline
Main pipeline for building High Level ↔ TTP and IoC ↔ TTP edges
"""

import json
import hashlib
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from .data_loader import DataLoader
from .candidate_generator import CandidateGenerator
from .llm_verifier import LLMVerifier, OpenAIVerifier
from .models import HighLevelNode, IoCNode, TTPNode, Edge


class CTIAugmentorPipeline:
    """Main CTIAugmentor pipeline"""
    
    def __init__(
        self,
        delta: int = 0,
        verifier_threshold: float = 0.8,
        model: str = "gpt-5",
        verifier: Optional[LLMVerifier] = None,
        check_split: bool = False
    ):
        """
        Initialize pipeline
        
        Args:
            delta: Distance threshold for candidate generation (default 0)
            verifier_threshold: Confidence threshold for edge verification (default 0.8)
            model: LLM model name (default "gpt-5")
            verifier: LLM verifier instance (if None, creates OpenAIVerifier)
            check_split: Whether to check sentence alignment (default False)
        """
        self.delta = delta
        self.verifier_threshold = verifier_threshold
        self.model = model
        self.check_split = check_split
        
        self.data_loader = DataLoader()
        self.candidate_generator = CandidateGenerator(delta=delta)
        
        if verifier is None:
            self.verifier = OpenAIVerifier(model=model)
        else:
            self.verifier = verifier
    
    def process(
        self,
        highlevel_json_path: str,
        ioc_json_path: str,
        ttp_json_path: str,
        input_txt_path: str,
        mitre_json_path: str
    ) -> Dict[str, Any]:
        """
        Process inputs and generate augmented graph
        
        Args:
            highlevel_json_path: Path to CTIHighLevelExtractor output
            ioc_json_path: Path to CTIIoCExtractor output
            ttp_json_path: Path to CTITagger output
            input_txt_path: Path to original CTI report text
            mitre_json_path: Path to MITRE ATT&CK JSON (for TTP descriptions)
        
        Returns:
            Dictionary containing nodes and edges
        """
        print("📖 Loading input data...")
        
        # Load data
        highlevel_entities = self.data_loader.load_highlevel_json(highlevel_json_path)
        ioc_data = self.data_loader.load_ioc_json(ioc_json_path)
        ttp_behaviors = self.data_loader.load_ttp_json(ttp_json_path)
        input_text = self.data_loader.load_input_txt(input_txt_path)
        ttp_description_map = self._load_ttp_descriptions(mitre_json_path)
        
        print(f"  - Loaded {len(highlevel_entities)} high-level entities")
        print(f"  - Loaded {len(ioc_data)} IoCs")
        print(f"  - Loaded {len(ttp_behaviors)} behaviors")
        print(f"  - Loaded {len(ttp_description_map)} TTP descriptions from MITRE")
        
        # Split text into sentences
        print("\n✂️  Splitting text into sentences...")
        sentences = self.data_loader.split_text(input_text)
        print(f"  - Split into {len(sentences)} sentences")
        
        # Build nodes
        print("\n🔨 Building nodes...")
        highlevel_nodes = self.data_loader.build_highlevel_nodes(highlevel_entities)
        ioc_nodes = self.data_loader.build_ioc_nodes(ioc_data)
        ttp_nodes = self.data_loader.build_ttp_nodes(ttp_behaviors)
        
        print(f"  - Built {len(highlevel_nodes)} high-level nodes (after deduplication)")
        print(f"  - Built {len(ioc_nodes)} IoC nodes (after deduplication)")
        print(f"  - Built {len(ttp_nodes)} TTP nodes (after deduplication)")
        
        # Check alignment if requested
        if self.check_split:
            print("\n🔍 Checking sentence alignment...")
            is_valid, errors = self.data_loader.check_alignment(
                sentences, highlevel_nodes, ioc_nodes, ttp_nodes
            )
            if not is_valid:
                print("❌ Alignment check failed!")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"  - {error}")
                if len(errors) > 10:
                    print(f"  ... and {len(errors) - 10} more errors")
                raise ValueError(f"Sentence alignment check failed with {len(errors)} errors")
            print("✅ Alignment check passed")
        
        # Generate candidates
        print("\n🎯 Generating candidate edges...")
        candidates = self.candidate_generator.generate_candidates(
            highlevel_nodes, ioc_nodes, ttp_nodes
        )
        print(f"  - Generated {len(candidates)} candidate edge instances")
        
        # Group candidates by evidence
        candidate_groups = self.candidate_generator.group_candidates_by_evidence(candidates)
        print(f"  - Grouped into {len(candidate_groups)} batches")
        
        # Combine all node maps for verifier
        all_nodes = {}
        all_nodes.update(highlevel_nodes)
        all_nodes.update(ioc_nodes)
        all_nodes.update(ttp_nodes)
        
        # Verify candidates in batches
        print("\n✅ Verifying candidates with LLM...")
        verified_edges = []
        rejected_edges = []  # Track edges that failed verification
        verified_edge_pairs = set()  # Track (node_u, node_v) pairs that are already verified
        verified_edge_ids = set()  # Track edge_id to prevent duplicates
        
        batch_num = 0
        for evidence_key, batch_candidates in candidate_groups.items():
            batch_num += 1
            if batch_num % 10 == 0:
                print(f"  Processing batch {batch_num}/{len(candidate_groups)}...")
            
            # Get context sentences (global indices -> actual sentences)
            evidence_indices = list(evidence_key)
            batch_context = [sentences[idx] for idx in evidence_indices]
            
            # Filter out candidates that are already verified
            filtered_candidates = []
            filtered_indices = []
            for i, candidate in enumerate(batch_candidates):
                edge_pair = (candidate.node_u_id, candidate.node_v_id)
                if edge_pair not in verified_edge_pairs:
                    filtered_candidates.append(candidate)
                    filtered_indices.append(i)
            
            if not filtered_candidates:
                continue
            
            # Verify batch
            verification_results = self.verifier.verify_batch(
                batch_context=batch_context,
                candidates=filtered_candidates,
                node_map=all_nodes,
                threshold=self.verifier_threshold,
                evidence_indices=evidence_indices,
                ttp_description_map=ttp_description_map
            )
            
            # Process results
            for result in verification_results:
                candidate_idx = result["candidate_index"]
                if candidate_idx >= len(filtered_candidates):
                    continue
                
                candidate = filtered_candidates[candidate_idx]
                
                # Get evidence sentences (global indices)
                evidence_sentences = [sentences[idx] for idx in candidate.evidence_indices_set]
                edge_id = self._generate_edge_id(candidate.node_u_id, candidate.node_v_id)
                
                # Create edge with same structure for both verified and rejected
                edge = Edge(
                    edge_id=edge_id,
                    node_u=candidate.node_u_id,
                    node_v=candidate.node_v_id,
                    edge_type=candidate.edge_type,
                    evidence={
                        "evidence_sentence_indices_global": candidate.evidence_indices_set,
                        "evidence_sentences": evidence_sentences,
                        "batch_local_mapping": {
                            "evidence_key": list(evidence_key),
                            "batch_index": batch_num
                        }
                    },
                    verification={
                        "confidence": result["confidence"],
                        "threshold": self.verifier_threshold,
                        "model": result.get("model", self.model),
                        "reasoning": result.get("reasoning", "")
                    }
                )
                
                if result["verified"]:
                    # Check if edge_id already exists (prevent duplicates within same batch or across batches)
                    if edge_id in verified_edge_ids:
                        continue
                    
                    verified_edges.append(edge)
                    verified_edge_pairs.add((candidate.node_u_id, candidate.node_v_id))
                    verified_edge_ids.add(edge_id)
                else:
                    # Edge failed verification - add to rejected_edges
                    rejected_edges.append(edge)

        # Supplement center-entity links to uncovered TTPs
        print("\n⭐ Supplementing center-entity TTP edges...")
        center_entity = self._select_potential_center_entity(highlevel_nodes)
        if center_entity is None:
            print("  - No potential center entity found in ThreatActor/Campaign/Malware categories")
        else:
            linked_ttp_node_ids = {
                edge.node_v for edge in verified_edges
                if edge.node_u == center_entity.node_id and edge.edge_type == "high_level-ttp"
            }
            center_candidates = self._build_center_ttp_candidates(ttp_nodes, linked_ttp_node_ids)
            print(
                f"  - Potential center entity: {center_entity.node_text} ({center_entity.node_type}), "
                f"unlinked TTP candidates: {len(center_candidates)}"
            )

            if center_candidates:
                center_verification = self.verifier.verify_center_entity_missing_ttps(
                    sentences=sentences,
                    center_entity={
                        "node_id": center_entity.node_id,
                        "node_text": center_entity.node_text,
                        "node_type": center_entity.node_type,
                        "sentence_indices": center_entity.sentence_indices,
                        "span_sentences": center_entity.span_sentences,
                    },
                    ttp_candidates=center_candidates,
                    threshold=self.verifier_threshold,
                    ttp_description_map=ttp_description_map
                )

                if not center_verification.get("is_center_entity", False):
                    print("  - LLM judged the candidate is not a center entity; supplementation skipped")
                else:
                    center_candidate_map = {
                        item["ttp_node_id"]: item for item in center_candidates
                    }
                    added_count = 0
                    for result in center_verification.get("results", []):
                        ttp_node_id = result.get("ttp_node_id", "")
                        if not ttp_node_id:
                            continue

                        candidate_meta = center_candidate_map.get(ttp_node_id)
                        if not candidate_meta:
                            continue

                        edge_id = self._generate_edge_id(center_entity.node_id, ttp_node_id)
                        evidence_indices = candidate_meta.get("evidence_sentence_indices_global", [])
                        evidence_sentences = [
                            sentences[idx] for idx in evidence_indices
                            if 0 <= idx < len(sentences)
                        ]

                        edge = Edge(
                            edge_id=edge_id,
                            node_u=center_entity.node_id,
                            node_v=ttp_node_id,
                            edge_type="high_level-ttp",
                            evidence={
                                "evidence_sentence_indices_global": evidence_indices,
                                "evidence_sentences": evidence_sentences,
                                "batch_local_mapping": {
                                    "evidence_key": evidence_indices,
                                    "batch_index": "center_entity_supplement"
                                }
                            },
                            verification={
                                "confidence": result.get("confidence", 0.0),
                                "threshold": self.verifier_threshold,
                                "model": result.get("model", self.model),
                                "reasoning": result.get("reasoning", ""),
                                "stage": "center_entity_supplement"
                            }
                        )

                        if result.get("verified", False):
                            if edge_id in verified_edge_ids:
                                continue
                            verified_edges.append(edge)
                            verified_edge_pairs.add((center_entity.node_id, ttp_node_id))
                            verified_edge_ids.add(edge_id)
                            added_count += 1
                        else:
                            rejected_edges.append(edge)

                    print(f"  - Added {added_count} supplementary center-entity edges")
        
        print(f"  - Verified {len(verified_edges)} edges (passed threshold {self.verifier_threshold})")
        print(f"  - Rejected {len(rejected_edges)} edges (below threshold {self.verifier_threshold})")
        
        # Update node edges lists
        print("\n🔗 Updating node edge references...")
        for edge in verified_edges:
            if edge.node_u in all_nodes:
                if edge.edge_id not in all_nodes[edge.node_u].edges:
                    all_nodes[edge.node_u].edges.append(edge.edge_id)
            if edge.node_v in all_nodes:
                if edge.edge_id not in all_nodes[edge.node_v].edges:
                    all_nodes[edge.node_v].edges.append(edge.edge_id)
        
        # Build output
        print("\n📊 Building output...")
        output = {
            "nodes": {
                "high_level": [node.to_dict() for node in highlevel_nodes.values()],
                "ttp": [node.to_dict() for node in ttp_nodes.values()],
                "ioc": [node.to_dict() for node in ioc_nodes.values()]
            },
            "edges": [edge.to_dict() for edge in verified_edges],
            "debug_edges": [edge.to_dict() for edge in rejected_edges]
        }
        
        print(f"  - Output: {len(output['nodes']['high_level'])} high-level nodes, "
              f"{len(output['nodes']['ttp'])} TTP nodes, {len(output['nodes']['ioc'])} IoC nodes, "
              f"{len(output['edges'])} edges, {len(output['debug_edges'])} debug_edges")
        
        return output

    @staticmethod
    def _select_potential_center_entity(
        highlevel_nodes: Dict[str, HighLevelNode]
    ) -> Optional[HighLevelNode]:
        """
        Select potential center entity from ThreatActor/Campaign/Malware categories
        by maximum span-sentence count.
        """
        allowed_types = {"ThreatActor", "Campaign", "MalwareFamily"}
        candidates = [
            node for node in highlevel_nodes.values()
            if node.node_type in allowed_types
        ]
        if not candidates:
            return None

        def score(node: HighLevelNode) -> int:
            if node.span_sentences:
                return len(node.span_sentences)
            return len(node.sentence_indices)

        candidates.sort(
            key=lambda node: (
                score(node),
                len(node.sentence_indices),
                node.node_text,
                node.node_id
            ),
            reverse=True
        )
        return candidates[0]

    @staticmethod
    def _extract_ttp_evidence_indices(ttp_node: TTPNode) -> List[int]:
        """Extract sorted sentence indices from all span ranges of a TTP node."""
        indices: Set[int] = set()
        for span_range in ttp_node.best_span_ranges:
            if not isinstance(span_range, list) or len(span_range) != 2:
                continue
            start_idx, end_idx = span_range
            if not isinstance(start_idx, int) or not isinstance(end_idx, int):
                continue
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            indices.update(range(start_idx, end_idx + 1))

        return sorted(indices)

    def _build_center_ttp_candidates(
        self,
        ttp_nodes: Dict[str, TTPNode],
        linked_ttp_node_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """Build candidate TTP list for center-entity supplementary verification."""
        candidates: List[Dict[str, Any]] = []
        for ttp_node in sorted(ttp_nodes.values(), key=lambda node: node.ttp_id):
            if ttp_node.node_id in linked_ttp_node_ids:
                continue
            evidence_indices = self._extract_ttp_evidence_indices(ttp_node)
            candidates.append({
                "ttp_node_id": ttp_node.node_id,
                "ttp_id": ttp_node.ttp_id,
                "ttp_node_text": ttp_node.node_text,
                "evidence_sentence_indices_global": evidence_indices,
            })
        return candidates

    @staticmethod
    def _load_ttp_descriptions(mitre_json_path: str) -> Dict[str, str]:
        """
        Load TTP descriptions from MITRE JSON.

        `contents` can be a JSON string; only `description` is used.
        """
        with open(mitre_json_path, 'r', encoding='utf-8') as f:
            mitre_data = json.load(f)

        descriptions: Dict[str, str] = {}

        if not isinstance(mitre_data, dict):
            return descriptions

        for key, value in mitre_data.items():
            if not isinstance(value, dict):
                continue

            ttp_id = str(value.get("mitre_id", key)).strip().upper()
            if not ttp_id.startswith("T"):
                continue

            contents = value.get("contents")
            description = ""

            if isinstance(contents, str):
                contents = contents.strip()
                if not contents:
                    continue

                try:
                    contents_json = json.loads(contents)
                    if isinstance(contents_json, dict):
                        description = str(contents_json.get("description", "")).strip()
                except json.JSONDecodeError:
                    # Fallback: keep raw text if not JSON-formatted.
                    description = contents
            elif isinstance(contents, dict):
                description = str(contents.get("description", "")).strip()

            if description and ttp_id not in descriptions:
                descriptions[ttp_id] = description

        return descriptions
    
    def _generate_edge_id(self, node_u_id: str, node_v_id: str) -> str:
        """Generate unique edge ID"""
        key = f"{node_u_id}|||{node_v_id}"
        hash_obj = hashlib.sha1(key.encode('utf-8'))
        return "EDGE_" + hash_obj.hexdigest()[:16]
