#!/usr/bin/env python3
"""
Stage 2: Behavior Span Locator
Uses a bag-of-words approach to map behavior_text back to a contiguous sentence range in the original text
"""

import re
from typing import List, Dict, Any, Tuple, Set
from collections import Counter

# Try to import NLTK (optional)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    
    # Download the required NLTK data (if not already present)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except:
            pass
except ImportError:
    NLTK_AVAILABLE = False
    # Use a simple fallback
    class PorterStemmer:
        def stem(self, word):
            # Simple stemming (fallback)
            if word.endswith('ing'):
                return word[:-3]
            elif word.endswith('ed'):
                return word[:-2]
            elif word.endswith('s'):
                return word[:-1]
            return word


# BEACON port: sentence splitter selection. BEACON_SENTENCE_SPLITTER=regex forces
# the built-in regex splitter (the behaviour of the reference runs, where the
# NLTK punkt_tab data was absent). Unset or "nltk" keeps the original
# "NLTK if available" behaviour. Stopwords/stemming are not affected.
import os as _beacon_os
USE_NLTK_SENTENCE_SPLITTER = NLTK_AVAILABLE and (
    _beacon_os.environ.get("BEACON_SENTENCE_SPLITTER", "").strip().lower() != "regex"
)

class SpanLocatorConfig:
    """Span Locator configuration class"""
    
    def __init__(
        self,
        lmax: int = 5,
        high_freq_threshold: float = 0.01,  # Top 1% high-frequency tokens are not scored
        enable_ngram: bool = True,
        ngram_bonus: float = 0.5,  # n-gram match bonus
        tie_break_prefer_shorter: bool = True  # True=prefer the shorter span, False=prefer the longer span
    ):
        self.lmax = lmax
        self.high_freq_threshold = high_freq_threshold
        self.enable_ngram = enable_ngram
        self.ngram_bonus = ngram_bonus
        self.tie_break_prefer_shorter = tie_break_prefer_shorter


class SpanLocator:
    """Behavior span locator"""
    
    def __init__(self, config: SpanLocatorConfig = None):
        """
        Initialize the locator
        
        Args:
            config: Configuration object
        """
        self.config = config or SpanLocatorConfig()
        self.stemmer = PorterStemmer()
        
        # Load stopwords
        if NLTK_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words('english'))
            except:
                self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        else:
            # Use a basic stopword list
            self.stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can'])
        
        # Synonym map (small dictionary)
        self.synonym_map = {
            'credential': 'credential',
            'creds': 'credential',
            'password': 'credential',
            'privilege escalation': 'privilege_escalation',
            'privesc': 'privilege_escalation',
            'command and control': 'command_control',
            'c2': 'command_control',
            'lateral movement': 'lateral_movement',
            'exfiltration': 'exfiltration',
            'data exfiltration': 'exfiltration'
        }
    
    def normalize_token(self, token: str) -> str:
        """
        Normalize a token: lowercase + synonym mapping
        
        Args:
            token: Raw token
        
        Returns:
            Normalized token
        """
        token_lower = token.lower()
        # Synonym mapping
        if token_lower in self.synonym_map:
            return self.synonym_map[token_lower]
        return token_lower
    
    def tokenize_and_normalize(self, text: str) -> List[str]:
        """
        Tokenize and normalize
        
        Args:
            text: Input text
        
        Returns:
            List of normalized tokens
        """
        # Simple tokenization: keep alphanumerics and hyphens
        tokens = re.findall(r'\b[a-zA-Z0-9\-]+\b', text.lower())
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]
        
        # Morphological normalization (stemming)
        tokens = [self.stemmer.stem(t) for t in tokens]
        
        # Synonym normalization
        tokens = [self.normalize_token(t) for t in tokens]
        
        return tokens
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split the text into a list of sentences
        
        Args:
            text: Raw text
        
        Returns:
            List of sentences
        """
        # Use the NLTK sentence splitter (if available)
        if USE_NLTK_SENTENCE_SPLITTER:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = self._split_sentences_fallback(text)
        else:
            sentences = self._split_sentences_fallback(text)
        
        # If splitting fails, return at least the whole text as one sentence
        if not sentences:
            sentences = [text]
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _split_sentences_fallback(self, text: str) -> List[str]:
        """Fallback sentence splitting method (without NLTK)"""
        sentences = []
        # Simple rule: sentence-ending punctuation (. ? !) followed by whitespace or a newline
        pattern = r'([.!?])\s+'
        parts = re.split(pattern, text)
        
        # Reassemble the sentences
        current_sentence = ""
        for i, part in enumerate(parts):
            if part in '.!?':
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Add the last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def compute_ngram_overlap(self, tokens1: List[str], tokens2: List[str], n: int = 2) -> int:
        """
        Compute the n-gram overlap count
        
        Args:
            tokens1: First token list
            tokens2: Second token list
            n: The n of the n-gram
        
        Returns:
            Number of overlapping n-grams
        """
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        ngrams1 = set(get_ngrams(tokens1, n))
        ngrams2 = set(get_ngrams(tokens2, n))
        return len(ngrams1 & ngrams2)
    
    def locate_span(
        self,
        behavior_text: str,
        raw_text: str
    ) -> Tuple[List[str], Tuple[int, int], float, Dict[str, Any]]:
        """
        Locate the best contiguous sentence range for behavior_text in raw_text
        
        Args:
            behavior_text: Extracted behavior text
            raw_text: Raw CTI report text
        
        Returns:
            (best_span_sentences, best_span_range, best_score, debug_info)
        """
        # 1. Sentence splitting
        sentences = self.split_sentences(raw_text)
        if not sentences:
            return [], (0, 0), 0.0, {"error": "Unable to split sentences"}
        
        # 2. Preprocessing and normalization
        behavior_tokens = self.tokenize_and_normalize(behavior_text)
        sentence_tokens = [self.tokenize_and_normalize(s) for s in sentences]
        
        # Count token frequencies over the full text (used for down-weighting)
        all_tokens = []
        for tokens in sentence_tokens:
            all_tokens.extend(tokens)
        token_freq = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # Compute the high-frequency token threshold
        if total_tokens > 0:
            high_freq_count = int(total_tokens * self.config.high_freq_threshold)
            high_freq_tokens = set()
            for token, count in token_freq.most_common(high_freq_count):
                high_freq_tokens.add(token)
        else:
            high_freq_tokens = set()
        
        # 3. Generate candidate contiguous spans and score them
        best_span = None
        best_score = -1.0
        best_range = (0, 0)
        candidates = []
        
        for start_idx in range(len(sentences)):
            for length in range(1, min(self.config.lmax + 1, len(sentences) - start_idx + 1)):
                end_idx = start_idx + length - 1
                span_sentences = sentences[start_idx:end_idx + 1]
                span_tokens = []
                for i in range(start_idx, end_idx + 1):
                    span_tokens.extend(sentence_tokens[i])
                
                # 4. BoW match scoring
                span_token_set = set(span_tokens)
                behavior_token_set = set(behavior_tokens)
                
                # Basic overlap
                overlap_tokens = span_token_set & behavior_token_set
                
                # Remove high-frequency tokens
                overlap_tokens = overlap_tokens - high_freq_tokens
                
                # Compute the weighted score
                score = 0.0
                for token in overlap_tokens:
                    # Weight: 1 / (1 + freq)
                    freq = token_freq.get(token, 0)
                    weight = 1.0 / (1.0 + freq)
                    score += weight
                
                # n-gram bonus
                if self.config.enable_ngram and len(behavior_tokens) >= 2:
                    bigram_overlap = self.compute_ngram_overlap(behavior_tokens, span_tokens, n=2)
                    score += bigram_overlap * self.config.ngram_bonus
                    
                    if len(behavior_tokens) >= 3:
                        trigram_overlap = self.compute_ngram_overlap(behavior_tokens, span_tokens, n=3)
                        score += trigram_overlap * self.config.ngram_bonus
                
                candidates.append({
                    "start": start_idx,
                    "end": end_idx,
                    "score": score,
                    "overlap_tokens": list(overlap_tokens)
                })
                
                # Update the best span
                is_better = False
                if score > best_score:
                    is_better = True
                elif score == best_score:
                    # Tie-break rules
                    if self.config.tie_break_prefer_shorter:
                        # Prefer the shorter span
                        current_length = end_idx - start_idx + 1
                        best_length = best_range[1] - best_range[0] + 1
                        if current_length < best_length:
                            is_better = True
                        elif current_length == best_length and start_idx < best_range[0]:
                            # Same length; prefer the earlier one
                            is_better = True
                    else:
                        # Prefer the longer span
                        current_length = end_idx - start_idx + 1
                        best_length = best_range[1] - best_range[0] + 1
                        if current_length > best_length:
                            is_better = True
                        elif current_length == best_length and start_idx < best_range[0]:
                            is_better = True
                
                if is_better:
                    best_score = score
                    best_range = (start_idx, end_idx)
                    best_span = span_sentences
        
        # 5. Output
        best_span_sentences = best_span if best_span else []
        
        # Sort candidates for debugging
        candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
        top_k = candidates_sorted[:10]  # top-10 candidates
        
        debug_info = {
            "top_candidates": [
                {
                    "start": c["start"],
                    "end": c["end"],
                    "score": c["score"],
                    "overlap_tokens": c["overlap_tokens"][:10]  # Show only the first 10 tokens
                }
                for c in top_k
            ],
            "total_candidates": len(candidates),
            "behavior_tokens": behavior_tokens[:20],  # Show only the first 20 tokens
            "high_freq_tokens_count": len(high_freq_tokens)
        }
        
        return best_span_sentences, best_range, best_score, debug_info

