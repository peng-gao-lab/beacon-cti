#!/usr/bin/env python3
"""
Sentence splitter - Must be identical to CTITagger's SpanLocator.split_sentences
"""

import re
from typing import List

# Try to import NLTK (optional)
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    
    # Download necessary NLTK data if not available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
except ImportError:
    NLTK_AVAILABLE = False


# BEACON port: sentence splitter selection. BEACON_SENTENCE_SPLITTER=regex forces
# the built-in regex splitter (the behaviour of the reference runs, where the
# NLTK punkt_tab data was absent). Unset or "nltk" keeps the original
# "NLTK if available" behaviour. Stopwords/stemming are not affected.
import os as _beacon_os
USE_NLTK_SENTENCE_SPLITTER = NLTK_AVAILABLE and (
    _beacon_os.environ.get("BEACON_SENTENCE_SPLITTER", "").strip().lower() != "regex"
)

class SentenceSplitter:
    """Sentence splitter (identical to CTITagger's SpanLocator)"""
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into sentences (identical to CTITagger's SpanLocator.split_sentences)
        
        Args:
            text: Raw text
            
        Returns:
            List of sentences
        """
        # Use NLTK sentence tokenizer if available
        if USE_NLTK_SENTENCE_SPLITTER:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = SentenceSplitter._split_sentences_fallback(text)
        else:
            sentences = SentenceSplitter._split_sentences_fallback(text)
        
        # If splitting fails, return entire text as one sentence
        if not sentences:
            sentences = [text]
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def _split_sentences_fallback(text: str) -> List[str]:
        """
        Fallback sentence splitting method (without NLTK)
        Identical to CTITagger's SpanLocator._split_sentences_fallback
        """
        sentences = []
        # Simple rule: sentence-ending punctuation (. ? !) followed by whitespace
        pattern = r'([.!?])\s+'
        parts = re.split(pattern, text)
        
        # Reconstruct sentences
        current_sentence = ""
        for i, part in enumerate(parts):
            if part in '.!?':
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Add last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences
