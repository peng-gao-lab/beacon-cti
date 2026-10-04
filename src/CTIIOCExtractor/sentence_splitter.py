#!/usr/bin/env python3
"""
Sentence splitter
Based on the ctitagger implementation
"""

import re
from typing import List

# Try to import NLTK (optional)
try:
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    
    # Download the required NLTK data (if not already downloaded)
    try:
        import nltk
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
    """Sentence splitter"""
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """
        Split text into a list of sentences (based on the ctitagger implementation)
        
        Args:
            text: raw text
        
        Returns:
            List of sentences
        """
        # Use NLTK's sentence tokenizer (if available)
        if USE_NLTK_SENTENCE_SPLITTER:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = SentenceSplitter._split_sentences_fallback(text)
        else:
            sentences = SentenceSplitter._split_sentences_fallback(text)
        
        # If splitting failed, at least return the whole text as a single sentence
        if not sentences:
            sentences = [text]
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def _split_sentences_fallback(text: str) -> List[str]:
        """Fallback sentence splitting method (without NLTK)"""
        sentences = []
        # Use a simple rule: sentence-ending punctuation (. ? !) followed by whitespace or a newline
        pattern = r'([.!?])\s+'
        parts = re.split(pattern, text)
        
        # Reassemble sentences
        current_sentence = ""
        for i, part in enumerate(parts):
            if part in '.!?':
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                current_sentence += part
        
        # Append the last sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return sentences

