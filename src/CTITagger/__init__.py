"""
CTITagger - CTI report processing pipeline
"""

from .core.ctitagger import CTITagger
from .core.llm_provider import LLMProvider, create_llm_provider
from .core.behavior_extractor import AttackBehavior
from .core.span_locator import SpanLocatorConfig
from .core.ttp_proposer import TtpCandidate
from .core.ttp_verifier import VerifiedTtp, VerifiedResult

__version__ = "1.0.0"
__all__ = [
    "CTITagger",
    "LLMProvider",
    "create_llm_provider",
    "AttackBehavior",
    "SpanLocatorConfig",
    "TtpCandidate",
    "VerifiedTtp",
    "VerifiedResult"
]

