"""BEACON: a thin YAML-driven driver around the five CTI components.

The five components (CTIHighLevelExtractor, CTITagger, CTIIOCExtractor,
CTIAugmentor, CTIAlignmentor) keep their own packages, prompts, algorithms and
command-line interfaces. This package only reads a YAML configuration,
enumerates inputs, runs the component entry points as subprocesses, writes the
lightweight "extract" views and records what happened.
"""

__version__ = "0.1.0"
