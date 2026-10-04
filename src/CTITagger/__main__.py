#!/usr/bin/env python3
"""
CTITagger module entry point
Supports invocation via python -m CTITagger

BEACON port: the original file inserted the CTITagger directory into sys.path
and imported the top-level module ``main``. The package is now imported
relatively so that it no longer shadows other components' modules.
"""

from .main import main

if __name__ == "__main__":
    main()
