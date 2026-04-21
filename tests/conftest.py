"""Shared pytest configuration.

Adds the repository root to ``sys.path`` so the flat-layout modules
(``recorder``, ``config``, ``diarizer``, ...) can be imported by tests
without requiring an editable install.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
