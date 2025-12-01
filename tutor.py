#!/usr/bin/env python3
"""
CS Tutor CLI Entry Point.

Usage:
    python tutor.py explain "binary search tree" --level beginner
    python tutor.py practice --topic algorithms --count 3
    python tutor.py quiz --topic sorting
    python tutor.py grade --question-file q.txt --answer-file a.txt
    python tutor.py chat

Or install and run:
    pip install -e .
    tutor explain "binary search tree"
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli.app import main

if __name__ == "__main__":
    main()

