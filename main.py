#!/usr/bin/env python3
"""Main entry point for Mistral NER training."""

import sys
from pathlib import Path

# Import and run the training script
sys.path.append(str(Path(__file__).parent))
from scripts.train import main

if __name__ == "__main__":
    main()