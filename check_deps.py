#!/usr/bin/env python3
"""Check available dependencies."""

import sys

dependencies = [
    "torch",
    "transformers", 
    "peft",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "evaluate",
    "seqeval",
    "numpy",
    "pyyaml",
    "dotenv",
    "wandb",
    "fastapi",
    "uvicorn",
    "pydantic"
]

print("Checking dependencies:")
print("-" * 40)

available = []
missing = []

for dep in dependencies:
    try:
        if dep == "pyyaml":
            __import__("yaml")
        elif dep == "dotenv":
            __import__("dotenv")
        else:
            __import__(dep)
        print(f"✓ {dep}")
        available.append(dep)
    except ImportError:
        print(f"✗ {dep}")
        missing.append(dep)

print("-" * 40)
print(f"Available: {len(available)}")
print(f"Missing: {len(missing)}")

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("\nTo install missing dependencies:")
    print("uv pip install pyyaml python-dotenv wandb")