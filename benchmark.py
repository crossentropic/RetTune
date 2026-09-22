#!/usr/bin/env python3
"""RetTune: Reproducible Hybrid Search Benchmark Testbed (Layer 1 RAG).

Root execution entrypoint delegating to src/cli.py.
Enforces strict zero-network execution offline unless --setup is explicitly specified.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rettune.cli import main

if __name__ == "__main__":
    sys.exit(main())
