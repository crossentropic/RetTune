#!/usr/bin/env python3
"""RetTune: Reproducible Hybrid Search Benchmark Testbed (Layer 1 RAG).

Root execution entrypoint delegating to rettune.cli.main.
Supports both installed package execution and zero-install checkout execution.
"""

import sys

try:
    from rettune.cli import main
except ModuleNotFoundError:
    from pathlib import Path
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from rettune.cli import main

if __name__ == "__main__":
    sys.exit(main())
