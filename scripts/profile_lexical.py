#!/usr/bin/env python3
"""CLI runner for Stage 1.3 Lexical Dynamics & Vocabulary Overlap Profiling (EDA).

Thin wrapper delegating directly to the unified RetTune CLI:
    python benchmark.py --stage eda ...
"""

from pathlib import Path
import sys
from typing import Optional, Sequence

# Ensure src directory is available on sys.path for direct script execution
try:
    from rettune.cli import main as cli_main
except ModuleNotFoundError:
    src_dir = Path(__file__).resolve().parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from rettune.cli import main as cli_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI execution entrypoint delegating to benchmark --stage eda."""
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    # Prepend --stage eda if not already explicitly specified
    if "--stage" not in raw_args:
        full_args = ["--stage", "eda"] + raw_args
    else:
        full_args = raw_args
    return cli_main(full_args)


if __name__ == "__main__":
    sys.exit(main())
