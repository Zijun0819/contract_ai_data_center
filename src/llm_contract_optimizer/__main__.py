"""Run the default case directly from VS Code or use the optional CLI arguments."""
from pathlib import Path
import sys

# VS Code's "Run Python File in Terminal" executes this file without a package
# context. Resolve the source directory from this file, independently of cwd.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_contract_optimizer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
