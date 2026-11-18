"""Make the project importable when pytest is invoked as a console script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
