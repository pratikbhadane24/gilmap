"""Repo-root pytest config: ensure the in-tree `gilmap/` package wins over
any partial site-packages install. Keeps `pytest` runnable straight after
`maturin develop` without depending on which files maturin chose to ship.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
