"""Wheel smoke test. CI installs the built wheel, then runs this."""
import os
import sys
import tempfile

tmp = tempfile.mkdtemp()
with open(os.path.join(tmp, "sm.py"), "w") as f:
    f.write("def sq(x):\n    return x * x\n")
sys.path.insert(0, tmp)

import gilmap
import sm

result = gilmap.map(sm.sq, [1, 2, 3, 4])
assert result == [1, 4, 9, 16], f"unexpected result: {result!r}"
print("smoke ok")
