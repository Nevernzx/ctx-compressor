"""Inspect public ModelScope frontend source to locate read-only search API."""
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
patterns = sys.argv[2:] or ['datasets', 'DatasetList']
for pattern in patterns:
    hits = [m.start() for m in re.finditer(re.escape(pattern), text)]
    print('PATTERN', pattern, 'COUNT', len(hits))
    for p in hits[:70]:
        print(text[max(0, p-180):p+350])
