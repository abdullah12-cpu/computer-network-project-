"""Save everything a script prints to a report file as well as the terminal."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


@contextmanager
def tee_stdout(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        old = sys.stdout
        sys.stdout = _Tee(old, fh)
        try:
            yield
        finally:
            sys.stdout = old
    print(f"report saved to {path}")
