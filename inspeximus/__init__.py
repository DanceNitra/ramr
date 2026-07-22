"""inspeximus — agent memory core (zero-dependency), PINNED SNAPSHOT vendored for reproducibility.

This is a frozen copy (v0.6.10) of the inspeximus library, vendored so the RAMR numbers in this repo
reproduce against exactly the code that produced them. The maintained library is newer and installable
with `pip install inspeximus`; this pinned copy is intentionally not updated in lock-step with it.
"""
from .core import Inspeximus  # noqa: F401
