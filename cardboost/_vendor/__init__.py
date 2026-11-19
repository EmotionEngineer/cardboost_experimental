"""Battle-tested CardBoost engine (R71--R87 lineage), vendored verbatim.

The modules in this package are the *unaltered* research engine that produced
every metric anchor in `golden/canonical_r87.json` (rounds R71--R87).  They are
kept byte-identical to the certified sources so that `cardboost` reproduces the
canonical 16-fold table bit-exactly; all packaging, API and export code lives
outside this directory and only *wraps* these modules.

Import side effect: this directory is prepended to ``sys.path`` so that the
vendored modules can import each other by their historical flat names
(``import r71_universal``, ``import r77_stage`` ...).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__all__ = []
