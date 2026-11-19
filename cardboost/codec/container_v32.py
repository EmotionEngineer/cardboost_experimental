"""Container v3.x binary codec: ``to_bytes`` / ``from_bytes``.

Thin wrapper over the certified engine codec (``r76_container31.emit_v31`` /
``decode_card_from_bytes``).  The wire format is the R83--R87 Container v3.1
(gzip-9 collapsed, `moved_train == 0` certified): tags 1-4 rays, 5 grid (p1),
6 bilinear (m2), 9 main-sum ray (msum), 10 coordination-of-missingness
(q_miss), 13/14 safe tiers.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .. import _vendor  # noqa: F401
import r76_container31 as C  # noqa: E402


GZIP_MAGIC = b"\x1f\x8b"


@dataclass
class WalletReport:
    raw_bytes: int
    stored_bytes: int
    used_gzip: bool
    moved_train: int


def emit(card, Xtr, budget: Optional[int] = 4096):
    """Emit (payload, info) with optional rd_squeeze under a hard budget."""
    import r83_3_wallet14 as W
    payload, info = C.emit_v31(card, Xtr)
    if budget is not None and info["bytes"] > budget:
        card, payload, info, _ = W.rd_squeeze(card, Xtr, budget)
    return payload, info


def to_bytes(card, Xtr=None, budget: Optional[int] = 4096) -> bytes:
    """Serialize a card to the Container v3.x bytes (min(raw, gzip-9))."""
    if Xtr is None:
        Xtr = np.zeros((1, max((len(card.cuts), 1))))
    payload, info = emit(card, Xtr, budget=budget)
    gz = gzip.compress(payload, 9)
    return gz if len(gz) < len(payload) else payload


def from_bytes(blob: bytes):
    """Decode Container bytes (gzip auto-detected) to a DecodedCard."""
    if blob[:2] == GZIP_MAGIC:
        blob = gzip.decompress(blob)
    return C.decode_card_from_bytes(blob)


def wallet(card, Xtr, budget: Optional[int] = 4096) -> WalletReport:
    payload, info = emit(card, Xtr, budget=budget)
    gz = gzip.compress(payload, 9)
    return WalletReport(raw_bytes=int(info["bytes"]),
                        stored_bytes=min(len(gz), len(payload)),
                        used_gzip=len(gz) < len(payload),
                        moved_train=int(info.get("moved", 0)))


def certify_decode(card, Xtr, Xte=None, budget: Optional[int] = 4096) -> dict:
    """decode(encode(card)) vs in-memory: max score diff and moved_train."""
    payload, info = emit(card, Xtr, budget=budget)
    dec = C.decode_card_from_bytes(payload)
    X = Xtr if Xte is None else Xte
    s1 = np.asarray(C.model_score(card, X), float)
    s2 = np.asarray(C.model_score(dec, X), float)
    return dict(wallet_bytes=int(info["bytes"]),
                max_diff=float(np.max(np.abs(s1 - s2))) if len(s1) else 0.0,
                moved_train=int(info.get("moved", 0)),
                dec_card=dec, payload=payload)
