"""QR-code export of the container bytes (model in a QR image).

Uses the optional ``qrcode`` package (``pip install cardboost[qr]``).  The QR
payload is the Container v3.x bytes themselves, so
``from_bytes(qr_readback_payload)`` rebuilds the model.  For models > ~2.9 KB
the container is gzipped first (gzip happens automatically in ``to_bytes``).
"""
from __future__ import annotations

import json
from typing import Optional


def to_qr(card, path: str, Xtr=None, budget: Optional[int] = 4096,
          box_size: int = 6, border: int = 4, meta: Optional[dict] = None) -> str:
    """Render the model QR to `path` (PNG/SVG by extension). Returns path."""
    try:
        import qrcode
    except ImportError as e:  # pragma: no cover
        raise ImportError("to_qr requires the optional dependency 'qrcode': "
                          "pip install cardboost[qr]") from e
    from .container_v32 import to_bytes
    blob = to_bytes(card, Xtr=Xtr, budget=budget)
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=box_size, border=border)
    qr.add_data(blob)
    qr.make(fit=True)
    img = qr.make_image()
    img.save(path)
    if meta:
        with open(path + ".meta.json", "w") as fh:
            json.dump(dict(meta, n_bytes=len(blob)), fh, indent=2)
    return path


def to_qr_svg(card, path: str, Xtr=None, budget: Optional[int] = 4096) -> str:
    """Render the model QR as a standalone SVG (no PNG dependency)."""
    to_qr(card, path if path.endswith(".svg") else path + ".svg",
          Xtr=Xtr, budget=budget)
    return path
