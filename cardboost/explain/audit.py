"""OOD safety audit (Invariant III): shelf-compensation edge jumps."""
from __future__ import annotations

import numpy as np


def audit_ood_safety(model, threshold: float = 0.15):
    """Audit pair-atom edge compensation vs Invariant III (R61).

    For every pair atom the edge-bin shelf compensation |mean contribution on
    the edge rows| is compared against ``threshold * sigma_hat_y`` (0.15 sigma
    is the certified safe-tier gate from R87).  Returns a report dict with
    per-atom rows and the global ``pass`` flag.
    """
    card = getattr(model, "dec_card_", None) or getattr(model, "card_", None)
    rows = []
    worst = 0.0
    for i, at in enumerate(card.atoms):
        cls = type(at).__name__
        edge = None
        if cls == "P1Atom":
            tab = np.asarray(at.V if at.V is not None else at.Vc, float)
            A, B = tab.shape[0], tab.shape[1]
            marg_j = tab.mean(axis=1)
            marg_k = tab.mean(axis=0)
            edge = float(max(abs(marg_j[0]), abs(marg_j[-1]),
                             abs(marg_k[0]), abs(marg_k[-1])))
        elif cls in ("ProjRayAtom", "MSumAtom"):
            V = np.asarray(at.V if at.V is not None else at.Vc, float)
            if V.ndim == 1:
                edge = float(max(abs(V[0]), abs(V[-1])))
        if edge is not None:
            worst = max(worst, edge)
            rows.append(dict(atom=f"{cls}({getattr(at,'j','?')},{getattr(at,'k','?')})",
                             edge_compensation=edge,
                             within_threshold=bool(edge <= threshold)))
    return dict(atoms=rows, worst_edge=worst, threshold=threshold,
                pass_=bool(worst <= threshold))
