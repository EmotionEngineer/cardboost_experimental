"""r82_nomcon.py -- R82 Part 2 (H11): #5 permutation-invariant nominal Potts +
#6 error-correcting multiclass contrast rotation.

#5 (user plan 2026-11-13): for categorical / low-cardinality features
   (u_j <= 12 unique values, non-monotone level profile) the bins are ORDERED
   by the smoothed score G_b/(H_b+0.5) BEFORE potts_dp; levels return to the
   original bin indexing afterwards (labels permuted back).  Continuous
   features (u_j > 12) are NEVER permuted (guard: protects cal and continuous
   aps sensors).  Deploy: 0 bytes.

#6 (triage #6: "K-1 independent fits after a fixed orthogonal rotation, no
   shared partitions"): the OOF residual covariance C = R_OOF^T R_OOF/n is
   decomposed C = Q Lambda Q^T; in the final 1-D update pass potts_dp runs in
   the principal-contrast basis Q with INDEPENDENT segmentations per contrast
   (the K-th, ~zero-eigenvalue common direction is left unfused -- a common
   logit shift is invisible to softmax), then V <- V_rot Q^T.
"""
from __future__ import annotations

import numpy as np

from r71_universal import potts_dp


def _smooth_key(zc, H):
    """per-bin smoothed score G_b/(H_b+0.5); for multiclass rows: projection on
    the top principal component of the smoothed level matrix."""
    zc = np.asarray(zc, float)
    H = np.asarray(H, float)
    if zc.ndim == 1:
        return zc * H / (H + 0.5)
    Sm = zc * (H / (H + 0.5))[:, None]
    # top PC of rows (bins)
    if np.max(np.abs(Sm)) < 1e-15:
        return np.zeros(len(Sm))
    u, s, vt = np.linalg.svd(Sm, full_matrices=False)
    return Sm @ vt[0]


def _monotone(vals):
    if len(vals) <= 2:
        return True
    d = np.diff(vals)
    return bool(np.all(d >= -1e-15) or np.all(d <= 1e-15))


def nom_perm(zc, H, u_j):
    """#5: bin permutation by smoothed score, or None (feature stays ordered).
    Criteria (user): u_j <= 12 unique values AND non-monotone profile."""
    if u_j > 12:
        return None
    H = np.asarray(H, float)
    key = _smooth_key(zc, H)
    live = H > 1e-12
    if int(live.sum()) <= 2:
        return None
    if _monotone(key[live]):
        return None
    return np.argsort(key, kind="stable")


def _fuse_1d(z, H, kap):
    """single potts_dp fusion of one contrast column -> fused levels."""
    lab = potts_dp(np.asarray(z, float).copy(),
                   np.where(H > 0, H, 1e-9), kap)
    out = np.zeros(len(z), float)
    for seg in np.unique(lab):
        sel = lab == seg
        w = float(H[sel].sum())
        out[sel] = float((z[sel] * H[sel]).sum() / w) if w > 0 else 0.0
    out[H <= 1e-12] = 0.0
    return out


def contrast_Q(R_oof):
    """Q for #6: eigenvectors of the OOF error covariance, Lambda descending."""
    R = np.asarray(R_oof, float)
    C = R.T @ R / max(len(R), 1)
    lam, Q = np.linalg.eigh(C)
    return Q[:, ::-1]           # columns: descending eigenvalue


def contrast_fuse(C2, H, Q, kap, nom=False):
    """#6: independent per-contrast fusion in the Q basis, then V_rot Q^T.
    The K-th column (smallest Lambda, ~common direction) is left unfused
    (single global level).  nom=True additionally sorts each contrast column's
    bins by its own smoothed score before its potts_dp (nominal features --
    harmless for monotone profiles, order-free grouping otherwise)."""
    C2 = np.asarray(C2, float)
    H = np.asarray(H, float)
    Q = np.asarray(Q, float)
    A = C2 @ Q
    m, K = A.shape
    out = np.zeros_like(A)
    for c in range(K - 1):
        z = A[:, c]
        if nom:
            key = z * H / (H + 0.5)
            perm = np.argsort(key, kind="stable")
            fused = _fuse_1d(z[perm], H[perm], kap)
            out[:, c] = 0.0
            out[perm, c] = fused
        else:
            out[:, c] = _fuse_1d(z, H, kap)
    w = H
    gm = float((A[:, K - 1] * w).sum() / max(float(w.sum()), 1e-12))
    out[:, K - 1] = gm
    return out @ Q.T
