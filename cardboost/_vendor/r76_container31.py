"""r76_container31.py — corrected v3.1 container (Idea 12) for Stage 5.

Fixes over r76_stage5.emit_v3:
  * lattice bounds follow bin_idx's right-closed routing: x_left = max{x < cut},
    x_right = min{x >= cut}, acceptance float64(x_left) < t <= float64(x_right)
    -> 0 moved rows by construction;
  * dead-cut collapse PRUNES the cut stream (a cut is dead iff the quantized
    level is flat across it for every class), runs are stored over the merged
    bin grid + NaN bin;
  * atom states (p1/g4/m2) serialize into an ATOMS section (f64 cuts, f16 state).

Gates: 0 moved rows, |dLL| <= 0.001 (whole card incl. f16 atom state),
aps kappa0 container <= 7 KB gzip.
"""
import gzip
import struct

import numpy as np

import r76_mech as M
import r76_stage5 as T   # quantize_levels, collapse_runs


def _lattice_recut(x_sorted, cuts, use_f16=True):
    """Rebuild cut values under the routing lattice; returns (rec_cuts, codes, vals)."""
    rec, codes, vals = [], [], bytearray()
    prev = 0.0
    for cut in cuts:
        below = x_sorted[x_sorted < cut]
        above = x_sorted[x_sorted >= cut]
        xl = float(below[-1]) if len(below) else -np.inf
        xr = float(above[0]) if len(above) else np.inf
        ok = False
        if use_f16:
            d = np.float16(cut - prev)
            t = prev + float(np.float64(d))
            if float(xl) < t <= float(xr):
                vals += struct.pack("<e", float(d))
                codes.append(0)
                prev = t
                rec.append(t)
                ok = True
        if not ok:
            t32 = float(np.float32(cut))
            if float(xl) < t32 <= float(xr):
                vals += struct.pack("<f", t32)
                codes.append(1)
            else:
                t32 = float(cut)
                vals += struct.pack("<d", t32)
                codes.append(2)
            prev = t32
            rec.append(t32)
    return np.asarray(rec), codes, vals


def _pack_codes(codes):
    nb = (2 * len(codes) + 7) // 8
    tb = bytearray(nb)
    for c, t in enumerate(codes):
        tb[c // 4] |= (t & 3) << (2 * (c % 4))
    return bytes(tb)


USE_F32_ATOMS = True   # R79 step-3: False -> f16 atom states (ucuts keep _f32_floor)
COMPRESS_UCUTS = False  # R109 Formula 2 (frozen R87): lossless active-cut packing of
                        # 1D ray tables (g4): store only the S-1 boundaries where the
                        # fused level changes + S shelf levels; routing is bit-identical
                        # (bin_idx side='right', NaN keeps its own trailing bin).


def _state_dt():
    return np.float32 if USE_F32_ATOMS else np.float16


def _state_w():
    return 4 if USE_F32_ATOMS else 2


def _pack_active_cuts(ucuts, st):
    """R109 Formula 2: lossless packing of a 1D step table.

    ucuts: (nu,) sorted cuts; st: (nu+2, ...) levels -- rows 0..nu are the real
    regions of bin_idx(side='right'), row nu+1 is the NaN bin.  Returns
    (cuts', st') where cuts' keeps only the boundaries at which the routed
    level actually changes and st' holds the shelf levels (+ unchanged NaN
    row).  For every u in R and for NaN the routed level value is identical;
    packing is idempotent.  Layouts that do not match (nu+2 rows) pass through
    unchanged.
    """
    ucuts = np.asarray(ucuts, float)
    st = np.asarray(st)
    nu = len(ucuts)
    if nu == 0 or st.shape[0] != nu + 2:
        return ucuts, st
    d = np.unique(ucuts)                       # distinct sorted boundaries
    cnt = np.searchsorted(ucuts, d, side="right")  # level index above boundary
    shelves_idx = np.concatenate([[0], cnt])   # region level rows, in order
    lev = st[shelves_idx]
    if st.ndim == 1:
        keep = np.concatenate([[True], lev[1:] != lev[:-1]])
    else:
        keep = np.concatenate([[True], (lev[1:] != lev[:-1]).any(axis=1)])
    cuts_p = d[keep[1:]]
    lev_p = lev[keep]
    st_p = np.concatenate([lev_p, st[nu + 1:]], axis=0)
    return cuts_p, st_p


def _f32_floor(x):
    """f32 quantisation of CUT arrays, one-sided (toward -inf).

    R78: plain round-to-nearest f32 on ucuts flips every tied row whose u
    exactly equals a cut (ECDF ranks repeat), moving it across a level gap ->
    |dR2| ~ 4e-4 on a single card.  Flooring keeps all u == cut rows on their
    original side of the boundary ('right' searchsorted); only rows strictly
    inside the 1-ulp window can move.
    """
    v = np.asarray(x, float)
    q = np.asarray(v, np.float32)
    back = np.nextafter(q, np.float32(-np.inf))
    return np.where(q.astype(np.float64) > v, back, q)


def make_rank_grid(xs, n=96):
    """Fit-side twin of the decoded ECDF rank grid (R84 canary 1).

    Reproduces decode_atoms_from_bytes' knot arithmetic EXACTLY: positions =
    xs[np.linspace(0, norig-1, n).astype(int)] with the f32-floor first knot and
    f16 cumulative deltas; ranks = (np.linspace(0, norig-1, n) + 1)/norig (the
    decoder's float-linspace formula, NOT the integer knot positions); then
    dedup tied knots keeping the LAST of each block (post-block rank).
    Fitting g4 atoms against this grid makes fit == deploy at score level.
    """
    xs = np.asarray(xs, float)
    norig = len(xs)
    n = int(min(n, norig))
    idx = np.linspace(0, norig - 1, n).astype(int)
    g = xs[idx]
    out = [float(_f32_floor(g[:1])[0])]
    prev = out[0]
    for t in g[1:]:
        dv = np.float16(float(t) - prev)
        out.append(prev + float(np.float64(dv)))
        prev = out[-1]
    g = np.array(out)
    r = (np.linspace(0, norig - 1, n) + 1.0) / norig
    if len(g) > 1:
        last = np.ones(len(g), bool)
        last[:-1] = g[1:] != g[:-1]
        g, r = g[last], r[last]
    return g, r


def _atom_payloads(card, rank_grid=96):
    """Serialize atom states: per atom (type, j, k, f64 cuts, f16 state).

    R77 atoms (r77_mech) get their own tags: 4=ProjRay (g4 ray, ECDF rank grid
    shared per parent), 5=P1 (product-Potts rectangle), 6=M2 (bilinear on the
    mains' levels, pos-tail flag)."""
    import r77_mech as R77
    out = bytearray()
    # ---- shared ECDF rank-grid table (parents used by g4 atoms) ------------
    parents = {}
    parents_n = {}
    for a in card.atoms:
        if isinstance(a, R77.ProjRayAtom):
            for feat, xs in ((a.j, a.xs_j), (a.k, a.xs_k)):
                if feat not in parents:
                    parents[feat] = np.asarray(xs, float)
                    parents_n[feat] = int(getattr(a, "ecdf_n", {}).get(feat, len(xs)) \
                                          if isinstance(getattr(a, "ecdf_n", None), dict) \
                                          else getattr(a, "ecdf_n", 0) or len(xs))
    out += struct.pack("<B", len(parents))
    for feat, xs in parents.items():
        n = min(rank_grid, len(xs))
        idx = np.linspace(0, len(xs) - 1, n).astype(int)
        grid = xs[idx]
        # R83: lattice-style f16 cumulative DELTAS (first knot f32-floor) --
        # absolute f16 x-knots move by up to 0.03 at x~34 (eps), flipping the
        # ranks of hundreds of tied rows; delta encoding keeps positions to
        # ~1e-4 at the same byte cost (+2 B per parent).
        buf = bytearray(struct.pack("<f", float(_f32_floor(grid[:1])[0])))
        prev = float(_f32_floor(grid[:1])[0])
        for t in grid[1:]:
            dv = np.float16(float(t) - prev)
            buf += struct.pack("<e", float(dv))
            prev = prev + float(np.float64(dv))
        out += struct.pack("<BHI", feat, n, int(parents_n.get(feat, len(xs))))
        out += bytes(buf)
    out += struct.pack("<B", len(card.atoms))
    for a in card.atoms:
        if isinstance(a, R77.ProjRayAtom):
            out += struct.pack("<BBB", 4, a.j, a.k)
            out += struct.pack("<ff", float(a.a), float(a.b))
            out += struct.pack("<dddd", float(a.xmin_j), float(a.xmax_j),
                               float(a.xmin_k), float(a.xmax_k))
            st = a.Vc if a.Vc is not None else a.V
            uc = a.ucuts
            if COMPRESS_UCUTS and uc is not None and st is not None and len(uc):
                uc, st = _pack_active_cuts(uc, st)   # R109 Formula 2 (lossless)
            out += struct.pack("<H", len(uc) if uc is not None else 0)
            if uc is not None and len(uc):
                out += _f32_floor(uc).tobytes()
            if st is None:
                out += struct.pack("<HH", 0, 1)
            elif a.Vc is not None:
                out += struct.pack("<HH", st.shape[0], st.shape[1])
                out += np.asarray(st, _state_dt()).tobytes()
            else:
                out += struct.pack("<HH", st.shape[0], 1)
                out += np.asarray(st, _state_dt()).tobytes()
            marg = bool(getattr(a, "mj", None) is not None)
            out += struct.pack("<B", 1 if marg else 0)
            if marg:
                mj = np.asarray(a.mj, _state_dt())
                mk = np.asarray(a.mk, _state_dt())
                out += struct.pack("<HH", mj.size, mk.size)
                out += mj.tobytes() + mk.tobytes()
        elif isinstance(a, R77.P1Atom):
            out += struct.pack("<BBB", 5, a.j, a.k)
            out += struct.pack("<HH", len(a.cj), len(a.ck))
            out += np.asarray(a.cj, np.float64).tobytes()
            out += np.asarray(a.ck, np.float64).tobytes()
            out += struct.pack("<dddd", float(a.xs_j[0]), float(a.xs_j[-1]),
                               float(a.xs_k[0]), float(a.xs_k[-1]))
            if a.Vc is not None:
                out += struct.pack("<HHH", a.Vc.shape[0], a.Vc.shape[1], a.Vc.shape[2])
                out += np.asarray(a.Vc, _state_dt()).tobytes()
            else:
                out += struct.pack("<HHH", a.V.shape[0], a.V.shape[1], 1)
                out += np.asarray(a.V, _state_dt()).tobytes()
        elif isinstance(a, R77.M2Atom):
            out += struct.pack("<BBB", 6, a.j, a.k)
            out += struct.pack("<B", 1 if getattr(a, "pos", False) else 0)
            out += struct.pack("<HH", len(a.cj), len(a.ck))
            out += np.asarray(a.cj, np.float64).tobytes()
            out += np.asarray(a.ck, np.float64).tobytes()
            out += struct.pack("<HH", len(a.zj), len(a.zk))
            out += np.asarray(a.zj, _state_dt()).tobytes()
            out += np.asarray(a.zk, _state_dt()).tobytes()
            g = np.atleast_1d(np.asarray(a.gamma, float))
            out += struct.pack("<B", g.size) + g.tobytes()
        elif isinstance(a, R77.MSumAtom):
            # R85 Canary 3, tag 9 + R105 clamp: Main-Sum Ray -- zj/zk f16
            # (len = lcj+2/lck+2), cj/ck f32-at-fit, u_lo/u_hi f32 (value-space
            # clamp range), ucuts f32-floor, state _state_dt.
            out += struct.pack("<BBB", 9, a.j, a.k)
            out += struct.pack("<ff", float(a.u_lo), float(a.u_hi))
            out += struct.pack("<B", len(a.cj))
            out += np.asarray(a.cj, np.float32).tobytes()
            out += struct.pack("<B", len(a.ck))
            out += np.asarray(a.ck, np.float32).tobytes()
            out += np.asarray(a.zj, np.float16).tobytes()
            out += np.asarray(a.zk, np.float16).tobytes()
            st = a.Vc if a.Vc is not None else a.V
            out += struct.pack("<B", len(a.ucuts) if a.ucuts is not None else 0)
            if a.ucuts is not None and len(a.ucuts):
                out += _f32_floor(a.ucuts).tobytes()
            if a.Vc is not None:
                out += struct.pack("<BB", a.Vc.shape[0], a.Vc.shape[1])
                out += np.asarray(a.Vc, _state_dt()).tobytes()
            else:
                out += struct.pack("<BB", len(a.V) if a.V is not None else 0, 1)
                if a.V is not None:
                    out += np.asarray(a.V, _state_dt()).tobytes()
        elif type(a).__name__ == "QMissAtom":
            # R112 (frozen R87 Finalist 4), tag 10: qmiss -- w_j=|V_j[b_NaN]|
            # восстанавливается декодером из мейнов (0 B); u_lo/u_hi f32 +
            # активные разрезы f32 + полки (nu+2) _state_dt (~29-40 B).
            st = np.asarray(a.V if a.V is not None else np.zeros(0), float)
            uc = np.asarray(a.ucuts, float) if a.ucuts is not None else np.zeros(0)
            if len(uc) and st.shape[0] == len(uc) + 2:
                uc, st = _pack_active_cuts(uc, st)
            out += struct.pack("<BBB", 10, 0, 0)
            out += struct.pack("<ff", float(a.u_lo), float(a.u_hi))
            out += struct.pack("<B", min(len(uc), 255))
            if len(uc):
                out += _f32_floor(uc).tobytes()
            out += struct.pack("<B", len(st))
            if len(st):
                out += np.asarray(st, _state_dt()).tobytes()
        elif isinstance(a, M.RectAtom):
            out += struct.pack("<BBB", 1, a.j, a.k)
            out += struct.pack("<HH", len(a.cj), len(a.ck))
            out += np.asarray(a.cj, np.float64).tobytes()
            out += np.asarray(a.ck, np.float64).tobytes()
            st = a.Vc if a.Vc is not None else a.V
            out += struct.pack("<HHH", st.shape[0], st.shape[1],
                               st.shape[2] if st.ndim == 3 else 1)
            out += np.asarray(st, _state_dt()).tobytes()
        elif isinstance(a, M.GrammarAtom):
            out += struct.pack("<BBB", 2, a.j, a.k)
            for u in (a.u1, a.u2):
                out += struct.pack("<B", 1 if u.sign > 0 else 0)
                out += struct.pack("<HH", len(u.cj), len(u.ck))
                out += np.asarray(u.cj, np.float64).tobytes()
                out += np.asarray(u.ck, np.float64).tobytes()
                out += struct.pack("<H", len(u.ucuts))
                out += _f32_floor(u.ucuts).tobytes()
                st = u.Vc if u.Vc is not None else u.V
                out += struct.pack("<HH", st.shape[0], st.shape[1] if st.ndim == 2 else 1)
                out += np.asarray(st, _state_dt()).tobytes()
        elif isinstance(a, M.BilinearAtom):
            out += struct.pack("<BBB", 3, a.j, a.k)
            out += struct.pack("<HH", len(a.cj), len(a.ck))
            out += np.asarray(a.cj, np.float64).tobytes()
            out += np.asarray(a.ck, np.float64).tobytes()
            out += struct.pack("<HH", len(a.zj), len(a.zk))
            out += np.asarray(a.zj, _state_dt()).tobytes()
            out += np.asarray(a.zk, _state_dt()).tobytes()
            g = np.atleast_1d(np.asarray(a.gamma, np.float64))
            out += struct.pack("<B", g.size) + g.tobytes()
        elif type(a).__name__ in ("P1Swap2A", "P1Swap2B"):
            out += struct.pack("<BBB", 7 if type(a).__name__ == "P1Swap2A" else 8,
                               a.j, a.k)
            out += a.wire()
        else:
            raise TypeError(f"atom type {type(a)} not serializable")
    return bytes(out)


def emit_v31(card, Xtr, mode="int8", use_f16=True, collapse=True):
    kind = card.kind
    K = 1 if kind != "multiclass" else len(np.atleast_1d(card.b0))
    kind_id = {"regression": 0, "binary": 1, "multiclass": 2}[kind]
    p = len(card.cuts)
    out = bytearray()
    out += struct.pack("<BBBH", kind_id, K, 0, p)

    moved = 0
    b0_shift = np.zeros(K)   # R83: quantization mean drift absorbed into b0 (N9.1)
    feat_streams = []
    for j in range(p):
        cuts = np.asarray(card.cuts[j], float)
        V = np.asarray(card.V[j], float)
        if V.ndim == 1:
            V = V[:, None]
        xj = Xtr[:, j]
        fin = np.isfinite(xj)
        order = np.sort(xj[fin])
        # quantized level table over ORIGINAL bins
        qs = []
        for k in range(K):
            q, scale, q0, recon = T.quantize_levels(V[:, k], mode)
            qs.append(q)
        qs = np.stack(qs, axis=1)                       # (nb, K)
        nb = qs.shape[0]
        # dead cuts: flat quantized level across the cut for every class
        if collapse and nb == len(cuts) + 2:
            dead = np.all(qs[:-2] == qs[1:-1], axis=1)  # cut m between bins m and m+1
        else:
            dead = np.zeros(len(cuts), bool)
        surv = np.where(~dead)[0]
        rec_cuts, codes, vals = _lattice_recut(order, cuts[surv], use_f16=use_f16)
        # merged-bin map: new_bin(i) = #{surv cuts with original index < i}
        gmap = np.cumsum(np.concatenate([[0], (~dead).astype(int)]))
        # last REAL bin index is nb-2; NaN bin nb-1
        gmap = gmap[: nb - 1]                      # for real bins 0..nb-2
        gmap = np.concatenate([gmap, [len(surv) + 1]])   # NaN bin -> trailing bin
        # moved test: routing through reconstructed surviving cuts == group of old routing
        old_bins = M.bin_idx(xj[fin], cuts)
        new_bins = M.bin_idx(xj[fin], rec_cuts) if len(rec_cuts) else np.zeros(fin.sum(), int)
        moved += int(np.sum(gmap[old_bins] != new_bins))
        # level runs over the merged grid
        lv = bytearray()
        for k in range(K):
            qk = qs[:, k]
            newq = np.empty(len(surv) + 2, np.int16)
            for i in range(nb):
                newq[gmap[i]] = qk[i]
            runs = T.collapse_runs(newq) if collapse else [(i, int(newq[i]))
                                                           for i in range(len(newq))]
            q, scale, q0, _ = T.quantize_levels(V[:, k], mode)
            rec_merged = newq.astype(float) * scale
            b0_shift[k] += float(np.mean(V[:, k] - rec_merged[gmap]))
            lv += struct.pack("<fB", scale, q0 & 0xFF)
            if len(runs) < 255:
                lv += struct.pack("<B", len(runs))
            else:
                lv += struct.pack("<BH", 255, len(runs))
            span = 2 if len(newq) > 255 else 1
            for end, qv in runs:
                lv += (struct.pack("<H", end) if span == 2 else struct.pack("<B", end))
                if mode == "int16":
                    lv += struct.pack("<h", int(np.clip(qv, -32767, 32767)))
                elif mode == "int8":
                    lv += struct.pack("<b", int(np.clip(qv, -127, 127)))
                else:
                    lv += struct.pack("<B", int(np.clip(qv, 0, 255)))
        feat_streams.append((bool(len(cuts) or np.any(V != 0)),
                             struct.pack("<B", min(len(surv), 255))
                             + (struct.pack("<H", len(surv)) if len(surv) >= 255 else b"")
                             + _pack_codes(codes) + bytes(vals) + bytes(lv)))
    bitmap = bytearray((p + 7) // 8)
    for j, (live, _) in enumerate(feat_streams):
        if live:
            bitmap[j // 8] |= 1 << (j % 8)
    out += bytes(bitmap)
    for live, buf in feat_streams:
        if live:
            out += buf
    out += _atom_payloads(card)
    b0 = np.atleast_1d(np.asarray(card.b0, float)) + b0_shift
    for k in range(K):
        out += struct.pack("<d", float(b0[k]))
    out += struct.pack("<d", 1.0)
    # R83 Part 3: native IsoLink-OOF section (f16 knots, <= 16 nodes = <= 64 B)
    out += _isolink_bytes(card)
    raw = bytes(out)
    gz = gzip.compress(raw, 9)
    info = dict(raw=len(raw), gzip=len(gz), bytes=min(len(raw), len(gz)), moved=moved,
                mode=mode, f16=use_f16, collapse=collapse)
    return raw, info


def decode_v31(card, mode="int8", collapse=True):
    """Reconstruct the quantized model: pruned reconstructed cuts, merged levels,
    f16 atom states. Returns (cuts, V, b0_shift, atoms_rebuilt)."""
    new_cuts, newV, b0_shift = [], [], 0.0
    for j, cuts in enumerate(card.cuts):
        cuts = np.asarray(cuts, float)
        V = np.asarray(card.V[j], float)
        multi = V.ndim == 2
        Vk = V[:, None] if not multi else V
        nb, K = Vk.shape
        qs, recs = [], []
        shift_k = np.zeros(K)
        for k in range(K):
            q, scale, q0, recon = T.quantize_levels(Vk[:, k], mode)
            qs.append(q)
        qs = np.stack(qs, axis=1)
        if collapse and nb == len(cuts) + 2:
            dead = np.all(qs[:-2] == qs[1:-1], axis=1)
        else:
            dead = np.zeros(len(cuts), bool)
        surv = np.where(~dead)[0]
        order_dummy = None
        # same lattice reconstruction as the emitter (needs the training values)
        # -> the emitter's rec cuts are recomputed by the caller via card context;
        # for gate scoring we route with the PRUNED ORIGINAL cuts (identical bins)
        new_cuts.append(cuts[surv])
        gmap = np.cumsum(np.concatenate([[0], (~dead).astype(int)]))[: nb - 1]
        gmap = np.concatenate([gmap, [len(surv) + 1]])
        recs = np.empty((len(surv) + 2, K))
        for k in range(K):
            qk = qs[:, k]
            nq = np.empty(len(surv) + 2, np.int16)
            for i in range(nb):
                nq[gmap[i]] = qk[i]
            q, scale, q0, _ = T.quantize_levels(Vk[:, k], mode)
            if mode in ("int8", "int16"):
                rec = nq.astype(float) * scale
            else:
                rec = (nq - q0).astype(float) * scale
            shift_k[k] = float((Vk[:, k] - np.asarray(
                [rec[gmap[i]] for i in range(nb)])).mean())
            recs[:, k] = rec
        b0_shift = b0_shift + (shift_k if multi else float(shift_k[0]))
        newV.append(recs if multi else recs[:, 0])
    # atom states -> f16 round trip
    atoms = []
    for a in card.atoms:
        atoms.append(_atom_f16_roundtrip(a))
    return new_cuts, newV, b0_shift, atoms


def _atom_f16_roundtrip(a):
    """f32 state round trip (R78: container v3.1 states moved f16 -> f32)."""
    import copy
    import r77_mech as R77
    b = copy.deepcopy(a)
    if isinstance(b, R77.ProjRayAtom):
        if b.ucuts is not None:
            b.ucuts = _f32_floor(b.ucuts).astype(float)
        if b.V is not None:
            b.V = np.asarray(b.V, _state_dt()).astype(float)
        if b.Vc is not None:
            b.Vc = np.asarray(b.Vc, _state_dt()).astype(float)
        return b
    if isinstance(b, R77.P1Atom):
        if b.V is not None:
            b.V = np.asarray(b.V, _state_dt()).astype(float)
        if b.Vc is not None:
            b.Vc = np.asarray(b.Vc, _state_dt()).astype(float)
        return b
    if isinstance(b, R77.M2Atom):
        b.zj = np.asarray(b.zj, _state_dt()).astype(float)
        b.zk = np.asarray(b.zk, _state_dt()).astype(float)
        b.gamma = np.asarray(b.gamma, _state_dt()).astype(float)
        return b
    if type(b).__name__ in ("P1Swap2A", "P1Swap2B"):
        # R84 Canary 2 (kinds 7/8): состояния f16 по спеку (границы/каты/уровни),
        # f32-нормализация -- потеряless расширение уже-f16 значений.
        if getattr(b, "V", None) is not None:
            b.V = np.asarray(b.V, _state_dt()).astype(float)
        for nm in ("bnd_a", "bnd_b", "cj", "ck"):
            if getattr(b, nm, None) is not None:
                setattr(b, nm, np.asarray(getattr(b, nm), np.float16).astype(float))
        if getattr(b, "leaves", None):
            b.leaves = [(r1, r2, c1, c2, float(np.float16(v)))
                        for (r1, r2, c1, c2, v) in b.leaves]
        b.lmap = None
        return b
    if isinstance(b, M.RectAtom):
        b.V = np.asarray(b.V, np.float32).astype(float)
        if b.Vc is not None:
            b.Vc = np.asarray(b.Vc, np.float16).astype(float)
    elif isinstance(b, M.GrammarAtom):
        for u in (b.u1, b.u2):
            u.ucuts = np.asarray(u.ucuts, np.float16).astype(float)
            if u.V is not None:
                u.V = np.asarray(u.V, np.float16).astype(float)
            if u.Vc is not None:
                u.Vc = np.asarray(u.Vc, np.float16).astype(float)
    elif isinstance(b, M.BilinearAtom):
        b.zj = np.asarray(b.zj, np.float16).astype(float)
        b.zk = np.asarray(b.zk, np.float16).astype(float)
        b.gamma = np.asarray(np.float16(b.gamma)) if np.size(b.gamma) == 1 else \
            np.asarray(b.gamma, np.float16).astype(float)
    return b


# ----------------------------------------------------------------------------
# byte-level atom decode (decode(encode) bit-exact check; self-contained bytes)
# ----------------------------------------------------------------------------
def decode_atoms_from_bytes(payload, kind, n_classes=0, return_offset=False,
                             mains=None):
    """Rebuild atoms from _atom_payloads bytes.  g4 atoms get the f16 rank GRID
    as their ECDF support (coarse step ECDF), so encode(decode(encode(card)))
    reproduces the same bytes.  return_offset=True -> (atoms, consumed)."""
    import r77_mech as R77
    p = memoryview(payload)
    off = 0

    def take(n):
        nonlocal off
        b = bytes(p[off:off + n])
        off += n
        return b

    npar = struct.unpack("<B", take(1))[0]
    grids = {}
    for _ in range(npar):
        feat, n, norig = struct.unpack("<BHI", take(7))
        g = np.empty(n)
        g[0] = struct.unpack("<f", take(4))[0]
        prev = g[0]
        for i in range(1, n):
            dv = struct.unpack("<e", take(2))[0]
            g[i] = prev + float(np.float64(dv))
            prev = g[i]
        idx = np.linspace(0, norig - 1, n)
        g_all = g
        r_all = (idx + 1.0) / norig
        # R83 #14: dedup tied knots keeping the LAST of each duplicate block.
        # A train-value block of mass m makes the ECDF jump at one x; the old
        # f16 grid kept the block's INNER ranks, so np.interp at the tie served
        # the pre-block rank (bias -m per row of the block; -0.1 R^2 on the
        # geo-heavy cal_spat f2).  Keeping the last knot gives the post-block
        # rank -- the trained searchsorted semantics -- at zero extra bytes.
        if len(g_all) > 1:
            last = np.ones(len(g_all), bool)
            last[:-1] = g_all[1:] != g_all[:-1]
            g_all, r_all = g_all[last], r_all[last]
        grids[feat] = (g_all, r_all, norig)
    nat = struct.unpack("<B", take(1))[0]
    atoms = []
    for _ in range(nat):
        tag, j, k = struct.unpack("<BBB", take(3))
        if tag == 4:
            a, b = struct.unpack("<ff", take(8))
            x0, x1, x2, x3 = struct.unpack("<dddd", take(32))
            nu = struct.unpack("<H", take(2))[0]
            ucuts = np.frombuffer(take(4 * nu), np.float32).astype(float) if nu else np.array([])
            nr, nc = struct.unpack("<HH", take(4))
            st = np.frombuffer(take(_state_w() * nr * nc), _state_dt()).astype(float)
            at = R77.ProjRayAtom(j, k, a, b, grids[j][0], grids[k][0])
            at.rank_interp = (grids[j][0], grids[j][1], grids[k][0], grids[k][1])
            at.ecdf_n = {j: grids[j][2], k: grids[k][2]}
            at.xmin_j, at.xmax_j, at.xmin_k, at.xmax_k = x0, x1, x2, x3
            at.ucuts = ucuts
            if nc > 1:
                at.Vc = st.reshape(nr, nc)
                at.V = None
            else:
                at.V = st.reshape(nr)
            flag = struct.unpack("<B", take(1))[0]
            if flag:
                n1, n2 = struct.unpack("<HH", take(4))
                mj = np.frombuffer(take(_state_w() * n1), _state_dt()).astype(float)
                mk = np.frombuffer(take(_state_w() * n2), _state_dt()).astype(float)
                at.mj = mj
                at.mk = mk
            at.ensure(kind, n_classes)
            atoms.append(at)
        elif tag == 5:
            lcj, lck = struct.unpack("<HH", take(4))
            cj = np.frombuffer(take(8 * lcj), np.float64).copy()
            ck = np.frombuffer(take(8 * lck), np.float64).copy()
            bx0, bx1, bx2, bx3 = struct.unpack("<dddd", take(32))
            r0, c0, k0 = struct.unpack("<HHH", take(6))
            st = np.frombuffer(take(_state_w() * r0 * c0 * k0), _state_dt()).astype(float)
            xsj = np.array([bx0, bx1])
            xsk = np.array([bx2, bx3])
            at = R77.P1Atom(j, k, cj, ck, xsj, xsk)
            if k0 > 1:
                at.Vc = st.reshape(r0, c0, k0)
                at.V = None
            else:
                at.V = st.reshape(r0, c0)
            at.ensure(kind, n_classes)
            atoms.append(at)
        elif tag == 6:
            pos = struct.unpack("<B", take(1))[0]
            lcj, lck = struct.unpack("<HH", take(4))
            cj = np.frombuffer(take(8 * lcj), np.float64).copy()
            ck = np.frombuffer(take(8 * lck), np.float64).copy()
            nzj, nzk = struct.unpack("<HH", take(4))
            zj = np.frombuffer(take(_state_w() * nzj), _state_dt()).astype(float)
            zk = np.frombuffer(take(_state_w() * nzk), _state_dt()).astype(float)
            gs = struct.unpack("<B", take(1))[0]
            gamma = np.frombuffer(take(8 * gs), np.float64).copy()
            at = R77.M2Atom(j, k, zj, zk, pos=bool(pos))
            at.gamma = float(gamma[0]) if gamma.size == 1 else gamma
            at.set_cuts(cj, ck)
            at.ensure(kind, n_classes)
            atoms.append(at)
        elif tag in (7, 8):
            import r84_p1swap as _PS
            cls = _PS.P1Swap2A if tag == 7 else _PS.P1Swap2B
            at, used = cls.unwire(j, k, p[off:])
            off += used
            at.ensure(kind, n_classes)
            atoms.append(at)
        elif tag == 9:
            u_lo, u_hi = struct.unpack("<ff", take(8))
            lcj = struct.unpack("<B", take(1))[0]
            cj = np.frombuffer(take(4 * lcj), np.float32).astype(float)
            lck = struct.unpack("<B", take(1))[0]
            ck = np.frombuffer(take(4 * lck), np.float32).astype(float)
            zj = np.frombuffer(take(2 * (lcj + 2)), np.float16).astype(float)
            zk = np.frombuffer(take(2 * (lck + 2)), np.float16).astype(float)
            nu = struct.unpack("<B", take(1))[0]
            ucuts = np.frombuffer(take(4 * nu), np.float32).astype(float) if nu else np.zeros(0)
            nr, nc = struct.unpack("<BB", take(2))
            st = np.frombuffer(take(_state_w() * nr * nc), _state_dt()).astype(float) \
                if nr * nc else np.zeros(0)
            at = R77.MSumAtom(j, k, zj, zk, cj, ck, u_lo, u_hi)
            at.ucuts = ucuts
            if nc > 1:
                at.Vc = st.reshape(nr, nc)
                at.V = None
            elif nr > 0:
                at.V = st.reshape(nr)
            at.ensure(kind, n_classes)
            atoms.append(at)
        elif tag == 10:
            # R112 (frozen R87): qmiss -- w из декодированных мейнов (0 B на w)
            import r87_qmiss as QMiss
            u_lo, u_hi = struct.unpack("<ff", take(8))
            nu = struct.unpack("<B", take(1))[0]
            ucuts = np.frombuffer(take(4 * nu), np.float32).astype(float) if nu else np.zeros(0)
            ns = struct.unpack("<B", take(1))[0]
            st10 = np.frombuffer(take(_state_w() * ns), _state_dt()).astype(float) if ns else np.zeros(0)
            if mains is None:
                raise ValueError("tag 10 (qmiss) needs mains for w reconstruction")
            cuts_all, V_all = mains
            w = np.zeros(len(V_all))
            for jj in range(len(V_all)):
                Vj = np.asarray(V_all[jj], float)
                cj_ = np.asarray(cuts_all[jj], float)
                if Vj.ndim == 1 and len(Vj) == len(cj_) + 2:
                    w[jj] = abs(float(Vj[len(cj_) + 1]))
            at = QMiss.QMissAtom(w, u_lo, u_hi)
            at.ucuts = ucuts
            at.V = st10 if ns else None
            at.ensure(kind, n_classes)
            atoms.append(at)
        elif tag == 1:
            raise NotImplementedError("r76 RectAtom byte-decode")
        elif tag == 2:
            raise NotImplementedError("r76 GrammarAtom byte-decode")
        elif tag == 3:
            raise NotImplementedError("r76 BilinearAtom byte-decode")
        else:
            raise ValueError(tag)
    if return_offset:
        return atoms, off
    return atoms


# ----------------------------------------------------------------------------
# R83 Part 3: IsoLink section + full bytewise card decode (fresh-process cert)
# ----------------------------------------------------------------------------
def apply_isolink(iso, eta):
    """Piecewise-linear monotone link g(eta) with clamped extrapolation.
    Canonical implementation; r83_ceil.iso_apply delegates here.  Knots are
    float16 arrays (fitted f16, R83 spec) so fit-side and decode-side compute
    on bit-identical nodes."""
    if iso is None:
        return np.asarray(eta, float).copy()
    kx, kv = iso
    if len(kx) < 2:
        return np.asarray(eta, float).copy()
    out = np.interp(eta, kx, kv)
    out = np.minimum(np.maximum(out, kv.min()), kv.max())
    return out


def _isolink_bytes(card):
    iso = getattr(card, "isolink", None)
    if iso is None:
        return struct.pack("<B", 0)
    kx, kv = iso
    n = len(kx)
    if n > 16:
        raise ValueError("IsoLink knot budget is 16")
    return (struct.pack("<BB", 1, n)
            + np.asarray(kx, np.float16).tobytes()
            + np.asarray(kv, np.float16).tobytes())


class DecodedCard:
    """Container-form model rebuilt from bytes: mains + atoms (+ optional
    IsoLink held as .isolink).  .score mirrors JointCard.score arithmetic
    order exactly, so decode-side and normalize-side scores are bit-identical."""

    def __init__(self, kind, cuts, V, b0, atoms, isolink=None):
        self.kind = kind
        self.cuts = cuts
        self.V = V
        self.b0 = b0
        self.atoms = list(atoms)
        self.isolink = isolink
        self.meta = {}

    def score(self, X):
        X = np.asarray(X, float)
        n = len(X)
        if self.kind == "multiclass":
            K = len(self.b0)
            eta = np.tile(self.b0, (n, 1))
        else:
            eta = np.full(n, float(self.b0))
        for j in range(len(self.cuts)):
            eta = eta + self.V[j][M.bin_idx(X[:, j], self.cuts[j])]
        for a in self.atoms:
            eta = eta + a.contrib(X)
        return eta


def model_score(card, X):
    """Deployed prediction: IsoLink g(eta) on top of the additive score for
    kind == 'reg' cards carrying .isolink (regression only by construction)."""
    s = card.score(X)
    iso = getattr(card, "isolink", None)
    if iso is None:
        return s
    return apply_isolink(iso, s)


def decode_card_from_bytes(payload):
    """Full decode(encode(card)): mains + b0 + atoms + IsoLink from raw bytes.
    Returns a DecodedCard.  Pure function of the payload (fresh-process safe)."""
    p = memoryview(payload)
    off = 0

    def take(nbytes):
        nonlocal off
        b = bytes(p[off:off + nbytes])
        off += nbytes
        return b

    kind_id, K, _res, pp = struct.unpack("<BBBH", take(5))
    kind = {0: "regression", 1: "binary", 2: "multiclass"}[kind_id]
    nbm = (pp + 7) // 8
    bitmap = take(nbm)
    cuts_all, V_all = [], []
    for j in range(pp):
        live = (bitmap[j // 8] >> (j % 8)) & 1
        if not live:
            cuts_all.append(np.array([]))
            V_all.append(np.zeros((2, K)) if kind == "multiclass" else np.zeros(2))
            continue
        b0 = take(1)[0]
        # stream header: u8 min(n,255); if that byte == 255 -> u16 true n follows
        if b0 == 255:
            n = struct.unpack("<H", take(2))[0]
        else:
            n = b0
        ncodes = (2 * n + 7) // 8
        codeb = take(ncodes)
        codes = []
        for c in range(n):
            codes.append((codeb[c // 4] >> (2 * (c % 4))) & 3)
        rec = []
        prev = 0.0
        for c in codes:
            if c == 0:
                d = struct.unpack("<e", take(2))[0]
                t = prev + float(np.float64(np.float16(d)))
                prev = t
            elif c == 1:
                t = struct.unpack("<f", take(4))[0]
                prev = t
            else:
                t = struct.unpack("<d", take(8))[0]
                prev = t
            rec.append(t)
        rec_cuts = np.asarray(rec)
        nlev = n + 2
        span = 2 if nlev > 255 else 1
        Vk = np.zeros((nlev, K))
        for k in range(K):
            scale, q0 = struct.unpack("<fB", take(5))
            nb_run = take(1)[0]
            if nb_run == 255:
                nb_run = struct.unpack("<H", take(2))[0]
            newq = np.empty(nlev, float)
            start = 0
            for _r in range(nb_run):
                end = struct.unpack("<H" if span == 2 else "<B",
                                    take(2 if span == 2 else 1))[0]
                qv = struct.unpack("<b", take(1))[0]   # deployed mode = int8
                newq[start:end + 1] = qv
                start = end + 1
            Vk[:, k] = newq * float(scale)
        cuts_all.append(rec_cuts)
        V_all.append(Vk if kind == "multiclass" else Vk[:, 0])
    atoms, aoff = decode_atoms_from_bytes(bytes(p[off:]), kind, K if kind == "multiclass" else 0,
                                          return_offset=True,
                                          mains=(cuts_all, V_all))
    off += aoff
    b0v = [struct.unpack("<d", take(8))[0] for _ in range(K)]
    _trail = struct.unpack("<d", take(8))[0]
    has_iso = take(1)[0]
    iso = None
    if has_iso:
        n = take(1)[0]
        xk = np.frombuffer(take(2 * n), np.float16).copy()
        kv = np.frombuffer(take(2 * n), np.float16).copy()
        iso = (xk, kv)
    b0_arr = np.asarray(b0v) if kind == "multiclass" else float(b0v[0])
    return DecodedCard(kind, cuts_all, V_all, b0_arr, atoms, iso)


def normalize_v31(card, Xtr, mode="int8", use_f16=True, collapse=True):
    """Container-form deploy card: exactly what decode(emit(card)) yields,
    built in-process.  Returns (DecodedCard, info).  The serve-side in-memory
    model IS this object, so a fresh-process decode reproduces it bit-for-bit
    (max|score diff| == 0.0 by construction, cross-process certified)."""
    payload, info = emit_v31(card, Xtr, mode=mode, use_f16=use_f16, collapse=collapse)
    return decode_card_from_bytes(payload), info


def rd_sensitivity(card, Xtr, ytr):
    """Per-feature table sensitivity sum_b H_b * dV_b^2 (rate-distortion #14):
    H_b = routed train mass in bin b, dV_b = level gap to the neighbouring bin
    (largest of class dims).  Low score -> cheapest to coarsen."""
    out = []
    ytr = np.asarray(ytr, float)
    for j, cuts in enumerate(card.cuts):
        b = M.bin_idx(Xtr[:, j], np.asarray(cuts, float))
        nb = len(np.asarray(card.V[j], float).reshape(len(np.asarray(card.V[j])), -1))
        H = np.bincount(b, minlength=nb).astype(float)
        V = np.asarray(card.V[j], float)
        if V.ndim == 1:
            V = V[:, None]
        dV = np.max(np.abs(np.diff(V, axis=0)), axis=1) if len(V) > 1 else np.array([0.0])
        dV2 = np.concatenate([dV, [0.0]])
        out.append((float(np.sum(H * dV2 ** 2)), j))
    return sorted(out)

