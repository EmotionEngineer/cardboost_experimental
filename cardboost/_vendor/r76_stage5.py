#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r76_stage5.py — Idea 11 (kappa continuation) + Idea 12 (int8/uint8 quantized container).

Idea 11: kappa continuation inside the 24/48-pass ctl budget —
  * high->1: kappa_mult = 4.0 for passes 1-6, then 1.00 for passes 7-24;
  * low ->1: kappa_mult = 0.25 -> 0.50 -> 1.00 over the first 8 passes, then 1.00.
  (No unpenalized joint OLS refit anywhere — the Stage-0 lesson.)

Idea 12: quantized printing —
  * levels quantised to an int8 grid (v = scale*q) or a uint8 grid (v = scale*(q-q0),
    the grid always contains 0) per feature per class;
  * post-quant dead-cut collapse: adjacent segments with equal q merge;
  * cut stream = N10 f16 deltas with f32 fallback and f64 escape, each accepted only
    while float64(x_left) < t <= float64(x_right] keeps every train row's routing;
  * quantisation shifts are absorbed into the tail f64 b0[K] (N9.1);
  * container = v3' layout (header kind/K/flags/n_feat + live-feature bitmap + per-feature
    cut stream + run-length quantised level table + tail b0/T); bytes = min(raw, gzip-9).

Gates (pre-registered): 0 moved rows; |dLL| <= 0.001 held-out; aps kappa0 <= 7 KB.

Usage: python3 r76_stage5.py --panel aps [--out stage5.jsonl]
"""
from __future__ import annotations

import argparse
import gzip
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r76_lib as L
import r76_stage31 as S
import r71_universal as r71
from r71_universal import bin_idx


# ----------------------------------------------------------------------------
# Idea 11: kappa schedules
# ----------------------------------------------------------------------------
def sched_high_to1(it):
    return 4.0 if it < 6 else 1.0


def sched_low_to1(it):
    if it < 2:
        return 0.25
    if it < 5:
        return 0.50
    return 1.0


# ----------------------------------------------------------------------------
# Idea 12: quantized container
# ----------------------------------------------------------------------------
def _f16_ok(x_left, t, x_right):
    """Lattice acceptance: the reconstructed cut must keep every train row's routing."""
    return float(x_left) < t <= float(x_right)


def quantize_levels(V, mode):
    """Per-level quantisation.  Returns (q array int, scale, q0, recon)."""
    V = np.asarray(V, float)
    if mode == "int16":
        scale = max(float(np.max(np.abs(V))) / 32767.0, 1e-12) if V.size else 1.0
        q = np.clip(np.round(V / scale), -32767, 32767).astype(np.int16)
        recon = q.astype(float) * scale
        return q, scale, 0, recon
    if mode == "int8":
        scale = max(float(np.max(np.abs(V))) / 127.0, 1e-12) if V.size else 1.0
        q = np.clip(np.round(V / scale), -127, 127).astype(np.int16)
        recon = q.astype(float) * scale
        return q, scale, 0, recon
    # uint8 grid with q0 anchor, grid always contains 0 (r74 N9.1)
    vmin = float(np.min(V)) if V.size else 0.0
    vmax = float(np.max(V)) if V.size else 0.0
    lo = min(vmin, 0.0)
    hi = max(vmax, 0.0)
    scale = max((hi - lo) / 255.0, 1e-12)
    q0 = int(round(-lo / scale))
    q = np.clip(np.round(V / scale) + q0, 0, 255).astype(np.int16)
    recon = (q - q0).astype(float) * scale
    return q, scale, q0, recon


def collapse_runs(q):
    """Dead-cut collapse: (end_index, q) runs of the per-bin quantised level array."""
    runs = []
    start = 0
    for i in range(1, len(q) + 1):
        if i == len(q) or q[i] != q[i - 1]:
            runs.append((i - 1, int(q[i - 1])))
            start = i
    return runs


def emit_v3(card, Xtr, mode="int8", use_f16=True, collapse=True):
    """Serialize a JointCard into the v3'-like container.  Returns (buf, info)."""
    kind = card.kind
    K = 1 if kind != "multiclass" else len(np.atleast_1d(card.b0))
    kind_id = {"regression": 0, "binary": 1, "multiclass": 2}[kind]
    p = len(card.cuts)
    out = bytearray()
    out += struct.pack("<BBBH", kind_id, K, 0, p)

    # per-feature content prep
    feat_streams = []
    moved = 0
    for j in range(p):
        cuts = np.asarray(card.cuts[j], float)
        V = np.asarray(card.V[j], float)
        if V.ndim == 1:
            V = V[:, None]
        xj = Xtr[:, j]
        fin = np.isfinite(xj)
        # --- cut stream ---------------------------------------------------
        typebits = []
        vals = bytearray()
        prev = 0.0
        # routing lattice bounds per cut: max value <= cut  vs  min value > cut
        order = np.sort(xj[fin])
        for c, cut in enumerate(cuts):
            left = order[order <= cut]
            right = order[order > cut]
            xl = float(left[-1]) if len(left) else -np.inf
            xr = float(right[0]) if len(right) else np.inf
            ok16 = False
            if use_f16:
                d = np.float16(cut - prev)
                t = prev + float(np.float64(d))
                if _f16_ok(xl, t, xr):
                    vals += struct.pack("<e", float(d))
                    typebits.append(0)
                    ok16 = True
                    prev = t
            if not ok16:
                t32 = np.float32(cut)
                if _f16_ok(xl, float(t32), xr):
                    vals += struct.pack("<f", float(t32))
                    typebits.append(1)
                    prev = float(t32)
                else:
                    vals += struct.pack("<d", float(cut))
                    typebits.append(2)
                    prev = float(cut)
        # pack 2-bit type codes
        nbits = 2 * len(cuts)
        nbytes = (nbits + 7) // 8
        tb = bytearray(nbytes)
        for c, t in enumerate(typebits):
            tb[c // 4] |= (t & 3) << (2 * (c % 4))
        # --- level stream -------------------------------------------------
        lv = bytearray()
        qtables = []
        for k in range(K):
            Vk = V[:, k]
            q, scale, q0, recon = quantize_levels(Vk, mode)
            qtables.append((q, recon))
            # quantisation shift absorbed into b0 later
            if collapse:
                runs = collapse_runs(q)
            else:
                runs = [(i, int(q[i])) for i in range(len(q))]
            lv += struct.pack("<fB", scale, q0 & 0xFF)
            n_runs = len(runs)
            if n_runs < 255:
                lv += struct.pack("<B", n_runs)
            else:
                lv += struct.pack("<BH", 255, n_runs)
            span = 2 if max(len(q), 1) > 255 else 1
            for end, qv in runs:
                lv += (struct.pack("<H", end) if span == 2 else struct.pack("<B", end))
                if mode == "int8":
                    lv += struct.pack("<b", int(np.clip(qv, -127, 127)))
                else:
                    lv += struct.pack("<B", int(np.clip(qv, 0, 255)))
        # routing check on the reconstructed cuts (0 moved rows gate)
        rec_cuts = []
        prev = 0.0
        ci = 0
        for c in range(len(cuts)):
            tcode = (tb[c // 4] >> (2 * (c % 4))) & 3
            if tcode == 0:
                d = struct.unpack_from("<e", vals, ci)[0]; ci += 2
                prev = prev + float(np.float64(np.float16(d)))
            elif tcode == 1:
                v = struct.unpack_from("<f", vals, ci)[0]; ci += 4
                prev = float(np.float32(v))
            else:
                v = struct.unpack_from("<d", vals, ci)[0]; ci += 8
                prev = float(v)
            rec_cuts.append(prev)
        moved += int(np.sum(bin_idx(xj, np.asarray(rec_cuts)) != bin_idx(xj, cuts)))
        live = bool(len(cuts) or np.any(V != 0))
        feat_streams.append((live, struct.pack("<B", min(len(cuts), 255))
                             + (struct.pack("<H", len(cuts)) if len(cuts) >= 255 else b"")
                             + bytes(tb) + bytes(vals) + bytes(lv), qtables))
        # silence flake8
        _ = (xl, xr)

    # bitmap + streams
    bitmap = bytearray((p + 7) // 8)
    for j, (live, _, _) in enumerate(feat_streams):
        if live:
            bitmap[j // 8] |= 1 << (j % 8)
    out += bytes(bitmap)
    for live, buf, _ in feat_streams:
        if live:
            out += buf
    # tail: f64 b0[K] + f64 T  (b0 already carries the absorbed shifts)
    b0 = np.atleast_1d(np.asarray(card.b0, float))
    for k in range(K):
        out += struct.pack("<d", float(b0[k]))
    out += struct.pack("<d", 1.0)
    raw = bytes(out)
    gz = gzip.compress(raw, 9)
    info = dict(raw=len(raw), gzip=len(gz), bytes=min(len(raw), len(gz)), moved=moved,
                mode=mode, f16=use_f16, collapse=collapse)
    return raw, info


def decode_levels_only(card, mode="int8", collapse=True):
    """Quantise + reconstruct the level tables (for the |dLL| gate).  Cuts untouched.
    Per-feature level shifts are returned for b0 absorption (N9.1)."""
    newV = []
    b0_shift = 0.0
    for V in card.V:
        V = np.asarray(V, float)
        multi = V.ndim == 2
        Vk = V[:, None] if not multi else V
        recs = []
        shift_k = np.zeros(Vk.shape[1])
        for k in range(Vk.shape[1]):
            q, scale, q0, recon = quantize_levels(Vk[:, k], mode)
            if collapse:
                runs = collapse_runs(q)
                rq = np.empty_like(q)
                s = 0
                for end, qv in runs:
                    rq[s:end + 1] = qv
                    s = end + 1
                if mode == "int8":
                    rec = rq.astype(float) * scale
                else:
                    rec = (rq - q0).astype(float) * scale
            else:
                rec = recon
            shift_k[k] = float((Vk[:, k] - rec).mean())
            recs.append(rec)
        b0_shift = b0_shift + shift_k if multi else b0_shift + float(shift_k[0])
        newV.append(np.stack(recs, axis=1) if multi else recs[0])
    return newV, b0_shift


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------
def run(args):
    panels = L.build_panels()
    data, folds = panels[args.panel]
    kind = data["kind"]
    K = int(np.max(data["y"])) + 1 if kind == "multiclass" else 0
    X, y = data["X"], data["y"]
    for fold, (tr, te) in enumerate(folds):
        t0 = time.time()
        # ---- Idea 11 arms --------------------------------------------------
        arms = {
            "kappa1": dict(kappa_mult=1.0),
            "high_to1": dict(kappa_sched=sched_high_to1),
            "low_to1": dict(kappa_sched=sched_low_to1),
        }
        for tag, kw in arms.items():
            card = S.fit_joint(X[tr], y[tr], kind, atoms=[], n_classes=K, **kw)
            m = r71.metric(kind, y[te], card.score(X[te]))[0]
            row = dict(stage="5", panel=args.panel, fold=fold, arm=tag, metric=m,
                       passes=card.meta["passes"], secs=round(time.time() - t0, 1))
            if kind != "regression":
                fac = lambda X_, y_, k_, _kw=kw: S.fit_joint(X_, y_, k_, atoms=[],
                                                             n_classes=K, **_kw)
                cm = S.calib_metrics(kind, X[tr], y[tr], X[te], y[te], fac)
                row.update(cm)
            L.log_jsonl(args.out, row)
            print("  ", row, flush=True)
        # ---- Idea 12: quantized container on kappa1 / kappa0 ---------------
        for ktag, km in (("fused", 1.0), ("kappa0", 0.0)):
            card = S.fit_joint(X[tr], y[tr], kind, atoms=[], n_classes=K, kappa_mult=km)
            m_full = r71.metric(kind, y[te], card.score(X[te]))[0]
            for mode in ("int8", "uint8"):
                raw, info = emit_v3(card, X[tr], mode=mode, use_f16=True, collapse=True)
                newV, shift = decode_levels_only(card, mode=mode, collapse=True)
                b0 = np.atleast_1d(np.asarray(card.b0, float)).copy()
                b0 = b0 + np.atleast_1d(shift)      # N9.1 intercept absorption
                qcard = S.JointCard(kind, card.cuts, newV, b0, [])
                m_q = r71.metric(kind, y[te], qcard.score(X[te]))[0]
                row = dict(stage="5", panel=args.panel, fold=fold, arm=f"q_{ktag}_{mode}",
                           metric=m_q, metric_full=m_full, dLL=abs(m_q - m_full),
                           bytes=info["bytes"], raw=info["raw"], gz=info["gzip"],
                           moved=info["moved"], secs=round(time.time() - t0, 1))
                L.log_jsonl(args.out, row)
                print("  ", row, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="stage5.jsonl")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
