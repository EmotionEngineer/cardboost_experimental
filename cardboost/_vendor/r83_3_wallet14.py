"""r83_3_wallet14.py -- R83 Part 3 (#14): rate-distortion wallet + 16/16
fresh-process certification of the frozen unified config

    card_v7 (R83) = card_p_law + s12 (#12) + nom_perm (#5) + IsoLink-OOF (H7)

16 folds: cal_rand f0..f4, cal_spat f0..f4, anneal f0..f2, aps f0..f2.
Config contract (Rule 12.B-3, no per-task selectors):
  * cal    : Stage-1 s12 recipe (r83_run), nom_perm flag on (bit-identical to
             the saved cards: all u_j > 12 -- spot-checked), IsoLink-OOF for
             kind == "reg" (f16 knots at fit, <= 64 B).
  * anneal : R82 canary recipe (STEP3_TYPES star pool, legacy screen,
             K_pairs=0, nom_perm=True, oof_merge=False) -> 14b anchor 0.02814.
  * aps    : same recipe -> Toof anchor 0.03367; the oof_merge variant is
             measured SEPARATELY (3 folds) and is NOT in card_v7.

Per fold the deploy pipeline is: emit_v31 (incl. native IsoLink section) ->
deploy-card := decode(emit) in-process ("in-mem") -> FRESH subprocess decodes
the same payload -> certification row:
  * metric (R2 / 14b LL / Toof LL) before (in-mem) and after (decoded);
  * max|decoded.score - in_mem.score| == 0.0 (cross-process bit-exactness),
    plus fidelity vs the raw float fit (reference column);
  * moved_train == 0, raw/gz/v31 bytes, wallet <= 4096 B on every fold;
  * rate-distortion allocator #14 (Fisher-sensitivity coarsening / u8-affine
    table snap) fires only if a fold exceeds budget.

Universal Score (TOP_UNIVERSAL convention): mean over 4 panels of
  ratio-to-best (R2: model/best; LL: best/model), best = xgboost 0.8558 /
  QUARTZ-spatial 0.5957 / card_fused 0.0328 (Toof) / EBM-0 0.0285.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import warnings

import numpy as np

import r76_container31 as C
import r76_lib as L
import r76_mech as M
import r76_stage31 as S
import r77_mech as R
import r77_stage as T
import r83_ceil as CE
from r71_universal import metric as r71_metric
from r71_universal import fit_T as r71_fit_T

warnings.filterwarnings("ignore")

STEP3_TYPES = ("g4a", "g4b", "g4c", "g4d", "p1", "m2", "m2_pos")
CAL_TYPES = ("g4a", "g4b", "g4c", "g4d", "p1", "m2")
WALLET = 4096

UNIV_BEST = dict(california_random=0.8558, california_spatial=0.5957,
                 anneal=0.0328, aps=0.0285)


def _cfg(nom_perm=True):
    R.P1_SHRINK_POST = False
    R.P1_BINS = 8
    R.HANDLE_A = False
    R.QGATE = None
    T.set_fitcfg(m=32, kappa_sched=None, warm_passes=6, lr="formula",
                 km="formula", warm_lr=1.0, base_passes=None,
                 base_conv_tol=None, solver="gs", nom_perm=bool(nom_perm))


# ---------------------------------------------------------------- IsoLink deploy
def iso_deploy(eta_oof, y, sigma2, n_nodes=16, ridge=0.1, seed=109):
    """Prequential admission exactly as r83_ceil.iso_arm; returns the DEPLOYED
    knots (f16, iso_fit quantizes at fit) or None when not admitted."""
    n = len(y)
    parts = CE._parts(n, seed)
    tot = 0.0
    for val in parts:
        tr = np.setdiff1d(np.arange(n), val)
        knots = CE.iso_fit(eta_oof[tr], y[tr], n_nodes, ridge)
        tot += CE.nats_gain(y[val], eta_oof[val],
                            CE.iso_apply(knots, eta_oof[val]), sigma2)
    admitted = bool(tot > 0.5 * np.log(n))
    if not admitted:
        return None, False, tot
    return CE.iso_fit(eta_oof, y, n_nodes, ridge), True, tot


def calib_factory(admitted, n_classes, oof_merge=False, seed=109):
    """Replica of run_fold77's calibration factory (r77_stage.py:591-615):
    refit base + admitted atoms + warm joint backfit (+ optional oof_merge)."""
    def fac(Xf, yf, kf):
        b = T._fitj(Xf, yf, kf, atoms=[], n_classes=n_classes)
        if not admitted:
            if oof_merge:
                import r82_merge as RM
                b = RM.merge_card(b, Xf, yf, kf, n_classes, n_sub=5, seed=seed)
            return b
        e = b.score(Xf)
        aa = []
        for (at_, j, k) in admitted:
            try:
                at = R.make_atom77(at_, j, k, b, X=Xf)
            except (IndexError, ValueError):
                continue
            at.ensure(kf, n_classes)
            at.refit(Xf, yf, e, kf, kappa_mult=1.0, n_classes=n_classes)
            e = e + at.contrib(Xf)
            aa.append(at)
        cm2 = T._fitj(Xf, yf, kf, atoms=aa, n_classes=n_classes, warm_card=b,
                      passes=T.FITCFG["warm_passes"], conv_tol=0.0)
        if oof_merge:
            import r82_merge as RM
            cm2 = RM.merge_card(cm2, Xf, yf, kf, n_classes, n_sub=5, seed=seed)
        return cm2
    return fac


def fit_calib_spec(admitted, kind, n_classes, Xtr, ytr, oof_merge=False):
    if kind == "regression":
        return None, None
    fac = calib_factory(admitted, n_classes, oof_merge=oof_merge)
    eta_oof = L.toof_eta(fac, Xtr, ytr, kind, folds=5, seed=109)
    try:
        Tcal = float(r71_fit_T(ytr, eta_oof, kind))
    except Exception:
        Tcal = 1.0
    spec14 = L.fit_T14(ytr, eta_oof, kind, True)
    return Tcal, spec14


def apply_metric(kind, y, score, Tcal, spec14):
    out = {}
    if kind == "regression":
        out["R2"] = r71_metric(kind, y, score)[0]
        return out
    out["Toof"] = r71_metric(kind, y, np.asarray(score) / Tcal)[0]
    out["14b"] = r71_metric(kind, y, L.apply_T14(spec14, score))[0]
    return out


# ---------------------------------------------------------------- fold fitting
def fold_fit(args):
    panel, fold, tr_idx, te_idx, data, out, tag, oof_merge = args
    t0 = time.time()
    kind = data["kind"]
    X, y = data["X"], data["y"]
    n_classes = int(np.max(y)) + 1 if kind == "multiclass" else 0
    Xtr, ytr = X[tr_idx], y[tr_idx]
    save_dir = f"r76_results/cards_r83_3/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    _cfg(nom_perm=True)

    if panel.startswith("california"):
        # Stage-1 card already fitted by r83_run (s12 recipe); reload it.
        pkl = f"r76_results/cards_r83/{panel}/{panel}_f{fold}.pkl"
        d = pickle.load(open(pkl, "rb"))
        card = d["card"]
        admitted = d.get("admitted", [])
    else:
        T.run_fold77(panel, fold, tr_idx, te_idx, data, out,
                     K_pairs=0, K_refresh=120, max_admit=40,
                     star_types=STEP3_TYPES, pool_mode="active",
                     screen_mode="legacy", save_dir=save_dir,
                     oof_merge=oof_merge)
        d = pickle.load(open(f"{save_dir}/{panel}_f{fold}.pkl", "rb"))
        card = d["card"]
        admitted = d.get("admitted", [])

    iso = None
    iso_meta = dict(admitted=False, gain=0.0)
    if kind == "regression":
        eta_tr = np.asarray(card.score(Xtr), float)
        sigma2 = float(np.mean((ytr - eta_tr) ** 2))
        rng = np.random.default_rng(109)
        parts = [np.asarray(p, int) for p in
                 np.array_split(rng.permutation(len(ytr)), 5)]
        eta_oof = np.zeros(len(ytr))
        for val in parts:
            ftr = np.setdiff1d(np.arange(len(ytr)), val)
            c = T._fitj(Xtr[ftr], ytr[ftr], "regression", atoms=[], n_classes=0)
            eta_oof[val] = np.asarray(c.score(Xtr[val]), float)
        iso, adm, gain = iso_deploy(eta_oof, ytr, sigma2)
        iso_meta = dict(admitted=bool(adm), gain=round(float(gain), 2))
        card.isolink = iso

    # calibration spec (Toof T / 14b per-class beta), fold-fit on the train side
    Tcal, spec14 = fit_calib_spec(admitted, kind, n_classes, Xtr, ytr,
                                  oof_merge=False)

    payload, info = C.emit_v31(card, Xtr)
    v31 = info["bytes"]
    rd_log = []
    if v31 > WALLET:
        card, payload, info, rd_log = rd_squeeze(card, Xtr, WALLET)
        v31 = info["bytes"]

    bundle = dict(panel=panel, fold=fold, kind=kind, n_classes=n_classes,
                  tr_idx=tr_idx, te_idx=te_idx, Tcal=Tcal, spec14=spec14,
                  admitted=admitted, iso=iso_meta, info=info, rd=rd_log)
    bdir = f"r76_results/cert_r83/{tag}"
    os.makedirs(bdir, exist_ok=True)
    with open(f"{bdir}/payload.bin", "wb") as fh:
        fh.write(payload)
    with open(f"{bdir}/bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    with open(f"{bdir}/fitcard.pkl", "wb") as fh:
        pickle.dump(dict(card=card, Xtr=Xtr, ytr=ytr), fh)
    print(f"FIT_DONE {tag} [{time.time()-t0:.0f}s] v31={v31} iso={iso_meta}",
          flush=True)
    return tag


def rd_squeeze(card, Xtr, budget, mode="int8"):
    """Rate-distortion allocator #14: until the payload fits `budget`, snap the
    LEAST Fisher-sensitive main table (sum_b H_b dV_b^2) to a u8-affine grid,
    then progressively coarser (256/64/16/8 levels).  No refit (R48 ban).
    Terminates: every step consumes a (feature, level) pair from a finite set."""
    log = []
    payload, info = C.emit_v31(card, Xtr, mode=mode)
    ladder = [256, 64, 16, 8]
    done = set()
    progressed = True
    while info["bytes"] > budget and progressed:
        progressed = False
        sens = C.rd_sensitivity(card, Xtr, None)
        for _, j in sens:
            V = np.asarray(card.V[j], float)
            if not np.any(V != 0):
                continue
            for lev in ladder:
                if (j, lev) in done:
                    continue
                done.add((j, lev))
                vmin = min(float(V.min()), 0.0)
                vmax = max(float(V.max()), 0.0)
                grid = np.linspace(vmin, vmax, lev)
                Vq = grid[np.abs(grid[None, :] - V.reshape(-1, 1)).argmin(axis=1)]
                card.V[j] = Vq.reshape(V.shape)
                log.append(dict(feat=int(j), levels=int(lev)))
                payload, info = C.emit_v31(card, Xtr, mode=mode)
                progressed = True
                break
            if progressed:
                break
    return card, payload, info, log


# ---------------------------------------------------------------- certification
def certify_fold(tag, child_py):
    bdir = f"r76_results/cert_r83/{tag}"
    bundle = pickle.load(open(f"{bdir}/bundle.pkl", "rb"))
    fc = pickle.load(open(f"{bdir}/fitcard.pkl", "rb"))
    payload = open(f"{bdir}/payload.bin", "rb").read()
    kind, n_classes = bundle["kind"], bundle["n_classes"]
    data = PANELS[bundle["panel"]][0]
    X, y = data["X"], data["y"]
    Xte, yte = X[bundle["te_idx"]], y[bundle["te_idx"]]
    Xtr = fc["Xtr"]

    card_mem = C.decode_card_from_bytes(payload)      # in-mem deploy form
    fit_card = fc["card"]

    # in-process reference scores
    import pickle as _pk
    with open(f"{bdir}/child_in.pkl", "wb") as fh:
        _pk.dump(dict(Xte=Xte, yte=yte, kind=kind, n_classes=n_classes,
                      Tcal=bundle["Tcal"], spec14=bundle["spec14"],
                      inmem_pred=np.asarray(C.model_score(card_mem, Xte), float),
                      fit_pred=np.asarray(C.model_score(fit_card, Xte), float)),
                 fh)
    r = subprocess.run([sys.executable, child_py, "--cert-child", bdir],
                       capture_output=True, text=True, timeout=1200)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise RuntimeError(f"child failed for {tag}: {r.stdout[-2000:]} {r.stderr[-2000:]}")
    ch = json.loads(line[-1])

    met_mem = apply_metric(kind, yte, C.model_score(card_mem, Xte),
                           bundle["Tcal"], bundle["spec14"])
    met_fit = apply_metric(kind, yte, C.model_score(fit_card, Xte),
                           bundle["Tcal"], bundle["spec14"])
    return dict(panel=bundle["panel"], fold=bundle["fold"], tag=tag,
                metric_mem=met_mem, metric_dec=ch["metric_dec"],
                metric_fit=met_fit,
                maxdiff_mem_dec=ch["maxdiff"],
                maxdiff_fit_mem=float(np.max(np.abs(
                    np.asarray(C.model_score(fit_card, Xte), float)
                    - np.asarray(C.model_score(card_mem, Xte), float)))),
                moved_train=bundle["info"]["moved"],
                raw=bundle["info"]["raw"], gz=bundle["info"]["gzip"],
                v31=bundle["info"]["bytes"], iso=bundle["iso"],
                rd=bundle["rd"])


def child_main(bdir):
    import pickle as _pk
    d = _pk.load(open(f"{bdir}/child_in.pkl", "rb"))
    payload = open(f"{bdir}/payload.bin", "rb").read()
    card = C.decode_card_from_bytes(payload)
    pred = np.asarray(C.model_score(card, d["Xte"]), float)
    maxdiff = float(np.max(np.abs(pred - np.asarray(d["inmem_pred"], float))))
    met = apply_metric(d["kind"], d["yte"], pred, d["Tcal"], d["spec14"])
    print(json.dumps(dict(maxdiff=maxdiff, metric_dec=met)))
    return 0


# ---------------------------------------------------------------- main
PANELS = {}


def main():
    global PANELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-child", default=None)
    ap.add_argument("--out", default="r83_3.jsonl")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    if a.cert_child:
        sys.exit(child_main(a.cert_child))

    t0 = time.time()
    panels = L.build_panels()
    PANELS = panels
    folds = {}
    for pname in ("california_spatial", "california_random", "anneal", "aps"):
        data, splits = panels[pname]
        for f, (tr, te) in enumerate(splits):
            folds[f"{pname}_f{f}"] = (pname, f, tr, te, data)

    jobs = []
    for tag, (pname, f, tr, te, data) in folds.items():
        if a.only and tag not in a.only.split(","):
            continue
        jobs.append((pname, f, tr, te, data, a.out, tag, False))
    import multiprocessing as mp
    with mp.Pool(2) as pool:
        for _ in pool.imap_unordered(fold_fit, jobs):
            pass
    print(f"== fit done [{time.time()-t0:.0f}s]", flush=True)

    rows = []
    for tag in sorted(folds):
        if a.only and tag not in a.only.split(","):
            continue
        row = certify_fold(tag, os.path.abspath(__file__))
        rows.append(row)
        L.log_jsonl(a.out, dict(stage="r83_3", **row))
        print(f"CERT {tag}: mem={row['metric_mem']} dec={row['metric_dec']} "
              f"diff={row['maxdiff_mem_dec']} moved={row['moved_train']} "
              f"v31={row['v31']}", flush=True)

    # ---- aps oof_merge variant (measurement only, NOT in card_v7) ----------
    data, splits = panels["aps"]
    for f, (tr, te) in enumerate(splits):
        tag = f"aps_merge_f{f}"
        if a.only and tag not in a.only.split(","):
            continue
        job = ("aps", f, tr, te, data, a.out, tag, True)
        fold_fit(job)
        row = certify_fold(tag, os.path.abspath(__file__))
        rows.append(row)
        L.log_jsonl(a.out, dict(stage="r83_3_merge", **row))
        print(f"CERT {tag}: mem={row['metric_mem']} v31={row['v31']}", flush=True)

    # ---- summary ----------------------------------------------------------
    core = [r for r in rows if not r["tag"].startswith("aps_merge")]
    print("\n== 16/16 CERT TABLE (card_v7 R83) ==")
    for r in core:
        print(r["tag"], "mem", {k: round(v, 6) for k, v in r["metric_mem"].items()},
              "dec", {k: round(v, 6) for k, v in r["metric_dec"].items()},
              "fit", {k: round(v, 6) for k, v in r["metric_fit"].items()},
              "maxdiff", r["maxdiff_mem_dec"], "moved", r["moved_train"],
              "raw/gz/v31", r["raw"], r["gz"], r["v31"])
    wallets = [r["v31"] for r in core]
    print("wallet max:", max(wallets), "mean:", int(np.mean(wallets)),
          "16/16 <= 4096:", all(w <= WALLET for w in wallets))
    print("maxdiff all zero:", all(r["maxdiff_mem_dec"] == 0.0 for r in core))
    print("moved all zero:", all(r["moved_train"] == 0 for r in core))

    # Universal Score (deployed metrics; anneal in Toof currency per TOP_UNIVERSAL)
    per = {}
    for r in core:
        per.setdefault(r["panel"], []).append(r["metric_mem"])
    ratios = {}
    for p, ms in per.items():
        if p.startswith("california"):
            m = float(np.mean([x["R2"] for x in ms]))
            ratios[p] = m / UNIV_BEST[p]
        elif p == "anneal":
            m = float(np.mean([x["Toof"] for x in ms]))
            ratios[p] = UNIV_BEST[p] / m
        else:
            m = float(np.mean([x["Toof"] for x in ms]))
            ratios[p] = UNIV_BEST[p] / m
    us = float(np.mean(list(ratios.values())))
    print("Universal Score card_v7 (R83):", round(us, 4),
          {k: round(v, 4) for k, v in ratios.items()},
          "worst:", min(ratios, key=ratios.get))
    L.log_jsonl(a.out, dict(stage="r83_3_summary",
                                 wallet_max=max(wallets),
                                 wallet_mean=int(np.mean(wallets)),
                                 maxdiff_max=max(r["maxdiff_mem_dec"] for r in core),
                                 moved_max=max(r["moved_train"] for r in core),
                                 universal_score=us, ratios=ratios,
                                 per_panel={p: {k: float(np.mean([x[k] for x in ms]))
                                                for k in ms[0]} for p, ms in per.items()},
                                 secs=round(time.time() - t0, 1)))
    print(f"== r83_3 done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
