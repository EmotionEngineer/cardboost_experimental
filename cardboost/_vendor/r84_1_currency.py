"""r84_1_currency.py -- R84 (Фаза 5) Канарейка 1 на замороженной card_v7 (R83).

Три компонента (флаги --flags csv; при выключенных движок бит-в-бит R79/R83):
  currency : единая валюта девианса для kind=="reg" -- дисперсия potts_dp делится
             на замороженную OOF-дисперсию остатков базовой карты sigma2_oof
             (5-субфолдный OOF общего _fitj, те же сплиты seed=109, что у IsoLink;
             на cal_rand f0 = 0.330 при якоре плана ~0.31).  kappa_reg ~ sigma2*kappa.
  mbic     : штраф Zhang-Siegmund за малую массу сегмента в potts_dp:
             +0.5*ln(sum H / H_seg) на сегмент (защита тонких краевых бинов).
  fit_rank : g4-атомы фитируются по квантованной решётке rk_j (make_rank_grid,
             n=96 keep-last f16-дельты) -- fit == deploy на уровне скоров.

Замер: 16 фолдов (cal rebuild по конвенции r83_run K_pairs=20/s12_pure/CAL_TYPES;
anneal/aps -- конвенция W с nom_perm=True), свежепроцессная сертификация
decode(encode(card)), кошелёк min(raw,gz) <= 4096.  База сравнения -- строки R83
(fit .80055/.50231/14b .02814/Toof .03298; deploy .79981/.49478/.02814/.03298).
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

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/user/google_drive")

import r71_universal as U
import r76_container31 as C
import r76_lib as L
import r77_mech as R
import r77_stage as T
import r83_3_wallet14 as W

CAL_TYPES = ("g4a", "g4b", "g4c", "g4d", "p1", "m2")
WALLET = 4096
FLAGS = dict(currency=False, mbic=False, fit_rank=False)


def _cfg(nom_perm):
    R.P1_SHRINK_POST = False
    R.P1_BINS = 8
    R.HANDLE_A = False
    R.QGATE = None
    R.R84_FIT_RANK = FLAGS["fit_rank"]
    T.set_fitcfg(m=32, kappa_sched=None, warm_passes=6, lr="formula",
                 km="formula", warm_lr=1.0, base_passes=None,
                 base_conv_tol=None, solver="gs", nom_perm=nom_perm)


def sigma2_oof(Xtr, ytr):
    """Frozen OOF residual variance of the base card's skeleton (mains, atoms=[]).
    Same 5-subfold split convention (seed 109) as the IsoLink admission."""
    rng = np.random.default_rng(109)
    parts = [np.asarray(p, int) for p in
             np.array_split(rng.permutation(len(ytr)), 5)]
    eta = np.zeros(len(ytr))
    for val in parts:
        ftr = np.setdiff1d(np.arange(len(ytr)), val)
        c = T._fitj(Xtr[ftr], ytr[ftr], "regression", atoms=[], n_classes=0)
        eta[val] = np.asarray(c.score(Xtr[val]), float)
    return float(np.mean((ytr - eta) ** 2))


def fold_fit(args):
    panel, fold, tr_idx, te_idx, data, out, tag = args
    t0 = time.time()
    kind = data["kind"]
    X, y = data["X"], data["y"]
    n_classes = int(np.max(y)) + 1 if kind == "multiclass" else 0
    Xtr, ytr = X[tr_idx], y[tr_idx]
    save_dir = f"r84_1_results/cards/{tag}"
    os.makedirs(save_dir, exist_ok=True)
    _cfg(nom_perm=not panel.startswith("california"))

    sig2 = 1.0
    if kind == "regression" and FLAGS["currency"]:
        sig2 = sigma2_oof(Xtr, ytr)
    U.R84_COST["scale"] = 1.0 / sig2 if (kind == "regression" and FLAGS["currency"]) else 1.0
    U.R84_COST["mbic"] = FLAGS["mbic"]

    if panel.startswith("california"):
        T.run_fold77(panel, fold, tr_idx, te_idx, data, out,
                     K_pairs=20, K_refresh=120, max_admit=40,
                     star_types=CAL_TYPES, frag_mode="w1", screen_mode="s12_pure",
                     save_dir=save_dir)
    else:
        T.run_fold77(panel, fold, tr_idx, te_idx, data, out,
                     K_pairs=0, K_refresh=120, max_admit=40,
                     star_types=W.STEP3_TYPES, pool_mode="active",
                     screen_mode="legacy", save_dir=save_dir, oof_merge=False)
    d = pickle.load(open(f"{save_dir}/{panel}_f{fold}.pkl", "rb"))
    card = d["card"]
    admitted = d.get("admitted", [])

    iso = None
    iso_meta = dict(admitted=False, gain=0.0)
    if kind == "regression":
        eta_tr = np.asarray(card.score(Xtr), float)
        sigma2 = float(np.mean((ytr - eta_tr) ** 2))
        iso, adm, gain = W.iso_deploy(*_iso_oof(Xtr, ytr), sigma2)
        iso_meta = dict(admitted=bool(adm), gain=round(float(gain), 2))
        card.isolink = iso

    Tcal, spec14 = W.fit_calib_spec(admitted, kind, n_classes, Xtr, ytr,
                                    oof_merge=False)
    payload, info = C.emit_v31(card, Xtr)
    v31 = info["bytes"]
    rd_log = []
    if v31 > WALLET:
        card, payload, info, rd_log = W.rd_squeeze(card, Xtr, WALLET)
        v31 = info["bytes"]

    bundle = dict(panel=panel, fold=fold, kind=kind, n_classes=n_classes,
                  tr_idx=tr_idx, te_idx=te_idx, Tcal=Tcal, spec14=spec14,
                  admitted=admitted, iso=iso_meta, info=info, rd=rd_log,
                  sigma2_oof=round(sig2, 5))
    bdir = f"r84_1_results/cert/{tag}"
    os.makedirs(bdir, exist_ok=True)
    with open(f"{bdir}/payload.bin", "wb") as fh:
        fh.write(payload)
    with open(f"{bdir}/bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    with open(f"{bdir}/fitcard.pkl", "wb") as fh:
        pickle.dump(dict(card=card, Xtr=Xtr, ytr=ytr), fh)
    U.R84_COST["scale"] = 1.0
    U.R84_COST["mbic"] = False
    print(f"FIT_DONE {tag} [{time.time()-t0:.0f}s] v31={v31} sigma2_oof={sig2:.4f} "
          f"scale={1.0/sig2:.3f} iso={iso_meta}", flush=True)
    return tag


def _iso_oof(Xtr, ytr):
    rng = np.random.default_rng(109)
    parts = [np.asarray(p, int) for p in
             np.array_split(rng.permutation(len(ytr)), 5)]
    eta_oof = np.zeros(len(ytr))
    for val in parts:
        ftr = np.setdiff1d(np.arange(len(ytr)), val)
        c = T._fitj(Xtr[ftr], ytr[ftr], "regression", atoms=[], n_classes=0)
        eta_oof[val] = np.asarray(c.score(Xtr[val]), float)
    return eta_oof, ytr


def certify_fold(tag, child_py):
    bdir = f"r84_1_results/cert/{tag}"
    bundle = pickle.load(open(f"{bdir}/bundle.pkl", "rb"))
    fc = pickle.load(open(f"{bdir}/fitcard.pkl", "rb"))
    payload = open(f"{bdir}/payload.bin", "rb").read()
    kind, n_classes = bundle["kind"], bundle["n_classes"]
    data = W.PANELS[bundle["panel"]][0]
    X, y = data["X"], data["y"]
    Xte, yte = X[bundle["te_idx"]], y[bundle["te_idx"]]
    Xtr = fc["Xtr"]

    card_mem = C.decode_card_from_bytes(payload)
    fit_card = fc["card"]

    with open(f"{bdir}/child_in.pkl", "wb") as fh:
        pickle.dump(dict(Xte=Xte, yte=yte, kind=kind, n_classes=n_classes,
                         Tcal=bundle["Tcal"], spec14=bundle["spec14"],
                         inmem_pred=np.asarray(C.model_score(card_mem, Xte), float),
                         fit_pred=np.asarray(C.model_score(fit_card, Xte), float)), fh)
    r = subprocess.run([sys.executable, child_py, "--cert-child", bdir],
                       capture_output=True, text=True, timeout=1200)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if not line:
        raise RuntimeError(f"child failed for {tag}: {r.stdout[-2000:]} {r.stderr[-2000:]}")
    ch = json.loads(line[-1])

    met_mem = W.apply_metric(kind, yte, C.model_score(card_mem, Xte),
                             bundle["Tcal"], bundle["spec14"])
    met_fit = W.apply_metric(kind, yte, C.model_score(fit_card, Xte),
                             bundle["Tcal"], bundle["spec14"])
    return dict(panel=bundle["panel"], fold=bundle["fold"], tag=tag,
                metric_mem=met_mem, metric_dec=ch["metric_dec"], metric_fit=met_fit,
                maxdiff_mem_dec=ch["maxdiff"],
                maxdiff_fit_mem=float(np.max(np.abs(
                    np.asarray(C.model_score(fit_card, Xte), float)
                    - np.asarray(C.model_score(card_mem, Xte), float)))),
                moved_train=bundle["info"]["moved"], v31=bundle["info"]["bytes"],
                raw=bundle["info"]["raw"], gz=bundle["info"]["gzip"],
                sigma2_oof=bundle.get("sigma2_oof"), rd=bundle["rd"])


def main():
    global FLAGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert-child", default=None)
    ap.add_argument("--out", default="r84_1.jsonl")
    ap.add_argument("--only", default="")
    ap.add_argument("--flags", default="currency,mbic,fit_rank")
    ap.add_argument("--control", action="store_true",
                    help="switches OFF (rebuild control): cal f0 must reproduce R83 fit .79341")
    a = ap.parse_args()
    if a.cert_child:
        sys.exit(W.child_main(a.cert_child))
    if a.control:
        FLAGS = dict(currency=False, mbic=False, fit_rank=False)
    else:
        FLAGS = dict(currency="currency" in a.flags, mbic="mbic" in a.flags,
                     fit_rank="fit_rank" in a.flags)
    print("FLAGS:", FLAGS, flush=True)

    t0 = time.time()
    panels = L.build_panels()
    W.PANELS = panels
    folds = {}
    for pname in ("california_spatial", "california_random", "anneal", "aps"):
        data, splits = panels[pname]
        for f, (tr, te) in enumerate(splits):
            folds[f"{pname}_f{f}"] = (pname, f, tr, te, data)

    jobs = []
    for tag, (pname, f, tr, te, data) in folds.items():
        if a.only and tag not in a.only.split(","):
            continue
        jobs.append((pname, f, tr, te, data, a.out, tag))
    import multiprocessing as mp
    with mp.Pool(2) as pool:
        for _ in pool.imap_unordered(fold_fit, jobs):
            pass
    print(f"== fit done [{time.time()-t0:.0f}s]", flush=True)

    rows = []
    for tag, _ in folds.items():
        if a.only and tag not in a.only.split(","):
            continue
        row = certify_fold(tag, os.path.abspath(__file__))
        rows.append(row)
        L.log_jsonl(a.out, dict(stage="r84_1", flags=FLAGS, **row))
        print(f"CERT {tag}: fit={ {k: round(v, 5) for k, v in row['metric_fit'].items()} } "
              f"dec={ {k: round(v, 5) for k, v in row['metric_dec'].items()} } "
              f"diff={row['maxdiff_mem_dec']} moved={row['moved_train']} "
              f"v31={row['v31']} s2oof={row['sigma2_oof']}", flush=True)

    per_fit, per_dec = {}, {}
    for r in rows:
        per_fit.setdefault(r["panel"], []).append(r["metric_fit"])
        per_dec.setdefault(r["panel"], []).append(r["metric_dec"])
    for p in per_fit:
        for k in per_fit[p][0]:
            df = float(np.mean([x[k] for x in per_fit[p]]))
            dd = float(np.mean([x[k] for x in per_dec[p]]))
            print(f"  {p} {k}: fit {df:.5f} deploy {dd:.5f}", flush=True)
    wallets = [r["v31"] for r in rows]
    print("  wallet max/mean:", max(wallets), int(np.mean(wallets)),
          "16/16<=4096:", all(w <= WALLET for w in wallets),
          "maxdiff all 0:", all(r["maxdiff_mem_dec"] == 0.0 for r in rows),
          "moved all 0:", all(r["moved_train"] == 0 for r in rows), flush=True)
    L.log_jsonl(a.out, dict(stage="r84_1_summary", flags=FLAGS,
                            per_fit={p: {k: float(np.mean([x[k] for x in ms]))
                                         for k in ms[0]} for p, ms in per_fit.items()},
                            per_deploy={p: {k: float(np.mean([x[k] for x in ms]))
                                            for k in ms[0]} for p, ms in per_dec.items()},
                            wallet_max=max(wallets)))
    print(f"== r84_1 done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
