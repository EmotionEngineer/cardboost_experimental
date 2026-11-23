"""r77_stage.py — R77 ledger + fold runner (blueprint §1-2, §5 steps 1-2).

Ledger v2 (fixes r76_stage31 cause C and the manual spatial switch):
  * regression nats = sum_s dSSE_s / (2*sigma2_oof)  (r76 used dSSE/2 without
    sigma^2 -> ~3x understatement on California);
  * entry price = kappa_block + ln M with kappa_block = 0.5*ln(N_train),
    M = total candidates screened;
  * >= 4/5 subfolds positive (kept);
  * stagewise admission: greedy by net with K-best refresh; the admitted chain
    refits on the evolving residual (cached per subfold);
  * Frag-Guard (universal, no column names): j* = argmax_j sum_b H_jb V_j[b]^2
    of the fitted base; per subfold the validation rows are cut into 3 blocks
    along x_j* and 2 random half-bags; candidate is FRAGILE if any block gain
    < -kappa_block or the half-bag signs disagree  ->  rejected (this is what
    kills p1:6-7 on cal_spat without any manual ledger switch; Rule 12.B-3).

One code path for all 4 panels.  sw14 is dead (r76 registry) and not built.
"""
import time

import numpy as np

import r76_lib as L
import r76_stage31 as S
import r71_universal as r71
import r77_mech as R

C2D_HALF_GUARD = False

r71 = L.r71  # loader-verified engine handle

# ---- run configuration (step-1 defaults; r77_b overrides for step 2) --------
FITCFG = dict(m=32, kappa_sched=None, warm_passes=6,
              lr=1.0, km=None, warm_lr=1.0, gram_lr=False,
              base_passes=None, base_conv_tol=None)


def set_fitcfg(**kw):
    FITCFG.update(kw)


def _fitj(X, y, kind, **kw):
    kw.setdefault("m", FITCFG["m"])
    warm = kw.get("warm_card") is not None
    if FITCFG["kappa_sched"] is not None and not warm:
        kw.setdefault("kappa_sched", FITCFG["kappa_sched"])
    if warm:
        # warm joint-backfit continues at kappa_mult=1.0: the Idea-11 warmup is
        # for COLD fits only (measured: km=4 warm oversmooths anneal mains,
        # 14b 0.021 -> 0.042 with zero atoms admitted)
        kw.setdefault("kappa_mult", 1.0)
        lr = float(FITCFG.get("warm_lr", 1.0))
    else:
        # R79 dimension law (user): lr(p) = km(p) = min(1.0, sqrt(10/p)) when
        # set to "formula" -- one closed rule, no per-panel switches (12.B-3).
        law = min(1.0, (10.0 / X.shape[1]) ** 0.5)
        law_use = law
        if FITCFG.get("p_active_law") or FITCFG.get("p_phase_law"):
            # R108 (frozen 2026-11-16, Q3a): p-law вход = число НЕКОНСТАНТНЫХ
            # колонок (>=2 distinct finite values; anneal 38 -> 18).  passes и
            # conv_tol остаются от СЫРОГО p -- меняются только шаг lr и цена km.
            # R109 (frozen R87 Finalist 1): фазовый порог мёртвых колонок --
            # p -> p_act ТОЛЬКО когда p_dead > sqrt(p) (Poisson-fluctuation
            # threshold): cal p_dead=0<=2.83 и aps p_dead=2<=13.04 остаются
            # бит-в-бит на сыром p; anneal p_dead=20>6.16 переключается.
            n_act = 0
            for _j in range(X.shape[1]):
                _c = X[:, _j]
                if np.unique(_c[np.isfinite(_c)]).size >= 2:
                    n_act += 1
            use_act = True
            if FITCFG.get("p_phase_law") and not FITCFG.get("p_active_law"):
                use_act = (X.shape[1] - n_act) > (X.shape[1] ** 0.5)
            if use_act:
                law_use = min(1.0, (10.0 / max(n_act, 1)) ** 0.5)
        km = FITCFG.get("km")
        if FITCFG.get("gram_lr"):
            # R85 Canary 1 (frozen spec): per-feature lr_j = km_j from X-only
            # rank-collinearity (r85_gram); passes/conv_tol from the SLOWEST
            # feature (min rule).  When min_j lr_j == 1.0 nothing is overridden
            # and kappa_mult is the same scalar 1.0 as the lr==1.0 path today
            # (cal neutrality is bit-identical by construction).
            import r85_gram as G
            lr_v, km_v = G.gram_lr_vec(X)
            lr_eff = float(lr_v.min())
            if FITCFG["kappa_sched"] is None and "kappa_mult" not in kw:
                kw["kappa_mult"] = km_v if lr_eff < 1.0 else 1.0
            if lr_eff < 1.0:
                kw.setdefault("lr", lr_v)
                bp = FITCFG.get("base_passes") or min(48, int(np.ceil(24.0 / lr_eff)))
                kw.setdefault("passes", int(bp))
                kw.setdefault("max_passes", max(int(bp), 48))
                kw.setdefault("conv_tol", FITCFG.get("base_conv_tol") or 1e-5 * lr_eff)
            lr = 1.0
        else:
            if FITCFG["kappa_sched"] is None and km is not None and "kappa_mult" not in kw:
                kw["kappa_mult"] = law_use if km == "formula" else float(km)
            lr_spec = FITCFG.get("lr", 1.0)
            lr = law_use if lr_spec == "formula" else float(lr_spec)
    if lr != 1.0:
        kw.setdefault("lr", lr)
        # user R78 rule: per-pass level steps shrink by 1/lr, so the fixed-point
        # threshold must scale too (else the loop exits before the pass budget)
        if not warm:
            # R108/R109: длина обучения (passes/conv_tol) считается от СЫРОГО p
            lr_len = law if ((FITCFG.get("p_active_law") or FITCFG.get("p_phase_law"))
                             and lr_spec == "formula") else lr
            bp = FITCFG.get("base_passes") or min(48, int(np.ceil(24.0 / lr_len)))
            kw.setdefault("passes", int(bp))
            kw.setdefault("max_passes", max(int(bp), 48))
            kw.setdefault("conv_tol", FITCFG.get("base_conv_tol") or 1e-5 * lr_len)
    kw.setdefault("solver", FITCFG.get("solver", "gs"))
    kw.setdefault("nom_perm", bool(FITCFG.get("nom_perm", False)))
    return S.fit_joint(X, y, kind, **kw)


# ----------------------------------------------------------------------------
# nats currency
# ----------------------------------------------------------------------------
def nats_pieces(kind, y, eta_b, eta_a, sigma2=1.0, T=1.0):
    """Per-row nats gains (positive = candidate improves)."""
    if kind == "regression":
        return ((y - eta_b / T) ** 2 - (y - eta_a / T) ** 2) / (2.0 * sigma2)
    pb = 1.0 / (1.0 + np.exp(-np.clip(eta_b / T, -80, 80))) if kind == "binary" \
        else _softmax(eta_b / T)
    pa = 1.0 / (1.0 + np.exp(-np.clip(eta_a / T, -80, 80))) if kind == "binary" \
        else _softmax(eta_a / T)
    if kind == "binary":
        eps = 1e-12
        lb = -(y * np.log(np.clip(pb, eps, 1)) + (1 - y) * np.log(np.clip(1 - pb, eps, 1)))
        la = -(y * np.log(np.clip(pa, eps, 1)) + (1 - y) * np.log(np.clip(1 - pa, eps, 1)))
        return lb - la
    Y1 = np.eye(eta_b.shape[1])[y.astype(int)]
    eps = 1e-12
    return (-(Y1 * np.log(np.clip(pb, eps, 1))).sum(1)) + (Y1 * np.log(np.clip(pa, eps, 1))).sum(1)


def _softmax(E):
    Z = np.exp(E - E.max(axis=1, keepdims=True))
    return Z / Z.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# the stagewise dual-gate ledger
# ----------------------------------------------------------------------------
class Ledger:
    """5-subfold prequential ledger with cached bases, stagewise chain and
    Frag-Guard.  All candidates are evaluated against the current chain state."""

    def __init__(self, X, y, kind, n_classes, sigma2=1.0, seed=109, n_sub=5,
                 kappa_block=0.0, lnM=0.0, jstar=0, base_passes=24, base_tol=1e-5,
                 frag_mode="w1"):
        rng = np.random.default_rng(seed)
        self.X, self.y, self.kind, self.K = X, y, kind, n_classes
        self.sigma2 = max(float(sigma2), 1e-9)
        self.kappa_block = float(kappa_block)
        self.lnM = float(lnM)
        self.jstar = int(jstar)
        self.frag_mode = frag_mode
        self.parts = [np.asarray(p, int) for p in np.array_split(rng.permutation(len(y)), n_sub)]
        self.n_sub = n_sub
        self.base = []
        self.eta_tr = []      # chain state on sub-train (evolving)
        self.eta_va = []      # chain state on validation rows (evolving)
        self.T = []
        self.chain = []       # fitted chain atoms per subfold (for reference)
        for s in range(n_sub):
            va = self.parts[s]
            tr = np.concatenate([self.parts[g] for g in range(n_sub) if g != s])
            b = _fitj(X[tr], y[tr], kind, atoms=[], n_classes=n_classes,
                            passes=base_passes, max_passes=max(base_passes, 48),
                            conv_tol=base_tol)
            self.base.append((b, tr, va))
            self.eta_tr.append(b.score(X[tr]))
            self.eta_va.append(b.score(X[va]))
            if kind == "regression":
                self.T.append(1.0)
            else:
                # Toof-ledger (step 3): temperature on TRUE OOF scores — the
                # validation rows are out-of-sample for this subfold's base fit.
                try:
                    self.T.append(float(r71.fit_T(y[va], self.eta_va[s], kind)))
                except Exception:
                    self.T.append(1.0)
            self.chain.append([])

    def eval(self, cands):
        """cands: list of (tag, atype, j, k).  Returns dict tag -> record."""
        out = {}
        X, y = self.X, self.y
        for tag, atype, j, k in cands:
            gains = np.zeros(self.n_sub)
            pieces = []   # (block gains[3], half gains[2]) per subfold
            if self.frag_mode == "or_guard":
                # user 6.2: min_block > -kappa on >=1 axis of {j*, r_j, r_k}
                axes = tuple(dict.fromkeys([self.jstar, j, k]))
            else:
                axes = (self.jstar,)   # blueprint 2.2: block check along j*
            for s in range(self.n_sub):
                b, tr, va = self.base[s]
                try:
                    at = R.make_atom77(atype, j, k, b, X[tr])
                except (IndexError, ValueError):
                    # degenerate support on this sub-train (all-NaN / near-empty
                    # sensor): unbuildable candidate -> zero OOF contribution,
                    # ledger math for buildable candidates unchanged.
                    continue
                at.ensure(self.kind, self.K)
                at.refit(X[tr], y[tr], self.eta_tr[s], self.kind,
                         kappa_mult=1.0, n_classes=self.K)
                eta2_va = self.eta_va[s] + at.contrib(X[va])
                T = self.T[s]
                pc = nats_pieces(self.kind, y[va], self.eta_va[s], eta2_va,
                                 self.sigma2, T)
                gains[s] = float(pc.sum())
                # frag pieces: 3 blocks along each axis rank, 2 half-bags
                axis_bg = []
                for ax in axes:
                    xj = X[va, ax]
                    order = np.argsort(np.where(np.isfinite(xj), xj, np.inf), kind="stable")
                    blocks = np.array_split(order, 3)
                    axis_bg.append([float(pc[bl].sum()) if len(bl) else 0.0 for bl in blocks])
                rng = np.random.default_rng(65 + s)
                perm = rng.permutation(len(va))
                h1 = float(pc[perm[: len(va) // 2]].sum())
                h2 = float(pc[perm[len(va) // 2:]].sum())
                pieces.append((axis_bg, (h1, h2)))
            fragile = False
            for axis_bg, (h1, h2) in pieces:
                if self.frag_mode == "or_guard":
                    if not any(min(bg) > -self.kappa_block for bg in axis_bg):
                        fragile = True
                else:
                    if min(min(bg) for bg in axis_bg) < -(self.kappa_block + self.lnM):
                        fragile = True
                if h1 * h2 < 0:
                    c2d_ok = (C2D_HALF_GUARD and atype in ("p1", "m2")
                              and np.all(gains > 0)
                              and min(h1, h2) > -0.25 * (self.kappa_block + self.lnM))
                    if not c2d_ok:
                        fragile = True
            dq = 0  # filled by caller when needed
            out[tag] = dict(tag=tag, type=atype, j=j, k=k,
                            gain=float(gains.sum()), subs=gains.round(3).tolist(),
                            subs_pos=int((gains > 0).sum()), frag=fragile, dq=dq)
        return out

    def extend_chain(self, atype, j, k):
        """Admit one atom: fit it per subfold on the current chain residual."""
        for s in range(self.n_sub):
            b, tr, va = self.base[s]
            try:
                at = R.make_atom77(atype, j, k, b, self.X[tr])
            except (IndexError, ValueError):
                continue
            at.ensure(self.kind, self.K)
            at.refit(self.X[tr], self.y[tr], self.eta_tr[s], self.kind,
                     kappa_mult=1.0, n_classes=self.K)
            self.eta_tr[s] = self.eta_tr[s] + at.contrib(self.X[tr])
            self.eta_va[s] = self.eta_va[s] + at.contrib(self.X[va])
            self.chain[s].append((atype, j, k))


# ----------------------------------------------------------------------------
# contour 2 of the universal dual-ledger: axis-block prequential screen
# ----------------------------------------------------------------------------
class AxisBlockScreen:
    """Block-prequential screen along one feature axis: the 5 subfolds are 5
    CONTIGUOUS rank blocks; each sub-train therefore has a HOLE where the
    validation block sits, so a candidate must interpolate across the hole.
    This generalizes the r76 spatial-ledger instrument to any axis (parents rj,
    rk and the dominant main j*), with no column names (Rule 12.B-3)."""

    def __init__(self, X, y, kind, n_classes, axis, sigma2=1.0, n_sub=5):
        self.X, self.y, self.kind, self.K = X, y, kind, n_classes
        self.sigma2 = max(float(sigma2), 1e-9)
        self.axis = int(axis)
        xa = X[:, self.axis]
        order = np.argsort(np.where(np.isfinite(xa), xa, np.inf), kind="stable")
        self.parts = [np.asarray(p, int) for p in np.array_split(order, n_sub)]
        self.n_sub = n_sub
        self.base = []
        for s in range(n_sub):
            va = self.parts[s]
            tr = np.concatenate([self.parts[g] for g in range(n_sub) if g != s])
            b = _fitj(X[tr], y[tr], kind, atoms=[], n_classes=n_classes,
                            passes=24, max_passes=48, conv_tol=1e-5)
            self.base.append((b, tr, va))

    def gains(self, atype, j, k):
        g = np.zeros(self.n_sub)
        for s in range(self.n_sub):
            b, tr, va = self.base[s]
            try:
                at = R.make_atom77(atype, j, k, b, self.X[tr])
            except (IndexError, ValueError):
                continue   # degenerate support: zero gain on this subfold
            at.ensure(self.kind, self.K)
            at.refit(self.X[tr], self.y[tr], b.score(self.X[tr]), self.kind,
                     kappa_mult=1.0, n_classes=self.K)
            eta_va = b.score(self.X[va])
            eta2 = eta_va + at.contrib(self.X[va])
            g[s] = float(nats_pieces(self.kind, self.y[va], eta_va, eta2,
                                     self.sigma2, 1.0).sum())
        return g


# ----------------------------------------------------------------------------
# fold runner
# ----------------------------------------------------------------------------
def run_fold77(panel, fold, tr_idx, te_idx, data, out, K_pairs=20,
               star_types=("g4a", "g4b", "g4c", "g4d", "p1", "m2"),
               K_refresh=20, max_admit=40, seed=109, pool_mode="screen",
               save_dir=None, ledger_mode="toof", frag_mode="w1", anchor_oos=False,
               screen_mode="legacy", oof_merge=False, contrast_rot=False):
    kind = data["kind"]
    X, y = data["X"], data["y"]
    n_classes = int(np.max(y)) + 1 if kind == "multiclass" else 0
    Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
    t0 = time.time()

    # R82-2 (#6): OOF error-contrast basis Q (multiclass only)
    Qrot = None
    if contrast_rot and kind == "multiclass":
        import r82_nomcon as NC
        rngq = np.random.default_rng(seed)
        parts = [np.asarray(pp, int) for pp in
                 np.array_split(rngq.permutation(len(ytr)), 5)]
        Rr = np.zeros((len(ytr), n_classes))
        for s5 in range(5):
            va = parts[s5]
            ftr = np.concatenate([parts[g] for g in range(5) if g != s5])
            bq = _fitj(Xtr[ftr], ytr[ftr], kind, atoms=[], n_classes=n_classes)
            Sq = np.asarray(bq.score(Xtr[va]), float)
            Pq = np.exp(Sq - Sq.max(axis=1, keepdims=True))
            Pq /= Pq.sum(axis=1, keepdims=True)
            Yq = np.zeros_like(Pq)
            Yq[np.arange(len(va)), ytr[va].astype(int)] = 1.0
            Rr[va] = Yq - Pq
        Qrot = NC.contrast_Q(Rr)
    rot_kw = {"final_rot": Qrot} if Qrot is not None else {}
    base = _fitj(Xtr, ytr, kind, atoms=[], n_classes=n_classes, **rot_kw)
    m0, _ = r71.metric(kind, yte, base.score(Xte))
    L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="base", metric=m0,
                          passes=base.meta["passes"], secs=round(time.time() - t0, 1)))
    print("   base", round(m0, 5), {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in base.meta.items() if k in ("passes", "delta")},
          flush=True)
    if kind != "regression":
        cm = S.calib_metrics(kind, Xtr, ytr, Xte, yte,
                             lambda Xa, ya, ka: _fitj(Xa, ya, ka, atoms=[],
                                                            n_classes=n_classes))
        for lbl, v in cm.items():
            L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="base_" + lbl,
                                  metric=v, secs=round(time.time() - t0, 1)))
            print("   base_" + lbl, round(v, 5), flush=True)

    # ---- dominant main j* (Rule 12.B-3: no column names, learned weights) ----
    wj = []
    for j in range(len(base.cuts)):
        Vj = np.asarray(base.V[j], float)
        Hj = np.asarray(base.meta["H"][j], float)
        if Vj.ndim == 2:
            Vj = np.linalg.norm(Vj, axis=1)
        wj.append(float((Hj * Vj ** 2).sum()))
    jstar = int(np.argmax(wj))
    L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="jstar",
                          j=jstar, share=float(wj[jstar] / max(sum(wj), 1e-12)),
                          secs=round(time.time() - t0, 1)))
    print("   jstar", jstar, "share", round(wj[jstar] / max(sum(wj), 1e-12), 3), flush=True)

    # ---- candidates: top-K pairs x (4-ray star + p1 + m2) ------------------
    eta_tr = base.score(Xtr)
    if pool_mode == "active":
        # step 3: ALL pairs among active mains (V_j != 0), name-blind
        act = []
        for j in range(len(base.cuts)):
            Vj = np.asarray(base.V[j], float)
            if float(np.max(np.abs(Vj))) > 1e-12 and np.isfinite(Xtr[:, j]).any():
                act.append(j)
        pairs = [(j, k) for i, j in enumerate(act) for k in act[i + 1:]]
    else:
        if screen_mode == "legacy":
            pairs = S.pair_screen(Xtr, ytr, eta_tr, kind, K_pairs)
        else:
            import r81_screen12 as SC
            pairs = SC.pair_screen12(Xtr, ytr, eta_tr, kind, K_pairs,
                                     mode="pure" if screen_mode == "s12_pure"
                                     else "hybrid")
    cands = []
    for (jj, kk) in pairs:
        for at in star_types:
            if at == "mod":
                for g in range(Xtr.shape[1]):
                    if g != jj and g != kk:
                        cands.append((f"mod{g}:{jj}-{kk}", f"mod{g}", jj, kk))
            elif at == "mcond":
                continue        # independent family, built below
            else:
                cands.append((f"{at}:{jj}-{kk}", at, jj, kk))
    if "mcond" in star_types:
        # R82 Part 1 (H8): missingness-conditioned candidates (j | M_k),
        # support gate >=25 rows in both modes, top-12 by orthogonal score.
        import r82_mcond as MC
        for (jj, kk, _sc) in MC.screen_mcond(Xtr, ytr, eta_tr, kind,
                                             K=12, min_rows=25):
            cands.append((f"mcond{jj}|{kk}:{jj}-{kk}", "mcond", jj, kk))
    M_tot = max(len(cands), 2)
    # R81-5 (H6): explicit context price ln(p-2) for mod candidates
    mod_price = float(np.log(max(Xtr.shape[1] - 2, 1)))
    kappa_block = 0.5 * np.log(max(len(ytr), 2))
    lnM = float(np.log(M_tot))

    if ledger_mode == "preq":
        # ---- R76 canonical prequential dual-gate (spec: T = toof) ----------
        import r76_stage31 as S31
        try:
            toof = float(r71.fit_T(ytr, eta_tr, kind))
        except Exception:
            toof = float(np.mean((ytr - eta_tr) ** 2)) if kind == "regression" else 1.0
        preq_types = ("g4", "p1", "m2")   # r76 atom vocabulary (mechanism check)
        if screen_mode == "legacy":
            preq_pairs = S.pair_screen(Xtr, ytr, eta_tr, kind, 12)
        else:
            import r81_screen12 as SC
            preq_pairs = SC.pair_screen12(Xtr, ytr, eta_tr, kind, 12,
                                          mode="pure" if screen_mode == "s12_pure"
                                          else "hybrid")
        preq_cands = [(f"{at}:{j}-{k}", at, j, k) for (j, k) in preq_pairs for at in preq_types]
        gains = S31.prequential_gains(preq_cands, Xtr, ytr, kind, n_classes,
                                      seed=seed, split="random")
        kappa_child = 0.5 * np.log(max(len(ytr), 2))
        lnM_p = float(np.log(3 * max(len(preq_pairs), 1)))
        admitted = []
        admitted_tags = set()
        for (j, k) in preq_pairs:
            best = None
            for at in preq_types:
                tag = f"{at}:{j}-{k}"
                g = np.asarray(gains[tag], float)
                if kind == "regression":
                    g = g / max(toof, 1e-9)
                net = float(g.sum()) - (kappa_child + lnM_p)
                n_pos = int((g > 0).sum())
                v = dict(tag=tag, type=at, j=j, k=k, net=net, gain=float(g.sum()),
                         subs_pos=n_pos, dq=1,
                         frag=bool(n_pos < int(np.ceil(0.8 * len(g)))),
                         subs=g.round(3).tolist())
                L.log_jsonl(out, dict(stage="77preq", panel=panel, fold=fold,
                                      arm="ledger", **v,
                                      secs=round(time.time() - t0, 1)))
                if v["frag"] or net <= 0:
                    continue
                if best is None or net > best["net"]:
                    best = v
            if best is not None:
                admitted.append(best)
                admitted_tags.add(best["tag"])
                L.log_jsonl(out, dict(stage="77preq", panel=panel, fold=fold,
                                      arm="admit", **best, n_admit=len(admitted),
                                      secs=round(time.time() - t0, 1)))
                print(f"   preq_admit#{len(admitted)}", best["tag"], "net",
                      round(best["net"], 1), "gain", round(best["gain"], 1),
                      "subs_pos", best["subs_pos"], flush=True)
        if admitted:
            atoms = []
            for v in admitted:
                at = S31.make_atom(v["type"], v["j"], v["k"], base, X=Xtr)
                at.ensure(kind, n_classes)
                at.refit(Xtr, ytr, eta_tr, kind, kappa_mult=1.0, n_classes=n_classes)
                atoms.append(at)
            card = S.fit_joint(Xtr, ytr, kind, atoms=atoms, n_classes=n_classes,
                               warm_card=base, passes=6, conv_tol=0.0)
        else:
            card = base
        card = CalibCard(card, kind, n_classes)
        m1, _ = r71.metric(kind, yte, card.score(Xte))
        L.log_jsonl(out, dict(stage="77preq", panel=panel, fold=fold, arm="arbitration",
                              metric=m1, n_admit=len(admitted),
                              types=[v["type"] for v in admitted],
                              secs=round(time.time() - t0, 1)))
        print("   arbitration", round(m1, 5), "n_admit", len(admitted), flush=True)
        if kind != "regression":
            cm = S.calib_metrics(kind, Xtr, ytr, Xte, yte,
                                 lambda Xa, ya, ka: _fitj(Xa, ya, ka, atoms=[],
                                                          n_classes=n_classes))
            for lbl, v in cm.items():
                L.log_jsonl(out, dict(stage="77preq", panel=panel, fold=fold,
                                      arm="arbitration_" + lbl, metric=v,
                                      n_admit=len(admitted),
                                      secs=round(time.time() - t0, 1)))
                print("   arbitration_" + lbl, round(v, 5), flush=True)
        return
    # ---- (toof path continues) -------------------------------------------
    sigma2 = 1.0
    if kind == "regression":
        sigma2 = float(np.mean((ytr - eta_tr) ** 2))
    led = Ledger(Xtr, ytr, kind, n_classes, sigma2=sigma2, seed=seed,
                 kappa_block=kappa_block, lnM=lnM, jstar=jstar, frag_mode=frag_mode)
    print(f"   ledger: {len(cands)} cands, sigma2={sigma2:.3f}, "
          f"barrier={kappa_block + lnM:.2f} nats", flush=True)

    admitted = []      # list of dicts in admit order
    admitted_tags = set()
    recs = led.eval(cands)
    _top = sorted((v["gain"] for v in recs.values()), reverse=True)[:5]
    print(f"   ledger_top_nats={[round(x, 2) for x in _top]} "
          f"(barrier {kappa_block + lnM:.2f})", flush=True)
    # contour 2 (REJECTED config W2/W4: over-strict) — kept off; W1 ledger only
    live = list(recs.values())
    for v in live:
        v["net"] = v["gain"] - (kappa_block + lnM
                                + (mod_price if str(v["type"]).startswith("mod")
                                   else 0.0))
        L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="ledger", **v,
                              secs=round(time.time() - t0, 1)))
    while len(admitted) < max_admit:
        best = None
        for v in live:
            if (v["tag"] in admitted_tags or v["frag"] or False
                    or v["net"] <= 0 or v["subs_pos"] < 4):
                continue
            if str(v["type"]).startswith("mod"):
                # strong heredity (H6): modulation needs an admitted pair atom
                if not any(a2["j"] == v["j"] and a2["k"] == v["k"]
                           and not str(a2["type"]).startswith("mod")
                           for a2 in admitted):
                    continue
            if best is None or v["net"] > best["net"]:
                best = v
        if best is None:
            break
        admitted.append(best)
        admitted_tags.add(best["tag"])
        led.extend_chain(best["type"], best["j"], best["k"])
        L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="admit", **best,
                              n_admit=len(admitted), secs=round(time.time() - t0, 1)))
        print(f"   admit#{len(admitted)}", best["tag"], "net", round(best["net"], 1),
              "gain", round(best["gain"], 1), "subs", best["subs_pos"], flush=True)
        # K-best refresh on the updated chain residual
        pool = [v for v in live if v["tag"] not in admitted_tags]
        pool.sort(key=lambda v: -v["net"])
        refresh = [(v["tag"], v["type"], v["j"], v["k"]) for v in pool[:K_refresh]]
        if not refresh:
            break
        rr = led.eval(refresh)
        for v in live:
            if v["tag"] in rr:
                old_net = v["net"]
                v.update(rr[v["tag"]])
                v["net"] = v["gain"] - (kappa_block + lnM
                                        + (mod_price
                                           if str(v["type"]).startswith("mod")
                                           else 0.0))
                if abs(v["net"] - old_net) > 1e-9:
                    L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold,
                                          arm="ledger_refresh", **v,
                                          n_admit=len(admitted),
                                          secs=round(time.time() - t0, 1)))

    # ---- deployed construction: full-train stagewise chain + warm backfit ---
    ats = []
    eta_cur = base.score(Xtr)
    for v in admitted:
        at = R.make_atom77(v["type"], v["j"], v["k"], base, X=Xtr)
        at.ensure(kind, n_classes)
        at.refit(Xtr, ytr, eta_cur, kind, kappa_mult=1.0, n_classes=n_classes)
        eta_cur = eta_cur + at.contrib(Xtr)
        ats.append(at)
    if ats:
        card = _fitj(Xtr, ytr, kind, atoms=ats, n_classes=n_classes,
                           warm_card=base, passes=FITCFG['warm_passes'],
                           conv_tol=0.0, **rot_kw)
    else:
        card = base   # zero admits -> deployed == base (invariant)
    if oof_merge:
        # R82-2 (Opus #5): prequential OOF segment merger, post-fit
        import r82_merge as RM
        card = RM.merge_card(card, Xtr, ytr, kind, n_classes, n_sub=5, seed=seed)
        print("   oof_merge merges:", card.meta.get("oof_merges"), flush=True)
    if anchor_oos and ats:
        # ---- OOS-Ray Anchor (user §2.1): after warm6 the edge bins of the
        # admitted-atom features carry the atom-compensation drift; OOS test rows
        # clamp the atom to 0 and hit an artificial step.  Extend cuts with the
        # train-support thresholds and give the two new OOS bins the PRE-warm
        # base edge levels -> chain behaviour outside [x_min, x_max], warm6
        # inside; 0 train rows change prediction.
        feats = sorted({int(v["j"]) for v in admitted} | {int(v["k"]) for v in admitted})
        for j in feats:
            xj = Xtr[:, j]
            fin = xj[np.isfinite(xj)]
            lo, hi = float(fin.min()), float(fin.max())
            Vb = np.asarray(base.V[j], float)
            Vw = np.asarray(card.V[j], float)
            card.cuts[j] = np.concatenate([[lo - 1e-7],
                                           np.asarray(card.cuts[j], float),
                                           [hi + 1e-7]])
            if Vw.ndim == 1:
                card.V[j] = np.concatenate([[Vb[0]], Vw, [Vb[-1]]])
            else:
                card.V[j] = np.vstack([Vb[0:1], Vw, Vb[-1:]])
        d_tr = float(np.max(np.abs(card.score(Xtr) - base.score(Xtr))))
        L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="anchor_oos",
                              feats=feats, drift_train_vs_base=d_tr,
                              secs=round(time.time() - t0, 1)))
        print("   anchor_oos feats", feats, "train drift vs base", round(d_tr, 4), flush=True)
    if save_dir:
        import pickle, os
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{panel}_f{fold}.pkl"), "wb") as fh:
            pickle.dump(dict(card=card, kind=kind, n_classes=n_classes,
                             admitted=[(v["type"], v["j"], v["k"]) for v in admitted]), fh)
    m1, _ = r71.metric(kind, yte, card.score(Xte))
    L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="arbitration",
                          metric=m1, n_admit=len(admitted),
                          types=[v["type"] for v in admitted],
                          secs=round(time.time() - t0, 1)))
    print("   arbitration", round(m1, 5), "n_admit", len(admitted), flush=True)

    # ablation arm: the stagewise chain WITHOUT the warm joint backfit — shows
    # how much edge-fold movement comes from mains drift in the backfit
    class ChainModel:
        def __init__(self, base, atoms):
            self.b, self.atoms = base, atoms

        def score(self, X):
            s = self.b.score(X)
            for a in self.atoms:
                s = s + a.contrib(X)
            return s

    m_c, _ = r71.metric(kind, yte, ChainModel(base, ats).score(Xte))
    L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="chain_only",
                          metric=m_c, n_admit=len(admitted),
                          secs=round(time.time() - t0, 1)))
    print("   chain_only", round(m_c, 5), flush=True)
    if kind != "regression":
        def fac(Xf, yf, kf):
            b = _fitj(Xf, yf, kf, atoms=[], n_classes=n_classes, **rot_kw)
            if not admitted:
                if oof_merge:
                    import r82_merge as RM
                    b = RM.merge_card(b, Xf, yf, kf, n_classes, n_sub=5, seed=seed)
                return b      # deployed = base invariant (zero admits)
            e = b.score(Xf)
            aa = []
            for v in admitted:
                try:
                    at = R.make_atom77(v["type"], v["j"], v["k"], b, X=Xf)
                except (IndexError, ValueError):
                    continue   # degenerate support in this calibration subfold
                at.ensure(kf, n_classes)
                at.refit(Xf, yf, e, kf, kappa_mult=1.0, n_classes=n_classes)
                e = e + at.contrib(Xf)
                aa.append(at)
            cm2 = _fitj(Xf, yf, kf, atoms=aa, n_classes=n_classes,
                               warm_card=b, passes=FITCFG['warm_passes'],
                               conv_tol=0.0, **rot_kw)
            if oof_merge:
                import r82_merge as RM
                cm2 = RM.merge_card(cm2, Xf, yf, kf, n_classes, n_sub=5, seed=seed)
            return cm2
        cm = S.calib_metrics(kind, Xtr, ytr, Xte, yte, fac)
        for lbl, v in cm.items():
            L.log_jsonl(out, dict(stage="77a", panel=panel, fold=fold, arm="arbitration_" + lbl,
                                  metric=v, n_admit=len(admitted),
                                  secs=round(time.time() - t0, 1)))
            print("   arbitration_" + lbl, round(v, 5), flush=True)
    return m0, m1
