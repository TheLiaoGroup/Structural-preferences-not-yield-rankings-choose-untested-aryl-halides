"""Simulation settings, replicate draws and summaries of the interval calibration (Supplementary Note 6).
The layouts come from layout_inputs.py; run_calibration.py runs the replicates and writes the outputs.
"""
import argparse

import hashlib

import math

import os

import pickle

import sys

import tempfile

import time

from multiprocessing import Pool

import numpy as np

import pandas as pd

from scipy.special import expit

from scipy.stats import beta as beta_dist

from scipy.stats import norm

import design_extended_bba as E

import gated_layouts as G

import lib_bba as L

DEV_ROOT, CONFIRM_ROOT = 20261102, 20261103

BASE = dict(sa2=0.325, sb2=0.325, sd2=0.35, v=4.0, sgam2=1.0, f=0.3, tau=0.5, sq2=0.0, ss=0.0, calib="",
            sdelta2=0.0, h="", spe2=0.0, spn2=0.0, stau=0.0, omega=False, miss="", rho=1.0)

REPRESENTATIVE = ("Acylation", "Buchwald-Hartwig", "Kumada", "Ullmann", "Sonogashira")

MISSING = ("mar 0.1", "mar 0.3", "mnar 0.2")

LAYOUTS = {}

def to_yield(v, f):
    """Recorded yield on 0-1: an affine map of the latent value v that puts the fraction
    max(f, 0.005) of wells at or below 0.01 and 0.5% at or above 0.99, then clipping to [0, 1]."""
    lo, hi = np.quantile(v, [max(f, 0.005), 0.995])
    w = 0.98 / (hi - lo)
    return np.clip(0.01 + w * (v - lo), 0.0, 1.0), (0.01 - w * lo, w)

def noise_sd(prm):
    """Well noise: condition interaction g and residual e, each half of v times the central
    pair-effect variance (1.0); fixed across worlds, so removing a side does not remove noise."""
    return math.sqrt(prm["v"] / 2), math.sqrt(prm["v"] / 2)

def calibration(prm, rng, n_plate, e_of_plate_or_well):
    if prm["calib"] == "pair":
        eta = rng.normal(-prm["ss"] ** 2 / 2, prm["ss"], n_plate)
        return np.exp(eta), "plate"
    if prm["calib"] == "elec":
        eta = rng.normal(-prm["ss"] ** 2 / 2, prm["ss"], e_of_plate_or_well)
        return np.exp(eta), "elec"
    return None, ""

def h_of(prm, x):
    if not prm["h"]:
        return x
    part = prm["h"].split()
    if part[0] == "lin":
        return float(part[1]) * x
    return np.where(x < 0, float(part[1]) * x, float(part[2]) * x)

def draw_reaction(lay, prm, rng, a=None, b=None):
    """Latent values of one reaction's wells (Gate A and B worlds): z = a + b + d + gamma + g."""
    ne, nn = len(lay.e_name), len(lay.n_name)
    a = rng.normal(0, math.sqrt(prm["sa2"]), ne) if a is None else a
    b = rng.normal(0, math.sqrt(prm["sb2"]), nn) if b is None else b
    d = rng.normal(0, math.sqrt(prm["sd2"]), lay.n_pairs)
    gam = rng.normal(0, math.sqrt(prm["sgam2"]), len(lay.conds))
    sg, se = noise_sd(prm)
    e_, n_ = lay.pe[lay.w_pair], lay.pn[lay.w_pair]
    z = a[e_] + b[n_] + d[lay.w_pair] + gam[lay.w_cond] + rng.normal(0, sg, len(lay.w_pair))
    v = z + rng.normal(0, se, len(z))
    y, _ = to_yield(v, prm["f"])
    return y, z

def draw_contrast(cl, prm, rng):
    """Latent values of one contrast's wells: family A as draw_reaction; family B adds the switch
    effect tau_c, the rescaling h, side-specific terms a', b', a pair term delta and, in H1, an
    electrophile term that varies by condition. Calibration and plate constants per plate."""
    ne, nn, nc = len(cl.e_name), len(cl.n_name), len(cl.conds)
    e_, n_, p_, c_ = cl.pe[cl.w_pair], cl.pn[cl.w_pair], cl.w_pair, cl.w_cond
    a = rng.normal(0, math.sqrt(prm["sa2"]), ne)
    b = rng.normal(0, math.sqrt(prm["sb2"]), nn)
    d = rng.normal(0, math.sqrt(prm["sd2"]), cl.n_pairs)
    gam = rng.normal(0, math.sqrt(prm["sgam2"]), nc)
    tau = prm["tau"] + (rng.normal(0, prm["stau"], nc) if prm["stau"] else 0.0)
    isB = cl.w_family == "B"
    x = a[e_] + b[n_] + d[p_]
    zB = tau[c_] if np.ndim(tau) else np.full(len(x), tau)
    zB = zB + h_of(prm, x)
    if prm["spe2"]:
        ap = rng.normal(0, math.sqrt(prm["spe2"]), ne)
        zB = zB + ap[e_]
        if prm["omega"]:
            om = rng.normal(0, math.sqrt(prm["spe2"] / 2), (ne, nc))
            zB = zB + om[e_, c_]
    if prm["spn2"]:
        zB = zB + rng.normal(0, math.sqrt(prm["spn2"]), nn)[n_]
    if prm["sdelta2"]:
        zB = zB + rng.normal(0, math.sqrt(prm["sdelta2"]), cl.n_pairs)[p_]
    sg, se = noise_sd(prm)
    z = np.where(isB, zB, x) + gam[c_] + rng.normal(0, sg, len(x))
    n_plate = cl.w_plate.max() + 1
    s, kind = calibration(prm, rng, n_plate, ne)
    if kind == "plate":
        z = s[cl.w_plate] * z
    elif kind == "elec":
        z = s[e_] * z
    if prm["sq2"]:
        z = z + rng.normal(0, math.sqrt(prm["sq2"]), n_plate)[cl.w_plate]
    v = z + rng.normal(0, se, len(z))
    y, _ = to_yield(v, prm["f"])
    return y, z

def remove_wells(z, miss, rng):
    """World G: wells kept after removal at random, or more often at low latent yield."""
    kind, p = miss.split()
    p = float(p)
    if kind == "mar":
        return rng.random(len(z)) >= p
    zs = (z - z.mean()) / z.std()
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expit(mid - zs).mean() < p else (lo, mid)
    return rng.random(len(z)) >= expit((lo + hi) / 2 - zs)

def rebuild_reaction(lay, keep):
    wells, pairs, conds = lay.args
    w = wells[keep].reset_index(drop=True)
    kept = pairs.merge(w[["elec", "nuc"]].drop_duplicates(), on=["elec", "nuc"])
    return G.ReactionLayout(lay.rxn, w, kept, conds, w.plate if "plate" in w else None)

def rebuild_contrast(cl, keep):
    units, wells, hist_pairs, hist_conds, families = cl.args
    return G.ContrastLayout(cl.rxn, cl.fa, cl.fb, units, wells[keep].reset_index(drop=True), hist_pairs,
                            hist_conds, families)

def clipped_mean(mu, sd):
    """E[min(max(X, 0), 1)] for X ~ N(mu, sd^2)."""
    a0, a1 = (0 - mu) / sd, (1 - mu) / sd
    return sd * (norm.pdf(a0) - norm.pdf(a1)) + mu * (norm.cdf(a1) - norm.cdf(a0)) + (1 - norm.cdf(a1))

def side_variances(X):
    """Variance of the row and column effects of a full grid, less the share of the residual that
    averaging over the other side leaves in them (balanced two-way layout)."""
    n1, n2 = X.shape
    g = X.mean()
    r, c = X.mean(1) - g, X.mean(0) - g
    res = X - g - r[:, None] - c[None, :]
    ms = float((res ** 2).sum() / ((n1 - 1) * (n2 - 1)))
    return float(r.var(ddof=1) - ms / n2), float(c.var(ddof=1) - ms / n1), r, c, res

def observed_share(prm, rng, n=200, nc=20, draws=4):
    """The additive share of pair effects on the recorded 0-1 scale that a world's latent share
    becomes: noise-free expected yields of a full n x n grid at nc conditions, averaged over
    conditions, split into side and pair parts (Gate A and B cells; a diagnostic)."""
    sg, se = noise_sd(prm)
    shares = []
    for _ in range(draws):
        mu = (rng.normal(0, math.sqrt(prm["sa2"]), n)[:, None] + rng.normal(0, math.sqrt(prm["sb2"]), n)[None, :]
              + rng.normal(0, math.sqrt(prm["sd2"]), (n, n)))
        gam = rng.normal(0, math.sqrt(prm["sgam2"]), nc)
        allv = np.concatenate([(mu + gam[c] + rng.normal(0, sg, (n, n)) + rng.normal(0, se, (n, n))).ravel()
                               for c in range(nc)])
        lo, hi = np.quantile(allv, [max(prm["f"], 0.005), 0.995])
        w = 0.98 / (hi - lo)
        EY = np.mean([clipped_mean(0.01 + w * (mu + gam[c] - lo), w * math.sqrt(sg ** 2 + se ** 2)) for c in range(nc)], axis=0)
        VE, VN, *_ = side_variances(EY)
        shares.append((VE + VN) / EY.var(ddof=1))
    total = prm["sa2"] + prm["sb2"] + prm["sd2"]
    return {"latent additive share": (prm["sa2"] + prm["sb2"]) / total if total else np.nan,
            "observed additive share": float(np.mean(shares))}

def oracle(cl, prm, rng, n_mol=200, draws=8, max_unit_pairs=120):
    """Population truths of a contrast in a world: the side variances of the (historical-baseline-
    adjusted and unadjusted) differential response over populations of molecules, and the
    population cross-unit covariances of Section 7.7. Computed on a full grid of n_mol x n_mol
    new molecules at the contrast's own conditions, with the noise-free expected yield of every
    pair and condition (a clipped normal mean) and a realized noisy copy for the target variance
    and the adjustment coefficients. Independent of the estimators."""
    nc = len(cl.conds)
    condB = np.zeros(nc, bool)
    condB[np.unique(cl.w_cond[cl.w_family == "B"])] = True
    hist = np.asarray(cl.hist.conds)
    cix = {c: i for i, c in enumerate(cl.conds)}
    units = [([cix[c] for c in u["A"]], [cix[c] for c in u["B"]]) for u in cl.units]
    used = sorted(set(hist) | {c for u in units for c in u[0] + u[1]})
    sg, se = noise_sd(prm)
    acc = []
    n = n_mol
    for _ in range(draws):
        a = rng.normal(0, math.sqrt(prm["sa2"]), n)
        b = rng.normal(0, math.sqrt(prm["sb2"]), n)
        x = a[:, None] + b[None, :] + rng.normal(0, math.sqrt(prm["sd2"]), (n, n))
        gam = rng.normal(0, math.sqrt(prm["sgam2"]), nc)
        tau = prm["tau"] + (rng.normal(0, prm["stau"], nc) if prm["stau"] else np.zeros(nc))
        extra = np.zeros((n, n))
        ap = rng.normal(0, math.sqrt(prm["spe2"]), n) if prm["spe2"] else np.zeros(n)
        om = rng.normal(0, math.sqrt(prm["spe2"] / 2), (n, nc)) if prm["omega"] else np.zeros((n, nc))
        if prm["spn2"]:
            extra = extra + rng.normal(0, math.sqrt(prm["spn2"]), n)[None, :]
        if prm["sdelta2"]:
            extra = extra + rng.normal(0, math.sqrt(prm["sdelta2"]), (n, n))
        hx = h_of(prm, x)
        if prm["calib"] == "pair":
            s = np.exp(rng.normal(-prm["ss"] ** 2 / 2, prm["ss"], (n, n)))
        elif prm["calib"] == "elec":
            s = np.exp(rng.normal(-prm["ss"] ** 2 / 2, prm["ss"], n))[:, None] * np.ones((1, n))
        else:
            s = np.ones((n, n))
        q = rng.normal(0, math.sqrt(prm["sq2"]), (n, n)) if prm["sq2"] else np.zeros((n, n))
        mu = {}
        for c in used:
            mu[c] = (tau[c] + hx + ap[:, None] + om[:, c][:, None] + extra if condB[c] else x) + gam[c]
        V = {c: s * (mu[c] + rng.normal(0, sg, (n, n))) + q + rng.normal(0, se, (n, n)) for c in used}
        allv = np.concatenate([V[c].ravel() for c in used])
        lo, hi = np.quantile(allv, [max(prm["f"], 0.005), 0.995])
        w = 0.98 / (hi - lo)
        m0 = 0.01 - w * lo
        Y = {c: np.clip(m0 + w * V[c], 0, 1) for c in used}
        EY = {c: clipped_mean(m0 + w * (s * mu[c] + q), w * np.sqrt(s ** 2 * sg ** 2 + se ** 2)) for c in used}
        rec = {}
        if cl.kind != "regime":
            D = [Y[i[0]] - Y[j[0]] for i, j in units]
            ED = [EY[i[0]] - EY[j[0]] for i, j in units]
            t = np.mean([d_ - d_.mean() for d_ in D], axis=0)
            Et = np.mean([d_ - d_.mean() for d_ in ED], axis=0)
            Bh = np.mean([Y[c] for c in hist], axis=0)
            Bh = Bh - Bh.mean()
            EB = np.mean([EY[c] for c in hist], axis=0)
            EB = EB - EB.mean()
            lam = float(np.cov(t.ravel(), Bh.ravel())[0, 1] / Bh.var(ddof=1))
            c0 = float(t.mean() - lam * Bh.mean())
            for name, real, exp_ in (("adjusted", t - c0 - lam * Bh, Et - c0 - lam * EB), ("unadjusted", t, Et)):
                VE, VN, *_ = side_variances(exp_)
                VT = float(real.var(ddof=1))
                rec.update({f"{name} A_E": (VE - VN) / VT, f"{name} R2_ME": VE / VT, f"{name} R2_MN": VN / VT,
                            f"{name} I_EN": VE / VT, f"{name} I_NE": VN / VT})
            rec["lambda"] = lam
            ED_units = ED
        else:
            ED_units = [np.mean([EY[c] for c in A], axis=0) - np.mean([EY[c] for c in B], axis=0) for A, B in units]
        dis = cl.disjoint if cl.kind != "regime" else [(0, 1)]
        if len(dis) > max_unit_pairs:
            dis = [dis[i] for i in rng.choice(len(dis), max_unit_pairs, replace=False)]
        parts = []
        for u, v in dis:
            _, _, ru, cu, res_u = side_variances(ED_units[u])
            _, _, rv, cv, res_v = side_variances(ED_units[v])
            cross = float((res_u * res_v).sum() / ((n - 1) * (n - 1)))
            cee = float(np.cov(ru, rv)[0, 1] - cross / n)
            cnn = float(np.cov(cu, cv)[0, 1] - cross / n)
            cdd = float(np.cov(ED_units[u].ravel(), ED_units[v].ravel())[0, 1])
            parts.append((cee, cnn, cdd))
        cee, cnn, cdd = np.mean(parts, axis=0)
        rec.update({"C_EE": cee, "C_NN": cnn, "C_Delta": cdd,
                    "electrophile ratio": cee / (cee + cnn) if cee + cnn > 0 else np.nan,
                    "additive ratio": (cee + cnn) / cdd if cdd > 0 else np.nan})
        acc.append(rec)
    return {k: float(np.nanmean([r[k] for r in acc])) for k in acc[0]}

def rows_from(check, k, est, lo, hi, se, cat):
    return {"check": check, "degree": k, "estimate": est, "low": lo, "high": hi, "se": se, "category": cat}

def run_a(c, rng):
    lay, prm = LAYOUTS["A"][c["layout"]], c["prm"]
    y, z = draw_reaction(lay, prm, rng)
    if prm["miss"]:
        keep = remove_wells(z, prm["miss"], rng)
        lay, y = rebuild_reaction(lay, keep), y[keep]
    f = G.reaction_fits(lay, y)
    out = [rows_from(f"rho_P {r['side']}", r["degree"], r["rho_P"], r["low"], r["high"], r["se"], r["category"])
           for r in G.gate_a_partner(lay, f)]
    for r in G.gate_a_prediction(lay, y):
        for m in ("R2_ME", "R2_MN", "R2_MEN", "I_EN", "I_NE"):
            if r.get("status") != "ok":
                out.append(rows_from(m, r["degree"], np.nan, np.nan, np.nan, np.nan, r["status"]))
            else:
                lo, hi = r[f"{m} low"], r[f"{m} high"]
                out.append(rows_from(m, r["degree"], r[m], lo, hi, r[f"{m} se"], G.category(lo, hi)))
    return out

def run_b(c, rng):
    prm = c["prm"]
    lays = LAYOUTS["A"]
    shared = {s: rng.normal(0, 1.0, len(LAYOUTS["B names"][s])) for s in ("elec", "nuc")}
    fits, used = {}, {}
    for rc in sorted(lays):
        lay = lays[rc]
        mix = lambda s, var, n: math.sqrt(var) * (math.sqrt(prm["rho"]) * shared[s][LAYOUTS["B map"][rc][s]]
                                                   + math.sqrt(1 - prm["rho"]) * rng.normal(0, 1.0, n))
        a = mix("elec", prm["sa2"], len(lay.e_name))
        b = mix("nuc", prm["sb2"], len(lay.n_name))
        v_r = (1.0, 4.0, 16.0)[int(hashlib.sha256(rc.encode()).hexdigest(), 16) % 3]
        y, z = draw_reaction(lay, {**prm, "v": v_r}, rng, a, b)
        if prm["miss"]:
            keep = remove_wells(z, prm["miss"], rng)
            lay, y = rebuild_reaction(lay, keep), y[keep]
        fits[rc], used[rc] = G.reaction_fits(lay, y), lay
    out = []
    for k in G.DEGREES:
        res = G.gate_b(used, fits, k, bootstrap=2000 if k == 2 else 0, rng=rng)
        pr = res["pooled relative"]
        if pr is None:
            for m in ("D_G", "kappa", "D_A"):
                out.append(rows_from(m, k, np.nan, np.nan, np.nan, np.nan, G.NOT_ESTIMABLE))
        else:
            out.append(rows_from("D_G", k, pr["D_G"], pr["D_G low"], pr["D_G high"], pr["D_G se"], pr["D_G category"]))
            out.append(rows_from("kappa", k, pr["kappa"], pr["kappa low"], pr["kappa high"], pr["kappa se"],
                                 G.category(pr["kappa low"], pr["kappa high"])))
            out.append(rows_from("D_A", k, pr["D_A"], pr["D_A low"], pr["D_A high"], pr["D_A se"], ""))
        if k == 2:
            po = res["pooled observed"]
            out.append(rows_from("rho_R pooled, jackknife", k, po["weighted"], po["weighted low"], po["weighted high"],
                                 po["weighted se"], ""))
            out.append(rows_from("rho_R pooled, node bootstrap", k, po["weighted"], po.get("bootstrap low"),
                                 po.get("bootstrap high"), np.nan, ""))
    return out

def run_c(c, rng):
    cl, prm = LAYOUTS["C"][c["layout"]], c["prm"]
    y, z = draw_contrast(cl, prm, rng)
    if prm["miss"]:
        keep = remove_wells(z, prm["miss"], rng)
        cl, y = rebuild_contrast(cl, keep), y[keep]
    B = G.historical_baseline(cl, y)
    out = []
    for r in G.gate_c_prediction(cl, y, B):
        for m in ("A_E", "R2_ME", "R2_MN", "I_EN", "I_NE"):
            name = f"{m}, {r['target']}"
            if r.get("status") != "ok":
                out.append(rows_from(name, 2, np.nan, np.nan, np.nan, np.nan, r["status"]))
            else:
                lo, hi = r[f"{m} low"], r[f"{m} high"]
                out.append(rows_from(name, 2, r[m], lo, hi, r[f"{m} se"], G.category(lo, hi)))
    for r in G.gate_c_ranking(cl, y):
        out.append(rows_from(f"D_C {r['side']}", 2, r.get("D_C", np.nan), r.get("D_C low", np.nan),
                             r.get("D_C high", np.nan), r.get("D_C se", np.nan), r.get("category", r["status"])))
    cv = G.gate_c_covariances(cl, y)
    for m in ("electrophile ratio", "additive ratio"):
        out.append(rows_from(m, 2, cv.get(m, np.nan), np.nan, np.nan, np.nan, cv["status"]))
    return out

RUN = {"A": run_a, "B": run_b, "C": run_c}

def init(path):
    LAYOUTS.update(pickle.loads(open(path, "rb").read()))

def task(args):
    """Replicates lo..hi-1 of one cell, or the population truths of a Gate C cell (rep = -1)."""
    c, root, lo, hi = args
    rows = []
    for rep in range(lo, hi):
        rng = np.random.default_rng(np.random.SeedSequence(root, spawn_key=(c["cell"], max(rep, 0) if rep >= 0 else 10 ** 6)))
        if rep < 0:
            truth = (oracle(LAYOUTS["C"][c["layout"]], c["prm"], rng) if c["gate"] == "C"
                     else observed_share(c["prm"], rng))
            return [{"cell": c["cell"], "oracle": truth}]
        try:
            for r in RUN[c["gate"]](c, rng):
                rows.append({"cell": c["cell"], "rep": rep, **r})
        except Exception as err:  # reported, never dropped
            rows.append({"cell": c["cell"], "rep": rep, "check": "(replicate)", "category": f"error: {err!r}"[:300]})
    return rows

EVIDENCE = ("supporting evidence", "contrary evidence", "insufficient evidence")

NULL_C = {"N0", "B", "C1", "D", "H0", "N0+G", "F_N"}

SCENARIO_C = {"C2", "E1", "E2"}

def is_central(prm):
    return all(prm[k] == BASE[k] for k in ("v", "f", "tau"))

def truth_and_role(c, check, oracle_values):
    """(truth, role) of a check in a cell. Roles: 'null' (size is checked), 'alternative' (power
    is reported), 'scenario' (no chemical side change, but the observed-scale estimand can be
    non-zero; the support rate is reported), 'diagnostic' (no pass rule)."""
    w, prm = c["world"].replace("+G", ""), c["prm"]
    if c["gate"] == "A":
        null_side = {"F_E": "nuc", "F_N": "elec"}.get(w)
        if check.startswith("rho_P"):
            return (0.0, "null") if check.endswith(null_side or "-") else (None, "diagnostic")
        empty = {"nuc": ("R2_MN", "I_NE"), "elec": ("R2_ME", "I_EN")}.get(null_side, ())
        return (None, "null") if check in empty else (None, "diagnostic")
    if c["gate"] == "B":
        if check == "D_G":
            return (None, "null") if w == "A0" else (None, "alternative")
        if check == "kappa":
            return 6 / math.pi * math.asin(prm["rho"] / 2), "coverage"
        return None, "diagnostic"
    # Gate C
    if check.startswith("D_C"):
        side = check.split()[1]
        moved = (side == "elec" and w in {"F_E", "H1"}) or (side == "nuc" and w == "F_N")
        if moved:
            return None, "alternative"
        if side == "elec" and w == "C2":
            return None, "scenario"
        return None, "null"
    if check in ("electrophile ratio", "additive ratio"):
        return oracle_values.get(check), ("bias" if w in {"D", "F_E", "F_N"} and is_central(prm) else "diagnostic")
    metric, target = check.split(", ", 1)
    key = f"{'adjusted' if target == 'adjusted' else 'unadjusted'} {metric}"
    truth = oracle_values.get(key)
    if metric != "A_E":
        return truth, "coverage"
    if w in NULL_C:
        return truth, "null"
    return truth, ("scenario" if w in SCENARIO_C else "alternative")

def level_of(c, check):
    return G.LEVEL_PRIMARY if (c["gate"] == "C" and tuple(c["layout"]) in G.PRIMARY and check.startswith("A_E")) else G.LEVEL

def exact_ci(k, n):
    if n == 0:
        return np.nan, np.nan
    lo = beta_dist.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)

def summarise(cells_, rows, oracles):
    """One row per cell, check and degree, with the pre-specified rules of the annex (Section 6)."""
    D = pd.DataFrame(rows)
    out = []
    by_cell = {c["cell"]: c for c in cells_}
    for (cell, check, k), g in D[D.check != "(replicate)"].groupby(["cell", "check", "degree"], sort=True):
        c = by_cell[cell]
        truth, role = truth_and_role(c, check, oracles.get(cell, {}))
        level = level_of(c, check)
        alpha = (1 - level) / 2
        n = len(g)
        ok = g[g.category.isin(EVIDENCE) | ((g.category == "ok") & np.isfinite(g.estimate.astype(float)))
               | ((g.category == "") & np.isfinite(g.estimate.astype(float)))]
        structural = int((g.category == G.NOT_ESTIMABLE).sum())
        numerical = int((g.category == G.NUMERICAL).sum())
        undefined = int((g.category == G.UNDEFINED).sum())
        errors = int(g.category.astype(str).str.startswith("error").sum())
        m = len(ok)
        est = ok.estimate.astype(float)
        lo, hi = ok.low.astype(float), ok.high.astype(float)
        has_int = np.isfinite(lo) & np.isfinite(hi)
        sup = int((ok.category == "supporting evidence").sum())
        con = int((ok.category == "contrary evidence").sum())
        rec = {"cell": cell, "gate": c["gate"], "layout": " / ".join(c["layout"]) if isinstance(c["layout"], tuple) else c["layout"],
               "world": c["world"], **{k2: v for k2, v in c["prm"].items() if v != BASE[k2]},
               "check": check, "degree": k, "role": role, "level": level, "replicates": n, "estimable": m,
               "structurally not estimable": structural, "numerical failures": numerical, "undefined": undefined,
               "errors": errors,
               "mean estimate": float(est.mean()) if m else np.nan, "sd estimate": float(est.std(ddof=1)) if m > 1 else np.nan,
               "mean se": float(ok.se.astype(float).mean()) if m else np.nan, "truth": truth}
        if m:
            p = sup / m
            rec.update({"support rate": p, "support mc se": math.sqrt(p * (1 - p) / m),
                        "support ci low": exact_ci(sup, m)[0], "support ci high": exact_ci(sup, m)[1],
                        "contrary rate": con / m})
        if m and has_int.any():
            mu = float(est.mean())
            rec["bias-eliminated coverage (diagnostic)"] = float(((lo <= mu) & (mu <= hi))[has_int].mean())
            if truth is not None and np.isfinite(truth):
                cov = float(((lo <= truth) & (truth <= hi))[has_int].mean())
                rec.update({"bias": mu - truth, "coverage": cov, "coverage mc se": math.sqrt(cov * (1 - cov) / has_int.sum())})
        elif m and truth is not None:
            rec["bias"] = float(est.mean()) - truth
        # rules
        verdicts = []
        if n and (numerical + errors) / n > 0.10:
            verdicts.append("fail: numerical failures above 10%")
        if structural == n:
            verdicts.append("structurally not estimable")
        if role == "null" and m:
            bound = alpha + 2 * math.sqrt(alpha * (1 - alpha) / m)
            rec["size bound"] = bound
            verdicts.append("size pass" if rec["support rate"] <= bound else "fail: size")
        if role in ("null", "coverage", "alternative", "scenario") and "coverage" in rec:
            bound = level - 2 * math.sqrt(level * (1 - level) / has_int.sum())
            rec["coverage bound"] = bound
            verdicts.append("coverage pass" if rec["coverage"] >= bound else "fail: coverage")
        if role == "bias" and "bias" in rec:
            verdicts.append("bias pass" if abs(rec["bias"]) <= 0.10 else "fail: bias")
        rec["verdict"] = "; ".join(verdicts) or "reported"
        out.append(rec)
    S = pd.DataFrame(out)
    # family-wise error and simultaneous coverage of the two primary contrasts
    fw = []
    P = S[(S.gate == "C") & (S.check == "A_E, adjusted")]
    prim = [" / ".join(k) for k in G.PRIMARY]
    keycols = [c for c in BASE if c in P.columns]
    for _, g in P[P.layout.isin(prim)].groupby(["world"] + keycols, dropna=False):
        if len(g) != 2:
            continue
        a, b = g.iloc[0], g.iloc[1]
        rec = {"world": a.world, **{k: a[k] for k in keycols if pd.notna(a[k])}, "check": "A_E, adjusted: both primary contrasts"}
        if a.role == "null":
            p1, p2 = a["support rate"], b["support rate"]
            f = 1 - (1 - p1) * (1 - p2)
            m = min(a.estimable, b.estimable)
            bound = 0.025 + 2 * math.sqrt(0.025 * 0.975 / m)
            rec.update({"family-wise support rate": f, "size bound": bound,
                        "verdict": "size pass" if f <= bound else "fail: family-wise size"})
        if pd.notna(a.get("coverage")) and pd.notna(b.get("coverage")):
            sim = a.coverage * b.coverage
            bound = 0.95 - 2 * math.sqrt(0.95 * 0.05 / min(a.estimable, b.estimable))
            rec.update({"simultaneous coverage": sim, "simultaneous bound": bound})
            rec["verdict"] = (rec.get("verdict", "") + "; " if rec.get("verdict") else "") + \
                ("simultaneous coverage pass" if sim >= bound else "fail: simultaneous coverage")
        fw.append(rec)
    return S, pd.DataFrame(fw)
