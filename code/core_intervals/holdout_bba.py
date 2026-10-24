"""Agreement on the second condition set: within-reaction agreement with the first set, cross-reaction agreement and retention, with their resampling intervals."""
import hashlib

import itertools

import sys

import numpy as np

import pandas as pd

import analysis_bba as A

import design_extended_bba as E

import diagnose_transfer_bba as DG

import extended_transfer_bba as X

import lib_bba as L

import rank_bba as RK

SEED = 20260929

SEED_PLANTED = 20260930

N_BOOT = 2000

THRESHOLD = 0.5

SIDES = ("elec", "nuc")

def slice_of(conditions_file, pairs_file):
    """Pairs, and each reaction's conditions in coverage-rank order."""
    c = pd.read_csv(L.OUT / conditions_file, dtype={"condition_id": str})
    if "rank" in c.columns:
        c = c.sort_values(["rxn_class", "rank"], kind="stable")
    ranked = c.groupby("rxn_class", sort=True).condition_id.apply(list).to_dict()
    return pd.read_csv(L.OUT / pairs_file), ranked

def designs():
    return {"analysed": slice_of("extended_conditions_bba.csv", "extended_pairs_bba.csv"),
            "held": slice_of("holdout_conditions_bba.csv", "holdout_pairs_bba.csv")}

def expected_counts(d):
    """Cells and comparisons the criteria must score, from the molecule lists alone."""
    (pa, _), (ph, rh) = d["analysed"], d["held"]
    reactions = sorted(rh)
    within = sum(1 for rc in reactions for s in SIDES
                 if len(set(pa[pa.rxn_class == rc][s]) & set(ph[ph.rxn_class == rc][s])) >= X.MIN_SHARED)
    cross = sum(1 for s in SIDES for a, b in itertools.combinations(reactions, 2)
                if frozenset((a, b)) not in E.SAME_REACTION
                and len(set(ph[ph.rxn_class == a][s]) & set(ph[ph.rxn_class == b][s])) >= X.MIN_SHARED)
    return within, cross

def contributions(rows, pairs, ranked, halves=True):
    """Each molecule's contribution per reaction and side from the whole slice and, if asked,
    the reliability from the two halves of its conditions by parity of coverage rank."""
    full = X.main_effects(rows, pairs, {rc: set(v) for rc, v in ranked.items()})
    if not halves:
        return full, None
    parts = [X.main_effects(rows, pairs, {rc: {c for i, c in enumerate(v) if i % 2 == parity}
                                          for rc, v in ranked.items()})
             for parity in (0, 1)]
    rel = {}
    for rc in ranked:
        for side in SIDES:
            a, b = parts[0][rc][side], parts[1][rc][side]
            shared = a.index.intersection(b.index).sort_values()
            r = RK.spearman(a[shared], b[shared])
            rel[(rc, side)] = 2 * r / (1 + r) if r > -1 else np.nan
    return full, rel

def corrected(ceiling):
    return lambda u, v: RK.spearman(u, v) / ceiling if ceiling > 0 else np.nan

def selection(u, v):
    return X.retention(u, v)[0]

def pool(cells, rng, n_boot):
    """cells: (va, vb, statistic). The estimate is the mean of the statistic over cells weighted
    by the molecules each holds; the interval is the percentile interval over n_boot resamples of
    molecules within each cell."""
    n = np.array([len(va) for va, _, _ in cells], float)
    obs = np.array([f(va, vb) for va, vb, f in cells])
    draws = np.full((len(cells), n_boot), np.nan)
    for k, (va, vb, f) in enumerate(cells):
        idx = rng.integers(0, len(va), (n_boot, len(va)))
        draws[k] = [f(va[i], vb[i]) if len(np.unique(i)) >= X.MIN_SHARED else np.nan for i in idx]
    keep = np.isfinite(obs)
    estimate = float(np.average(obs[keep], weights=n[keep]))
    ok = np.isfinite(draws)
    w = np.where(ok, n[:, None], 0.0)
    pooled = (np.where(ok, draws, 0.0) * w).sum(axis=0) / w.sum(axis=0)
    lo, hi = np.percentile(pooled, [2.5, 97.5])
    return estimate, float(lo), float(hi)

def score(rows, d, rng, n_boot=N_BOOT):
    """C1, C2 and E1 from rows carrying yields."""
    (pa, ra), (ph, rh) = d["analysed"], d["held"]
    reactions = sorted(rh)
    full_a, _ = contributions(rows, pa, ra, halves=False)
    full_h, rel_h = contributions(rows, ph, rh)

    cells1 = []
    for rc in reactions:
        for side in SIDES:
            xa, xh = full_a[rc][side], full_h[rc][side]
            shared = xa.index.intersection(xh.index).sort_values()
            if len(shared) >= X.MIN_SHARED:
                cells1.append({"rxn_class": rc, "side": side, "n_shared": len(shared),
                               "rho": RK.spearman(xa[shared], xh[shared]),
                               "_va": xa[shared].values, "_vh": xh[shared].values})
    c1 = pool([(c["_va"], c["_vh"], RK.spearman) for c in cells1], rng, n_boot)

    recs, cells2, cells_sel = [], [], []
    for side in SIDES:
        for a, b in itertools.combinations(reactions, 2):
            xa, xb = full_h[a][side], full_h[b][side]
            shared = xa.index.intersection(xb.index).sort_values()
            if len(shared) < X.MIN_SHARED:
                continue
            va, vb = xa[shared].values, xb[shared].values
            ceiling = float(np.sqrt(max(rel_h[(a, side)], 0) * max(rel_h[(b, side)], 0)))
            same = frozenset((a, b)) in E.SAME_REACTION
            recs.append({"side": side, "rxn_a": a, "rxn_b": b, "n_shared": len(shared),
                         "same_reaction": same, "rho_obs": RK.spearman(va, vb), "ceiling": ceiling,
                         "rho_disattenuated": corrected(ceiling)(va, vb),
                         "retention": selection(va, vb) if len(shared) >= X.MIN_FOR_SELECTION else np.nan})
            if not same:
                cells2.append((va, vb, corrected(ceiling)))
                if len(shared) >= X.MIN_FOR_SELECTION:
                    cells_sel.append((va, vb, selection))
    T = pd.DataFrame(recs)
    c2 = pool(cells2, rng, n_boot)
    e1 = pool(cells_sel, rng, n_boot)

    reliability = pd.DataFrame([{"rxn_class": rc, "side": s, "reliability": v}
                                for (rc, s), v in rel_h.items()])
    return {"C1": c1, "C2": c2, "E1": e1, "cells": pd.DataFrame(cells1).drop(columns=["_va", "_vh"]),
            "transfer": T, "reliability": reliability,
            "full_a": full_a, "full_h": full_h}

WORLDS = {"independent": {"reaction": 1.0}, "shared": {"molecule": 1.0},
          "condition-specific": {"slice": 1.0},
          "mostly shared": {"molecule": 0.7, "reaction": 0.3},
          "mostly condition-specific": {"reaction": 0.3, "slice": 0.7}}

EXPECT = {"independent": (True, True), "shared": (True, False), "condition-specific": (False, True),
          "mostly shared": (True, False), "mostly condition-specific": (False, True)}

def amines(full_h):
    """E3: held-out contributions of the amines shared by Buchwald-Hartwig amination and SNAr,
    by the class of the frozen diagnosis."""
    bh, sn = full_h["Buchwald-Hartwig"]["nuc"], full_h["SNAr"]["nuc"]
    shared = sorted(bh.index.intersection(sn.index).sort_values())
    t = pd.DataFrame({"mol": shared, "Buchwald-Hartwig": bh[shared].values, "SNAr": sn[shared].values})
    t["class"] = t.mol.map(DG.classify)
    rest = t[t["class"] != "primary aryl amine"]
    return (t, RK.spearman(t["Buchwald-Hartwig"], t["SNAr"]),
            RK.spearman(rest["Buchwald-Hartwig"], rest["SNAr"]))

def verdicts(res):
    return (res["C1"][0] > THRESHOLD and res["C1"][1] > THRESHOLD,
            res["C2"][0] < THRESHOLD and res["C2"][2] < THRESHOLD)


