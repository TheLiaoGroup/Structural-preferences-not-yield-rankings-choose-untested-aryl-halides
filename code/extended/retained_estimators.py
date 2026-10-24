"""Estimators of the second Buchwald-Hartwig campaign: contributions within connected blocks, reliabilities and
the comparisons with the analysed reactions."""
from __future__ import annotations
import itertools
import math
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import classification as S
SEED, BOOT = 20261001, 2000
N_COND, MIN_COND, MIN_PAIRS, MIN_SIDE = 20, 10, 50, 10
MIN_SHARED, MIN_BLOCK_PAIRS = 5, 10
CAMPAIGN = "Buchwald-Hartwig(CMC)"  # the second Buchwald-Hartwig campaign
SAME_REACTION = [{"Suzuki", "Suzuki Bpin"}, {"Buchwald-Hartwig", "Buchwald-Hartwig(CMC)"}]

def factor(x):
    codes, names = pd.factorize(np.asarray(x))
    return codes, names


def two_way(y, left, right):
    """Least squares y ~ left + right with sparse indicators (the frozen analysis_bba.two_way_fit
    returns only the centred left effects; both factors are needed here)."""
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    li, ln = factor(left)
    ri, rn = factor(right)
    n = len(y)
    X = sparse.csr_matrix((np.ones(2 * n), (np.repeat(np.arange(n), 2),
                                           np.ravel(np.column_stack([li, len(ln) + ri])))),
                          shape=(n, len(ln) + len(rn)))
    beta = lsqr(X, np.asarray(y, float), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    return pd.Series(beta[:len(ln)], index=ln), pd.Series(beta[len(ln):], index=rn)


def bipartite_components(a, b):
    ai, an = factor(a)
    bi, bn = factor(b)
    size = len(an) + len(bn)
    g = coo_matrix((np.ones(len(ai)), (ai, len(an) + bi)), shape=(size, size))
    lab = connected_components(g, directed=False)[1]
    return lab[ai]


def largest_pairs(pairs):
    lab = bipartite_components(pairs.elec.values, pairs.nuc.values)
    return pairs[lab == np.bincount(lab).argmax()]


def contributions(rows, A):
    """Both stages of the frozen estimator on one reaction's wells (single condition block)."""
    pairs = largest_pairs(rows[["elec", "nuc"]].drop_duplicates())
    rows = rows.merge(pairs, on=["elec", "nuc"])
    eff = A.two_way_fit(rows["y"].values, rows.elec + " >> " + rows.nuc, rows.condition_id)
    table = pd.DataFrame([k.split(" >> ") for k in eff.index], columns=["elec", "nuc"]).assign(p=eff.values)
    st = A.stage_two(table)
    return {"elec": st["a"], "nuc": st["b"]}, len(rows), len(table)


def contributions_blocks(rows, A):
    """For reactions whose conditions come in kits run on disjoint pairs (BH(CMC) kits A/B/C):
    stage one within each connected pair-condition block, stage two with block intercepts."""
    rows = rows.copy()
    rows["pair"] = rows.elec + " >> " + rows.nuc
    rows["block"] = bipartite_components(rows.pair.values, rows.condition_id.values)
    sizes = rows.groupby("block").pair.nunique()
    rows = rows[rows.block.isin(sizes[sizes >= MIN_BLOCK_PAIRS].index)]
    pieces = []
    for b, g in rows.groupby("block"):
        eff = A.two_way_fit(g["y"].values, g.pair, g.condition_id)
        pieces.append(pd.DataFrame({"pair": eff.index, "p": eff.values, "block": b}))
    t = pd.concat(pieces, ignore_index=True)
    t[["elec", "nuc"]] = t.pair.str.split(" >> ", expand=True)
    t = t.merge(largest_pairs(t[["elec", "nuc"]].drop_duplicates()), on=["elec", "nuc"])
    ei, en = factor(t.elec)
    ni, nn = factor(t.nuc)
    bi, bn = factor(t.block)
    X = np.zeros((len(t), 1 + (len(en) - 1) + (len(nn) - 1) + (len(bn) - 1)))
    X[:, 0] = 1
    r = np.arange(len(t))
    X[r[ei > 0], ei[ei > 0]] = 1
    X[r[ni > 0], len(en) - 1 + ni[ni > 0]] = 1
    X[r[bi > 0], len(en) + len(nn) - 2 + bi[bi > 0]] = 1
    beta = np.linalg.lstsq(X, t.p.values, rcond=None)[0]
    a = np.concatenate([[0.0], beta[1:len(en)]])
    b = np.concatenate([[0.0], beta[len(en):len(en) + len(nn) - 1]])
    return ({"elec": pd.Series(a - a.mean(), index=en), "nuc": pd.Series(b - b.mean(), index=nn)},
            len(rows), len(t), int(len(bn)))


def coverage_order(rows):
    cov = (rows.drop_duplicates(["condition_id", "elec", "nuc"]).groupby("condition_id").size()
           .rename("n").reset_index().sort_values(["n", "condition_id"], ascending=[False, True]))
    return cov.condition_id.tolist()


def halves_reliability(rows, A, G, blocks=False):
    """Condition halves alternating in coverage order (within each block when blocks=True);
    both stages refitted in each half; Spearman-Brown of the half-to-half rank correlation."""
    rows = rows.copy()
    if blocks:
        rows["pair"] = rows.elec + " >> " + rows.nuc
        rows["block"] = bipartite_components(rows.pair.values, rows.condition_id.values)
    else:
        rows["block"] = 0
    half = {}
    for _, g in rows.groupby("block"):
        for k, c in enumerate(coverage_order(g)):
            half[c] = k % 2
    rows["half"] = rows.condition_id.map(half)
    fits = []
    for h in (0, 1):
        sub = rows[rows.half == h].drop(columns=["block", "half"] + (["pair"] if blocks else []))
        fits.append(contributions_blocks(sub, A)[0] if blocks else contributions(sub, A)[0])
    out = {}
    for side in ("elec", "nuc"):
        common = fits[0][side].index.intersection(fits[1][side].index)
        r = G.spearman(fits[0][side][common], fits[1][side][common]) if len(common) >= MIN_SHARED else np.nan
        out[side] = {"molecules": int(len(common)), "r_half": r,
                     "reliability": 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan}
    return out


def pool_draws(cells, rng):
    """Shared-molecule-weighted mean of corrected correlations, with molecule resamples within each
    comparison (as chemistry_readings.pool, returning the draws for contrasts)."""
    n = np.array([len(va) for va, _, _ in cells], float)
    obs = np.array([f(va, vb) for va, vb, f in cells])
    draws = np.full((len(cells), BOOT), np.nan)
    for k, (va, vb, f) in enumerate(cells):
        idx = rng.integers(0, len(va), (BOOT, len(va)))
        draws[k] = [f(va[i], vb[i]) if len(np.unique(i)) >= MIN_SHARED else np.nan for i in idx]
    keep = np.isfinite(obs)
    ok = np.isfinite(draws)
    w = np.where(ok, n[:, None], 0.0)
    pooled = (np.where(ok, draws, 0.0) * w).sum(axis=0) / w.sum(axis=0)
    est = float(np.average(obs[keep], weights=n[keep])) if keep.any() else np.nan
    return est, pooled


def summarize(est, draws):
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi)}


def campaign(rows, frozen_series, frozen_rel, labels, lookup, A, G):
    """The second Buchwald-Hartwig campaign: contributions fitted within each connected block of
    pairs and conditions, condition-half reliabilities, and its rank agreement with every analysed
    reaction on the molecules they share, corrected by both reliabilities."""
    g = rows[rows.rxn_class == CAMPAIGN]
    if g.empty:
        raise RuntimeError(f"No usable rows for {CAMPAIGN}")
    fit, n_wells, n_pairs, n_blocks = contributions_blocks(g, A)
    rel = halves_reliability(g, A, G, blocks=True)
    top = coverage_order(g)[:N_COND]
    metal = S.metal_of(lookup, CAMPAIGN, top)
    ekind, _ = S.kind_of(fit["elec"].index.tolist())
    nkind, _ = S.kind_of(fit["nuc"].index.tolist())
    info = pd.DataFrame([{"reaction": CAMPAIGN, "wells": n_wells, "pairs": n_pairs, "blocks": n_blocks, "metal": metal,
                          "electrophile": ekind, "nucleophile": nkind,
                          "reliability_elec": rel["elec"]["reliability"], "reliability_nuc": rel["nuc"]["reliability"]}])
    comps = []
    for r in labels.index:
        for side in ("elec", "nuc"):
            f = frozen_series.get((r, side))
            if f is None:
                continue
            shared = f.index.intersection(fit[side].index)
            if len(shared) < MIN_SHARED:
                continue
            ceiling = math.sqrt(max(rel[side]["reliability"], 0) * max(frozen_rel[(r, side)], 0))
            fn = lambda u, v, c=ceiling: G.spearman(u, v) / c if c > 0 else np.nan
            va, vb = fit[side][shared].values, f[shared].values
            comps.append({"side": side, "analysed_reaction": r, "same_reaction": {CAMPAIGN, r} in SAME_REACTION,
                          "shared": len(shared), "rho": G.spearman(va, vb), "rho_corrected": fn(va, vb)})
    return fit, info, pd.DataFrame(comps)


