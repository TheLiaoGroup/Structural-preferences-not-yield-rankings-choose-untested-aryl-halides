"""Agreement of molecule contributions between reactions, retention of a top-ten selection, and their resampling intervals."""
import hashlib

import itertools

import numpy as np

import pandas as pd

from scipy.stats import spearmanr

import analysis_bba as A

import design_extended_bba as E

import lib_bba as L

import rank_bba as RK

SEED = 20260927

N_BOOT = 2000

MIN_SHARED = 5

MIN_FOR_SELECTION = 20

TOP_K = 10

def main_effects(rows, pairs, conditions):
    """Main effects of both sides for every reaction of the extended set."""
    out = {}
    for rc, p in pairs.groupby("rxn_class"):
        g = rows[(rows.rxn_class == rc) & rows.condition_id.isin(conditions[rc])]
        g = g.merge(p[["elec", "nuc"]], on=["elec", "nuc"])
        eff = A.two_way_fit(g["yield"].values, g.elec + " >> " + g.nuc, g.condition_id)
        key = pd.DataFrame([k.split(" >> ") for k in eff.index], columns=["elec", "nuc"])
        fit = A.stage_two(key.assign(p=eff.values))
        out[rc] = {"elec": fit["a"], "nuc": fit["b"]}
    return out

def retention(va, vb, k=TOP_K):
    """How much of the benefit of choosing good building blocks survives choosing them in the
    wrong reaction. 1 = as good as choosing in B; 0 = no better than random. Both selections go
    through RK.top_k_mean, so a tie straddling tenth place is shared rather than broken by the
    arithmetic; deviation 4."""
    if len(va) < k + 2:
        return np.nan, np.nan, np.nan, np.nan
    picked = RK.top_k_mean(va, vb, k)
    best = RK.top_k_mean(vb, vb, k)
    everything = float(np.mean(vb))
    span = best - everything
    return (picked - everything) / span if span > 1e-12 else np.nan, picked, best, everything


def reproduce(rows, pairs, conds):
    conditions = conds.groupby("rxn_class").condition_id.apply(set).to_dict()
    rows = rows[rows.rxn_class.isin(conditions)]
    reactions = sorted(conditions)
    print(f"\n{len(reactions)} reactions, {len(pairs):,} pairs, {len(set(pairs.elec) | set(pairs.nuc))} molecules")

    full = main_effects(rows, pairs, conditions)
    # Halved by parity of coverage rank, as both addenda state and as transfer_bba.py does.
    # An earlier version of this script sorted the identifiers as strings instead; deviation 2.
    rank = (pd.read_csv(L.OUT / "extended_conditions_bba.csv", dtype={"condition_id": str})
            .groupby("rxn_class").condition_id.apply(list).to_dict())
    assert set(rank) == set(conditions) and all(set(rank[rc]) == conditions[rc] for rc in rank)
    halves = []
    for parity in (0, 1):
        sel = {rc: {c for i, c in enumerate(rank[rc]) if i % 2 == parity} for rc in reactions}
        halves.append(main_effects(rows, pairs, sel))

    rel = []
    for rc in reactions:
        for side in ("elec", "nuc"):
            a, b = halves[0][rc][side], halves[1][rc][side]
            shared = a.index.intersection(b.index)
            r = RK.spearman(a[shared], b[shared])
            rel.append({"rxn_class": rc, "side": side, "n_molecules": len(shared),
                        "r_half": r, "reliability": 2 * r / (1 + r) if r > -1 else np.nan})
    REL = pd.DataFrame(rel)
    print("\n== reliability of a main effect (same reaction, disjoint half of its conditions)")
    print(f"   {REL.reliability.min():.3f} to {REL.reliability.max():.3f} over {len(REL)} reaction-side cells; "
          f"median {REL.reliability.median():.3f}")
    rmap = REL.set_index(["rxn_class", "side"]).reliability

    rng = np.random.default_rng(SEED)
    rec, boots = [], []
    for side in ("elec", "nuc"):
        for a, b in itertools.combinations(reactions, 2):
            xa, xb = full[a][side], full[b][side]
            shared = xa.index.intersection(xb.index)
            if len(shared) < MIN_SHARED:
                continue
            va, vb = xa[shared].values, xb[shared].values
            ceiling = float(np.sqrt(max(rmap[(a, side)], 0) * max(rmap[(b, side)], 0)))
            obs = RK.spearman(va, vb)
            ret, picked, best, mean_all = (retention(va, vb) if len(shared) >= MIN_FOR_SELECTION
                                           else (np.nan,) * 4)
            same = frozenset((a, b)) in E.SAME_REACTION
            rec.append({"side": side, "rxn_a": a, "rxn_b": b, "n_shared": len(shared),
                        "same_reaction": same, "rho_obs": obs, "ceiling": ceiling,
                        "rho_disattenuated": obs / ceiling if ceiling > 0 else np.nan,
                        "retention": ret, "B_mean_A_selected": picked, "B_mean_B_best": best,
                        "B_mean_all": mean_all})
            idx = rng.integers(0, len(shared), (N_BOOT, len(shared)))
            rho_b = np.array([RK.spearman(va[i], vb[i]) if len(set(i)) >= MIN_SHARED else np.nan
                              for i in idx])
            ret_b = (np.array([retention(va[i], vb[i])[0] for i in idx])
                     if len(shared) >= MIN_FOR_SELECTION else np.full(N_BOOT, np.nan))
            boots.append({"same": same, "n": len(shared), "ceiling": ceiling,
                          "rho": rho_b, "ret": ret_b})
    T = pd.DataFrame(rec)

    def pooled(observed, values, weights, label, extra=""):
        """The estimate is the shared-count-weighted mean over the comparisons themselves, which
        is what a reader recomputes from the released files; the resamples give only the
        interval. An earlier version of this script reported the mean of the resamples as the
        estimate; deviation 3."""
        w = np.where(np.isfinite(values), weights, 0.0)
        if w.sum(axis=0).min() == 0:
            return
        p = np.nansum(np.where(np.isfinite(values), values, 0.0) * w, axis=0) / w.sum(axis=0)
        keep = np.isfinite(observed)
        point = float(np.average(observed[keep], weights=weights[keep, 0]))
        lo, hi = np.percentile(p, [2.5, 97.5])
        print(f"   {label}: {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]{extra}")
        return point, lo, hi

    cross = [b for b in boots if not b["same"]]
    n = np.array([b["n"] for b in cross], float)[:, None]
    ceil = np.array([b["ceiling"] for b in cross])[:, None]
    rho = np.vstack([b["rho"] for b in cross])
    ret = np.vstack([b["ret"] for b in cross])
    print(f"\n== {len(cross)} comparisons between different reactions")
    obs = T.loc[~T.same_reaction, "rho_obs"].to_numpy()
    cor = T.loc[~T.same_reaction, "rho_disattenuated"].to_numpy()
    kept_ret = T.loc[~T.same_reaction, "retention"].to_numpy()
    summary = {}
    summary["rho_obs"] = pooled(obs, rho, n, "pooled rank correlation")
    summary["rho_disattenuated"] = pooled(cor, rho / ceil, n,
                                          "pooled, corrected for measurement error")
    keep = np.isfinite(ret).any(axis=1)
    summary["retention"] = pooled(kept_ret[keep], ret[keep], n[keep],
                                  "pooled retention of a top-ten selection")

    print(f"\n== the same-reaction entry, reported apart and not pooled")
    print(T[T.same_reaction].round(3).to_string(index=False))

    sel = T[T.retention.notna()]
    print(f"\n== carrying a top-{TOP_K} selection, {len(sel)} comparisons with >= {MIN_FOR_SELECTION} shared molecules")
    show = sel.sort_values("retention").copy()
    for c in ("B_mean_A_selected", "B_mean_B_best", "B_mean_all"):
        show[c] = (show[c] * 100).round(1)
    print(show[["side", "rxn_a", "rxn_b", "n_shared", "rho_obs", "retention",
                "B_mean_all", "B_mean_A_selected", "B_mean_B_best"]].to_string(index=False))
    print(f"   retention below zero (worse than picking at random) in "
          f"{int((sel.retention < 0).sum())} of {len(sel)}")

    print(f"\n== every comparison")
    print(T.sort_values(["side", "rho_disattenuated"], ascending=[True, False]).round(3).to_string(index=False))
    REL.to_csv(L.OUT / "extended_reliability_bba.csv", index=False)
    T.to_csv(L.OUT / "extended_transfer_bba.csv", index=False)
    pd.DataFrame([{"quantity": k, "estimate": v[0], "ci_low": v[1], "ci_high": v[2]}
                  for k, v in summary.items() if v]).to_csv(L.OUT / "extended_pooled_bba.csv", index=False)
    print("\nwrote extended_reliability/transfer/pooled_bba.csv")
