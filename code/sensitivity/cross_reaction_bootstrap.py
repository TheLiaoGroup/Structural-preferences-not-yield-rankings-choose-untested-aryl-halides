"""A molecule-cluster bootstrap for the pooled cross-reaction agreement (Supplementary Note 10).

Canonical estimators (unchanged): core_intervals/extended_transfer_bba.reproduce (first condition set: pooled observed and
reliability-corrected agreement over the 58 cross-reaction comparisons) and core_intervals/holdout_bba.score (second
condition set: C2, corrected). Point estimates are the shared-count-weighted means; canonical intervals resample
molecules independently within each comparison (2,000 resamples, seeds 20260927 and 20260929).

Step 1 rebuilds both canonical intervals exactly from the molecule effects of the main analysis, consuming the
random streams in the same order, so that the inputs are those of the main analysis.

Step 2, the sensitivity. Algorithm (fixed before running):
  * cluster key = the molecule identifier (the RDKit canonical tautomer used throughout);
    one identifier is one cluster on both sides and in every reaction;
  * universe U = the union of the shared-molecule sets of the comparisons being pooled;
  * one replicate draws |U| molecules from U with replacement, giving every molecule a count c_m;
  * every comparison uses only its own shared molecules, each repeated c_m times; a comparison with fewer than
    five distinct molecules drawn is missing in that replicate (the rule of the main analysis);
  * pooling weights stay the shared counts; the reliability ceilings stay fixed;
  * 2,000 replicates; percentile interval; a second seed is run as a Monte Carlo check.
Also reported, without recomputation: the reaction-level intervals of the main analysis for the pooled observed
agreement (delete-one-reaction jackknife and reaction-node bootstrap, node_bootstrap_pooled.csv).

Step 3 verifies the shortfall inputs only (20 of 31, the reason, D_A over 31, degrees 3 and 5); its
leave-one-reaction-out interval is not replaced.

Run from the repository root after the main steps: python -B code/sensitivity/cross_reaction_bootstrap.py
"""
import itertools
import json
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

REL = PKG = Path(__file__).resolve().parents[2]   # the repository root (code, data, output)
OUT = PKG / "output" / "sensitivity" / "cross_reaction_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG / "code" / "core_intervals"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import public_common as PC  # noqa: E402

L, E, G = PC.load_modules(output=OUT / "scratch")
import extended_transfer_bba as X  # noqa: E402
import rank_bba as RK  # noqa: E402

CR = REL / "output"
assert (X.SEED, X.N_BOOT, X.MIN_SHARED) == (20260927, 2000, 5)
H_SEED = 20260929
CLUSTER_SEEDS = {('first', 'A'): 16998050213382060219, ('first', 'B'): 9541573871950130687, ('second', 'A'): 9943103931723310213, ('second', 'B'): 16070036166452195286}   # fixed seeds of the two runs of the cluster bootstrap


def effects(path, label):
    f = pd.read_csv(path)
    f = f[(f.scale == "raw") & (f.label == label)]
    return {rc: {s: g[g.side == s].set_index("mol").main_effect.sort_index(kind="stable") for s in ("elec", "nuc")}
            for rc, g in f.groupby("rxn_class")}


full = effects(CR / "core/full_data_effects_bba.csv", "primary")
held = effects(CR / "core/holdout_molecule_effects_bba.csv", "validation")
rel1 = pd.read_csv(CR / "core_intervals/extended_reliability_bba.csv").set_index(["rxn_class", "side"]).reliability
relh = pd.read_csv(CR / "core/holdout_reliability_bba.csv").set_index(["rxn_class", "side"]).reliability
reactions1 = sorted(full)
reactions_h = sorted(pd.read_csv(CR / "core_intervals/holdout_conditions_bba.csv").rxn_class.unique())
assert len(reactions1) == 15


def comparisons(eff, rel, reactions, consume_same):
    """The comparison list of the main analysis in its own order; same-reaction entries kept only for stream consumption."""
    out = []
    for side in ("elec", "nuc"):
        for a, b in itertools.combinations(reactions, 2):
            xa, xb = eff[a][side], eff[b][side]
            shared = xa.index.intersection(xb.index).sort_values()
            if len(shared) < X.MIN_SHARED:
                continue
            same = frozenset((a, b)) in E.SAME_REACTION
            if same and not consume_same:
                continue
            ceiling = float(np.sqrt(max(rel[(a, side)], 0) * max(rel[(b, side)], 0)))
            out.append({"side": side, "rxn_a": a, "rxn_b": b, "same": same, "shared": list(shared),
                        "va": xa[shared].to_numpy(float), "vb": xb[shared].to_numpy(float), "ceiling": ceiling,
                        "rho_obs": RK.spearman(xa[shared].to_numpy(float), xb[shared].to_numpy(float))})
    return out


def pooled_point(cells, corrected):
    n = np.array([len(c["va"]) for c in cells], float)
    v = np.array([c["rho_obs"] / c["ceiling"] if corrected else c["rho_obs"] for c in cells])
    return float(np.average(v[np.isfinite(v)], weights=n[np.isfinite(v)]))


def pooled_draws(draws, cells, corrected):
    n = np.array([len(c["va"]) for c in cells], float)[:, None]
    ceil = np.array([c["ceiling"] for c in cells])[:, None]
    v = draws / ceil if corrected else draws
    ok = np.isfinite(v)
    w = np.where(ok, n, 0.0)
    return (np.where(ok, v, 0.0) * w).sum(0) / w.sum(0)


def within_draws(cells, rng):
    """The resampling of the main analysis: molecules within each comparison, in list order."""
    D = np.full((len(cells), X.N_BOOT), np.nan)
    for k, c in enumerate(cells):
        idx = rng.integers(0, len(c["va"]), (X.N_BOOT, len(c["va"])))
        D[k] = [RK.spearman(c["va"][i], c["vb"][i]) if len(set(i)) >= X.MIN_SHARED else np.nan for i in idx]
    return D


def cluster_draws(cells, seed, n_boot=X.N_BOOT):
    """Global molecule-cluster bootstrap: one count per molecule per replicate, shared by every comparison."""
    U = sorted(set(m for c in cells for m in c["shared"]))
    pos = {m: i for i, m in enumerate(U)}
    rng = np.random.default_rng(seed)
    D = np.full((len(cells), n_boot), np.nan)
    members = [np.array([pos[m] for m in c["shared"]]) for c in cells]
    for r in range(n_boot):
        cnt = np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U))
        for k, c in enumerate(cells):
            w = cnt[members[k]]
            if (w > 0).sum() < X.MIN_SHARED:
                continue
            rep = np.repeat(np.arange(len(w)), w)
            D[k, r] = RK.spearman(c["va"][rep], c["vb"][rep])
    return D, U


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


# ---- Step 1: rebuild the canonical intervals ----
all1 = comparisons(full, rel1, reactions1, consume_same=True)
D1 = within_draws(all1, np.random.default_rng(X.SEED))
cross1 = [c for c in all1 if not c["same"]]
keep = np.array([not c["same"] for c in all1])
assert len(cross1) > 0
ref1 = pd.read_csv(CR / "core_intervals/extended_pooled_bba.csv").set_index("quantity")
res = []
for corrected, q in ((False, "rho_obs"), (True, "rho_disattenuated")):
    point = pooled_point(cross1, corrected)
    lo, hi = ci(pooled_draws(D1[keep], cross1, corrected))
    assert abs(point - ref1.loc[q, "estimate"]) < 1e-12 and abs(lo - ref1.loc[q, "ci_low"]) < 1e-12 and abs(hi - ref1.loc[q, "ci_high"]) < 1e-12, (q, point, lo, hi)
    res.append({"condition_set": "first", "quantity": "pooled observed" if not corrected else "pooled reliability-corrected",
                "comparisons": 58, "estimate": point, "canonical_ci_low": lo, "canonical_ci_high": hi})

# Second set: C1 cells consume the stream first, then the C2 cells (cross-reaction only), as holdout_bba.score does.
rng_h = np.random.default_rng(H_SEED)
cells_c1 = []
for rc in reactions_h:
    for side in ("elec", "nuc"):
        xa, xh = full[rc][side], held[rc][side]
        shared = xa.index.intersection(xh.index).sort_values()
        if len(shared) >= X.MIN_SHARED:
            cells_c1.append({"va": xa[shared].to_numpy(float), "vb": xh[shared].to_numpy(float), "ceiling": 1.0,
                             "rho_obs": RK.spearman(xa[shared].to_numpy(float), xh[shared].to_numpy(float)), "shared": list(shared)})
Dc1 = within_draws(cells_c1, rng_h)
cross_h = comparisons(held, relh, reactions_h, consume_same=False)
Dh = within_draws(cross_h, rng_h)
refh = pd.read_csv(CR / "core_intervals/holdout_summary_bba.csv").set_index("quantity")
c1 = (pooled_point(cells_c1, False), *ci(pooled_draws(Dc1, cells_c1, False)))
c2 = (pooled_point(cross_h, True), *ci(pooled_draws(Dh, cross_h, True)))
r1, r2 = refh.loc["C1 within-reaction, analysed vs held-out"], refh.loc["C2 cross-reaction, held-out, corrected"]
assert np.allclose(c1, (r1.estimate, r1.ci_low, r1.ci_high), atol=1e-12, rtol=0), c1
assert np.allclose(c2, (r2.estimate, r2.ci_low, r2.ci_high), atol=1e-12, rtol=0), c2
res.append({"condition_set": "second", "quantity": "pooled reliability-corrected (C2)", "comparisons": len(cross_h),
            "estimate": c2[0], "canonical_ci_low": c2[1], "canonical_ci_high": c2[2]})
obs_h = (pooled_point(cross_h, False), *ci(pooled_draws(Dh, cross_h, False)))
res.append({"condition_set": "second", "quantity": "pooled observed (not a released quantity)", "comparisons": len(cross_h),
            "estimate": obs_h[0], "canonical_ci_low": obs_h[1], "canonical_ci_high": obs_h[2]})

# ---- Step 2: global molecule-cluster bootstrap ----
universe = {}
mc = {}
for tag, cells in (("first", cross1), ("second", cross_h)):
    for s in ("A", "B"):
        mc[(tag, s)] = cluster_draws(cells, CLUSTER_SEEDS[(tag, s)])
    universe[tag] = len(mc[(tag, "A")][1])
for r in res:
    tag = r["condition_set"]
    corrected = "corrected" in r["quantity"]
    cells = cross1 if tag == "first" else cross_h
    for s in ("A", "B"):
        lo, hi = ci(pooled_draws(mc[(tag, s)][0], cells, corrected))
        r[f"cluster_ci_low_seed{s}"], r[f"cluster_ci_high_seed{s}"] = lo, hi
        r[f"cluster_missing_share_seed{s}"] = float(np.isnan(mc[(tag, s)][0]).mean())
    r["canonical_width"] = r["canonical_ci_high"] - r["canonical_ci_low"]
    r["cluster_width"] = r["cluster_ci_high_seedA"] - r["cluster_ci_low_seedA"]
    r["width_change"] = r["cluster_width"] - r["canonical_width"]
    r["width_ratio"] = r["cluster_width"] / r["canonical_width"]
    r["cluster_universe_molecules"] = universe[tag]
# Reaction-level intervals of the main analysis for the first-set pooled observed agreement (not recomputed).
node = pd.read_csv(CR / "core_intervals/node_bootstrap_pooled.csv").set_index("degree").loc[2]
for r in res:
    if r["condition_set"] == "first" and r["quantity"] == "pooled observed":
        assert abs(node.weighted - r["estimate"]) < 1e-12
        r.update({"existing_reaction_jackknife_low": node["weighted low"], "existing_reaction_jackknife_high": node["weighted high"],
                  "existing_reaction_node_bootstrap_low": node["bootstrap low"], "existing_reaction_node_bootstrap_high": node["bootstrap high"]})
R = pd.DataFrame(res)

# Multiplicity of molecules across comparisons (why the intervals could differ).
mult = {}
for tag, cells in (("first", cross1), ("second", cross_h)):
    cnt = pd.Series([m for c in cells for m in c["shared"]]).value_counts()
    mult[tag] = {"appearances": int(cnt.sum()), "distinct_molecules": int(len(cnt)), "in_one_comparison": int((cnt == 1).sum()),
                 "in_two_or_more": int((cnt >= 2).sum()), "max_comparisons_per_molecule": int(cnt.max()),
                 "share_of_appearances_from_multi": float(cnt[cnt >= 2].sum() / cnt.sum())}

# ---- Step 3: shortfall verification (no new interval) ----
sf = {}
for k in (2, 3, 5):
    Rk = pd.read_csv(CR / f"core/gated_B_relative_k{k}_bba.csv")
    pr = G.pooled_relative(Rk)
    pos = (Rk.rho_P_a > 0) & (Rk.rho_P_b > 0)
    sf[k] = {"eligible": len(Rk), "used_for_D_G": int(pos.sum()), "excluded": int((~pos).sum()),
             "excluded_have_nonpositive_benchmark": bool(((Rk.rho_P_a[~pos] <= 0) | (Rk.rho_P_b[~pos] <= 0)).all()),
             "D_G": pr["D_G"], "D_G_low": pr["D_G low"], "D_G_high": pr["D_G high"], "D_A_all": pr["D_A"],
             "D_A_low": pr["D_A low"], "D_A_high": pr["D_A high"], "kappa": pr["kappa"], "kappa_low": pr["kappa low"],
             "kappa_high": pr["kappa high"], "molecule_appearances": int(Rk.molecules.sum()),
             "reactions": int(len(set(Rk.rxn_a) | set(Rk.rxn_b)))}
lo = pd.read_csv(CR / "core/shortfall_leave_one_reaction_out.csv")
sf[2]["loro_min"], sf[2]["loro_max"] = float(lo[lo.degree == 2].D_G.min()), float(lo[lo.degree == 2].D_G.max())
# Within the 20, are D_G components driven by the excluded 11? Report the arithmetic contrast on the 20 as well.
R2 = pd.read_csv(CR / "core/gated_B_relative_k2_bba.csv")
pos = (R2.rho_P_a > 0) & (R2.rho_P_b > 0)
sf[2]["D_A_on_the_20"] = G.relative_values(R2[pos])["D_A"]
sf[2]["D_A_on_the_11"] = G.relative_values(R2[~pos])["D_A"]

R.to_csv(OUT / "global_cluster_bootstrap.csv", index=False, lineterminator="\n", float_format="%.6f")
(OUT / "cross_reaction_summary.json").write_text(json.dumps({"multiplicity": mult, "shortfall": sf}, indent=1, default=float), encoding="utf-8")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(R.round(4).T.to_string())
print(json.dumps({"multiplicity": mult, "shortfall": sf}, indent=1, default=float))
