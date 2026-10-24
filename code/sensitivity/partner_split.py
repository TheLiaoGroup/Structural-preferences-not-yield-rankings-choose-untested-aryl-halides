"""Partner-split sensitivity reported in Supplementary Note 8: partner-half agreement of every set under the
20 further partitions.

The partner-half agreement (rho_P) is recomputed for all 15 reactions x 2 sides with the estimator of the main
analysis (partner_profiles.half_fits -> evaluable -> gated_bba.spearman, >= 10 molecules) under the canonical,
unsalted split and under the 20 deterministic salted splits (partner-salt-00 ... partner-salt-19). Nothing in
the estimator, the salts or the thresholds is changed. Degree 2 is the primary analysis; degrees 3 and 5 are
sensitivities.

Run from the repository root after the main steps: python -B code/sensitivity/partner_split.py
"""
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

REL = PKG = Path(__file__).resolve().parents[2]   # the repository root (code, data, output)
OUT = PKG / "output" / "sensitivity" / "partner_split"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG / "code" / "partner_profiles"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import partner_profiles as A  # noqa: E402

A.load(OUT / "scratch")
SPLITS = [None] + A.SALTS
DEGREES = (2, 3, 5)
assert len(A.SALTS) == 20 and A.SALTS[0] == "partner-salt-00" and A.SALTS[-1] == "partner-salt-19"

wells, conds, pairs = A.PC.released()
D = A.PC.design_inputs(wells, conds, pairs, "primary")
lay, ys = A.PC.layouts_from(*D, A.G)[:2]
assert len(lay) == 15

ref = pd.read_csv(REL / "output/core/gated_A_cells_bba.csv")
fig2a = pd.read_csv(REL / "output/presentation/source_data/Figure2a_portability_by_set.csv")
role = fig2a.set_index(["reaction", "side"]).role.to_dict()
assert len(ref) == 90

# One row per (reaction, side, split, degree).
long = []
for rxn in sorted(lay):
    for side in ("elec", "nuc"):
        assert A.split_matches_release(lay[rxn], side), (rxn, side)
        for salt in SPLITS:
            halves = A.half_fits(lay[rxn], ys[rxn], side, salt)
            fit_ok = halves[0] is not None and halves[1] is not None
            for k in DEGREES:
                ev = A.evaluable(lay[rxn], halves, side, k)
                rho = (A.G.spearman(halves[0][side][ev], halves[1][side][ev])
                       if len(ev) >= A.MIN_MOL else np.nan)
                long.append({"reaction": rxn, "side": side, "degree": k, "split": salt or "canonical",
                             "fit_ok": fit_ok, "evaluable": len(ev), "rho_P": rho})
L = pd.DataFrame(long)

# The canonical split must reproduce the main analysis cell for cell (molecules and rho_P, all 90 cells).
can = L[L.split == "canonical"].merge(ref, on=["reaction", "side", "degree"], validate="one_to_one")
assert len(can) == 90
assert (can.evaluable == can.molecules).all()
same = np.where(np.isnan(can.rho_P_x), np.isnan(can.rho_P_y), np.abs(can.rho_P_x - can.rho_P_y) < 1e-12)
assert same.all(), can.loc[~same, ["reaction", "side", "degree", "rho_P_x", "rho_P_y"]]

# Wide table: one row per set and degree; estimable sets are those the main analysis estimates.
rows = []
for (rxn, side, k), g in L.groupby(["reaction", "side", "degree"], sort=True):
    r = ref[(ref.reaction == rxn) & (ref.side == side) & (ref.degree == k)].iloc[0]
    if np.isnan(r.rho_P):
        continue
    c = g[g.split == "canonical"].iloc[0]
    alt = g[g.split != "canonical"].set_index("split").reindex(A.SALTS)
    v = alt.rho_P.to_numpy(float)
    ok = np.isfinite(v)
    vv = v[ok]
    rho_c = float(r.rho_C)   # the condition-half agreement of the main analysis on the same degree-k molecules
    all21 = np.r_[c.rho_P, vv]
    row = {"reaction": rxn, "side": side, "molecule_class": role.get((rxn, side), ""), "degree": k,
           "condition_half_rho_C": rho_c, "canonical_rho_P": c.rho_P, "canonical_evaluable": c.evaluable,
           "alt_estimable": int(ok.sum()), "alt_median": np.median(vv) if ok.any() else np.nan,
           "alt_q25": np.percentile(vv, 25) if ok.any() else np.nan, "alt_q75": np.percentile(vv, 75) if ok.any() else np.nan,
           "alt_min": vv.min() if ok.any() else np.nan, "alt_max": vv.max() if ok.any() else np.nan}
    row["canonical_minus_median"] = row["canonical_rho_P"] - row["alt_median"]
    # Rank of the canonical value among all 21 splits (1 = lowest; ties take the average rank).
    row["canonical_rank_of_21"] = float(pd.Series(all21).rank(method="average").iloc[0]) if ok.all() else np.nan
    row["alt_prop_above_0"] = float((vv > 0).mean()) if ok.any() else np.nan
    row["alt_prop_above_rho_C"] = float((vv > rho_c).mean()) if ok.any() else np.nan
    row["alt_evaluable_min"] = int(alt.evaluable.min())
    row["alt_evaluable_max"] = int(alt.evaluable.max())
    for s in A.SALTS:
        row[f"rho_{s[-2:]}"] = alt.rho_P[s]
    for s in A.SALTS:
        row[f"n_{s[-2:]}"] = int(alt.evaluable[s])
    rows.append(row)
W = pd.DataFrame(rows)
W2 = W[W.degree == 2]
assert len(W2) > 0, len(W2)

# Summary across sets.
summ = []
for k, g in W.groupby("degree"):
    d = g.canonical_minus_median.to_numpy()
    full = g[g.alt_estimable == 20]
    # The headline recomputed under each alternative split: median over sets of that split's rho_P.
    per_split = []
    for s in A.SALTS:
        x = g[f"rho_{s[-2:]}"].to_numpy(float)
        per_split.append((np.nanmedian(x), int(np.isfinite(x).sum())))
    hv = np.array([p for p, _ in per_split])
    summ.append({
        "degree": k, "sets": len(g),
        "headline_canonical_median_rho_P": float(np.median(g.canonical_rho_P)),
        "headline_canonical_q25": float(np.percentile(g.canonical_rho_P, 25)),
        "headline_canonical_q75": float(np.percentile(g.canonical_rho_P, 75)),
        "median_of_alt_medians": float(np.median(g.alt_median)),
        "q25_of_alt_medians": float(np.percentile(g.alt_median, 25)),
        "q75_of_alt_medians": float(np.percentile(g.alt_median, 75)),
        "median_of_all_alt_values": float(np.nanmedian(g[[f"rho_{s[-2:]}" for s in A.SALTS]].to_numpy(float))),
        "headline_per_alt_split_median": float(np.median(hv)), "headline_per_alt_split_min": float(hv.min()),
        "headline_per_alt_split_max": float(hv.max()),
        "sets_per_alt_split_min": min(n for _, n in per_split), "sets_per_alt_split_max": max(n for _, n in per_split),
        "canonical_minus_median_median": float(np.median(d)), "canonical_minus_median_mean": float(np.mean(d)),
        "canonical_minus_median_q25": float(np.percentile(d, 25)), "canonical_minus_median_q75": float(np.percentile(d, 75)),
        "canonical_minus_median_min": float(d.min()), "canonical_minus_median_max": float(d.max()),
        "sets_canonical_above_alt_median": int((d > 0).sum()), "sets_canonical_below_alt_median": int((d < 0).sum()),
        "sign_test_two_sided_p": float(__import__("scipy.stats", fromlist=["binomtest"]).binomtest(
            int((d > 0).sum()), int((d != 0).sum()), 0.5).pvalue),
        "mean_canonical_rank_of_21": float(full.canonical_rank_of_21.mean()), "sets_with_20_alternatives": len(full),
        "sets_canonical_below_all_20": int((full.canonical_rho_P < full.alt_min).sum()),
        "sets_canonical_above_all_20": int((full.canonical_rho_P > full.alt_max).sum()),
        "sets_abs_canonical_minus_median_gt_0.2": int((np.abs(d) > 0.2).sum()),
        "sets_alt_median_above_0": int((g.alt_median > 0).sum()),
        "sets_all_alt_above_0": int((g.alt_prop_above_0 == 1).sum()),
        "sets_any_alt_above_rho_C": int((g.alt_prop_above_rho_C > 0).sum()),
        "median_alt_IQR_width": float(np.median(g.alt_q75 - g.alt_q25)),
        "median_alt_range_width": float(np.median(g.alt_max - g.alt_min)),
    })
S = pd.DataFrame(summ)

W.to_csv(OUT / "partner_split_all_sets.csv", index=False, lineterminator="\n", float_format="%.6f")
S.to_csv(OUT / "partner_split_summary.csv", index=False, lineterminator="\n", float_format="%.6f")
L.to_csv(OUT / "partner_split_long.csv", index=False, lineterminator="\n", float_format="%.6f")
pd.set_option("display.width", 250)
print(W2[["reaction", "side", "canonical_evaluable", "condition_half_rho_C", "canonical_rho_P", "alt_estimable", "alt_median",
          "alt_q25", "alt_q75", "alt_min", "alt_max", "canonical_minus_median", "canonical_rank_of_21",
          "alt_prop_above_0", "alt_evaluable_min", "alt_evaluable_max"]].round(3).to_string(index=False))
print(S.T.to_string())
