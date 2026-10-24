"""Robustness of the agreement between the structural profiles of the two Buchwald-Hartwig campaigns
(Supplementary Note 17).

Inputs are the contributions of the main analysis: the first campaign's aryl halides on the first condition set
(output/core/molecule_contributions.csv) and the second campaign's (output/extended/campaign_contributions.csv).
Features, class rule (>= 3 molecules with and >= 3 without a feature) and the profile comparison (features
testable in both, >= 5) are those of code/extended/structure_profiles.py.

  check        shared-molecule ranking, condition-half reliability, full-library profile agreement,
               and leave-one- and leave-two-feature-out agreement.
  resampling   molecule bootstrap within each campaign (contributions held fixed); a feature is estimable in a
               replicate only if >= 3 distinct molecules carry it and >= 3 distinct molecules lack it; a
               replicate is usable only if >= 5 features are estimable in both campaigns.
  permutation  exact feature-label permutation: all 7! = 5,040 orderings of the second campaign's class
               differences.
  shared       both profiles recomputed on the aryl halides the campaigns share, using the same rules.
  ranking      the profile similarity of the second campaign to each of the nine analysed reactions, reported as
               a ranking only (no test of 0.96 against 0.80).

Run from the repository root after the main steps: python -B code/sensitivity/bh_campaign.py
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
OUT = PKG / "output" / "sensitivity" / "bh_campaign"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG / "code" / "extended"))
sys.path.insert(0, str(PKG / "code" / "core"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr, rankdata  # noqa: E402
import structure_profiles as SP  # noqa: E402
import rank_bba as RK  # noqa: E402

CR = REL / "output"
FEATS = SP.FEATURES + SP.ALSO
assert len(FEATS) == 11 and (SP.MIN_CLASS, SP.MIN_FEATURES) == (3, 5)
N_BOOT = 10000
SEED = 8802605442682611966   # fixed seed of the molecule resampling

mc = pd.read_csv(CR / "core/molecule_contributions.csv")
cc = pd.read_csv(CR / "extended/campaign_contributions.csv")
first = mc[(mc.reaction == "Buchwald-Hartwig") & (mc.side == "elec")][["SMILES", "contribution_pp"]]
second = cc[(cc.reaction == "Buchwald-Hartwig(CMC)") & (cc.side == "elec")][["SMILES", "contribution_pp"]]


def with_features(frame):
    rows = []
    for s, v in zip(frame.SMILES, frame.contribution_pp):
        f = SP.features(s)
        if f is not None:
            rows.append([s, v] + [int(f[k]) for k in FEATS])
    return pd.DataFrame(rows, columns=["SMILES", "contribution_pp"] + FEATS)


F1, F2 = with_features(first), with_features(second)
ref_f = pd.read_csv(CR / "extended/aryl_halide_features.csv").set_index("SMILES")
for F in (F1, F2):
    assert all((ref_f.loc[s, FEATS].to_numpy(int) == F.loc[F.SMILES == s, FEATS].to_numpy(int)[0]).all() for s in F.SMILES)


def profile(F, weights=None):
    """Class differences (pp) per feature, None where the class rule fails; weights are bootstrap counts."""
    x = F.contribution_pp.to_numpy(float)
    w = np.ones(len(F)) if weights is None else np.asarray(weights, float)
    out = {}
    for k in FEATS:
        m = F[k].to_numpy(int) == 1
        n1, n0 = int(((w > 0) & m).sum()), int(((w > 0) & ~m).sum())   # distinct molecules in each class
        if n1 < SP.MIN_CLASS or n0 < SP.MIN_CLASS:
            continue
        out[k] = np.sum(w[m] * x[m]) / np.sum(w[m]) - np.sum(w[~m] * x[~m]) / np.sum(w[~m])
    return pd.Series(out, dtype=float)


def compare(p, q):
    z = pd.concat([p, q], axis=1).dropna()
    if len(z) < SP.MIN_FEATURES:
        return None
    return len(z), float(pearsonr(z.iloc[:, 0], z.iloc[:, 1])[0]), float(spearmanr(z.iloc[:, 0], z.iloc[:, 1])[0]), list(z.index)


# ---- check against the values of the main analysis ----
P1, P2 = profile(F1), profile(F2)
n, r, rs, shared_feats = compare(P1, P2)
readings = json.loads((CR / "extended/structure_readings.json").read_text(encoding="utf-8"))["profiles"]["campaigns"]
assert n == readings["features"] == 7 and abs(r - readings["pearson"]) < 1e-12 and abs(rs - readings["spearman"]) < 1e-12
z = pd.concat([P1, P2], axis=1).dropna()
z.columns = ["first_campaign_pp", "second_campaign_pp"]
loo = {k: float(pearsonr(z.drop(k).iloc[:, 0], z.drop(k).iloc[:, 1])[0]) for k in z.index}
assert abs(min(loo.values()) - readings["leave_one_out_min"]) < 1e-12 and abs(max(loo.values()) - readings["leave_one_out_max"]) < 1e-12
l2o = {(a, b): float(pearsonr(z.drop([a, b]).iloc[:, 0], z.drop([a, b]).iloc[:, 1])[0]) for a, b in itertools.combinations(z.index, 2)}
named = ("five-membered ring at the halogen", "substituent beside the halogen")
slope = float(np.polyfit(z.first_campaign_pp, z.second_campaign_pp, 1)[0])
shared = sorted(set(F1.SMILES) & set(F2.SMILES))
x1 = F1.set_index("SMILES").loc[shared, "contribution_pp"].to_numpy()
x2 = F2.set_index("SMILES").loc[shared, "contribution_pp"].to_numpy()
rho_shared = RK.spearman(x1, x2)
camp = pd.read_csv(CR / "extended/campaign_comparisons.csv")
cref = camp[(camp.side == "elec") & (camp.analysed_reaction == "Buchwald-Hartwig")].iloc[0]
assert len(shared) == cref.shared and abs(rho_shared - cref.rho) < 1e-12
design = pd.read_csv(CR / "extended/campaign_design.csv").iloc[0]
class_sizes = {k: {"first_with": int(F1[k].sum()), "first_without": int((F1[k] == 0).sum()),
                   "second_with": int(F2[k].sum()), "second_without": int((F2[k] == 0).sum())} for k in FEATS}
verified = {"first_campaign_aryl_halides": len(F1), "second_campaign_aryl_halides": len(F2),
            "first_campaign_electrophiles_in_file": len(first), "second_campaign_electrophiles_in_file": len(second),
            "features_compared": n, "feature_list": shared_feats, "pearson": r, "spearman": rs,
            "leave_one_feature_out": loo, "leave_one_out_min": min(loo.values()), "leave_one_out_max": max(loo.values()),
            "leave_two_out_min": min(l2o.values()), "leave_two_out_max": max(l2o.values()),
            "leave_two_out_median": float(np.median(list(l2o.values()))),
            "leave_out_ring_and_ortho_pair": l2o.get(named, l2o.get(named[::-1])),
            "slope_second_on_first": slope, "shared_aryl_halides": len(shared), "shared_rank_agreement": rho_shared,
            "shared_rank_agreement_corrected_released": float(cref.rho_corrected),
            "second_campaign_reliability_elec": float(design.reliability_elec),
            "second_campaign_reliability_nuc": float(design.reliability_nuc),
            "class_sizes": class_sizes, "profile_values": z.round(6).to_dict(orient="index")}

# ---- molecule bootstrap within each campaign ----
rng = np.random.default_rng(SEED)
boot = []
for b in range(N_BOOT):
    w1 = np.bincount(rng.integers(0, len(F1), len(F1)), minlength=len(F1))
    w2 = np.bincount(rng.integers(0, len(F2), len(F2)), minlength=len(F2))
    c = compare(profile(F1, w1), profile(F2, w2))
    boot.append({"replicate": b, "usable": c is not None, "features": c[0] if c else np.nan,
                 "pearson": c[1] if c else np.nan, "spearman": c[2] if c else np.nan,
                 "feature_set": "; ".join(c[3]) if c else ""})
B = pd.DataFrame(boot)
U = B[B.usable]
boot_summary = {"replicates": N_BOOT, "usable": int(len(U)), "unusable_fewer_than_5_features": int((~B.usable).sum()),
                "features_in_usable": U.features.value_counts().sort_index().to_dict()}
for k in ("pearson", "spearman"):
    v = U[k].to_numpy()
    boot_summary[k] = {"median": float(np.median(v)), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
                       "share_above_0": float((v > 0).mean()), "share_above_0.8": float((v > 0.8).mean())}
# The same summaries on the replicates that kept all seven canonical features, so that the feature set is fixed.
U7 = U[U.features == 7]
boot_summary["all_seven_features"] = {"replicates": int(len(U7)), **{k: {"median": float(np.median(U7[k])), "p2.5": float(np.percentile(U7[k], 2.5)),
                                      "p97.5": float(np.percentile(U7[k], 97.5)), "share_above_0": float((U7[k] > 0).mean()),
                                      "share_above_0.8": float((U7[k] > 0.8).mean())} for k in ("pearson", "spearman")}}

# ---- exact feature-label permutation ----
a, b2 = z.first_campaign_pp.to_numpy(), z.second_campaign_pp.to_numpy()
perm = np.array(list(itertools.permutations(range(len(b2)))))
assert len(perm) == 5040
bp = b2[perm]
ac = a - a.mean()
bc = bp - bp.mean(1, keepdims=True)
pear = (bc @ ac) / np.sqrt((ac @ ac) * (bc * bc).sum(1))
ra = rankdata(a)
rb = rankdata(bp, axis=1)
spear = ((rb - rb.mean(1, keepdims=True)) @ (ra - ra.mean())) / np.sqrt(((ra - ra.mean()) ** 2).sum() * ((rb - rb.mean(1, keepdims=True)) ** 2).sum(1))
ident = np.flatnonzero((perm == np.arange(7)).all(1))[0]
assert abs(pear[ident] - r) < 1e-12 and abs(spear[ident] - rs) < 1e-12
P = pd.DataFrame({"permutation": ["-".join(map(str, p)) for p in perm], "is_observed": np.arange(5040) == ident,
                  "pearson": pear, "spearman": spear})
perm_summary = {"permutations": 5040, "feature_order": list(z.index),
                "pearson_observed": r, "pearson_at_or_above_observed": int((pear >= r - 1e-12).sum()),
                "pearson_p_exact": float((pear >= r - 1e-12).mean()),
                "pearson_rank_of_observed": int((pear > r + 1e-12).sum() + 1),
                "spearman_observed": rs, "spearman_at_or_above_observed": int((spear >= rs - 1e-12).sum()),
                "spearman_p_exact": float((spear >= rs - 1e-12).mean()),
                "spearman_rank_of_observed_best_tie": int((spear > rs + 1e-12).sum() + 1),
                "pearson_null_median": float(np.median(pear)), "pearson_null_97.5": float(np.percentile(pear, 97.5)),
                "spearman_null_median": float(np.median(spear)), "spearman_null_97.5": float(np.percentile(spear, 97.5))}

# ---- the shared aryl-halide library ----
S1, S2 = F1[F1.SMILES.isin(shared)].reset_index(drop=True), F2[F2.SMILES.isin(shared)].reset_index(drop=True)
assert len(S1) == len(S2) == len(shared)
shared_comparison = compare(profile(S1), profile(S2))
shared_rows = []
for k in FEATS:
    m = S1.set_index("SMILES").loc[shared, k]
    shared_rows.append({"feature": k, "shared_with": int(m.sum()), "shared_without": int((m == 0).sum()),
                   "estimable": bool(m.sum() >= SP.MIN_CLASS and (m == 0).sum() >= SP.MIN_CLASS),
                   "first_campaign_pp": profile(S1).get(k, np.nan), "second_campaign_pp": profile(S2).get(k, np.nan),
                   "in_full_profile_comparison": k in shared_feats})
shared_table = pd.DataFrame(shared_rows)
shared_library = {"shared_aryl_halides": len(shared), "estimable_features": int(shared_table.estimable.sum()),
                  "status": "ESTIMABLE" if shared_comparison else "NOT ESTIMABLE (fewer than five features meet >= 3 with and >= 3 without)",
                  "pearson": shared_comparison[1] if shared_comparison else None,
                  "spearman": shared_comparison[2] if shared_comparison else None,
                  "features": shared_comparison[0] if shared_comparison else None}

# ---- the profile ranking against the nine analysed reactions (no new test) ----
pr = pd.read_csv(CR / "extended/structure_profiles.csv")
pr = pr[pr.comparison.str.startswith("second Buchwald")][["reaction_a", "features", "pearson", "spearman", "record_rank_agreement", "record_shared"]]
pr = pr.sort_values("pearson", ascending=False).reset_index(drop=True)
pr["profile_rank"] = np.arange(1, len(pr) + 1)

# ---- write ----
B.to_csv(OUT / "BH_molecule_resampling.csv", index=False, lineterminator="\n", float_format="%.6f")
P.to_csv(OUT / "BH_feature_permutation_exact.csv", index=False, lineterminator="\n", float_format="%.6f")
shared_table.assign(status=shared_library["status"]).to_csv(OUT / "BH_shared_library_profile_sensitivity.csv", index=False,
                                                                          lineterminator="\n", float_format="%.6f")
pr.to_csv(OUT / "BH_profile_ranking_released.csv", index=False, lineterminator="\n", float_format="%.6f")
summary = {"verified": verified, "molecule_resampling": boot_summary, "feature_permutation": perm_summary, "shared_library": shared_library,
           "seed_molecule_resampling": SEED, "leave_two_out_all": {f"{a} | {b}": v for (a, b), v in l2o.items()}}
(OUT / "BH_robustness_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "leave_two_out_all"}, indent=1, ensure_ascii=False, default=float))
print(shared_table.round(2).to_string(index=False))
print(pr.round(3).to_string(index=False))
