"""Same-measure sensitivity and grouping-alone expectation for the structural profiles of Fig. 3b,c (Supplementary Note 18).

For the seven aryl-halide reactions with a structural profile, partner_profiles.compare already returns four
numbers on one matched set of aryl halides: molecule Spearman (the canonical ranking statistic), molecule
Pearson, profile Pearson (the canonical profile statistic) and profile Spearman. This script
  1. re-derives those four numbers on the canonical split for the first and second condition sets and checks
     them against A1_first_set.csv and A3_second_set.csv;
  2. computes the same four numbers under the 20 salted partner splits, for both condition sets
     (the first-set molecule Spearman and profile Pearson must equal A2_salts.csv);
  3. repeats the matched random-grouping control (partner_profiles.permutation_null: the same seed text, the
     same 10,000 permutations of whole feature rows) and scores each permutation with Pearson, as in the main
     analysis, and with Spearman, so that the profile statistic and its control use one metric.
No estimator, threshold, feature or split is changed.

Run from the repository root after the main steps: python -B code/sensitivity/profile_metrics.py
"""
import hashlib
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

REL = PKG = Path(__file__).resolve().parents[2]   # the repository root (code, data, output)
OUT = PKG / "output" / "sensitivity" / "profile_metrics"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG / "code" / "partner_profiles"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
import partner_profiles as A  # noqa: E402

A.load(OUT / "scratch")
CR = REL / "output"
labels = pd.read_csv(CR / "core/reaction_labels.csv", index_col=0)
aryl = sorted(labels.index[labels.electrophile == "aryl halide or sulfonate"])
assert len(aryl) > 0
wells, conds, pairs = A.PC.released()
LAY = {lab: A.PC.layouts_from(*A.PC.design_inputs(wells, conds, pairs, lab), A.G)[:2] for lab in ("primary", "validation")}
SETS = {"first set": "primary", "second set": "validation"}
REF = {"first set": pd.read_csv(CR / "partner_profiles/A1_first_set.csv").set_index("reaction"),
       "second set": pd.read_csv(CR / "partner_profiles/A3_second_set.csv").set_index("reaction")}
A2 = pd.read_csv(CR / "partner_profiles/A2_salts.csv").set_index(["salt", "reaction"])
METRICS = {"rho_molecule": "molecule_spearman", "pearson_molecule": "molecule_pearson",
           "r_structure": "profile_pearson", "spearman_structure": "profile_spearman"}


def permutation_draws(xa, xb, F, tested, seed_text, n=A.PERMUTATIONS):
    """partner_profiles.permutation_null, returning the permuted class differences of both halves."""
    rng = np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    Ft = F[:, tested].astype(float)
    c1 = Ft.sum(0)
    c0 = len(Ft) - c1
    Fp = Ft[np.argsort(rng.random((n, len(Ft))), axis=1)]

    def diffs(x):
        s1 = np.einsum("pmf,m->pf", Fp, x)
        return s1 / c1 - (x.sum() - s1) / c0
    return diffs(xa), diffs(xb)


def row_pearson(da, db):
    da, db = da - da.mean(1, keepdims=True), db - db.mean(1, keepdims=True)
    return (da * db).sum(1) / np.sqrt((da * da).sum(1) * (db * db).sum(1))


def row_spearman(da, db):
    return row_pearson(rankdata(da, axis=1), rankdata(db, axis=1))


canon, nulls = [], []
for tag, lab in SETS.items():
    lay, ys = LAY[lab]
    for rxn in aryl:
        row, data = A.compare(lay[rxn], A.half_fits(lay[rxn], ys[rxn]))
        ref = REF[tag].loc[rxn]
        for k in ("evaluable", "matched", "features", *METRICS):
            a, b = row.get(k, np.nan), ref[k]
            assert (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) < 1e-12, (tag, rxn, k, a, b)
        if data is None:
            continue
        da, db = permutation_draws(*data, f"A-permutation|{tag}|{rxn}")
        p_null, s_null = row_pearson(da, db), row_spearman(da, db)
        # The Pearson null of the main analysis must be reproduced draw for draw (checked through its summaries).
        assert abs(np.median(p_null) - ref.null_median) < 1e-12 and abs(np.percentile(p_null, 97.5) - ref["null_97.5"]) < 1e-12
        rec = {"set": tag, "reaction": rxn, "matched_aryl_halides": row["matched"], "features": row["features"],
               **{v: row[k] for k, v in METRICS.items()}}
        rec["profile_minus_molecule_canonical_mix"] = rec["profile_pearson"] - rec["molecule_spearman"]
        rec["profile_minus_molecule_pearson"] = rec["profile_pearson"] - rec["molecule_pearson"]
        rec["profile_minus_molecule_spearman"] = rec["profile_spearman"] - rec["molecule_spearman"]
        for name, obs, null in (("pearson", rec["profile_pearson"], p_null), ("spearman", rec["profile_spearman"], s_null)):
            rec[f"random_group_{name}_median"] = float(np.median(null))
            rec[f"random_group_{name}_97.5"] = float(np.percentile(null, 97.5))
            rec[f"random_group_{name}_p_upper"] = float((1 + (null >= obs - 1e-12).sum()) / (1 + len(null)))
        # Aggregation alone: how often a random grouping of the same class sizes beats the molecule-level statistic.
        rec["random_group_share_above_molecule_canonical_mix"] = float((p_null > rec["molecule_spearman"]).mean())
        rec["random_group_share_above_molecule_pearson"] = float((p_null > rec["molecule_pearson"]).mean())
        rec["random_group_share_above_molecule_spearman"] = float((s_null > rec["molecule_spearman"]).mean())
        canon.append(rec)
        nulls.append((tag, rxn, rec["profile_pearson"], rec["profile_spearman"], p_null, s_null))
C = pd.DataFrame(canon)
assert len(C) == 2 * C.reaction.nunique()
# The p values of the main analysis use ">=" on the same draws; the tolerance above only guards exact ties.
for _, r in C.iterrows():
    assert abs(r.random_group_pearson_p_upper - REF[r.set].loc[r.reaction].null_p_upper) < 1e-12, (r.set, r.reaction)

# The pooled random-grouping control (A5) with each statistic: mean over the seven reactions.
pooled = []
for tag in SETS:
    rr = [n for n in nulls if n[0] == tag]
    for name, j in (("pearson", 2), ("spearman", 3)):
        obs = np.mean([n[j] for n in rr])
        t = np.vstack([n[j + 2] for n in rr]).mean(0)
        pooled.append({"set": tag, "statistic": f"profile {name}", "reactions": len(rr), "mean_observed": obs,
                       "pooled_null_median": float(np.median(t)), "pooled_null_97.5": float(np.percentile(t, 97.5)),
                       "pooled_p": float((1 + (t >= obs - 1e-12).sum()) / (1 + len(t))),
                       "above_own_null_median": int(sum(n[j] > np.median(n[j + 2]) for n in rr))})
PO = pd.DataFrame(pooled)


def at_least(probs, k):
    """P(at least k successes) for independent trials with the given probabilities (Poisson binomial)."""
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum())


# The 6/7 count against aggregation alone: expected count of reactions whose random-grouping profile beats the
# molecule-level statistic, and the chance of reaching the observed count (descriptive; reactions treated as independent).
agg = []
for tag in SETS:
    g = C[C.set == tag]
    for name, col, diff in (("canonical_mix", "random_group_share_above_molecule_canonical_mix", "profile_minus_molecule_canonical_mix"),
                            ("pearson", "random_group_share_above_molecule_pearson", "profile_minus_molecule_pearson"),
                            ("spearman", "random_group_share_above_molecule_spearman", "profile_minus_molecule_spearman")):
        k = int((g[diff] > 0).sum())
        agg.append({"set": tag, "pairing": name, "observed_count": k, "reactions": len(g),
                    "expected_count_under_random_grouping": float(g[col].sum()),
                    "p_at_least_observed_under_random_grouping": at_least(g[col].to_numpy(), k)})
AG = pd.DataFrame(agg)
a5 = pd.read_csv(CR / "partner_profiles/A5_pooled_permutation.csv").set_index("set")
for tag in SETS:
    p = PO[(PO.set == tag) & (PO.statistic == "profile pearson")].iloc[0]
    assert abs(p.pooled_p - a5.loc[tag, "pooled_p"]) < 1e-12 and p.above_own_null_median == a5.loc[tag, "above_own_null_median"]

# The 20 salted splits, both condition sets.
alt = []
for tag, lab in SETS.items():
    lay, ys = LAY[lab]
    for salt in A.SALTS:
        for rxn in sorted(C.reaction.unique()):
            row, data = A.compare(lay[rxn], A.half_fits(lay[rxn], ys[rxn], salt=salt))
            rec = {"set": tag, "reaction": rxn, "salt": salt, "evaluable": row["evaluable"], "matched": row["matched"],
                   "features": row.get("features", np.nan), **{v: row.get(k, np.nan) for k, v in METRICS.items()}}
            if tag == "first set":
                ref = A2.loc[(salt, rxn)]
                for k in ("evaluable", "matched", "features", "rho_molecule", "r_structure"):
                    a, b = row.get(k, np.nan), ref[k]
                    assert (pd.isna(a) and pd.isna(b)) or abs(float(a) - float(b)) < 1e-12, (salt, rxn, k, a, b)
            alt.append(rec)
S = pd.DataFrame(alt)
for a, b, name in (("profile_pearson", "molecule_spearman", "canonical_mix"), ("profile_pearson", "molecule_pearson", "pearson"),
                   ("profile_spearman", "molecule_spearman", "spearman")):
    S[f"profile_above_molecule_{name}"] = np.where(S[a].notna() & S[b].notna(), S[a] > S[b], np.nan)

# Count of reactions (of 7) in which the profile exceeds the molecule statistic, per split and pairing.
counts = []
for (tag, salt), g in S.groupby(["set", "salt"]):
    rec = {"set": tag, "split": salt, "reactions_with_profile": int(g.profile_pearson.notna().sum())}
    for name in ("canonical_mix", "pearson", "spearman"):
        rec[f"profile_above_{name}"] = int((g[f"profile_above_molecule_{name}"] == 1).sum())
    counts.append(rec)
for tag in SETS:
    g = C[C.set == tag]
    counts.append({"set": tag, "split": "canonical", "reactions_with_profile": len(g),
                   "profile_above_canonical_mix": int((g.profile_minus_molecule_canonical_mix > 0).sum()),
                   "profile_above_pearson": int((g.profile_minus_molecule_pearson > 0).sum()),
                   "profile_above_spearman": int((g.profile_minus_molecule_spearman > 0).sum())})
K = pd.DataFrame(counts)

# Per reaction over the 20 splits: share of splits where the profile is higher, by pairing.
per = (S.groupby(["set", "reaction"]).agg(splits_with_profile=("profile_pearson", lambda x: int(x.notna().sum())),
                                          share_above_canonical_mix=("profile_above_molecule_canonical_mix", "mean"),
                                          share_above_pearson=("profile_above_molecule_pearson", "mean"),
                                          share_above_spearman=("profile_above_molecule_spearman", "mean"),
                                          molecule_spearman_median=("molecule_spearman", "median"),
                                          molecule_pearson_median=("molecule_pearson", "median"),
                                          profile_pearson_median=("profile_pearson", "median"),
                                          profile_spearman_median=("profile_spearman", "median"))
       .reset_index())

C.to_csv(OUT / "same_metric_results.csv", index=False, lineterminator="\n", float_format="%.6f")
S.to_csv(OUT / "same_metric_alternative_splits.csv", index=False, lineterminator="\n", float_format="%.6f")
K.to_csv(OUT / "same_metric_counts_by_split.csv", index=False, lineterminator="\n")
per.to_csv(OUT / "same_metric_alternative_by_reaction.csv", index=False, lineterminator="\n", float_format="%.4f")
PO.to_csv(OUT / "same_metric_pooled_random_grouping.csv", index=False, lineterminator="\n", float_format="%.6f")
AG.to_csv(OUT / "same_metric_count_vs_random_grouping.csv", index=False, lineterminator="\n", float_format="%.4f")
print(C[["set", "reaction", "random_group_share_above_molecule_canonical_mix", "random_group_share_above_molecule_pearson",
         "random_group_share_above_molecule_spearman"]].round(3).to_string(index=False))
print(AG.round(4).to_string(index=False))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(C.round(3).to_string(index=False))
print(PO.round(4).to_string(index=False))
print(K[K.split == "canonical"].to_string(index=False))
print(K[K.split != "canonical"].groupby("set")[["reactions_with_profile", "profile_above_canonical_mix", "profile_above_pearson",
                                               "profile_above_spearman"]].describe().round(2).T.to_string())
print(per.round(3).to_string(index=False))
