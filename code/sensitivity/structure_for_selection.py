"""Choosing aryl halides by a reaction's structural profile instead of by their own records (Fig. 3d, Supplementary Note 19).

Score of a molecule m under a source profile: s(m) = sum over features f of d_f * x_f(m), where x_f is the feature
indicator of Supplementary Note 17 and d_f the source's class difference (mean contribution with the feature minus
without, percentage points) for every one of the eleven features carried by at least three source molecules and absent
from at least three. Agreement is Spearman's correlation of the score (or of the source contributions) with the target
contributions. The top ten follow Supplementary Note 11: molecules tied at the tenth place share the places left; the
gain is their mean target contribution minus the mean of the molecules chosen from, and the target's own top ten among
the same molecules give the gain available.

Molecules without a record: source = each analysed reaction with at least ten aryl halides and five testable features,
target = the aryl halides of the second Buchwald-Hartwig campaign that the first campaign never ran; the ten are chosen
among those molecules only. Agreement over all of the campaign's aryl halides and, for the shared ones, against their
own first-campaign contributions is reported alongside, and the same unseen molecules are chosen by simpler rules built from the first
campaign: one feature at a time, a least-squares fit on the features, and structural similarity.
Molecules with a record: for each reaction of Fig. 3b, source = partner half A, target = partner half B on the same
molecules, under the fixed partition and the 20 further partitions of Supplementary Note 18 (first condition set); the
molecules' own half-A contributions are the alternative to the profile. Contributions are rounded to twelve decimals
(fraction scale) before ranking, as for every rank correlation of the analysis.

Run from the repository root after the main steps: python -B code/sensitivity/structure_for_selection.py
"""
import json
import os
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
REL = Path(__file__).resolve().parents[2]   # the repository root (code, data, output)
OUT = REL / "output" / "sensitivity" / "structure_for_selection"
CORE = REL / "output" / "core"
EXT = REL / "output" / "extended"
PROF = REL / "output" / "partner_profiles"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REL / "code" / "partner_profiles"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs  # noqa: E402
from rdkit.Chem import rdFingerprintGenerator  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
import partner_profiles as A  # noqa: E402

FEATURES = ["halogen next to ring N", "N–H", "substituent beside the halogen", "chloride leaving group", "iodide leaving group",
            "sulfonate leaving group", "heteroaromatic ring at the halogen", "five-membered ring at the halogen",
            "fused ring at the halogen", "electron-withdrawing group", "electron-donating group"]
_c = {}


def canon(s):
    if s not in _c:
        _c[s] = Chem.MolToSmiles(Chem.MolFromSmiles(s))
    return _c[s]


feat = pd.read_csv(EXT / "aryl_halide_features.csv")
feat["c"] = feat.SMILES.map(canon)
feat = feat.drop_duplicates("c").set_index("c")


def series(df, value):
    x = df.assign(c=df.SMILES.map(canon))
    x = x[x.c.isin(feat.index)]
    assert not x.c.duplicated().any()
    return x.set_index("c")[value].astype(float)


def class_differences(values, X):
    """Class differences of the features testable on these molecules (>= 3 with, >= 3 without)."""
    cols = [j for j in range(X.shape[1]) if 3 <= X[:, j].sum() <= len(X) - 3]
    return cols, np.array([values[X[:, j] == 1].mean() - values[X[:, j] == 0].mean() for j in cols])


def profile_score(source_values, source_X, target_X):
    cols, d = class_differences(np.asarray(source_values, float), source_X)
    return target_X[:, cols] @ d, len(cols)


def top_ten(pred, target, k=10):
    """Mean target value of the k highest-ranked molecules, molecules tied at the k-th place sharing the places left."""
    pred, target = np.round(np.asarray(pred, float), 12), np.asarray(target, float)
    cut = np.sort(pred)[::-1][k - 1]
    above, tied = pred > cut, pred == cut
    return (target[above].sum() + (k - above.sum()) * target[tied].mean()) / k


def gains(pred, target):
    target = np.asarray(target, float)
    base = target.mean()
    return top_ten(pred, target) - base, top_ten(target, target) - base


# ---- molecules without a record: the second Buchwald-Hartwig campaign ----
mc = pd.read_csv(CORE / "molecule_contributions.csv")
mc = mc[mc.side == "elec"]
cc = pd.read_csv(EXT / "campaign_contributions.csv")
target = series(cc[cc.side == "elec"], "contribution_pp")
first_bh = series(mc[mc.reaction == "Buchwald-Hartwig"], "contribution_pp")
shared = target.index.intersection(first_bh.index)
unseen = target.index.difference(first_bh.index)
assert len(shared) > 0 and len(unseen) > 0 and len(unseen) + len(shared) == len(target)
Xt, Xu, Xs = (feat.loc[i, FEATURES].to_numpy(float) for i in (target.index, unseen, shared))
rows = []
for rxn in sorted(mc.reaction.unique()):
    src = series(mc[mc.reaction == rxn], "contribution_pp")
    if len(src) < 10:
        continue
    Xsrc = feat.loc[src.index, FEATURES].to_numpy(float)
    s_new, nf = profile_score(src.to_numpy(), Xsrc, Xu)
    if nf < 5:
        continue
    s_all, _ = profile_score(src.to_numpy(), Xsrc, Xt)
    g, best = gains(s_new, target[unseen])
    rows.append(dict(source=rxn, features=nf, rho_unseen=spearmanr(s_new, target[unseen]).statistic,
                     rho_all=spearmanr(s_all, target).statistic, top_ten_gain_pp=g, gain_available_pp=best, share=g / best))
campaign = pd.DataFrame(rows).sort_values("rho_unseen", ascending=False).reset_index(drop=True)
assert campaign.source.iloc[0] == "Buchwald-Hartwig"
campaign.to_csv(OUT / "campaign_selection_by_profile.csv", index=False)
campaign.assign(molecules=len(unseen)).to_csv(OUT / "Figure3d_selection_without_record.csv", index=False, float_format="%.15g",
                                              lineterminator="\n")
s_shared, _ = profile_score(first_bh.to_numpy(), feat.loc[first_bh.index, FEATURES].to_numpy(float), Xs)
summary = {"n_target": int(len(target)), "n_shared": int(len(shared)), "n_unseen": int(len(unseen)),
           "shared_rank_rho": float(spearmanr(first_bh[shared], target[shared]).statistic),
           "shared_profile_rho": float(spearmanr(s_shared, target[shared]).statistic)}

# ---- the same unseen molecules chosen by simpler rules built from the first campaign ----
# One feature at a time (that feature's class difference); a least-squares fit on the same features; and structural
# similarity: each molecule scored by the mean first-campaign contribution of its k nearest aryl halides (ECFP4,
# radius 2, 2,048 bits, Tanimoto) or by the similarity-weighted mean over all of them.
y_new = target[unseen].to_numpy()
v_src = first_bh.to_numpy()
X_src = feat.loc[first_bh.index, FEATURES].to_numpy(float)
cols, d = class_differences(v_src, X_src)
rules = {"structural profile": Xu[:, cols] @ d}
for j, dj in zip(cols, d):
    if Xu[:, j].min() < Xu[:, j].max():   # A constant feature cannot rank the target molecules.
        rules[f"one feature: {FEATURES[j]}"] = dj * Xu[:, j]
beta = np.linalg.lstsq(np.column_stack([np.ones(len(X_src)), X_src[:, cols]]), v_src, rcond=None)[0]
rules["least squares on the same features"] = np.column_stack([np.ones(len(Xu)), Xu[:, cols]]) @ beta
fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
f_src = [fp.GetFingerprint(Chem.MolFromSmiles(m)) for m in first_bh.index]
sim = np.array([DataStructs.BulkTanimotoSimilarity(fp.GetFingerprint(Chem.MolFromSmiles(m)), f_src) for m in unseen])
near = np.argsort(-sim, axis=1, kind="stable")
for k in (1, 3, 5, 10):
    rules[f"nearest aryl halides, k = {k}"] = v_src[near[:, :k]].mean(axis=1)
rules["similarity-weighted, all"] = (sim * v_src).sum(axis=1) / sim.sum(axis=1)
base_rows = []
for name, s in rules.items():
    g, best = gains(s, y_new)
    base_rows.append(dict(rule=name, rho=spearmanr(s, y_new).statistic, top_ten_gain_pp=g, gain_available_pp=best))
baselines = pd.DataFrame(base_rows)
assert abs(baselines.iloc[0].top_ten_gain_pp - campaign.set_index("source").loc["Buchwald-Hartwig", "top_ten_gain_pp"]) < 1e-9
baselines.to_csv(OUT / "campaign_selection_baselines.csv", index=False, float_format="%.15g", lineterminator="\n")
similar = baselines[baselines.rule.str.startswith(("nearest", "similarity"))].sort_values("top_ten_gain_pp")
knn = rules[similar.rule.iloc[-1]]   # the similarity rule with the largest gain
singles = baselines[baselines.rule.str.startswith("one feature")].sort_values("top_ten_gain_pp")
single = rules[singles.rule.iloc[-1]]   # the single feature with the largest gain
lsq = rules["least squares on the same features"]
rng = np.random.default_rng(20261014)
boot = []
for _ in range(5000):   # Resample the unseen molecules with replacement.
    b = rng.integers(0, len(y_new), len(y_new))
    gp, gk = gains(rules["structural profile"][b], y_new[b])[0], gains(knn[b], y_new[b])[0]
    rp = spearmanr(rules["structural profile"][b], y_new[b]).statistic
    boot.append((rp, gp, gp - gk, gp - gains(lsq[b], y_new[b])[0], gp - gains(single[b], y_new[b])[0],
                 rp - spearmanr(lsq[b], y_new[b]).statistic))
boot = np.array(boot)
summary["median_max_similarity"] = float(np.median(sim.max(axis=1)))
summary["bootstrap"] = {"replicates": 5000, "seed": 20261014, "similarity_rule": similar.rule.iloc[-1],
                        "rho_95": [float(x) for x in np.percentile(boot[:, 0], [2.5, 97.5])],
                        "gain_95": [float(x) for x in np.percentile(boot[:, 1], [2.5, 97.5])],
                        "gain_minus_similarity_95": [float(x) for x in np.percentile(boot[:, 2], [2.5, 97.5])],
                        "share_profile_above_similarity": float((boot[:, 2] > 0).mean()),
                        "single_feature_rule": singles.rule.iloc[-1],
                        "gain_minus_least_squares_95": [float(x) for x in np.percentile(boot[:, 3], [2.5, 97.5])],
                        "share_profile_above_least_squares": float((boot[:, 3] > 0).mean()),
                        "gain_minus_single_feature_95": [float(x) for x in np.percentile(boot[:, 4], [2.5, 97.5])],
                        "share_profile_above_single_feature": float((boot[:, 4] > 0).mean()),
                        "rho_minus_least_squares_95": [float(x) for x in np.percentile(boot[:, 5], [2.5, 97.5])]}

# ---- molecules with a record: partner halves under the fixed and 20 further partitions ----
A.load(OUT / "scratch")
wells, conds, pairs = A.PC.released()
lay, ys = A.PC.layouts_from(*A.PC.design_inputs(wells, conds, pairs, "primary"), A.G)[:2]
reactions = pd.read_csv(PROF / "A1_first_set.csv").dropna(subset=["r_structure"]).reaction.tolist()
assert len(reactions) > 0 and len(A.SALTS) == 20
long = []
for rxn in reactions:
    lt = lay[rxn]
    assert A.split_matches_release(lt, "elec"), rxn
    for salt in [None] + A.SALTS:
        halves = A.half_fits(lt, ys[rxn], "elec", salt)
        ev = A.evaluable(lt, halves, "elec", 2)
        rows_ = [A.feature_row(n) for n in lt.e_name[ev]]
        keep = [i for i, r in enumerate(rows_) if r is not None]
        # contributions rounded to twelve decimals (fraction scale), as for every rank correlation (Supplementary Note 8)
        a = 100 * np.round(halves[0]["elec"][ev][keep], 12)
        b = 100 * np.round(halves[1]["elec"][ev][keep], 12)
        X = np.array([rows_[i] for i in keep], float)
        s, nf = profile_score(a, X, X)
        g_rank, best = gains(a, b)
        g_prof, _ = gains(s, b)
        long.append(dict(reaction=rxn, partition=salt or "fixed", molecules=len(a), features=nf,
                         rho_ranking=spearmanr(a, b).statistic, rho_profile=spearmanr(s, b).statistic,
                         top_ten_ranking_pp=g_rank, top_ten_profile_pp=g_prof, gain_available_pp=best))
L = pd.DataFrame(long)
L.to_csv(OUT / "partner_selection_by_partition.csv", index=False)

# The fixed partition must equal the released partner-half contributions (molecules and values).
ph = pd.read_csv(CORE / "partner_half_contributions_bba.csv")
ph = ph[(ph.side == "elec") & ph.evaluable_k2]
for rxn in reactions:
    x = ph[ph.reaction == rxn].pivot_table(index="molecule", columns="half", values="contribution").dropna().round(12) * 100
    x = x[x.index.map(canon).isin(feat.index)]
    r = L[(L.reaction == rxn) & (L.partition == "fixed")].iloc[0]
    assert len(x) == r.molecules and abs(spearmanr(x.A, x.B).statistic - r.rho_ranking) < 1e-9, rxn

fixed = L[L.partition == "fixed"].set_index("reaction")
alt = L[L.partition != "fixed"].assign(diff=lambda d: d.top_ten_profile_pp - d.top_ten_ranking_pp)
agg = alt.groupby("reaction").agg(partitions=("diff", "size"), profile_higher=("diff", lambda v: int((v > 0).sum())),
                                  median_difference_pp=("diff", "median"), median_ranking_pp=("top_ten_ranking_pp", "median"),
                                  median_profile_pp=("top_ten_profile_pp", "median"), median_available_pp=("gain_available_pp", "median"),
                                  median_rho_ranking=("rho_ranking", "median"), median_rho_profile=("rho_profile", "median"))
partners = fixed[["molecules", "features", "rho_ranking", "rho_profile", "top_ten_ranking_pp", "top_ten_profile_pp",
                  "gain_available_pp"]].join(agg).reset_index()
assert (partners.partitions == 20).all()
partners.to_csv(OUT / "partner_selection_by_profile.csv", index=False)
(OUT / "campaign_shared.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(campaign.round(2).to_string(index=False))
print(json.dumps(summary, indent=1))
print(partners.round(2).to_string(index=False))
