"""Structural-feature contrasts and profiles of the aryl halides.

Uses the contributions on the first and second condition sets and in the second Buchwald-Hartwig
campaign, read from the core and extended outputs.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import chi2, mannwhitneyu, pearsonr, spearmanr
from pathlib import Path
import json
RESULTS = None
CORE_RESULTS = None
def write_csv(frame, path, index=False):
    frame.to_csv(path, index=index, lineterminator="\n", float_format="%.17g")
def write_json(value, path):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
def input_path(name):
    return (RESULTS if name == "campaign_contributions.csv" else CORE_RESULTS) / name

RDLogger.DisableLog("rdApp.*")
MIN_FEATURES = 5
ARYL = "aryl halide or sulfonate"
MIN_CLASS = 3
FEATURES = ["halogen next to ring N", "N–H", "substituent beside the halogen", "chloride leaving group"]
ALSO = ["iodide leaving group", "sulfonate leaving group", "heteroaromatic ring at the halogen", "five-membered ring at the halogen",
        "fused ring at the halogen", "electron-withdrawing group", "electron-donating group"]
Q = Chem.MolFromSmarts
NEXT_TO_N = [Q("[F,Cl,Br,I]-c:[n;X2;H0;+0]"), Q("O(-c:[n;X2;H0;+0])S(=O)(=O)")]
NEXT_TO_ANY_N = [Q("[F,Cl,Br,I]-c:n"), Q("O(-c:n)S(=O)(=O)")]   # pyrrole-type nitrogens counted too (check only)
NH = Q("[#7;!H0]")
SITE = [("I", Q("[I]-c")), ("Br", Q("[Br]-c")), ("sulfonate", Q("O(-c)S(=O)(=O)")), ("Cl", Q("[Cl]-c")), ("F", Q("[F]-c"))]
EWG = [Q(x) for x in ("a-C#N", "a-[N+](=O)[O-]", "a-C(F)(F)F", "a-C(=O)", "a-S(=O)(=O)")]
EDG = [Q(x) for x in ("a-[OX2][CX4]", "a-[NX3;H0;!$(N-C=O);!$(N-S)]([CX4])[CX4]")]


def features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    f = {"SMILES": smiles, "halogen next to ring N": int(any(mol.HasSubstructMatch(q) for q in NEXT_TO_N)),
         "N–H": int(mol.HasSubstructMatch(NH)), "halogen next to any ring N": int(any(mol.HasSubstructMatch(q) for q in NEXT_TO_ANY_N))}
    for name, q in SITE:
        hits = mol.GetSubstructMatches(q)
        if hits:
            carbon = mol.GetAtomWithIdx(hits[0][1])
            beside = any(not nn.GetIsAromatic() and nn.GetAtomicNum() > 1
                         for n in carbon.GetNeighbors() if n.GetIsAromatic()
                         for nn in n.GetNeighbors() if nn.GetIdx() != carbon.GetIdx())
            rings = [r for r in mol.GetRingInfo().AtomRings() if carbon.GetIdx() in r]
            ring = min(rings, key=len)
            return {**f, "leaving group": name, "substituent beside the halogen": int(beside), "chloride leaving group": int(name == "Cl"),
                    "iodide leaving group": int(name == "I"), "sulfonate leaving group": int(name == "sulfonate"),
                    "heteroaromatic ring at the halogen": int(any(mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring)),
                    "five-membered ring at the halogen": int(len(ring) == 5),
                    "fused ring at the halogen": int(any(set(r) & set(ring) and set(r) != set(ring) for r in mol.GetRingInfo().AtomRings())),
                    "electron-withdrawing group": int(any(mol.HasSubstructMatch(q) for q in EWG)),
                    "electron-donating group": int(any(mol.HasSubstructMatch(q) for q in EDG))}
    return None   # not an aryl (pseudo)halide, e.g. a benzyl bromide; left out of every class


def contributions():
    labels = pd.read_csv(input_path("reaction_labels.csv"), index_col=0)
    aryl = set(labels.index[labels.electrophile == ARYL])
    first = pd.read_csv(input_path("molecule_contributions.csv"))
    second = pd.read_csv(input_path("holdout_molecule_effects_bba.csv"))
    second = second.assign(contribution_pp=100 * second.main_effect).rename(columns={"rxn_class": "reaction", "mol": "SMILES"})
    campaign = pd.read_csv(input_path("campaign_contributions.csv"))
    frames = {"first condition set": first[first.reaction.isin(aryl)],
              "second condition set": second[second.reaction.isin(aryl)],
              "second Buchwald-Hartwig campaign": campaign}
    out = pd.concat([f[f.side == "elec"][["reaction", "SMILES", "contribution_pp"]].assign(data=k) for k, f in frames.items()], ignore_index=True)
    if set(out.data) != set(frames) or out.duplicated(["data", "reaction", "SMILES"]).any():
        raise RuntimeError("Contribution sets incomplete or duplicated")
    return out, labels


def main():
    C, labels = contributions()
    F = [features(s) for s in sorted(C.SMILES.unique())]
    skipped = sorted(s for s, f in zip(sorted(C.SMILES.unique()), F) if f is None)
    F = pd.DataFrame([f for f in F if f is not None])
    if len(skipped) > 5:
        raise RuntimeError(f"{len(skipped)} electrophiles without an aryl leaving group")
    D = C.merge(F, on="SMILES", validate="many_to_one")
    rows = []
    for (data, rc), g in D.groupby(["data", "reaction"], sort=True):
        for feat in FEATURES + ALSO + ["halogen next to any ring N"]:
            a, b = g.contribution_pp[g[feat] == 1], g.contribution_pp[g[feat] == 0]
            if len(a) < MIN_CLASS or len(b) < MIN_CLASS:
                continue
            rows.append({"data": data, "feature": feat, "reaction": rc, "metal": labels.metal.get(rc, "Pd"),
                         "with_n": len(a), "without_n": len(b), "with_mean_pp": a.mean(), "without_mean_pp": b.mean(),
                         "difference_pp": a.mean() - b.mean(), "mann_whitney_p": mannwhitneyu(a, b).pvalue})
    T = pd.DataFrame(rows)
    if T.empty:
        raise RuntimeError("No class differences computed")
    alt = T[T.feature == "halogen next to any ring N"]
    T = T[T.feature != "halogen next to any ring N"].reset_index(drop=True)
    # Benjamini-Hochberg q values over every test of each set of contributions.
    T["bh_q"] = np.nan
    for data, g in T.groupby("data"):
        p_ = g.mann_whitney_p.to_numpy()
        order = np.argsort(p_)
        q = p_[order] * len(p_) / np.arange(1, len(p_) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        T.loc[g.index[order], "bh_q"] = np.minimum(q, 1)
    T["examined_as"] = np.where(T.feature.isin(FEATURES), "reported", "also examined")
    summary = {"aryl_halides": len(F), "not_aryl_halides": skipped}
    for feat in FEATURES + ALSO:
        t = T[T.feature == feat]
        first = t[t.data == "first condition set"].set_index("reaction")
        s = {"reactions": len(first), "higher": int((first.difference_pp > 0).sum()), "lower": int((first.difference_pp < 0).sum()),
             "higher_p05": int(((first.difference_pp > 0) & (first.mann_whitney_p < .05)).sum()),
             "lower_p05": int(((first.difference_pp < 0) & (first.mann_whitney_p < .05)).sum()),
             "min_pp": float(first.difference_pp.min()), "max_pp": float(first.difference_pp.max())}
        for other in ("second condition set",):
            o = t[t.data == other].set_index("reaction").difference_pp
            k = first.difference_pp.index.intersection(o.index)
            key = other.split()[0]
            if len(k) < 4:
                if feat in FEATURES:
                    raise RuntimeError(f"Too few reactions to compare {feat} on the {other}")
                s[f"{key}_reactions"] = len(k)
                continue
            s[f"{key}_reactions"] = len(k)
            s[f"{key}_same_sign"] = int((np.sign(first.difference_pp[k]) == np.sign(o[k])).sum())
            s[f"{key}_rank_agreement"] = float(spearmanr(first.difference_pp[k], o[k])[0])
        later = t[t.data.eq("second condition set")]
        later = later.assign(first=later.reaction.map(first.difference_pp)).dropna(subset=["first"])
        s["later_tests"] = len(later)
        s["later_same_sign_as_first"] = int((np.sign(later.difference_pp) == np.sign(later["first"])).sum())
        s["all_tests"] = len(t)
        s["all_tests_lower"] = int((t.difference_pp < 0).sum())
        s["first_q05"] = int((first.bh_q < .05).sum())
        # Heterogeneity across reactions on the first set: Cochran's Q with Welch standard errors.
        se2 = []
        for rc in first.index:
            g = D[(D.data == "first condition set") & (D.reaction == rc)]
            a, b = g.contribution_pp[g[feat] == 1], g.contribution_pp[g[feat] == 0]
            se2.append(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        w = 1 / np.array(se2)
        d = first.difference_pp.to_numpy()
        Qs = float(np.sum(w * (d - np.sum(w * d) / np.sum(w)) ** 2))
        s["heterogeneity_Q"], s["heterogeneity_df"] = Qs, len(d) - 1
        s["heterogeneity_p"] = float(chi2.sf(Qs, len(d) - 1)) if len(d) > 1 else float("nan")
        s["heterogeneity_I2"] = float(max(0.0, (Qs - (len(d) - 1)) / Qs)) if Qs > 0 else 0.0
        summary[feat] = s
    first_all = T[T.data == "first condition set"]
    summary["first_set_tests"] = {"tests": len(first_all), "q05": int((first_all.bh_q < .05).sum()),
                                  "q05_reported": int((first_all[first_all.examined_as == "reported"].bh_q < .05).sum())}
    XN = "halogen next to ring N"
    # The ring-nitrogen contrast among aryl bromides only (first set).
    br = D[(D.data == "first condition set") & (D["leaving group"] == "Br")]
    rows_br = []
    for rc, g in br.groupby("reaction"):
        a, b = g.contribution_pp[g[XN] == 1], g.contribution_pp[g[XN] == 0]
        if len(a) >= MIN_CLASS and len(b) >= MIN_CLASS:
            rows_br.append({"reaction": rc, "with_n": len(a), "without_n": len(b), "difference_pp": a.mean() - b.mean(),
                            "mann_whitney_p": mannwhitneyu(a, b).pvalue})
    BR = pd.DataFrame(rows_br).set_index("reaction")
    allx = T[(T.data == "first condition set") & (T.feature == XN)].set_index("reaction").difference_pp
    kb = BR.index.intersection(allx.index)
    summary["bromides_only"] = {"reactions": len(kb), "same_sign": int((np.sign(BR.difference_pp[kb]) == np.sign(allx[kb])).sum()),
                                **{rc: float(BR.difference_pp[rc]) for rc in kb}}
    # The ring-nitrogen contrast on the molecules two reactions share, for every pair whose first-set contrasts differ in sign.
    fd = D[D.data == "first condition set"]
    series = {rc: g.set_index("SMILES") for rc, g in fd.groupby("reaction")}
    shared_rows = []
    for a_, b_ in [(x, y) for i, x in enumerate(sorted(allx.index)) for y in sorted(allx.index)[i + 1:]]:
        if np.sign(allx[a_]) == np.sign(allx[b_]):
            continue
        sh = series[a_].index.intersection(series[b_].index)
        flag = series[a_].loc[sh, XN]
        if (flag == 1).sum() < MIN_CLASS or (flag == 0).sum() < MIN_CLASS:
            continue
        da = series[a_].loc[sh].groupby(XN).contribution_pp.mean()
        db = series[b_].loc[sh].groupby(XN).contribution_pp.mean()
        shared_rows.append({"reaction_a": a_, "reaction_b": b_, "shared": len(sh), "shared_with_feature": int((flag == 1).sum()),
                            "difference_a_pp": da[1] - da[0], "difference_b_pp": db[1] - db[0],
                            "all_molecules_a_pp": allx[a_], "all_molecules_b_pp": allx[b_]})
    SH = pd.DataFrame(shared_rows)
    if SH.empty:
        raise RuntimeError("No reaction pair for the shared-molecule check")
    keeps = (np.sign(SH.difference_a_pp) == np.sign(SH.all_molecules_a_pp)) & (np.sign(SH.difference_b_pp) == np.sign(SH.all_molecules_b_pp))
    summary["shared_molecules"] = {"pairs": len(SH), "both_signs_kept": int(keeps.sum()),
                                   "opposite_on_shared": int((np.sign(SH.difference_a_pp) != np.sign(SH.difference_b_pp)).sum())}
    hu = SH[((SH.reaction_a == "Heck") & (SH.reaction_b == "Ullmann")) | ((SH.reaction_a == "Ullmann") & (SH.reaction_b == "Heck"))]
    if len(hu) == 1:
        r = hu.iloc[0]
        summary["shared_molecules"]["Heck_Ullmann"] = {"shared": int(r.shared), "with_feature": int(r.shared_with_feature),
                                                       "Heck_pp": float(r.difference_a_pp if r.reaction_a == "Heck" else r.difference_b_pp),
                                                       "Ullmann_pp": float(r.difference_b_pp if r.reaction_a == "Heck" else r.difference_a_pp)}
    write_csv(SH, RESULTS / "structure_shared_molecules.csv")
    # Structural profiles.
    prof = {d: g.pivot(index="feature", columns="reaction", values="difference_pp") for d, g in T.groupby("data")}

    def compare(x, y):
        z = pd.concat([x, y], axis=1).dropna()
        if len(z) < MIN_FEATURES:
            return None
        return {"features": len(z), "pearson": float(pearsonr(z.iloc[:, 0], z.iloc[:, 1])[0]),
                "spearman": float(spearmanr(z.iloc[:, 0], z.iloc[:, 1])[0])}
    rows_p = []
    P1 = prof["first condition set"]
    for later in ("second condition set",):
        for rc in P1.columns:
            if rc in prof[later]:
                c = compare(P1[rc], prof[later][rc])
                if c:
                    rows_p.append({"comparison": "same reaction, " + later, "reaction_a": rc, "reaction_b": rc, **c})
    transfer = pd.read_csv(input_path("extended_transfer_bba.csv"))
    transfer = transfer[(transfer.side == "elec") & ~transfer.same_reaction].set_index(["rxn_a", "rxn_b"])
    for (a_, b_), r in transfer.iterrows():
        if a_ in P1 and b_ in P1:
            c = compare(P1[a_], P1[b_])
            if c:
                rows_p.append({"comparison": "two analysed reactions", "reaction_a": a_, "reaction_b": b_, **c,
                               "record_rank_agreement": float(r.rho_obs), "record_shared": int(r.n_shared)})
    NEW = prof["second Buchwald-Hartwig campaign"]
    campaign = pd.read_csv(RESULTS / "campaign_comparisons.csv")
    for nr in NEW.columns:
        for rc in P1.columns:
            c = compare(P1[rc], NEW[nr])
            if not c:
                continue
            rec = campaign[(campaign.analysed_reaction == rc) & (campaign.side == "elec")]
            rows_p.append({"comparison": "second Buchwald-Hartwig campaign against an analysed reaction", "reaction_a": rc, "reaction_b": nr, **c,
                           "record_rank_agreement": float(rec.rho.iloc[0]) if len(rec) else np.nan,
                           "record_shared": int(rec.shared.iloc[0]) if len(rec) else np.nan})
    PR = pd.DataFrame(rows_p)
    within2 = PR[PR.comparison == "same reaction, second condition set"].pearson
    between = PR[PR.comparison == "two analysed reactions"]
    bh = PR[(PR.comparison == "second Buchwald-Hartwig campaign against an analysed reaction") & (PR.reaction_b == "Buchwald-Hartwig(CMC)")].set_index("reaction_a")
    if "Buchwald-Hartwig" not in bh.index:
        raise RuntimeError("The two Buchwald-Hartwig campaigns share too few features")
    x = pd.concat([P1["Buchwald-Hartwig"], NEW["Buchwald-Hartwig(CMC)"]], axis=1).dropna()
    loo = [pearsonr(x.drop(k).iloc[:, 0], x.drop(k).iloc[:, 1])[0] for k in x.index]
    summary["profiles"] = {
        "within_second": {"reactions": len(within2), "median": float(within2.median()), "min": float(within2.min()), "max": float(within2.max())},
        "between": {"pairs": len(between), "median": float(between.pearson.median()),
                    "profile_vs_record_rank_correlation": float(spearmanr(between.pearson, between.record_rank_agreement)[0])},
        "campaigns": {"features": int(bh.loc["Buchwald-Hartwig", "features"]), "pearson": float(bh.loc["Buchwald-Hartwig", "pearson"]),
                      "spearman": float(bh.loc["Buchwald-Hartwig", "spearman"]), "leave_one_out_min": float(min(loo)), "leave_one_out_max": float(max(loo)),
                      "closest_analysed": str(bh.pearson.idxmax()), "analysed_compared": len(bh),
                      "record_rank_agreement": float(bh.loc["Buchwald-Hartwig", "record_rank_agreement"]),
                      "record_shared": int(bh.loc["Buchwald-Hartwig", "record_shared"])}}
    write_csv(PR, RESULTS / "structure_profiles.csv")
    # The same contrast with pyrrole-type ring nitrogens counted as well.
    a1 = alt[alt.data == "first condition set"].set_index("reaction").difference_pp
    m1 = T[(T.feature == "halogen next to ring N") & (T.data == "first condition set")].set_index("reaction").difference_pp
    k = a1.index.intersection(m1.index)
    summary["halogen next to any ring N"] = {"molecules_reclassified": int((F["halogen next to any ring N"] != F["halogen next to ring N"]).sum()),
                                             "reactions_in_both": len(k), "same_sign": int((np.sign(a1[k]) == np.sign(m1[k])).sum()),
                                             "reactions_only_with_any_N": sorted(set(a1.index) - set(m1.index))}
    # A 2-bromopyridine used in Heck, Suzuki-Miyaura and Sonogashira couplings.
    anchor = "COC(=O)c1ccnc(Br)c1"
    first = C[C.data == "first condition set"].copy()   # ranked among all of the reaction's electrophiles
    first["from_top"] = first.groupby("reaction").contribution_pp.rank(ascending=False, method="min")
    first["from_bottom"] = first.groupby("reaction").contribution_pp.rank(ascending=True, method="min")
    a = first[first.SMILES == anchor].set_index("reaction")
    if not {"Heck", "Suzuki", "Sonogashira"} <= set(a.index):
        raise RuntimeError("Anchor molecule missing from a reaction")
    summary["anchor"] = {"SMILES": anchor, "name": "methyl 2-bromoisonicotinate",
                         **{rc: {"contribution_pp": float(a.contribution_pp[rc]), "from_top": int(a.from_top[rc]),
                                 "from_bottom": int(a.from_bottom[rc]), "molecules": int((first.reaction == rc).sum())} for rc in a.index}}
    write_csv(F, RESULTS / "aryl_halide_features.csv")
    write_csv(T, RESULTS / "structure_contrasts.csv")
    write_json(summary, RESULTS / "structure_readings.json")
    pd.set_option("display.width", 250)
    for feat in FEATURES + ALSO:
        print("==", feat, summary[feat])
        print(T[T.feature == feat].pivot(index="reaction", columns="data", values="difference_pp").round(1).to_string())
    print(summary["anchor"])
    print(summary["halogen next to any ring N"])
    for k in ("first_set_tests", "bromides_only", "shared_molecules", "profiles"):
        print(k, summary[k])
    print(SH.round(2).to_string())



