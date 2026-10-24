"""Further values of the main text and Supplementary Notes 12 and 17: prediction gains with intervals above zero,
like-for-like arylamine leads by ligand family, and the profiles of Suzuki-Miyaura coupling with boronic acids and
with pinacol boronates, read from the outputs of the main analysis.

Run from the repository root after the main steps: python -B code/sensitivity/additional_sensitivities.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
REL = Path(__file__).resolve().parents[2]; OUT = REL / "output" / "sensitivity" / "additional"; OUT.mkdir(parents=True, exist_ok=True)
CR = REL / "output"
res = {}
# Extended Data Table 1: prediction gains with intervals above zero (degree 2).
g = pd.read_csv(CR / "core/gated_A_prediction_bba.csv")
g = g[g.degree == 2]
assert len(g) == 15
el = g[["reaction", "I_EN", "I_EN low", "I_EN high"]].rename(columns={"I_EN": "est", "I_EN low": "low", "I_EN high": "high"}).assign(side="electrophile")
nu = g[["reaction", "I_NE", "I_NE low", "I_NE high"]].rename(columns={"I_NE": "est", "I_NE low": "low", "I_NE high": "high"}).assign(side="nucleophile")
pg = pd.concat([el, nu], ignore_index=True)
res["prediction"] = {"partner_sides": len(pg), "interval_above_zero": int((pg.low > 0).sum()),
                     "reactions_with_one": int(pg[pg.low > 0].reaction.nunique()),
                     "reactions": sorted(pg[pg.low > 0].reaction.unique()),
                     "point_estimate_negative": int((pg.est < 0).sum()), "point_estimate_positive_interval_spans_zero": int(((pg.est > 0) & (pg.low <= 0)).sum()),
                     "interval_below_zero": int((pg.high < 0).sum())}
pg.to_csv(OUT / "prediction_gains_all_sides.csv", index=False, float_format="%.4f")
# Benjamini-Hochberg q over the 14 random-grouping tests (7 reactions x 2 condition sets), Pearson profile.
a1 = pd.read_csv(CR / "partner_profiles/A1_first_set.csv").dropna(subset=["null_p_upper"])
a3 = pd.read_csv(CR / "partner_profiles/A3_second_set.csv").dropna(subset=["null_p_upper"])
p = pd.concat([a1.assign(set="first"), a3.assign(set="second")])[["set", "reaction", "r_structure", "null_p_upper"]].reset_index(drop=True)
assert len(p) > 0
o = np.argsort(p.null_p_upper.to_numpy())
q = p.null_p_upper.to_numpy()[o] * 14 / np.arange(1, 15)
q = np.minimum.accumulate(q[::-1])[::-1]
p.loc[p.index[o], "bh_q_over_14"] = np.minimum(q, 1)
res["random_group_tests"] = p.round(4).to_dict(orient="records")
# Like-for-like arylamine lead by ligand family (Buchwald-Hartwig).
a = pd.read_csv(CR / "core/amine_classes_by_condition_family.csv")
lead = {}
for fam, t in a[a.reaction == "Buchwald-Hartwig"].groupby("condition_family"):
    ar = t[t["class"] == "primary aryl amine"]
    ot = t[t["class"] != "primary aryl amine"]
    lead[fam] = {"arylamines_mean_pp": float(ar.mean_pp.iloc[0]), "others_mean_pp": float(np.average(ot.mean_pp, weights=ot.amines)),
                 "lead_pp": float(ar.mean_pp.iloc[0] - np.average(ot.mean_pp, weights=ot.amines)), "arylamines": int(ar.amines.iloc[0]),
                 "others": int(ot.amines.sum())}
res["brettphos"] = lead
# Kumada electrophiles without an aryl leaving group, and amide-coupling nucleophiles with a sulfonamide.
mc = pd.read_csv(CR / "core/molecule_contributions.csv")
acyl = Chem.MolFromSmarts("[c,C]C(=O)[Cl,Br,I]")
kum = mc[(mc.reaction == "Kumada") & (mc.side == "elec")]
res["kumada_acyl_halides"] = [s for s in kum.SMILES if Chem.MolFromSmiles(s).HasSubstructMatch(acyl)]
res["kumada_electrophiles"] = len(kum)
sulf = Chem.MolFromSmarts("[NX3;!H0]S(=O)(=O)")
am = mc[(mc.reaction == "Acylation") & (mc.side == "nuc")]
hits = [s for s in am.SMILES if Chem.MolFromSmiles(s).HasSubstructMatch(sulf)]
res["amide_sulfonamide_nucleophiles"] = {"count": len(hits), "of": len(am), "smiles": hits}
# Which electrophiles were left out of the aryl-halide features (Methods: benzylic bromide, alkenyl triflate, benzoyl chloride).
import importlib.util
spec = importlib.util.spec_from_file_location("sp", REL / "code" / "extended" / "structure_profiles.py")
sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
labels = pd.read_csv(CR / "core/reaction_labels.csv", index_col=0)
aryl = set(labels.index[labels.electrophile == "aryl halide or sulfonate"])
el = mc[mc.reaction.isin(aryl) & (mc.side == "elec")]
res["no_aryl_leaving_group"] = sorted({(r, s) for r, s in zip(el.reaction, el.SMILES) if sp.features(s) is None})
# Fig. 2d numbers.
f2 = pd.read_csv(CR / "presentation/source_data/Figure2d_amine_class_summary.csv")
res["fig2d_class_summary"] = f2.to_dict(orient="records")
s4 = pd.read_csv(CR / "presentation/tables/TableS4.csv")
res["tableS4_columns"] = list(s4.columns)
json.dump(res, open(OUT / "additional_sensitivities.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(res, indent=1, ensure_ascii=False, default=str))

# Suzuki-Miyaura with boronic acids against pinacol boronates (descriptive): profiles compared by the rule of the
# main analysis (features testable in both, >= 5; Pearson and Spearman).
from scipy.stats import pearsonr, spearmanr
T = pd.read_csv(CR / "extended/structure_contrasts.csv")
sb = {}
for data in ("first condition set", "second condition set"):
    P = T[T.data == data].pivot(index="feature", columns="reaction", values="difference_pp")
    z = P[["Suzuki", "Suzuki Bpin"]].dropna()
    sb[data] = {"features": len(z), "pearson": float(pearsonr(z.Suzuki, z["Suzuki Bpin"])[0]) if len(z) >= 5 else None,
                "spearman": float(spearmanr(z.Suzuki, z["Suzuki Bpin"])[0]) if len(z) >= 5 else None,
                "values": z.round(2).to_dict(orient="index")}
res["suzuki_vs_bpin_profiles"] = sb
pr = pd.read_csv(CR / "extended/structure_profiles.csv")
bt = pr[pr.comparison == "two analysed reactions"].pearson
res["between_reaction_profiles"] = {"pairs": int(len(bt)), "median": float(bt.median()), "share_negative": float((bt < 0).mean())}
json.dump(res, open(OUT / "additional_sensitivities.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: res[k] for k in ("suzuki_vs_bpin_profiles", "between_reaction_profiles")}, indent=1, ensure_ascii=False))
