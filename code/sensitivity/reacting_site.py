"""The halogen beside a pyridine-type ring nitrogen, counted only where it is the halogen that reacts (Supplementary Note 17).

The structural feature "halogen next to ring N" matches any aryl halogen (or sulfonate) on a carbon bonded to a
pyridine-type ring nitrogen, whichever halogen couples. Here the coupling site of every aryl-halide electrophile in
the reactions where the feature is tested is read from the recorded target products: the halogen (or sulfonate) that
the electrophile and nucleophile together carry and the target product does not. A molecule keeps the feature only
when that group sits beside the ring nitrogen. The class difference of the main analysis (mean contribution with the feature minus without)
is then recomputed on the first and second condition sets, and the second Buchwald-Hartwig campaign's value is put
into the profile comparison of the two campaigns (Pearson over the features testable in both). Each reaction's class
difference is also set beside the feature's coefficient in a least-squares fit of the contributions on every feature
testable in that reaction (at least three molecules with it and three without), so that features that occur together
are fitted jointly.

Run from the repository root after the main steps: python -B code/sensitivity/reacting_site.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr

sys.dont_write_bytecode = True
RDLogger.DisableLog("rdApp.*")
REL = Path(__file__).resolve().parents[2]   # the repository root (code, data, output)
OUT = REL / "output" / "sensitivity" / "reacting_site"
CORE = REL / "output" / "core"
EXT = REL / "output" / "extended"
WELLS = REL / "data" / "experimental_wells.csv"
OUT.mkdir(parents=True, exist_ok=True)
FEATURE = "halogen next to ring N"
HAL = {"F", "Cl", "Br", "I"}
_mols = {}


def mol(s):
    if s not in _mols:
        _mols[s] = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return _mols[s]


def canon(s):
    m = mol(s)
    return Chem.MolToSmiles(m) if m is not None else s


def counts(m):
    return Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetSymbol() in HAL | {"S"})


def pyridine_n(atom):
    return atom.GetSymbol() == "N" and atom.GetIsAromatic() and atom.GetTotalNumHs() == 0 and atom.GetDegree() == 2 \
        and atom.GetFormalCharge() == 0


def beside_n(smiles, lost):
    """1 if every atom of the reacting group sits on a carbon bonded to a pyridine-type N, 0 if none does, None if unknown."""
    m = mol(smiles)
    if "S" in lost:   # a sulfonate reacts (a triflate also loses its fluorines)
        hits = [any(pyridine_n(n) for n in c.GetNeighbors()) for o in m.GetAtoms() if o.GetSymbol() == "O"
                for c in o.GetNeighbors() if c.GetIsAromatic() and any(x.GetSymbol() == "S" for x in o.GetNeighbors())]
    else:
        hits = [any(pyridine_n(n) for n in a.GetNeighbors()[0].GetNeighbors()) for a in m.GetAtoms()
                if a.GetSymbol() in lost and a.GetDegree() == 1 and a.GetNeighbors()[0].GetIsAromatic()]
    if not hits or len(set(hits)) > 1:
        return None
    return int(hits[0])


sc = pd.read_csv(EXT / "structure_contrasts.csv")
first_ref = sc[(sc.data == "first condition set") & (sc.feature == FEATURE)].set_index("reaction").difference_pp
second_ref = sc[(sc.data == "second condition set") & (sc.feature == FEATURE)].set_index("reaction").difference_pp
CAMPAIGN = "Buchwald-Hartwig(CMC)"
reactions = sorted(set(first_ref.index) | {CAMPAIGN})
feats = pd.read_csv(EXT / "aryl_halide_features.csv")
labelled = dict(zip(feats.SMILES.map(canon), feats[FEATURE]))
FEATURES = ["halogen next to ring N", "N–H", "substituent beside the halogen", "chloride leaving group", "iodide leaving group",
            "sulfonate leaving group", "heteroaromatic ring at the halogen", "five-membered ring at the halogen",
            "fused ring at the halogen", "electron-withdrawing group", "electron-donating group"]
rows_x = dict(zip(feats.SMILES.map(canon), feats[FEATURES].to_numpy(float)))

w = pd.read_csv(WELLS, usecols=["rxn_class", "electrophile_smiles", "nucleophile_smiles", "product_smiles", "measurement_status"])
w = w[w.measurement_status.isin(["quantified", "zero_by_rule"])]
w = w[w.rxn_class.isin(reactions)].dropna(subset=["product_smiles"])
w = w.drop_duplicates(["rxn_class", "electrophile_smiles", "nucleophile_smiles", "product_smiles"])
lost = {}
for r in w.itertuples():
    e, n, p = mol(r.electrophile_smiles), mol(r.nucleophile_smiles), mol(r.product_smiles)
    if e is None or n is None or p is None:
        continue
    ce, cn, cp = counts(e), counts(n), counts(p)
    group = tuple(sorted(k for k in ce if ce[k] + cn[k] - cp[k] > 0))
    lost.setdefault((r.rxn_class, canon(r.electrophile_smiles)), Counter())[group] += 1
rows = []
for (rxn, smi), c in lost.items():
    if smi in labelled:
        group = c.most_common(1)[0][0]
        rows.append(dict(reaction=rxn, SMILES=smi, reacting_group="+".join(group), feature=int(labelled[smi]),
                         reacting_site_beside_n=beside_n(smi, group)))
site = pd.DataFrame(rows).sort_values(["reaction", "SMILES"])
site.to_csv(OUT / "reacting_site_members.csv", index=False)
cls = site[site.feature == 1]
moved = cls[cls.reacting_site_beside_n == 0]
assert len(site[(site.feature == 0) & (site.reacting_site_beside_n == 1)]) == 0
assert cls.reacting_site_beside_n.isna().sum() + (cls.reacting_site_beside_n == 1).sum() + len(moved) == len(cls)
off = set(zip(moved.reaction, moved.SMILES))


def joint(values):
    """The feature's coefficient when every testable feature is fitted together (least squares with an intercept)."""
    v = values.assign(c=values.SMILES.map(canon))
    v = v[v.c.isin(rows_x)]
    X = np.array([rows_x[c] for c in v.c])
    cols = [j for j in range(X.shape[1]) if 3 <= X[:, j].sum() <= len(X) - 3]
    assert 0 in cols
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X[:, cols]]), v.contribution_pp.to_numpy(), rcond=None)[0]
    return float(beta[1 + cols.index(0)])


def difference(values, rxn, site_specific):
    v = values.assign(c=values.SMILES.map(canon))
    v = v[v.c.isin(labelled)]
    lab = np.array([0 if site_specific and (rxn, c) in off else labelled[c] for c in v.c], dtype=bool)
    x = v.contribution_pp.to_numpy()
    return float(x[lab].mean() - x[~lab].mean())


mc = pd.read_csv(CORE / "molecule_contributions.csv")
mc = mc[mc.side == "elec"]
h = pd.read_csv(CORE / "holdout_molecule_effects_bba.csv", usecols=["scale", "label", "rxn_class", "side", "mol", "main_effect"])
h = h[(h.scale == "raw") & (h.label == "validation") & (h.side == "elec")]
h = pd.DataFrame({"reaction": h.rxn_class, "SMILES": h.mol, "contribution_pp": 100 * h.main_effect})
out = []
for cset, data, ref in (("first", mc, first_ref), ("second", h, second_ref)):
    for rxn in ref.index:
        d = data[data.reaction == rxn]
        any_ = difference(d, rxn, False)
        assert abs(any_ - ref[rxn]) < 1e-6, (cset, rxn, any_, ref[rxn])   # the class difference of the main analysis, reproduced
        out.append(dict(condition_set=cset, reaction=rxn, released_pp=ref[rxn], reacting_site_pp=difference(d, rxn, True),
                        joint_fit_pp=joint(d)))
out = pd.DataFrame(out)
assert len(out) == 2 * out.reaction.nunique()
out.to_csv(OUT / "reacting_site_differences.csv", index=False)

cc = pd.read_csv(EXT / "campaign_contributions.csv")
cc = cc[cc.side == "elec"]
c2 = sc[sc.data == "second Buchwald-Hartwig campaign"].set_index("feature").difference_pp
c1 = sc[(sc.data == "first condition set") & (sc.reaction == "Buchwald-Hartwig")].set_index("feature").difference_pp
common = c1.index.intersection(c2.index)
assert abs(difference(cc, CAMPAIGN, False) - c2[FEATURE]) < 1e-6 and len(common) == 7
c2s = c2.copy()
c2s[FEATURE] = difference(cc, CAMPAIGN, True)
summary = {"class_entries": int(len(cls)), "reacting_site_beside_n": int((cls.reacting_site_beside_n == 1).sum()),
           "reacting_site_elsewhere": int(len(moved)), "unresolved": int(cls.reacting_site_beside_n.isna().sum()),
           "molecules_elsewhere": sorted(set(moved.SMILES)), "entries_elsewhere": sorted(map(list, off)),
           "campaign_profile_released": float(pearsonr(c1[common], c2[common])[0]),
           "campaign_profile_reacting_site": float(pearsonr(c1[common], c2s[common])[0]),
           "signs_unchanged": bool((np.sign(out.released_pp) == np.sign(out.reacting_site_pp)).all()),
           "joint_fit_signs_unchanged": int((np.sign(out.released_pp) == np.sign(out.joint_fit_pp)).sum())}
(OUT / "reacting_site_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(out.round(1).to_string())
print(json.dumps(summary, indent=1))
