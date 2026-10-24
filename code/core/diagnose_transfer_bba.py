"""Amine classes, and four cross-reaction comparisons examined molecule by molecule."""
import itertools

import numpy as np

import pandas as pd

from rdkit import Chem

from scipy.stats import spearmanr

import analysis_bba as A

import lib_bba as L

import rank_bba as RK

PATTERNS = [
    ("azole N-H", "[nH]"),
    ("amide or lactam N-H", "[NX3;H1,H2]-[#6]=[O,S]"),
    ("sulfonamide N-H", "[NX3;H1,H2]-S(=O)=O"),
    ("primary aryl amine", "[NX3;H2;!$(N-[#6]=[O,S,N]);!$(N-[S,P]=O)]-a"),
    ("secondary aryl amine", "[NX3;H1;!$(N-[#6]=[O,S,N]);!$(N-[S,P]=O)](-a)-[#6]"),
    ("secondary alkyl amine", "[NX3;H1;!$(N-[#6]=[O,S,N]);!$(N-[S,P]=O);!$(N-a);!$(N-[#7,#8])]([CX4])[CX4]"),
    ("primary alkyl amine", "[NX3;H2;!$(N-[#6]=[O,S,N]);!$(N-[S,P]=O);!$(N-[#7,#8])]-[CX4]"),
]

PATTERNS = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in PATTERNS]

PAIRS_TO_CHECK = [("Buchwald-Hartwig", "SNAr", "nuc"), ("Acylation", "Chan-Lam", "nuc"),
                  ("Heck", "Suzuki", "elec"), ("SNAr", "Suzuki", "elec")]

def classify(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for name, pattern in PATTERNS:
        if mol.HasSubstructMatch(pattern):
            return name
    return "other"


def reproduce(rows):
    P = A.pair_effects(rows, "raw")
    eff = {}
    for rc in L.REACTIONS:
        fit = A.stage_two(P[P.rxn_class == rc].reset_index(drop=True))
        eff[rc] = {"elec": fit["a"], "nuc": fit["b"]}

    for a, b, side in PAIRS_TO_CHECK:
        xa, xb = eff[a][side], eff[b][side]
        shared = sorted(xa.index.intersection(xb.index))
        t = pd.DataFrame({"mol": shared, a: xa[shared].values, b: xb[shared].values})
        t["class"] = t.mol.map(classify)
        overall = RK.spearman(t[a], t[b])
        print(f"\n{'=' * 100}\n== {a} vs {b}, {side} side: {len(t)} shared molecules, "
              f"Spearman {overall:+.3f}")

        print("\n-- by class")
        for name, g in t.groupby("class"):
            r = RK.spearman(g[a], g[b]) if len(g) >= 5 else np.nan
            print(f"   {name:22s} n={len(g):3d}  within-class Spearman "
                  f"{'   n/a' if not np.isfinite(r) else f'{r:+.3f}'}"
                  f"   mean {a[:12]} {g[a].mean():+.4f}   mean {b[:12]} {g[b].mean():+.4f}")

        print("\n-- drop one class at a time")
        for name in sorted(t["class"].unique()):
            s = t[t["class"] != name]
            r = RK.spearman(s[a], s[b]) if len(s) >= 5 else np.nan
            print(f"   without {name:22s} n={len(s):3d}  Spearman {r:+.3f}"
                  f"   ({'sign flips' if np.sign(r) != np.sign(overall) else 'same sign'})")

        infl = []
        for i in range(len(t)):
            s = t.drop(t.index[i])
            infl.append(RK.spearman(s[a], s[b]) - overall)
        t["influence"] = infl
        print("\n-- the five molecules that most hold the correlation where it is "
              "(removing them moves it most)")
        for _, r in t.reindex(t.influence.abs().sort_values(ascending=False).index).head(5).iterrows():
            print(f"   move {r.influence:+.4f}  {a[:12]} {r[a]:+.4f}  {b[:12]} {r[b]:+.4f}  "
                  f"{r['class']:22s} {r.mol}")

        print(f"\n-- every shared molecule, sorted by its {a} effect")
        for _, r in t.sort_values(a, ascending=False).iterrows():
            print(f"   {a[:12]:12s} {r[a]:+.4f}   {b[:12]:12s} {r[b]:+.4f}   "
                  f"{r['class']:22s} {r.mol}")
        t.to_csv(L.OUT / f"diagnose_{a.replace('-', '')}_{b.replace('-', '')}_{side}_bba.csv", index=False)
