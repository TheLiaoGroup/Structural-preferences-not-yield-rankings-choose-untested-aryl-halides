"""Reaction labels from the condition reagents and the canonical substrates."""
import numpy as np
import pandas as pd
from types import SimpleNamespace
LOOKUP_TYPE = {"Buchwald-Hartwig(CMC)":"B-H (CMC)"}
KIND = {"aryl halide or sulfonate": "[c]-[F,Cl,Br,I,$(O-S(=O)(=O)-[#6,#9])]",
        "carbonyl": "[CX3;!$(C-[OX2H1])](=O)([#6,#1])",
        "carboxylic acid": "[CX3](=O)[OX2H1]",
        "arylboronic acid": "c-B(-O)-O",
        "N–H": "[N,n;!H0;!$(N-C=O);!$(N-S(=O)=O)]"}
def metal_labels(conditions):
    """The metal named in at least 15 of a reaction's 20 analyzed conditions (PEPPSI counts as Pd)."""
    import re
    out = {}
    for rc, g in conditions.groupby("rxn_class"):
        hits = {}
        for text in g.reagents.fillna(""):
            present = {m for m in ("Pd", "Cu", "Ni", "Fe") if re.search(rf"(?<![A-Za-z]){m}(?![a-z])", text)}
            if "PEPPSI" in text:
                present.add("Pd")
            for m in present:
                hits[m] = hits.get(m, 0) + 1
        major = [m for m, k in hits.items() if k >= 15]
        out[rc] = major[0] if len(major) == 1 else ("none" if not hits else "mixed " + "/".join(sorted(hits)))
    return out


def kind_labels(pairs, side):
    from rdkit import Chem
    pats = {k: Chem.MolFromSmarts(v) for k, v in KIND.items()}
    out = {}
    for rc, p in pairs.groupby("rxn_class"):
        mols = [Chem.MolFromSmiles(s) for s in p[side].unique()]
        share = {k: np.mean([m.HasSubstructMatch(q) for m in mols]) for k, q in pats.items()}
        best = max(share, key=share.get)
        out[rc] = best if share[best] >= 0.8 else "other"
    return out

CR=SimpleNamespace(KIND=KIND, metal_labels=metal_labels, kind_labels=kind_labels)
def condition_names(lookup, rxn_class, condition_ids):
    g = lookup[lookup.reaction_type == LOOKUP_TYPE[rxn_class]].set_index("condition_id")
    cols = [f"Reagent{i}_Standardized Name" for i in range(1, 7)]
    return ["; ".join(str(v) for v in g.loc[c, cols] if isinstance(v, str) and v.strip())
            if c in g.index else "" for c in condition_ids]


def metal_of(lookup, rxn_class, condition_ids):
    frame = pd.DataFrame({"rxn_class": rxn_class,
                          "reagents": condition_names(lookup, rxn_class, condition_ids)})
    return CR.metal_labels(frame)[rxn_class]


def kind_of(molecules, rxn_class="x"):
    """SMARTS class carried by >= 80% of the molecules (chemistry_readings.kind_labels) and the
    N–H share, which the frozen grouping needs for nucleophiles."""
    mols = [m for m in molecules if m]
    if not mols:
        return "none", float("nan")
    frame = pd.DataFrame({"rxn_class": rxn_class, "m": mols})
    from rdkit import Chem
    nh = Chem.MolFromSmarts(CR.KIND["N–H"])
    return CR.kind_labels(frame, "m")[rxn_class], float(np.mean([Chem.MolFromSmiles(m).HasSubstructMatch(nh) for m in mols]))
