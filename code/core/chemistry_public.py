"""Chemical readings of the fitted contributions that the manuscript reports in yield points.

Three questions, each answered from the released primary-design wells with the frozen two-stage
estimator (analysis_bba.two_way_fit, then analysis_bba.stage_two):

1. Amine classes by condition family. Is the primary (hetero)arylamine advantage in
   Buchwald-Hartwig amination carried by the BrettPhos-type ligands that make up nine of its
   twenty conditions? Is the SNAr disadvantage the same under acid, organic-base and
   inorganic-base conditions? Each family is refitted on its own wells, within its largest
   connected piece.
2. Amine classes by number of partners. A molecule met with one partner has a contribution
   equal to that pair's adjusted yield; the class contrast is repeated for molecules with at
   least two and three partners.
3. Retrospective selection against cross-reaction agreement. The gain from choosing ten building
   blocks by their record in another reaction, set against the rank agreement of the two
   reactions on their shared molecules.
"""
from __future__ import annotations
import importlib
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu
from public_common import *

PAA = "primary aryl amine"
BH, SNAR = "Buchwald-Hartwig", "SNAr"
SNAR_FAMILY = {"AcOH": "acid", "TFA": "acid",
               "2,6-lutidine": "organic base", "TEA": "organic base", "Py": "organic base",
               "DIPEA": "organic base", "DBU": "organic base",
               "K2CO3": "inorganic base", "Cs2CO3": "inorganic base", "KOAc": "inorganic base",
               "NaOtAm": "inorganic base", "NaOMe": "inorganic base", "": "no additive"}
AGREEMENT_CUT = 0.25


def largest_piece(rows):
    pairs = rows[["elec", "nuc"]].drop_duplicates()
    ei, en = pd.factorize(pairs.elec)
    ni, nn = pd.factorize(pairs.nuc)
    size = len(en) + len(nn)
    graph = coo_matrix((np.ones(len(pairs)), (ei, len(en) + ni)), shape=(size, size))
    labels = connected_components(graph, directed=False)[1][ei]
    keep = pairs[labels == np.bincount(labels).argmax()]
    return rows.merge(keep, on=["elec", "nuc"])


def nucleophile_contributions(A, rows):
    rows = largest_piece(rows)
    eff = A.two_way_fit(rows["yield"].values, rows.elec + " >> " + rows.nuc, rows.condition_id)
    table = pd.DataFrame([k.split(" >> ") for k in eff.index], columns=["elec", "nuc"]).assign(p=eff.values)
    partners = table.groupby("nuc").elec.nunique()
    b = A.stage_two(table)["b"]
    return pd.DataFrame({"mol": b.index, "contribution_pp": 100 * b.values,
                         "partners": partners.reindex(b.index).values}), len(rows)


def class_contrast(frame, label):
    paa = frame[frame["class"] == PAA].contribution_pp
    rest = frame[frame["class"] != PAA].contribution_pp
    if len(paa) < 3 or len(rest) < 3:
        raise RuntimeError(f"Too few amines for the class contrast: {label}")
    return {"paa_n": len(paa), "paa_mean_pp": paa.mean(), "other_n": len(rest), "other_mean_pp": rest.mean(),
            "difference_pp": paa.mean() - rest.mean(), "mann_whitney_p": mannwhitneyu(paa, rest).pvalue}


KIND = {"aryl halide or sulfonate": "[c]-[F,Cl,Br,I,$(O-S(=O)(=O)-[#6,#9])]",
        "carbonyl": "[CX3;!$(C-[OX2H1])](=O)([#6,#1])",
        "carboxylic acid": "[CX3](=O)[OX2H1]",
        "arylboronic acid": "c-B(-O)-O",
        "N–H": "[N,n;!H0;!$(N-C=O);!$(N-S(=O)=O)]"}


def pairs_primary(pairs):
    p = pairs[(pairs.design == "twenty conditions") & (pairs["set"] == "analysed")]
    return p.rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})


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


def reaction_labels(conditions, pairs):
    """Each reaction's metal (named in at least 15 of its 20 analysed conditions) and the kind of
    molecule that at least 80% of its molecules on each side carry."""
    metal = metal_labels(conditions)
    ekind, nkind = kind_labels(pairs, "elec"), kind_labels(pairs, "nuc")
    return pd.DataFrame({"metal": metal, "electrophile": ekind, "nucleophile": nkind}).sort_index()


def main():
    L, E, G = legacy()
    A = importlib.import_module("analysis_bba")
    D = importlib.import_module("diagnose_transfer_bba")
    wells, conditions, pairs = released()
    w, c, _ = design_inputs(wells, conditions, pairs, "primary")
    effects_all = None

    # 1. Condition families.
    bh = c[c.rxn_class == BH]
    brett = bh.reagents.str.contains("BrettPhos")
    families = {BH: {"BrettPhos-type ligand": bh[brett].condition_id.tolist(),
                     "other ligands": bh[~brett].condition_id.tolist()}}
    sn = c[c.rxn_class == SNAR].assign(family=lambda f: f.reagents.map(SNAR_FAMILY))
    if sn.family.isna().any():
        raise RuntimeError(f"Unclassified SNAr reagents: {sorted(set(sn.reagents[sn.family.isna()]))}")
    families[SNAR] = {k: g.condition_id.tolist() for k, g in sn.groupby("family")}
    classes, contrasts, by_family = [], [], {}
    for reaction, fams in families.items():
        rows = w[w.rxn_class == reaction]
        if len(rows) == 0:
            raise RuntimeError(f"No primary wells for {reaction}")
        for family, ids in {"all twenty": c[c.rxn_class == reaction].condition_id.tolist(), **fams}.items():
            if not ids:
                raise RuntimeError(f"Empty condition family {reaction}/{family}")
            contrib, n_wells = nucleophile_contributions(A, rows[rows.condition_id.isin(ids)])
            contrib["class"] = contrib.mol.map(D.classify)
            by_family[(reaction, family)] = contrib.set_index("mol")
            for cls, g in contrib.groupby("class"):
                classes.append({"reaction": reaction, "condition_family": family, "conditions": len(ids),
                                "wells": n_wells, "class": cls, "amines": len(g), "mean_pp": g.contribution_pp.mean()})
            contrasts.append({"reaction": reaction, "condition_family": family, "conditions": len(ids),
                              "wells": n_wells, "minimum_partners": 1, **class_contrast(contrib, family)})
            if family == "all twenty":
                for k in (2, 3):
                    sub = contrib[contrib.partners >= k]
                    contrasts.append({"reaction": reaction, "condition_family": family, "conditions": len(ids),
                                      "wells": n_wells, "minimum_partners": k, **class_contrast(sub, f"{family} k>={k}")})
    a, b = by_family[(BH, "BrettPhos-type ligand")], by_family[(BH, "other ligands")]
    shared = a.index.intersection(b.index)
    ligand_agreement = {"amines": len(shared),
                        "rank_agreement": G.spearman(a.loc[shared, "contribution_pp"], b.loc[shared, "contribution_pp"])}
    write_csv(pd.DataFrame(classes), RESULTS / "amine_classes_by_condition_family.csv")
    write_csv(pd.DataFrame(contrasts), RESULTS / "amine_class_contrasts.csv")

    # Shared-amine panel (Fig. 2d): molecule-level class tests on the full-data contributions.
    effects = pd.read_csv(RESULTS / "full_data_effects_bba.csv")
    effects = effects[(effects.scale == "raw") & (effects.label == "primary")]
    if effects.duplicated(["rxn_class", "side", "mol"]).any():
        raise RuntimeError("Duplicate full-data contributions")
    eff = {r: effects[(effects.rxn_class == r) & (effects.side == "nuc")].set_index("mol").main_effect for r in (BH, SNAR)}
    common = sorted(eff[BH].index.intersection(eff[SNAR].index))
    panel = pd.DataFrame({"mol": common, "BH_pp": 100 * eff[BH][common].values, "SNAr_pp": 100 * eff[SNAR][common].values})
    panel["class"] = panel.mol.map(D.classify)
    shared_tests = {side: class_contrast(panel.rename(columns={side: "contribution_pp"}), side) for side in ("BH_pp", "SNAr_pp")}
    write_csv(panel, RESULTS / "shared_amines_BH_SNAr.csv")
    # The same shared amines on the twenty held-out conditions per reaction.
    held = pd.read_csv(RESULTS / "holdout_molecule_effects_bba.csv")
    if set(held.scale) != {"raw"}:
        raise RuntimeError("Held-out molecule effects on an unexpected scale")
    held = {r: held[(held.rxn_class == r) & (held.side == "nuc")].set_index("mol").main_effect for r in (BH, SNAR)}
    held_panel = panel.assign(BH_pp=panel.mol.map(held[BH]) * 100, SNAr_pp=panel.mol.map(held[SNAR]) * 100)
    held_tests = {}
    for side in ("BH_pp", "SNAr_pp"):
        sub = held_panel.dropna(subset=[side]).rename(columns={side: "contribution_pp"})
        held_tests[side] = class_contrast(sub, "held-out " + side)

    # 3. Selection gain against cross-reaction agreement (19 recorded directions).
    transfer = pd.read_csv(RESULTS / "extended_transfer_bba.csv")
    sel = transfer[(~transfer.same_reaction) & (transfer.n_shared >= 20)].copy()
    if len(sel) == 0:
        raise RuntimeError("No selection directions")
    sel["source_gain_pp"] = 100 * (sel.B_mean_A_selected - sel.B_mean_all)
    sel["target_gain_pp"] = 100 * (sel.B_mean_B_best - sel.B_mean_all)
    sel["share_of_target_gain"] = sel.source_gain_pp / sel.target_gain_pp
    similar, opposite = sel[sel.rho_obs >= AGREEMENT_CUT], sel[sel.rho_obs < 0]
    selection = {"directions": len(sel), "above_random": int((sel.source_gain_pp > 0).sum()),
                 "mean_gain_pp": sel.source_gain_pp.mean(), "median_gain_pp": sel.source_gain_pp.median(),
                 "agreement_vs_gain_rank_correlation": G.spearman(sel.rho_obs, sel.source_gain_pp),
                 "agreement_cut": AGREEMENT_CUT,
                 "similar_n": len(similar), "similar_mean_gain_pp": similar.source_gain_pp.mean(),
                 "similar_median_share": similar.share_of_target_gain.median(),
                 "opposite_n": len(opposite), "opposite_mean_gain_pp": opposite.source_gain_pp.mean(),
                 "opposite_below_random": int((opposite.source_gain_pp < 0).sum())}
    write_csv(sel[["side", "rxn_a", "rxn_b", "n_shared", "rho_obs", "source_gain_pp", "target_gain_pp",
                   "share_of_target_gain"]], RESULTS / "selection_vs_agreement.csv")
    # 4. Each reaction's metal and the kind of molecule on each side (labels from conditions and
    #    substrates only).
    labels = reaction_labels(c, pairs_primary(pairs))
    write_csv(labels.rename_axis("rxn_class").reset_index(), RESULTS / "reaction_labels.csv")
    write_csv(labels.rename_axis("rxn_class").reset_index(), RESULTS / "reaction_labels_bba.csv")
    # Every molecule's contribution in every reaction, as a lookup table for readers (Table S10).
    lookup = effects.assign(contribution_pp=100 * effects.main_effect)[["rxn_class", "side", "mol", "contribution_pp", "n_pairs"]]
    lookup = lookup.rename(columns={"rxn_class": "reaction", "mol": "SMILES", "n_pairs": "partners"}).sort_values(["reaction", "side", "contribution_pp"], ascending=[True, True, False])
    write_csv(lookup, RESULTS / "molecule_contributions.csv")
    write_json({"bh_ligand_family_agreement": ligand_agreement, "shared_amine_tests": shared_tests,
                "shared_amine_tests_held_out_conditions": held_tests,
                "selection": selection}, RESULTS / "chemistry_readings.json")
    print(pd.DataFrame(contrasts).round(3).to_string(index=False))
    print(ligand_agreement, shared_tests, selection, sep="\n")


if __name__ == "__main__":
    main()
