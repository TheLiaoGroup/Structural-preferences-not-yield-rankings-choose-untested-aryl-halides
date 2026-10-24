"""Table transforms of Supplementary Tables 1-6."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from definitions import REACTION_NAMES, REACTIONS, REAGENT_COLUMNS

L = SimpleNamespace(REACTIONS=REACTIONS, REAGENT_COLUMNS=REAGENT_COLUMNS)
S = SimpleNamespace(SCHEMES={key: {"name": value} for key, value in REACTION_NAMES.items()})


def reaction_name(rc):
    """A reaction as its scheme names it (Figure 1a, Figure S1), SNAr without markup; a reaction
    type without a scheme keeps the collection's identifier."""
    return S.SCHEMES[rc]["name"].replace("<sub>N</sub>", "N") if rc in S.SCHEMES else rc

def fixed(values, places):
    """Numbers printed to a fixed number of places, so 0.30 is not shown as 0.3; missing stays empty."""
    return ["" if pd.isna(v) else f"{v:.{places}f}" for v in values]

B = SimpleNamespace(reaction_name=reaction_name, fixed=fixed)


def condition_table(rows, conds, names, label):
    """One row per condition of one set, in coverage order, with its composition."""
    order = {rc: {c: i for i, c in enumerate(g.condition_id)} for rc, g in conds.groupby("rxn_class")}
    cov = (rows.drop_duplicates(["rxn_class", "condition_id", "elec", "nuc"])
               .groupby(["rxn_class", "condition_id"]).size().rename("pairs_covered"))
    first = rows.drop_duplicates(["rxn_class", "condition_id"]).set_index(["rxn_class", "condition_id"])
    s1 = []
    for rc, g in conds.groupby("rxn_class"):
        for cid in g.condition_id:
            r = first.loc[(rc, cid)]
            reagents = [str(names.get(r[c], r[c])) for c in L.REAGENT_COLUMNS if r[c]]
            s1.append({"reaction": rc, "set": label, "rank": order[rc][cid] + 1, "condition": cid,
                       "reagents": "; ".join(reagents) or "none",
                       "solvent": names.get(r.solvent_id, r.solvent_id),
                       "temperature_C": float(r.temp_id), "time_h": float(r.time_id),
                       "pairs_covered": int(cov.loc[(rc, cid)])})
    return pd.DataFrame(s1)

def tables(sets, names, load):
    """sets: (label, rows, conditions) per set of conditions listed in Table S1."""
    S1 = pd.concat([condition_table(rows, conds, names, label) for label, rows, conds in sets], ignore_index=True)
    S1 = S1.sort_values(["reaction", "set", "rank"])
    assert not S1.reagents.str.contains("CMP").any(), "a reagent identifier has no name"

    design = load("extended_design_bba.csv")
    rel = load("extended_reliability_bba.csv").pivot(
        index="rxn_class", columns="side", values="reliability")
    assert design.in_extended_set.all(), "Table S2 lists the fifteen analysed reactions"
    S2 = design.drop(columns="in_extended_set").rename(columns={"rxn_class": "reaction"})
    S2["reliability, electrophile"] = S2.reaction.map(rel.get("elec")).round(3)
    S2["reliability, nucleophile"] = S2.reaction.map(rel.get("nuc")).round(3)
    S2 = S2.sort_values("pairs", ascending=False)

    S3 = load("extended_transfer_bba.csv")
    held = load("holdout_transfer_bba.csv")[["side", "rxn_a", "rxn_b", "n_shared",
                                                             "rho_disattenuated"]]
    S3 = S3.merge(held.rename(columns={"n_shared": "held out: shared molecules",
                                       "rho_disattenuated": "held out: corrected"}),
                  on=["side", "rxn_a", "rxn_b"], how="left")
    for c in ("B_mean_A_selected", "B_mean_B_best", "B_mean_all"):
        S3[c] = (S3[c] * 100).round(1)
    S3 = S3.rename(columns={"rho_obs": "rank correlation", "ceiling": "measurement ceiling",
                            "rho_disattenuated": "corrected", "n_shared": "shared molecules",
                            "same_reaction": "same reaction",
                            "B_mean_all": "B: all shared (points)",
                            "B_mean_A_selected": "B: A's best ten (points)",
                            "B_mean_B_best": "B: B's own best ten (points)"})
    S3 = S3.sort_values(["same reaction", "side", "corrected"], ascending=[True, True, False]).round(3)

    S4 = load("diagnose_BuchwaldHartwig_SNAr_nuc_bba.csv").rename(
        columns={"mol": "SMILES", "Buchwald-Hartwig": "contribution, Buchwald-Hartwig",
                 "SNAr": "contribution, SNAr",
                 "influence": "change in the overall correlation when removed"})
    held4 = load("holdout_amines_bba.csv").rename(
        columns={"mol": "SMILES", "Buchwald-Hartwig": "held out: Buchwald-Hartwig",
                 "SNAr": "held out: SNAr"}).drop(columns="class")
    S4 = S4.merge(held4, on="SMILES", how="left")
    S4 = S4.sort_values("contribution, Buchwald-Hartwig", ascending=False).round(4)
    return {"TableS1": S1, "TableS2": S2, "TableS3": S3, "TableS4": S4}

def for_readers(T):
    """Column names and values as the reader of the Supporting Information sees them; the sorting
    in tables() uses the working names."""
    whole = lambda s: s.astype(int) if (s == s.round()).all() else s
    yes_no = {True: "yes", False: "no"}
    name = B.reaction_name
    S1 = T["TableS1"].rename(columns={"rank": "coverage rank", "temperature_C": "temperature (°C)",
                                      "time_h": "time (h)", "pairs_covered": "pairs covered"})
    S1["temperature (°C)"], S1["time (h)"] = whole(S1["temperature (°C)"]), whole(S1["time (h)"])
    S1["reaction"] = S1.reaction.map(name)
    S2 = T["TableS2"].rename(columns={"median_partners_elec": "median partners per electrophile",
                                      "median_partners_nuc": "median partners per nucleophile"})
    S2["reaction"] = S2.reaction.map(name)
    for c in ("median partners per electrophile", "median partners per nucleophile"):
        S2[c] = B.fixed(S2[c], 1)
    for c in ("reliability, electrophile", "reliability, nucleophile"):
        S2[c] = B.fixed(S2[c], 3)
    S3 = T["TableS3"].rename(columns={"rxn_a": "reaction A", "rxn_b": "reaction B"})
    S3["side"] = S3.side.map({"elec": "electrophile", "nuc": "nucleophile"})
    S3["same reaction"] = S3["same reaction"].map(yes_no)
    S3["reaction A"], S3["reaction B"] = S3["reaction A"].map(name), S3["reaction B"].map(name)
    for c in ("rank correlation", "measurement ceiling", "corrected", "retention", "held out: corrected"):
        S3[c] = B.fixed(S3[c], 3)
    for c in ("B: A's best ten (points)", "B: B's own best ten (points)", "B: all shared (points)"):
        S3[c] = B.fixed(S3[c], 1)
    S4 = T["TableS4"].rename(columns=lambda c: c.replace("Buchwald-Hartwig", "Buchwald–Hartwig"))
    for c in S4.columns:
        if c not in ("SMILES", "class"):
            S4[c] = B.fixed(S4[c], 4)
    for column in (S3.side, S3["same reaction"]):
        assert column.notna().all(), f"{column.name}: a value has no reader's form"
    out = {"TableS1": S1, "TableS2": S2, "TableS3": S3, "TableS4": S4}
    assert all(len(out[k]) == len(T[k]) for k in T)
    return out

def dense_tables(D):
    h1 = D["results"].query("test == 'H1' and scale == 'raw' and label == 'primary'").set_index("rxn_class")
    h2 = D["results"].query("test == 'H2' and scale == 'raw' and label == 'primary'").set_index("rxn_class")
    h3 = D["results"].query("test == 'H3' and scale == 'raw' and label == 'primary'").set_index("rxn_class")
    rows = D["rows"]
    floor = (rows.assign(hit=rows["yield"] >= 0.20).groupby(["rxn_class", "elec", "nuc"]).hit.max()
                 .groupby("rxn_class").mean())
    spec = D["spectrum"].query("scale == 'raw' and label == 'primary'").set_index("rxn_class")
    t1 = pd.DataFrame({
        "reaction": [reaction_name(r) for r in L.REACTIONS],
        "pairs": [int(h1.loc[r, "n_pairs"]) for r in L.REACTIONS],
        "degrees of freedom for the remainder": [int(h1.loc[r, "interaction_df"]) for r in L.REACTIONS],
        "additive share": fixed([h1.loc[r, "estimate"] for r in L.REACTIONS], 3),
        "95% CI": [f"{h1.loc[r,'ci_low']:.3f} to {h1.loc[r,'ci_high']:.3f}" for r in L.REACTIONS],
        "reliability of a pair effect": fixed([h1.loc[r, "reliability"] for r in L.REACTIONS], 3),
        "pairs reaching 20% yield": [f"{floor[r]:.1%}" for r in L.REACTIONS],
        "look-up gain (yield points)": fixed([h2.loc[r, "estimate"] * 100 for r in L.REACTIONS], 1),
        "variance ratio, electrophile to nucleophile": fixed([h3.loc[r, "estimate"] for r in L.REACTIONS], 2),
        "leading singular value share": fixed([spec.loc[r, "first_singular_share"] for r in L.REACTIONS], 3),
    })
    rel = D["rel"].pivot(index="rxn_class", columns="side", values="reliability").round(3)
    des = D["design"].query("in_extended_set").set_index("rxn_class")
    agree = D["hold_cells"].pivot(index="rxn_class", columns="side", values="rho").round(3)
    t2 = pd.DataFrame({
        "reaction": [reaction_name(r) for r in des.index], "conditions": des.conditions.values,
        "pairs": des.pairs.values,
        "electrophiles": des.electrophiles.values, "nucleophiles": des.nucleophiles.values,
        "reliability, electrophile side": fixed(rel.loc[des.index, "elec"].values, 3),
        "reliability, nucleophile side": fixed(rel.loc[des.index, "nuc"].values, 3),
        "agreement with twenty other conditions, electrophile side": fixed(agree.loc[des.index, "elec"].values, 3),
        "agreement with twenty other conditions, nucleophile side": fixed(agree.loc[des.index, "nuc"].values, 3),
    }).sort_values("pairs", ascending=False, kind="stable")
    return t1, t2
