"""Source Data of the Article figures, one table per panel, named by the Article's figure numbers.

Built from the panel tables (output/presentation/panel_data), the partner-profile outputs
(output/partner_profiles), the core contributions and Supplementary Table 2; nothing is fitted here.
Writes output/presentation/source_data."""
from __future__ import annotations

import pandas as pd

from presentation_io import ROOT, OUTPUT, SOURCES as PANELS, TABLES, read_csv
import plot_figures as BF

OUT = OUTPUT / "source_data"
PROFILES = ROOT / "output/partner_profiles"
CORE = ROOT / "output/core"
COPIES = (  # panel table -> Source Data of the Article figure
    ("Figure1b_three_changes.csv", "Figure1b_three_changes.csv"),
    ("Figure2b_pooled_portability.csv", "Figure1c_pooled_portability.csv"),
    ("Figure2a_portability_by_set.csv", "Figure2a_portability_by_set.csv"),
    ("Figure3b_structure_effects.csv", "Figure2c_structure_effects.csv"),
    ("Figure3d_BH_SNAr_amines.csv", "Figure2d_BH_SNAr_amines.csv"),
    ("Figure3d_amine_class_summary.csv", "Figure2d_amine_class_summary.csv"),
    ("Figure4a_heldout_amide_pairs.csv", "Figure4a_heldout_amide_pairs.csv"),
    ("Figure4a_metrics.csv", "Figure4a_metrics.csv"),
    ("Figure4b_selection_vs_agreement.csv", "Figure4b_selection_vs_agreement.csv"),
    ("Figure3c_Heck_Suzuki_molecules.csv", "Figure4c_Heck_Suzuki_molecules.csv"),
    ("Figure1a_amide_matrix.csv", "ExtendedDataFig2_amide_matrix.csv"),
    ("ExtendedDataFig2_defined_benchmark.csv", "ExtendedDataFig3_defined_benchmark.csv"),
    ("ExtendedDataFig2_pooled_shortfall.csv", "ExtendedDataFig3_pooled_shortfall.csv"),
    ("ExtendedDataFig3_undefined_benchmark.csv", "ExtendedDataFig4_undefined_benchmark.csv"),
    ("ExtendedDataFig4a_leave_one_reaction_out.csv", "ExtendedDataFig5a_leave_one_reaction_out.csv"),
    ("ExtendedDataFig4b_outcome_scale.csv", "ExtendedDataFig5b_outcome_scale.csv"),
    ("ExtendedDataFig5_reaction_map.csv", "ExtendedDataFig6_reaction_map.csv"),
    ("ExtendedDataFig6bc_profile_comparisons.csv", "ExtendedDataFig7ab_profile_comparisons.csv"),
    ("ExtendedDataFig7_prediction_all_reactions.csv", "ExtendedDataFig8_prediction_all_reactions.csv"),
)
WRITTEN = []   # names of the Source Data tables written


def save(frame, name, inputs, derivation):
    if len(frame) == 0:
        raise RuntimeError(f"empty panel table: {name}")
    frame.to_csv(OUT / name, index=False, float_format="%.15g", lineterminator="\n")
    WRITTEN.append(name)


def build_article_sources():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Source Data output must start empty: {OUT.relative_to(ROOT)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for old, new in COPIES:
        save(read_csv(PANELS / old), new, [PANELS / old], f"panel table {old}")

    # Fig. 1a: design counts of the fifteen reactions (no yields).
    mc_path, pe_path, s2_path = CORE / "molecule_contributions.csv", CORE / "full_data_pair_effects_bba.csv", TABLES / "TableS2.csv"
    mc, pe, s2 = read_csv(mc_path), read_csv(pe_path), read_csv(s2_path)
    assert len(s2) == 15, "Table S2 lists the fifteen analysed reactions"
    k = mc.groupby(["side", "SMILES"]).reaction.nunique()
    shared = set(k[k > 1].index)
    rows = [{"reaction": rx, "electrophile": BF.ROLE[rx][0], "nucleophile": BF.ROLE[rx][1],
             "electrophiles": int(g.side.eq("elec").sum()), "nucleophiles": int(g.side.eq("nuc").sum()),
             "pairs": int(pe.rxn_class.eq(rx).sum()),
             "shared_molecules": int(sum((s, m) in shared for s, m in zip(g.side, g.SMILES)))}
            for rx, g in mc.groupby("reaction")]
    t = pd.DataFrame(rows).sort_values(["pairs", "reaction"], ascending=[False, True])
    assert len(t) == 15 and t.pairs.sum() == int(s2.pairs.sum())
    assert t.electrophiles.sum() == int(s2.electrophiles.sum()) and t.nucleophiles.sum() == int(s2.nucleophiles.sum())
    save(t, "Figure1a_reactions.csv", [mc_path, pe_path, s2_path],
         "design counts per reaction (no yields); shared = molecules that occur on the same side in at least one other analysed reaction")

    from figures_1_4 import condition_composition_data
    save(condition_composition_data(), "ExtendedDataFig1_condition_composition.csv",
         [ROOT / "data/metadata/base_condition_roles.csv"],
         "the twenty analysed conditions of each core array, with reagent and solvent composition")

    # Fig. 2b: SNAr aryl halides, condition halves against the fixed and 20 further partner partitions.
    f2a_path = PANELS / "Figure2a_portability_by_set.csv"
    a1_path, a2_path, a3_path = (PROFILES / n for n in ("A1_first_set.csv", "A2_salts.csv", "A3_second_set.csv"))
    f2a, a1, a2, a3 = (read_csv(p) for p in (f2a_path, a1_path, a2_path, a3_path))
    sn = f2a[(f2a.reaction == "SNAr") & (f2a.side == "elec")]
    assert len(sn) == 1
    sn = sn.iloc[0]
    rel_split = a1[a1.reaction == "SNAr"].iloc[0]
    assert abs(rel_split.rho_molecule - sn.rho_P) < 1e-4 and int(rel_split.matched) == int(sn.molecules)
    salts = a2[(a2.reaction == "SNAr")].dropna(subset=["rho_molecule"])
    assert len(salts) == 20
    rows = [{"agreement": "condition halves", "partition": "", "aryl_halides": int(sn.molecules), "rank_agreement": sn.rho_C},
            {"agreement": "partner halves", "partition": "fixed", "aryl_halides": int(rel_split.matched),
             "rank_agreement": rel_split.rho_molecule}]
    rows += [{"agreement": "partner halves", "partition": r.salt, "aryl_halides": int(r.matched),
              "rank_agreement": r.rho_molecule} for r in salts.itertuples()]
    save(pd.DataFrame(rows), "Figure2b_SNAr_partitions.csv", [f2a_path, a1_path, a2_path],
         "SNAr electrophile side: condition-half and fixed partner-half Spearman from Figure 2a; 20 salted partner partitions from A2")

    # Fig. 3a: the second Buchwald-Hartwig campaign against the nine analysed reactions.
    ed_path = PANELS / "ExtendedDataFig6bc_profile_comparisons.csv"
    ed = read_csv(ed_path)
    q = ed[(ed.comparison == "second Buchwald-Hartwig campaign against an analysed reaction") & (ed.reaction_b == "Buchwald-Hartwig(CMC)")].copy()
    assert len(q) == 9, len(q)
    q = q.rename(columns={"reaction_a": "analysed_reaction", "pearson": "profile_similarity", "spearman": "profile_spearman",
                          "record_rank_agreement": "aryl_halide_rank_agreement", "record_shared": "shared_aryl_halides"})
    q = q[["analysed_reaction", "features", "profile_similarity", "profile_spearman", "shared_aryl_halides", "aryl_halide_rank_agreement"]]
    q["ranking_eligible"] = q.shared_aryl_halides >= 10
    q["profile_rank"] = q.profile_similarity.rank(ascending=False, method="min").astype(int)
    q["ranking_rank_among_eligible"] = q.loc[q.ranking_eligible, "aryl_halide_rank_agreement"].rank(ascending=False, method="min")
    assert q.loc[q.analysed_reaction == "Buchwald-Hartwig", "profile_rank"].item() == 1
    save(q.sort_values("profile_similarity", ascending=False), "Figure3a_campaign_recognition.csv", [ed_path],
         "rows of Extended Data Fig. 7a,b for the second Buchwald-Hartwig campaign; ranking ranks among reactions sharing >= 10 aryl halides")

    # Fig. 3b and 3c: partner halves, molecule ranking against structural profile, and the permutation null.
    both = pd.concat([a1.assign(condition_set="first"), a3.assign(condition_set="second")], ignore_index=True)
    both = both.dropna(subset=["rho_molecule", "r_structure"]).rename(columns={"matched": "aryl_halides"})
    assert both.groupby("condition_set").size().to_dict() == {"first": 7, "second": 7}
    save(both[["reaction", "condition_set", "aryl_halides", "features", "rho_molecule", "pearson_molecule",
               "r_structure", "spearman_structure"]], "Figure3b_partner_change.csv", [a1_path, a3_path],
         "A1 (first condition set) and A3 (second): fixed partner split, aryl halides evaluable in both halves")
    c = both[["reaction", "condition_set", "aryl_halides", "features", "r_structure", "null_median", "null_2.5",
              "null_97.5", "null_p_upper", "pearson_molecule"]].copy()
    c["above_null_97.5"] = c.r_structure > c["null_97.5"]
    save(c, "Figure3c_random_grouping.csv", [a1_path, a3_path],
         "A1/A3: 10,000 permutations of whole 11-feature rows among the aryl halides, one permutation for both halves")

    # Fig. 3d: aryl halides without a record, chosen by each reaction's structural profile (sensitivity step).
    d_path = ROOT / "output/sensitivity/structure_for_selection/Figure3d_selection_without_record.csv"
    assert len(read_csv(d_path)) == 9   # copied byte for byte: re-serializing would change the last digit of some values
    (OUT / d_path.name).write_bytes(d_path.read_bytes())
    WRITTEN.append(d_path.name)
    return WRITTEN
