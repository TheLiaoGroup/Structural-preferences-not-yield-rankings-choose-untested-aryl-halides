"""Panel tables of the figures, computed from the analysis outputs."""
import numpy as np
import pandas as pd
from definitions import ROLE, STRUCTURE_FEATURES, STRUCTURE_SETS
from presentation_io import SOURCES, read_upstream

WRITTEN = []   # names of the panel tables written
def save_panel(frame, name, inputs=(), derivation="", units="dimensionless"):
    if len(frame) == 0:
        raise RuntimeError(f"Empty panel table: {name}")
    target = SOURCES / name
    frame.to_csv(target, index=False, float_format="%.15g")
    WRITTEN.append(name)

def bool_series(series):
    return series.astype(str).str.lower().eq("true")

def pooled_table(degree):
    frame, path = read_upstream(f"gated_B_pooled_k{degree}_bba.csv")
    values = frame.iloc[0].to_dict() if "D_G" in frame.columns else dict(zip(frame.iloc[:, 0], frame.iloc[:, 1]))
    return values, path

def build_source_data():
    """All scientific display transformations are performed and saved here."""
    SOURCES.mkdir(parents=True, exist_ok=True)
    WRITTEN.clear()

    # Figure 1a: the amide-coupling matrix, ordered by the fitted contributions.
    pair_effects, pe = read_upstream("full_data_pair_effects_bba.csv")
    effects, ep = read_upstream("full_data_effects_bba.csv")
    effects = effects.loc[effects.scale.eq("raw") & effects.label.eq("primary")]
    amide = pair_effects.loc[pair_effects.rxn_class.eq("Acylation")].copy()
    acid = effects.loc[effects.rxn_class.eq("Acylation") & effects.side.eq("elec")].set_index("mol").main_effect
    amine = effects.loc[effects.rxn_class.eq("Acylation") & effects.side.eq("nuc")].set_index("mol").main_effect
    amide["acid_contribution_pp"] = 100 * amide.elec.map(acid)
    amide["amine_contribution_pp"] = 100 * amide.nuc.map(amine)
    amide["pair_effect_pp"] = 100 * amide.p
    amide["acid_rank"] = amide.acid_contribution_pp.rank(method="dense", ascending=False).astype(int)
    amide["amine_rank"] = amide.amine_contribution_pp.rank(method="dense", ascending=False).astype(int)
    if amide[["acid_contribution_pp", "amine_contribution_pp"]].isna().any().any():
        raise RuntimeError("Amide pair without a fitted contribution")
    save_panel(amide.drop(columns="p"), "Figure1a_amide_matrix.csv", [pe, ep],
               "Condition-adjusted pair effects of all measured amide pairs, rows and columns ordered by the fitted acid and amine contributions.",
               "yield percentage points")
    save_panel(pd.DataFrame([
        {"change": "Conditions", "kept": "same molecule, same partners", "question": "Is the record repeatable?"},
        {"change": "Partners", "kept": "same molecule, same reaction", "question": "Does it carry to new partners?"},
        {"change": "Reaction", "kept": "same molecule", "question": "Does it carry to another reaction?"}]),
        "Figure1b_three_changes.csv", derivation="Conceptual layout of the three comparisons; not measurements.", units="not applicable")

    # Figure 2a: condition-half and partner-half agreement for every estimable cell.
    cells, cp = read_upstream("gated_A_cells_bba.csv")
    cells = cells.loc[cells.degree.eq(2)].copy()
    cells["estimable"] = cells.rho_P.notna()
    cells["interval_above_zero"] = cells.low.gt(0) & cells.estimable
    cells["role"] = [ROLE[r][0 if s == "elec" else 1] for r, s in zip(cells.reaction, cells.side)]
    transfer_all, tpa = read_upstream("extended_transfer_bba.csv")
    tr = transfer_all.loc[~bool_series(transfer_all.same_reaction) & transfer_all.n_shared.ge(5)]
    long = pd.concat([tr.rename(columns={"rxn_a": "reaction"})[["reaction", "side", "rho_obs", "n_shared"]],
                      tr.rename(columns={"rxn_b": "reaction"})[["reaction", "side", "rho_obs", "n_shared"]]], ignore_index=True)
    between = pd.DataFrame([{"reaction": rc, "side": sd, "between_mean": float(np.average(g.rho_obs, weights=g.n_shared)),
                             "between_comparisons": len(g), "between_shared_molecules": int(g.n_shared.sum())}
                            for (rc, sd), g in long.groupby(["reaction", "side"])])
    cells = cells.merge(between, on=["reaction", "side"], how="left", validate="one_to_one")
    save_panel(cells, "Figure2a_portability_by_set.csv", [cp, tpa],
               "Degree-two cells; condition-half (rho_C) and partner-half (rho_P) rank correlations on the same molecules, and the shared-molecule-weighted mean full-data rank correlation of the set with each other reaction sharing at least five molecules.")
    pred, pp = read_upstream("gated_A_prediction_bba.csv")
    pred = pred.loc[pred.degree.eq(2)].copy()
    increments = []
    for _, row in pred.iterrows():
        for side, key in (("elec", "I_EN"), ("nuc", "I_NE")):
            increments.append({"reaction": row.reaction, "side": side, "role": ROLE[row.reaction][0 if side == "elec" else 1],
                               "scored_pairs": row.scored, "increment": row[key], "low": row[key + " low"],
                               "high": row[key + " high"], "interval_above_zero": row[key + " low"] > 0})
    increments = pd.DataFrame(increments)
    pairs, pq = read_upstream("amide_heldout_pair_predictions.csv", "pair_holdout_predictions_bba.csv")
    degree_filter = pairs.degree.eq(2) if "degree" in pairs else bool_series(pairs.eligible_k2)
    pairs = pairs.loc[pairs.reaction.eq("Acylation") & degree_filter].copy()
    pairs["degree"] = 2
    pairs["target_pp"] = 100 * pairs.target
    pairs["prediction_pp"] = 100 * pairs.predicted_MEN
    save_panel(pairs, "Figure4a_heldout_amide_pairs.csv", [pq], "Amide scored pairs, target and prediction multiplied by 100.", "yield percentage points")
    metrics = pd.DataFrame([{"scored_pairs": len(pairs), "prediction_rmse_pp": float(np.sqrt(np.mean((pairs.target - pairs.predicted_MEN) ** 2)) * 100),
                             "reaction_mean_rmse_pp": float(np.sqrt(np.mean((pairs.target - pairs.baseline_M0) ** 2)) * 100)}])
    save_panel(metrics, "Figure4a_metrics.csv", [pq], "RMSE from saved pair-level targets and predictions.", "yield percentage points")

    # Figure 3a: cross-reaction agreement against each reaction's own partner reproducibility.
    comparisons, gp = read_upstream("gated_B_relative_k2_bba.csv")
    defined = comparisons.rho_P_a.gt(0) & comparisons.rho_P_b.gt(0)
    comparisons["geometric_benchmark_defined"] = defined
    comparisons["geometric_benchmark"] = np.sqrt((comparisons.rho_P_a * comparisons.rho_P_b).where(defined))
    comparisons["D_G_comparison"] = comparisons.geometric_benchmark - comparisons.rho_R_half
    comparisons["minimum_partners"] = 2
    save_panel(comparisons.loc[defined], "ExtendedDataFig2_defined_benchmark.csv", [gp], "Comparisons whose two within-reaction partner correlations are both positive.")
    pooled, pooled_inputs, omissions = [], [], []
    for degree in (2, 3):
        values, mp = pooled_table(degree)
        pooled_inputs.append(mp)
        pooled.append({"minimum_partners": degree, "D_G": float(values["D_G"]), "low": float(values["D_G low"]),
                       "high": float(values["D_G high"]), "positive_benchmark_comparisons": int(float(values["used for D_G"])),
                       "eligible_comparisons": int(float(values["comparisons"])), "interval_method": "95% delete-one-reaction jackknife"})
        relative, rp = read_upstream(f"gated_B_relative_k{degree}_bba.csv")
        pooled_inputs.append(rp)
        for reaction in sorted(set(relative.rxn_a) | set(relative.rxn_b)):
            subset = relative.loc[relative.rxn_a.ne(reaction) & relative.rxn_b.ne(reaction)]
            subset = subset.loc[subset.rho_P_a.gt(0) & subset.rho_P_b.gt(0)]
            if len(subset):
                gaps = np.sqrt(subset.rho_P_a * subset.rho_P_b) - subset.rho_R_half
                omissions.append({"minimum_partners": degree, "reaction_left_out": reaction,
                                  "D_G": np.average(gaps, weights=subset.molecules), "remaining_positive_comparisons": len(subset)})
    save_panel(pd.DataFrame(pooled), "ExtendedDataFig2_pooled_shortfall.csv", pooled_inputs, "Pooled shortfall copied from the analysis output.")
    selection, selp = read_upstream("selection_vs_agreement.csv")
    save_panel(selection, "Figure4b_selection_vs_agreement.csv", [selp],
               "Gain of the source reaction's top ten over the shared-molecule mean in the receiving reaction, against the full-data rank agreement of the two reactions.",
               "yield percentage points; rank correlation dimensionless")

    # Figure 4: two cross-reaction comparisons molecule by molecule.
    transfer, tp = read_upstream("extended_transfer_bba.csv")
    transfer = transfer.loc[~bool_series(transfer.same_reaction) & transfer.n_shared.ge(5)].copy()

    def paired(a, b, side):
        cols = ["mol", "main_effect", "n_pairs", "source_well_ids"]
        names = lambda k: {"main_effect": f"{k}_contribution", "n_pairs": f"{k}_n_pairs", "source_well_ids": f"{k}_source_well_ids"}
        x = effects.loc[effects.rxn_class.eq(a) & effects.side.eq(side), cols].rename(columns=names("x"))
        y = effects.loc[effects.rxn_class.eq(b) & effects.side.eq(side), cols].rename(columns=names("y"))
        m = x.merge(y, on="mol", validate="one_to_one")
        m["x_contribution_pp"], m["y_contribution_pp"] = 100 * m.x_contribution, 100 * m.y_contribution
        agreement = transfer.loc[transfer.rxn_a.eq(a) & transfer.rxn_b.eq(b) & transfer.side.eq(side)]
        if len(agreement) != 1:
            raise RuntimeError((a, b, side))
        m["rho_obs"] = float(agreement.rho_obs.iloc[0])
        return m
    positive = paired("Heck", "Suzuki", "elec")
    if positive.empty:
        raise RuntimeError("No aryl bromides shared by the Heck and Suzuki-Miyaura reactions")
    # Drawn examples come from molecules measured with at least two partners in both reactions.
    eligible = positive[["x_n_pairs", "y_n_pairs"]].min(axis=1).ge(2)
    total = (positive.x_contribution_pp.rank() + positive.y_contribution_pp.rank())[eligible]
    positive["drawn"] = 0
    for k, i in enumerate(list(total.nlargest(2).index) + list(total.nsmallest(2).index), 1):
        positive.loc[i, "drawn"] = k
    save_panel(positive, "Figure3c_Heck_Suzuki_molecules.csv", [ep, tp],
               "Full-data electrophile contributions of the shared aryl bromides; drawn = among molecules with at least two partners in both reactions, the two highest and two lowest in summed rank.", "yield percentage points")
    negative = paired("Buchwald-Hartwig", "SNAr", "nuc")
    classes, clp = read_upstream("shared_amines_BH_SNAr.csv")
    negative = negative.merge(classes[["mol", "class"]], on="mol", validate="one_to_one")
    negative["primary_arylamine"] = negative["class"].eq("primary aryl amine")
    if negative.empty or not negative.primary_arylamine.any():
        raise RuntimeError("No shared amines, or no primary (hetero)arylamine among them")
    eligible = negative[["x_n_pairs", "y_n_pairs"]].min(axis=1).ge(2)
    gap = (negative.x_contribution_pp - negative.y_contribution_pp)[eligible]
    negative["drawn"] = 0
    picks = list(gap[negative.primary_arylamine[eligible]].nlargest(2).index) + list(gap[~negative.primary_arylamine[eligible]].nsmallest(2).index)
    for k, i in enumerate(picks, 1):
        negative.loc[i, "drawn"] = k
    save_panel(negative, "Figure3d_BH_SNAr_amines.csv", [ep, tp, clp],
               "Full-data amine contributions with SMARTS classes; drawn = among amines with at least two partners in both reactions, the two primary (hetero)arylamines most favoured in Pd amination relative to SNAr and the two other amines most favoured in SNAr.", "yield percentage points")
    save_panel(negative.groupby("class", as_index=False).agg(n=("mol", "size"), mean_BH_pp=("x_contribution_pp", "mean"), mean_SNAr_pp=("y_contribution_pp", "mean")),
               "Figure3d_amine_class_summary.csv", [ep, clp], "Class means of the plotted contributions.", "yield percentage points")
    # Extended Data Fig. 6: structural profiles within and between reactions.
    profiles, prp = read_upstream("structure_profiles.csv")
    save_panel(profiles, "ExtendedDataFig6bc_profile_comparisons.csv", [prp],
               "Pearson and Spearman correlations of two reactions' (or two condition sets') class differences over the features testable in both, with the rank agreement of their shared aryl halides where defined.")
    sc_all, scp_all = read_upstream("structure_contrasts.csv")
    first_bh = sc_all.loc[sc_all.data.eq("first condition set") & sc_all.reaction.eq("Buchwald-Hartwig")].set_index("feature").difference_pp
    second_bh = sc_all.loc[sc_all.data.eq("second Buchwald-Hartwig campaign") & sc_all.reaction.eq("Buchwald-Hartwig(CMC)")].set_index("feature").difference_pp
    camp = pd.concat({"first_campaign_pp": first_bh, "second_campaign_pp": second_bh}, axis=1).dropna().rename_axis("feature").reset_index()
    if len(camp) < 5:
        raise RuntimeError("Too few shared features for the campaign panel")
    save_panel(camp, "ExtendedDataFig6a_campaign_profiles.csv", [scp_all],
               "Class differences of the features testable in both Buchwald-Hartwig campaigns (first condition set of the analysed campaign; all usable wells of the second).", "yield percentage points")
    # Figure 3b: what each reaction rewards in an aryl halide, on two condition sets.
    contrasts, scp = read_upstream("structure_contrasts.csv")
    shown = contrasts.loc[contrasts.feature.isin(list(STRUCTURE_FEATURES)) & contrasts.data.isin(list(STRUCTURE_SETS))]
    if set(shown.feature) != set(STRUCTURE_FEATURES) or set(shown.data) != set(STRUCTURE_SETS):
        raise RuntimeError("Structure panel is missing a feature or a condition set")
    save_panel(shown, "Figure3b_structure_effects.csv", [scp],
               "Mean contribution of the aryl halides carrying the feature minus that of those without it, per reaction and condition set; reactions with at least three molecules in each class.",
               "yield percentage points")


    # Intervals of the core and campaign estimates.
    held, hp = read_upstream("holdout_summary_bba.csv")
    ext, xp = read_upstream("extended_pooled_bba.csv")
    held_ci, ext_ci = held.set_index("quantity"), ext.set_index("quantity")
    c1 = held_ci.loc["C1 within-reaction, analysed vs held-out"]
    c2 = held_ci.loc["C2 cross-reaction, held-out, corrected"]
    t1 = ext_ci.loc["rho_disattenuated"]
    bootstrap, bootstrap_path = read_upstream("PUBLIC_BOOTSTRAP_RESULTS.csv")
    if bootstrap.analysis_id.duplicated().any():
        raise RuntimeError("Duplicate campaign interval identity")
    summary = bootstrap.set_index("analysis_id")
    rel_all, rlp = read_upstream("extended_reliability_bba.csv")
    quart = lambda v: (float(v.median()), float(v.quantile(.25)), float(v.quantile(.75)))
    mC, mP = quart(rel_all.r_half), quart(cells.loc[cells.estimable].rho_P)
    same = summary.loc["campaign.same_reaction"]
    wb = pd.DataFrame([
        {"kind": "conditions", "label": "1st set, halves", "estimate": mC[0], "ci_low": mC[1], "ci_high": mC[2], "interval": "interquartile range"},
        {"kind": "conditions", "label": "1st vs 2nd set", "estimate": c1.estimate, "ci_low": c1.ci_low, "ci_high": c1.ci_high, "interval": "95% CI"},
        {"kind": "partners", "label": "1st set, halves", "estimate": mP[0], "ci_low": mP[1], "ci_high": mP[2], "interval": "interquartile range"},
        {"kind": "reactions", "label": "1st set", "estimate": t1.estimate, "ci_low": t1.ci_low, "ci_high": t1.ci_high, "interval": "95% CI"},
        {"kind": "reactions", "label": "2nd set", "estimate": c2.estimate, "ci_low": c2.ci_low, "ci_high": c2.ci_high, "interval": "95% CI"},
        {"kind": "reactions", "label": "B–H, 2nd campaign", "estimate": same["estimate"], "ci_low": same["ci_low"], "ci_high": same["ci_high"], "interval": "95% CI"}])
    save_panel(wb, "Figure2b_pooled_portability.csv", [rlp, cp, hp, xp, bootstrap_path],
               "Conditions: median (interquartile range) condition-half rank correlation over the 30 sets, and pooled agreement of the analyzed with the second condition set. Partners: median (interquartile range) partner-half rank correlation over the estimable sets. Reactions: reliability-corrected rank correlation over shared molecules pooled over comparisons on each condition set, and between Buchwald-Hartwig amination and its second campaign.")

    # Supporting Information figures.
    save_panel(increments, "ExtendedDataFig7_prediction_all_reactions.csv", [pp], "All fifteen reactions, both sides.")
    save_panel(comparisons.loc[~defined], "ExtendedDataFig3_undefined_benchmark.csv", [gp], "Comparisons with a non-positive within-reaction partner correlation.")
    save_panel(pd.DataFrame(omissions), "ExtendedDataFig4a_leave_one_reaction_out.csv", pooled_inputs, "Shortfall recomputed from per-comparison estimates after omitting each reaction.")
    sensitivity, sp = read_upstream("gated_B_secondary_bba.csv")
    labels = {"design-eligible comparisons (6.1)": "Raw yield", "logit scale (6.6)": "Logit yield",
              "threshold 0.20 scale (6.6)": "Yield ≥20%", "without SNAr and Chan-Lam coupling (6.6)": "Without SNAr and Chan–Lam"}
    sens = sensitivity.loc[sensitivity.analysis.isin(labels)].copy()
    if len(sens) != len(labels):
        raise RuntimeError("Missing outcome-scale sensitivity rows")
    sens["label"] = sens.analysis.map(labels)
    sens = sens.rename(columns={"relative D_G": "D_G", "relative D_G low": "low", "relative D_G high": "high",
                                "relative used for D_G": "positive_benchmark_comparisons", "relative comparisons": "eligible_comparisons"})
    save_panel(sens[["analysis", "label", "D_G", "low", "high", "positive_benchmark_comparisons", "eligible_comparisons"]],
               "ExtendedDataFig4b_outcome_scale.csv", [sp], "Each scale redefines the positive-benchmark subset.")
    save_panel(transfer[["side", "rxn_a", "rxn_b", "n_shared", "rho_obs"]], "ExtendedDataFig5_reaction_map.csv", [tp], "All full-data cross-reaction rank agreements.")
    print(f"Built {len(WRITTEN)} panel tables.")
