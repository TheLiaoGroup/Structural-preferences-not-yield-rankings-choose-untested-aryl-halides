"""Numerical sources for the reaction-specific readings in Figure 3.

The fitted contributions, structural contrasts, feature definitions and exclusions are
those of the core and extended analyses. This step aligns the eight displayed features
between condition sets, summarizes their replication, and ranks the illustrated molecule.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from classification import metal_labels
from structure_profiles import ARYL, MIN_CLASS, MIN_FEATURES

FIRST = "first condition set"
SECOND = "second condition set"
FEATURES = (
    "halogen next to ring N",
    "N\u2013H",
    "five-membered ring at the halogen",
    "substituent beside the halogen",
    "fused ring at the halogen",
    "heteroaromatic ring at the halogen",
    "electron-withdrawing group",
    "electron-donating group",
)
MOLECULE = "COC(=O)c1ccnc(Br)c1"
MOLECULE_NAME = "methyl 2-bromoisonicotinate"
MOLECULE_REACTIONS = ("Sonogashira", "Heck", "Suzuki", "Suzuki Bpin")
MEASURES = ("with_n", "without_n", "with_mean_pp", "without_mean_pp", "difference_pp",
            "mann_whitney_p", "bh_q")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def reaction_order(contrasts, labels, conditions):
    """Group the eligible profiles by metals read from their first twenty recipes.

    A profile requires the existing minimum of five testable features. Groups are
    ordered by decreasing reaction count, then metal name; reactions within each
    group are alphabetical. No reaction-specific plotting order is supplied.
    """
    first = contrasts[(contrasts.data == FIRST) & contrasts.feature.isin(FEATURES)]
    second = contrasts[(contrasts.data == SECOND) & contrasts.feature.isin(FEATURES)]
    first_counts = first.groupby("reaction").feature.nunique()
    second_counts = second.groupby("reaction").feature.nunique()
    aryl = labels.index[labels.electrophile == ARYL]
    included = sorted(reaction for reaction in aryl if first_counts.get(reaction, 0) >= MIN_FEATURES)
    require(bool(included), "No aryl-halide reaction has a testable structural profile")
    recipes = conditions[(conditions.design == "twenty conditions") & (conditions["set"] == "analysed")]
    require(not recipes.duplicated(["rxn_class", "condition_id"]).any(), "Duplicate first-set condition recipe")
    metals = metal_labels(recipes)
    counts = recipes.groupby("rxn_class").size()
    for reaction in included:
        require(counts.get(reaction, 0) == 20, f"Expected twenty first-set recipes for {reaction}")
        require(metals.get(reaction) == labels.loc[reaction, "metal"],
                f"Reagent-derived metal disagrees with core labels for {reaction}")
    order = pd.DataFrame({"reaction": included, "metal": [metals[reaction] for reaction in included]})
    group_sizes = order.groupby("metal").size()
    groups = sorted(group_sizes.index, key=lambda metal: (-int(group_sizes[metal]), metal))
    order["metal_group_order"] = order.metal.map({metal: i + 1 for i, metal in enumerate(groups)})
    order["metal_group_size"] = order.metal.map(group_sizes)
    order = order.sort_values(["metal_group_order", "reaction"], kind="stable").reset_index(drop=True)
    order.insert(0, "row_order", np.arange(1, len(order) + 1))
    order["first_tested_features"] = order.reaction.map(first_counts).astype(int)
    order["second_tested_features"] = order.reaction.map(second_counts).fillna(0).astype(int)
    return order


def feature_tables(contrasts, order):
    """Return the complete first-set matrix, paired cells, and one pooled summary."""
    selected = contrasts[contrasts.reaction.isin(order.reaction) & contrasts.feature.isin(FEATURES) &
                         contrasts.data.isin((FIRST, SECOND))]
    require(not selected.duplicated(["data", "reaction", "feature"]).any(),
            "Duplicate reaction-feature contrast in a condition set")
    require(np.isfinite(selected.difference_pp.to_numpy()).all(), "Nonfinite structural contrast")
    require(((selected.with_n >= MIN_CLASS) & (selected.without_n >= MIN_CLASS)).all(),
            "A structural contrast fails the existing minimum class size")
    identifiers = ["reaction", "metal", "row_order", "metal_group_order"]
    feature_order = pd.DataFrame({"feature": FEATURES, "feature_order": np.arange(1, len(FEATURES) + 1)})
    grid = order[identifiers].merge(feature_order, how="cross")
    first = selected[selected.data == FIRST][["reaction", "feature", *MEASURES]]
    second = selected[selected.data == SECOND][["reaction", "feature", *MEASURES]]
    matrix = grid.merge(first, on=["reaction", "feature"], how="left", validate="one_to_one", indicator=True)
    matrix["tested"] = matrix.pop("_merge").eq("both")
    matrix = matrix[[*identifiers, "feature", "feature_order", "tested", *MEASURES]]
    matrix = matrix.sort_values(["row_order", "feature_order"], kind="stable").reset_index(drop=True)
    paired = first.merge(second, on=["reaction", "feature"], suffixes=("_first", "_second"), validate="one_to_one")
    paired = grid.merge(paired, on=["reaction", "feature"], how="inner", validate="one_to_one")
    paired = paired.rename(columns={f"{measure}_{which}": f"{which}_{measure}"
                                    for measure in MEASURES for which in ("first", "second")})
    paired = paired.sort_values(["row_order", "feature_order"], kind="stable").reset_index(drop=True)
    paired["same_sign"] = np.sign(paired.first_difference_pp) == np.sign(paired.second_difference_pp)
    require(len(paired) >= 2 and paired.first_difference_pp.nunique() > 1 and
            paired.second_difference_pp.nunique() > 1, "Too few varying paired contrasts for Pearson correlation")
    summary = pd.DataFrame([{
        "reactions": len(order), "features": len(FEATURES), "candidate_cells": len(grid),
        "paired_cells": len(paired), "same_sign_cells": int(paired.same_sign.sum()),
        "same_sign_fraction": float(paired.same_sign.mean()),
        "pearson_r": float(pearsonr(paired.first_difference_pp, paired.second_difference_pp).statistic),
    }])
    return matrix, paired, summary


def molecule_readings(contributions, reactions=MOLECULE_REACTIONS):
    """Ordinal rank of the illustrated molecule among each reaction's electrophiles.

    Rank 1 is the largest unrounded contribution. Exact ties are resolved by
    ascending dataset SMILES, giving deterministic ranks and denominators.
    """
    electrophiles = contributions[(contributions.side == "elec") & contributions.reaction.isin(reactions)]
    require(not electrophiles.duplicated(["reaction", "SMILES"]).any(), "Duplicate molecule contribution")
    require(np.isfinite(electrophiles.contribution_pp.to_numpy()).all(), "Nonfinite molecule contribution")
    rows = []
    for reaction in reactions:
        ranked = electrophiles[electrophiles.reaction == reaction].sort_values(
            ["contribution_pp", "SMILES"], ascending=[False, True], kind="stable").reset_index(drop=True)
        match = ranked[ranked.SMILES == MOLECULE]
        require(len(match) == 1, f"Illustrated molecule is absent or duplicated in {reaction}")
        row = match.iloc[0]
        rows.append({"reaction": reaction, "SMILES": MOLECULE, "molecule_name": MOLECULE_NAME,
                     "contribution_pp": float(row.contribution_pp), "rank": int(match.index[0]) + 1,
                     "total_molecules": len(ranked)})
    return pd.DataFrame(rows)


def run(package, core, extended, output):
    package, core, extended, output = (Path(path).resolve() for path in (package, core, extended, output))
    contrasts = pd.read_csv(extended / "structure_contrasts.csv")
    labels = pd.read_csv(core / "reaction_labels.csv", index_col=0)
    conditions = pd.read_csv(package / "data/metadata/base_condition_roles.csv", dtype={"condition_id": str})
    contributions = pd.read_csv(core / "molecule_contributions.csv")
    order = reaction_order(contrasts, labels, conditions)
    matrix, paired, summary = feature_tables(contrasts, order)
    molecule = molecule_readings(contributions)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {
        "aryl_reaction_order.csv": order,
        "aryl_feature_matrix.csv": matrix,
        "paired_feature_contrasts.csv": paired,
        "paired_feature_summary.csv": summary,
        "methyl_2_bromoisonicotinate.csv": molecule,
    }.items():
        table.to_csv(output / name, index=False, lineterminator="\n", float_format="%.17g")
    aryl = labels.index[labels.electrophile == ARYL]
    first_counts = contrasts[(contrasts.data == FIRST) & contrasts.feature.isin(FEATURES)].groupby("reaction").feature.nunique()
    metadata = {
        "features_in_display_order": list(FEATURES),
        "feature_definitions": "code/extended/structure_profiles.py: features; existing reacting-site and SMARTS definitions",
        "contrast_inputs": "output/extended/structure_contrasts.csv",
        "contribution_inputs": "output/core/molecule_contributions.csv; electrophile side; first condition set",
        "reaction_inclusion": f"Aryl-halide-or-sulfonate electrophile label and at least {MIN_FEATURES} testable displayed features on the first condition set",
        "minimum_molecules_per_feature_class": MIN_CLASS,
        "excluded_reactions": [{"reaction": reaction, "first_tested_features": int(first_counts.get(reaction, 0))}
                               for reaction in sorted(set(aryl) - set(order.reaction))],
        "metal_inputs": "data/metadata/base_condition_roles.csv; design=twenty conditions; set=analysed",
        "metal_rule": "Existing classification.metal_labels: a metal in at least 15 of 20 reagent recipes; PEPPSI counts as Pd; otherwise mixed observed metals or none",
        "metal_order": "Descending number of included reactions, then ascending metal label; ascending reaction name within each group",
        "sign_rule": "np.sign(first_difference_pp) == np.sign(second_difference_pp); two exact zeros agree",
        "correlation": "Pearson correlation of all cells testable on both condition sets; no significance filter",
        "missing_cells": "Not testable under the existing class-size rule; missing contrasts are never replaced by zero",
        "molecule_rank_rule": "Descending full-precision contribution_pp, exact ties resolved by ascending dataset SMILES; rank 1 is highest",
        "molecule_rank_denominator": "All fitted electrophiles in that reaction on the first condition set, without an additional structural subset",
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                                             allow_nan=False) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print(molecule[["reaction", "contribution_pp", "rank", "total_molecules"]].to_string(index=False))
    return order, matrix, paired, summary, molecule


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--core-output", type=Path)
    parser.add_argument("--extended-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.package, args.core_output or args.package / "output/core",
        args.extended_output or args.package / "output/extended",
        args.output or args.package / "output/reaction_readings")
