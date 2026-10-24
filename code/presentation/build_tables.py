"""Supplementary Tables 1-11 and Extended Data Table 1, from the analysis outputs and the well membership.

No fitting or resampling occurs. The condition coverage of Table S1 and the threshold fraction of Table S5
are counted on the analysis sets; every effect, interval and diagnostic comes from the analysis outputs.
"""
import csv
from pathlib import Path

import pandas as pd

import legacy_table_transforms as legacy
from extended_data_table import table1
from presentation_io import ROOT, SOURCES, TABLES, read_csv, read_upstream


DESIGN_COLUMNS = ["rxn_class", "conditions", "pairs", "electrophiles", "nucleophiles",
                  "median_partners_elec", "median_partners_nuc", "in_extended_set"]


def load(name):
    if name == "holdout_amines_bba.csv":
        # Table S4 uses only the second-set contributions; its class column is
        # dropped immediately. The molecular effects and classes come from the core step.
        effects, _ = read_upstream("holdout_molecule_effects_bba.csv")
        effects = effects.loc[effects.scale.eq("raw") & effects.side.eq("nuc") &
                              effects.rxn_class.isin(("Buchwald-Hartwig", "SNAr"))]
        both = effects.pivot(index="mol", columns="rxn_class", values="main_effect")
        both = both[["Buchwald-Hartwig", "SNAr"]].dropna().sort_index().reset_index()
        classes, _ = read_upstream("shared_amines_BH_SNAr.csv")
        return both.merge(classes[["mol", "class"]], on="mol", how="left", validate="one_to_one")
    frame, _ = read_upstream(name)
    if name == "extended_design_bba.csv":
        return frame[DESIGN_COLUMNS]
    return frame


def selected_rows_and_conditions():
    """Bind original counting inputs to the recorded public membership only."""
    columns = ["well_id", "component", "rxn_class", "condition_id", "electrophile_smiles",
               "nucleophile_smiles", "yield", "measurement_status"]
    wells = read_csv(ROOT / "data/experimental_wells.csv", usecols=columns,
                     keep_default_na=False, dtype={"condition_id": str}, float_precision="round_trip")
    wells = wells.loc[wells.component.eq("existing_release") & wells.measurement_status.isin(["quantified", "zero_by_rule"])].copy()
    wells["yield"] = pd.to_numeric(wells["yield"], errors="raise")
    wells = wells.rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})
    members = read_csv(ROOT / "data/metadata/analysis_well_membership.csv", keep_default_na=False)
    status = read_csv(ROOT / "data/experimental_wells.csv", usecols=["well_id", "measurement_status"], keep_default_na=False)
    members = members[members.well_id.isin(status.well_id[status.measurement_status.isin(["quantified", "zero_by_rule"])])]
    sets = []
    primary = None
    for label, membership, prefix in (
        ("analysed", "primary", "extended"),
        ("held out", "second_condition_set", "holdout"),
    ):
        conds = read_csv(ROOT / f"output/core/{prefix}_conditions_bba.csv",
                         keep_default_na=False, dtype={"condition_id": str})
        conds = conds.sort_values(["rxn_class", "rank"], kind="stable")
        if conds.duplicated(["rxn_class", "condition_id"]).any():
            raise RuntimeError(f"Duplicate condition metadata: {prefix}")
        ids = set(members.loc[members.analysis_set.eq(membership), "well_id"])
        rows = wells.loc[wells.well_id.isin(ids)].copy()
        if set(rows.well_id) != ids:
            raise RuntimeError(f"Public well membership cannot be resolved: {membership}")
        if label == "analysed":
            primary = rows.copy()
        composition = conds[["rxn_class", "condition_id", "reagents", "solvent", "temperature_C", "time_h"]]
        rows = rows.merge(composition, on=["rxn_class", "condition_id"], how="left", validate="many_to_one")
        if rows[["reagents", "solvent", "temperature_C", "time_h"]].isna().any().any():
            raise RuntimeError(f"Selected well lacks condition composition: {prefix}")
        # Public composition already uses reader names. Putting that same string
        # in one reagent field preserves the original join/none display rule.
        rows["reagent1_id"] = rows.reagents.where(rows.reagents.ne("none"), "")
        for index in range(2, 7):
            rows[f"reagent{index}_id"] = ""
        rows["solvent_id"] = rows.solvent
        rows["temp_id"] = rows.temperature_C
        rows["time_id"] = rows.time_h
        sets.append((label, rows, conds))
    return sets, primary


def build_tables():
    TABLES.mkdir(parents=True, exist_ok=True)
    sets, primary = selected_rows_and_conditions()
    tables = legacy.for_readers(legacy.tables(sets, {}, load))
    s5, s6 = legacy.dense_tables({
        "results": load("results_bba.csv"), "rows": primary,
        "spectrum": load("residual_spectrum_bba.csv"),
        "rel": load("extended_reliability_bba.csv"), "design": load("extended_design_bba.csv"),
        "hold_cells": load("holdout_cells_bba.csv"),
    })
    tables.update(TableS5=s5, TableS6=s6)
    direct = {
        "TableS7": "selection_vs_agreement.csv", "TableS8": "amine_class_contrasts.csv",
        "TableS9": "molecule_contributions.csv", "TableS10": "campaign_comparisons.csv",
        "TableS11": "structure_contrasts.csv",
    }
    tables.update({label: load(name) for label, name in direct.items()})
    records = []
    for label in sorted(tables, key=lambda name: int(name.removeprefix("TableS"))):
        table = tables[label]
        path = TABLES / (label + ".csv")
        table.to_csv(path, index=False, lineterminator="\n")
        records.append({"file": path.name, "rows": len(table)})
    ed = table1()
    ed_path = TABLES / "ExtendedDataTable1.md"
    ed_path.write_text(ed + "\n", encoding="utf-8")
    rows = [line.strip().strip("|").split("|") for line in ed.splitlines() if line.startswith("|")]
    ed_csv = TABLES / "ExtendedDataTable1.csv"
    with ed_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([item.strip() for item in rows[0]])
        writer.writerows([item.strip() for item in row] for row in rows[2:])
    return records
