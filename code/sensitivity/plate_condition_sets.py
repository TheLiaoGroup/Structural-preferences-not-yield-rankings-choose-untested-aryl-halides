"""Audit physical plates across the core and second condition sets without changing recorded labels.

Run: python -B code/sensitivity/plate_condition_sets.py
Writes pair-level and plate-label audits plus the condition-set summary from the released data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/core"))

import pandas as pd
from design_gated_bba import physical_plate_ids

KEYS = ["rxn_class", "electrophile_smiles", "nucleophile_smiles"]
SETS = ["primary", "second_condition_set"]


def plate_audit(package=ROOT):
    package = Path(package)
    source = package / "data/experimental_wells.csv"
    membership = package / "data/metadata/analysis_well_membership.csv"
    wells = pd.read_csv(source, usecols=["well_id", *KEYS, "plate", "measurement_status"],
                        dtype=str, keep_default_na=False)
    members = pd.read_csv(membership, dtype=str, keep_default_na=False)
    wells = wells.merge(members, on="well_id", validate="one_to_one", how="outer", indicator=True)
    if not wells._merge.eq("both").all():
        raise ValueError("analysis-set membership does not cover exactly the released wells")
    wells["physical_plate"] = physical_plate_ids(wells.plate)
    labels = wells[["plate", "physical_plate"]].drop_duplicates().sort_values(["physical_plate", "plate"])
    measured = wells[wells.measurement_status.isin(["quantified", "zero_by_rule"])]
    units = measured.groupby(["analysis_set", *KEYS]).agg(
        recorded_plate_units=("plate", "nunique"), physical_plate_units=("physical_plate", "nunique"))
    grouped = measured[measured.analysis_set.isin(SETS) & measured.physical_plate.ne("")]
    pairs = grouped.groupby(KEYS + ["analysis_set"]).agg(
        physical_plates=("physical_plate", lambda s: frozenset(s)),
        recorded_plates=("plate", lambda s: frozenset(s)), wells=("well_id", "size"))
    paired = pairs.unstack("analysis_set").dropna(subset=[("physical_plates", s) for s in SETS])
    rows = []
    for key, row in paired.iterrows():
        first, second = (row[("physical_plates", s)] for s in SETS)
        union = first | second
        record = dict(zip(KEYS, key))
        record.update({"primary_physical_plates": "; ".join(sorted(first)),
                       "second_physical_plates": "; ".join(sorted(second)),
                       "primary_recorded_plates": "; ".join(sorted(row[("recorded_plates", "primary")])),
                       "second_recorded_plates": "; ".join(sorted(row[("recorded_plates", "second_condition_set")])),
                       "primary_wells": int(row[("wells", "primary")]),
                       "second_wells": int(row[("wells", "second_condition_set")]),
                       "primary_physical_count": len(first), "second_physical_count": len(second),
                       "union_physical_count": len(union), "one_shared_physical_plate": len(union) == 1,
                       "any_shared_physical_plate": bool(first & second), "identical_plate_sets": first == second,
                       "second_has_additional_plate": bool(second - first),
                       "primary_has_additional_plate": bool(first - second)})
        if len(union) == 1:
            category = "both sets on the same single physical plate"
        elif first == second:
            category = "both sets share the same multiple physical plates"
        elif second - first and first - second:
            category = "each set includes a physical plate absent from the other"
        elif second - first:
            category = "second set includes an additional physical plate"
        else:
            category = "primary set includes an additional physical plate"
        record["plate_category"] = category
        rows.append(record)
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("no substrate pairs have a recorded plate in both condition sets")
    summary = {
        "physical_plate_definition": "Complete recorded prefix through -PL followed by its full integer; ignore subsequent assay suffixes such as LC1, LC2 and DE1.",
        "measurement_domain": "Quantified and zero_by_rule wells in the primary and second_condition_set analysis sets.",
        "pairs_with_plate_in_both_sets": len(detail),
        "pairs_sharing_a_plate": int(detail.any_shared_physical_plate.sum()),
        "pairs_on_one_shared_physical_plate": int(detail.one_shared_physical_plate.sum()),
        "pairs_with_multiple_physical_plates": int(detail.union_physical_count.gt(1).sum()),
        "pairs_with_identical_plate_sets": int(detail.identical_plate_sets.sum()),
        "pairs_with_different_plate_sets": int((~detail.identical_plate_sets).sum()),
        "pairs_with_second_set_additional_plate": int(detail.second_has_additional_plate.sum()),
        "pairs_with_primary_set_additional_plate": int(detail.primary_has_additional_plate.sum()),
        "categories": {str(k): int(v) for k, v in detail.plate_category.value_counts().sort_index().items()},
        "plate_units_by_analysis_set": {
            str(key): {"recorded_pair_plate_units": int(g.recorded_plate_units.sum()),
                       "physical_pair_plate_units": int(g.physical_plate_units.sum()),
                       "pairs_with_collapsed_assay_suffixes": int(g.recorded_plate_units.ne(g.physical_plate_units).sum())}
            for key, g in units.groupby("analysis_set")},
        "input_sha256": {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (source, membership)},
    }
    assert sum(summary["categories"].values()) == len(detail)
    assert summary["pairs_on_one_shared_physical_plate"] + summary["pairs_with_multiple_physical_plates"] == len(detail)
    return summary, detail, labels


def catalyst_counts(package=ROOT):
    conditions = pd.read_csv(Path(package) / "data/metadata/base_condition_roles.csv")
    conditions = conditions[conditions["set"].eq("analysed")]
    counts = {}
    for reaction, key in (("Suzuki", "boronic_acid"), ("Suzuki Bpin", "pinacol_boronate")):
        frame = conditions[conditions.rxn_class.eq(reaction)]
        if len(frame) != 20:
            raise ValueError(f"{reaction}: expected twenty first-set conditions")
        counts[f"{key}_conditions_with_PPh3"] = int(frame.reagents.str.contains("PPh3", regex=False).sum())
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.package / "output/sensitivity/plate_condition_sets"
    out.mkdir(parents=True, exist_ok=True)
    summary, detail, labels = plate_audit(args.package)
    summary.update(catalyst_counts(args.package))
    detail.to_csv(out / "plate_pair_membership.csv", index=False, lineterminator="\n")
    labels.to_csv(out / "recorded_to_physical_plate.csv", index=False, lineterminator="\n")
    (out / "plate_condition_sets_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    legacy = args.package / "output/sensitivity/boronic_acids_across_roles/condition_sets.json"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
