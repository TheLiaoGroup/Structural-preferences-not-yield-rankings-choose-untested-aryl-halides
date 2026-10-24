"""Layouts of the fifteen analysed reactions for the simulations, in sorted order.

Only design columns are read; no recorded yield is loaded. The fifteen primary layouts are built; the
calibration covers gates A and B.
"""
from pathlib import Path
import itertools
import json
import sys
import numpy as np
import pandas as pd

STAGING = Path(__file__).resolve().parents[2]
KEYS = ["rxn_class", "condition_id", "elec", "nuc", "well_id"]


def canonical_wells(frame):
    frame = frame.copy()
    for col in KEYS + ["plate"]:
        frame[col] = frame[col].fillna("").astype(str)
    assert not frame.duplicated(["rxn_class", "condition_id", "elec", "nuc"]).any()
    return frame.sort_values(KEYS, kind="stable").reset_index(drop=True)


def primary_frames(staging=STAGING):
    staging = Path(staging)
    use = ["well_id", "rxn_class", "condition_id", "electrophile_smiles", "nucleophile_smiles", "plate",
           "measurement_status"]
    w = pd.read_csv(staging / "data/experimental_wells.csv", usecols=use, dtype=str, keep_default_na=False)
    membership = pd.read_csv(staging / "data/metadata/analysis_well_membership.csv", dtype=str, keep_default_na=False)
    primary_ids = set(membership.loc[membership.analysis_set.eq("primary"), "well_id"])
    w = w[w.well_id.isin(primary_ids) & w.measurement_status.isin(["quantified", "zero_by_rule"])]
    w = w.drop(columns="measurement_status").rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})
    c = pd.read_csv(staging / "data/metadata/base_condition_roles.csv",
                    usecols=["design", "set", "rxn_class", "coverage_rank", "condition_id"],
                    dtype={"condition_id": str}, keep_default_na=False)
    c = c[c.design.eq("twenty conditions") & c["set"].eq("analysed")]
    c = c.sort_values(["rxn_class", "coverage_rank", "condition_id"], kind="stable").reset_index(drop=True)
    p = pd.read_csv(staging / "data/metadata/base_pair_roles.csv",
                    usecols=["design", "set", "rxn_class", "electrophile_smiles", "nucleophile_smiles"],
                    dtype=str, keep_default_na=False)
    p = p[p.design.eq("twenty conditions") & p["set"].eq("analysed")]
    p = p.rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})
    p = p.sort_values(["rxn_class", "elec", "nuc"], kind="stable").reset_index(drop=True)
    return canonical_wells(w), c, p


def build_layouts(G, staging=STAGING):
    staging = Path(staging)
    w, c, p = primary_frames(staging)
    A = {}
    for rc, pairs in p.groupby("rxn_class", sort=True):
        wells = w[w.rxn_class.eq(rc)].reset_index(drop=True)
        conditions = c[c.rxn_class.eq(rc)].condition_id.tolist()
        A[rc] = G.ReactionLayout(rc, wells, pairs, conditions, wells.plate)
        assert list(A[rc].e_name) == sorted(A[rc].e_name)
        assert list(A[rc].n_name) == sorted(A[rc].n_name)
    C = {}  # gate C is not calibrated
    names = {side: sorted(set().union(*[set(l.e_name if side == "elec" else l.n_name) for l in A.values()]))
             for side in ("elec", "nuc")}
    bmap = {rc: {side: np.searchsorted(names[side], l.e_name if side == "elec" else l.n_name)
                for side in ("elec", "nuc")} for rc, l in A.items()}
    return {"A": A, "C": C, "B names": names, "B map": bmap}


def load_cells(staging=STAGING):
    obj = json.loads((Path(staging) / "data/metadata/calibration_cells.json").read_text(encoding="utf-8"))
    assert obj["confirmation_root"] == 20261103 and obj["replicates_per_cell"] == 500
    cells = obj["cells"]
    assert [c["cell"] for c in cells] == list(range(112))
    assert {g: sum(c["gate"] == g for c in cells) for g in "ABC"} == {"A": 100, "B": 12, "C": 0}
    return cells


def comparison_frame(layouts, E):
    rows = []
    for side in ("elec", "nuc"):
        for a, b in itertools.combinations(sorted(layouts["A"]), 2):
            names_a = layouts["A"][a].e_name if side == "elec" else layouts["A"][a].n_name
            names_b = layouts["A"][b].e_name if side == "elec" else layouts["A"][b].n_name
            if len(set(names_a) & set(names_b)) >= E.MIN_SHARED:
                rows.append({"side": side, "rxn_a": a, "rxn_b": b,
                             "same_reaction": frozenset((a, b)) in E.SAME_REACTION})
    result = pd.DataFrame(rows).sort_values(["side", "rxn_a", "rxn_b"], kind="stable").reset_index(drop=True)
    assert (~result.same_reaction).sum() > 0
    return result
