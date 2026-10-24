"""Input adapter: the wells, their analysis sets and the conditions and pairs of each design."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import pandas as pd

STAGING = Path(__file__).resolve().parents[2]
RESULTS = (STAGING / "output" / "core").resolve()
ROOT = STAGING
DESIGNS = {
    "primary": ("twenty conditions", "analysed", "twenty_conditions", "01_primary_analysis", "extended"),
    "validation": ("twenty conditions", "held out", "twenty_conditions", "06_validation_conditions", "holdout"),
}
RENAMES = {"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"}

def released():
    wells = pd.read_csv(STAGING / "data" / "experimental_wells.csv", keep_default_na=False, dtype={"condition_id": str}, low_memory=False, float_precision="round_trip")
    # Measured wells only: a well with a product signal but no usable yield, or without a screening
    # record, is missing and is never read as zero.
    wells = wells[(wells.component == "existing_release") & wells.measurement_status.isin(["quantified", "zero_by_rule"])].copy()
    wells["source_well_id"] = wells.well_id
    wells["yield"] = pd.to_numeric(wells["yield"], errors="raise")
    member = pd.read_csv(STAGING / "data" / "metadata" / "analysis_well_membership.csv")
    for design, role, column, _, _ in DESIGNS.values():
        if column not in wells:
            wells[column] = ""
    membership_labels = {"primary": "primary", "validation": "second_condition_set"}
    for label, (_, role, column, _, _) in DESIGNS.items():
        ids = set(member.loc[member.analysis_set == membership_labels[label], "well_id"])
        wells.loc[wells.well_id.isin(ids), column] = role
    conditions = pd.read_csv(STAGING / "data" / "metadata" / "base_condition_roles.csv", keep_default_na=False, dtype={"condition_id": str})
    pairs = pd.read_csv(STAGING / "data" / "metadata" / "base_pair_roles.csv", keep_default_na=False)
    return wells, conditions, pairs

def design_inputs(wells, conditions, pairs, label):
    design, role, column, _, _ = DESIGNS[label]
    w = wells[wells[column] == role].copy()
    c = conditions[(conditions.design == design) & (conditions["set"] == role)].copy()
    c = c.sort_values(["rxn_class", "coverage_rank"], kind="stable")
    p = pairs[(pairs.design == design) & (pairs["set"] == role)].copy()
    return w.rename(columns=RENAMES), c, p.rename(columns=RENAMES)

def write_csv(frame, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, lineterminator="\n", float_format="%.17g")
    return path

def write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def load_modules(output=None):
    global RESULTS
    if output is not None:
        RESULTS = Path(output).resolve()
    RESULTS.mkdir(parents=True, exist_ok=True)
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    L = importlib.import_module("lib_bba")
    L.OUT = RESULTS
    L.LOGS = RESULTS
    E = importlib.import_module("design_extended_bba")
    G = importlib.import_module("gated_bba")
    return L, E, G

legacy = load_modules

def layouts_from(wells, conditions, pairs, G):
    layouts, ys, rows = {}, {}, {}
    for reaction, p in pairs.groupby("rxn_class", sort=True):
        c = conditions[conditions.rxn_class == reaction].sort_values("coverage_rank")
        w = wells[wells.rxn_class == reaction].reset_index(drop=True)
        layouts[reaction] = G.ReactionLayout(reaction, w, p, c.condition_id.tolist(), w.plate)
        ys[reaction] = w["yield"].to_numpy(float)
        rows[reaction] = w
    return layouts, ys, rows

def molecule_count(frame):
    for a, b in (("elec", "nuc"), ("electrophile_smiles", "nucleophile_smiles")):
        if a in frame and b in frame:
            return len(set(frame[a]) | set(frame[b]))
    return frame.molecule.nunique() if "molecule" in frame else None

def pair_id(reaction, elec, nuc):
    return "pair-" + hashlib.sha256(f"{reaction}|{elec}|{nuc}".encode()).hexdigest()[:24]
