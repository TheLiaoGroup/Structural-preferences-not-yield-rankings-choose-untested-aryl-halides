"""Build the analysed dataset of the Article from the raw high-throughput tables.

    python build_dataset.py <raw_dir> <package_dir> --frozen-metadata <registration_dir>
    python build_dataset.py <raw_dir> <package_dir> --rederive

Reads every reaction table in <raw_dir> (one table per array) and the condition dictionary
'0_total condition_*.xlsx', and writes into <package_dir>/data:

    experimental_wells.csv                 one row per well of the analysed designs, with its measurement status
    metadata/analysis_well_membership.csv  the analysis set of every well
    metadata/base_condition_roles.csv      the twenty conditions of each design, with their recipes
    metadata/base_pair_roles.csv           the substrate pairs of each design
    metadata/extended_conditions.csv       the conditions of the second Buchwald-Hartwig campaign

and into the folder of this script corpus_accounting.csv (every raw table, its records and its use).

Measurement status of a well:
    quantified               numeric yield
    zero_by_rule             blank yield, screen found no product        -> yield 0
    detected_not_quantified  blank yield, screen found product, no usable chromatographic yield -> missing
    no_screen_record         blank yield, no screening record            -> missing
Missing wells are written with an empty yield and are never used as zeros.

Design of each array (the rule of Supplementary Note 2): among measured wells (quantified or zero by
rule) that passed the platform check, leaving out a substrate pair recorded more than once under the
same condition, the twenty conditions covering the most substrate pairs (ties
broken by condition label); the pairs measured under at least ten of them; the largest connected
set of those pairs. An array enters when that set has at least 50 pairs and ten molecules on each
side. The second condition set applies the same rule to the remaining conditions of each array.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RDLogger.DisableLog("rdApp.*")
N_CONDITIONS, MIN_CONDITIONS, MIN_PAIRS, MIN_MOLECULES = 20, 10, 50, 10
MEASURED = ("quantified", "zero_by_rule")
# Arrays whose first substrate column is the electrophile; in all others it is the nucleophile.
ELECTROPHILE_FIRST = {"Kumada", "Reductive-amination", "Suzuki", "Suzuki Bpin", "Allylic-substitution",
                      "Minisci", "Pd-carbonylation"}
SECOND_CAMPAIGN = "Buchwald-Hartwig(CMC)"   # a second campaign of the Buchwald-Hartwig amination
NOT_A_COUPLING = {"Finkelstein"}            # halide exchange of a single substrate
DICTIONARY_NAME = {
    "Acylation": "acylation", "Allylic-substitution": "allylic-substitution", "Buchwald-Hartwig": "buchwald",
    "Buchwald-Hartwig(CMC)": "B-H (CMC)", "C-O competitive reactions": "C-N or C-O competitive reactions",
    "Chan-Lam": "chan-lam", "Esterification": "esterification", "Finkelstein": "finkelstein",
    "Grignard": "grignard", "Heck": "heck", "Hiyama": "hiyama", "Kumada": "kumada", "Minisci": "minisci",
    "Pd-carbonylation": "pd-carbonylation", "Reductive-amination": "reductive-amination", "SNAr": "snar",
    "Sonogashira": "sonogashira", "Suzuki": "suzuki", "Suzuki Bpin": "suzuki(bpin)", "Ullmann": "ullmann",
}
WELL_COLUMNS = ["well_id", "name0", "component", "rxn_class", "condition_id", "electrophile_smiles", "nucleophile_smiles",
                "product_smiles", "plate", "measurement_status", "yield"]

_TAUTOMER = rdMolStandardize.TautomerEnumerator()
_TAUTOMER.SetRemoveSp3Stereo(False)
_TAUTOMER.SetRemoveBondStereo(False)
_CANON = {}


def canonical(smiles):
    """Molecule identity: RDKit canonical tautomer with stereochemistry retained."""
    smiles = smiles.strip()
    if smiles not in _CANON:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"unparseable SMILES: {smiles}")
        _CANON[smiles] = Chem.MolToSmiles(_TAUTOMER.Canonicalize(mol))
    return _CANON[smiles]


def array_name(path):
    return re.sub(r"(_\d+)?_?\.(csv|xlsx)$", "", Path(path).name)


def exclusion_path(raw_dir):
    root = Path(__file__).resolve().parents[2]
    choices = [Path(raw_dir).parent / "metadata/EXCLUDE_wells.csv",
               root / "data/metadata/EXCLUDE_wells.csv",
               root / "03_GITHUB_ZENODO/data/metadata/EXCLUDE_wells.csv",
               root.parent / "00_FOUNDATION_REPAIR_261018/out/EXCLUDE_wells.csv"]
    for path in choices:
        if path.is_file():
            return path
    raise FileNotFoundError("Supply the authoritative EXCLUDE_wells.csv with --exclusions")


def exclude_wells(raw, csv_path, raw_dir):
    """Exclude reconciled-record conflicts before measurement status or design selection."""
    csv_path = Path(csv_path)
    excluded = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    ids = excluded.name0.str.strip()
    if ids.empty or ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Exclusion name0 values must be nonempty and unique")
    if raw.name0.duplicated().any():
        raise ValueError("Raw name0 identifiers are not unique")
    selected = raw.name0.str.strip().isin(set(ids))
    removed = raw.loc[selected]
    checksum = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if not selected.any():
        audit_path = Path(raw_dir) / "corpus_exclusion_audit.json"
        if not audit_path.is_file():
            raise ValueError("Exclusion list matched zero wells in unaudited input")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["exclusion_sha256"] != checksum:
            raise ValueError("Sanitized corpus exclusion authority differs")
        prior = set(audit["excluded_name0"])
        if prior != set(ids) or audit["excluded_records"] != len(prior):
            raise ValueError("Sanitized corpus does not account for the exclusion list")
        if len(raw) != audit["retained_records"]:
            raise ValueError("Sanitized corpus row count differs from its audit")
        for row in audit["per_array"]:
            if int((raw.rxn_class == row["array"]).sum()) != row["retained_records"]:
                raise ValueError(f"Sanitized array count differs: {row['array']}")
    else:
        if set(removed.name0) != set(ids):
            raise ValueError("Full source corpus does not contain every exclusion ID")
        per_array = [{"array": name, "original_records": len(group),
                      "retained_records": int((~selected.loc[group.index]).sum()),
                      "excluded_records": int(selected.loc[group.index].sum())}
                     for name, group in raw.groupby("rxn_class", sort=True)]
        audit = {"schema": 1, "exclusion_sha256": checksum,
                 "original_records": len(raw), "retained_records": int((~selected).sum()),
                 "excluded_records": int(selected.sum()), "excluded_name0": sorted(removed.name0),
                 "per_array": per_array}
    clean = raw.loc[~selected].copy()
    if clean.name0.isin(set(ids)).any():
        raise ValueError("An excluded well survived filtering")
    if audit["original_records"] - len(clean) != audit["excluded_records"]:
        raise ValueError("Exclusion counts do not reconcile")
    clean.attrs["exclusion_audit"] = audit
    return clean


def read_raw(raw_dir, exclusions=None):
    frames = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*"))):
        if Path(path).name.startswith("0_") or Path(path).suffix.lower() not in (".csv", ".xlsx"):
            continue
        if path.endswith(".csv"):
            t = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        else:   # the records are on the first sheet
            t = pd.read_excel(path, sheet_name=0, dtype=str, keep_default_na=False)
        t["rxn_class"] = array_name(path)
        if "source_row" in t:
            source_row = pd.to_numeric(t.source_row, errors="raise")
            if source_row.duplicated().any() or (source_row <= 0).any() or (source_row % 1 != 0).any():
                raise ValueError(f"Invalid source_row identifiers: {path}")
            t["source_row"] = source_row.astype(int)
        else:
            t["source_row"] = np.arange(1, len(t) + 1)
        frames.append(t)
    raw = pd.concat(frames, ignore_index=True)
    raw[raw.columns.difference(["source_row"])] = raw[raw.columns.difference(["source_row"])].fillna("")
    raw = exclude_wells(raw, exclusions or exclusion_path(raw_dir), raw_dir)
    number = pd.to_numeric(raw["yield"].str.strip(), errors="coerce")
    blank = raw["yield"].str.strip() == ""
    if (number.isna() & ~blank).any():
        raise ValueError("non-numeric yield text")
    clf = pd.to_numeric(raw["yield_clf"].str.strip(), errors="coerce")
    raw["measurement_status"] = np.select(
        [~blank, clf == 0, clf == 1], ["quantified", "zero_by_rule", "detected_not_quantified"], "no_screen_record")
    raw["yield"] = np.where(raw.measurement_status == "zero_by_rule", "0", raw["yield"].str.strip())
    raw["yield"] = np.where(raw.measurement_status.isin(MEASURED), raw["yield"], "")
    raw["checked"] = raw.check_passed.str.strip().str.lower() == "true"
    first = raw.rxn_class.isin(ELECTROPHILE_FIRST)
    s1 = raw.sub_1_smiles.map(canonical)
    s2 = raw.sub_2_smiles.map(lambda s: canonical(s) if s.strip() else "")
    raw["electrophile_smiles"] = np.where(first, s1, s2)
    raw["nucleophile_smiles"] = np.where(first, s2, s1)
    raw["plate"] = raw.name0.str.strip().str.replace(r"-[A-Z]+\d+$", "", regex=True)
    # A substrate pair recorded more than once under the same condition of an array is left out.
    raw["repeated"] = raw.duplicated(["rxn_class", "condition_id", "electrophile_smiles", "nucleophile_smiles"], keep=False)
    return raw


def frozen_design(rows, metadata, rxn_class, set_name):
    conditions = pd.read_csv(Path(metadata) / "base_condition_roles.csv", dtype={"condition_id": str}, keep_default_na=False)
    pairs = pd.read_csv(Path(metadata) / "base_pair_roles.csv", keep_default_na=False)
    take = lambda frame: frame[(frame.design == "twenty conditions") &
                              (frame["set"] == set_name) & (frame.rxn_class == rxn_class)]
    keep = take(conditions).sort_values("coverage_rank", kind="stable")
    registered = take(pairs).rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})
    counts = (rows[rows.condition_id.isin(keep.condition_id)]
              .groupby(["electrophile_smiles", "nucleophile_smiles"]).condition_id.nunique()
              .rename("n_conditions").reset_index()
              .rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"}))
    # A fully excluded pair has no response to fit; all other registered members remain fixed.
    p = registered.drop(columns=["n_conditions"]).merge(counts, on=["elec", "nuc"], validate="one_to_one")
    return keep[["condition_id"]], p


def largest_component(pairs):
    a = pd.factorize(pairs.elec)[0]
    b = pd.factorize(pairs.nuc)[0]
    n = a.max() + 1
    graph = coo_matrix((np.ones(len(a)), (a, b + n)), shape=(n + b.max() + 1,) * 2)
    labels = connected_components(graph, directed=False)[1][a]
    return pairs[labels == np.bincount(labels).argmax()]


def design(rows):
    """Twenty best-covered conditions, pairs measured under at least ten, largest connected set."""
    rows = rows.rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})
    cover = (rows.drop_duplicates(["condition_id", "elec", "nuc"]).groupby("condition_id").size()
             .rename("n_pairs").reset_index().sort_values(["n_pairs", "condition_id"], ascending=[False, True]))
    keep = cover.head(N_CONDITIONS).reset_index(drop=True)
    counted = (rows[rows.condition_id.isin(keep.condition_id)].groupby(["elec", "nuc"]).condition_id.nunique()
               .rename("n_conditions").reset_index())
    counted = counted[counted.n_conditions >= MIN_CONDITIONS]
    pairs = largest_component(counted) if len(counted) >= 3 else counted.iloc[0:0]
    return keep, pairs


def enters(pairs):
    return len(pairs) >= MIN_PAIRS and pairs.elec.nunique() >= MIN_MOLECULES and pairs.nuc.nunique() >= MIN_MOLECULES


def recipes(dictionary, rxn_class, condition_ids):
    d = dictionary[dictionary.reaction_type == DICTIONARY_NAME[rxn_class]].set_index("condition_id")
    missing = set(condition_ids) - set(d.index)
    if missing:
        raise ValueError(f"{rxn_class}: conditions missing from the dictionary: {sorted(missing)}")
    d = d.loc[list(condition_ids)]
    names = d[[f"Reagent{i}_Standardized Name" for i in range(1, 7)]].apply(
        lambda r: "; ".join(x.strip() for x in r if x.strip()), axis=1)
    return pd.DataFrame({"condition_id": list(condition_ids), "reagents": names.values,
                         "solvent": d["Standardized Name"].str.strip().values,
                         "temperature_C": d.temp_id.astype(float).values, "time_h": d.time_id.astype(float).values})


def main(raw_dir, package, exclusions=None, frozen_metadata=None):
    raw_dir, package = Path(raw_dir), Path(package)
    raw = all_raw = read_raw(raw_dir, exclusions)
    exclusion_audit = all_raw.attrs["exclusion_audit"]
    excluded_counts = {r["array"]: r for r in exclusion_audit["per_array"]}
    dictionary = pd.read_excel(next(raw_dir.glob("0_total condition_*.xlsx")), dtype=str, keep_default_na=False)
    dictionary.columns = [c.strip() for c in dictionary.columns]
    dictionary = dictionary.apply(lambda c: c.str.replace("‘", "", regex=False).str.strip())
    raw = raw[~raw.repeated]
    usable = raw.checked & raw.measurement_status.isin(MEASURED)

    wells, conditions, pairs, accounting = [], [], [], []
    for rxn_class in sorted(raw.rxn_class.unique()):
        r = raw[raw.rxn_class == rxn_class]
        row = {"array": rxn_class, "records": excluded_counts[rxn_class]["original_records"],
               "excluded_version_conflict": excluded_counts[rxn_class]["excluded_records"],
               "retained_records": int((all_raw.rxn_class == rxn_class).sum()),
               "repeated_pair_condition": int((all_raw.repeated & (all_raw.rxn_class == rxn_class)).sum()), **{s: int((r.measurement_status == s).sum()) for s in
               ("quantified", "zero_by_rule", "detected_not_quantified", "no_screen_record")},
               "failed_platform_check": int((~r.checked).sum())}
        if rxn_class in NOT_A_COUPLING:
            accounting.append({**row, "use": "not used: halide exchange of a single substrate", "wells_used": 0})
            continue
        if rxn_class == SECOND_CAMPAIGN:
            w = r[r.checked].copy()
            w["analysis_set"], w["component"] = "additional_campaign", "additional_campaign"
            wells.append(w)
            accounting.append({**row, "use": "second Buchwald-Hartwig campaign: every checked well",
                               "wells_used": int(w.measurement_status.isin(MEASURED).sum())})
            continue
        measured_rows = r[usable.loc[r.index]]
        keep1, p1 = (frozen_design(measured_rows, frozen_metadata, rxn_class, "analysed")
                     if frozen_metadata else design(measured_rows))
        if not len(p1) or (not frozen_metadata and not enters(p1)):
            accounting.append({**row, "use": f"not used: design has {len(p1)} pairs, "
                               f"{p1.elec.nunique()} electrophiles, {p1.nuc.nunique()} nucleophiles", "wells_used": 0})
            continue
        rest = r[usable.loc[r.index] & ~r.condition_id.isin(keep1.condition_id)]
        keep2, p2 = (frozen_design(measured_rows, frozen_metadata, rxn_class, "held out")
                     if frozen_metadata else design(rest))
        if set(keep1.condition_id) & set(keep2.condition_id):
            raise ValueError(f"Condition sets overlap: {rxn_class}")
        used = 0
        for label, set_name, keep, p in (("primary", "analysed", keep1, p1), ("second_condition_set", "held out", keep2, p2)):
            w = r[r.checked & r.condition_id.isin(keep.condition_id)].merge(
                p[["elec", "nuc"]].rename(columns={"elec": "electrophile_smiles", "nuc": "nucleophile_smiles"}),
                on=["electrophile_smiles", "nucleophile_smiles"])
            w["analysis_set"], w["component"] = label, "existing_release"
            wells.append(w)
            used += int(w.measurement_status.isin(MEASURED).sum())
            c = recipes(dictionary, rxn_class, keep.condition_id)
            c.insert(0, "coverage_rank", np.arange(1, len(c) + 1))
            c.insert(0, "rxn_class", rxn_class)
            c.insert(0, "set", set_name)
            c.insert(0, "design", "twenty conditions")
            conditions.append(c)
            pairs.append(p.rename(columns={"elec": "electrophile_smiles", "nuc": "nucleophile_smiles"})
                         .assign(design="twenty conditions", set=set_name, rxn_class=rxn_class)
                         [["design", "set", "rxn_class", "electrophile_smiles", "nucleophile_smiles", "n_conditions"]])
        accounting.append({**row, "use": "core design and second condition set", "wells_used": used})

    W = pd.concat(wells, ignore_index=True)
    key = ["rxn_class", "condition_id", "electrophile_smiles", "nucleophile_smiles"]
    measured = W[W.measurement_status.isin(MEASURED)]
    if measured.duplicated(key).any():
        raise ValueError(f"{int(measured.duplicated(key).sum())} repeated measured wells in the analysed designs")
    W["well_id"] = [f"{a}:{n:06d}" for a, n in zip(W.rxn_class, W.source_row)]
    W = W.sort_values(["component", "rxn_class", "source_row"], kind="stable").reset_index(drop=True)
    data = package / "data"
    (data / "metadata").mkdir(parents=True, exist_ok=True)
    (data / "metadata/EXCLUDE_wells.csv").write_bytes(Path(exclusions or exclusion_path(raw_dir)).read_bytes())
    (data / "metadata/exclusion_processing.json").write_text(
        json.dumps(exclusion_audit, indent=2) + "\n", encoding="utf-8")
    W[WELL_COLUMNS].to_csv(data / "experimental_wells.csv", index=False, lineterminator="\n")
    W[["well_id", "analysis_set"]].to_csv(data / "metadata/analysis_well_membership.csv", index=False, lineterminator="\n")
    pd.concat(conditions).to_csv(data / "metadata/base_condition_roles.csv", index=False, lineterminator="\n")
    pd.concat(pairs).to_csv(data / "metadata/base_pair_roles.csv", index=False, lineterminator="\n")
    cmc = W[W.component == "additional_campaign"].condition_id.unique()
    ext = dictionary[(dictionary.reaction_type == DICTIONARY_NAME[SECOND_CAMPAIGN]) & dictionary.condition_id.isin(cmc)]
    if len(ext) != len(cmc):
        raise ValueError("second-campaign conditions missing from the dictionary")
    ext = ext.drop(columns=["note"])
    ext.insert(0, "rxn_class", SECOND_CAMPAIGN)
    ext = ext[["rxn_class", "condition_id"] + [c for c in ext.columns if c not in ("rxn_class", "condition_id")]]
    def natural(labels):
        parts = labels.str.extract(r"([A-Z]+)(\d+)")
        return parts[0] + parts[1].str.zfill(4)
    ext.sort_values("condition_id", key=natural, kind="stable").to_csv(
        data / "metadata/extended_conditions.csv", index=False, lineterminator="\n")
    A = pd.DataFrame(accounting)
    A.to_csv(data / "metadata/corpus_accounting.csv", index=False, lineterminator="\n")
    print(A.to_string(index=False))
    print(W.groupby(["analysis_set", "measurement_status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("--exclusions", type=Path)
    membership = parser.add_mutually_exclusive_group(required=True)
    membership.add_argument("--frozen-metadata", type=Path,
                            help="Retain registered condition/pair memberships; remove only fully excluded pairs")
    membership.add_argument("--rederive", action="store_true",
                            help="Explicitly rederive the design for the separate eligibility comparison")
    args = parser.parse_args()
    main(args.raw_dir, args.package, args.exclusions, args.frozen_metadata)
