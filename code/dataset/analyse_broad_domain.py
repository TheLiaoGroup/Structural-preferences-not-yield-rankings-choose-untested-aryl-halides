"""Corpus accounting and agreement over all measured conditions and connected pairs.

Run from the project root with ``python -B 05_REPRODUCE/02_DATASET/analyse_broad_domain.py``.
The raw tables are needed; this analysis does not restrict arrays to the Article's designs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import build_dataset as D

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT if (ROOT / "data/experimental_wells.csv").is_file() else ROOT / "03_GITHUB_ZENODO"
sys.path.insert(0, str(PACKAGE / "code/core"))
import gated_bba as G
import design_extended_bba as E

OUT = PACKAGE / "output/broad_domain" if PACKAGE == ROOT else Path(__file__).resolve().parent / "output"
RAW = PACKAGE / "data/raw_corpus" if PACKAGE == ROOT else ROOT / "05_REPRODUCE/00_RAW_DATA"
SIDES = ("elec", "nuc")
MIN_CORRELATION_MOLECULES = G.MIN_MOLECULES


def component_labels(left, right):
    if not len(left):
        return np.array([], dtype=int)
    li, ln = pd.factorize(left)
    ri, rn = pd.factorize(right)
    graph = coo_matrix((np.ones(len(left)), (li, ri + len(ln))),
                       shape=(len(ln) + len(rn),) * 2)
    return connected_components(graph, directed=False)[1][li]


def largest_pair_domain(rows):
    """The core estimator's largest substrate component, without a minimum-pair gate."""
    pairs = rows[list(SIDES)].drop_duplicates()
    if pairs.empty:
        return rows.copy(), 0
    labels = component_labels(pairs.elec, pairs.nuc)
    counts = np.bincount(labels)
    candidates = np.flatnonzero(counts == counts.max())
    chosen = min(candidates, key=lambda c: min(pairs.elec.to_numpy()[labels == c]))
    keep = pairs.loc[labels == chosen]
    return rows.merge(keep, on=list(SIDES), validate="many_to_one"), len(np.unique(labels))


def fit(rows):
    """Stage one per pair-condition block; stage two includes block intercepts.

    This is retained_estimators.contributions_blocks' model. All block sizes are
    allowed here; rank-deficient contributions are explicitly unreported.
    """
    inputs = rows
    domain_rows, substrate_components = largest_pair_domain(rows)
    info = {"input_measured": len(inputs), "input_conditions": inputs.condition_id.nunique(),
            "input_pairs": len(inputs[list(SIDES)].drop_duplicates()),
            "substrate_components": substrate_components,
            "excluded_connectivity_measured": len(inputs) - len(domain_rows),
            "stage_one_measured": len(inputs),
            "domain_measured": len(domain_rows), "domain_conditions": domain_rows.condition_id.nunique(),
            "domain_pairs": len(domain_rows[list(SIDES)].drop_duplicates())}
    info["excluded_connectivity_pairs"] = info["input_pairs"] - info["domain_pairs"]
    info["excluded_connectivity_conditions"] = info["input_conditions"] - info["domain_conditions"]
    if domain_rows.empty:
        return {"effects": {s: pd.Series(dtype=float) for s in SIDES}, "rows": domain_rows,
                "stage_one_rows": domain_rows,
                "info": {**info, "status": "empty_domain", "pair_condition_blocks": 0}}
    rows = rows.copy()
    rows["pair"] = rows.elec + " >> " + rows.nuc
    rows["block"] = component_labels(rows.pair, rows.condition_id)
    pieces, zero_blocks = [], 0
    for block, group in rows.groupby("block", sort=True):
        pi, pn = pd.factorize(group.pair)
        ci, cn = pd.factorize(group.condition_id)
        y = group["yield"].to_numpy(float)
        values, _, stop = G.stage_one(y, pi, ci, len(pn), len(cn))
        exact_zero = not np.any(y)
        if stop not in (1, 2) and not (stop == 0 and exact_zero):
            raise RuntimeError(f"Stage-one numerical failure: {stop}; block {block}")
        zero_blocks += int(stop == 0 and exact_zero)
        pieces.append(pd.DataFrame({"pair": pn, "p": values, "block": block}))
    pairs = pd.concat(pieces, ignore_index=True)
    pairs[list(SIDES)] = pairs.pair.str.split(" >> ", expand=True)
    keep = domain_rows[list(SIDES)].drop_duplicates()
    pairs = pairs.merge(keep, on=list(SIDES), validate="many_to_one")
    domain_rows = rows.merge(keep, on=list(SIDES), validate="many_to_one")
    ei, en = pd.factorize(pairs.elec)
    ni, nn = pd.factorize(pairs.nuc)
    bi, bn = pd.factorize(pairs.block)
    k = len(en) + len(nn) + len(bn) - 2
    result = G.contributions(pairs.elec, pairs.nuc, pairs.p) if len(bn) == 1 else None
    rank = k
    if result is None:
        # Block intercepts remove the independent stage-one origin of every block.
        X = np.zeros((len(pairs), k))
        X[:, 0] = 1
        r = np.arange(len(pairs))
        X[r[ei > 0], ei[ei > 0]] = 1
        X[r[ni > 0], len(en) - 1 + ni[ni > 0]] = 1
        X[r[bi > 0], len(en) + len(nn) - 2 + bi[bi > 0]] = 1
        Q, R, piv = qr(X, mode="economic", pivoting=True)
        tol = np.finfo(float).eps * max(X.shape) * abs(R[0, 0])
        rank = int((np.abs(np.diag(R)) > tol).sum())
        beta = np.zeros(k)
        beta[piv[:rank]] = solve_triangular(R[:rank, :rank], Q[:, :rank].T @ pairs.p.to_numpy())
        a = np.r_[0.0, beta[1:len(en)]]
        b = np.r_[0.0, beta[len(en):len(en) + len(nn) - 1]]
        result = ({"elec": pd.Series(a - a.mean(), index=en), "nuc": pd.Series(b - b.mean(), index=nn)}
                  if rank == k else {s: pd.Series(dtype=float) for s in SIDES})
    if not all(np.isfinite(result[s]).all() for s in SIDES):
        raise RuntimeError("Nonfinite stage-two contributions")
    info.update({"pair_condition_blocks": len(bn), "zero_response_stage_one_blocks": zero_blocks,
                 "stage_two_columns": k, "stage_two_rank": rank,
                 "status": "identified" if rank == k else "unidentified_block_adjusted_contributions"})
    for side, names in (("elec", en), ("nuc", nn)):
        info.update({f"{side}_molecules": len(names),
                     f"{side}_comparable_molecules": len(result[side]),
                     f"{side}_excluded_unidentified_molecules": len(names) - len(result[side])})
    return {"effects": result, "rows": domain_rows, "stage_one_rows": rows, "info": info}


def correlation(a, b, minimum):
    shared = a.index.intersection(b.index)
    status, rho = "ok", np.nan
    if len(shared) < minimum:
        status = "insufficient_shared_molecules"
    else:
        constant_a = len(np.unique(np.round(a.loc[shared], 12))) < 2
        constant_b = len(np.unique(np.round(b.loc[shared], 12))) < 2
        if constant_a or constant_b:
            status = "constant_both" if constant_a and constant_b else "constant_left" if constant_a else "constant_right"
        else:
            rho = G.spearman(a.loc[shared], b.loc[shared])
            if not np.isfinite(rho):
                raise RuntimeError("Nonfinite correlation of nonconstant identified effects")
    return {"n_shared": len(shared), "rho": rho, "status": status}


def condition_halves(rows):
    """Alternate coverage order within each pair-condition block, as in the campaign estimator."""
    half_of = {}
    for _, group in rows.groupby("block", sort=True):
        order = (group.groupby("condition_id").size().rename("coverage").reset_index()
                 .sort_values(["coverage", "condition_id"], ascending=[False, True]).condition_id)
        half_of.update({c: h % 2 for h, c in enumerate(order)})
    return [rows[rows.condition_id.map(half_of) == h] for h in (0, 1)]


def partner_halves(pairs, array, side):
    """The Article's exact SHA-256 partition, before connectivity restriction in fit()."""
    other = "nuc" if side == "elec" else "elec"
    order = pairs.assign(key=[hashlib.sha256(f"{array}|{side}|{m}|{p}".encode()).hexdigest()
                              for m, p in zip(pairs[side], pairs[other])])
    order = order.sort_values([side, "key"], kind="stable")
    order["half"] = order.groupby(side).cumcount() % 2
    return [order.loc[order.half == h, list(SIDES)] for h in (0, 1)]


def transformation(array):
    return "Suzuki-Miyaura" if array in ("Suzuki", "Suzuki Bpin") else (
        "Buchwald-Hartwig" if array == D.SECOND_CAMPAIGN else array)


def accounting(raw):
    audit = raw.attrs["exclusion_audit"]
    per_array = {row["array"]: row for row in audit["per_array"]}
    if len(per_array) != len(audit["per_array"]) or set(per_array) != set(raw.rxn_class):
        raise ValueError("Exclusion audit arrays do not match the retained corpus")
    if (len(raw) != audit["retained_records"] or
            audit["original_records"] != audit["retained_records"] + audit["excluded_records"]):
        raise ValueError("Exclusion audit totals do not reconcile")
    wells = pd.read_csv(PACKAGE / "data/experimental_wells.csv", keep_default_na=False)
    membership = pd.read_csv(PACKAGE / "data/metadata/analysis_well_membership.csv")
    wells = wells.merge(membership, on="well_id", validate="one_to_one")
    selected = set(wells.rxn_class)
    records = []
    for name, group in raw.groupby("rxn_class", sort=True):
        source = per_array[name]
        if (source["retained_records"] != len(group) or
                source["original_records"] != len(group) + source["excluded_records"]):
            raise ValueError(f"Exclusion audit does not reconcile: {name}")
        w = wells[wells.rxn_class == name]
        measured = group.measurement_status.isin(D.MEASURED)
        used = w.measurement_status.isin(D.MEASURED)
        counts = {f"{label}_measured": int((used & (w.analysis_set == label)).sum())
                  for label in ("primary", "second_condition_set", "additional_campaign")}
        other_measured = int((~group.repeated & group.checked & measured).sum()) - sum(counts.values())
        record = {"array": name, "transformation": transformation(name),
                  "raw_records": source["original_records"], "retained_records": len(group),
                  "excluded_version_conflict": source["excluded_records"], "used_array": name in selected,
                  "excluded_array_records": len(group) if name not in selected else 0,
                  "repeated_records": int(group.repeated.sum()) if name in selected else 0,
                  "failed_check_nonrepeated": int((~group.repeated & ~group.checked).sum()) if name in selected else 0,
                  "missing_checked_nonrepeated": int((~group.repeated & group.checked & ~measured).sum()) if name in selected else 0,
                  "measured_outside_design": other_measured if name in selected else 0,
                  **counts, "released_missing": int((~used).sum())}
        parts = (record["excluded_version_conflict"] + record["excluded_array_records"] + record["repeated_records"] +
                 record["failed_check_nonrepeated"] + record["missing_checked_nonrepeated"] +
                 record["measured_outside_design"] + sum(counts.values()))
        if parts != record["raw_records"]:
            raise ValueError(f"Corpus accounting does not reconcile: {name}, {parts} != {record['raw_records']}")
        records.append(record)
    table = pd.DataFrame(records)
    if (int(table.raw_records.sum()) != audit["original_records"] or
            int(table.excluded_version_conflict.sum()) != audit["excluded_records"]):
        raise ValueError("Per-array exclusion audit does not match the corpus totals")
    expected = (20, 410983, 19, 16, 14)
    actual = (len(table), int(table.raw_records.sum()), int((table.array != "Finkelstein").sum()),
              int(table.used_array.sum()), table.loc[table.used_array, "transformation"].nunique())
    if actual != expected:
        raise ValueError(f"Corpus benchmark differs: {actual}, expected {expected}")
    return table


def main():
    global PACKAGE, RAW, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    PACKAGE, RAW, OUT = args.package.resolve(), args.raw_dir.resolve(), args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = D.read_raw(RAW)
    reconciliation = accounting(raw)
    reconciliation.to_csv(OUT / "corpus_reconciliation.csv", index=False, lineterminator="\n")
    eligible = raw[~raw.repeated & raw.checked & raw.measurement_status.isin(D.MEASURED) &
                   ~raw.rxn_class.isin(D.NOT_A_COUPLING)].copy()
    eligible = eligible.rename(columns={"electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})
    eligible["yield"] = pd.to_numeric(eligible["yield"], errors="raise")
    if eligible.rxn_class.nunique() != 19:
        raise ValueError("Expected measured observations in all nineteen dual-substrate arrays")
    cells, domains, fit_domains, effects, full, excluded = [], [], [], [], {}, []

    def tracked_fit(rows, array, partition):
        result = fit(rows)
        fit_domains.append({"array": array, "partition": partition, **result["info"]})
        return result

    for name, group in eligible.groupby("rxn_class", sort=True):
        print(f"All-condition domain: {name}", flush=True)
        fitted = tracked_fit(group, name, "full")
        rows = fitted["rows"]
        full[name] = fitted["effects"]
        pairs = rows[["elec", "nuc"]].drop_duplicates()
        membership = pd.MultiIndex.from_frame(group[list(SIDES)]).isin(pd.MultiIndex.from_frame(pairs))
        omitted = group.loc[~membership, ["rxn_class", "source_row", "condition_id", "elec", "nuc", "measurement_status", "yield"]].copy()
        omitted["reason"] = "used_in_stage_one_only_outside_largest_stage_two_substrate_component"
        excluded.append(omitted)
        halves = [tracked_fit(half, name, f"condition_{h}")
                  for h, half in enumerate(condition_halves(fitted["stage_one_rows"]))]
        domains.append({"array": name, "transformation": transformation(name), **fitted["info"]})
        for side in SIDES:
            for molecule, value in fitted["effects"][side].items():
                effects.append({"array": name, "side": side, "molecule": molecule, "contribution": value})
            partner_pairs = partner_halves(pairs, name, side)
            partner_fits = [tracked_fit(rows.merge(p, on=list(SIDES), validate="many_to_one"),
                                       name, f"partner_{side}_{h}") for h, p in enumerate(partner_pairs)]
            cond = correlation(halves[0]["effects"][side], halves[1]["effects"][side], MIN_CORRELATION_MOLECULES)
            degree = pairs.groupby(side).size()
            pa, pb = [part["effects"][side] for part in partner_fits]
            common = pa.index.intersection(pb.index)
            common = common[degree.loc[common] >= 2]
            partner = correlation(pa.loc[common], pb.loc[common], MIN_CORRELATION_MOLECULES)
            matched = common.intersection(halves[0]["effects"][side].index).intersection(halves[1]["effects"][side].index)
            matched_cond = correlation(halves[0]["effects"][side].loc[matched],
                                       halves[1]["effects"][side].loc[matched], MIN_CORRELATION_MOLECULES)
            unavailable = lambda fs: any(f["info"]["status"] == "unidentified_block_adjusted_contributions" for f in fs)
            if unavailable(halves):
                cond["status"] = matched_cond["status"] = "unidentified_block_adjusted_contributions"
            if unavailable(partner_fits):
                partner["status"] = "unidentified_block_adjusted_contributions"
                matched_cond["status"] = "unidentified_partner_domain"
            cells.append({"array": name, "side": side, "n_condition_shared": cond["n_shared"],
                          "rho_condition": cond["rho"], "condition_status": cond["status"],
                          "n_partner_shared": partner["n_shared"], "rho_partner": partner["rho"],
                          "partner_status": partner["status"], "n_matched_condition_shared": matched_cond["n_shared"],
                          "rho_condition_on_partner_domain": matched_cond["rho"],
                          "matched_condition_status": matched_cond["status"]})
    cross = []
    for a, b in itertools.combinations(sorted(full), 2):
        for side in SIDES:
            same = transformation(a) == transformation(b)
            if same != (frozenset((a, b)) in E.SAME_REACTION):
                raise ValueError("Transformation labels disagree with the core same-reaction definition")
            result = correlation(full[a][side], full[b][side], E.MIN_SHARED)
            unidentified = any(d["status"] == "unidentified_block_adjusted_contributions"
                               for d in domains if d["array"] in (a, b))
            if unidentified:
                result["status"] = "unidentified_block_adjusted_contributions"
            cross.append({"array_a": a, "array_b": b, "transformation_a": transformation(a),
                          "transformation_b": transformation(b), "side": side, **result,
                          "same_transformation": same,
                          "included": not same and result["status"] == "ok",
                          "exclusion_reason": "same_transformation" if same else result["status"] if result["status"] != "ok" else ""})
    cells, cross = pd.DataFrame(cells), pd.DataFrame(cross)
    cross_valid = cross[cross.included]
    if cross_valid.empty or not cells.rho_condition.notna().any() or not cells.rho_partner.notna().any():
        raise ValueError("An agreement domain is empty")
    for table, name in ((cells, "agreement_cells"), (cross, "cross_reaction_agreement"),
                        (pd.DataFrame(domains), "estimand_domains"), (pd.DataFrame(effects), "molecule_contributions"),
                        (pd.DataFrame(fit_domains), "fit_domains"), (pd.concat(excluded, ignore_index=True), "connectivity_excluded_wells")):
        table.to_csv(OUT / f"{name}.csv", index=False, lineterminator="\n", float_format="%.17g")
    matched = cells[cells.rho_partner.notna() & cells.rho_condition_on_partner_domain.notna()]
    summary = {
        "raw_tables": len(reconciliation), "raw_records": int(reconciliation.raw_records.sum()),
        "excluded_version_conflict": int(reconciliation.excluded_version_conflict.sum()),
        "retained_records": int(reconciliation.retained_records.sum()),
        "dual_substrate_arrays": len(full), "article_arrays": int(reconciliation.used_array.sum()),
        "article_transformations": int(reconciliation.loc[reconciliation.used_array, "transformation"].nunique()),
        "condition_half": {"median": float(cells.rho_condition.median()), "cells": int(cells.rho_condition.notna().sum())},
        "partner_half": {"median": float(cells.rho_partner.median()), "cells": int(cells.rho_partner.notna().sum())},
        "matched_condition_partner_domain": {"cells": len(matched),
            "condition_median": float(matched.rho_condition_on_partner_domain.median()),
            "partner_median": float(matched.rho_partner.median())},
        "cross_reaction": {"median": float(cross_valid.rho.median()), "comparisons": len(cross_valid),
            "shared_count_weighted_mean": float(np.average(cross_valid.rho, weights=cross_valid.n_shared))},
        "full_domains": {"eligible_measured": int(sum(d["input_measured"] for d in domains)),
                         "domain_measured": int(sum(d["domain_measured"] for d in domains)),
                         "excluded_connectivity_measured": int(sum(d["excluded_connectivity_measured"] for d in domains)),
                         "identified_arrays": int(sum(d["status"] == "identified" for d in domains)),
                         "identified_domain_measured": int(sum(d["domain_measured"] for d in domains if d["status"] == "identified")),
                         "unidentified_domain_measured": int(sum(d["domain_measured"] for d in domains if d["status"] != "identified")),
                         "unidentified_arrays": [d["array"] for d in domains if d["status"] != "identified"]},
        "condition_statuses": cells.condition_status.value_counts().to_dict(),
        "partner_statuses": cells.partner_status.value_counts().to_dict(),
        "cross_comparison_exclusions": cross.loc[~cross.included, "exclusion_reason"].value_counts().to_dict(),
        "domain_definition": "All nineteen dual-substrate arrays; checked measured nonrepeated wells; all conditions and pairs enter stage one, fitted separately in every pair-condition block. Stage two uses the largest substrate-pair connected component, includes a block intercept, and has its rank checked. Rows outside the stage-two component still inform stage-one condition adjustment and are listed separately. No minimum block-size gate is imposed. Condition halves alternate coverage order within each full stage-one block. Partner halves use the Article's deterministic SHA-256 partition of the full fit's stage-two pairs. Within-array correlations need ten shared molecules; cross-transformation correlations need five. Constant and unidentifiable comparisons are explicitly excluded. The two Buchwald-Hartwig campaigns and two Suzuki arrays are same-transformation comparisons and excluded from cross-transformation aggregation. The primary summaries are unweighted medians; the shared-count-weighted cross-transformation mean is a separate statistic.",
    }
    (OUT / "agreement_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # A reversal requires the author to review the chemistry before any manuscript rewrite.
    if not summary["condition_half"]["median"] > summary["partner_half"]["median"] > summary["cross_reaction"]["median"]:
        raise RuntimeError(f"Main agreement ordering changed: {summary}")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
