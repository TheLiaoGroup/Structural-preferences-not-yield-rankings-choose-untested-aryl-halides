"""Source Data for pooled molecular comparisons and cross-role boronic acids."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CORE_CODE = ROOT / "code/core"
CATEGORIES = ("conditions", "partners", "reactions")
SHEETS = {
    "Figure2a_cloud_points.csv": "Fig2a_points",
    "Figure2a_cloud_summary.csv": "Fig2a_clouds",
    "Figure2a_aggregate_summary.csv": "Fig2a_aggregates",
    "Figure2c_selection_summary.csv": "Fig2c_selection",
    "Figure2d_boronic_molecules.csv": "Fig2d_molecules",
    "Figure2d_boronic_summary.csv": "Fig2d_summary",
}


def _read(path):
    return pd.read_csv(path, float_precision="round_trip")


def _true(series):
    return series.astype(str).str.lower().eq("true")


def _core_modules():
    if str(CORE_CODE) not in sys.path:
        sys.path.insert(0, str(CORE_CODE))
    import analysis_bba
    import gated_bba
    return analysis_bba, gated_bba


def _paired_points(category, reaction_a, reaction_b, side, a, b, shared):
    """Standardize a complete reaction-side estimate before matching molecules."""
    a, b = a.dropna().round(12), b.dropna().round(12)
    x_mean, y_mean = float(a.mean()), float(b.mean())
    x_sd, y_sd = float(a.std(ddof=0)), float(b.std(ddof=0))
    if min(x_sd, y_sd) <= 0:
        raise ValueError(f"Constant contribution group: {category}/{reaction_a}/{reaction_b}/{side}")
    shared = pd.Index(shared).sort_values()
    return pd.DataFrame({
        "category": category,
        "group_key": f"{category}|{reaction_a}|{reaction_b}|{side}",
        "reaction_a": reaction_a, "reaction_b": reaction_b, "side": side,
        "SMILES": shared,
        "x_contribution_pp": 100 * a[shared].to_numpy(),
        "y_contribution_pp": 100 * b[shared].to_numpy(),
        "x_z": (a[shared].to_numpy() - x_mean) / x_sd,
        "y_z": (b[shared].to_numpy() - y_mean) / y_sd,
        "x_group_mean_pp": 100 * x_mean, "y_group_mean_pp": 100 * y_mean,
        "x_group_sd_pp": 100 * x_sd, "y_group_sd_pp": 100 * y_sd,
        "x_standardization_n": len(a), "y_standardization_n": len(b),
    })


def _condition_points(package, results, A, G):
    cols = ["well_id", "rxn_class", "condition_id", "electrophile_smiles", "nucleophile_smiles", "yield",
            "component", "measurement_status"]
    wells = pd.read_csv(package / "data/experimental_wells.csv", usecols=cols,
                        dtype={"condition_id": str}, float_precision="round_trip")
    membership = _read(package / "data/metadata/analysis_well_membership.csv")
    ids = set(membership.loc[membership.analysis_set.eq("primary"), "well_id"])
    wells = wells.loc[wells.well_id.isin(ids) & wells.component.eq("existing_release") &
                      wells.measurement_status.isin(["quantified", "zero_by_rule"])].rename(columns={
        "electrophile_smiles": "elec", "nucleophile_smiles": "nuc"})
    conditions = pd.read_csv(results / "core/extended_conditions_bba.csv", dtype={"condition_id": str})
    frames, records = [], []
    for reaction, rows in wells.groupby("rxn_class", sort=True):
        order = conditions.loc[conditions.rxn_class.eq(reaction)].sort_values("rank").condition_id.tolist()
        halves = []
        for parity in (0, 1):
            subset = rows.loc[rows.condition_id.isin(order[parity::2])]
            pair = A.two_way_fit(subset["yield"].to_numpy(),
                                 subset.elec + " >> " + subset.nuc, subset.condition_id)
            keys = pd.DataFrame([key.split(" >> ") for key in pair.index], columns=["elec", "nuc"])
            fit = A.stage_two(keys.assign(p=pair.values))
            halves.append({"elec": fit["a"], "nuc": fit["b"]})
        for side in ("elec", "nuc"):
            a, b = halves[0][side], halves[1][side]
            shared = a.index.intersection(b.index)
            records.append({"rxn_class": reaction, "side": side, "r_half": G.spearman(a[shared], b[shared])})
            frames.append(_paired_points("conditions", reaction, reaction, side, a, b, shared))
    saved = _read(results / "core/extended_reliability_bba.csv")
    check = pd.DataFrame(records).merge(saved, on=["rxn_class", "side"], suffixes=("_new", "_saved"), validate="one_to_one")
    if len(check) != len(saved) or not np.allclose(check.r_half_new, check.r_half_saved, atol=1e-12, rtol=0):
        raise ValueError("Condition-half contributions do not reproduce the released rank correlations")
    return frames


def _partner_points(results, G):
    values = _read(results / "core/partner_half_contributions_bba.csv")
    values["evaluable_k2"] = _true(values.evaluable_k2)
    cells = _read(results / "core/gated_A_cells_bba.csv")
    cells = cells.loc[cells.degree.eq(2) & cells.rho_P.notna()]
    frames = []
    for row in cells.itertuples():
        group = values.loc[values.reaction.eq(row.reaction) & values.side.eq(row.side)]
        a = group.loc[group.half.eq("A")].set_index("molecule").contribution
        b = group.loc[group.half.eq("B")].set_index("molecule").contribution
        eligible = set(group.loc[group.evaluable_k2, "molecule"])
        shared = a.index.intersection(b.index).intersection(pd.Index(sorted(eligible)))
        shared = shared[np.isfinite(a[shared]) & np.isfinite(b[shared])]
        if len(shared) != row.molecules or not np.isclose(G.spearman(a[shared], b[shared]), row.rho_P, atol=1e-12):
            raise ValueError(f"Partner-half membership or correlation differs: {row.reaction}/{row.side}")
        frames.append(_paired_points("partners", row.reaction, row.reaction, row.side, a, b, shared))
    return frames


def _reaction_points(results, G):
    effects = _read(results / "core/full_data_effects_bba.csv")
    effects = effects.loc[effects.scale.eq("raw") & effects.label.eq("primary")]
    full = {(reaction, side): g.set_index("mol").main_effect
            for (reaction, side), g in effects.groupby(["rxn_class", "side"], sort=True)}
    transfer = _read(results / "core/extended_transfer_bba.csv")
    transfer = transfer.loc[~_true(transfer.same_reaction) & transfer.n_shared.ge(5)]
    frames = []
    for row in transfer.itertuples():
        a, b = full[row.rxn_a, row.side], full[row.rxn_b, row.side]
        shared = a.index.intersection(b.index)
        if len(shared) != row.n_shared or not np.isclose(G.spearman(a[shared], b[shared]), row.rho_obs, atol=1e-12):
            raise ValueError(f"Cross-reaction membership or correlation differs: {row.rxn_a}/{row.rxn_b}/{row.side}")
        frames.append(_paired_points("reactions", row.rxn_a, row.rxn_b, row.side, a, b, shared))
    return frames, transfer


def _boronic_sources(results):
    directory = results / "sensitivity/boronic_acids_across_roles"
    members = _read(directory / "boronic_acids_across_roles_members.csv")
    summary = _read(directory / "boronic_acids_across_roles.csv")
    members = members.loc[members.condition_set.eq("first")].rename(columns={"mol": "SMILES"}).copy()
    summary = summary.loc[summary.condition_set.eq("first")].copy()
    if len(summary) != 1 or len(members) != int(summary.n_shared.iloc[0]):
        raise ValueError("Boronic-acid summary and membership differ")
    observed = float(spearmanr(members.suzuki_pp, members.chanlam_pp).statistic)
    if not np.isclose(observed, summary.rho_obs.iloc[0], atol=1e-12):
        raise ValueError("Boronic-acid rank agreement does not reproduce the saved output")
    flag = _true(members.five_membered)
    for column, expected in (("suzuki_pp", "suzuki_five_minus_other_pp"), ("chanlam_pp", "chanlam_five_minus_other_pp")):
        contrast = float(members.loc[flag, column].mean() - members.loc[~flag, column].mean())
        if not np.isclose(contrast, summary[expected].iloc[0], atol=1e-10):
            raise ValueError(f"Boronic-acid feature contrast differs: {column}")
    return members.sort_values("SMILES", kind="stable"), summary


def build_sources(results, out, package=None):
    """Write only to the explicit output directory; return workbook-sheet paths."""
    results, out = Path(results), Path(out)
    package = ROOT if package is None else Path(package)
    A, G = _core_modules()
    frames = _condition_points(package, results, A, G) + _partner_points(results, G)
    cross, transfer = _reaction_points(results, G)
    points = pd.concat(frames + cross, ignore_index=True)
    points = points.sort_values(["category", "group_key", "SMILES"], kind="stable").reset_index(drop=True)
    summaries = []
    for category in CATEGORIES:
        group = points.loc[points.category.eq(category)]
        summaries.append({"category": category,
                          "rho_pooled": float(spearmanr(group.x_z, group.y_z).statistic),
                          "n_points": len(group), "n_groups": group.group_key.nunique(),
                          "n_unique_molecules": group.SMILES.nunique(),
                          "standardization": "complete reaction-side estimate; population SD; contributions rounded to 12 decimals"})
    cloud_summary = pd.DataFrame(summaries)
    if not np.all(np.diff(cloud_summary.rho_pooled.to_numpy()) < 0):
        raise RuntimeError("STOP: pooled point-cloud ordering is not conditions > partners > reactions")
    reliability = _read(results / "core/extended_reliability_bba.csv")
    cells = _read(results / "core/gated_A_cells_bba.csv")
    cells = cells.loc[cells.degree.eq(2) & cells.rho_P.notna()]
    aggregate = pd.DataFrame([
        {"category": "conditions", "estimate": reliability.r_half.median(), "groups": len(reliability),
         "definition": "median condition-half Spearman across reaction-side groups"},
        {"category": "partners", "estimate": cells.rho_P.median(), "groups": len(cells),
         "definition": "median partner-half Spearman across estimable reaction-side groups"},
        {"category": "reactions", "estimate": np.average(transfer.rho_disattenuated, weights=transfer.n_shared),
         "groups": len(transfer), "definition": "shared-count-weighted mean of reliability-corrected cross-reaction Spearman"},
    ])
    boronic_molecules, boronic_summary = _boronic_sources(results)
    amines = transfer.loc[transfer.rxn_a.eq("Buchwald-Hartwig") & transfer.rxn_b.eq("SNAr") & transfer.side.eq("nuc")].copy()
    if len(amines) != 1:
        raise ValueError("Expected one Buchwald-Hartwig to SNAr amine comparison")
    amines["target_top_ten_shortfall_pp"] = 100 * (amines.B_mean_B_best - amines.B_mean_A_selected)
    tables = {"Figure2a_cloud_points.csv": points, "Figure2a_cloud_summary.csv": cloud_summary,
              "Figure2a_aggregate_summary.csv": aggregate,
              "Figure2c_selection_summary.csv": amines,
              "Figure2d_boronic_molecules.csv": boronic_molecules,
              "Figure2d_boronic_summary.csv": boronic_summary}
    out.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_csv(out / filename, index=False, float_format="%.17g", lineterminator="\n")
    return {SHEETS[name]: out / name for name in tables}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "output")
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    for sheet, path in build_sources(options.results, options.out).items():
        print(f"{sheet}: {path}")
