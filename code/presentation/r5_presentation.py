"""R5 Article locations for the reproducible numerical panel tables."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

import figure1_selection as F1
import plot_figures as BF
import r4_figures as R4
import r5_style as S

ROOT = Path(__file__).resolve().parents[2]
ED_MAP = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}


def relocated(name):
    if name.startswith("Figure3"):
        return name.replace("Figure3", "Figure4", 1)
    if name.startswith("Figure4a"):
        return name.replace("Figure4a", "Figure5b", 1)
    if name.startswith("Figure4b"):
        return name.replace("Figure4b", "Figure5c", 1)
    if name.startswith("Figure4c"):
        return name.replace("Figure4c", "Figure5d", 1)
    for old, new in ED_MAP.items():
        if name.startswith(f"ExtendedDataFig{old}"):
            return name.replace(f"ExtendedDataFig{old}", f"ExtendedDataFig{new}", 1)
    return name


SOURCE_MAP = []
for producer, original, previous, sheet in R4.SOURCE_MAP:
    if previous == "Figure2a_pooled_portability.csv":
        continue
    if previous == "ExtendedDataFig3_portability_by_set.csv":
        final, sheet = "Figure2b_portability_by_set.csv", "Fig2b_sets"
    else:
        final = relocated(previous)
        sheet = final.removesuffix(".csv").replace("ExtendedDataFig", "ED").replace("SupplementaryFig", "SI").replace("Figure", "F")
        if len(sheet) > 31:
            sheet = sheet[:24] + f"_{len(SOURCE_MAP):02d}"
    SOURCE_MAP.append((producer, original, final, sheet))

FIRST_SHEETS = {
    "Figure3_shared_molecules.csv": "Fig3b_shared",
    "Figure3_structural_profiles.csv": "Fig3b_profiles",
    "Figure3_untested_molecules.csv": "Fig3cd_untested",
    "Figure3_campaign_composition.csv": "Fig3a_campaigns",
    "Figure1_archive_counts.csv": "Fig1_archive",
    "Figure1_array_sizes.csv": "Fig1_arrays",
}


def overview_molecules(candidates):
    ordered = candidates.sort_values("profile_selection_rank")
    pattern = Chem.MolFromSmarts("[Br]-c:[n;X2;H0;+0]")
    bromine_adjacent = ordered.SMILES.map(lambda smiles: Chem.MolFromSmiles(smiles).HasSubstructMatch(pattern))
    selected = ordered.loc[ordered.top_ten_selection_weight.gt(0) & bromine_adjacent].head(4)
    controls = ordered.loc[ordered.top_ten_selection_weight.eq(0)].head(2)
    if len(selected) != 4 or len(controls) != 2:
        raise ValueError("The overview requires four source-ranked selections and two source-ranked controls")
    return selected, controls


class Presentation(R4.Presentation):
    def __init__(self, results, out, readings=None):
        super().__init__(results, out, readings)
        self.aliases = {previous: relocated(previous) for _, _, previous, _ in R4.SOURCE_MAP}
        self.aliases["ExtendedDataFig3_portability_by_set.csv"] = "Figure2b_portability_by_set.csv"

    def tab(self, name):
        path = self.out / "source_data" / name
        if not path.exists():
            path = self.out / "source_data" / self.aliases.get(name, name)
        return pd.read_csv(path, float_precision="round_trip")

    def sources(self):
        from r5_pointclouds import build_sources
        directory = self.out / "source_data"
        directory.mkdir(parents=True, exist_ok=True)
        bases = {"source": self.source, "readings": self.readings,
                 "selection": self.results / "sensitivity/structure_for_selection", "tables": self.results / "presentation/tables"}
        sheets = {}
        for producer, original, final, sheet in SOURCE_MAP:
            shutil.copyfile(bases[producer] / original, directory / final)
            sheets[sheet] = directory / final
        rank, profiles, selected, values = F1.read_data(self.results)
        for column, smarts in (
                ("any_halogen_beside_pyridine_n", "[F,Cl,Br,I]-c:[n;X2;H0;+0]"),
                ("bromine_beside_pyridine_n", "[Br]-c:[n;X2;H0;+0]")):
            pattern = Chem.MolFromSmarts(smarts)
            selected[column] = selected.SMILES.map(lambda smiles: Chem.MolFromSmiles(smiles).HasSubstructMatch(pattern))
        for name, frame in (("Figure3_shared_molecules.csv", rank), ("Figure3_structural_profiles.csv", profiles),
                            ("Figure3_untested_molecules.csv", selected)):
            frame.to_csv(directory / name, index=False, float_format="%.17g", lineterminator="\n")
        core = pd.read_csv(self.results / "core/molecule_contributions.csv")
        campaign = pd.read_csv(self.results / "extended/campaign_contributions.csv")
        designs = pd.read_csv(self.results / "core/design_counts.csv").set_index("rxn_class")
        campaign_conditions = pd.read_csv(ROOT / "data/metadata/extended_conditions.csv")
        campaign_condition_count = campaign_conditions.loc[
            campaign_conditions.rxn_class.eq("Buchwald-Hartwig(CMC)"), "condition_id"].nunique()
        compositions = pd.DataFrame([
            dict(campaign="Buchwald-Hartwig 1", electrophiles=values["source_n"],
                 amines=core.loc[core.reaction.eq("Buchwald-Hartwig") & core.side.eq("nuc"), "SMILES"].nunique(),
                 conditions=int(designs.loc["Buchwald-Hartwig", "conditions"])),
            dict(campaign="Buchwald-Hartwig 2", electrophiles=values["target_n"],
                 amines=campaign.loc[campaign.side.eq("nuc"), "SMILES"].nunique(), conditions=campaign_condition_count),
        ])
        compositions.to_csv(directory / "Figure3_campaign_composition.csv", index=False, lineterminator="\n")
        wells = pd.read_csv(ROOT / "data/experimental_wells.csv", low_memory=False)
        measured = wells.loc[wells.measurement_status.isin(["quantified", "zero_by_rule"])]
        unique = set(measured.electrophile_smiles.dropna()) | set(measured.nucleophile_smiles.dropna())
        counts = pd.DataFrame([dict(experiments=len(measured), arrays=measured.rxn_class.nunique(),
                                    transformations=measured.rxn_class.replace({"Buchwald-Hartwig(CMC)": "Buchwald-Hartwig", "Suzuki Bpin": "Suzuki"}).nunique(), substrates=len(unique))])
        counts.to_csv(directory / "Figure1_archive_counts.csv", index=False, lineterminator="\n")
        arrays = measured.groupby("rxn_class", sort=True).size().rename("measured_wells").reset_index()
        arrays["display_name"] = arrays.rxn_class.map(
            lambda name: S.reaction_name("Buchwald-Hartwig") + " 2" if name == "Buchwald-Hartwig(CMC)" else S.reaction_name(name))
        arrays.to_csv(directory / "Figure1_array_sizes.csv", index=False, lineterminator="\n")
        self.values = values
        (self.out / "figure3_metrics.json").write_text(json.dumps(values, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        for name, sheet in FIRST_SHEETS.items():
            sheets[sheet] = directory / name
        sheets.update(build_sources(self.results, directory))
        return sheets

    def export(self, name, fig):
        S.finalize(fig)
        BF.finalize = S.finalize
        actual = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
        if actual != self.titles[name]:
            raise ValueError(f"{name}: panel titles differ: {actual!r} != {self.titles[name]!r}")
        qa = BF.export_figure(fig, self.out / "figures" / name, creator="r5_presentation.py")
        overlaps = R4.F23.overlaps(fig)
        if overlaps:
            raise ValueError(f"{name}: rendered text overlaps: {overlaps}")
        qa.update(panel_titles=actual, text_overlap_pairs=0, word_placement_scale=1,
                  word_placed_min_text_pt=qa["min_text_pt"])
        self.qa[name] = qa
        plt.close(fig)


def final_sheets():
    from r5_pointclouds import SHEETS
    return {**FIRST_SHEETS, **{final: sheet for _, _, final, sheet in SOURCE_MAP}, **SHEETS}
