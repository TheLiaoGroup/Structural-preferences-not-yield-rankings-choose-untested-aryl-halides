"""Build R4 figures, Source Data and workbook, with isolated preview paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
FIRST_SHEETS = {
    "Figure1_shared_molecules.csv": "Fig1ab_shared",
    "Figure1_structural_profiles.csv": "Fig1c_profiles",
    "Figure1_untested_molecules.csv": "Fig1d_untested",
    "Figure1_illustrated_structures.csv": "Fig1_structures",
}


def final_sheets():
    """Final CSV-to-workbook mapping, available without reading data or plotting."""
    from r5_presentation import final_sheets as current_sheets
    return current_sheets()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "output")
    parser.add_argument("--out", type=Path, default=ROOT / "output/natchem_presentation")
    parser.add_argument("--readings", type=Path, help="Alternate formal reaction-readings outputs")
    parser.add_argument("--only", nargs="*", help="Render only these R4 figures for isolated QA")
    args = parser.parse_args()
    import r4_figures as R4
    import figure1_selection as F1
    import source_data_workbook as W
    presentation = R4.Presentation(args.results, args.out, args.readings)
    sheets = presentation.sources()
    presentation.render(args.only)
    first = {}
    if not args.only or "Figure1" in args.only:
        F1.export(args.out / "figures", args.results)
    metric_path = args.out / "figures/Figure1_source_data/Figure1_metrics.json"
    if metric_path.exists():
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        qa = metrics["export_qa"]
        qa["panel_titles"] = qa["left_titles"]
        qa["text_overlap_pairs"] = metrics["qa"]["text_overlap_pairs"]
        scale = min(1., 6.8 * 25.4 / qa["width_mm"], 7.6 * 25.4 / qa["height_mm"])
        qa["word_placement_scale"] = scale
        qa["word_placed_min_text_pt"] = qa["min_text_pt"] * scale
        presentation.qa["Figure1"] = qa
        (args.out / "figure_qa.json").write_text(json.dumps(presentation.qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, sheet in FIRST_SHEETS.items():
        origin = args.out / "figures/Figure1_source_data" / name
        if origin.exists():
            target = args.out / "source_data" / name
            shutil.copyfile(origin, target)
            first[sheet] = target
    if len(first) == 4:
        W.build({**first, **sheets}, args.out / "Source_Data.xlsx")
    titles = {"Figure1": ["Two Buchwald–Hartwig campaigns", "The same molecules change rank",
                           "Structural preferences persist", "Choose molecules absent from campaign 1"], **R4.TITLES}
    (args.out / "panel_titles.json").write_text(json.dumps(titles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mapping = [{"producer": producer, "original_csv": old, "final_csv": new, "workbook_sheet": sheet}
               for producer, old, new, sheet in R4.SOURCE_MAP]
    mapping = [{"producer": "figure1_selection", "original_csv": name, "final_csv": name, "workbook_sheet": sheet}
               for name, sheet in FIRST_SHEETS.items()] + mapping
    (args.out / "source_data_map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    drawings = []
    for name, table, panel in (("Figure2", "Figure2c_BH_SNAr_amines.csv", "c"),
                                ("Figure4", "Figure4c_Heck_Suzuki_molecules.csv", "c")):
        frame = presentation.tab(table)
        for row in frame[frame.drawn.fillna(0).gt(0)].sort_values("drawn").itertuples():
            orient = R4.F23.ORIENT.get(row.mol, 0)
            drawings.append({"figure": name, "panel": panel, "SMILES": row.mol,
                             "number": int(row.drawn), "asset": R4.BF.CD.molecule(row.mol, orient)})
    for feature, _, smiles in R4.FEATURES:
        drawings.append({"figure": "Figure3", "panel": "a", "feature": feature,
                         "SMILES": smiles, "asset": R4.BF.CD.molecule(smiles, -1)})
    smiles = presentation.tab("Figure3c_methyl_2_bromoisonicotinate.csv").SMILES.iloc[0]
    drawings.append({"figure": "Figure3", "panel": "c", "SMILES": smiles, "asset": R4.BF.CD.molecule(smiles, -1)})
    for reaction in presentation.tab("ExtendedDataFig1_reaction_designs.csv").reaction:
        drawings.append({"figure": "ExtendedDataFig1", "panel": "a", "reaction": reaction,
                         "asset": R4.BF.CD.scheme(reaction)})
    illustrated = args.out / "source_data/Figure1_illustrated_structures.csv"
    if illustrated.exists():
        for row in R4.pd.read_csv(illustrated).itertuples():
            drawings.append({"figure": "Figure1", "panel": row.panel, "SMILES": row.SMILES,
                             "asset": R4.BF.CD.molecule(row.SMILES)})
    (args.out / "drawing_map.json").write_text(json.dumps(drawings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Nature Chemistry R4: {len(first) + len(sheets)} Source Data sheets; main Figures 1–4; Extended Data 1–8; SI Fig. 1")


if __name__ == "__main__":
    from r5_finalize import main as current_main
    current_main()
