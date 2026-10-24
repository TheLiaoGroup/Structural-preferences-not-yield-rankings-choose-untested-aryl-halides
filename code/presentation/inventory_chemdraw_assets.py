"""Inventory structures and visible titles without exporting or changing analysis data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.text import Text
from rdkit import Chem

import plot_figures as BF
import figures_1_4 as F14
import figures_2_3 as F23
import figure1_selection as F1


def inventory(out):
    out.mkdir(parents=True, exist_ok=True)
    current = ""
    structures, figures = {}, []
    original_structure, original_save = BF.structure, BF.save_figure

    def capture_structure(ax, smiles, cx, cy, scale, orient=None):
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
        orientation = None if orient is None else int(orient)
        key = (canonical, orientation)
        if current == "Figure1":
            panel = "a" if ax.get_position().x0 < .5 else "d"
        elif current == "Figure3":
            panel = "d" if any(frame.function == "structure_strip" for frame in inspect.stack()) else "c"
        elif current == "Figure5":
            panel = "c"
        else:
            panel = "unassigned"
        item = structures.setdefault(key, {"asset_id": "mol_" + hashlib.sha256(
            (canonical + "|" + str(orientation)).encode()).hexdigest()[:12],
            "canonical_smiles": canonical, "smiles": smiles, "orient": orientation,
            "panels": set(), "occurrences": 0, "old_scales": set()})
        item["panels"].add(current + panel)
        item["occurrences"] += 1
        item["old_scales"].add(float(scale))
        return original_structure(ax, smiles, cx, cy, scale, orient)

    def capture_figure(fig, name):
        titles = []
        for i, ax in enumerate(fig.axes):
            for location in ("left", "center", "right"):
                title = ax.get_title(loc=location)
                if title:
                    titles.append({"axes": i, "location": location, "text": title})
        figures.append({"figure": name, "axes_titles": titles,
                        "panel_titles": [t.get_text() for t in fig.findobj(Text)
                                         if t.get_gid() == "panel-title"]})
        plt.close(fig)

    BF.structure, BF.save_figure = capture_structure, capture_figure
    try:
        for name, make in (("Figure1", F1.figure1), ("Figure2", F14.figure1),
                           ("Figure3", F23.figure2), ("Figure4", F23.figure3),
                           ("Figure5", F14.figure4), ("ExtendedDataFig1", F14.ed_fig1),
                           ("ExtendedDataFig2", F14.ed_fig2)):
            current = name
            BF.style()
            result = make()
            capture_figure(result[0] if isinstance(result, tuple) else result, name)
        current = "ExtendedData"
        BF.plot_figures()
    finally:
        BF.structure, BF.save_figure = original_structure, original_save
    rows = []
    for key in sorted(structures, key=lambda x: (x[0], -99 if x[1] is None else x[1])):
        item = structures[key]
        item["panels"] = ";".join(sorted(item["panels"]))
        item["old_scales"] = ";".join(str(v) for v in sorted(item["old_scales"]))
        rows.append(item)
    fields = ["asset_id", "canonical_smiles", "smiles", "orient", "panels", "occurrences", "old_scales"]
    with (out / "unique_structures.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "figure_titles.json").write_text(json.dumps(figures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"unique_assets": len(rows), "structure_placements": sum(r["occurrences"] for r in rows),
                      "figures": len(figures), "out": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    inventory(parser.parse_args().out)
