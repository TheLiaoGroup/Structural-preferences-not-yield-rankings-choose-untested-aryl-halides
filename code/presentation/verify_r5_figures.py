"""Independently verify R5 panel data, vector/raster exports and source tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import openpyxl
import pandas as pd
from PIL import Image
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
NAMES = ([f"Figure{k}" for k in range(1, 6)] +
         [f"ExtendedDataFig{k}" for k in range(1, 8)] + ["SupplementaryFig1"])
CATEGORIES = ("conditions", "partners", "reactions")
DEFINITIONS = {
    "conditions": "median condition-half Spearman across reaction-side groups",
    "partners": "median partner-half Spearman across estimable reaction-side groups",
    "reactions": "shared-count-weighted mean of reliability-corrected cross-reaction Spearman",
}
APPROVED = {
    "conditions": (1643, 30, 1014, .9353755278504314),
    "partners": (886, 24, 601, .5462824193715299),
    "reactions": (1113, 58, 292, .12951375021871697),
}


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def truth(values):
    return values.astype(str).str.lower().eq("true")


def spearman(x, y):
    return float(pd.Series(np.asarray(x)).rank().corr(pd.Series(np.asarray(y)).rank()))


def normalize(text):
    text = text.replace("S$_\\mathregular{N}$Ar", "SNAr")
    return re.sub(r"\s+", "", text).casefold()


def close(a, b, atol=1e-10):
    return bool(np.isclose(a, b, rtol=0, atol=atol))


def cloud_checks(results, points, summary, aggregate):
    errors, inventory = [], []
    rel = read(results / "core/extended_reliability_bba.csv")
    cells = read(results / "core/gated_A_cells_bba.csv")
    cells = cells.loc[cells.degree.eq(2) & cells.rho_P.notna()]
    transfer = read(results / "core/extended_transfer_bba.csv")
    transfer = transfer.loc[~truth(transfer.same_reaction) & transfer.n_shared.ge(5)]
    effects = read(results / "core/full_data_effects_bba.csv")
    effects = effects.loc[effects.scale.eq("raw") & effects.label.eq("primary")]
    complete = {(reaction, side): frame.set_index("mol").main_effect
                for (reaction, side), frame in effects.groupby(["rxn_class", "side"])}
    halves = read(results / "core/partner_half_contributions_bba.csv")
    expected_aggregates = (rel.r_half.median(), cells.rho_P.median(),
                           np.average(transfer.rho_disattenuated, weights=transfer.n_shared))
    if tuple(summary.category) != CATEGORIES or tuple(aggregate.category) != CATEGORIES:
        errors.append("Cloud and aggregate summaries must each contain the three ordered categories")
        return errors, inventory
    if set(points.category) != set(CATEGORIES):
        errors.append("Point-table categories differ from the three scientific comparisons")
    if points.duplicated(["category", "group_key", "SMILES"]).any():
        errors.append("Point table duplicates a molecule within one comparison")
    molecule_sets = []
    for category, expected in zip(CATEGORIES, expected_aggregates):
        group = points.loc[points.category.eq(category)]
        row = summary.loc[summary.category.eq(category)].iloc[0]
        agg = aggregate.loc[aggregate.category.eq(category)].iloc[0]
        molecule_sets.append(set(group.SMILES))
        observed = (len(group), group.group_key.nunique(), group.SMILES.nunique(),
                    spearman(group.x_z, group.y_z))
        for found, saved in zip(observed[:3], (row.n_points, row.n_groups, row.n_unique_molecules)):
            if found != saved:
                errors.append(f"{category}: point-membership counts differ from the summary")
        if not close(observed[3], row.rho_pooled):
            errors.append(f"{category}: pooled rank correlation differs from the displayed points")
        if observed[:3] != APPROVED[category][:3] or not close(observed[3], APPROVED[category][3]):
            errors.append(f"{category}: membership or pooled estimate differs from the approved R5 baseline")
        if agg.definition != DEFINITIONS[category] or not close(agg.estimate, expected):
            errors.append(f"{category}: headline aggregate definition or value differs from formal analysis")
        if close(agg.estimate, row.rho_pooled, atol=1e-8):
            errors.append(f"{category}: headline aggregate has been substituted for the pooled cloud correlation")
        for axis in ("x", "y"):
            z = (group[f"{axis}_contribution_pp"] - group[f"{axis}_group_mean_pp"]) / group[f"{axis}_group_sd_pp"]
            if not np.allclose(z, group[f"{axis}_z"], rtol=0, atol=1e-10):
                errors.append(f"{category}: {axis} standardization does not match the complete-side moments")
        for key, members in group.groupby("group_key", sort=True):
            first = members.iloc[0]
            reaction, side = first.reaction_a, first.side
            expected_values = None
            if category == "conditions":
                formal = rel.loc[rel.rxn_class.eq(reaction) & rel.side.eq(side)]
                expected_r = formal.r_half.iloc[0] if len(formal) == 1 else np.nan
                expected_molecules = set(complete[reaction, side].index)
            elif category == "partners":
                formal = cells.loc[cells.reaction.eq(reaction) & cells.side.eq(side)]
                expected_r = formal.rho_P.iloc[0] if len(formal) == 1 else np.nan
                half = halves.loc[halves.reaction.eq(reaction) & halves.side.eq(side)]
                a = half.loc[half.half.eq("A")].set_index("molecule").contribution
                b = half.loc[half.half.eq("B")].set_index("molecule").contribution
                eligible = set(half.loc[truth(half.evaluable_k2), "molecule"])
                shared = a.index.intersection(b.index)
                expected_molecules = set(shared[np.isfinite(a[shared]) & np.isfinite(b[shared])]) & eligible
                expected_values = (a, b)
                if len(formal) == 1 and len(members) != formal.molecules.iloc[0]:
                    errors.append(f"{key}: partner membership differs from the formal eligible cell")
            else:
                formal = transfer.loc[transfer.rxn_a.eq(reaction) & transfer.rxn_b.eq(first.reaction_b) & transfer.side.eq(side)]
                expected_r = formal.rho_obs.iloc[0] if len(formal) == 1 else np.nan
                a, b = complete[reaction, side], complete[first.reaction_b, side]
                expected_molecules = set(a.index) & set(b.index)
                expected_values = (a, b)
                if len(formal) == 1 and len(members) != formal.n_shared.iloc[0]:
                    errors.append(f"{key}: shared-molecule membership differs from formal transfer")
            if set(members.SMILES) != expected_molecules:
                errors.append(f"{key}: molecule identities differ from the formal eligible comparison")
            elif expected_values is not None:
                for axis, values in zip(("x", "y"), expected_values):
                    expected_pp = 100 * values.loc[members.SMILES].round(12).to_numpy()
                    if not np.allclose(members[f"{axis}_contribution_pp"], expected_pp, rtol=0, atol=1e-10):
                        errors.append(f"{key}: {axis} contributions are attached to the wrong molecule")
            if not close(spearman(members.x_contribution_pp, members.y_contribution_pp), expected_r):
                errors.append(f"{key}: within-comparison ranks differ from the formal result")
        inventory.append(dict(category=category, appearances=observed[0], groups=observed[1],
                              unique_molecules=observed[2], pooled_spearman=observed[3],
                              headline_estimate=float(agg.estimate), headline_definition=agg.definition))
    if any(a == b for i, a in enumerate(molecule_sets) for b in molecule_sets[i+1:]):
        errors.append("Clouds were incorrectly reduced to a common molecular universe")
    if not all(a["pooled_spearman"] > b["pooled_spearman"] for a, b in zip(inventory, inventory[1:])):
        errors.append("STOP: pooled cloud order is not conditions > partners > reactions")
    return errors, inventory


def svg_points(value):
    match = re.fullmatch(r"\s*([0-9.]+)(mm|pt|px|in|cm)?\s*", value or "")
    scale = {None: .75, "px": .75, "pt": 1, "mm": 72/25.4, "cm": 72/2.54, "in": 72}
    if not match:
        raise ValueError(f"Unsupported SVG dimension: {value!r}")
    return float(match[1]) * scale[match[2]]


def snar_script_origins(page):
    """Bind a smaller, lowered N to the surrounding S and Ar, including rotated labels."""
    characters = [{**char, "size": span["size"]}
                  for block in page.get_text("rawdict")["blocks"] if "lines" in block
                  for line in block["lines"] for span in line["spans"] for char in span["chars"]]
    origins = set()
    for index in range(1, len(characters) - 2):
        before, script, after, last = characters[index-1:index+3]
        if (before["c"], script["c"], after["c"], last["c"]) != ("S", "N", "A", "r"):
            continue
        base = before["size"]
        if not ((close(base, 8, atol=.01) or close(base, 9, atol=.01))
                and close(after["size"], base, atol=.01) and close(last["size"], base, atol=.01)
                and close(script["size"], .7 * base, atol=.01)):
            continue
        start, finish, middle = [np.asarray(char["origin"]) for char in (before, after, script)]
        advance = finish - start
        distance = np.linalg.norm(advance)
        if not 0 < distance < 2 * base:
            continue
        along = advance / distance
        offset = middle - start
        drop = np.dot(offset, [-along[1], along[0]])
        if 0 < np.dot(offset, along) < distance and .07 * base < drop < .5 * base:
            origins.add(tuple(round(value, 3) for value in script["origin"]))
    return origins


def figure_checks(directory, names, titles, qa, errors):
    records = []
    for name in names:
        paths = {suffix: directory / f"figures/{name}.{suffix}" for suffix in ("svg", "pdf", "png")}
        if not all(path.is_file() for path in paths.values()):
            errors.append(f"{name}: SVG, PDF and PNG masters are required")
            continue
        root = ET.parse(paths["svg"]).getroot()
        if "SNAr" in paths["svg"].read_text(encoding="utf-8"):
            errors.append(f"{name}: SNAr must use an actual subscript N in the SVG")
        svg_images = sum(node.tag.rsplit("}", 1)[-1] == "image" for node in root.iter())
        if name == "Figure1" and svg_images:
            errors.append(f"{name}: SVG contains an embedded raster image")
        with pymupdf.open(paths["pdf"]) as document:
            if len(document) != 1:
                errors.append(f"{name}: PDF requires exactly one page")
            page = document[0]
            width, height = page.rect.width, page.rect.height
            if width * 25.4 / 72 > 180.001:
                errors.append(f"{name}: PDF exceeds 180 mm width")
            raster_images = len(page.get_images())
            if name == "Figure1" and raster_images:
                errors.append(f"{name}: PDF contains a raster image")
            spans = [s for b in page.get_text("dict")["blocks"] if "lines" in b
                     for line in b["lines"] for s in line["spans"] if s["text"].strip()]
            visible = "".join(s["text"] for s in spans)
            lowered_n = snar_script_origins(page)
            bad_fonts, outside, negative_hyphens = [], [], []
            for span in spans:
                size, value = span["size"], span["text"]
                # Native chemical subscripts/superscripts retain their 6 pt setting.
                chemistry_script = close(size, 6, atol=.01) and re.fullmatch(r"[0-9+\-\u2212]+", value)
                snar_script = value == "N" and tuple(round(v, 3) for v in span["origin"]) in lowered_n
                if not (close(size, 8, atol=.01) or close(size, 9, atol=.01) or chemistry_script or snar_script):
                    bad_fonts.append((value, size))
                box = pymupdf.Rect(span["bbox"])
                if box.x0 < -.6 or box.y0 < -.6 or box.x1 > width+.6 or box.y1 > height+.6:
                    outside.append(value)
                if re.search(r"(?<![A-Za-z0-9])-(?=\d|\.\d)", value):
                    negative_hyphens.append(value)
            if bad_fonts:
                errors.append(f"{name}: glyphs outside the 8/9 pt scale (chemical scripts and bound SNAr subscripts allowed): {bad_fonts}")
            if outside:
                errors.append(f"{name}: PDF text extends outside the physical canvas: {outside}")
            if negative_hyphens:
                errors.append(f"{name}: numeric signs use hyphens: {negative_hyphens}")
            overlaps = []
            atoms = {"H", "N", "O", "S", "Br", "Cl", "R", "X", "F", "P", "B", "Si", "Mg", "I"}
            for a, b in combinations([s for s in spans if s["size"] >= 7.9], 2):
                if a["text"] in atoms and b["text"] in atoms:
                    continue
                box_a, box_b = pymupdf.Rect(a["bbox"]), pymupdf.Rect(b["bbox"])
                intersection = box_a & box_b
                if (not intersection.is_empty and intersection.get_area() > 1 and
                        intersection.get_area() / min(box_a.get_area(), box_b.get_area()) > .15):
                    overlaps.append([a["text"], b["text"]])
            if overlaps:
                errors.append(f"{name}: overlapping PDF text spans: {overlaps}")
            for title in titles.get(name, []):
                if normalize(title) not in normalize(visible):
                    errors.append(f"{name}: left panel title missing from visible PDF: {title}")
            for axis, expected in (("width", width), ("height", height)):
                if not close(svg_points(root.get(axis)), expected, atol=.02):
                    errors.append(f"{name}: SVG and PDF physical {axis} differ")
        with Image.open(paths["png"]) as image:
            pixels = image.size
            if abs(pixels[0] - width/72*600) > 1.1 or abs(pixels[1] - height/72*600) > 1.1:
                errors.append(f"{name}: PNG pixel dimensions differ from the PDF at 600 dpi")
            if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
                errors.append(f"{name}: PNG has transparent pixels")
        saved = qa.get(name, {})
        if saved.get("left_titles") != titles.get(name):
            errors.append(f"{name}: actual get_title(loc='left') inventory differs from panel_titles.json")
        if saved.get("undersized_text") or saved.get("outside_canvas_text") or saved.get("text_overlap_pairs", 0):
            errors.append(f"{name}: renderer reports unresolved text layout failures")
        records.append(dict(figure=name, width_mm=width/72*25.4, height_mm=height/72*25.4,
                            pixels=pixels, min_glyph_pt=min(s["size"] for s in spans),
                            native_chemical_script_spans=sum(s["size"] < 7.9 for s in spans),
                            titles=titles.get(name, []), raster_images=raster_images,
                            measured_text_overlaps=overlaps,
                            text_geometry_measured_from_pdf=True))
    return records


def workbook_checks(directory, errors):
    mapping = json.loads((directory / "source_data_map.json").read_text(encoding="utf-8"))
    workbook = openpyxl.load_workbook(directory / "Source_Data.xlsx", read_only=True, data_only=True)
    expected = {row["workbook_sheet"]: row["final_csv"] for row in mapping}
    if len(expected) != len(mapping) or set(workbook.sheetnames) != set(expected):
        errors.append("Source Data workbook sheets do not match the unique final source map")
    for sheet, filename in expected.items():
        with (directory / "source_data" / filename).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        cells = list(workbook[sheet].values)
        headers = tuple(None if value == "" else value for value in rows[0])
        if not cells or tuple(cells[0]) != headers or len(cells) != len(rows):
            errors.append(f"{sheet}: source headers or row count differ from {filename}")
            continue
        mismatch = False
        for i, (saved, row) in enumerate(zip(cells[1:], rows[1:]), 2):
            for a, b in zip(saved, row):
                equal = (a is None and b == "") or a == b
                if isinstance(a, (int, float)):
                    try:
                        equal = bool(np.isclose(a, float(b), atol=1e-11, rtol=1e-13))
                    except ValueError:
                        equal = False
                if not equal:
                    errors.append(f"{sheet} row {i}: workbook value differs from {filename}")
                    mismatch = True
                    break
            if mismatch:
                break
    workbook.close()
    return len(expected)


def compare_trees(reference, current, errors, *, scientific=False):
    def inventory(root):
        return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*") if p.is_file() and
                (not scientific or p.relative_to(root).parts[0] not in ("presentation", "natchem_presentation"))}
    before, after = inventory(reference), inventory(current)
    if before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        errors.append(f"{'Scientific baseline' if scientific else 'Independent reproduction'} byte mismatch: {changed}")
    return len(before)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "output")
    parser.add_argument("--presentation", type=Path)
    parser.add_argument("--package", type=Path, default=ROOT)
    parser.add_argument("--only", nargs="+", choices=NAMES, help="Check an explicitly partial rendering preview")
    parser.add_argument("--manuscript", type=Path, help="Require main/ED panel titles to occur verbatim in the manuscript")
    parser.add_argument("--supplementary", type=Path, help="SI source required to check SupplementaryFig1 titles when --manuscript is supplied")
    parser.add_argument("--baseline", type=Path, help="Earlier output tree; scientific outputs must remain byte-identical")
    parser.add_argument("--reproduction", type=Path, help="Independent presentation tree; all files must be byte-identical")
    parser.add_argument("--report", type=Path, help="Optional external JSON report; do not put it in an output tree being compared")
    args = parser.parse_args()
    directory = args.presentation or args.results / "natchem_presentation"
    errors, report = [], {"partial_preview": bool(args.only), "limitations": [
        "Automated text geometry does not certify leader-line routing or visual aesthetics; rendered pages require visual review.",
        "Ordinary text must be 8 or 9 pt; exceptions are 6 pt chemical scripts and 70%-size N geometrically bound below S and Ar."]}
    source = directory / "source_data"
    points, summary, aggregate = [read(source / name) for name in (
        "Figure2a_cloud_points.csv", "Figure2a_cloud_summary.csv", "Figure2a_aggregate_summary.csv")]
    failures, report["clouds"] = cloud_checks(args.results, points, summary, aggregate)
    errors.extend(failures)
    controls = {}
    replaced = summary.copy()
    replaced["rho_pooled"] = aggregate.estimate.to_numpy()
    controls["aggregate_substitution_rejected"] = bool(cloud_checks(args.results, points, replaced, aggregate)[0])
    controls["missing_membership_rejected"] = bool(cloud_checks(args.results, points.iloc[1:], summary, aggregate)[0])
    relabelled = points.copy()
    relabelled.loc[relabelled.index[0], "SMILES"] = "invalid-molecule-label"
    controls["wrong_molecule_identity_rejected"] = bool(cloud_checks(args.results, relabelled, summary, aggregate)[0])
    wrong = aggregate.copy()
    wrong.loc[wrong.category.eq("reactions"), "estimate"] = .15
    controls["old_015_estimate_rejected"] = bool(cloud_checks(args.results, points, summary, wrong)[0])
    report["negative_controls"] = controls
    if not all(controls.values()):
        errors.append("A scientific negative control was incorrectly accepted")
    wells = read(args.package / "data/experimental_wells.csv")
    measured = wells.loc[wells.measurement_status.isin(["quantified", "zero_by_rule"])]
    actual = (len(measured), measured.rxn_class.nunique(),
              measured.rxn_class.replace({"Buchwald-Hartwig(CMC)": "Buchwald-Hartwig", "Suzuki Bpin": "Suzuki"}).nunique(),
              len(set(measured.electrophile_smiles.dropna()) | set(measured.nucleophile_smiles.dropna())))
    archive = read(source / "Figure1_archive_counts.csv").iloc[0]
    if actual != tuple(archive[["experiments", "arrays", "transformations", "substrates"]]) or actual != (172915, 16, 14, 1042):
        errors.append("Figure 1 archive scope differs from the measured-well dataset or approved R5 scope")
    report["archive"] = dict(zip(("experiments", "arrays", "transformations", "substrates"), actual))
    titles = json.loads((directory / "panel_titles.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "figure_qa.json").read_text(encoding="utf-8"))
    names = args.only or NAMES
    if not args.only and (set(titles) != set(NAMES) or set(qa) != set(NAMES)):
        errors.append("Full R5 title/QA inventories must contain five main figures, seven ED figures and one SI figure")
    report["figures"] = figure_checks(directory, names, titles, qa, errors)
    if args.manuscript:
        manuscript = normalize(args.manuscript.read_text(encoding="utf-8"))
        for name in names:
            if name.startswith("Supplementary"):
                continue
            for title in titles[name]:
                if normalize(title) not in manuscript:
                    errors.append(f"{name}: panel title is absent from manuscript/caption: {title}")
        if "SupplementaryFig1" in names and not args.supplementary:
            errors.append("SupplementaryFig1: --supplementary PATH is required alongside --manuscript for a full R5 title check; provide the SI Markdown source")
    if args.supplementary and "SupplementaryFig1" in names:
        supplementary = normalize(args.supplementary.read_text(encoding="utf-8"))
        for title in titles["SupplementaryFig1"]:
            if normalize(title) not in supplementary:
                errors.append(f"SupplementaryFig1: panel title is absent from SI/caption: {title}")
    report["workbook_sheets"] = workbook_checks(directory, errors)
    if args.baseline:
        report["baseline_files_compared"] = compare_trees(args.baseline, args.results, errors, scientific=True)
    if args.reproduction:
        report["reproduction_files_compared"] = compare_trees(args.reproduction, directory, errors)
    report["errors"], report["passed"] = errors, not errors
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
