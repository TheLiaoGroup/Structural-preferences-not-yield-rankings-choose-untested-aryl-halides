"""Source Data workbook: one sheet per figure panel and one for Extended Data Table 1.

Every cell holds the value of the corresponding source data CSV: a number in canonical form as that number,
written at full precision and shown with the CSV's decimals, and everything else as the CSV text.
"""
import csv
import math
import re
from pathlib import Path

import openpyxl.cell._writer as cell_writer
from openpyxl import Workbook
from openpyxl.compat import safe_string
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
SHEETS = {  # source data file -> sheet, in figure order
    "Figure1a_reactions.csv": "Fig1a",
    "Figure1b_three_changes.csv": "Fig1b",
    "Figure1c_pooled_portability.csv": "Fig1c",
    "Figure2a_portability_by_set.csv": "Fig2a",
    "Figure2b_SNAr_partitions.csv": "Fig2b",
    "Figure2c_structure_effects.csv": "Fig2c",
    "Figure2d_BH_SNAr_amines.csv": "Fig2d_amines",
    "Figure2d_amine_class_summary.csv": "Fig2d_classes",
    "Figure3a_campaign_recognition.csv": "Fig3a",
    "Figure3b_partner_change.csv": "Fig3b",
    "Figure3c_random_grouping.csv": "Fig3c",
    "Figure3d_selection_without_record.csv": "Fig3d",
    "Figure4a_heldout_amide_pairs.csv": "Fig4a_pairs",
    "Figure4a_metrics.csv": "Fig4a_metrics",
    "Figure4b_selection_vs_agreement.csv": "Fig4b",
    "Figure4c_Heck_Suzuki_molecules.csv": "Fig4c",
    "ExtendedDataFig1_condition_composition.csv": "EDFig1_conditions",
    "ExtendedDataFig2_amide_matrix.csv": "EDFig2",
    "ExtendedDataFig3_defined_benchmark.csv": "EDFig3_benchmark",
    "ExtendedDataFig3_pooled_shortfall.csv": "EDFig3_shortfall",
    "ExtendedDataFig4_undefined_benchmark.csv": "EDFig4",
    "ExtendedDataFig5a_leave_one_reaction_out.csv": "EDFig5a",
    "ExtendedDataFig5b_outcome_scale.csv": "EDFig5b",
    "ExtendedDataFig6_reaction_map.csv": "EDFig6",
    "ExtendedDataFig7ab_profile_comparisons.csv": "EDFig7ab",
    "ExtendedDataFig8_prediction_all_reactions.csv": "EDFig8",
    "ExtendedDataTable1.csv": "EDTable1",
}
INTEGER = re.compile(r"-?(0|[1-9]\d*)$")
NUMBER = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?$")


def full_precision(value):
    """openpyxl writes numbers with 16 significant digits; a double can need 17. Write the shortest exact form."""
    return repr(float(value)) if isinstance(value, float) and math.isfinite(value) else safe_string(value)


cell_writer.safe_string = full_precision


def decimals_shown(text):
    """A number format that keeps the CSV's trailing zeros (0.90 stays 0.90); None where Excel shows the text as is."""
    m = re.fullmatch(r"-?\d+\.(\d+)", text)
    return "0." + "0" * len(m.group(1)) if m and text.endswith("0") else None


def build(files, path):
    """Write the workbook; files maps each sheet name to its CSV, in sheet order."""
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.creator = ""
    for sheet, source in files.items():
        with open(source, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        ws = wb.create_sheet(sheet)
        ws.append(rows[0])
        for row in rows[1:]:
            ws.append([None if t == "" else int(t) if INTEGER.match(t) else float(t) if NUMBER.match(t) else t for t in row])
            for c, t in zip(ws[ws.max_row], row):
                if NUMBER.match(t) and decimals_shown(t):
                    c.number_format = decimals_shown(t)
        ws.freeze_panes = "A2"
        for c in ws[1]:
            c.font = Font(bold=True)
        for j, col in enumerate(zip(*rows[:300]), 1):
            ws.column_dimensions[get_column_letter(j)].width = min(60, max(len(t) for t in col) + 2)
    wb.save(path)
    fix_timestamps(Path(path))


def fix_timestamps(path, stamp=(2026, 10, 14, 0, 0, 0)):
    """Give the document properties and every archive entry one fixed date, so the same tables give the same bytes."""
    import zipfile
    tmp = path.with_name(path.name + ".tmp")
    iso = "%04d-%02d-%02dT%02d:%02d:%02dZ" % stamp
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*", lambda m: m.group(1) + iso.encode(), data)
            info = zipfile.ZipInfo(item.filename, date_time=stamp)
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, item.external_attr
            zout.writestr(info, data)
    tmp.replace(path)


def sources(source_data, tables):
    """The CSV of every sheet: the figure source data, and Extended Data Table 1 from the tables."""
    figures = {name for name in SHEETS if name != "ExtendedDataTable1.csv"}
    present = {p.name for p in Path(source_data).glob("*.csv")}
    if present != figures:
        raise RuntimeError(f"Source data files differ from the sheet list: {sorted(present ^ figures)}")
    return {sheet: (Path(tables) if name == "ExtendedDataTable1.csv" else Path(source_data)) / name
            for name, sheet in SHEETS.items()}


if __name__ == "__main__":
    out = ROOT / "output" / "presentation"
    build(sources(out / "source_data", out / "tables"), out / "Source_Data.xlsx")
