"""Prepare release copies of the source tables, excluding authoritative well IDs.

The source folder is read only. Reaction tables retain their file types and
original one-based ``source_row`` values; dictionary files are copied verbatim.
No yields, screening flags, molecule identities or analysis roles are derived.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell


def _source_rows(table: pd.DataFrame, label: str) -> pd.DataFrame:
    table = table.copy()
    if "source_row" not in table:
        table["source_row"] = range(1, len(table) + 1)
    else:
        values = table.source_row.astype(str).str.strip()
        if not values.str.fullmatch(r"[1-9][0-9]*").all() or values.duplicated().any():
            raise ValueError(f"{label}: source_row must contain unique positive integers")
        table["source_row"] = values.map(int)
    return table


def _filter(table: pd.DataFrame, excluded: set[str], label: str):
    if "name0" not in table:
        raise ValueError(f"{label}: missing name0 measurement identifiers")
    table = _source_rows(table, label)
    names = table.name0.astype(str).str.strip()
    matched = names.isin(excluded)
    removed = names[matched].tolist()
    if len(removed) != len(set(removed)):
        raise ValueError(f"{label}: an excluded name0 occurs more than once")
    retained = table.loc[~matched].copy()
    if retained.name0.astype(str).str.strip().isin(excluded).any():
        raise AssertionError(f"{label}: excluded measurements remain")
    return retained, removed


def _write_excel(sheets: dict[str, pd.DataFrame], path: Path, first_sheet: str):
    workbook = Workbook(write_only=True)
    for name, table in sheets.items():
        sheet = workbook.create_sheet(name)
        if name == first_sheet and len(table.columns):
            sheet.append([WriteOnlyCell(sheet, str(column)) for column in table.columns])
        for row in table.itertuples(index=False, name=None):
            cells = []
            for value in row:
                cell = WriteOnlyCell(sheet, value)
                if isinstance(value, str):
                    cell.data_type = "s"
                cells.append(cell)
            sheet.append(cells)
    workbook.save(path)
    workbook.close()


def prepare(raw_dir: Path, dest_dir: Path, exclusion_csv: Path) -> dict:
    raw_dir, dest_dir, exclusion_csv = (Path(p).resolve() for p in (raw_dir, dest_dir, exclusion_csv))
    if raw_dir == dest_dir or raw_dir.is_relative_to(dest_dir) or dest_dir.is_relative_to(raw_dir):
        raise ValueError("Source and destination must be separate, non-nested folders")
    authority_bytes = exclusion_csv.read_bytes()
    authority = pd.read_csv(io.BytesIO(authority_bytes), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "name0" not in authority:
        raise ValueError("Exclusion CSV must contain name0")
    names = authority.name0.str.strip()
    if names.empty or names.eq("").any() or names.duplicated().any():
        raise ValueError("Exclusion name0 values must be nonblank and unique, with at least one ID")
    excluded = set(names)
    sources = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in (".csv", ".xlsx"))
    reactions = [p for p in sources if not p.name.startswith("0_")]
    if not reactions:
        raise ValueError("No reaction source tables found")
    expected_files = {p.name for p in sources} | {"corpus_exclusion_audit.json"}
    if dest_dir.exists():
        unexpected = sorted(p.name for p in dest_dir.iterdir() if p.name not in expected_files)
        if unexpected:
            raise ValueError(f"Destination contains files outside this corpus: {unexpected}")

    audit = {"schema": 1, "exclusion_sha256": hashlib.sha256(authority_bytes).hexdigest(),
             "original_records": 0, "retained_records": 0, "excluded_records": 0,
             "excluded_name0": [], "per_array": []}
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="corpus_prepare_", dir=dest_dir.parent) as temporary:
        stage = Path(temporary)
        for source in sources:
            target = stage / source.name
            if source.name.startswith("0_"):
                shutil.copyfile(source, target)
                if target.read_bytes() != source.read_bytes():
                    raise AssertionError(f"Dictionary copy differs: {source.name}")
                continue
            if source.suffix.lower() == ".csv":
                table = pd.read_csv(source, dtype=str, keep_default_na=False, encoding="utf-8-sig")
                sheets = {"": table}
            else:
                with pd.ExcelFile(source) as workbook:
                    sheets = {name: pd.read_excel(workbook, sheet_name=name, header=0 if i == 0 else None,
                                                   dtype=str, keep_default_na=False)
                              for i, name in enumerate(workbook.sheet_names)}
                table = next(iter(sheets.values()))
            first_sheet = next(iter(sheets))
            if excluded.intersection(str(column).strip() for column in table.columns):
                raise ValueError(f"{source.name}: excluded IDs in measurement column headers")
            retained, removed = _filter(table, excluded, source.name)
            sheets[first_sheet] = retained
            entry = {"array": re.sub(r"(_\d+)?_?\.(csv|xlsx)$", "", source.name, flags=re.IGNORECASE),
                     "source_file": source.name, "original_records": len(table),
                     "retained_records": len(retained), "excluded_records": len(removed),
                     "excluded_name0": removed}
            for name, extra in list(sheets.items())[1:]:
                # Auxiliary sheets retain their original first row, including any field labels.
                matches = extra.apply(lambda column: column.astype(str).str.strip().isin(excluded))
                matched_rows = matches.any(axis=1)
                extra_removed = sorted(set(extra.where(matches).stack().astype(str).str.strip()))
                sheets[name] = extra.loc[~matched_rows].copy()
                entry.setdefault("auxiliary_sheets", []).append(
                    {"sheet": name, "original_records": len(extra),
                     "retained_records": len(sheets[name]), "excluded_records": len(extra) - len(sheets[name]),
                     "excluded_name0": extra_removed})
            if source.suffix.lower() == ".csv":
                retained.to_csv(target, index=False, lineterminator="\n")
            else:
                _write_excel(sheets, target, first_sheet)
            for key in ("original_records", "retained_records", "excluded_records"):
                audit[key] += entry[key]
            audit["excluded_name0"].extend(removed)
            audit["per_array"].append(entry)
        if not audit["excluded_records"]:
            raise ValueError("Exclusion CSV matched zero source measurements")
        if len(audit["excluded_name0"]) != len(set(audit["excluded_name0"])):
            raise ValueError("An excluded name0 occurs in multiple reaction arrays")
        if set(audit["excluded_name0"]) != excluded:
            missing = sorted(excluded - set(audit["excluded_name0"]))
            raise ValueError(f"Exclusion IDs absent from the source corpus: {missing}")
        if audit["original_records"] != audit["retained_records"] + audit["excluded_records"]:
            raise AssertionError("Corpus exclusion counts do not reconcile")
        (stage / "corpus_exclusion_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        metadata = dest_dir.parent / "metadata"
        metadata.mkdir(exist_ok=True)
        (metadata / "EXCLUDE_wells.csv").write_bytes(authority_bytes)
        if (metadata / "EXCLUDE_wells.csv").read_bytes() != authority_bytes:
            raise AssertionError("Copied exclusion CSV differs from the authority")
        dest_dir.mkdir(exist_ok=True)
        for source in stage.iterdir():
            shutil.copyfile(source, dest_dir / source.name)
            if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256((dest_dir / source.name).read_bytes()).digest():
                raise AssertionError(f"Release copy differs: {source.name}")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("dest_dir", type=Path)
    parser.add_argument("exclusion_csv", type=Path)
    args = parser.parse_args()
    audit = prepare(args.raw_dir, args.dest_dir, args.exclusion_csv)
    print(json.dumps({key: audit[key] for key in ("schema", "original_records", "retained_records", "excluded_records")}, indent=2))


if __name__ == "__main__":
    main()
