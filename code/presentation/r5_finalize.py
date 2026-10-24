"""Generate the five-figure Article and its final numbered panel sources."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import r4_figures as R4
import r5_evidence as E
import r5_extended as X
import r5_figure2 as F2
import r5_overview as O
import r5_presentation as P
import r5_style as S
import source_data_workbook as W


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=P.ROOT / "output")
    parser.add_argument("--out", type=Path, default=P.ROOT / "output/natchem_presentation")
    parser.add_argument("--readings", type=Path)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    S.style()
    p = P.Presentation(args.results, args.out, args.readings)
    p.titles = {"Figure1": [O.TITLE], "Figure2": F2.TITLES, **E.TITLES}
    p.titles.update({f"ExtendedDataFig{new}": R4.TITLES[f"ExtendedDataFig{old}"] for old, new in P.ED_MAP.items()})
    p.titles["SupplementaryFig1"] = R4.TITLES["SupplementaryFig1"]
    sheets = p.sources()
    p.out.joinpath("figures").mkdir(parents=True, exist_ok=True)
    # The existing ED producers supply unchanged statistical data; only final style and locations change.
    S.BF.finalize = S.finalize
    for module in (R4, R4.F14, R4.F23, S.BF):
        for name, colour in (("BLUE", S.PURPLE), ("ORANGE", S.GREY), ("TEAL", S.TEAL), ("RED", S.GREY), ("PURPLE", S.PURPLE)):
            setattr(module, name, colour)
    makers = [("Figure2", lambda: F2.render(p)), ("Figure3", lambda: E.figure3(p)),
              ("Figure4", lambda: E.figure4(p)), ("Figure5", lambda: E.figure5(p))]
    makers.extend((name, lambda fn=fn: fn(p)) for name, fn in X.MAKERS.items())
    if not args.only or "Figure1" in args.only:
        overview = O.draw(p)
        p.qa["Figure1"].update(O.render(overview))
        print("Figure1: editable SVG scene and native ChemDraw vectors", flush=True)
    panel_geometry = {"revision": "R7"}
    for name, maker in makers:
        if args.only and name not in args.only:
            continue
        figure = maker()
        p.export(name, figure)
        if name in {"Figure2", "Figure5"}:
            producer = F2 if name == "Figure2" else E
            panel_geometry[name] = producer.geometry_metadata(figure, p.out / "figures" / f"{name}.svg")
        print(f"{name}: final physical size and panel titles checked", flush=True)
    (p.out / "r7_panel_geometry.json").write_text(json.dumps(panel_geometry, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    W.build(sheets, p.out / "Source_Data.xlsx")
    mapping = [{"producer": producer, "original_csv": old, "final_csv": final, "workbook_sheet": sheet}
               for producer, old, final, sheet in P.SOURCE_MAP]
    mapping += [{"producer": "r5_presentation", "final_csv": name, "workbook_sheet": sheet}
                for name, sheet in P.final_sheets().items() if name not in {row[2] for row in P.SOURCE_MAP}]
    for filename, value in (("source_data_map.json", mapping), ("panel_titles.json", p.titles), ("figure_qa.json", p.qa)):
        (p.out / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"R7: {len(sheets)} Source Data sheets; five main figures and seven Extended Data figures", flush=True)


if __name__ == "__main__":
    main()
