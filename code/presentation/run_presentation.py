"""Panel tables, the Source Data of the Article figures, Supplementary Tables 1-11, Extended Data Table 1,
the figures and the Source Data workbook.

Runs after the analysis steps. It performs only deterministic display and counting transformations and
rendering; it never fits or resamples a model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from presentation_io import ROOT, OUTPUT, SOURCES, TABLES, BINDINGS


def dependencies():
    paths = set(BINDINGS.values())
    # The primary prediction file is produced by core; the alternate spelling
    # is documented for the original transform but is not a separate requirement.
    paths.discard("output/core/pair_holdout_predictions_bba.csv")
    paths.update(("data/experimental_wells.csv", "data/metadata/analysis_well_membership.csv", "code/assets/Figure1a_schemes.json"))
    paths.update(f"output/core/{prefix}_conditions_bba.csv" for prefix in ("extended", "holdout"))
    paths.update(f"output/partner_profiles/{name}" for name in ("A1_first_set.csv", "A2_salts.csv", "A3_second_set.csv"))
    paths.add("output/sensitivity/structure_for_selection/Figure3d_selection_without_record.csv")
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--sources-only", action="store_true", help="Build source CSVs and tables without rendering")
    stages.add_argument("--plots-only", action="store_true", help="Render the panel tables and Source Data already built")
    args = parser.parse_args()
    missing = [path for path in dependencies() if not (ROOT / path).is_file()] if not args.plots_only else []
    if missing:
        raise SystemExit("Missing outputs of earlier steps:\n" + "\n".join(missing))
    if not args.plots_only:
        for directory in (SOURCES, TABLES):
            if directory.exists() and any(directory.iterdir()):
                raise RuntimeError(f"Presentation output must start empty: {directory.relative_to(ROOT)}")
        from build_sources import build_source_data
        from build_tables import build_tables
        from article_sources import build_article_sources
        build_source_data()
        build_tables()
        build_article_sources()
    if not args.sources_only:
        import plot_figures
        plot_figures.OUT.mkdir(parents=True, exist_ok=True)
        plot_figures.plot_figures()
        import figures_2_3
        import figures_1_4
        figures_2_3.main()
        figures_1_4.main()
    if not args.plots_only:
        import source_data_workbook
        source_data_workbook.build(source_data_workbook.sources(OUTPUT / "source_data", TABLES), OUTPUT / "Source_Data.xlsx")
    files = sum(1 for path in OUTPUT.rglob("*") if path.is_file())
    print(f"Presentation complete: {files} files in {OUTPUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
