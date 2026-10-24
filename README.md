# Structural preferences, not yield rankings, choose untested aryl halides: data and code

Experimental data and analysis code for the study of building-block contributions and structural profiles across coupling reactions (Lin Tan, Yaxian Kong, Chan Zhu, Kai Xue and Kuangbiao Liao).

## Contents

| Folder | Contents |
|---|---|
| `data/` | 175,021 wells from the analysed designs: 172,915 measured wells and 2,106 wells with missing yields; metadata define the analysis sets |
| `code/` | Analysis and figure-generation code |
| `data/raw_corpus/` | Twenty reaction tables and two condition/compound dictionaries used for corpus reconstruction and the all-condition analysis |
| `source_data/` | `Source_Data.xlsx`: 42 sheets underlying Figures 1-5, Extended Data Figs 1-7, Extended Data Table 1 and Supplementary Fig. 1, arranged by panel |

## Data

The source corpus contains 410,983 records in 20 tables; the supplied tables contain 410,587 retained records. After excluding Finkelstein halide exchange, 406,909 records in 19 coupling arrays spanning 17 transformations remain. This study uses 16 arrays spanning 14 transformations: 15 arrays contribute the primary and second condition sets, and a second Buchwald-Hartwig campaign provides the additional campaign.

For each primary design, measured wells passing the platform check are used to order conditions by the number of substrate pairs they cover, with ties broken by condition label. The twenty best-covered conditions are retained, followed by pairs measured under at least ten of those conditions and their largest connected component. The design must contain at least 50 pairs and ten molecules on each side. The same rule selects the second set from the remaining conditions. Repeated records for a substrate pair under the same condition are excluded. The additional campaign includes all wells passing the platform check. Allylic substitution, Minisci and Pd carbonylation do not meet the design thresholds.

`data/experimental_wells.csv` has one row per well:

| Column | Content |
|---|---|
| `well_id` | Well identifier |
| `name0` | Original measurement identifier |
| `component` | `existing_release`: primary and second condition sets; `additional_campaign`: second Buchwald-Hartwig campaign |
| `rxn_class` | Array label; the two Buchwald-Hartwig campaigns and the Suzuki and Suzuki Bpin arrays have separate labels |
| `condition_id` | Reaction condition; identifiers are specific to each array |
| `electrophile_smiles`, `nucleophile_smiles` | The two substrates |
| `product_smiles` | Product |
| `plate` | Recorded plate label |
| `measurement_status` | `quantified`, `zero_by_rule`, `detected_not_quantified` or `no_screen_record`, defined below |
| `yield` | Yield as a fraction from 0 to 1; missing yields are empty |

Every well was first screened for product by DESI-TOF-MS. Wells with product, and one to three wells without product as a check, were quantified by UPLC-UV/MS. Numeric yields have status `quantified`. A blank yield with `yield_clf = 0` has status `zero_by_rule` and yield 0. A blank yield with `yield_clf = 1` has status `detected_not_quantified`; a blank yield without a screening flag has status `no_screen_record`. These last two statuses retain empty yields and are excluded from yield analyses. Missing yields are never filled with zero.

| Analysis set | Quantified | Zero by rule | Measured total | Detected, not quantified | No screen record | Exported rows |
|---|---:|---:|---:|---:|---:|---:|
| Primary | 74,950 | 5,223 | 80,173 | 603 | 327 | 81,103 |
| Second condition set | 71,270 | 3,842 | 75,112 | 590 | 398 | 76,100 |
| Additional Buchwald-Hartwig campaign | 17,574 | 56 | 17,630 | 188 | 0 | 17,818 |
| Total | 163,794 | 9,121 | 172,915 | 1,381 | 725 | 175,021 |

The primary designs contain 4,258 substrate pairs and 300 condition recipes; the second condition set contains 4,091 pairs and 300 further condition recipes. The additional campaign contains 285 conditions and 215 exported pairs, of which 214 have measured yields and 213 belong to the connected component used for fitting molecular contributions.

`data/metadata/` defines the analyses:

| File | Content |
|---|---|
| `analysis_well_membership.csv` | The analysis set of every exported well |
| `base_condition_roles.csv` | 600 primary and second-set condition recipes: coverage rank, reagents, solvent, temperature and time |
| `base_pair_roles.csv` | Substrate pairs of each primary and second-set design |
| `extended_conditions.csv` | The 285 conditions of the additional campaign: reagents, amounts, solvent, temperature and time |
| `calibration_cells.json` | Interval-calibration simulation settings |
| `EXCLUDE_wells.csv` | Measurement identifiers excluded from all analyses |
| `exclusion_processing.json`, `corpus_accounting.csv` | Corpus and analysis membership counts |

## Reproduce

Use Python 3.13.3 and install the pinned dependencies in a fresh virtual environment. From this folder:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -B run_analysis.py --all
```

On macOS or Linux, use `.venv/bin/python` in place of `.venv/Scripts/python`.

`python -B run_analysis.py` runs the core, interval, structural-profile, independent and sensitivity analyses, then generates the tables, figures and source data. `--all` additionally runs the cross-reaction heterogeneity test and simulation-based interval calibration. Calibration uses eight processes by default; `--workers N` changes that number. Random draws use fixed seeds. Each run replaces its generated output folders.

The all-condition analysis is included in the standard run. Its `corpus_reconciliation.csv` gives per-array counts; `estimand_domains.csv`, `fit_domains.csv` and `agreement_cells.csv` give comparison domains, design-matrix ranks and agreement estimates. It can also be run separately with `python -B code/dataset/analyse_broad_domain.py`; it reads the full corpus in `data/raw_corpus/` and writes `output/broad_domain/`. Its summaries are unweighted medians over identifiable comparisons; the shared-count-weighted cross-transformation mean is a separate quantity. Rebuild the analysed dataset from those same inputs with `python -B code/dataset/build_dataset.py data/raw_corpus . --rederive`.

`output/independent/` contains a second, separately written implementation of the main estimates; it reproduces them to numerical precision.

## Outputs

The command generates `output/`:

| Folder | Contents |
|---|---|
| `output/core`, `output/core_dense`, `output/core_intervals` | Molecular contributions, agreement across condition halves, partner halves and reactions, held-out pair prediction, second condition set and intervals |
| `output/extended`, `output/extended_intervals` | Additional Buchwald-Hartwig campaign and structural profiles |
| `output/reaction_readings` | Aryl-halide feature matrix, reagent-derived reaction order, paired feature contrasts between condition sets, and the illustrated methyl 2-bromoisonicotinate contributions and ranks |
| `output/partner_profiles` | Structural profiles across partner halves |
| `output/independent` | Independent implementation of principal core estimates |
| `output/sensitivity` | Sensitivity analyses and further values used in the Article and Supplementary Information |
| `output/broad_domain` | Corpus reconciliation, all-condition fit domains and agreement across conditions, partners and transformations |
| `output/presentation` | Supplementary Tables 1-11 (`tables`), Extended Data Table 1, intermediate panel source data (`source_data`) and the intermediate `Source_Data.xlsx` |
| `output/natchem_presentation` | Final Figures 1-5, Extended Data Figs 1-7, Supplementary Fig. 1, panel source data and the final 42-sheet `Source_Data.xlsx` |
| `output/heterogeneity`, `output/calibration` | Additional analyses run by `--all` |

`output/calibration/validation_gated_bba.csv` contains the interval-calibration results for all 112 simulation settings.

The supplied `source_data/Source_Data.xlsx` contains 42 sheets. The final figures and their panel data are in `output/natchem_presentation/figures/` and `output/natchem_presentation/source_data/`. After the analysis stages, `code/presentation/finalize_nature_chemistry.py` generates the display items below; `source_data_map.json` maps each panel CSV to its workbook sheet.

| Display item | Sheets in `Source_Data.xlsx` |
|---|---|
| Figure 1 | `Fig1_archive`, `Fig1_arrays` |
| Figure 2 | `Fig2*`, `F2*` |
| Figure 3 | `Fig3*` |
| Figure 4 | `F4*` |
| Figure 5 | `F5*` |
| Extended Data Fig. 1 | `ED1_*` |
| Extended Data Fig. 2 | `ED2_*` |
| Extended Data Fig. 3 | `ED3*` |
| Extended Data Fig. 4 | `ED4_*` |
| Extended Data Fig. 5 | `ED5*` |
| Extended Data Fig. 6 | `ED6*` |
| Extended Data Fig. 7 | `ED7*` |
| Extended Data Table 1 | `ExtendedDataTable1` |
| Supplementary Fig. 1 | `SI1*` |

An asterisk matches the remaining characters in a sheet name.

`code/assets/ChemDraw/` contains editable CDXML masters and their native ChemDraw SVG exports, based on the existing Nature-style settings. All structures and reaction schemes use 10.8 pt bonds and Arial 8 pt atom labels; chemically necessary subscripts retain their native smaller glyphs. Figure code places these vectors at their recorded physical sizes. ChemDraw is not required to reproduce the figures from the supplied assets.

## Citation

Please cite the Article and the Zenodo record https://doi.org/10.5281/zenodo.23583744.

## Data note

A subset of the amide-coupling measurements was previously reported by Zhang et al. (*Chem. Sci.* **16**, 11809-11822, 2025).
