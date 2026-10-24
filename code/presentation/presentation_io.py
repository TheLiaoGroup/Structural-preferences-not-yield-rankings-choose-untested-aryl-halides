"""Paths of the analysis outputs read by the presentation step."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/presentation"
SOURCES = OUTPUT / "panel_data"  # panel tables; the Source Data of the Article figures go to source_data
TABLES = OUTPUT / "tables"
BINDINGS = {'full_data_pair_effects_bba.csv': 'output/core/full_data_pair_effects_bba.csv', 'full_data_effects_bba.csv': 'output/core/full_data_effects_bba.csv', 'gated_A_cells_bba.csv': 'output/core/gated_A_cells_bba.csv', 'extended_transfer_bba.csv': 'output/core/extended_transfer_bba.csv', 'gated_A_prediction_bba.csv': 'output/core/gated_A_prediction_bba.csv', 'amide_heldout_pair_predictions.csv': 'output/core/amide_heldout_pair_predictions.csv', 'pair_holdout_predictions_bba.csv': 'output/core/pair_holdout_predictions_bba.csv', 'gated_B_relative_k2_bba.csv': 'output/core/gated_B_relative_k2_bba.csv', 'gated_B_relative_k3_bba.csv': 'output/core/gated_B_relative_k3_bba.csv', 'gated_B_pooled_k2_bba.csv': 'output/core/gated_B_pooled_k2_bba.csv', 'gated_B_pooled_k3_bba.csv': 'output/core/gated_B_pooled_k3_bba.csv', 'selection_vs_agreement.csv': 'output/core/selection_vs_agreement.csv', 'shared_amines_BH_SNAr.csv': 'output/core/shared_amines_BH_SNAr.csv', 'extended_reliability_bba.csv': 'output/core/extended_reliability_bba.csv', 'gated_B_secondary_bba.csv': 'output/core/gated_B_secondary_bba.csv', 'holdout_transfer_bba.csv': 'output/core/holdout_transfer_bba.csv', 'diagnose_BuchwaldHartwig_SNAr_nuc_bba.csv': 'output/core/diagnose_BuchwaldHartwig_SNAr_nuc_bba.csv', 'holdout_molecule_effects_bba.csv': 'output/core/holdout_molecule_effects_bba.csv', 'results_bba.csv': 'output/core/results_bba.csv', 'residual_spectrum_bba.csv': 'output/core/residual_spectrum_bba.csv', 'holdout_cells_bba.csv': 'output/core/holdout_cells_bba.csv', 'reaction_labels.csv': 'output/core/reaction_labels.csv', 'amine_class_contrasts.csv': 'output/core/amine_class_contrasts.csv', 'molecule_contributions.csv': 'output/core/molecule_contributions.csv', 'holdout_summary_bba.csv': 'output/core_intervals/holdout_summary_bba.csv', 'extended_pooled_bba.csv': 'output/core_intervals/extended_pooled_bba.csv', 'structure_profiles.csv': 'output/extended/structure_profiles.csv', 'structure_contrasts.csv': 'output/extended/structure_contrasts.csv', 'PUBLIC_BOOTSTRAP_RESULTS.csv': 'output/extended_intervals/PUBLIC_BOOTSTRAP_RESULTS.csv', 'campaign_comparisons.csv': 'output/extended/campaign_comparisons.csv', 'extended_design_bba.csv': 'output/core/design_counts.csv'}

def path_for(name):
    path = ROOT / BINDINGS[name]
    if not path.is_file():
        raise FileNotFoundError(f"Output of an earlier step required: {path.relative_to(ROOT)}")
    return path

def read_csv(path, **kwargs):
    path = Path(path).resolve()
    return pd.read_csv(path, **kwargs)

def read_upstream(*names):
    for name in names:
        if name in BINDINGS:
            path = path_for(name)
            return read_csv(path), path
    raise KeyError(f"Unbound scientific producer: {names}")
