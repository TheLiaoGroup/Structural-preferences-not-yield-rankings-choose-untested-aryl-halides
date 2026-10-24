"""Counterexamples for the Figure 3 numerical summaries.

Run with python -B -m unittest discover -s code/extended -p 'test_*.py'.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reaction_readings as R


def contrasts(reactions=("Alpha", "Beta", "Gamma")):
    rows = []
    for reaction in reactions:
        for i, feature in enumerate(R.FEATURES[:5]):
            for data in (R.FIRST, R.SECOND):
                rows.append({"reaction": reaction, "feature": feature, "data": data,
                             "metal": "Pd", "with_n": 3, "without_n": 4,
                             "with_mean_pp": float(i), "without_mean_pp": 0.0,
                             "difference_pp": float(i), "mann_whitney_p": 0.1, "bh_q": 0.2})
    return pd.DataFrame(rows)


class MetalOrderTests(unittest.TestCase):
    def setUp(self):
        self.table = contrasts()
        self.labels = pd.DataFrame({"metal": ["Cu", "Pd", "Pd"], "electrophile": [R.ARYL] * 3},
                                   index=["Alpha", "Beta", "Gamma"])
        self.conditions = pd.DataFrame([
            {"rxn_class": reaction, "condition_id": str(i), "design": "twenty conditions",
             "set": "analysed", "reagents": "CuI" if reaction == "Alpha" else "PEPPSI"}
            for reaction in self.labels.index for i in range(20)
        ])

    def test_grouping_reads_reagents_and_is_invariant_to_input_order(self):
        order = R.reaction_order(self.table.sample(frac=1, random_state=3), self.labels,
                                 self.conditions.sample(frac=1, random_state=4))
        self.assertEqual(order.reaction.tolist(), ["Beta", "Gamma", "Alpha"])
        self.assertEqual(order.metal.tolist(), ["Pd", "Pd", "Cu"])
        self.assertEqual(order.metal_group_order.tolist(), [1, 1, 2])
        self.assertEqual(order.row_order.tolist(), [1, 2, 3])

    def test_disagreement_with_reagent_derived_metal_is_rejected(self):
        self.labels.loc["Alpha", "metal"] = "Pd"
        with self.assertRaisesRegex(RuntimeError, "metal.*Alpha"):
            R.reaction_order(self.table, self.labels, self.conditions)

    def test_underpowered_profile_is_excluded(self):
        reduced = self.table[~((self.table.reaction == "Gamma") &
                               (self.table.feature == R.FEATURES[4]))]
        order = R.reaction_order(reduced, self.labels, self.conditions)
        self.assertEqual(set(order.reaction), {"Alpha", "Beta"})


class PairedCellTests(unittest.TestCase):
    def setUp(self):
        self.order = pd.DataFrame([{"reaction": "Example", "metal": "Pd", "row_order": 1,
                                    "metal_group_order": 1}])
        self.table = contrasts(("Example",))
        values = [(1.0, 2.0), (-2.0, -4.0), (0.0, 0.0), (3.0, -1.0), (9.0, 8.0)]
        for feature, (first, second) in zip(R.FEATURES, values):
            self.table.loc[(self.table.feature == feature) & (self.table.data == R.FIRST), "difference_pp"] = first
            self.table.loc[(self.table.feature == feature) & (self.table.data == R.SECOND), "difference_pp"] = second
        self.table = self.table[~((self.table.feature == R.FEATURES[4]) & (self.table.data == R.SECOND))]

    def test_only_complete_pairs_enter_summary_and_untested_cells_stay_missing(self):
        matrix, paired, summary = R.feature_tables(self.table.sample(frac=1, random_state=7), self.order)
        self.assertEqual(len(matrix), 8)
        self.assertEqual(int(matrix.tested.sum()), 5)
        self.assertTrue(matrix.loc[~matrix.tested, "difference_pp"].isna().all())
        self.assertEqual(len(paired), 4)
        self.assertEqual(paired.same_sign.tolist(), [True, True, True, False])
        self.assertEqual(summary.iloc[0].paired_cells, 4)
        self.assertEqual(summary.iloc[0].same_sign_cells, 3)
        self.assertEqual(summary.iloc[0].same_sign_fraction, 0.75)
        self.assertAlmostEqual(summary.iloc[0].pearson_r, 17 / np.sqrt(975))
        self.assertEqual(paired.first_difference_pp.tolist(), [1.0, -2.0, 0.0, 3.0])
        self.assertEqual(paired.second_difference_pp.tolist(), [2.0, -4.0, 0.0, -1.0])

    def test_duplicate_reaction_feature_is_rejected(self):
        duplicate = pd.concat([self.table, self.table.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            R.feature_tables(duplicate, self.order)


class MoleculeRankTests(unittest.TestCase):
    def test_rank_descends_by_full_precision_then_breaks_ties_by_smiles(self):
        frame = pd.DataFrame([
            {"reaction": "Example", "side": "elec", "SMILES": "Z", "contribution_pp": 5.0},
            {"reaction": "Example", "side": "elec", "SMILES": R.MOLECULE, "contribution_pp": 5.0},
            {"reaction": "Example", "side": "elec", "SMILES": "A", "contribution_pp": 5.0},
            {"reaction": "Example", "side": "nuc", "SMILES": "X", "contribution_pp": 9.0},
        ])
        ranked = R.molecule_readings(frame, reactions=("Example",))
        self.assertEqual(ranked.iloc[0]["rank"], 2)
        self.assertEqual(ranked.iloc[0].total_molecules, 3)
        frame.loc[frame.SMILES == "Z", "contribution_pp"] = 5.0000000001
        changed = R.molecule_readings(frame.sample(frac=1, random_state=2), reactions=("Example",))
        self.assertEqual(changed.iloc[0]["rank"], 3)

    def test_duplicate_molecule_is_rejected_before_ranking(self):
        frame = pd.DataFrame([{"reaction": "Example", "side": "elec", "SMILES": R.MOLECULE,
                               "contribution_pp": 1.0}] * 2)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            R.molecule_readings(frame, reactions=("Example",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
