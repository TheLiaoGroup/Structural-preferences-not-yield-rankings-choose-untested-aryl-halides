"""Shared helpers: molecule identity, connected sets and the design selection."""
import datetime

import hashlib

import os

import sys

from pathlib import Path

import numpy as np

import pandas as pd

import rdkit

from rdkit import Chem, RDLogger

from rdkit.Chem.MolStandardize import rdMolStandardize

from scipy.sparse import coo_matrix

from scipy.sparse.csgraph import connected_components

RDLogger.DisableLog("rdApp.*")

REACTIONS = ("Acylation", "Buchwald-Hartwig", "Chan-Lam", "Heck", "SNAr", "Suzuki")

SIDES = {"Acylation": (2, 1), "Buchwald-Hartwig": (2, 1), "Chan-Lam": (2, 1),
         "Heck": (2, 1), "SNAr": (2, 1), "Suzuki": (1, 2)}

N_CONDITIONS = 20

MIN_CONDITIONS = 10

REAGENT_COLUMNS = [f"reagent{i}_id" for i in range(1, 7)]

_TAUTOMER = rdMolStandardize.TautomerEnumerator()

_TAUTOMER.SetRemoveSp3Stereo(False)

_TAUTOMER.SetRemoveBondStereo(False)

def canonical(smiles):
    """Molecule identity: RDKit canonical tautomer with stereochemistry retained."""
    return Chem.MolToSmiles(_TAUTOMER.Canonicalize(Chem.MolFromSmiles(smiles)))

def coverage_order(rows):
    """Conditions of each reaction in selection order: most distinct substrate pairs first,
    ties broken by the condition identifier that sorts first as a character string."""
    cov = (rows.drop_duplicates(["rxn_class", "condition_id", "elec", "nuc"])
               .groupby(["rxn_class", "condition_id"]).size().rename("n_pairs").reset_index())
    cov = cov.sort_values(["rxn_class", "n_pairs", "condition_id"], ascending=[True, False, True])
    cov["rank"] = cov.groupby("rxn_class").cumcount() + 1
    return cov.reset_index(drop=True)

def component_labels(left, right):
    """Connected piece each edge of a two-way layout belongs to."""
    ia, ib = pd.factorize(np.asarray(left))[0], pd.factorize(np.asarray(right))[0]
    n_left = ia.max() + 1
    edges = (np.ones(len(ia)), (ia, ib + n_left))
    labels = connected_components(coo_matrix(edges, shape=(n_left + ib.max() + 1,) * 2),
                                  directed=False)[1]
    return labels[ia]

def largest_component(pairs):
    """Keep the pairs in the largest connected piece of one reaction's electrophile-nucleophile
    graph; main effects are identified only within a connected piece."""
    labels = component_labels(pairs.elec, pairs.nuc)
    return pairs[labels == np.bincount(labels).argmax()]

def interaction_df(pairs):
    """Residual degrees of freedom of the additive model on a connected set of pairs."""
    return len(pairs) - (pairs.elec.nunique() + pairs.nuc.nunique() - 1)

def select_design(rows):
    """The frozen design: the N_CONDITIONS best-covered conditions of each reaction, then the
    pairs measured at MIN_CONDITIONS or more of them, restricted to the largest component."""
    order = coverage_order(rows)
    conditions = order[order["rank"] <= N_CONDITIONS]
    kept = rows.merge(conditions[["rxn_class", "condition_id"]], on=["rxn_class", "condition_id"])
    counted = (kept.groupby(["rxn_class", "elec", "nuc"]).condition_id.nunique()
                   .rename("n_conditions").reset_index())
    counted = counted[counted.n_conditions >= MIN_CONDITIONS]
    pairs = pd.concat([largest_component(g) for _, g in counted.groupby("rxn_class")],
                      ignore_index=True)
    return conditions, pairs

def apply_design(rows, conditions, pairs):
    """Rows at the frozen conditions and pairs."""
    out = rows.merge(conditions[["rxn_class", "condition_id"]], on=["rxn_class", "condition_id"])
    return out.merge(pairs[["rxn_class", "elec", "nuc"]], on=["rxn_class", "elec", "nuc"])

WORK = Path(__file__).resolve().parent
OUT = WORK.parents[1] / "output" / "core"
LOGS = OUT
