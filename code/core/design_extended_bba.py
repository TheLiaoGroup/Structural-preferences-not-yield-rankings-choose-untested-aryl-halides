"""The design rule of the analysed reactions."""
import itertools

import numpy as np

import pandas as pd

import lib_bba as L

ALL_SIDES = {
    "Acylation": (2, 1), "Allylic-substitution": (1, 2), "Buchwald-Hartwig": (2, 1),
    "Buchwald-Hartwig(CMC)": (2, 1), "C-O competitive reactions": (2, 1), "Chan-Lam": (2, 1),
    "Esterification": (2, 1), "Grignard": (2, 1), "Heck": (2, 1), "Hiyama": (2, 1),
    "Kumada": (1, 2), "Minisci": (1, 2), "Pd-carbonylation": (1, 2),
    "Reductive-amination": (1, 2), "SNAr": (2, 1), "Sonogashira": (2, 1), "Suzuki": (1, 2),
    "Suzuki Bpin": (1, 2), "Ullmann": (2, 1),
}

SAME_REACTION = {frozenset(("Buchwald-Hartwig", "Buchwald-Hartwig(CMC)")),
                 frozenset(("Suzuki", "Suzuki Bpin"))}

MIN_PAIRS = 50

MIN_MOLECULES = 10

MIN_SHARED = 5

def design_for(rows):
    """The design rule applied to one reaction: twenty best-covered
    conditions, pairs measured at ten or more, largest connected piece."""
    cov = (rows.drop_duplicates(["condition_id", "elec", "nuc"]).groupby("condition_id").size()
               .rename("n").reset_index().sort_values(["n", "condition_id"], ascending=[False, True]))
    keep = cov.head(L.N_CONDITIONS)
    sub = rows[rows.condition_id.isin(keep.condition_id)]
    counted = (sub.groupby(["elec", "nuc"]).condition_id.nunique().rename("n").reset_index())
    counted = counted[counted.n >= L.MIN_CONDITIONS]
    if len(counted) < 3:
        return counted.iloc[0:0], keep
    return L.largest_component(counted), keep
