"""Arylboronic acids used in both Suzuki-Miyaura coupling and Chan-Lam coupling, and the plate of each pair's two condition sets.

The comparisons between reactions (Supplementary Note 10) pair molecules on the same side. Arylboronic acids are the
nucleophile of Suzuki-Miyaura coupling and are recorded as the electrophile of Chan-Lam coupling, so the boronic acids
the two reactions share are never compared there. This script compares them with the estimators of Supplementary
Note 10, unchanged:
  * rank agreement = Spearman correlation of the full-data contributions of the main analysis over the shared molecules, on the
    first condition set (label primary) and on the second (label validation);
  * corrected agreement = that correlation divided by the geometric mean of the two condition-half reliabilities;
  * 95% interval = percentile interval over 2,000 resamples of the shared molecules (seed 20261013);
  * five-membered heteroaryl boronic acids (boron on an aromatic carbon of a five-membered ring) against the others,
    mean contribution difference and two-sided Mann-Whitney test, in each reaction.
It also counts, from the wells, the pairs whose plate is recorded in both condition sets and how many of
them carry both sets on one plate, and, from the condition table, the first-set conditions of the two Suzuki-Miyaura
campaigns that use triphenylphosphine-ligated palladium.

Run from the repository root after the main steps: python -B code/sensitivity/boronic_acids_across_roles.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import mannwhitneyu, spearmanr
from plate_condition_sets import plate_audit, catalyst_counts

sys.dont_write_bytecode = True
REL = Path(__file__).resolve().parents[2]   # the repository root (code, data, output)
OUT = REL / "output" / "sensitivity" / "boronic_acids_across_roles"
CR = REL / "output" / "core"
META = REL / "data" / "metadata"
OUT.mkdir(parents=True, exist_ok=True)
FIVE = Chem.MolFromSmarts("[B;X3]-[c,n,o,s;r5]")


def effects(name, label):
    e = pd.read_csv(CR / name, usecols=["scale", "label", "rxn_class", "side", "mol", "main_effect"])
    e = e[(e.scale == "raw") & (e.label == label)]
    s = e[(e.rxn_class == "Suzuki") & (e.side == "nuc")].set_index("mol").main_effect
    c = e[(e.rxn_class == "Chan-Lam") & (e.side == "elec")].set_index("mol").main_effect
    return s, c


def reliability(name):
    r = pd.read_csv(CR / name).set_index(["rxn_class", "side"]).reliability
    return float(np.sqrt(r[("Suzuki", "nuc")] * r[("Chan-Lam", "elec")]))


rows, members = [], []
for cset, eff, rel in (("first", "full_data_effects_bba.csv", "extended_reliability_bba.csv"),
                       ("second", "holdout_molecule_effects_bba.csv", "holdout_reliability_bba.csv")):
    s, c = effects(eff, "primary" if cset == "first" else "validation")
    shared = sorted(set(s.index) & set(c.index))
    assert len(shared) >= 5, (cset, len(shared))
    a, b = s[shared].to_numpy(), c[shared].to_numpy()
    rho = spearmanr(a, b).statistic
    ceiling = reliability(rel)
    rng = np.random.default_rng(20261013)
    boot = []
    for _ in range(2000):
        k = rng.integers(0, len(shared), len(shared))
        if len(set(k)) >= 5:
            boot.append(spearmanr(a[k], b[k]).statistic)
    boot = np.array(boot)
    five = np.array([Chem.MolFromSmiles(m).HasSubstructMatch(FIVE) for m in shared])
    assert 3 <= five.sum() <= len(shared) - 3
    rows.append(dict(condition_set=cset, n_shared=len(shared), rho_obs=rho, ceiling=ceiling, rho_corrected=rho / ceiling,
                     ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5), n_boot=len(boot),
                     n_five_membered=int(five.sum()),
                     suzuki_five_minus_other_pp=100 * (a[five].mean() - a[~five].mean()),
                     chanlam_five_minus_other_pp=100 * (b[five].mean() - b[~five].mean()),
                     suzuki_mw_p=mannwhitneyu(a[five], a[~five]).pvalue, chanlam_mw_p=mannwhitneyu(b[five], b[~five]).pvalue))
    members += [dict(condition_set=cset, mol=m, five_membered=bool(f), suzuki_pp=100 * x, chanlam_pp=100 * y)
                for m, f, x, y in zip(shared, five, a, b)]

pd.DataFrame(rows).to_csv(OUT / "boronic_acids_across_roles.csv", index=False)
pd.DataFrame(members).to_csv(OUT / "boronic_acids_across_roles_members.csv", index=False)

# A shared plate is distinct from both complete sets occupying the same single plate.
plates, _, _ = plate_audit(REL)
plates.update(catalyst_counts(REL))
(OUT / "condition_sets.json").write_text(json.dumps(plates, indent=1), encoding="utf-8")
print(pd.DataFrame(rows).round(3).to_string())
print(plates)
