"""Decomposition of the six dense reactions on the second condition set, on the primary and logit scales."""
from pathlib import Path
import os
import sys

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
import pandas as pd
import public_common as P

PACKAGE = Path(__file__).resolve().parents[2]


def main():
    output = PACKAGE / 'output' / 'core_dense'
    core = PACKAGE / 'output' / 'core'
    L, _, _ = P.load_modules(output)
    import analysis_bba as A
    wells, _, _ = P.released()
    rows = wells.rename(columns=P.RENAMES)
    # The conditions and pairs of the second condition set, as written by the core step.
    L.OUT = core
    hc = pd.read_csv(L.OUT / 'holdout_conditions_bba.csv', dtype={'condition_id': str})
    hp = pd.read_csv(L.OUT / 'holdout_pairs_bba.csv')
    six_c = hc[hc.rxn_class.isin(L.REACTIONS)]
    six = L.apply_design(rows[rows.rxn_class.isin(L.REACTIONS)], six_c, hp[hp.rxn_class.isin(L.REACTIONS)])
    order = six_c[['rxn_class','condition_id','rank']]
    R = pd.concat([A.run(six, order)['results'],
                   A.run(six, order, 'logit', 'logit sensitivity')['results']], ignore_index=True)
    R.to_csv(output/'holdout_R1_bba.csv', index=False)


if __name__ == '__main__':
    main()
