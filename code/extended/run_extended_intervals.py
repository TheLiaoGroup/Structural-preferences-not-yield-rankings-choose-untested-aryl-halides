"""Interval of the agreement between the two Buchwald-Hartwig campaigns.

The second campaign's contributions are refitted in memory with the estimator of the extended step
and checked against its recorded design. Its two comparisons with the first campaign (aryl halides
and amines) are pooled; molecules are resampled within each comparison, 2,000 times from seed
20261001. pool_draws and summarize are those of retained_estimators.
"""
from __future__ import annotations
import argparse
import importlib
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
SEED, RESAMPLES = 20261001, 2000
ANALYSIS = 'campaign.same_reaction'
sys.path[:0] = [str(PACKAGE/'code/extended'), str(PACKAGE/'code/core')]
import retained_estimators as T
import public_common


def csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def cell(key, left, right, ceiling, G):
    shared = sorted(set(left.index) & set(right.index))
    if len(shared) < T.MIN_SHARED:
        return None
    assert len(shared) == len(set(shared))
    va, vb = left.loc[shared].to_numpy(), right.loc[shared].to_numpy()
    fn = lambda u,v,c=ceiling: G.spearman(u,v)/c if c > 0 else np.nan
    return {'key':key, 'SMILES':shared, 'values':(va,vb,fn),
            'shared':len(shared), 'rho':G.spearman(va,vb), 'rho_corrected':fn(va,vb)}


def run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_names = [
        'data/experimental_wells.csv',
        'output/core/molecule_contributions.csv',
        'output/core/reaction_labels.csv',
        'output/core/extended_reliability_bba.csv',
        'output/extended/campaign_design.csv',
    ]
    _, _, G = public_common.load_modules(output)
    A = importlib.import_module('analysis_bba')
    # Same parser and fit input rows as the extended stage; the public order is
    # applied to the fitted identity populations.
    wells = csv(PACKAGE/input_names[0], dtype={'condition_id':str,'source_row':str},
                keep_default_na=False, low_memory=False)
    rows = wells[(wells.component=='additional_campaign')&wells.measurement_status.isin(['quantified','zero_by_rule'])].copy()
    rows = rows.rename(columns={'electrophile_smiles':'elec','nucleophile_smiles':'nuc','yield':'y'})
    rows['y'] = pd.to_numeric(rows.y, errors='raise')
    core = csv(PACKAGE/input_names[1]).assign(c=lambda d:d.contribution_pp/100)
    core_series = {(r,s):g.set_index('SMILES').c for (r,s),g in core.groupby(['reaction','side'])}
    labels = csv(PACKAGE/input_names[2], index_col=0).sort_index(kind='stable')
    base_rel = csv(PACKAGE/input_names[3])
    rel0 = {(r.rxn_class,r.side):r.reliability for r in base_rel.itertuples()}
    design = csv(PACKAGE/input_names[4], float_precision='round_trip').set_index('reaction')
    g = rows[rows.rxn_class==T.CAMPAIGN]
    fit,nw,npairs,nb = T.contributions_blocks(g,A)
    assert (nw,npairs,nb) == tuple(int(design.loc[T.CAMPAIGN,k]) for k in ('wells','pairs','blocks'))
    print('Campaign fit reconstructed with the inputs of the extended stage', flush=True)
    cells = []
    for side in ('elec','nuc'):
        for r in sorted(labels.index):
            if {T.CAMPAIGN,r} not in T.SAME_REACTION:
                continue
            f = core_series.get((r,side))
            if f is None:
                continue
            ceiling = math.sqrt(max(design.loc[T.CAMPAIGN,'reliability_'+side],0)*max(rel0[(r,side)],0))
            # The campaign is reaction_a, the analysed reaction reaction_b.
            c = cell((side,T.CAMPAIGN,r),fit[side],f,ceiling,G)
            if c is not None:
                cells.append(c)
    assert len(cells) == 2 and [c['key'] for c in cells] == sorted(c['key'] for c in cells)
    rng = np.random.default_rng(SEED)
    assert type(rng.bit_generator).__name__ == 'PCG64'
    est, draws = T.pool_draws([c['values'] for c in cells], rng)
    result = {**T.summarize(est, draws), 'comparisons': len(cells), 'shared_molecules': sum(c['shared'] for c in cells)}
    pd.DataFrame({ANALYSIS: draws}).to_csv(output/'BOOTSTRAP_DRAWS.csv',index=False,lineterminator='\n',float_format='%.17g')
    pd.DataFrame([{'analysis_id':ANALYSIS,**result,'seed':SEED,'resamples':RESAMPLES}]).to_csv(
        output/'PUBLIC_BOOTSTRAP_RESULTS.csv',index=False,lineterminator='\n',float_format='%.17g')
    print(f'{ANALYSIS}: {result}', flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=PACKAGE/'output/extended_intervals')
    args=parser.parse_args()
    run(args.output)
