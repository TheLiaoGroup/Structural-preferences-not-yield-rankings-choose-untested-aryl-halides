"""The second Buchwald-Hartwig campaign and the structural profiles.

The campaign's contributions are fitted with the frozen estimator within each connected block of
pairs and conditions, with condition-half reliabilities, and compared with every analysed reaction
on the molecules they share. The structural contrasts and profiles of the aryl halides follow
(structure_profiles.py).
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import importlib
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
# Match the canonical pipeline's process environment before NumPy initializes
# its BLAS runtime; setting these after importing NumPy is too late.
for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
import pandas as pd
import retained_estimators as T
import structure_profiles as SP

CORE_INPUTS = (
    'molecule_contributions.csv', 'reaction_labels.csv', 'extended_reliability_bba.csv',
    'holdout_molecule_effects_bba.csv', 'extended_transfer_bba.csv',
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write_csv(frame, path):
    # These tables use pandas' default shortest float representation;
    # the structural calculations read the contribution table.
    frame.to_csv(path, index=False, lineterminator='\n')


def run(package, core, output):
    package, core, output = package.resolve(), core.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = ('data/experimental_wells.csv', 'data/metadata/extended_conditions.csv')
    for name in CORE_INPUTS:
        require((core / name).is_file(), f'Core result not yet generated: {name}')
    sys.path[:0] = [str(package / 'code/core')]
    import public_common
    _, _, G = public_common.load_modules(output)
    A = importlib.import_module('analysis_bba')
    # yield uses the pandas default numeric CSV parser. Replacing this with
    # float(text) changes some low bits and can alter unrounded Mann-Whitney tie
    # ranks downstream. The public source strings remain unchanged.
    wells = pd.read_csv(package / required[0], dtype={'condition_id': str, 'source_row': str},
                        keep_default_na=False, low_memory=False)
    rows = wells[(wells.component == 'additional_campaign') & wells.measurement_status.isin(['quantified', 'zero_by_rule'])].copy()
    require(set(rows.rxn_class) == {T.CAMPAIGN}, 'The additional campaign must be the second Buchwald-Hartwig campaign')
    rows = rows.rename(columns={'electrophile_smiles': 'elec', 'nucleophile_smiles': 'nuc', 'yield': 'y'})
    rows['y'] = pd.to_numeric(rows.y, errors='raise')
    lookup = pd.read_csv(package / required[1], dtype=str, keep_default_na=False)
    original = pd.read_csv(core / 'molecule_contributions.csv').assign(c=lambda d: d.contribution_pp / 100)
    series = {(rc, side): group.set_index('SMILES').c for (rc, side), group in original.groupby(['reaction', 'side'])}
    labels = pd.read_csv(core / 'reaction_labels.csv', index_col=0)
    base_reliability = pd.read_csv(core / 'extended_reliability_bba.csv')
    rel0 = {(r.rxn_class, r.side): r.reliability for r in base_reliability.itertuples()}
    fit, design, comparisons = T.campaign(rows, series, rel0, labels, lookup, A, G)
    contributions = pd.DataFrame([{'reaction': T.CAMPAIGN, 'side': side, 'SMILES': mol, 'contribution_pp': 100 * value}
                                  for side in ('elec', 'nuc') for mol, value in fit[side].items()])
    for name, frame in {'campaign_design.csv': design, 'campaign_comparisons.csv': comparisons,
                        'campaign_contributions.csv': contributions}.items():
        write_csv(frame, output / name)
    print('Campaign fits complete', flush=True)
    SP.RESULTS, SP.CORE_RESULTS = output, core
    with (output / 'structure_console.txt').open('w', encoding='utf-8') as console:
        with redirect_stdout(console):
            SP.main()
    print('Structural contrasts/profiles complete', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--core-output', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.package, args.core_output or args.package/'output/core', args.output or args.package/'output/extended')
