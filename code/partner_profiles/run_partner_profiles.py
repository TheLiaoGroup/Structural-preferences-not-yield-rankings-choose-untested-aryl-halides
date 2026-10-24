"""Structural profiles across partner halves (Supplementary Note 18).

Reads the wells and design metadata through the core adapter, and the core and extended outputs;
writes A1–A5 to output/partner_profiles:
A1, first condition set (fixed partition, with the same-permutation null); A2, the 20 salted
partitions; A3, second condition set; A4, pairs measured in both condition sets; A5, the null
pooled over the seven reactions with a profile."""
from pathlib import Path
import os
import sys
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parents[2]
sys.path.append(str(PACKAGE / 'code/core'))


def main():
    output = PACKAGE / 'output/partner_profiles'
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PACKAGE / 'code/partner_profiles'))
    import numpy as np
    import pandas as pd
    import partner_profiles as A
    A.load(output)
    core, extended = PACKAGE / 'output/core', PACKAGE / 'output/extended'

    def write(frame, name):
        if len(frame) == 0:
            raise RuntimeError(f'empty output {name}')
        frame.to_csv(output / name, index=False, lineterminator='\n', float_format='%.17g')

    labels = pd.read_csv(core / 'reaction_labels.csv', index_col=0)
    aryl = sorted(labels.index[labels.electrophile == 'aryl halide or sulfonate'])
    assert len(aryl) > 0
    wells, conds, pairs = A.PC.released()
    D = {lab: A.PC.design_inputs(wells, conds, pairs, lab) for lab in ('primary', 'validation')}
    LAY = {lab: A.PC.layouts_from(*D[lab], A.G)[:2] for lab in D}
    assert len(LAY['primary'][0]) == len(LAY['validation'][0]) == 15

    # Consistency with the core and extended outputs: unsalted split, half contributions, rho_P and features.
    ref_c = pd.read_csv(core / 'partner_half_contributions_bba.csv')
    ref_g = pd.read_csv(core / 'gated_A_cells_bba.csv')
    ref_f = pd.read_csv(extended / 'aryl_halide_features.csv').set_index('SMILES')
    lay1, ys1 = LAY['primary']
    checked, worst = 0, 0.0
    for rxn in aryl:
        assert A.split_matches_release(lay1[rxn]), rxn
        halves = A.half_fits(lay1[rxn], ys1[rxn])
        for h, arr in enumerate(halves):
            r = ref_c[(ref_c.reaction == rxn) & (ref_c.side == 'elec') & (ref_c.half == 'AB'[h])].set_index('molecule').contribution.dropna()
            s = pd.Series(arr['elec'], index=lay1[rxn].e_name).dropna()
            assert set(r.index) == set(s.index), (rxn, h)
            worst = max(worst, float((r - s.reindex(r.index)).abs().max()))
            checked += len(r)
        row, _ = A.compare(lay1[rxn], halves)
        g = ref_g[(ref_g.reaction == rxn) & (ref_g.side == 'elec') & (ref_g.degree == 2)].iloc[0]
        assert row['evaluable'] == g.molecules and ((np.isnan(row['rho_P_release_rule']) and np.isnan(g.rho_P))
                                                    or abs(row['rho_P_release_rule'] - g.rho_P) < 1e-12), rxn
    assert checked > 0 and worst < 1e-12, (checked, worst)
    assert len(ref_f) > 0 and all(A.feature_row(s) == [int(ref_f.loc[s, k]) for k in A.FEATS] for s in ref_f.index)

    def one_set(label, lay, ys):
        rows = []
        for rxn in aryl:
            row, data = A.compare(lay[rxn], A.half_fits(lay[rxn], ys[rxn]))
            row = {'set': label, 'reaction': rxn, **row}
            if data is not None:
                null = A.permutation_null(*data, f'A-permutation|{label}|{rxn}')
                assert len(null) == A.PERMUTATIONS and np.isfinite(null).all()
                row.update({'null_median': float(np.median(null)), 'null_2.5': float(np.percentile(null, 2.5)),
                            'null_97.5': float(np.percentile(null, 97.5)),
                            'null_p_upper': float((1 + (null >= row['r_structure']).sum()) / (1 + len(null)))})
                row['_null'] = null
            rows.append(row)
        return rows

    rows1, rows3 = one_set('first set', *LAY['primary']), one_set('second set', *LAY['validation'])
    cols = ['set', 'reaction', 'evaluable', 'rho_P_release_rule', 'matched', 'no_aryl_leaving_group', 'features', 'rho_molecule',
            'pearson_molecule', 'r_structure', 'spearman_structure', 'feature_list', 'null_median', 'null_2.5', 'null_97.5', 'null_p_upper']
    write(pd.DataFrame([{k: r.get(k, np.nan) for k in cols} for r in rows1]), 'A1_first_set.csv')
    write(pd.DataFrame([{k: r.get(k, np.nan) for k in cols} for r in rows3]), 'A3_second_set.csv')

    a2 = pd.DataFrame([{'salt': salt, 'reaction': rxn, **{k: row.get(k, np.nan) for k in ('evaluable', 'matched', 'features', 'rho_molecule', 'r_structure')}}
                       for salt in A.SALTS for rxn in aryl
                       for row, _ in [A.compare(lay1[rxn], A.half_fits(lay1[rxn], ys1[rxn], salt=salt))]])
    assert len(a2) == 20 * len(aryl)
    write(a2, 'A2_salts.csv')

    rows4 = []
    for rxn in aryl:
        common, n1, n2, lc = A.common_layouts(rxn, D['primary'], D['validation'])
        res = {t: (lc[t][0], A.half_fits(*lc[t])) for t in ('first', 'second')}
        row = {'reaction': rxn, 'common_pairs': len(common), 'pairs_first': n1, 'pairs_second': n2,
               'same_evaluable_set': bool(np.array_equal(A.evaluable(*res['first']), A.evaluable(*res['second'])))}
        for t in ('first', 'second'):
            r, _ = A.compare(*res[t])
            row.update({f'{k}_{t}': r.get(k, np.nan) for k in ('matched', 'features', 'rho_molecule', 'r_structure')})
        rows4.append(row)
    write(pd.DataFrame(rows4), 'A4_common_pairs.csv')

    rows5 = []
    for tag, rows in (('first set', rows1), ('second set', rows3)):
        rr = [r for r in rows if '_null' in r]
        assert len(rr) == 7
        obs, N = np.array([r['r_structure'] for r in rr]), np.vstack([r['_null'] for r in rr])
        t_null = N.mean(0)
        rows5.append({'set': tag, 'reactions': len(rr), 'mean_r_structure': float(obs.mean()),
                      'pooled_null_median': float(np.median(t_null)), 'pooled_null_97.5': float(np.percentile(t_null, 97.5)),
                      'pooled_p': float((1 + (t_null >= obs.mean()).sum()) / (1 + len(t_null))),
                      'above_own_null_median': int(sum(o > np.median(n) for o, n in zip(obs, N)))})
    write(pd.DataFrame(rows5), 'A5_pooled_permutation.csv')


if __name__ == '__main__':
    main()
