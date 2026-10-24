"""The estimators of code/core/gated_bba.py with sorted index order (six factorize calls with sort=True), as
used by the interval calibration and the heterogeneity test. Estimation and simulation definitions are
unchanged; the well, pair and condition order comes from layout_inputs.py.
"""
import hashlib
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import LinAlgError, cho_factor, cho_solve
from scipy.sparse.linalg import lsqr
from scipy.stats import t as t_dist
import design_extended_bba as E
import design_gated_bba as DG
import lib_bba as L
import rank_bba as RK
K_FOLDS = DG.K_FOLDS
DEGREES = DG.DEGREES
MIN_MOLECULES, MIN_SCORED, MIN_COMPARISON, MIN_JACKKNIFE = (10, 20, 5, 10)
PRIMARY = (('Kumada', 'Pd', 'Ni'), ('Suzuki', 'Pd', 'Ni'))
SECONDARY = (('Kumada', 'Pd', 'Fe'), ('Kumada', 'Ni', 'Fe'))
PREDICTED = 'elec'
LEVEL, LEVEL_PRIMARY = (0.95, 0.975)
MODELS = ('M0', 'ME', 'MN', 'MEN')
NOT_ESTIMABLE, NUMERICAL = ('not estimable by design', 'numerical failure')
UNDEFINED = 'undefined: a within-family agreement is not positive'

def design_one(wp, wc, n_p, n_c):
    n = len(wp)
    cols = np.empty(2 * n, dtype=np.int64)
    cols[0::2], cols[1::2] = (wp, n_p + wc)
    return sparse.csr_matrix((np.ones(2 * n), (np.repeat(np.arange(n), 2), cols)), shape=(n, n_p + n_c))

def stage_one(y, wp, wc, n_p, n_c, X=None):
    """Stage one (Section 4.1): the LSQR call of analysis_bba.two_way_fit, keeping the
    condition effects and the stop code. Pair effects are centred; condition effects are shifted
    by the mean pair effect, so that mean(y - gamma_c) over a fitted pair's wells is its centred
    pair effect. A stop code other than 1 or 2 is a numerical failure."""
    if X is None:
        X = design_one(wp, wc, n_p, n_c)
    res = lsqr(X, np.asarray(y, float), atol=1e-12, btol=1e-12, iter_lim=20000)
    beta, istop = (res[0], int(res[1]))
    shift = beta[:n_p].mean()
    return (beta[:n_p] - shift, beta[n_p:] + shift, istop)

def stage_two(ei, ni, ne, nn, p):
    """Stage two (Section 4.2): p = mu + a_e + b_n with an intercept and one level dropped per
    side, the design of analysis_bba.stage_two, solved through its normal equations (assembled
    from counts). Returns the contributions centred by the unweighted mean over molecules and the
    intercept that goes with them, or None when the normal equations are singular (a disconnected
    set)."""
    p = np.asarray(p, float)
    k = ne + nn - 1
    M = np.zeros((ne + nn, ne + nn))
    np.add.at(M, (ei, ne + ni), 1.0)
    M += M.T
    M[np.arange(ne), np.arange(ne)] = np.bincount(ei, minlength=ne)
    M[ne + np.arange(nn), ne + np.arange(nn)] = np.bincount(ni, minlength=nn)
    keep = np.r_[np.arange(1, ne), ne + np.arange(1, nn)]
    XtX = np.empty((k, k))
    XtX[0, 0] = len(p)
    XtX[0, 1:] = XtX[1:, 0] = M.diagonal()[keep]
    XtX[1:, 1:] = M[np.ix_(keep, keep)]
    Xty = np.r_[p.sum(), np.bincount(ei, p, ne)[1:], np.bincount(ni, p, nn)[1:]]
    try:
        beta = cho_solve(cho_factor(XtX), Xty)
    except LinAlgError:
        return None
    a = np.concatenate([[0.0], beta[1:ne]])
    b = np.concatenate([[0.0], beta[ne:]])
    return (a - a.mean(), b - b.mean(), beta[0] + a.mean() + b.mean())

def piece_mask(elec, nuc):
    """Pairs of the largest connected piece: most pairs, at least three; a tie goes to the piece
    holding the alphabetically first electrophile (the rule of design_gated_bba.largest_piece)."""
    elec, nuc = (np.asarray(elec), np.asarray(nuc))
    if len(elec) < 3:
        return np.zeros(len(elec), bool)
    labels = L.component_labels(elec, nuc)
    counts = np.bincount(labels)
    best = np.flatnonzero(counts == counts.max())
    chosen = min(best, key=lambda c: min(elec[labels == c])) if len(best) > 1 else best[0]
    return labels == chosen

def contributions(elec, nuc, p):
    """Stage two on a connected set of pairs given by molecule names; Series per side, or None."""
    ei, en = pd.factorize(np.asarray(elec))
    ni, nn_ = pd.factorize(np.asarray(nuc))
    fit = stage_two(ei, ni, len(en), len(nn_), p)
    if fit is None:
        return None
    a, b, _ = fit
    return {'elec': pd.Series(a, index=en), 'nuc': pd.Series(b, index=nn_)}

def jackknife(estimate, values, level):
    """The delete-one-molecule jackknife interval of Section 4.7: (se, low, high, n)."""
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    n = len(v)
    if n < MIN_JACKKNIFE or not np.isfinite(estimate):
        return (np.nan, np.nan, np.nan, n)
    se = math.sqrt((n - 1) / n * float(np.sum((v - v.mean()) ** 2)))
    q = float(t_dist.ppf(0.5 + level / 2, n - 1))
    return (se, estimate - q * se, estimate + q * se, n)

def category(low, high, status='ok'):
    """Section 4.6, or the reason no evidence category applies."""
    if status != 'ok':
        return status
    if not (np.isfinite(low) and np.isfinite(high)):
        return NUMERICAL
    if low > 0:
        return 'supporting evidence'
    if high < 0:
        return 'contrary evidence'
    return 'insufficient evidence'

def average_ranks(x):
    """Ranks 1..n with ties at their average rank."""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(cnt)
    return ((ends - cnt + 1 + ends) / 2.0)[inv]

def spearman(a, b):
    """Spearman's rho as rank_bba.spearman computes it: values rounded to twelve decimals, ties at
    their average rank, Pearson's r of the ranks. Written out to avoid scipy.stats.spearmanr's
    overhead inside the simulations; test_gated_bba.py checks that the two agree."""
    a, b = (np.asarray(a, float), np.asarray(b, float))
    if len(a) < 3:
        return np.nan
    ra = average_ranks(np.round(a, RK.DECIMALS))
    rb = average_ranks(np.round(b, RK.DECIMALS))
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / den) if den > 0 else np.nan

def half_arrays(lay, pairs, p):
    """Stage two on a connected set of layout pairs: contributions as arrays over the layout's
    molecules, NaN where a molecule is absent; None when the fit is singular."""
    le, ei = np.unique(lay.pe[pairs], return_inverse=True)
    ln, ni = np.unique(lay.pn[pairs], return_inverse=True)
    fit = stage_two(ei, ni, len(le), len(ln), p)
    if fit is None:
        return None
    a, b, _ = fit
    A = np.full(len(lay.e_name), np.nan)
    Bn = np.full(len(lay.n_name), np.nan)
    A[le], Bn[ln] = (a, b)
    return {'elec': A, 'nuc': Bn}

def deletion(lay, key, pairs, side_del, mol):
    """Positions within `pairs` kept when one molecule and its pairs are deleted and the rest is
    cut to its largest piece. Structure only, so it is cached on the layout."""
    ck = (key, side_del, int(mol))
    got = lay.cache.get(ck)
    if got is None:
        codes = lay.pe[pairs] if side_del == 'elec' else lay.pn[pairs]
        keep = np.flatnonzero(codes != mol)
        if len(keep) >= 3:
            keep = keep[piece_mask(lay.e_name[lay.pe[pairs[keep]]], lay.n_name[lay.pn[pairs[keep]]])]
        else:
            keep = keep[:0]
        lay.cache[ck] = got = keep
    return got

def named(lay, arrays, side):
    """An array of contributions as a Series by molecule name, absent molecules dropped."""
    s = pd.Series(arrays[side], index=lay.e_name if side == 'elec' else lay.n_name)
    return s[np.isfinite(s.values)]

class Sub:
    """A subset of a layout's pairs, with their wells at a subset of conditions, coded for stage
    one. Pairs without a well at those conditions are dropped; the rest are cut to their largest
    connected piece when `connect` is set."""

    def __init__(self, lay, pair_idx, cond_ok=None, connect=True):
        pair_idx = np.asarray(pair_idx, int)
        w = np.flatnonzero(np.isin(lay.w_pair, pair_idx) & (True if cond_ok is None else cond_ok[lay.w_cond]))
        present = np.unique(lay.w_pair[w])
        if connect and len(present):
            present = present[piece_mask(lay.e_name[lay.pe[present]], lay.n_name[lay.pn[present]])]
            w = w[np.isin(lay.w_pair[w], present)]
        self.pairs = present
        self.wells = w
        self.wp = np.searchsorted(present, lay.w_pair[w])
        self.wc, self.conds = pd.factorize(lay.w_cond[w])
        self.X = design_one(self.wp, self.wc, len(present), len(self.conds)) if len(w) else None
        self.n_layout_conds = len(lay.conds)

    def fit(self, y):
        """Stage one on this subset: (pair effects, condition effects by layout condition, stop)."""
        p, g, stop = stage_one(y[self.wells], self.wp, self.wc, len(self.pairs), len(self.conds), self.X)
        gamma = np.full(self.n_layout_conds, np.nan)
        gamma[self.conds] = g
        return (p, gamma, stop)

class ReactionLayout:
    """The analysed design of one reaction: its pairs and usable wells at
    its twenty conditions, the partner halves and folds of Sections 4.4-4.5 and the condition
    halves of Gate A. Built from metadata only."""

    def __init__(self, rxn, wells, pairs, cond_order, plate=None):
        self.rxn = rxn
        self.args = (wells.reset_index(drop=True), pairs, list(cond_order))
        pairs = pairs.reset_index(drop=True)[['elec', 'nuc']]
        pe, self.e_name = pd.factorize(pairs.elec.values, sort=True)
        pn, self.n_name = pd.factorize(pairs.nuc.values, sort=True)
        self.e_name, self.n_name = (np.asarray(self.e_name), np.asarray(self.n_name))
        self.pe, self.pn = (pe, pn)
        self.n_pairs = len(pairs)
        key = pd.MultiIndex.from_frame(pairs)
        self.w_pair = key.get_indexer(pd.MultiIndex.from_frame(wells[['elec', 'nuc']]))
        assert (self.w_pair >= 0).all(), f'{rxn}: a well outside the design'
        self.conds = list(cond_order)
        cix = {c: i for i, c in enumerate(self.conds)}
        self.w_cond = wells.condition_id.map(cix).to_numpy()
        assert not pd.isna(self.w_cond).any(), f"{rxn}: a well outside the design's conditions"
        self.w_cond = self.w_cond.astype(int)
        self.w_plate = pd.factorize(pd.Series(self.w_pair).astype(str) + '|' + (DG.physical_plate_ids(plate).values if plate is not None else ''), sort=True)[0]
        assert not pd.DataFrame({'p': self.w_pair, 'c': self.w_cond}).duplicated().any()
        self.deg = {'elec': np.bincount(pe), 'nuc': np.bincount(pn)}
        self.cache = {}
        self.full = Sub(self, np.arange(self.n_pairs), connect=False)
        self.halves = {}
        for side in ('elec', 'nuc'):
            parts = DG.partner_halves(pairs, rxn, side)
            self.halves[side] = [Sub(self, key.get_indexer(pd.MultiIndex.from_frame(h[['elec', 'nuc']])), connect=False) for h in parts]
        self.cond_halves = [Sub(self, np.arange(self.n_pairs), np.arange(len(self.conds)) % 2 == h) for h in (0, 1)]
        fold = np.array([DG.fold_of(rxn, e, n) for e, n in zip(pairs.elec, pairs.nuc)])
        self.folds = []
        for f in range(K_FOLDS):
            out_f = np.flatnonzero(fold != f)
            train = out_f[piece_mask(pairs.elec.values[out_f], pairs.nuc.values[out_f])]
            es, ns = (set(pe[train]), set(pn[train]))
            test = np.array([i for i in np.flatnonzero(fold == f) if pe[i] in es and pn[i] in ns], int)
            self.folds.append((Sub(self, train, connect=False), test))
        self.pair_wells = pd.Series(np.arange(len(self.w_pair))).groupby(self.w_pair).apply(np.asarray)

def reaction_fits(lay, y):
    """Full-data and partner-half contributions of one reaction (arrays over the layout's
    molecules, and Series by name); None where a fit failed."""
    out = {'stop': [], 'y': y, 'halves': {}, 'halves_s': {}}
    p, _, stop = lay.full.fit(y)
    out['stop'].append(stop)
    out['full'] = half_arrays(lay, lay.full.pairs, p) if stop in (1, 2) else None
    out['full_s'] = {s: named(lay, out['full'], s) for s in ('elec', 'nuc')} if out['full'] else None
    for side in ('elec', 'nuc'):
        vals = []
        for sub in lay.halves[side]:
            ph, _, st = sub.fit(y)
            out['stop'].append(st)
            vals.append((ph, half_arrays(lay, sub.pairs, ph) if st in (1, 2) else None))
        out['halves'][side] = vals
        out['halves_s'][side] = [named(lay, v, side) if v is not None else None for _, v in vals]
    return out

def evaluable(lay, fits, side, k):
    """Molecules (layout indices) with a contribution in both partner halves and full-design
    degree >= k."""
    a, b = (v[1] for v in fits['halves'][side])
    if a is None or b is None:
        return np.array([], int)
    idx = np.flatnonzero(np.isfinite(a[side]) & np.isfinite(b[side]))
    return idx[lay.deg[side][idx] >= k]

def gate_a_partner(lay, fits, inference=True):
    """Strict partner-half correlation per side and degree, with its jackknife interval, and the
    condition-half correlation on the same molecules (descriptive)."""
    rows = []
    cond = []
    for sub in lay.cond_halves:
        ph, _, st = sub.fit(fits['y'])
        cond.append(half_arrays(lay, sub.pairs, ph) if st in (1, 2) else None)
    for side in ('elec', 'nuc'):
        (pa, ca), (pb, cb) = fits['halves'][side]
        ev = {k: evaluable(lay, fits, side, k) for k in DEGREES}
        est = {k: spearman(ca[side][ev[k]], cb[side][ev[k]]) if len(ev[k]) >= MIN_MOLECULES else np.nan for k in DEGREES}
        jk = {k: [] for k in DEGREES}
        if inference and ca is not None and (cb is not None):
            subs = lay.halves[side]
            held = [(set(lay.pe[s.pairs]), set(lay.pn[s.pairs])) for s in subs]
            units = sorted({('elec', int(m)) for h in held for m in h[0]} | {('nuc', int(m)) for h in held for m in h[1]})
            for side_del, mol in units:
                vals = []
                for h, (s, p_, c_) in enumerate(zip(subs, (pa, pb), (ca, cb))):
                    if mol not in held[h][0 if side_del == 'elec' else 1]:
                        vals.append(c_)
                        continue
                    keep = deletion(lay, ('half', side, h), s.pairs, side_del, mol)
                    vals.append(half_arrays(lay, s.pairs[keep], p_[keep]) if len(keep) >= 3 else None)
                va, vb = vals
                for k in DEGREES:
                    if va is None or vb is None or len(ev[k]) < MIN_MOLECULES:
                        continue
                    idx = ev[k][ev[k] != mol] if side_del == side else ev[k]
                    idx = idx[np.isfinite(va[side][idx]) & np.isfinite(vb[side][idx])]
                    jk[k].append(spearman(va[side][idx], vb[side][idx]))
        for k in DEGREES:
            n = len(ev[k])
            status = 'ok' if n >= MIN_MOLECULES else NOT_ESTIMABLE
            if status == 'ok' and (ca is None or cb is None):
                status = NUMERICAL
            if inference and status == 'ok':
                se, lo, hi, nj = jackknife(est[k], jk[k], LEVEL)
            else:
                se, lo, hi, nj = (np.nan, np.nan, np.nan, 0)
            rc = np.nan
            if cond[0] is not None and cond[1] is not None and n:
                m = ev[k][np.isfinite(cond[0][side][ev[k]]) & np.isfinite(cond[1][side][ev[k]])]
                rc = spearman(cond[0][side][m], cond[1][side][m]) if len(m) >= MIN_MOLECULES else np.nan
            rows.append({'reaction': lay.rxn, 'side': side, 'degree': k, 'molecules': n, 'rho_P': est[k], 'se': se, 'low': lo, 'high': hi, 'jackknife n': nj, 'category': category(lo, hi, status if inference else 'no inference'), 'rho_C': rc, 'delta_CP': rc - est[k]})
    return rows

def fold_predictions(lay, y, fold_sub, test):
    """One fold of the end-to-end prediction (Section 5.1 iii): stage one on the training piece,
    targets of the scorable test pairs from their own wells with the training condition effects,
    and the four models' predictions. Returns (test pairs, targets, predictions by model, stop)."""
    p, gamma, stop = fold_sub.fit(y)
    if stop not in (1, 2):
        return None
    tr = fold_sub.pairs
    te, ti = (lay.pe[tr], lay.pn[tr])
    ne, nn = (len(lay.e_name), len(lay.n_name))
    t_list, keep = ([], [])
    for i in test:
        w = lay.pair_wells[i]
        g = gamma[lay.w_cond[w]]
        ok = np.isfinite(g)
        if ok.any():
            t_list.append(float(np.mean(y[w][ok] - g[ok])))
            keep.append(i)
    test = np.asarray(keep, int)
    if not len(test):
        return (test, np.array([]), {m: np.array([]) for m in MODELS}, stop)
    return (test, np.asarray(t_list), side_models(te, ti, p, lay.pe[test], lay.pn[test], ne, nn), stop)

def side_models(te, ti, m, qe, qn, ne, nn):
    """M0, ME, MN and MEN fitted on training targets m of pairs (te, ti) and evaluated at test
    pairs (qe, qn); molecule indices are global to the layout."""
    m = np.asarray(m, float)
    sum_e, cnt_e = (np.bincount(te, m, ne), np.bincount(te, None, ne))
    sum_n, cnt_n = (np.bincount(ti, m, nn), np.bincount(ti, None, nn))
    out = {'M0': np.full(len(qe), m.mean()), 'ME': sum_e[qe] / cnt_e[qe], 'MN': sum_n[qn] / cnt_n[qn]}
    le, e_codes = np.unique(te, return_inverse=True)
    ln, n_codes = np.unique(ti, return_inverse=True)
    fit = stage_two(e_codes, n_codes, len(le), len(ln), m)
    if fit is None:
        out['MEN'] = np.full(len(qe), np.nan)
    else:
        a, b, mu = fit
        out['MEN'] = mu + a[np.searchsorted(le, qe)] + b[np.searchsorted(ln, qn)]
    return out

def prediction_metrics(pe, pn, t, pred, level, inference=True, level_a=None):
    """SSE of the four models on the same scored pairs; R2 against M0, increments and A_s, with
    the scoring-stage jackknife over molecules (predictions held fixed). A_E and A_N take
    `level_a` (the 97.5% of the primary contrasts), the rest `level`. Returns a dict."""
    e2 = {m: (t - pred[m]) ** 2 for m in MODELS}
    sse = {m: float(e2[m].sum()) for m in MODELS}

    def metrics(s):
        d = {'R2_ME': 1 - s['ME'] / s['M0'], 'R2_MN': 1 - s['MN'] / s['M0'], 'R2_MEN': 1 - s['MEN'] / s['M0'], 'I_EN': (s['MN'] - s['MEN']) / s['M0'], 'I_NE': (s['ME'] - s['MEN']) / s['M0'], 'A_E': (s['MN'] - s['ME']) / s['M0']}
        d['A_N'] = -d['A_E']
        return d
    est = metrics(sse)
    out = {'scored': len(t), **{f'SSE_{m}': sse[m] for m in MODELS}, **{f'RMSE_{m}': math.sqrt(sse[m] / len(t)) if len(t) else np.nan for m in MODELS}}
    if inference and len(t):
        ue, ie = np.unique(pe, return_inverse=True)
        un, inn = np.unique(pn, return_inverse=True)
        per_e = {m: np.bincount(ie, e2[m], len(ue)) for m in MODELS}
        per_n = {m: np.bincount(inn, e2[m], len(un)) for m in MODELS}
        deleted = [metrics({m: sse[m] - per_e[m][j] for m in MODELS}) for j in range(len(ue))]
        deleted += [metrics({m: sse[m] - per_n[m][j] for m in MODELS}) for j in range(len(un))]
    for key, value in est.items():
        out[key] = value
        if inference and len(t):
            se, lo, hi, nj = jackknife(value, [d[key] for d in deleted], level_a if level_a and key in ('A_E', 'A_N') else level)
        else:
            se, lo, hi, nj = (np.nan, np.nan, np.nan, 0)
        out[f'{key} se'], out[f'{key} low'], out[f'{key} high'] = (se, lo, hi)
    out['jackknife n'] = nj
    return out

def gate_a_prediction(lay, y, inference=True):
    """Ten-fold partner-held-out prediction of pair targets; one row per degree threshold."""
    got, stops = ([], [])
    for sub, test in lay.folds:
        r = fold_predictions(lay, y, sub, test)
        if r is None:
            stops.append(False)
            continue
        stops.append(True)
        got.append(r[:3])
    rows = []
    if not all(stops):
        return [{'reaction': lay.rxn, 'degree': k, 'status': NUMERICAL} for k in DEGREES]
    test = np.concatenate([g[0] for g in got])
    t = np.concatenate([g[1] for g in got])
    pred = {m: np.concatenate([g[2][m] for g in got]) for m in MODELS}
    for k in DEGREES:
        s = (lay.deg['elec'][lay.pe[test]] >= k) & (lay.deg['nuc'][lay.pn[test]] >= k)
        rec = {'reaction': lay.rxn, 'degree': k}
        if s.sum() < MIN_SCORED:
            rows.append({**rec, 'scored': int(s.sum()), 'status': NOT_ESTIMABLE})
            continue
        met = prediction_metrics(lay.pe[test][s], lay.pn[test][s], t[s], {m: v[s] for m, v in pred.items()}, LEVEL, inference)
        rows.append({**rec, **met, 'status': 'ok'})
    return rows

def comparison_list():
    """The cross-reaction comparisons of the design."""
    B = pd.read_csv(L.OUT / 'gated_gateB_eligibility_bba.csv')
    B = B[~B.same_reaction]
    assert len(B) > 0
    return list(zip(B.side, B.rxn_a, B.rxn_b))

def gate_b(layouts, fits, k=2, inference=True, bootstrap=2000, rng=None):
    """Section 6: observed transfer (6.1) and the relative claim (6.2) over the design-eligible
    comparisons at degree k. `fits` holds reaction_fits per reaction."""
    comps = comparison_list()
    obs, rel = ([], [])
    for side, ra, rb in comps:
        fa, fb = (fits[ra]['full_s'], fits[rb]['full_s'])
        if fa is None or fb is None:
            obs.append((side, ra, rb, 0, np.nan))
        else:
            shared = fa[side].index.intersection(fb[side].index)
            obs.append((side, ra, rb, len(shared), spearman(fa[side][shared], fb[side][shared])))
        la, lb = (layouts[ra], layouts[rb])
        names = lambda lay, idx: (lay.e_name if side == 'elec' else lay.n_name)[idx]
        ea, eb = (set(names(la, evaluable(la, fits[ra], side, k))), set(names(lb, evaluable(lb, fits[rb], side, k))))
        M = sorted(ea & eb)
        if len(M) < MIN_COMPARISON:
            continue
        A1, A2 = fits[ra]['halves_s'][side]
        B1, B2 = fits[rb]['halves_s'][side]
        pa = spearman(A1[M], A2[M])
        pb = spearman(B1[M], B2[M])
        cross = np.mean([spearman(x[M], z[M]) for x in (A1, A2) for z in (B1, B2)])
        rel.append({'side': side, 'rxn_a': ra, 'rxn_b': rb, 'molecules': len(M), 'rho_P_a': pa, 'rho_P_b': pb, 'rho_R_half': cross})
    O = pd.DataFrame(obs, columns=['side', 'rxn_a', 'rxn_b', 'shared', 'rho_R'])
    R = pd.DataFrame(rel)
    out = {'observed': O, 'relative': R}
    out['pooled observed'] = pooled_observed(O, inference, bootstrap, rng)
    out['pooled relative'] = pooled_relative(R, inference) if len(R) else None
    return out

def weighted_mean(v, w):
    v, w = (np.asarray(v, float), np.asarray(w, float))
    ok = np.isfinite(v) & (w > 0)
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan

def pooled_observed(O, inference=True, bootstrap=2000, rng=None):
    """Observed rho_R pooled three ways, with the reaction-level jackknife and a node bootstrap."""
    O = O[np.isfinite(O.rho_R) & (O.shared >= MIN_COMPARISON)]
    z = np.arctanh(np.clip(O.rho_R.values, -0.999999, 0.999999))
    est = {'weighted': weighted_mean(O.rho_R, O.shared), 'fisher': float(np.tanh(z.mean())), 'median': float(np.median(O.rho_R))}
    res = {'comparisons': len(O), **est}
    if not inference:
        return res
    reactions = sorted(set(O.rxn_a) | set(O.rxn_b))
    loro = [weighted_mean(*O[(O.rxn_a != r) & (O.rxn_b != r)][['rho_R', 'shared']].T.values) for r in reactions]
    se, lo, hi, n = jackknife(est['weighted'], loro, LEVEL)
    res.update({'weighted se': se, 'weighted low': lo, 'weighted high': hi, 'reactions': n, 'loro min': float(np.nanmin(loro)), 'loro max': float(np.nanmax(loro))})
    if bootstrap and rng is not None:
        idx = {r: i for i, r in enumerate(reactions)}
        ia, ib = (O.rxn_a.map(idx).values, O.rxn_b.map(idx).values)
        boots = []
        for _ in range(bootstrap):
            cnt = np.bincount(rng.integers(0, len(reactions), len(reactions)), minlength=len(reactions))
            mult = cnt[ia] * cnt[ib]
            boots.append(weighted_mean(O.rho_R.values, O.shared.values * mult))
        res['bootstrap low'], res['bootstrap high'] = (float(q) for q in np.nanquantile(boots, [0.025, 0.975]))
    return res

def relative_values(R):
    """D_A, D_G and kappa-bar over a set of eligible comparisons (Section 6.2)."""
    w = R.molecules.values.astype(float)
    da = weighted_mean(0.5 * (R.rho_P_a + R.rho_P_b) - R.rho_R_half, w)
    pos = (R.rho_P_a > 0) & (R.rho_P_b > 0)
    g = np.sqrt(R.rho_P_a[pos] * R.rho_P_b[pos])
    dg = weighted_mean(g - R.rho_R_half[pos], w[pos]) if pos.any() else np.nan
    kb = float(np.sum(w[pos] * R.rho_R_half[pos]) / np.sum(w[pos] * g)) if pos.any() and np.sum(w[pos] * g) > 0 else np.nan
    return {'D_A': da, 'D_G': dg, 'kappa': kb, 'used for D_G': int(pos.sum()), 'excluded': int((~pos).sum())}

def pooled_relative(R, inference=True):
    est = relative_values(R)
    res = {'comparisons': len(R), **est}
    if inference:
        reactions = sorted(set(R.rxn_a) | set(R.rxn_b))
        dele = [relative_values(R[(R.rxn_a != r) & (R.rxn_b != r)]) for r in reactions]
        for key in ('D_G', 'kappa', 'D_A'):
            se, lo, hi, n = jackknife(est[key], [d[key] for d in dele], LEVEL)
            res.update({f'{key} se': se, f'{key} low': lo, f'{key} high': hi})
        res['reactions'] = n
        res['D_G category'] = category(res['D_G low'], res['D_G high'])
    return res

class ContrastLayout:
    """One activation contrast: its condition-pair units, the
    differential-response piece and its folds, the scaling-adjusted piece and its folds, the four
    family halves, the per-unit pieces and the condition-disjoint unit pairs, plus the analysed
    wells that give each pair its historical baseline. Built from metadata only."""

    def __init__(self, rxn, fa, fb, units, wells, hist_pairs, hist_conds, families):
        self.rxn, self.fa, self.fb = (rxn, fa, fb)
        self.args = (units, wells.reset_index(drop=True), hist_pairs, hist_conds, families)
        self.kind = units.design.iloc[0]
        self.primary = (rxn, fa, fb) in PRIMARY
        self.level = LEVEL_PRIMARY if self.primary else LEVEL
        pairs = wells[['elec', 'nuc']].drop_duplicates().sort_values(['elec', 'nuc']).reset_index(drop=True)
        pe, e_name = pd.factorize(pairs.elec.values, sort=True)
        pn, n_name = pd.factorize(pairs.nuc.values, sort=True)
        self.e_name, self.n_name, self.pe, self.pn = (np.asarray(e_name), np.asarray(n_name), pe, pn)
        self.n_pairs = len(pairs)
        key = pd.MultiIndex.from_frame(pairs)
        self.w_pair = key.get_indexer(pd.MultiIndex.from_frame(wells[['elec', 'nuc']]))
        self.conds = sorted(set(wells.condition_id))
        cix = {c: i for i, c in enumerate(self.conds)}
        self.w_cond = wells.condition_id.map(cix).to_numpy().astype(int)
        self.w_plate = pd.factorize(pd.Series(self.w_pair).astype(str) + '|' + DG.physical_plate_ids(wells.plate).values, sort=True)[0]
        fam_of = dict(zip(families.condition_id, families.family))
        self.w_family = np.array(['B' if fam_of.get(c) == fb else 'A' for c in wells.condition_id])
        self.pair_wells = pd.Series(np.arange(len(self.w_pair))).groupby(self.w_pair).apply(np.asarray)
        self.cache = {}
        cell = {(p, c): w for w, (p, c) in enumerate(zip(self.w_pair, self.w_cond))}
        assert len(cell) == len(self.w_pair), 'two wells in one pair-condition cell'
        runs = {c: set(self.w_pair[self.w_cond == cix[c]]) for c in self.conds}
        hk = key.get_indexer(pd.MultiIndex.from_frame(hist_pairs[['elec', 'nuc']]))
        self.hist_pairs = np.sort(hk[hk >= 0])
        hc = np.isin(np.arange(len(self.conds)), [cix[c] for c in hist_conds if c in cix])
        self.hist = Sub(self, self.hist_pairs, hc, connect=False)
        self.units = []
        if self.kind == 'regime':
            for h in sorted(set(units['replicate or half'])):
                s = units[units['replicate or half'] == h]
                a, b = (sorted(set(s['condition A'])), sorted(set(s['condition B'])))
                need = math.ceil(len(a) / 2)
                ok = {p for p in set().union(*[runs.get(c, set()) for c in a + b]) if sum((p in runs.get(c, set()) for c in a)) >= need and sum((p in runs.get(c, set()) for c in b)) >= need}
                self.units.append({'A': a, 'B': b, 'pairs': ok})
        else:
            for a, b in zip(units['condition A'], units['condition B']):
                self.units.append({'A': [a], 'B': [b], 'pairs': runs.get(a, set()) & runs.get(b, set())})
        delta = np.array(sorted(set().union(*[u['pairs'] for u in self.units])), int)
        m = piece_mask(self.e_name[pe[delta]], self.n_name[pn[delta]])
        self.piece = delta[m]
        in_piece = set(self.piece)
        self.deg_piece = {'elec': np.bincount(pe[self.piece], minlength=len(e_name)), 'nuc': np.bincount(pn[self.piece], minlength=len(n_name))}
        if self.kind != 'regime':
            rows = [(k, p, cell[p, cix[u['A'][0]]], cell[p, cix[u['B'][0]]]) for k, u in enumerate(self.units) for p in sorted(u['pairs'] & in_piece)]
            D = np.array(rows, int)
            self.d_unit, self.d_pair, self.d_a, self.d_b = (D[:, 0], D[:, 1], D[:, 2], D[:, 3])
            hs = set(self.hist_pairs)
            with_hist = np.array([p for p in self.piece if p in hs], int)
            adj = with_hist[piece_mask(self.e_name[pe[with_hist]], self.n_name[pn[with_hist]])]
            self.variants = {'unadjusted': self._folds(self.piece), 'adjusted': self._folds(adj)}
        if self.kind == 'regime':
            halves = [u['A'] for u in self.units] + [u['B'] for u in self.units]
        else:
            order = lambda fam, cs: sorted(cs, key=lambda c: hashlib.sha256(f'{rxn}|{fam}|{c}'.encode()).hexdigest())
            uA = order(fa, sorted(set(units['condition A'])))
            uB = order(fb, sorted(set(units['condition B'])))
            halves = [uA[0::2], uA[1::2], uB[0::2], uB[1::2]]
        self.family_halves = []
        for cs in halves:
            q = np.array(sorted(in_piece & set().union(*[runs.get(c, set()) for c in cs])), int)
            q = q[piece_mask(self.e_name[pe[q]], self.n_name[pn[q]])] if len(q) >= 3 else q[:0]
            ok = np.isin(np.arange(len(self.conds)), [cix[c] for c in cs])
            self.family_halves.append(Sub(self, q, ok, connect=False))
        self.unit_subs = []
        for u in self.units:
            q = np.array(sorted(u['pairs'] & in_piece), int)
            m = piece_mask(self.e_name[pe[q]], self.n_name[pn[q]]) if len(q) >= 3 else np.zeros(len(q), bool)
            self.unit_subs.append(q[m])
        conds_of = [set(u['A']) | set(u['B']) for u in self.units]
        self.disjoint = [(i, j) for i in range(len(self.units)) for j in range(i + 1, len(self.units)) if not conds_of[i] & conds_of[j]]
        self.cov_struct = self._cov_structure() if self.kind != 'regime' else None

    def _cov_structure(self):
        """Section 7.7 for matched designs, structure only: each unit's piece coded for stage two,
        and for each condition-disjoint unit pair the positions of the common evaluation set."""
        units = []
        for u, q in enumerate(self.unit_subs):
            if len(q) < 3:
                units.append(None)
                continue
            sel = np.flatnonzero(self.d_unit == u)
            sel = sel[np.isin(self.d_pair[sel], q)]
            sel = sel[np.argsort(self.d_pair[sel])]
            assert np.array_equal(self.d_pair[sel], q)
            le, ei = np.unique(self.pe[q], return_inverse=True)
            ln, ni = np.unique(self.pn[q], return_inverse=True)
            units.append({'q': q, 'sel': sel, 'le': le, 'ln': ln, 'ei': ei, 'ni': ni})
        pairs = []
        for u, v in self.disjoint:
            U, V = (units[u], units[v])
            if U is None or V is None:
                continue
            common, iu, iv = np.intersect1d(U['q'], V['q'], return_indices=True)
            e, n = (self.pe[common], self.pn[common])
            ok = np.isin(e, U['le']) & np.isin(e, V['le']) & np.isin(n, U['ln']) & np.isin(n, V['ln'])
            if ok.sum() < 3:
                continue
            pairs.append((u, v, iu[ok], iv[ok], np.searchsorted(U['le'], e[ok]), np.searchsorted(V['le'], e[ok]), np.searchsorted(U['ln'], n[ok]), np.searchsorted(V['ln'], n[ok])))
        return (units, pairs)

    def _folds(self, pairs):
        fold = np.array([DG.fold_of(self.rxn, self.e_name[self.pe[p]], self.n_name[self.pn[p]]) for p in pairs])
        out = []
        for f in range(K_FOLDS):
            rest = pairs[fold != f]
            train = rest[piece_mask(self.e_name[self.pe[rest]], self.n_name[self.pn[rest]])]
            es, ns = (set(self.pe[train]), set(self.pn[train]))
            test = np.array([p for p in pairs[fold == f] if self.pe[p] in es and self.pn[p] in ns], int)
            out.append((np.sort(train), test))
        return {'pairs': pairs, 'folds': out}

def delta_values(cl, y):
    return y[cl.d_a] - y[cl.d_b]

def differential_fold(cl, dvals, train, test):
    """Section 7.3 in one fold: Delta = tau_k + delta_ij + e fitted on the training pairs; training
    targets are the centred delta_ij, test targets mean(Delta - tau_k) over their units."""
    sel = np.isin(cl.d_pair, train)
    wp = np.searchsorted(train, cl.d_pair[sel])
    wu, units = pd.factorize(cl.d_unit[sel])
    p, tau, stop = stage_one(dvals[sel], wp, wu, len(train), len(units))
    if stop not in (1, 2):
        return None
    tau_of = np.full(len(cl.units), np.nan)
    tau_of[units] = tau
    t = []
    for q in test:
        s = cl.d_pair == q
        v = dvals[s] - tau_of[cl.d_unit[s]]
        v = v[np.isfinite(v)]
        t.append(float(v.mean()) if len(v) else np.nan)
    return (p, np.asarray(t, float))

def gate_c_prediction(cl, y, B=None, inference=True):
    """Primary estimand of Section 7.5 on the historical-baseline-adjusted differential response,
    the unadjusted target on the same pairs, and the unadjusted target on its own larger piece."""
    if cl.kind == 'regime':
        return []
    dvals = delta_values(cl, y)
    out = []
    for name, variant, adjust in (('adjusted', 'adjusted', True), ('unadjusted, same pairs', 'adjusted', False), ('unadjusted, own piece', 'unadjusted', False)):
        V = cl.variants[variant]
        tests, targets, preds, failed = ([], [], {m: [] for m in MODELS}, False)
        for train, test in V['folds']:
            got = differential_fold(cl, dvals, train, test)
            if got is None:
                failed = True
                break
            m_train, t_test = got
            keep = np.isfinite(t_test)
            test, t_test = (test[keep], t_test[keep])
            if adjust:
                bt, bq = (B[train], B[test])
                X = np.column_stack([np.ones(len(train)), bt])
                c, lam = np.linalg.lstsq(X, m_train, rcond=None)[0]
                m_train = m_train - (c + lam * bt)
                t_test = t_test - (c + lam * bq)
            pr = side_models(cl.pe[train], cl.pn[train], m_train, cl.pe[test], cl.pn[test], len(cl.e_name), len(cl.n_name))
            tests.append(test)
            targets.append(t_test)
            for mdl in MODELS:
                preds[mdl].append(pr[mdl])
        rec = {'reaction': cl.rxn, 'family A': cl.fa, 'family B': cl.fb, 'target': name, 'level': cl.level}
        if failed:
            out.append({**rec, 'status': NUMERICAL})
            continue
        test = np.concatenate(tests)
        t = np.concatenate(targets)
        pred = {mdl: np.concatenate(v) for mdl, v in preds.items()}
        s = (cl.deg_piece['elec'][cl.pe[test]] >= 2) & (cl.deg_piece['nuc'][cl.pn[test]] >= 2)
        if s.sum() < MIN_SCORED:
            out.append({**rec, 'scored': int(s.sum()), 'status': NOT_ESTIMABLE})
            continue
        met = prediction_metrics(cl.pe[test][s], cl.pn[test][s], t[s], {m: v[s] for m, v in pred.items()}, LEVEL, inference, level_a=cl.level)
        out.append({**rec, **met, 'status': 'ok'})
    return out

def historical_baseline(cl, y):
    """B_ij: each pair's centred pair effect from stage one on the analysed wells (read wells)."""
    p, _, stop = cl.hist.fit(y)
    B = np.full(cl.n_pairs, np.nan)
    if stop in (1, 2):
        B[cl.hist.pairs] = p
    return B

def family_half_contributions(cl, y):
    out = []
    for sub in cl.family_halves:
        if not len(sub.pairs):
            out.append(None)
            continue
        p, _, stop = sub.fit(y)
        out.append((p, half_arrays(cl, sub.pairs, p)) if stop in (1, 2) else None)
    return out

def ranking_molecules(cl, side):
    """Molecules (layout indices) with degree >= 2 in the piece of every family half (the
    design's count)."""
    common = None
    for sub in cl.family_halves:
        codes = cl.pe[sub.pairs] if side == 'elec' else cl.pn[sub.pairs]
        v, c = np.unique(codes, return_counts=True)
        keep = set(v[c >= 2])
        common = keep if common is None else common & keep
    return np.array(sorted(common or ()), int)

def ranking_statistics(vals, side, idx):
    a1, a2, b1, b2 = (v[side][idx] for v in vals)
    raa, rbb = (spearman(a1, a2), spearman(b1, b2))
    cross = float(np.mean([spearman(x, z) for x in (a1, a2) for z in (b1, b2)]))
    g = math.sqrt(raa * rbb) if raa > 0 and rbb > 0 else np.nan
    return {'rho_AA': raa, 'rho_BB': rbb, 'rho_AB': cross, 'D_C': g - cross if np.isfinite(g) else np.nan, 'kappa_C': cross / g if np.isfinite(g) and g > 0 else np.nan}

def gate_c_ranking(cl, y, inference=True):
    """Section 7.6: within-family and cross-family agreement of contributions from the four family
    halves; D_C with its delete-one-molecule jackknife (95%)."""
    fh = family_half_contributions(cl, y)
    rows = []
    for side in ('elec', 'nuc'):
        mols = ranking_molecules(cl, side)
        rec = {'reaction': cl.rxn, 'family A': cl.fa, 'family B': cl.fb, 'side': side, 'molecules': len(mols)}
        if len(mols) < MIN_MOLECULES:
            rows.append({**rec, 'status': NOT_ESTIMABLE})
            continue
        if any((f is None or f[1] is None for f in fh)):
            rows.append({**rec, 'status': NUMERICAL})
            continue
        vals = [f[1] for f in fh]
        est = ranking_statistics(vals, side, mols)
        se = lo = hi = np.nan
        nj = 0
        if inference and np.isfinite(est['D_C']):
            held = [(set(cl.pe[s.pairs]), set(cl.pn[s.pairs])) for s in cl.family_halves]
            units = sorted({('elec', int(m)) for h in held for m in h[0]} | {('nuc', int(m)) for h in held for m in h[1]})
            dele = []
            for side_del, mol in units:
                vv = []
                for h, (sub, (p, v)) in enumerate(zip(cl.family_halves, fh)):
                    if mol not in held[h][0 if side_del == 'elec' else 1]:
                        vv.append(v)
                        continue
                    keep = deletion(cl, ('family', h), sub.pairs, side_del, mol)
                    vv.append(half_arrays(cl, sub.pairs[keep], p[keep]) if len(keep) >= 3 else None)
                if any((x is None for x in vv)):
                    continue
                idx = mols[mols != mol] if side_del == side else mols
                idx = idx[np.all([np.isfinite(x[side][idx]) for x in vv], axis=0)]
                if len(idx) >= 3:
                    dele.append(ranking_statistics(vv, side, idx)['D_C'])
            se, lo, hi, nj = jackknife(est['D_C'], dele, LEVEL)
        cat = (category(lo, hi) if np.isfinite(est['D_C']) else UNDEFINED) if inference else 'no inference'
        rows.append({**rec, **est, 'D_C se': se, 'D_C low': lo, 'D_C high': hi, 'jackknife n': nj, 'category': cat, 'status': 'ok'})
    return rows

def unit_delta(cl, y, u):
    """Delta of one unit over its pairs: a condition pair's difference, or, for the regime design,
    the difference of the two families' stage-one pair effects within the half."""
    q = cl.unit_subs[u]
    if not len(q):
        return (q, np.array([]))
    unit = cl.units[u]
    if cl.kind != 'regime':
        sel = (cl.d_unit == u) & np.isin(cl.d_pair, q)
        order = np.argsort(cl.d_pair[sel])
        return (cl.d_pair[sel][order], (y[cl.d_a[sel]] - y[cl.d_b[sel]])[order])
    eff = []
    cix = {c: i for i, c in enumerate(cl.conds)}
    for fam in ('A', 'B'):
        ok = np.isin(np.arange(len(cl.conds)), [cix[c] for c in unit[fam]])
        sub = Sub(cl, q, ok, connect=False)
        p, _, stop = sub.fit(y)
        if stop not in (1, 2):
            return (q[:0], np.array([]))
        eff.append(pd.Series(p, index=sub.pairs))
    both = eff[0].index.intersection(eff[1].index)
    return (np.asarray(both), (eff[0][both] - eff[1][both]).values)

def gate_c_covariances(cl, y):
    """Section 7.7: cross-unit covariances of side contributions and of Delta on the common
    evaluation sets of condition-disjoint unit pairs, weighted by their sizes; no truncation."""
    if cl.cov_struct is not None:
        return covariances_matched(cl, y)
    fits = {}
    for u in range(len(cl.units)):
        q, d = unit_delta(cl, y, u)
        if len(q) < 3:
            continue
        v = contributions(cl.e_name[cl.pe[q]], cl.n_name[cl.pn[q]], d)
        if v is not None:
            fits[u] = (pd.Series(d, index=q), v)
    acc, total = (np.zeros(5), 0.0)
    for u, v in cl.disjoint:
        if u not in fits or v not in fits:
            continue
        (du, cu), (dv, cv) = (fits[u], fits[v])
        P = du.index.intersection(dv.index)
        e, n = (cl.e_name[cl.pe[P]], cl.n_name[cl.pn[P]])
        ok = np.isin(e, cu['elec'].index) & np.isin(e, cv['elec'].index) & np.isin(n, cu['nuc'].index) & np.isin(n, cv['nuc'].index)
        if ok.sum() < 3:
            continue
        P, e, n = (P[ok], e[ok], n[ok])
        au, av = (cu['elec'][e].values, cv['elec'][e].values)
        bu, bv = (cu['nuc'][n].values, cv['nuc'][n].values)
        cov = lambda x, z: float(np.mean((x - x.mean()) * (z - z.mean())))
        terms = np.array([cov(au, av), cov(bu, bv), cov(au, bv), cov(bu, av), cov(du[P].values, dv[P].values)])
        acc += len(P) * terms
        total += len(P)
    return covariance_record(cl, acc, total)

def covariance_record(cl, acc, total):
    rec = {'reaction': cl.rxn, 'family A': cl.fa, 'family B': cl.fb, 'unit pairs': len(cl.disjoint), 'evaluation pairs': total}
    if total == 0:
        return {**rec, 'status': NOT_ESTIMABLE}
    C = acc / total
    add = C[:4].sum()
    rec.update({'C_EE': C[0], 'C_NN': C[1], 'C_EN': C[2], 'C_NE': C[3], 'C_Delta': C[4], 'additive': add, 'electrophile ratio': C[0] / add if add > 0 else np.nan, 'additive ratio': add / C[4] if C[4] > 0 else np.nan, 'status': 'ok'})
    rec['ratio outside [0, 1]'] = bool(any((np.isfinite(x) and (not 0 <= x <= 1) for x in (rec['electrophile ratio'], rec['additive ratio']))))
    return rec

def covariances_matched(cl, y):
    """gate_c_covariances for matched designs on the precomputed structure."""
    units, pairs = cl.cov_struct
    dv = y[cl.d_a] - y[cl.d_b]
    fit = {}
    for u, U in enumerate(units):
        if U is None:
            continue
        d = dv[U['sel']]
        f = stage_two(U['ei'], U['ni'], len(U['le']), len(U['ln']), d)
        if f is not None:
            fit[u] = (d, f[0], f[1])
    acc, total = (np.zeros(5), 0.0)
    cov = lambda x, z: float(np.mean((x - x.mean()) * (z - z.mean())))
    for u, v, iu, iv, eu, ev, nu, nv in pairs:
        if u not in fit or v not in fit:
            continue
        (du, au, bu), (dw, av, bv) = (fit[u], fit[v])
        x1, x2, z1, z2 = (au[eu], av[ev], bu[nu], bv[nv])
        terms = np.array([cov(x1, x2), cov(z1, z2), cov(x1, z2), cov(z1, x2), cov(du[iu], dw[iv])])
        acc += len(iu) * terms
        total += len(iu)
    return covariance_record(cl, acc, total)

def gate_d(layouts, fits, k=2):
    """Section 8: each molecule's reputation from its standardized contributions in the other
    reactions, against its contribution in the reaction at hand; pooled, by reaction and by the
    number of other reactions, with tail odds ratios and leave-one-reaction-out ranges."""
    z = {}
    for rc, f in fits.items():
        if f['full_s'] is None:
            continue
        lay = layouts[rc]
        for side in ('elec', 'nuc'):
            s = f['full_s'][side]
            names = lay.e_name if side == 'elec' else lay.n_name
            deg = pd.Series(lay.deg[side], index=names)
            s = s[deg.reindex(s.index).values >= k]
            if len(s) >= 2 and s.std() > 0:
                z[rc, side] = (s - s.mean()) / s.std()
    rows = []
    for (rc, side), s in z.items():
        for m, v in s.items():
            others = [z[r2, side][m] for r2, sd in z if sd == side and r2 != rc and (m in z[r2, side].index)]
            if others:
                rows.append({'reaction': rc, 'side': side, 'molecule': m, 'value': v, 'reputation': float(np.mean(others)), 'other reactions': len(others)})
    D = pd.DataFrame(rows)
    if D.empty:
        return (D, pd.DataFrame())
    D['stratum'] = D['other reactions'].clip(upper=3)

    def summary(g):
        rho = spearman(g.value, g.reputation) if len(g) >= MIN_MOLECULES else np.nan
        low_r, low_g = (g.value <= g.value.quantile(0.2), g.reputation <= g.reputation.quantile(0.2))
        top_r, top_g = (g.value >= g.value.quantile(0.8), g.reputation >= g.reputation.quantile(0.8))

        def odds(x, w):
            a, b = ((x & w).sum() + 0.5, (x & ~w).sum() + 0.5)
            c, d = ((~x & w).sum() + 0.5, (~x & ~w).sum() + 0.5)
            return float(a * d / (b * c))
        return pd.Series({'molecules': len(g), 'rho': rho, 'bottom-fifth OR': odds(low_r, low_g), 'top-fifth OR': odds(top_r, top_g)})
    S = [summary(D).rename('pooled')]
    S += [summary(g).rename(f'reaction {rc}') for rc, g in D.groupby('reaction')]
    S += [summary(g).rename(f'other reactions {k2}') for k2, g in D.groupby('stratum')]
    loro = [spearman(*D[D.reaction != rc][['value', 'reputation']].T.values) for rc in sorted(D.reaction.unique())]
    out = pd.DataFrame(S)
    out['loro min'], out['loro max'] = (float(np.nanmin(loro)), float(np.nanmax(loro)))
    return (D, out)

def analysed_set():
    c = pd.read_csv(L.OUT / 'extended_conditions_bba.csv', dtype={'condition_id': str})
    return set(zip(c.rxn_class, c.condition_id))

def reaction_layouts(meta):
    """The fifteen analysed designs, from usable-well metadata (no yield)."""
    P = pd.read_csv(L.OUT / 'extended_pairs_bba.csv')
    C = pd.read_csv(L.OUT / 'extended_conditions_bba.csv', dtype={'condition_id': str})
    out = {}
    for rc, p in P.groupby('rxn_class'):
        conds = list(C[C.rxn_class == rc].condition_id)
        w = meta[(meta.rxn_class == rc) & meta.condition_id.isin(conds)].merge(p[['elec', 'nuc']], on=['elec', 'nuc'])
        out[rc] = ReactionLayout(rc, w, p, conds, w.plate if 'plate' in w else None)
    return out

