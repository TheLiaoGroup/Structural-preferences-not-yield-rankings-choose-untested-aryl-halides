"""Structural profiles across partner halves (Supplementary Note 18; Fig. 3b,c and Fig. 2b).

When an aryl halide's partners change, how well does a reaction's structural profile hold, compared
with the ranking of the individual aryl halides? The functions reuse the release's design, partner
split, two-stage fits, evaluability rule, Spearman (twelve-decimal ties) and aryl-halide feature
definitions. The only additions are an optional salt in the partner-split hash, layouts on the
pairs common to two condition sets, and a same-permutation null.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "code" / "extended"))
sys.path.insert(0, str(PACKAGE / "code" / "core"))
import public_common as PC  # noqa: E402

assert PC.STAGING.resolve() == PACKAGE.resolve(), PC.STAGING
L = E = G = DG = SP = None
FEATS = []
MIN_CLASS = MIN_FEATURES = MIN_MOL = None
SALTS = [f"partner-salt-{i:02d}" for i in range(20)]
PERMUTATIONS = 10000
SIDE = "elec"
_FEATURE_ROWS = {}


def load(output):
    """Import the frozen modules with their scratch outputs directed to this stage's folder."""
    global L, E, G, DG, SP, FEATS, MIN_CLASS, MIN_FEATURES, MIN_MOL
    L, E, G = PC.load_modules(output=output)
    import design_gated_bba as dg
    import structure_profiles as sp
    DG, SP = dg, sp
    FEATS = SP.FEATURES + SP.ALSO
    assert len(FEATS) == 11 and len(set(FEATS)) == 11
    MIN_CLASS, MIN_FEATURES, MIN_MOL = SP.MIN_CLASS, SP.MIN_FEATURES, G.MIN_MOLECULES
    assert (MIN_CLASS, MIN_FEATURES, MIN_MOL) == (3, 5, 10)


def feature_row(smiles):
    """The release's eleven aryl-halide features, or None without an aryl leaving group."""
    if smiles not in _FEATURE_ROWS:
        f = SP.features(smiles)
        _FEATURE_ROWS[smiles] = None if f is None else [int(f[k]) for k in FEATS]
    return _FEATURE_ROWS[smiles]


def partner_halves(pairs, rxn, side, salt=None):
    """DG.partner_halves with an optional salt prepended to the hashed string."""
    other = "nuc" if side == "elec" else "elec"
    pre = "" if salt is None else f"{salt}|"
    order = pairs.assign(key=[hashlib.sha256(f"{pre}{rxn}|{side}|{m}|{q}".encode()).hexdigest()
                              for m, q in zip(pairs[side], pairs[other])])
    order = order.sort_values([side, "key"], kind="stable")
    order["half"] = order.groupby(side).cumcount() % 2
    return [DG.largest_piece(order[order.half == h][["elec", "nuc"]], f"{salt} half {h + 1}: {rxn}, {side}")
            for h in (0, 1)]


def layout_pairs(lay):
    return pd.DataFrame({"elec": lay.e_name[lay.pe], "nuc": lay.n_name[lay.pn]})


def split_matches_release(lay, side=SIDE):
    """The unsalted split reproduces the release's own partner halves edge for edge."""
    pairs = layout_pairs(lay)
    mine = partner_halves(pairs, lay.rxn, side)
    ref = DG.partner_halves(pairs, lay.rxn, side)
    return all(set(map(tuple, a[["elec", "nuc"]].to_numpy())) == set(map(tuple, b[["elec", "nuc"]].to_numpy()))
               for a, b in zip(mine, ref))


def half_fits(lay, y, side=SIDE, salt=None):
    """Stage one and two on each partner half; arrays over the layout's molecules (or None)."""
    pairs = layout_pairs(lay)
    key = pd.MultiIndex.from_frame(pairs)
    out = []
    for h in partner_halves(pairs, lay.rxn, side, salt):
        idx = key.get_indexer(pd.MultiIndex.from_frame(h[["elec", "nuc"]]))
        assert (idx >= 0).all()
        sub = G.Sub(lay, idx, connect=False)
        p, _, stop = sub.fit(y)
        out.append(G.half_arrays(lay, sub.pairs, p) if stop in (1, 2) else None)
    return out


def evaluable(lay, halves, side=SIDE, k=2):
    """The release rule: a contribution in both halves and full-design degree >= k."""
    a, b = halves
    if a is None or b is None:
        return np.array([], int)
    idx = np.flatnonzero(np.isfinite(a[side]) & np.isfinite(b[side]))
    return idx[lay.deg[side][idx] >= k]


def profile(values, F, tested):
    return np.asarray([values[F[:, j] == 1].mean() - values[F[:, j] == 0].mean() for j in tested])


def pearson(x, y):
    x, y = np.asarray(x, float) - np.mean(x), np.asarray(y, float) - np.mean(y)
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 0 else np.nan


def compare(lay, halves, side=SIDE):
    """Molecule ranking and structural profile on one matched set of aryl halides."""
    ev = evaluable(lay, halves, side)
    names = lay.e_name if side == "elec" else lay.n_name
    a, b = halves[0][side] if halves[0] else None, halves[1][side] if halves[1] else None
    row = {"evaluable": len(ev),
           "rho_P_release_rule": G.spearman(a[ev], b[ev]) if len(ev) >= MIN_MOL else np.nan}
    rows_f = [(i, feature_row(names[i])) for i in ev]
    keep = np.array([i for i, f in rows_f if f is not None], int)
    F = np.array([f for _, f in rows_f if f is not None], int).reshape(-1, len(FEATS))
    row["matched"] = len(keep)
    row["no_aryl_leaving_group"] = len(ev) - len(keep)
    if len(keep) < MIN_MOL:
        return row, None
    xa, xb = a[keep], b[keep]
    tested = [j for j in range(len(FEATS)) if (F[:, j] == 1).sum() >= MIN_CLASS and (F[:, j] == 0).sum() >= MIN_CLASS]
    row["features"] = len(tested)
    row["rho_molecule"] = G.spearman(xa, xb)
    row["pearson_molecule"] = pearson(xa, xb)
    if len(tested) < MIN_FEATURES:
        return row, None
    pa, pb = profile(xa, F, tested), profile(xb, F, tested)
    row["r_structure"] = pearson(pa, pb)
    row["spearman_structure"] = float(pd.Series(pa).corr(pd.Series(pb), method="spearman"))
    row["feature_list"] = "; ".join(FEATS[j] for j in tested)
    return row, (xa, xb, F, tested)


def permutation_null(xa, xb, F, tested, seed_text, n=PERMUTATIONS):
    """Whole eleven-feature rows permuted among the matched aryl halides; each draw applies one
    permutation to both halves, keeping prevalence, co-occurrence and class sizes."""
    rng = np.random.default_rng(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    Ft = F[:, tested].astype(float)
    c1 = Ft.sum(0)
    c0 = len(Ft) - c1
    Fp = Ft[np.argsort(rng.random((n, len(Ft))), axis=1)]

    def diffs(x):
        s1 = np.einsum("pmf,m->pf", Fp, x)
        return s1 / c1 - (x.sum() - s1) / c0

    da, db = diffs(xa), diffs(xb)
    da -= da.mean(1, keepdims=True)
    db -= db.mean(1, keepdims=True)
    return (da * db).sum(1) / np.sqrt((da * da).sum(1) * (db * db).sum(1))


def common_layouts(rxn, first, second):
    """Layouts of the pairs present in both condition sets, one per set, sharing one pair order."""
    (w1, c1, p1), (w2, c2, p2) = first, second
    pa, pb = p1[p1.rxn_class == rxn], p2[p2.rxn_class == rxn]
    common = pa.merge(pb[["elec", "nuc"]], on=["elec", "nuc"])[["rxn_class", "elec", "nuc"]].reset_index(drop=True)
    out = {}
    for tag, w, c in (("first", w1, c1), ("second", w2, c2)):
        ww = w[w.rxn_class == rxn].merge(common[["elec", "nuc"]], on=["elec", "nuc"]).reset_index(drop=True)
        cc = c[c.rxn_class == rxn].sort_values("coverage_rank").condition_id.tolist()
        out[tag] = (G.ReactionLayout(rxn, ww, common, cc, ww.plate), ww["yield"].to_numpy(float))
    assert list(out["first"][0].e_name) == list(out["second"][0].e_name)
    assert not set(out["first"][0].conds) & set(out["second"][0].conds), f"{rxn}: condition sets overlap"
    return common, len(pa), len(pb), out
