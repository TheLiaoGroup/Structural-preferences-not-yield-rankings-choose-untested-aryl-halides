"""Two-way fits of well yields to pair and condition effects, the pair-to-molecule stage and the descriptive decomposition of the six dense reactions."""
import numpy as np

import pandas as pd

from scipy import sparse

from scipy.sparse.linalg import lsqr

from scipy.stats import norm, spearmanr, ttest_1samp

import lib_bba as L

SEED = 20260925

MIN_JACKKNIFE_DF = 20

MAX_SKIPPED = 0.05

ADDITIVE_THRESHOLD = 0.5

CLIP = 0.01

ALPHA = 0.01

def transform(y, scale):
    if scale == "raw":
        return y
    z = np.clip(y, CLIP, 1 - CLIP)
    return np.log(z / (1 - z))

def two_way_fit(values, left, right):
    """Least squares of `values` on indicators of `left` and `right`; returns the left
    effects, centred, indexed by the categories of `left`."""
    li, l_names = pd.factorize(np.asarray(left))
    ri, r_names = pd.factorize(np.asarray(right))
    n, p = len(values), len(l_names) + len(r_names)
    rows = np.repeat(np.arange(n), 2)
    cols = np.empty(2 * n, dtype=np.int64)
    cols[0::2], cols[1::2] = li, len(l_names) + ri
    X = sparse.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, p))
    beta = lsqr(X, np.asarray(values, float), atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    effects = beta[:len(l_names)]
    return pd.Series(effects - effects.mean(), index=l_names)

def pair_effects(rows, scale="raw", conditions=None):
    """Stage one, per reaction: every pair's effect, condition effects removed."""
    out = []
    for rc, g in rows.groupby("rxn_class"):
        if conditions is not None:
            g = g[g.condition_id.isin(conditions[rc])]
        eff = two_way_fit(transform(g["yield"].values, scale),
                          g.elec + " >> " + g.nuc, g.condition_id)
        key = pd.DataFrame([k.split(" >> ") for k in eff.index], columns=["elec", "nuc"])
        out.append(key.assign(rxn_class=rc, p=eff.values))
    return pd.concat(out, ignore_index=True)

def ceiling(rows, scale, order):
    """Split each reaction's conditions by parity of coverage rank, fit stage one on each
    half, and return the error variance of the full pair effect and the split-half
    correlation. sigma2 = Var(pA - pB) / 4, the Spearman-Brown value."""
    halves = {}
    for parity in (0, 1):
        sel = {rc: set(g[g["rank"] % 2 == parity].condition_id) for rc, g in order.groupby("rxn_class")}
        halves[parity] = pair_effects(rows, scale, sel).rename(columns={"p": f"p{parity}"})
    both = halves[0].merge(halves[1], on=["rxn_class", "elec", "nuc"])
    out = []
    for rc, g in both.groupby("rxn_class"):
        spread = float(np.var(g.p0 - g.p1, ddof=1))
        out.append({"rxn_class": rc, "n_pairs": len(g), "sigma2_error": spread / 4,
                    "half_correlation": float(np.corrcoef(g.p0, g.p1)[0, 1]) if spread > 0 else np.nan})
    return pd.DataFrame(out)

def stage_two(pairs):
    """Fit p = a_elec + b_nuc + d on a connected set of pairs, with the hat diagonal that
    leave-one-pair-out needs. The design carries an intercept and drops one level per side,
    so it is full rank and can be factorised by QR."""
    ei, e_names = pd.factorize(pairs.elec.values)
    ni, n_names = pd.factorize(pairs.nuc.values)
    n = len(pairs)
    X = np.zeros((n, 1 + (len(e_names) - 1) + (len(n_names) - 1)))
    X[:, 0] = 1.0
    rows = np.arange(n)
    keep_e, keep_n = ei > 0, ni > 0
    X[rows[keep_e], ei[keep_e]] = 1.0
    X[rows[keep_n], len(e_names) - 1 + ni[keep_n]] = 1.0
    y = pairs.p.values
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    d = y - X @ beta
    h = np.einsum("ij,ij->i", Q, Q)
    a = np.concatenate([[0.0], beta[1:len(e_names)]])
    b = np.concatenate([[0.0], beta[len(e_names):]])
    return {"d": d, "h": h, "y": y, "df_int": n - X.shape[1],
            "ss_total": float(np.sum((y - y.mean()) ** 2)), "ss_resid": float(d @ d),
            "df_total": n - 1, "n_e": len(e_names), "n_n": len(n_names),
            "a": pd.Series(a - a.mean(), index=e_names), "b": pd.Series(b - b.mean(), index=n_names),
            "ei": ei, "ni": ni}

def shares(fit, sigma2):
    """Additive and interaction share of the pair-effect variance, measurement error removed."""
    if fit["df_int"] <= 0:
        return np.nan, np.nan, np.nan, np.nan
    V = fit["ss_total"] / fit["df_total"]
    R = fit["ss_resid"] / fit["df_int"]
    denom = V - sigma2
    if V < 1e-10 or denom <= 0:
        return np.nan, np.nan, V, R
    s_int = (R - sigma2) / denom
    return 1 - s_int, s_int, V, R

def leave_one_pair_out(fit):
    """Exact leave-one-pair-out RMSE of the additive model and of the reaction mean."""
    usable = fit["h"] < 1 - 1e-8
    if usable.sum() < 3:
        return np.nan, np.nan, 0, int(len(usable))
    y = fit["y"]
    err_add = fit["d"][usable] / (1 - fit["h"][usable])
    n = len(y)
    err_mean = (y - (n * y.mean() - y) / (n - 1))[usable]
    return (float(np.sqrt(np.mean(err_mean ** 2))), float(np.sqrt(np.mean(err_add ** 2))),
            int(usable.sum()), int((~usable).sum()))

def statistics(pairs, sigma2):
    """The two tested quantities from one set of pairs."""
    fit = stage_two(pairs)
    base, add, _, _ = leave_one_pair_out(fit)
    return shares(fit, sigma2)[0], base - add, fit

def jackknife(pairs, sigma2):
    """Standard errors by deleting one molecule at a time with all of its pairs. A molecule
    bootstrap would resample away a third of each side and distort a variance ratio; deleting
    one molecule leaves the grid almost intact, which is what this statistic needs."""
    values, skipped = [], 0
    molecules = [("elec", m) for m in pairs.elec.unique()] + [("nuc", m) for m in pairs.nuc.unique()]
    for side, mol in molecules:
        sub = pairs[pairs[side] != mol]
        if len(sub) < 3:
            skipped += 1
            continue
        keep = L.largest_component(sub)
        fit = stage_two(keep)
        if fit["df_int"] < MIN_JACKKNIFE_DF:
            skipped += 1
            continue
        s, g, _ = statistics(keep, sigma2)
        if not (np.isfinite(s) and np.isfinite(g)):
            skipped += 1
            continue
        values.append((s, g))
    if len(values) < 10:
        return np.array([np.nan, np.nan]), len(molecules), skipped
    V = np.asarray(values)
    m = len(V)
    se = np.sqrt((m - 1) / m * np.sum((V - V.mean(0)) ** 2, axis=0))
    return se, len(molecules), skipped

def one_sided_p(estimate, se, null):
    if not np.isfinite(estimate) or not np.isfinite(se) or se <= 0:
        return np.nan
    return float(norm.sf((estimate - null) / se))

def holm(p):
    p = np.asarray(p, float)
    out, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(np.where(np.isnan(p), 1.0, p))):
        running = max(running, min(1.0, (len(p) - rank) * (1.0 if np.isnan(p[i]) else p[i])))
        out[i] = running
    return out

def run(rows, order, scale="raw", label="primary", inference=True):
    """Every pre-registered computation on one scale."""
    P = pair_effects(rows, scale)
    C = ceiling(rows, scale, order).set_index("rxn_class")
    results, molecules, spectra = [], [], []
    for rc in L.REACTIONS:
        pairs = P[P.rxn_class == rc].reset_index(drop=True)
        sigma2 = C.loc[rc, "sigma2_error"]
        s_add, gain, fit = statistics(pairs, sigma2)
        base, add, n_pred, n_skip = leave_one_pair_out(fit)
        se, n_mol, skipped = (jackknife(pairs, sigma2) if inference
                              else (np.array([np.nan, np.nan]), 0, 0))
        share_skipped = skipped / max(1, n_mol)
        descriptive = share_skipped > MAX_SKIPPED
        for test, est, err, null, extra in (
                ("H1", s_add, se[0], ADDITIVE_THRESHOLD,
                 {"n_pairs": len(pairs), "interaction_df": fit["df_int"],
                  "var_pair_effect": shares(fit, sigma2)[2], "resid_mean_square": shares(fit, sigma2)[3],
                  "sigma2_error": sigma2, "reliability": C.loc[rc, "half_correlation"]}),
                ("H2", gain, se[1], 0.0,
                 {"rmse_mean_baseline": base, "rmse_additive": add, "n_predicted": n_pred,
                  "n_not_predictable": n_skip, "ceiling_rmse": float(np.sqrt(sigma2))})):
            results.append({"test": test, "scale": scale, "label": label, "rxn_class": rc,
                            "estimate": est, "se": err,
                            "ci_low": est - 1.96 * err, "ci_high": est + 1.96 * err,
                            "p": one_sided_p(est, err, null),
                            "n_molecules": n_mol, "jackknife_skipped": share_skipped,
                            "descriptive_only": descriptive, **extra})
        results.append({"test": "H3", "scale": scale, "label": label, "rxn_class": rc,
                        "estimate": float(np.var(fit["a"].values, ddof=1) / np.var(fit["b"].values, ddof=1)),
                        "var_electrophile": float(np.var(fit["a"].values, ddof=1)),
                        "var_nucleophile": float(np.var(fit["b"].values, ddof=1)),
                        "n_electrophiles": fit["n_e"], "n_nucleophiles": fit["n_n"]})
        for side, codes, names, eff in (("elec", fit["ei"], fit["a"].index, fit["a"]),
                                        ("nuc", fit["ni"], fit["b"].index, fit["b"])):
            g = pd.DataFrame({"code": codes, "abs_d": np.abs(fit["d"])}).groupby("code")
            idx = g.abs_d.mean().index.values
            molecules.append(pd.DataFrame({
                "scale": scale, "label": label, "rxn_class": rc, "side": side,
                "mol": np.asarray(names)[idx], "mean_abs_interaction": g.abs_d.mean().values,
                "n_pairs": g.size().values, "main_effect": eff.values[idx]}))
        M = np.zeros((fit["n_e"], fit["n_n"]))
        M[fit["ei"], fit["ni"]] = fit["d"]
        sv = np.linalg.svd(M, compute_uv=False)
        spectra.append({"scale": scale, "label": label, "rxn_class": rc,
                        "first_singular_share": float(sv[0] ** 2 / np.sum(sv ** 2)),
                        "n_singular_values": len(sv)})
    res = pd.DataFrame(results)
    # The claim is about the six reactions together. Each reaction's own interval is wide --
    # a reaction-level test would only detect additivity far above the threshold -- so the
    # test is a one-sample t over the six reaction estimates, with the reaction as the unit
    # of replication. The per-reaction rows are reported but carry no pass line.
    macro = []
    for test, null in (("H1", ADDITIVE_THRESHOLD), ("H2", 0.0)):
        vals = res.loc[res.test.eq(test), "estimate"].astype(float).values
        good = vals[np.isfinite(vals)]
        t, p = ttest_1samp(good, null, alternative="greater") if len(good) >= 3 else (np.nan, np.nan)
        se = float(np.std(good, ddof=1) / np.sqrt(len(good))) if len(good) >= 2 else np.nan
        macro.append({"test": f"{test} (six reactions)", "scale": scale, "label": label,
                      "rxn_class": "pooled", "estimate": float(np.mean(good)), "se": se,
                      "ci_low": float(np.mean(good)) - 2.571 * se,
                      "ci_high": float(np.mean(good)) + 2.571 * se,
                      "p": float(p), "t": float(t), "df": len(good) - 1,
                      "n_reactions": len(good),
                      "per_reaction": ", ".join(f"{v:.3f}" for v in vals)})
    res = pd.concat([res, pd.DataFrame(macro)], ignore_index=True)
    family = res.test.str.endswith("(six reactions)")
    assert int(family.sum()) == 2, f"Holm family has {int(family.sum())} tests, not 2"
    res.loc[family, "p_holm"] = holm(res.loc[family, "p"].values)
    direction = res.estimate > np.where(res.test.str.startswith("H1"), ADDITIVE_THRESHOLD, 0.0)
    res.loc[family, "criterion_met"] = (direction & (res.p_holm < ALPHA))[family]
    return {"results": res, "pair_effects": P.assign(scale=scale, label=label),
            "ceiling": C.reset_index().assign(scale=scale, label=label),
            "molecules": pd.concat(molecules, ignore_index=True),
            "residual_spectrum": pd.DataFrame(spectra)}

def secondary(rows, order, tables):
    """S1 transfer of a molecule's main effect between reactions; S2 ten and thirty conditions."""
    P = tables["pair_effects"]
    out = []
    for side in ("elec", "nuc"):
        eff = {rc: stage_two(P[P.rxn_class == rc].reset_index(drop=True))["a" if side == "elec" else "b"]
               for rc in L.REACTIONS}
        for i, a in enumerate(L.REACTIONS):
            for b in L.REACTIONS[i + 1:]:
                shared = eff[a].index.intersection(eff[b].index)
                if len(shared) >= 5:
                    out.append({"side": side, "rxn_a": a, "rxn_b": b, "n_shared": len(shared),
                                "spearman": float(spearmanr(eff[a][shared], eff[b][shared]).statistic)})
    s2 = []
    for k in (10, 30):
        sel = {rc: set(g[g["rank"] <= k].condition_id) for rc, g in order.groupby("rxn_class")}
        sub = rows[[c in sel[r] for r, c in zip(rows.rxn_class, rows.condition_id)]]
        s2.append(run(sub, order[order["rank"] <= k], "raw", label=f"{k} conditions")["results"])
    return {"S1_transfer": pd.DataFrame(out), "S2_conditions": pd.concat(s2, ignore_index=True)}

def write(tables, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}_bba.csv", index=False)
