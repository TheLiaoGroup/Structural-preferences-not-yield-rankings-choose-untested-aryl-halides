"""Functions of the heterogeneity test (Cochran's Q, condition-half agreement, noise calibration and the
simulated cross-reaction agreement), as in code/core/gated_secondary_bba.py, here bound to the sorted
layouts of code/calibration/gated_layouts.py.
"""
import numpy as np
import pandas as pd
import gated_layouts as G

HETEROGENEITY_SEED = 20261101

def cochran_q(O):
    """Cochran's Q over Fisher-z values of observed rho_R, weights n - 3."""
    O = O[np.isfinite(O.rho_R) & (O.shared > 3)]
    z = np.arctanh(np.clip(O.rho_R.values, -0.999999, 0.999999))
    w = O.shared.values - 3.0
    return float(np.sum(w * (z - np.sum(w * z) / np.sum(w)) ** 2))

def condition_half_agreement(lay, prm, rng, sims):
    """Mean over simulations and both sides of the Spearman correlation between contributions from
    the two condition halves -- the simulated counterpart of r_half in extended_reliability_bba.csv."""
    import validate_gated_bba as V
    vals = []
    for _ in range(sims):
        y, _ = V.draw_reaction(lay, prm, rng)
        h = [G.half_arrays(lay, s.pairs, s.fit(y)[0]) for s in lay.cond_halves]
        r = []
        for side in ("elec", "nuc"):
            ok = np.isfinite(h[0][side]) & np.isfinite(h[1][side])
            r.append(G.spearman(h[0][side][ok], h[1][side][ok]))
        vals.append(np.nanmean(r))
    return float(np.nanmean(vals))

def calibrate_reaction(lay, target, floor, rng, sims=50, steps=12):
    """Noise ratio v for one reaction, by bisection on log v, so that the simulated condition-half
    agreement matches the frozen one; the floor fraction is the observed one."""
    import validate_gated_bba as V
    lo, hi = np.log(0.01), np.log(100.0)
    for _ in range(steps):
        mid = (lo + hi) / 2
        got = condition_half_agreement(lay, {**V.BASE, "v": float(np.exp(mid)), "f": floor}, rng, sims)
        lo, hi = (mid, hi) if got > target else (lo, mid)
    return float(np.exp((lo + hi) / 2))

def simulated_transfer(layouts, prm_of, rho, rng, names, index):
    """Observed rho_R of the 58 comparisons in one simulation: latent effects shared across
    reactions with correlation rho, each reaction on its own layout, noise and floor."""
    import validate_gated_bba as V
    shared = {s: rng.normal(0, 1.0, len(names[s])) for s in ("elec", "nuc")}
    full = {}
    for rc, lay in layouts.items():
        prm = prm_of[rc]
        mix = lambda s, var, n: np.sqrt(var) * (np.sqrt(rho) * shared[s][index[rc][s]]
                                                + np.sqrt(1 - rho) * rng.normal(0, 1.0, n))
        y, _ = V.draw_reaction(lay, prm, rng, mix("elec", prm["sa2"], len(lay.e_name)),
                               mix("nuc", prm["sb2"], len(lay.n_name)))
        p, _, stop = lay.full.fit(y)
        arr = G.half_arrays(lay, lay.full.pairs, p)
        full[rc] = {s: G.named(lay, arr, s) for s in ("elec", "nuc")}
    rows = []
    for side, ra, rb in G.comparison_list():
        a, b = full[ra][side], full[rb][side]
        m = a.index.intersection(b.index)
        rows.append((side, ra, rb, len(m), G.spearman(a[m], b[m])))
    return pd.DataFrame(rows, columns=["side", "rxn_a", "rxn_b", "shared", "rho_R"])
