"""Secondary and sensitivity analyses of the cross-reaction comparisons, including the heterogeneity test."""
import itertools

import numpy as np

import pandas as pd

from rdkit import Chem

import analysis_bba as A

import gated_bba as G

import lib_bba as L

EXCLUDED_FOR_FLOOR = ("SNAr", "Chan-Lam")

ARYL_GRIGNARD = Chem.MolFromSmarts("[Mg]~c")

HETEROGENEITY_SEED = 20261101

BASES = ("K3PO4", "KF", "KOtBu", "Na2CO3")

OTHER = ("dinaphtho[2,1-b:1',2'-d]furan",)

LIGANDS = ("(S)-AntPhos", "1,10-Phen", "2,6-difluoroanilino(oxo)acetic acid", "AMidolePhos", "AdBippyPhos",
           "BINAM", "BTMPO", "BrettPhos", "CyJohnPhos", "CyPF-tBu", "DMPAO", "DPEO", "IPr.HCl", "JohnPhos",
           "L-proline", "Me4Phen", "Me4tBuXPhos", "MorDalPhos", "N(Me2)PhP(tBu)2", "N,N-dimethylglycine",
           "N-phenylpicolinamide", "PCy3", "RockPhos", "XPhos", "XantPhos", "dipivaloylmethane",
           "pivaloyltrifluoroacetone", "tBuBrettPhos", "tBuDavePhos", "tBuXphos", "trans-1,2-Cy-(NHMe)2")

ROLE = {**{n: "ligand" for n in LIGANDS}, **{n: "base" for n in BASES}, **{n: "other" for n in OTHER}}

def transform(y, scale):
    """The scales of Sections 6.6 and 7.8: as recorded, logit (clipped at 0.01 and 0.99, as
    analysis_bba.transform), and the indicator of the pre-specified yield threshold of 0.20."""
    if scale == "raw":
        return np.asarray(y, float)
    if scale == "logit":
        return A.transform(np.asarray(y, float), "logit")
    return (np.asarray(y, float) >= 0.20).astype(float)

def align(layout, rows):
    """Yields of `rows` in the order of the layout's wells; every well must be found once."""
    idx = pd.DataFrame({"elec": layout.e_name[layout.pe[layout.w_pair]], "nuc": layout.n_name[layout.pn[layout.w_pair]],
                        "condition_id": [layout.conds[c] for c in layout.w_cond]}).reset_index()
    got = idx.merge(rows[["elec", "nuc", "condition_id", "yield"]], on=["elec", "nuc", "condition_id"])
    assert len(got) == len(idx) and got["index"].is_unique, "a well is missing or duplicated"
    return got.sort_values("index")["yield"].values

def pooled(O, R):
    out = {f"observed {k}": v for k, v in G.pooled_observed(O, bootstrap=0).items()}
    if len(R):
        out.update({f"relative {k}": v for k, v in G.pooled_relative(R).items()})
    return out

def gate_b_secondary(layouts, y_of, A_cells, k=2):
    """6.1 design-eligible comparisons and the historical correction, 6.3 the selected subset, 6.6
    the logit and threshold scales and the set without SNAr and Chan-Lam coupling. No pass lines."""
    rows = []
    fits = {rc: G.reaction_fits(lay, y_of[rc]) for rc, lay in layouts.items()}
    base = G.gate_b(layouts, fits, k, bootstrap=0)
    O, R = base["observed"], base["relative"]
    eligible = O.merge(R[["side", "rxn_a", "rxn_b"]], on=["side", "rxn_a", "rxn_b"])
    rows.append({"analysis": "design-eligible comparisons (6.1)", **pooled(eligible, R)})
    frozen = pd.read_csv(L.OUT / "extended_transfer_bba.csv")
    cross = frozen[~frozen.same_reaction]
    rows.append({"analysis": "historical correction, frozen values (6.1)",
                 "observed weighted": G.weighted_mean(cross.rho_disattenuated, cross.n_shared),
                 "observed comparisons": len(cross)})
    ok = A_cells[(A_cells.degree == k) & (A_cells.category == "supporting evidence")]
    good = set(zip(ok.reaction, ok.side))
    sel = R[[(a, s) in good and (b, s) in good for s, a, b in zip(R.side, R.rxn_a, R.rxn_b)]]
    rows.append({"analysis": "selected on the same data: both reaction-sides with supporting rho_P (6.3)",
                 **pooled(O.merge(sel[["side", "rxn_a", "rxn_b"]], on=["side", "rxn_a", "rxn_b"]), sel)})
    keep = lambda D: D[~D.rxn_a.isin(EXCLUDED_FOR_FLOOR) & ~D.rxn_b.isin(EXCLUDED_FOR_FLOOR)]
    rows.append({"analysis": "without SNAr and Chan-Lam coupling (6.6)", **pooled(keep(O), keep(R))})
    for scale in ("logit", "threshold 0.20"):
        f2 = {rc: G.reaction_fits(lay, transform(y_of[rc], scale)) for rc, lay in layouts.items()}
        b2 = G.gate_b(layouts, f2, k, bootstrap=0)
        rows.append({"analysis": f"{scale} scale (6.6)", **pooled(b2["observed"], b2["relative"])})
    return pd.DataFrame(rows), O

def grouping(O):
    """6.4: the palladium contrast among aryl-halide electrophiles, with its exact node-label
    permutation p (all labellings of the non-palladium reactions), with and without Kumada
    coupling; and the N-H contrast, descriptive."""
    lab = pd.read_csv(L.OUT / "reaction_labels_bba.csv", index_col=0, encoding="utf-8")
    out = []
    E_ = O[O.side == "elec"]
    for label, drop in (("aryl halides", ()), ("aryl halides without Kumada coupling", ("Kumada",))):
        aryl = [r for r in lab.index[lab.electrophile == "aryl halide or sulfonate"] if r not in drop]
        pd_ = {r for r in aryl if lab.metal[r] == "Pd"}
        C = E_[E_.rxn_a.isin(aryl) & E_.rxn_b.isin(aryl)]

        def contrast(pd_set):
            n = C.rxn_a.isin(pd_set).astype(int) + C.rxn_b.isin(pd_set).astype(int)
            return (G.weighted_mean(C.rho_R[n == 2], C.shared[n == 2])
                    - G.weighted_mean(C.rho_R[n == 1], C.shared[n == 1]))
        observed = contrast(pd_)
        others = len(aryl) - len(pd_)
        perms = [contrast(set(aryl) - set(o)) for o in itertools.combinations(aryl, others)]
        out.append({"grouping": label, "reactions": len(aryl), "palladium": len(pd_), "comparisons": len(C),
                    "contrast": observed, "labellings": len(perms),
                    "p": float(np.mean([v >= observed - 1e-12 for v in perms]))})
    N_ = O[O.side == "nuc"]
    nh = [r for r in lab.index if lab.nucleophile[r] == "N–H"]
    C = N_[N_.rxn_a.isin(nh) & N_.rxn_b.isin(nh)]
    pdn = {r for r in nh if lab.metal[r] == "Pd"}
    n = C.rxn_a.isin(pdn).astype(int) + C.rxn_b.isin(pdn).astype(int)
    out.append({"grouping": "N-H nucleophiles (descriptive)", "reactions": len(nh), "palladium": len(pdn),
                "comparisons": len(C), "contrast": G.weighted_mean(C.rho_R[n == 0], C.shared[n == 0])
                - G.weighted_mean(C.rho_R[n == 1], C.shared[n == 1])})
    return pd.DataFrame(out)

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

def heterogeneity(layouts, y_of, O, sims_per_step=200, draws=10000, calib_sims=50, steps=12):
    """Section 6.5: Q of the observed comparisons against its distribution under a latent
    between-reaction correlation rho* calibrated on the observed scale."""
    import validate_gated_bba as V
    rng = np.random.default_rng(HETEROGENEITY_SEED)
    rel = pd.read_csv(L.OUT / "extended_reliability_bba.csv").groupby("rxn_class").r_half.mean()
    prm_of = {}
    for rc, lay in layouts.items():
        floor = float(np.mean(np.asarray(y_of[rc]) <= 0.01))
        v = calibrate_reaction(lay, float(rel[rc]), floor, rng, calib_sims, steps)
        prm_of[rc] = {**V.BASE, "v": v, "f": floor}
    names = {s: sorted(set().union(*[set(l.e_name if s == "elec" else l.n_name) for l in layouts.values()]))
             for s in ("elec", "nuc")}
    index = {rc: {s: np.searchsorted(names[s], l.e_name if s == "elec" else l.n_name) for s in ("elec", "nuc")}
             for rc, l in layouts.items()}
    target = G.pooled_observed(O, inference=False)["weighted"]
    lo, hi = 0.0, 1.0
    for _ in range(steps):
        rho = (lo + hi) / 2
        got = np.mean([G.pooled_observed(simulated_transfer(layouts, prm_of, rho, rng, names, index),
                                         inference=False)["weighted"] for _ in range(sims_per_step)])
        if abs(got - target) < 0.005:
            break
        lo, hi = (rho, hi) if got < target else (lo, rho)
    q_obs = cochran_q(O)
    q_sim = np.array([cochran_q(simulated_transfer(layouts, prm_of, rho, rng, names, index)) for _ in range(draws)])
    return pd.DataFrame([{"Q": q_obs, "comparisons": int((np.isfinite(O.rho_R) & (O.shared > 3)).sum()),
                          "rho*": rho, "calibrated pooled rho_R": got, "observed pooled rho_R": target,
                          "simulations": draws, "p": float(np.mean(q_sim >= q_obs))}]
                        ).assign(**{f"v {rc}": prm_of[rc]["v"] for rc in prm_of},
                                 **{f"floor {rc}": prm_of[rc]["f"] for rc in prm_of})

def one_role(units, role):
    """Near-matched condition pairs whose accompanying difference is one reagent of `role`, exchanged
    or present on one side only."""
    keep = []
    for a, b in zip(units["context only in A"].fillna(""), units["context only in B"].fillna("")):
        A_, B_ = ([x.strip() for x in str(s).split(";") if x.strip()] for s in (a, b))
        missing = [x for x in A_ + B_ if x not in ROLE]
        assert not missing, f"reagents without a pre-specified role: {missing}"
        keep.append(0 < len(A_) + len(B_) and len(A_) <= 1 and len(B_) <= 1 and all(ROLE[x] == role for x in A_ + B_))
    return units[np.array(keep, bool)]

def rebuilt(cl, keep):
    units, wells, hist_pairs, hist_conds, families = cl.args
    return G.ContrastLayout(cl.rxn, cl.fa, cl.fb, units, wells[keep].reset_index(drop=True), hist_pairs,
                            hist_conds, families)

def contrast_rows(cl, y, label):
    B = G.historical_baseline(cl, y)
    rows = [{"sensitivity": label, **r} for r in G.gate_c_prediction(cl, y, B)]
    rows += [{"sensitivity": label, "analysis": "ranking", **r} for r in G.gate_c_ranking(cl, y)]
    rows.append({"sensitivity": label, "analysis": "covariances", **G.gate_c_covariances(cl, y)})
    return rows

def gate_c_sensitivity(cls, y_c):
    rows = []
    for key, cl in cls.items():
        if cl.kind == "regime":
            continue
        y = y_c[key]
        rows += contrast_rows(cl, transform(y, "logit"), "logit scale")
        wells = cl.args[1]
        if cl.rxn == "Kumada":
            aryl = {n: Chem.MolFromSmiles(n).HasSubstructMatch(ARYL_GRIGNARD) for n in set(wells.nuc)}
            keep = wells.nuc.map(aryl).values
            rows += contrast_rows(rebuilt(cl, keep), y[keep], "aryl Grignard reagents only")
        if cl.kind == "near":
            units, _, hist_pairs, hist_conds, families = cl.args
            for role in ("ligand", "base"):
                u = one_role(units, role)
                label = f"condition pairs differing by one {role} ({len(u)} of {len(units)})"
                if len(u):
                    rows += contrast_rows(G.ContrastLayout(cl.rxn, cl.fa, cl.fb, u, wells, hist_pairs, hist_conds,
                                                           families), y, label)
                else:
                    rows.append({"sensitivity": label, "reaction": cl.rxn, "status": G.NOT_ESTIMABLE})
        in_contrast = wells.condition_id.isin(set(cl.args[0]["condition A"]) | set(cl.args[0]["condition B"]))
        physical = G.DG.physical_plate_ids(wells.plate)
        plates = wells.assign(physical_plate=physical)[in_contrast & physical.ne("")].groupby(["elec", "nuc"]).physical_plate.nunique()
        split = set(plates[plates > 1].index)
        if split:
            keep = np.array([(e, n) not in split for e, n in zip(wells.elec, wells.nuc)])
            rows += contrast_rows(rebuilt(cl, keep), y[keep], f"without the {len(split)} pairs on two plates")
    return pd.DataFrame(rows)
