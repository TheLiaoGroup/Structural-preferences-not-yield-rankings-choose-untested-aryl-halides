"""Independent implementation of the principal core estimates (Supplementary Note 15): alternating means and QR
solves, written separately from the canonical estimators, none of which is imported."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import hashlib,itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata,t as t_dist
PACKAGE=Path(__file__).resolve().parents[2]
HERE=PACKAGE/'output/independent'
COLS={'electrophile_smiles':'elec','nucleophile_smiles':'nuc'}
SIDES=('elec','nuc')
MODELS=('M0','ME','MN','MEN')
DEGREES=(2,3,5)
SAME={frozenset(('Suzuki','Suzuki Bpin')),frozenset(('Buchwald-Hartwig','Buchwald-Hartwig(CMC)'))}
CHECKS, HEADLINES, EFFECTS, PAIR_EFFECTS, SPLITS, PREDICTIONS = [], [], [], [], [], []
SOLVES=[]
def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return np.nan
    a = rankdata(np.round(a, 12)); b = rankdata(np.round(b, 12))
    a -= a.mean(); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan

def largest(g):
    """Frozen rule: most edges, at least 3; ties by smallest electrophile SMILES."""
    if len(g) < 3:
        return g.iloc[:0]
    ei, en = pd.factorize(g.elec); ni, nn = pd.factorize(g.nuc)
    graph = coo_matrix((np.ones(len(g)), (ei, len(en) + ni)),
                       shape=(len(en) + len(nn), len(en) + len(nn)))
    labels = connected_components(graph, directed=False)[1][ei]
    counts = np.bincount(labels)
    tied = np.flatnonzero(counts == counts.max())
    chosen = min(tied, key=lambda c: min(g.elec.to_numpy()[labels == c]))
    return g.iloc[np.flatnonzero(labels == chosen)]

def fit(w, label, record=False):
    """Fit pair+condition, then pair effects on molecular indicators.

    Each pair receives equal weight in stage two. Pair centring is unweighted over
    pairs; each molecular side is centred over molecules. Gamma includes the
    stage-one pair-centre shift, so held-out targets are mean(y - gamma_train).
    """
    pairs = w[["elec", "nuc"]].drop_duplicates().sort_values(["elec", "nuc"]).reset_index(drop=True)
    key = pd.MultiIndex.from_frame(pairs)
    wp = key.get_indexer(pd.MultiIndex.from_frame(w[["elec", "nuc"]]))
    wc, conditions = pd.factorize(w.condition_id)
    y = w["yield"].to_numpy(float)
    na, nb = len(pairs), len(conditions)
    ca, cb = np.bincount(wp, minlength=na), np.bincount(wc, minlength=nb)
    a, b = np.zeros(na), np.full(nb, y.mean())
    for iteration in range(20000):
        old = a.copy()
        a = np.bincount(wp, weights=y-b[wc], minlength=na) / ca
        b = np.bincount(wc, weights=y-a[wp], minlength=nb) / cb
        if np.max(np.abs(a-old)) < 1e-14:
            break
    else:
        raise ArithmeticError(f"Alternating projection did not converge: {label}")
    p, gamma = a-a.mean(), pd.Series(b+a.mean(), index=conditions)
    resid = y-p[wp]-gamma.to_numpy()[wc]
    normal_error = max(np.max(np.abs(np.bincount(wp, resid, na))),
                       np.max(np.abs(np.bincount(wc, resid, nb))))
    if normal_error > 1e-8:
        raise ArithmeticError(f"Normal-equation residual {normal_error}: {label}")
    ei, en = pd.factorize(pairs.elec); ni, nn = pd.factorize(pairs.nuc)
    x = np.zeros((len(pairs), len(en)+len(nn)-1))
    x[:, 0] = 1
    rows = np.arange(len(pairs))
    x[rows[ei > 0], ei[ei > 0]] = 1
    x[rows[ni > 0], len(en)-1+ni[ni > 0]] = 1
    beta, _, rank, _ = lstsq(x, p, lapack_driver="gelsy")
    if rank != x.shape[1]:
        raise ArithmeticError(f"Disconnected stage-two design: {label}")
    ae, an = np.r_[0., beta[1:len(en)]], np.r_[0., beta[len(en):]]
    mu = beta[0]+ae.mean()+an.mean()
    effects = {"elec": pd.Series(ae-ae.mean(), index=en),
               "nuc": pd.Series(an-an.mean(), index=nn)}
    pairs["effect"] = p
    SOLVES.append({"fit": label, "wells": len(w), "pairs": na,
                   "iterations": iteration+1, "stage1_normal_error": normal_error,
                   "stage2_rank": int(rank), "stage2_columns": x.shape[1]})
    if record:
        for side, values in effects.items():
            EFFECTS.extend({"fit": label, "side": side, "molecule": m,
                            "contribution": float(v)} for m, v in values.items())
        PAIR_EFFECTS.extend({"fit": label, "elec": r.elec, "nuc": r.nuc,
                             "pair_effect": float(r.effect)} for r in pairs.itertuples())
    return {"pairs": pairs, "gamma": gamma, "mu": float(mu), **effects}

def subset(w, pairs):
    return w.merge(pairs[["elec", "nuc"]], on=["elec", "nuc"], how="inner", validate="many_to_one")

def make_halves(pairs, reaction, side):
    other = "nuc" if side == "elec" else "elec"
    p = pairs[["elec", "nuc"]].copy()
    p["hash"] = [hashlib.sha256(f"{reaction}|{side}|{m}|{q}".encode()).hexdigest()
                 for m, q in zip(p[side], p[other])]
    p = p.sort_values([side, "hash"], kind="stable")
    p["half"] = p.groupby(side).cumcount() % 2
    halves = [largest(p[p.half == h]) for h in (0, 1)]
    for h, hp in enumerate(halves):
        SPLITS.extend({"reaction": reaction, "type": "partner", "side": side,
                       "partition": h, "elec": e, "nuc": n} for e, n in zip(hp.elec, hp.nuc))
    return halves

def prediction(w, reaction, pairs, degrees):
    fold = np.array([int(hashlib.sha256(f"{reaction}|{e}|{n}".encode()).hexdigest(), 16) % 10
                     for e, n in zip(pairs.elec, pairs.nuc)])
    scored = []
    for f in range(10):
        train = largest(pairs.iloc[np.flatnonzero(fold != f)])
        test = pairs.iloc[np.flatnonzero(fold == f)]
        test = test[test.elec.isin(train.elec) & test.nuc.isin(train.nuc)]
        fitted = fit(subset(w, train), f"analysed|{reaction}|fold{f}")
        for row in test.itertuples():
            tw = w[(w.elec == row.elec) & (w.nuc == row.nuc)]
            gamma = tw.condition_id.map(fitted["gamma"])
            ok = gamma.notna()
            if not ok.any():
                continue
            ps = fitted["pairs"]
            rec = {"reaction": reaction, "fold": f, "elec": row.elec, "nuc": row.nuc,
                   "target": float((tw.loc[ok, "yield"]-gamma[ok]).mean()),
                   "M0": float(ps.effect.mean()),
                   "ME": float(ps.loc[ps.elec == row.elec, "effect"].mean()),
                   "MN": float(ps.loc[ps.nuc == row.nuc, "effect"].mean()),
                   "MEN": fitted["mu"] + fitted["elec"][row.elec]+fitted["nuc"][row.nuc],
                   "degree_elec": int(degrees["elec"][row.elec]),
                   "degree_nuc": int(degrees["nuc"][row.nuc])}
            scored.append(rec)
        SPLITS.extend({"reaction": reaction, "type": "prediction-test", "side": "both",
                       "partition": f, "elec": e, "nuc": n} for e, n in zip(test.elec, test.nuc))
    PREDICTIONS.extend(scored)
    scored = pd.DataFrame(scored)
    rows = []
    for k in DEGREES:
        s = scored[(scored.degree_elec >= k) & (scored.degree_nuc >= k)]
        rec = {"reaction": reaction, "degree": k, "scored": len(s)}
        if len(s) >= 20:
            sse = {m: float(np.sum((s.target-s[m])**2)) for m in MODELS}
            rec.update({f"SSE_{m}": v for m, v in sse.items()})
            rec.update({f"RMSE_{m}": np.sqrt(v/len(s)) for m, v in sse.items()})
            rec.update({f"R2_{m}": 1-sse[m]/sse["M0"] for m in MODELS[1:]})
            rec.update(I_EN=(sse["MN"]-sse["MEN"])/sse["M0"],
                       I_NE=(sse["ME"]-sse["MEN"])/sse["M0"],
                       A_E=(sse["MN"]-sse["ME"])/sse["M0"],
                       A_N=(sse["ME"]-sse["MN"])/sse["M0"])
        rows.append(rec)
    return rows

def top_mean(source, target, k=10):
    source, target = np.round(np.asarray(source, float), 12), np.asarray(target, float)
    weight, remaining = np.zeros(len(source)), k
    for level in sorted(np.unique(source), reverse=True):
        pos = np.flatnonzero(source == level)
        take = min(remaining, len(pos))
        weight[pos] = take/len(pos); remaining -= take
        if remaining == 0:
            return float(weight @ target / k)
    raise ValueError("Insufficient selection candidates")

def cross_reaction(fits, reliabilities=None):
    rows = []
    for side in SIDES:
        for ra, rb in itertools.combinations(sorted(fits), 2):
            a, b = fits[ra][side], fits[rb][side]
            shared = sorted(set(a.index) & set(b.index))
            if len(shared) < 5:
                continue
            r = rho(a[shared], b[shared])
            rec = {"side": side, "rxn_a": ra, "rxn_b": rb, "shared": len(shared),
                   "same_reaction": frozenset((ra, rb)) in SAME, "rho_R": r}
            if reliabilities is not None:
                ceiling = np.sqrt(max(reliabilities[(ra, side)], 0)*max(reliabilities[(rb, side)], 0))
                rec["rho_corrected"] = r/ceiling
            if len(shared) >= 20:
                source_mean = top_mean(a[shared], b[shared]); target_mean = top_mean(b[shared], b[shared])
                rec.update(source_selected=source_mean, target_selected=target_mean,
                           retrospective_gap_pp=100*(target_mean-source_mean))
            rows.append(rec)
    return pd.DataFrame(rows)

def relative_stat(r):
    pos = r[(r.rho_P_a > 0) & (r.rho_P_b > 0)]
    dg = np.average(np.sqrt(pos.rho_P_a*pos.rho_P_b)-pos.rho_R_half, weights=pos.molecules)
    da = np.average((r.rho_P_a+r.rho_P_b)/2-r.rho_R_half, weights=r.molecules)
    return {"comparisons": len(r), "used for D_G": len(pos), "excluded": len(r)-len(pos),
            "D_G": float(dg), "D_A": float(da)}

def relative_comparisons(full_cross, halves, degrees):
    rows, pooled, shared_members = [], [], []
    for k in DEGREES:
        for c in full_cross[~full_cross.same_reaction].itertuples():
            ha, hb = halves[(c.rxn_a, c.side)], halves[(c.rxn_b, c.side)]
            shared = set(ha[0][c.side].index)&set(ha[1][c.side].index)&set(hb[0][c.side].index)&set(hb[1][c.side].index)
            shared = sorted(m for m in shared if degrees[(c.rxn_a, c.side)][m] >= k
                            and degrees[(c.rxn_b, c.side)][m] >= k)
            if len(shared) < 5:
                continue
            values = [[x[c.side][shared] for x in h] for h in (ha, hb)]
            rows.append({"degree": k, "side": c.side, "rxn_a": c.rxn_a, "rxn_b": c.rxn_b,
                         "molecules": len(shared), "rho_P_a": rho(*values[0]),
                         "rho_P_b": rho(*values[1]),
                         "rho_R_half": float(np.mean([rho(x, z) for x in values[0] for z in values[1]]))})
            shared_members.extend({"degree": k, "side": c.side, "rxn_a": c.rxn_a,
                                   "rxn_b": c.rxn_b, "molecule": m} for m in shared)
        r = pd.DataFrame([v for v in rows if v["degree"] == k])
        est = relative_stat(r)
        reactions = sorted(set(r.rxn_a)|set(r.rxn_b))
        deleted = [relative_stat(r[(r.rxn_a != rxn)&(r.rxn_b != rxn)]) for rxn in reactions]
        for metric in ("D_G", "D_A"):
            vals = np.array([v[metric] for v in deleted]); n = len(vals)
            se = np.sqrt((n-1)/n*np.sum((vals-vals.mean())**2))
            q = t_dist.ppf(.975, n-1)
            est.update({metric+" se": float(se), metric+" low": est[metric]-q*se,
                        metric+" high": est[metric]+q*se})
        pooled.append({"degree": k, **est})
    return pd.DataFrame(rows), pd.DataFrame(pooled), pd.DataFrame(shared_members)

def ring_n_halide(smiles):
    """1 if a halogen or sulfonate oxygen sits on an aromatic carbon bonded to a two-connected,
    uncharged, H-free aromatic nitrogen; 0 if the molecule has an aryl leaving group but no such
    site; None if it has no aryl leaving group. Written from the Methods text, atom by atom."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    found, leaving = 0, False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in ("F", "Cl", "Br", "I"):
            ok = True
        elif atom.GetSymbol() == "O" and atom.GetDegree() == 2:
            ok = any(n.GetSymbol() == "S" and sum(1 for b in n.GetBonds() if b.GetBondTypeAsDouble() == 2) >= 2
                     for n in atom.GetNeighbors())
        else:
            continue
        for c in atom.GetNeighbors():
            if not (ok and c.GetSymbol() == "C" and c.GetIsAromatic()):
                continue
            leaving = True
            if any(n.GetSymbol() == "N" and n.GetIsAromatic() and n.GetDegree() == 2 and n.GetTotalNumHs() == 0
                   and n.GetFormalCharge() == 0 for n in c.GetNeighbors()):
                found = 1
    return found if leaving else None
def main():
    HERE.mkdir(parents=True,exist_ok=True)
    # Original default pandas numeric parser is preserved; identity/membership only is adapted.
    wells=pd.read_csv(PACKAGE/'data/experimental_wells.csv',dtype={'condition_id':str}).rename(columns=COLS)
    wells=wells[(wells.component=='existing_release')&wells.measurement_status.isin(['quantified','zero_by_rule'])].copy()
    membership=pd.read_csv(PACKAGE/'data/metadata/analysis_well_membership.csv',dtype=str)
    wells['twenty_conditions']=''
    for label,scope in [('primary','analysed'),('second_condition_set','held out')]:
        wells.loc[wells.well_id.isin(set(membership.loc[membership.analysis_set==label,'well_id'])),'twenty_conditions']=scope
    conditions=pd.read_csv(PACKAGE/'data/metadata/base_condition_roles.csv',dtype={'condition_id':str})
    fits, reliabilities, counts = {}, {}, []
    partner_halves, degrees, partner_rows, prediction_rows = {}, {}, [], []
    condition_shared, condition_cells = [], []
    for scope in ("analysed", "held out"):
        selected = wells[wells.twenty_conditions == scope]
        scope_fits, scope_rel = {}, {}
        for reaction, w in selected.groupby("rxn_class", sort=True):
            print(f"Fitting {scope}: {reaction}, {len(w)} wells", flush=True)
            full = fit(w, f"{scope}|{reaction}|full", record=True)
            p = full["pairs"]
            cond = conditions[(conditions.design == "twenty conditions") & (conditions["set"] == scope)
                              & (conditions.rxn_class == reaction)]
            ch = []
            for parity in (0, 1):
                wc = w[w.condition_id.isin(cond.loc[cond.coverage_rank % 2 == parity, "condition_id"])]
                wc = subset(wc, largest(wc[["elec", "nuc"]].drop_duplicates()))
                ch.append(fit(wc, f"{scope}|{reaction}|condition{parity}", record=True))
            for side in SIDES:
                shared = sorted(set(ch[0][side].index)&set(ch[1][side].index))
                r = rho(ch[0][side][shared], ch[1][side][shared])
                scope_rel[(reaction, side)] = 2*r/(1+r)
                condition_cells.append({"set": scope, "rxn_class": reaction, "side": side,
                                        "n_molecules": len(shared), "r_half": r, "reliability": 2*r/(1+r)})
            scope_fits[reaction] = full
            counts.append({"set": scope, "rxn_class": reaction, "wells": len(w), "pairs": len(p),
                           "electrophiles": p.elec.nunique(), "nucleophiles": p.nuc.nunique()})
            if scope != "analysed":
                continue
            ds = {s: p.groupby(s).size() for s in SIDES}
            for side in SIDES:
                degrees[(reaction, side)] = ds[side]
                hp = make_halves(p, reaction, side)
                hs = [fit(subset(w, part), f"analysed|{reaction}|partner-{side}-{h}", record=True)
                      for h, part in enumerate(hp)]
                partner_halves[(reaction, side)] = hs
                both = sorted(set(hs[0][side].index)&set(hs[1][side].index))
                for k in DEGREES:
                    ev = [m for m in both if ds[side][m] >= k]
                    cc = [m for m in ev if m in ch[0][side].index and m in ch[1][side].index]
                    partner_rows.append({"reaction": reaction, "side": side, "degree": k, "molecules": len(ev),
                                         "rho_P": rho(hs[0][side][ev], hs[1][side][ev]) if len(ev) >= 10 else np.nan,
                                         "rho_C": rho(ch[0][side][cc], ch[1][side][cc]) if len(cc) >= 10 else np.nan})
            prediction_rows.extend(prediction(w, reaction, p, ds))
        fits[scope], reliabilities[scope] = scope_fits, scope_rel
    counts = pd.DataFrame(counts); partner_rows = pd.DataFrame(partner_rows)
    prediction_rows = pd.DataFrame(prediction_rows); condition_cells = pd.DataFrame(condition_cells)
    crosses = {scope: cross_reaction(fits[scope], reliabilities[scope]) for scope in fits}
    relative, relative_pool, members = relative_comparisons(crosses["analysed"], partner_halves, degrees)
    for reaction in sorted(fits["analysed"]):
        for side in SIDES:
            a, b = fits["analysed"][reaction][side], fits["held out"][reaction][side]
            shared = sorted(set(a.index)&set(b.index))
            condition_shared.append({"rxn_class": reaction, "side": side, "shared": len(shared),
                                     "rho": rho(a[shared], b[shared])})
    condition_shared = pd.DataFrame(condition_shared)
    tables = {"independent_counts.csv": counts, "independent_contributions.csv": pd.DataFrame(EFFECTS),
              "independent_pair_effects.csv": pd.DataFrame(PAIR_EFFECTS),
              "independent_partner_correlations.csv": partner_rows,
              "independent_prediction_metrics.csv": prediction_rows,
              "independent_predictions.csv": pd.DataFrame(PREDICTIONS),
              "independent_partition_membership.csv": pd.DataFrame(SPLITS),
              "independent_condition_reliability.csv": condition_cells,
              "independent_condition_validation.csv": condition_shared,
              "independent_cross_reaction.csv": crosses["analysed"],
              "independent_holdout_cross_reaction.csv": crosses["held out"],
              "independent_gateB_relative.csv": relative, "independent_gateB_pooled.csv": relative_pool,
              "independent_gateB_shared_molecules.csv": members, "independent_solver_diagnostics.csv": pd.DataFrame(SOLVES)}
    for name, table in tables.items():
        table.to_csv(HERE/name, index=False)

    # Independent structural contrast computation uses the same original feature function.
    structural=[]
    for scope,data in [('analysed','first condition set'),('held out','second condition set')]:
        for reaction in sorted(fits[scope]):
            e=fits[scope][reaction]['elec']
            flag=pd.Series({m:ring_n_halide(m) for m in e.index}).dropna()
            if (flag==1).sum()<3 or (flag==0).sum()<3:continue
            c=100*e[flag.index]
            structural.append({'data':data,'reaction':reaction,'feature':'halogen next to ring N',
                'difference_pp':c[flag==1].mean()-c[flag==0].mean()})
    pd.DataFrame(structural).to_csv(HERE/'independent_structure_contrasts.csv',index=False)

if __name__=='__main__':main()
