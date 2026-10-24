"""Partner halves, prediction folds and the eligibility of molecules and cross-reaction comparisons."""
import hashlib

import itertools

import math

import re

import networkx as nx

import numpy as np

import pandas as pd

import design_extended_bba as E

import lib_bba as L

DEGREES = (2, 3, 5)

COMMON = (2, 3)

M_PER_FAMILY = 6

MIN_DELTA_PAIRS, MIN_DELTA_MOLECULES, MIN_MATCHED = 30, 10, 2

K_FOLDS = 10

LEVERAGE_TOL = 1e-8

TIES = []

def largest_piece(pairs, where):
    """Largest connected piece by number of pairs. A piece needs three pairs. A tie between pieces
    of equal size is broken by the piece holding the alphabetically first electrophile SMILES, so
    the choice does not depend on row order; every choice is logged with whether it was a tie."""
    if len(pairs) < 3:
        TIES.append((where, False))
        return pairs.iloc[0:0]
    labels = L.component_labels(pairs.elec, pairs.nuc)
    counts = np.bincount(labels)
    best = np.flatnonzero(counts == counts.max())
    if len(best) > 1:
        chosen = min(best, key=lambda c: min(pairs.elec.values[labels == c]))
    else:
        chosen = best[0]
    TIES.append((where, len(best) > 1))
    return pairs[labels == chosen]

def fold_of(rxn, elec, nuc):
    """Ten-fold assignment of a pair for end-to-end prediction, by SHA-256; no yield."""
    return int(hashlib.sha256(f"{rxn}|{elec}|{nuc}".encode()).hexdigest(), 16) % K_FOLDS

READ = {"analysed": "extended_conditions_bba.csv", "held out": "holdout_conditions_bba.csv"}

METALS = ("Pd", "Ni", "Fe", "Cu")

CONTRASTS = {
    "Sonogashira": ("copper co-catalyst", {"Cu": {"Pd", "Cu"}, "no Cu": {"Pd"}}),
    "Suzuki": ("metal", {"Pd": {"Pd"}, "Ni": {"Ni"}}),
    "Kumada": ("metal", {"Pd": {"Pd"}, "Ni": {"Ni"}, "Fe": {"Fe"}}),
    "Buchwald-Hartwig": ("metal", {"Pd": {"Pd"}, "Cu": {"Cu"}}),
    "C-O competitive reactions": ("metal", {"Pd": {"Pd"}, "Cu": {"Cu"}}),
}

def metals_of(names):
    text = " ".join(names)
    found = {m for m in METALS if re.search(rf"(?<![A-Za-z]){m}(?![a-z])", text)}
    if "PEPPSI" in text:
        found.add("Pd")
    return found

def physical_plate_ids(values):
    """Retain the complete batch prefix through -PLdigits; assay suffixes do not name a new plate."""
    recorded = pd.Series(values, copy=False).fillna("").astype(str).str.strip()
    physical = recorded.str.extract(r"^(.*-PL[0-9]+)(?:-.*)?$", expand=False)
    invalid = recorded.ne("") & physical.isna()
    if invalid.any():
        raise ValueError(f"plate labels do not contain a terminal -PLnumber: {recorded[invalid].unique()[:5].tolist()}")
    return physical.fillna("")


def with_ids(rows):
    """Plate and screen as strings, empty where a source file does not record them."""
    for c in ("plate", "screen"):
        rows[c] = rows[c].fillna("").astype(str) if c in rows.columns else ""
    rows["plate"] = physical_plate_ids(rows.plate)
    return rows

def partner_halves(pairs, rxn, side):
    """The frozen partner split of the partner-half analysis. Each molecule on `side` orders its partners by
    SHA-256 of (reaction, side, molecule, partner) and sends them alternately to halves A and B, so
    every edge sits in exactly one half and no pair (hence no plate) is shared. A molecule with one
    partner lands in one half only. Each half is then cut to its largest connected piece. Returns
    the two edge sets. Uses no yield."""
    other = "nuc" if side == "elec" else "elec"
    order = pairs.assign(key=[hashlib.sha256(f"{rxn}|{side}|{m}|{q}".encode()).hexdigest()
                              for m, q in zip(pairs[side], pairs[other])])
    order = order.sort_values([side, "key"], kind="stable")
    order["half"] = order.groupby(side).cumcount() % 2
    halves = []
    for h in (0, 1):
        e = order[order.half == h][["elec", "nuc"]]
        halves.append(largest_piece(e, f"partner half {h + 1}: {rxn}, {side}"))
    return halves

def evaluable(P, deg):
    """Molecules with a contribution in both partner halves, by reaction and side and degree."""
    out = {}
    for rc, g in P.groupby("rxn_class"):
        for side in ("elec", "nuc"):
            a, b = partner_halves(g[["elec", "nuc"]], rc, side)
            both = set(a[side]) & set(b[side])
            out[(rc, side)] = both
    return out

def degrees(P):
    return {(rc, side): g.groupby(side)[other].nunique()
            for rc, g in P.groupby("rxn_class") for side, other in (("elec", "nuc"), ("nuc", "elec"))}

def gate_a(P, deg, ev):
    out = []
    for rc, g in P.groupby("rxn_class"):
        de, dn = deg[(rc, "elec")], deg[(rc, "nuc")]
        both = pd.DataFrame({"e": g.elec.map(de).values, "n": g.nuc.map(dn).values})
        for side, d in (("elec", de), ("nuc", dn)):
            rec = {"reaction": rc, "side": side, "molecules": len(d), "degree 1": int((d == 1).sum())}
            for k in DEGREES:
                rec[f"molecules degree >={k}"] = int((d >= k).sum())
            rec["molecules degree >=4 (two partners per half)"] = int((d >= 4).sum())
            for k in DEGREES:
                rec[f"evaluable in both partner halves, degree >={k}"] = int(sum(d[m] >= k for m in ev[(rc, side)]))
            for k in DEGREES:
                rec[f"pairs with both degrees >={k}"] = int(((both.e >= k) & (both.n >= k)).sum())
            rec["pairs"] = len(g)
            out.append(rec)
    return pd.DataFrame(out)

def gate_b(P, deg, ev):
    reactions = sorted(P.rxn_class.unique())
    partners = {(rc, side): g.groupby(side)["nuc" if side == "elec" else "elec"].apply(set)
                for rc, g in P.groupby("rxn_class") for side in ("elec", "nuc")}
    out = []
    for side in ("elec", "nuc"):
        for a, b in itertools.combinations(reactions, 2):
            shared = sorted(set(deg[(a, side)].index) & set(deg[(b, side)].index))
            if len(shared) < E.MIN_SHARED:
                continue
            dmin = np.array([min(deg[(a, side)][m], deg[(b, side)][m]) for m in shared])
            common = np.array([len(partners[(a, side)][m] & partners[(b, side)][m]) for m in shared])
            rec = {"side": side, "rxn_a": a, "rxn_b": b,
                   "same_reaction": frozenset((a, b)) in E.SAME_REACTION, "shared": len(shared),
                   "degree 1 in either": int((dmin == 1).sum())}
            for k in DEGREES:
                rec[f"shared degree >={k} in both"] = int((dmin >= k).sum())
            both_halves = [m for m in shared if m in ev[(a, side)] and m in ev[(b, side)]]
            for k in DEGREES:
                rec[f"shared evaluable in both halves of both, degree >={k}"] = int(
                    sum(min(deg[(a, side)][m], deg[(b, side)][m]) >= k for m in both_halves))
            for m in COMMON:
                rec[f"shared with >={m} common partners"] = int((common >= m).sum())
            rec["common partner molecules"] = len(set().union(*[partners[(a, side)][m] & partners[(b, side)][m]
                                                                 for m in shared]))
            out.append(rec)
    return pd.DataFrame(out)

def matched_design(g, pairs):
    """Pairs measured under both conditions of at least one matched condition pair; largest piece."""
    if not pairs:
        return {"pairs": 0, "electrophiles": 0, "nucleophiles": 0, "electrophiles >=2": 0, "nucleophiles >=2": 0}
    runs = g.drop_duplicates(["condition_id", "elec", "nuc"])
    got = [runs[runs.condition_id == a][["elec", "nuc"]].merge(runs[runs.condition_id == b][["elec", "nuc"]],
                                                                on=["elec", "nuc"]) for a, b in pairs]
    allp = pd.concat(got).drop_duplicates()
    piece = largest_piece(allp, "matched design")
    de, dn = piece.groupby("elec").nuc.nunique(), piece.groupby("nuc").elec.nunique()
    return {"pairs": len(piece), "electrophiles": int(de.size), "nucleophiles": int(dn.size),
            "electrophiles >=2": int((de >= 2).sum()), "nucleophiles >=2": int((dn >= 2).sum())}

def enough(md):
    return (md["pairs"] >= MIN_DELTA_PAIRS and md["electrophiles >=2"] >= MIN_DELTA_MOLECULES
            and md["nucleophiles >=2"] >= MIN_DELTA_MOLECULES)

def leverage(pairs):
    """Diagonal of the stage-two hat matrix (intercept plus one indicator per molecule less one per
    side), as analysis_bba.stage_two builds it; structure only."""
    ei = pd.factorize(pairs.elec.values)[0]
    ni = pd.factorize(pairs.nuc.values)[0]
    ne, nn = ei.max() + 1, ni.max() + 1
    X = np.zeros((len(pairs), ne + nn - 1))
    X[:, 0] = 1.0
    r = np.arange(len(pairs))
    X[r[ei > 0], ei[ei > 0]] = 1.0
    X[r[ni > 0], ne - 1 + ni[ni > 0]] = 1.0
    Q, _ = np.linalg.qr(X)
    return np.einsum("ij,ij->i", Q, Q)

def bridge_mask(pairs):
    G = nx.Graph()
    G.add_edges_from(("e|" + e, "n|" + n) for e, n in zip(pairs.elec, pairs.nuc))
    br = {frozenset(b) for b in nx.bridges(G)}
    return np.array([frozenset(("e|" + e, "n|" + n)) in br for e, n in zip(pairs.elec, pairs.nuc)])

def kfold_scorable(pairs, rxn, where):
    """Pairs a ten-fold end-to-end prediction can score: with the pair's fold removed, both of its
    molecules lie in the largest piece of the remaining pairs. The training graph is the same for
    every degree threshold; the threshold only selects which scorable pairs are reported."""
    fold = np.array([fold_of(rxn, e, n) for e, n in zip(pairs.elec, pairs.nuc)])
    ok = np.zeros(len(pairs), bool)
    for f in range(K_FOLDS):
        train = largest_piece(pairs[fold != f], f"{where}, fold {f}")
        es, ns = set(train.elec), set(train.nuc)
        test = np.flatnonzero(fold == f)
        ok[test] = [pairs.elec.values[i] in es and pairs.nuc.values[i] in ns for i in test]
    return ok

def gate_a_prediction(P, deg):
    out = []
    for rc, g in P.groupby("rxn_class"):
        g = g[["elec", "nuc"]].reset_index(drop=True)
        h = leverage(g)
        br = bridge_mask(g)
        assert ((h >= 1 - LEVERAGE_TOL) == br).all(), f"{rc}: leverage-one edges are not the bridges"
        ok = kfold_scorable(g, rc, f"ten-fold training: {rc}")
        de, dn = g.elec.map(deg[(rc, "elec")]).values, g.nuc.map(deg[(rc, "nuc")]).values
        rec = {"reaction": rc, "pairs": len(g), "bridge edges": int(br.sum()),
               "edges with leverage >= 1 - 1e-8": int((h >= 1 - LEVERAGE_TOL).sum())}
        for k in (1,) + DEGREES:
            both = (de >= k) & (dn >= k)
            rec[f"historical LOPO scorable, degree >={k}"] = int((both & ~br).sum())
            rec[f"ten-fold scorable, degree >={k}"] = int((both & ok).sum())
        out.append(rec)
    return pd.DataFrame(out)

def activation_units(U, contrast_pairs, P):
    """Per contrast: condition pairs, unique conditions, condition-disjoint contrasts, the
    differential-response graph and its ten-fold scorable pairs, the pairs with a historical pair
    effect, and the four family halves of the ranking comparison. Structure only."""
    out = []
    for (rc, fa, fb), s in contrast_pairs.groupby(["reaction", "family A", "family B"], sort=False):
        g = U[U.rxn_class == rc].drop_duplicates(["condition_id", "elec", "nuc"])
        kind = s.design.iloc[0]
        rows = list(zip(s["condition A"], s["condition B"], s["replicate or half"]))
        uA, uB = sorted(set(s["condition A"])), sorted(set(s["condition B"]))
        runs = {c: set(zip(x.elec, x.nuc)) for c, x in g[g.condition_id.isin(uA + uB)].groupby("condition_id")}
        if kind == "regime":
            disjoint = 1
            delta = set()
            for hf in sorted(set(s["replicate or half"])):
                a_c = sorted(set(s[s["replicate or half"] == hf]["condition A"]))
                b_c = sorted(set(s[s["replicate or half"] == hf]["condition B"]))
                need = math.ceil(len(a_c) / 2)
                every = set().union(*[runs.get(c, set()) for c in a_c + b_c])
                delta |= {q for q in every if sum(q in runs.get(c, set()) for c in a_c) >= need
                          and sum(q in runs.get(c, set()) for c in b_c) >= need}
            halves = {f"{fa} half {hf}": sorted(set(s[s["replicate or half"] == hf]["condition A"]))
                      for hf in sorted(set(s["replicate or half"]))}
            halves.update({f"{fb} half {hf}": sorted(set(s[s["replicate or half"] == hf]["condition B"]))
                           for hf in sorted(set(s["replicate or half"]))})
        else:
            disjoint = sum(1 for (a1, b1, _), (a2, b2, _) in itertools.combinations(rows, 2)
                           if not {a1, b1} & {a2, b2})
            delta = set().union(*[runs.get(a, set()) & runs.get(b, set()) for a, b, _ in rows])
            # each family's unique conditions, ordered by SHA-256 and dealt alternately into two
            # halves, so the halves differ in size by at most one condition
            order = lambda fam, cs: sorted(cs, key=lambda c: hashlib.sha256(f"{rc}|{fam}|{c}".encode()).hexdigest())
            halves = {f"{fam} half {h + 1}": order(fam, cs)[h::2] for fam, cs in ((fa, uA), (fb, uB)) for h in (0, 1)}
        D = pd.DataFrame(sorted(delta), columns=["elec", "nuc"])
        piece = largest_piece(D, f"differential-response graph: {rc} {fa}/{fb}")
        de, dn = piece.groupby("elec").nuc.nunique(), piece.groupby("nuc").elec.nunique()
        ok = kfold_scorable(piece.reset_index(drop=True), rc, f"ten-fold training, differential response: {rc} {fa}/{fb}")
        both2 = (piece.elec.map(de).values >= 2) & (piece.nuc.map(dn).values >= 2)
        hist = set(zip(P[P.rxn_class == rc].elec, P[P.rxn_class == rc].nuc))
        rec = {"reaction": rc, "family A": fa, "family B": fb, "design": kind,
               "condition pairs": len(rows), "unique conditions A": len(uA), "unique conditions B": len(uB),
               "condition-disjoint contrasts": disjoint,
               "differential-response pairs (piece)": len(piece), "electrophiles": int(de.size),
               "nucleophiles": int(dn.size), "electrophiles >=2": int((de >= 2).sum()),
               "nucleophiles >=2": int((dn >= 2).sum()),
               "ten-fold scorable pairs": int(ok.sum()), "ten-fold scorable, both degrees >=2": int((ok & both2).sum()),
               "pairs with a historical pair effect": int(sum((e, n) in hist for e, n in zip(piece.elec, piece.nuc)))}
        # the scaling-adjusted analysis uses only pairs with a historical pair effect, on its own piece
        adj = piece[[(e, n) in hist for e, n in zip(piece.elec, piece.nuc)]]
        adj = largest_piece(adj, f"scaling-adjusted graph: {rc} {fa}/{fb}").reset_index(drop=True)
        rec["scaling-adjusted: pairs (piece)"] = len(adj)
        rec["scaling-adjusted: ten-fold scorable pairs"] = int(kfold_scorable(
            adj, rc, f"ten-fold training, scaling-adjusted: {rc} {fa}/{fb}").sum()) if len(adj) else 0
        common = {"elec": None, "nuc": None}
        for name, cs in halves.items():
            q = pd.DataFrame(sorted(set(zip(piece.elec, piece.nuc)) & set().union(*[runs.get(c, set()) for c in cs])),
                             columns=["elec", "nuc"]) if cs else pd.DataFrame(columns=["elec", "nuc"])
            qp = largest_piece(q, f"family half {name}: {rc}") if len(q) else q
            qe, qn = qp.groupby("elec").nuc.nunique(), qp.groupby("nuc").elec.nunique()
            rec[f"{name}: conditions"] = len(cs)
            rec[f"{name}: pairs"] = len(qp)
            for side, d in (("elec", qe), ("nuc", qn)):
                keep = set(d[d >= 2].index)
                common[side] = keep if common[side] is None else common[side] & keep
        rec["ranking comparison: electrophiles in all four family halves (degree >=2)"] = len(common["elec"] or ())
        rec["ranking comparison: nucleophiles in all four family halves (degree >=2)"] = len(common["nuc"] or ())
        out.append(rec)
    return pd.DataFrame(out)
