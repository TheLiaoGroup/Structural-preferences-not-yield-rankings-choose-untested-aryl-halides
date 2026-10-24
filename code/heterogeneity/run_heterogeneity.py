"""Heterogeneity of the cross-reaction agreement: Cochran's Q of the observed comparisons against its
distribution in 10,000 corpora simulated on the real pair and condition layouts with one latent
correlation shared by all comparisons.

Each reaction's noise is first calibrated to its condition-half agreement, and the shared latent
correlation to the observed pooled agreement. Reads the wells and the core step's comparisons and
reliabilities; writes output/heterogeneity.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
HERE = Path(__file__).resolve().parent
STAGING = HERE.parents[1]
CORE = STAGING / "code/core"
SIM = STAGING / "code/calibration"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(SIM))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import gated_layouts as G
import layout_inputs as M
import validate_gated_bba as V
import heterogeneity_science as S

DRAW_COUNT = 10000


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, value):
    value.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def main():
    out = STAGING / "output/heterogeneity"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    core = STAGING / "output/core"
    w, c, p = M.primary_frames(STAGING)
    layouts, rows = {}, {}
    for rc, pairs in p.groupby("rxn_class", sort=True):
        rows[rc] = w[w.rxn_class.eq(rc)].reset_index(drop=True)
        conditions = c[c.rxn_class.eq(rc)].condition_id.tolist()
        layouts[rc] = G.ReactionLayout(rc, rows[rc], pairs, conditions, rows[rc].plate)
        assert list(layouts[rc].e_name) == sorted(layouts[rc].e_name)
        assert list(layouts[rc].n_name) == sorted(layouts[rc].n_name)
    assert len(layouts) == 15 and list(layouts) == sorted(layouts)
    comparisons = M.comparison_frame({"A": layouts}, G.E)
    write_csv(out / "gated_gateB_eligibility_bba.csv", comparisons)
    G.L.OUT = out
    observed = pd.read_csv(core / "gated_B_observed_k2_bba.csv").sort_values(["side", "rxn_a", "rxn_b"], kind="stable").reset_index(drop=True)
    assert not observed.duplicated(["side", "rxn_a", "rxn_b"]).any()
    assert list(zip(observed.side, observed.rxn_a, observed.rxn_b)) == G.comparison_list()
    write_csv(out / "observed_comparisons.csv", observed)
    rel = pd.read_csv(core / "extended_reliability_bba.csv").groupby("rxn_class").r_half.mean()
    # The yields are read with the parser of public_common.released.
    released = pd.read_csv(STAGING / "data/experimental_wells.csv", usecols=["well_id", "yield", "component", "measurement_status"],
                           keep_default_na=False, low_memory=False, float_precision="round_trip")
    released = released[released.component.eq("existing_release") & released.measurement_status.isin(["quantified", "zero_by_rule"])]
    assert released.well_id.is_unique
    yield_by_id = released.set_index("well_id")["yield"]
    y_of = {rc: yield_by_id.loc[rows[rc].well_id].to_numpy(float) for rc in layouts}
    assert all(np.isfinite(y).all() and len(y) == len(layouts[rc].w_pair) for rc, y in y_of.items())
    rng = np.random.default_rng(S.HETEROGENEITY_SEED)
    prm_of = {}
    for rc, lay in layouts.items():
        floor = float(np.mean(np.asarray(y_of[rc]) <= 0.01))
        v = S.calibrate_reaction(lay, float(rel[rc]), floor, rng, 50, 12)
        prm_of[rc] = {**V.BASE, "v": v, "f": floor}
        print(f"noise calibrated: {rc}, {time.time() - started:.0f} s", flush=True)
    write_json(out / "noise_calibration.json", prm_of)
    names = {s: sorted(set().union(*[set(l.e_name if s == "elec" else l.n_name) for l in layouts.values()]))
             for s in ("elec", "nuc")}
    index = {rc: {s: np.searchsorted(names[s], l.e_name if s == "elec" else l.n_name)
                  for s in ("elec", "nuc")} for rc, l in layouts.items()}
    target = G.pooled_observed(observed, inference=False)["weighted"]
    lo, hi = 0.0, 1.0
    calibration = []
    for step in range(12):
        rho = (lo + hi) / 2
        got = np.mean([G.pooled_observed(S.simulated_transfer(layouts, prm_of, rho, rng, names, index),
                                         inference=False)["weighted"] for _ in range(200)])
        calibration.append({"step": step, "rho": rho, "pooled": float(got)})
        print(f"latent calibration step {step}, {time.time() - started:.0f} s", flush=True)
        if abs(got - target) < 0.005:
            break
        lo, hi = (rho, hi) if got < target else (lo, rho)
    write_json(out / "latent_calibration.json", calibration)
    q_obs = S.cochran_q(observed)
    q_sim = []
    for draw in range(DRAW_COUNT):
        q_sim.append(S.cochran_q(S.simulated_transfer(layouts, prm_of, rho, rng, names, index)))
        if (draw + 1) % 1000 == 0:
            print(f"{draw + 1} of {DRAW_COUNT} simulated corpora, {time.time() - started:.0f} s", flush=True)
    write_csv(out / "simulated_Q.csv", pd.DataFrame({"draw": np.arange(len(q_sim)), "Q": q_sim}))
    q_sim = np.asarray(q_sim)
    assert len(q_sim) == DRAW_COUNT and np.isfinite(q_sim).all()
    exceedances = int(np.count_nonzero(q_sim >= q_obs))
    p_value = exceedances / DRAW_COUNT
    result = pd.DataFrame([{"Q": q_obs, "comparisons": int((np.isfinite(observed.rho_R) & (observed.shared > 3)).sum()),
                            "rho*": rho, "calibrated pooled rho_R": got, "observed pooled rho_R": target,
                            "simulations": DRAW_COUNT, "exceedances": exceedances, "p": p_value,
                            "p_display_3dp": f"{p_value:.3f}"}]).assign(
                                **{f"v {rc}": prm_of[rc]["v"] for rc in prm_of},
                                **{f"floor {rc}": prm_of[rc]["f"] for rc in prm_of})
    write_csv(out / "gated_B_heterogeneity_bba.csv", result)
    print(result.iloc[0, :9].to_json(), flush=True)


if __name__ == "__main__":
    main()
