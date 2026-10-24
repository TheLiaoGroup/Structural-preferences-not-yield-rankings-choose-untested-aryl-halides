"""Intervals of the core estimates: pooled agreement and retention on the first condition set, the second
condition set, and the reaction-node bootstrap of the pooled agreement at degrees 2, 3 and 5."""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import importlib
import numpy as np
from public_common import *

def summary_rows(res, suffix):
    labels = {"C1": f"C1 within-reaction, analysed vs {suffix}",
              "C2": f"C2 cross-reaction, {suffix}, corrected",
              "E1": "E1 retention of a top-ten selection"}
    return [{"quantity": labels[k], "estimate": res[k][0], "ci_low": res[k][1], "ci_high": res[k][2]}
            for k in ("C1", "C2", "E1")]

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    L, E, G = load_modules()
    X = importlib.import_module("extended_transfer_bba")
    H = importlib.import_module("holdout_bba")
    wells, conditions, pairs = released()
    rows = wells.rename(columns=RENAMES)
    sets = {key: design_inputs(wells, conditions, pairs, key) for key in DESIGNS}
    for label, (_, c, p) in sets.items():
        prefix = DESIGNS[label][4]
        write_csv(c.rename(columns={"coverage_rank": "rank"}), RESULTS / f"{prefix}_conditions_bba.csv")
        write_csv(p, RESULTS / f"{prefix}_pairs_bba.csv")

    print("Intervals on the first condition set", flush=True)
    _, c, p = sets["primary"]
    X.reproduce(rows, p, c)

    print("Intervals on the second condition set", flush=True)
    held = H.score(rows, H.designs(), np.random.default_rng(H.SEED))
    write_csv(pd.DataFrame(summary_rows(held, "held-out")), RESULTS / "holdout_summary_bba.csv")

    print("Reaction-node bootstrap, degrees 2, 3 and 5", flush=True)
    core_dir = STAGING / "output" / "core"
    rng = np.random.default_rng(20261101)
    node_rows = []
    for degree in (2, 3, 5):
        observed = pd.read_csv(core_dir / f"gated_B_observed_k{degree}_bba.csv")
        key = ["side", "rxn_a", "rxn_b"]
        if observed.duplicated(key).any():
            raise RuntimeError("Duplicate comparison identity")
        observed = observed.sort_values(key, kind="stable").reset_index(drop=True)
        result = G.pooled_observed(observed, bootstrap=2000, rng=rng)
        node_rows.append({"degree": degree, **result})
    write_csv(pd.DataFrame(node_rows), RESULTS / "node_bootstrap_pooled.csv")

if __name__ == "__main__":
    main()
