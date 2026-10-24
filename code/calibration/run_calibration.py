"""Calibration of the intervals on simulated data (Supplementary Note 6): 112 settings x 500 replicates on
the real pair and condition layouts of the fifteen analysed reactions.

Every setting and replicate draws from its own SeedSequence (root 20261103), so the results do not depend
on the number of worker processes or on their scheduling. Writes output/calibration.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import pickle
import sys
import time

sys.dont_write_bytecode = True
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
HERE = Path(__file__).resolve().parent
STAGING = HERE.parents[1]
CORE = STAGING / "code" / "core"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import gated_layouts as G
import layout_inputs as M
import validate_gated_bba as V

ROOT_SEED = 20261103
REPLICATES = 500
CHUNK = 25   # replicates per job


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def init_worker(layout_path, output):
    G.L.OUT = Path(output)
    V.L.OUT = Path(output)
    V.LAYOUTS.clear()
    with Path(layout_path).open("rb") as f:
        V.LAYOUTS.update(pickle.load(f))


def perform(job):
    cell, lo, hi = job
    rows = V.task((cell, ROOT_SEED, lo, hi))
    if lo >= 0:
        assert {int(r["rep"]) for r in rows} == set(range(lo, hi)), "A requested replicate produced no record"
    else:
        assert len(rows) == 1 and "oracle" in rows[0]
    # Records are passed on as plain JSON values (numpy scalars become Python numbers).
    return cell["cell"], lo, json.loads(json.dumps(rows, ensure_ascii=False, default=json_default, separators=(",", ":")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = STAGING / "output" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    layouts = M.build_layouts(G)
    layout_file = out / "layouts.pkl"
    with layout_file.open("wb") as f:
        pickle.dump(layouts, f, protocol=pickle.HIGHEST_PROTOCOL)
    # The cross-reaction comparisons of the fifteen layouts, read by the gate B simulations.
    M.comparison_frame(layouts, G.E).to_csv(out / "gated_gateB_eligibility_bba.csv", index=False, lineterminator="\n", float_format="%.17g")
    cells = M.load_cells()
    jobs = [(c, -1, 0) for c in cells]
    jobs += [(c, lo, lo + CHUNK) for c in cells for lo in range(0, REPLICATES, CHUNK)]
    # Oracles first, then the slower gate B settings.
    jobs.sort(key=lambda j: (j[1] >= 0, {"B": 0, "C": 1, "A": 2}[j[0]["gate"]], j[1], j[0]["cell"]))
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(layout_file), str(out))) as pool:
        futures = [pool.submit(perform, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            cell, lo, rows = future.result()
            results[(cell, lo)] = rows
            if done % 100 == 0 or done == len(jobs):
                print(f"{done} of {len(jobs)} jobs, {time.time() - started:.0f} s", flush=True)
    layout_file.unlink()
    rows, oracles, coverage = [], {}, []
    for c in cells:
        oracles[c["cell"]] = results[(c["cell"], -1)][0]["oracle"]
        observed_reps = set()
        for lo in range(0, REPLICATES, CHUNK):
            data = results[(c["cell"], lo)]
            observed_reps.update(int(r["rep"]) for r in data)
            rows.extend(data)
        assert observed_reps == set(range(REPLICATES))
        coverage.append({"cell": c["cell"], "gate": c["gate"], "replicates": len(observed_reps), "oracle": True})
    rows.sort(key=lambda r: (r["cell"], r["rep"], r.get("check", ""), r.get("degree", -1)))
    summary, familywise = V.summarise(cells, rows, oracles)
    summary.to_csv(out / "validation_gated_bba.csv", index=False, lineterminator="\n", float_format="%.17g")
    assert familywise.empty, "family-wise rows belong to gate C, which is not calibrated here"
    pd.DataFrame([{"cell": c, **oracles[c]} for c in sorted(oracles)]).to_csv(out / "validation_gated_truths_bba.csv", index=False, lineterminator="\n", float_format="%.17g")
    pd.DataFrame(coverage).to_csv(out / "replicate_coverage.csv", index=False, lineterminator="\n")
    errors = [r for r in rows if str(r.get("category", "")).startswith("error")]
    write_json(out / "replicate_errors.json", errors)
    print(f"{len(summary)} summary rows, {len(errors)} replicate errors, {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
