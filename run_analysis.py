"""Reproduce the analyses of the Article and its Supplementary Information.

    python run_analysis.py              analyses, sensitivity analyses, tables, figures and source data
    python run_analysis.py --all        also the heterogeneity test and the interval calibration
    python run_analysis.py --workers N  processes used by the interval calibration (default 8)

Each step runs in its own Python process and writes its results to output/<step>/.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [  # (output folder, script, what it computes)
    ("core", "code/core/run_core.py",
     "Molecule contributions; agreement across condition halves, partner halves and reactions; held-out pair prediction"),
    ("core_dense", "code/core/run_dense.py", "Decomposition of the six dense reactions on the second condition set"),
    ("core_intervals", "code/core_intervals/run_intervals.py", "Intervals of the core estimates"),
    ("extended", "code/extended/run_extended.py", "Second Buchwald-Hartwig campaign and structural profiles"),
    ("reaction_readings", "code/extended/reaction_readings.py",
     "Reaction-feature replication between condition sets and the illustrated aryl-halide contributions and ranks"),
    ("extended_intervals", "code/extended/run_extended_intervals.py",
     "Interval of the agreement between the two Buchwald-Hartwig campaigns"),
    ("partner_profiles", "code/partner_profiles/run_partner_profiles.py", "Structural profiles across partner halves"),
    ("independent", "code/independent/run_independent.py", "Independent implementation of the principal core estimates"),
    ("sensitivity/structure_for_selection", "code/sensitivity/structure_for_selection.py",
     "Choosing aryl bromides by a structural profile (Fig. 3d, Supplementary Note 19)"),
    ("presentation", "code/presentation/run_presentation.py",
     "Intermediate panel data, Supplementary Tables and Extended Data Table 1"),
    ("sensitivity/partner_split", "code/sensitivity/partner_split.py", "Partner-split sensitivity (Supplementary Note 8)"),
    ("sensitivity/profile_metrics", "code/sensitivity/profile_metrics.py",
     "Agreement measures of the structural profiles (Supplementary Note 18)"),
    ("sensitivity/bh_campaign", "code/sensitivity/bh_campaign.py",
     "Agreement between the two Buchwald-Hartwig campaigns (Supplementary Note 17)"),
    ("sensitivity/cross_reaction_bootstrap", "code/sensitivity/cross_reaction_bootstrap.py",
     "Molecule-cluster bootstrap of the cross-reaction agreement (Supplementary Note 10)"),
    ("sensitivity/additional", "code/sensitivity/additional_sensitivities.py",
     "Further values of the main text and Supplementary Notes 12 and 17"),
    ("sensitivity/boronic_acids_across_roles", "code/sensitivity/boronic_acids_across_roles.py",
     "Boronic acids in Suzuki-Miyaura and Chan-Lam coupling; plates and catalysts of the condition sets (Supplementary Notes 10, 14 and 17)"),
    ("sensitivity/reacting_site", "code/sensitivity/reacting_site.py",
     "The halogen beside a ring nitrogen counted at the coupling site (Supplementary Note 17)"),
    ("broad_domain", "code/dataset/analyse_broad_domain.py",
     "Corpus reconciliation and agreement across all measured conditions and connected pairs"),
    ("natchem_presentation", "code/presentation/finalize_nature_chemistry.py",
     "Nature Chemistry Figures 1-5, Extended Data Figures 1-7, Supplementary Fig. 1 and panel Source Data"),
]
SLOW_STEPS = [
    ("heterogeneity", "code/heterogeneity/run_heterogeneity.py", "Heterogeneity of the cross-reaction agreement"),
    ("calibration", "code/calibration/run_calibration.py", "Calibration of the intervals on simulated data"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also run the heterogeneity test and the interval calibration")
    parser.add_argument("--workers", type=int, default=8, help="processes used by the interval calibration")
    args = parser.parse_args()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    steps = STEPS + (SLOW_STEPS if args.all else [])
    started = time.time()
    for number, (folder, script, title) in enumerate(steps, 1):
        output = ROOT / "output" / folder
        if output.exists():
            shutil.rmtree(output)
        print(f"\n[{number}/{len(steps)}] {title}", flush=True)
        extra = (["--workers", str(args.workers)] if folder == "calibration" else
                 ["--sources-only"] if folder == "presentation" else [])
        t0 = time.time()
        if subprocess.run([sys.executable, "-B", str(ROOT / script), *extra], cwd=ROOT, env=env).returncode:
            sys.exit(f"Stopped: {script} failed.")
        print(f"[{number}/{len(steps)}] done in {time.time() - t0:.0f} s", flush=True)
    print(f"\nAll {len(steps)} steps finished in {(time.time() - started) / 60:.1f} min; results are in output/.")


if __name__ == "__main__":
    main()
