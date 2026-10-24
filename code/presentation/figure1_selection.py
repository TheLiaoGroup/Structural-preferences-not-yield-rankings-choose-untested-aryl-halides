"""Draw approved Figure 1 from analysis outputs, without changing those outputs.

Editable ChemDraw masters supply native SVG vectors at their original physical
scale. No structure bitmap is embedded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT if (ROOT / "code/presentation/plot_figures.py").exists() else ROOT / "03_GITHUB_ZENODO"
RESULTS = PACKAGE / "output"
OUT = (PACKAGE / "output/natchem_presentation/figures" if PACKAGE == ROOT else
       ROOT / "05_REPRODUCE/01_MANUSCRIPT/FINAL_FIGURE_MASTERS")
sys.path.insert(0, str(PACKAGE / "code/presentation"))
import plot_figures as BF  # noqa: E402

BLUE, RED, GREEN, GREY = "#236C9C", "#B64942", "#37806D", "#737373"
FEATURE_LABELS = {
    "halogen next to ring N": "Adjacent ring N",
    "substituent beside the halogen": "Adjacent substituent",
    "heteroaromatic ring at the halogen": "Heteroaromatic ring",
    "five-membered ring at the halogen": "Five-membered ring",
    "fused ring at the halogen": "Fused ring",
    "electron-withdrawing group": "Electron-withdrawing group",
    "electron-donating group": "Electron-donating group",
}
SELECTION_FEATURES = ["halogen next to ring N", "N–H", "substituent beside the halogen", "chloride leaving group",
                      "iodide leaving group", "sulfonate leaving group", "heteroaromatic ring at the halogen",
                      "five-membered ring at the halogen", "fused ring at the halogen",
                      "electron-withdrawing group", "electron-donating group"]


def read_data(results=RESULTS):
    core = pd.read_csv(results / "core/molecule_contributions.csv")
    campaign = pd.read_csv(results / "extended/campaign_contributions.csv")
    source = core[(core.reaction == "Buchwald-Hartwig") & (core.side == "elec")].set_index("SMILES").contribution_pp
    target = campaign[campaign.side == "elec"].set_index("SMILES").contribution_pp
    features = pd.read_csv(results / "extended/aryl_halide_features.csv").set_index("SMILES")
    if source.empty or target.empty or source.index.has_duplicates or target.index.has_duplicates:
        raise ValueError("Empty or nonunique campaign contributions")
    shared = source.index.intersection(target.index)
    unseen = target.index.difference(source.index)
    groups = [features.loc[lib.index, "leaving group"].value_counts() for lib in (source, target)]
    rank = pd.DataFrame({"SMILES": shared, "campaign_1_pp": source.loc[shared].values,
                         "campaign_2_pp": target.loc[shared].values})
    rank = rank.sort_values("campaign_1_pp", ascending=False, kind="stable").reset_index(drop=True)
    rank["molecule_id"] = np.arange(1, len(rank) + 1)
    rank["campaign_1_rank"] = rankdata(-rank.campaign_1_pp)
    rank["campaign_2_rank"] = rankdata(-rank.campaign_2_pp)
    rank["absolute_rank_change"] = abs(rank.campaign_1_rank - rank.campaign_2_rank)
    profiles = []
    for domain, libraries in (("full libraries", (source, target)),
                              ("same shared molecules", (source.loc[shared], target.loc[shared]))):
        for number, feature in enumerate(FEATURE_LABELS, 1):
            values, counts = [], []
            for library in libraries:
                flag = features.loc[library.index, feature].astype(bool)
                counts.append((int(flag.sum()), int((~flag).sum())))
                values.append(float(library.loc[flag].mean() - library.loc[~flag].mean()))
            if min(n for pair in counts for n in pair) < 3:
                if domain == "full libraries":
                    raise ValueError(f"Full profile feature not estimable: {feature}")
                continue
            profiles.append({"domain": domain, "feature_number": number, "feature": feature,
                             "campaign_1_pp": values[0], "campaign_2_pp": values[1],
                             "first_with": counts[0][0], "first_without": counts[0][1],
                             "second_with": counts[1][0], "second_without": counts[1][1]})
    profile = pd.DataFrame(profiles)
    full = profile[profile.domain.eq("full libraries")]
    same = profile[profile.domain.eq("same shared molecules")]
    score_features = [f for f in SELECTION_FEATURES if 3 <= features.loc[source.index, f].sum() <= len(source) - 3]
    score_effects = []
    for feature in score_features:
        flag = features.loc[source.index, feature].astype(bool)
        score_effects.append(source.loc[flag].mean() - source.loc[~flag].mean())
    score = pd.Series(features.loc[unseen, score_features].to_numpy(float) @ np.asarray(score_effects), index=unseen)
    selection = pd.DataFrame({"SMILES": unseen, "profile_score": score.to_numpy(),
                              "target_contribution_pp": target.loc[unseen].to_numpy()})
    selection = selection.sort_values(["profile_score", "SMILES"], ascending=[False, True]).reset_index(drop=True)
    selection["profile_selection_rank"] = np.arange(1, len(selection) + 1)
    selection["target_rank"] = rankdata(-selection.target_contribution_pp).astype(int)
    rounded_score = np.round(selection.profile_score.to_numpy(), 12)
    cut = np.sort(rounded_score)[-10]
    above, tied = rounded_score > cut, rounded_score == cut
    selection["top_ten_selection_weight"] = np.where(above, 1.0, np.where(tied, (10 - above.sum()) / tied.sum(), 0.0))
    reference = pd.read_csv(results / "sensitivity/structure_for_selection/campaign_selection_baselines.csv").set_index("rule")
    rank_r = float(spearmanr(rank.campaign_1_pp, rank.campaign_2_pp).statistic)
    full_r = float(pearsonr(full.campaign_1_pp, full.campaign_2_pp).statistic)
    shared_r = float(pearsonr(same.campaign_1_pp, same.campaign_2_pp).statistic)
    gain = float((selection.target_contribution_pp * selection.top_ten_selection_weight).sum() / 10 - selection.target_contribution_pp.mean())
    available = float(selection.nlargest(10, "target_contribution_pp").target_contribution_pp.mean() - selection.target_contribution_pp.mean())
    metrics = json.loads((results / "sensitivity/bh_campaign/BH_robustness_summary.json").read_text(encoding="utf-8"))
    for actual, expected in ((rank_r, metrics["verified"]["shared_rank_agreement"]),
                             (full_r, metrics["verified"]["pearson"]),
                             (shared_r, metrics["shared_library"]["pearson"]),
                             (gain, reference.loc["structural profile", "top_ten_gain_pp"]),
                             (available, reference.loc["structural profile", "gain_available_pp"])):
        if not np.isclose(actual, expected, atol=2e-7):
            raise ValueError(f"Recomputed Figure 1 statistic differs from analysis: {actual} versus {expected}")
    if len(same) != metrics["shared_library"]["features"] or len(shared) != metrics["verified"]["shared_aryl_halides"]:
        raise ValueError("Shared profile membership differs from analysis")
    values = {"source_n": len(source), "target_n": len(target), "shared_n": len(shared), "unseen_n": len(unseen),
              "source_leaving_groups": {str(k): int(v) for k, v in groups[0].items()},
              "target_leaving_groups": {str(k): int(v) for k, v in groups[1].items()},
              "rank_correlation": rank_r, "full_profile_correlation": full_r, "shared_profile_correlation": shared_r,
              "full_features": len(full), "shared_features": len(same), "selected_gain_pp": gain,
              "selection_features": score_features,
              "available_gain_pp": available,
              "selection_rank_correlation": float(spearmanr(score, target.loc[unseen]).statistic),
              "shared_slope_second_on_first": float(np.polyfit(same.campaign_1_pp, same.campaign_2_pp, 1)[0]),
              "shared_feature_ratio_min": float((same.campaign_2_pp / same.campaign_1_pp).min()),
              "shared_feature_ratio_max": float((same.campaign_2_pp / same.campaign_1_pp).max())}
    return rank, profile, selection, values


def structure(fig, rect, smiles, label, number=None):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    width, height = rect[2] * fig.get_figwidth() * 72, rect[3] * fig.get_figheight() * 72
    ax.set(xlim=(0, width), ylim=(0, height))
    BF.structure(ax, smiles, width / 2, height * .59, .72)
    for patch in ax.patches:
        vertices = patch.get_path().vertices
        if (vertices[:, 0].min() < 0 or vertices[:, 0].max() > width or
                vertices[:, 1].min() < 0 or vertices[:, 1].max() > height):
            raise ValueError(f"Structure drawing lies outside its allotted axes: {smiles}")
    ax.text(width / 2, height * .035, label, ha="center", va="bottom", fontsize=7.0, color=GREY)
    if number is not None:
        ax.text(0, height * .95, str(number), va="top", fontsize=8, weight="bold", color=GREEN)
    return ax


def figure1(results=RESULTS):
    rank, profile, selection, values = read_data(results)
    BF.style()
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(7.0, 7.25), facecolor="white")
    def header(x, y, letter, title):
        BF.figure_header(fig, x, y, letter, title)
    header(.035, .968, "a", "Two Buchwald–Hartwig campaigns")
    fig.text(.035, .930, f"Campaign 1: {values['source_n']} aryl (pseudo)halides", color=BLUE, fontsize=8.2)
    composition = ", ".join(f"{n} {name if name != 'sulfonate' else 'triflates'}" for name, n in values["source_leaving_groups"].items())
    fig.text(.035, .905, composition, fontsize=7, color=GREY)
    if values["target_leaving_groups"] != {"Br": values["target_n"]}:
        raise ValueError("The second campaign must contain only aryl bromides")
    fig.text(.035, .873, f"Campaign 2: {values['target_n']} aryl bromides; {values['target_n']} Br", color=RED, fontsize=8.2)
    fig.text(.035, .848, f"{values['shared_n']} shared aryl bromides; four shown", fontsize=7, color=GREY)
    show = rank.sort_values(["absolute_rank_change", "molecule_id"], ascending=[False, True]).head(4)
    for i, row in enumerate(show.itertuples()):
        structure(fig, [.035 + i % 2 * .225, .685 - i // 2 * .15, .21, .15], row.SMILES,
                  f"rank {row.campaign_1_rank:.0f} → {row.campaign_2_rank:.0f}", row.molecule_id)

    header(.56, .968, "b", "The same molecules change rank")
    b = fig.add_axes([.64, .568, .29, .335])
    illustrated = set(show.molecule_id)
    for row in rank.itertuples():
        colour = GREEN if row.molecule_id in illustrated else "#B3B3B3"
        b.plot([0, 1], [row.campaign_1_rank, row.campaign_2_rank], color=colour, lw=1.3 if row.molecule_id in illustrated else .65)
        b.scatter([0, 1], [row.campaign_1_rank, row.campaign_2_rank], s=12, color=colour, zorder=3)
        b.text(-.05, row.campaign_1_rank, str(row.molecule_id), ha="right", va="center", fontsize=6.8, color=colour)
        b.text(1.05, row.campaign_2_rank, str(row.molecule_id), ha="left", va="center", fontsize=6.8, color=colour)
    b.set(xlim=(-.18, 1.18), ylim=(len(rank) + .65, .35), xticks=[0, 1],
          xticklabels=["Campaign 1", "Campaign 2"], ylabel="Rank among the shared molecules")
    b.set_yticks([1, 5, 10, 15, len(rank)])
    b.spines[["bottom", "left", "top", "right"]].set_visible(False)
    b.tick_params(length=0)
    fig.text(.64, .931, f"{len(rank)} molecules; rank agreement {values['rank_correlation']:.2f}", fontsize=8)

    header(.035, .490, "c", "Structural preferences persist")
    c = fig.add_axes([.115, .205, .355, .225])
    c.axhline(0, color="#D8D8D8", lw=.6)
    c.axvline(0, color="#D8D8D8", lw=.6)
    full = profile[profile.domain.eq("full libraries")]
    same = profile[profile.domain.eq("same shared molecules")]
    c.scatter(full.campaign_1_pp, full.campaign_2_pp, s=28, color=GREEN, zorder=3)
    c.scatter(same.campaign_1_pp, same.campaign_2_pp, s=28, facecolors="white", edgecolors=BLUE, linewidths=1.05, zorder=3)
    offsets = {1: (4, 3), 2: (4, -9), 3: (-8, -8), 4: (4, -9), 5: (4, 3), 6: (4, -9), 7: (4, 3)}
    for row in profile.itertuples():
        offset = offsets[row.feature_number]
        if row.domain == "same shared molecules":
            offset = {1: (4, -8), 2: (-10, 3), 3: (4, -9), 4: (-10, 3), 5: (4, -8)}[row.feature_number]
        c.annotate(str(row.feature_number), (row.campaign_1_pp, row.campaign_2_pp), xytext=offset,
                   textcoords="offset points", fontsize=6.8,
                   color=GREEN if row.domain == "full libraries" else BLUE)
    c.set(xlim=(-34, 20), ylim=(-8.2, 5.8), xticks=[-30, -15, 0, 15], yticks=[-6, -3, 0, 3],
          xlabel="Campaign 1: feature difference (pp)", ylabel="Campaign 2: feature difference (pp)")
    c.spines[["top", "right"]].set_visible(False)
    legend = [Line2D([0], [0], marker="o", ls="", ms=4.5, color=GREEN,
                     label=f"Full libraries, 7 features: {values['full_profile_correlation']:.2f}"),
              Line2D([0], [0], marker="o", ls="", ms=4.5, mfc="white", mec=BLUE,
                     label=f"Same {len(rank)} molecules, 5 features: {values['shared_profile_correlation']:.2f}")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.065, .472), frameon=False,
               fontsize=7, handletextpad=.25, borderaxespad=0, labelspacing=.25)
    for i, label in enumerate(FEATURE_LABELS.values()):
        column = i // 4
        fig.text(.055 + .245 * column, .113 - (i % 4) * .025, f"{i+1}  {label}", fontsize=6.9, color=GREY)

    header(.56, .490, "d", "Choose molecules absent from campaign 1")
    fig.text(.56, .452, f"{len(selection)} untested aryl halides", fontsize=8.1)
    bars = fig.add_axes([.73, .300, .235, .115])
    bars.barh([1, 0], [values["selected_gain_pp"], values["available_gain_pp"]], color=[GREEN, "#DADADA"], height=.52)
    bars.set(yticks=[1, 0], yticklabels=["Profile-selected ten", "Best available ten"], xlim=(0, values["available_gain_pp"] * 1.22), xticks=[0, 5, 10])
    for y, value in ((1, values["selected_gain_pp"]), (0, values["available_gain_pp"])):
        bars.text(value + .15, y, f"{value:.1f}", va="center", fontsize=8, weight="bold")
    bars.spines[["left", "bottom", "top", "right"]].set_visible(False)
    bars.tick_params(axis="y", length=0, labelsize=7)
    bars.tick_params(axis="x", length=2, labelsize=7)
    fig.text(.56, .257, f"Yield gain above the {len(selection)}-molecule mean (pp)", fontsize=7.4)
    fig.text(.56, .225, "Three highest source-profile scores", fontsize=7.5, color=GREY)
    for i, row in enumerate(selection.head(3).itertuples()):
        structure(fig, [.55 + i * .148, .035, .145, .175], row.SMILES,
                  f"target {row.target_contribution_pp:+.1f} pp\nrank {row.target_rank} of {len(selection)}".replace("-", "−"))
    return fig, (rank, profile, selection, values)


def check_figure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    overlapping = []
    for i, (a, box_a) in enumerate(texts):
        for b, box_b in texts[i+1:]:
            if min(box_a.x1, box_b.x1) - max(box_a.x0, box_b.x0) > 1 and min(box_a.y1, box_b.y1) - max(box_a.y0, box_b.y0) > 1:
                overlapping.append([a.get_text(), b.get_text()])
    outside = [t.get_text() for t, box in texts if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height]
    if overlapping or outside:
        raise ValueError(f"Figure 1 text overlap: {overlapping}; outside figure: {outside}")
    return {"text_overlap_pairs": 0, "outside_figure": 0, "visible_texts": len(texts)}


def export(out=OUT, results=RESULTS, case="lower", svg_text="none"):
    fig, data = figure1(results)
    rank, profile, selection, values = data
    for text in fig.findobj(Text):
        if text.get_gid() == "panel-letter":
            text.set_text(text.get_text().upper() if case == "upper" else text.get_text().lower())
    plt.rcParams["svg.fonttype"] = svg_text
    BF.finalize(fig)
    checks = check_figure(fig)
    out.mkdir(parents=True, exist_ok=True)
    values["export_qa"] = BF.export_figure(fig, out / "Figure1", creator="build_figure1_storyboard.py")
    plt.close(fig)
    assets = out / "Figure1_source_data"
    assets.mkdir(exist_ok=True)
    rank.to_csv(assets / "Figure1_shared_molecules.csv", index=False)
    profile.to_csv(assets / "Figure1_structural_profiles.csv", index=False)
    selection.to_csv(assets / "Figure1_untested_molecules.csv", index=False)
    illustrated = rank.sort_values(["absolute_rank_change", "molecule_id"], ascending=[False, True]).head(4)
    structures = [(f"shared_{int(row.molecule_id)}", row.SMILES) for row in illustrated.itertuples()]
    structures += [(f"selected_{int(row.profile_selection_rank)}", row.SMILES) for row in selection.head(3).itertuples()]
    shown = illustrated.copy()
    shown["panel"] = "a"
    selected = selection.head(3).copy()
    selected["panel"] = "d"
    selected["illustration_rule"] = "source profile score descending, canonical SMILES ascending for ties"
    pd.concat([shown, selected], ignore_index=True).to_csv(assets / "Figure1_illustrated_structures.csv", index=False)
    for name, smiles in structures:
        (assets / f"{name}.svg").write_text(BF.depiction(smiles)[0], encoding="utf-8")
    values["qa"] = checks
    values["structure_artwork"] = "Editable ChemDraw CDXML masters with native SVG vectors; Nature-based style with larger labels and a uniform molecule bond length of 12.42 pt."
    (assets / "Figure1_metrics.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--results", type=Path, help="analysis output directory; defaults to PACKAGE/output")
    args = parser.parse_args()
    export(args.out, args.results or args.package / "output")
