"""R5 evidence figures, with unchanged analysis tables and native ChemDraw art."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import plot_figures as BF
import r5_style as S
from r5_figure2 import _r7_boundary, _r7_bounds, _r7_contains

TITLES = {
    "Figure3": [
        "Two campaigns share 17 aryl bromides",
        "Ranks change; structural preferences persist",
        "The profile recovers 6.8 of 10.4 pp",
        "Eleven molecules share ten selections",
    ],
    "Figure4": [
        "Each reaction rewards different structures",
        "Twenty new conditions preserve 90% of signs",
        "One aryl bromide ranks from 7/87 to 36/36",
    ],
    "Figure5": [
        "Choose the history that matches the candidate",
        "With a record, use the record",
        "No record: use this reaction's profile",
        "Borrow when the two reactions agree",
    ],
}

# Feature order is a visual contract shared by the strips and matrix.
FEATURES = [
    ("N–H", "N–H", "Brc1ccc2[nH]ncc2c1"),
    ("five-membered ring at the halogen", "5-membered\nring", "Brc1ccsc1"),
    ("halogen next to ring N", "Ring N\nbeside X", "Brc1ccccn1"),
    ("substituent beside the halogen", "Ortho\nsubstituent", "Cc1ccccc1Br"),
    ("fused ring at the halogen", "Fused\nring", "Brc1cccc2ccccc12"),
    ("heteroaromatic ring at the halogen", "Heteroaryl\nat X", "Brc1cccnc1"),
    ("electron-withdrawing group", "Electron-\nwithdrawing", "N#Cc1ccc(Br)cc1"),
    ("electron-donating group", "Electron-\ndonating", "COc1ccc(Br)cc1"),
]
PROFILE_LABELS = {
    "halogen next to ring N": "Ring N\nat Br",
    "substituent beside the halogen": "Ortho\ngroup",
    "heteroaromatic ring at the halogen": "Hetero-\naryl",
    "five-membered ring at the halogen": "5-ring",
    "fused ring at the halogen": "Fused\nring",
}
BONDS = {"Buchwald-Hartwig": "C–N", "Heck": "C–C", "Hiyama": "C–C",
         "Sonogashira": "C–C", "Suzuki": "C–C", "Suzuki Bpin": "C–C",
         "Ullmann": "C–O", "Kumada": "C–C", "SNAr": "C–N"}


def _figure(height_pt=538):
    S.style()
    return plt.figure(figsize=(S.WIDTH_IN, height_pt / 72), facecolor=S.WHITE)


def _rect(fig, left, bottom, width, height):
    return [left / (fig.get_figwidth() * 72), bottom / (fig.get_figheight() * 72),
            width / (fig.get_figwidth() * 72), height / (fig.get_figheight() * 72)]


def _axes(fig, left, bottom, width, height):
    return fig.add_axes(_rect(fig, left, bottom, width, height))


def _text(fig, x, y, text, **kwargs):
    fontsize = kwargs.pop("fontsize", 8)
    if fontsize == 9:
        kwargs.setdefault("gid", "key-number")
    return fig.text(x / (fig.get_figwidth() * 72), y / (fig.get_figheight() * 72), text,
                    fontsize=fontsize, **kwargs)


def _header(fig, name, letter, x, y):
    return S.header(fig, name, letter, x / (fig.get_figwidth() * 72),
                    y / (fig.get_figheight() * 72), TITLES)


def _native(fig, left, bottom, width, height, smiles, orient=None):
    return S.native(fig, _rect(fig, left, bottom, width, height), smiles, orient)


def _canvas(fig):
    ax = _axes(fig, 0, 0, fig.get_figwidth() * 72, fig.get_figheight() * 72)
    ax.set(xlim=(0, fig.get_figwidth() * 72), ylim=(0, fig.get_figheight() * 72))
    ax.set_axis_off()
    return ax


def _cmap():
    return LinearSegmentedColormap.from_list("r5-feature-difference", [S.RED, S.WHITE, S.BLUE])


def _read(p, name):
    frame = p.tab(name)
    if frame.empty:
        raise ValueError(f"R5 figure source is empty: {name}")
    return frame


def _models(p):
    source = pd.read_csv(Path(p.results) / "core/molecule_contributions.csv")
    target = pd.read_csv(Path(p.results) / "extended/campaign_contributions.csv")
    source = source[source.reaction.eq("Buchwald-Hartwig")]
    return source, target


def _rank_structure_leader(fig, native, label, molecule_id):
    """Point to the nearest native ink, leaving clear space at both ends."""
    renderer = fig.canvas.get_renderer()
    scale = 72 / fig.dpi
    bounds = label.get_window_extent(renderer)
    center = np.array([(bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2]) * scale
    segments = []
    for artist in native.get_children():
        if artist.get_gid() != "chemdraw":
            continue
        if hasattr(artist, "get_path"):
            path = artist.get_path().transformed(artist.get_transform())
            start = previous = None
            for vertices, code in path.iter_segments(curves=False):
                point = np.asarray(vertices[-2:]) * scale
                if code == MPath.MOVETO:
                    start = previous = point
                elif code == MPath.CLOSEPOLY:
                    segments.append((previous, start))
                    previous = start
                else:
                    segments.append((previous, point))
                    previous = point
        else:
            box = artist.get_window_extent(renderer)
            corners = np.array([[box.x0, box.y0], [box.x1, box.y0],
                                [box.x1, box.y1], [box.x0, box.y1]]) * scale
            segments.extend(zip(corners, np.roll(corners, -1, axis=0)))
    def closest(reference):
        nearest = []
        for a, b in segments:
            vector = b - a
            fraction = np.clip(np.dot(reference - a, vector) / max(np.dot(vector, vector), 1e-12), 0, 1)
            nearest.append(a + fraction * vector)
        return min(nearest, key=lambda point: np.linalg.norm(point - reference))

    target = closest(center)
    gap = 3.25
    side = -1 if target[0] < center[0] else 1
    first = np.array([(bounds.x0 if side == -1 else bounds.x1) * scale + side * gap, center[1]])
    # A short horizontal end keeps a displaced structure tied to its own rank.
    elbow = first + [side * 8, 0] if abs(target[1] - center[1]) > 2 else first
    target = closest(elbow)
    direction = (target - elbow) / np.linalg.norm(target - elbow)
    last = target - direction * gap
    size = np.array(fig.get_size_inches()) * 72
    points = [first, elbow, last] if not np.array_equal(first, elbow) else [first, last]
    line = Line2D(*(np.array(points) / size).T,
                  transform=fig.transFigure, color=S.PURPLE, lw=.5,
                  gid=f"r10-Figure3b-leader-{molecule_id}")
    fig.add_artist(line)


def figure3(p):
    rank = _read(p, "Figure3_shared_molecules.csv")
    profiles = _read(p, "Figure3_structural_profiles.csv")
    selection = _read(p, "Figure3_untested_molecules.csv")
    chosen = selection[selection.top_ten_selection_weight.gt(0)].copy()
    if len(rank) != 17 or len(selection) != 52 or len(chosen) != 11:
        raise ValueError("Figure 3 requires 17 shared, 52 unseen and 11 weighted selections")
    weights = chosen.top_ten_selection_weight.to_numpy()
    if not np.isclose(weights.sum(), 10) or not ((weights == 1).sum() == 6 and (weights == .8).sum() == 5):
        raise ValueError("Figure 3 selection weights differ from six full and five 0.8 selections")
    fig, width = _figure(), S.WIDTH_IN * 72
    art = _canvas(fig)

    _header(fig, "Figure3", "a", 7, 526)
    S.scheme(fig, _rect(fig, 8, 446, 244, 66), "Buchwald-Hartwig")
    _text(fig, 130, 500, "Pd / ligand / base", ha="center")
    source, target = _models(p)
    n1, n2 = source[source.side.eq("nuc")].SMILES.nunique(), target[target.side.eq("nuc")].SMILES.nunique()
    core_scope = pd.read_csv(Path(p.results) / "core/design_counts.csv")
    core_scope = core_scope[core_scope.rxn_class.eq("Buchwald-Hartwig")].iloc[0]
    first_conditions = pd.read_csv(Path(p.results) / "presentation/source_data/ExtendedDataFig1_condition_composition.csv")
    first_conditions = first_conditions[first_conditions.rxn_class.eq("Buchwald-Hartwig")]
    second_conditions = pd.read_csv(Path(p.results).parent / "data/metadata/extended_conditions.csv")
    second_conditions = second_conditions[second_conditions.rxn_class.eq("Buchwald-Hartwig(CMC)")]
    def span(values):
        lo, hi = min(values), max(values)
        return f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"
    comparison = [
        ("", "Campaign 1", "Campaign 2"),
        ("Aryl (pseudo)halides", f"{len(source[source.side.eq('elec')]):,}", f"{len(target[target.side.eq('elec')]):,} Br"),
        ("Amines", f"{n1:,}", f"{n2:,}"),
        ("Conditions", f"{int(core_scope.conditions):,}", f"{second_conditions.condition_id.nunique():,}"),
        ("Temperature / time", f"{span(first_conditions.temperature_C)} °C / {span(first_conditions.time_h)} h",
         f"{span(second_conditions.temp_id)} °C / {span(second_conditions.time_id)} h"),
    ]
    for i, (label, first, second) in enumerate(comparison):
        y = 507 - i * 11
        _text(fig, 256, y, label, va="center", color=S.MUTED)
        _text(fig, 382, y, first, ha="center", va="center", weight="bold" if i == 0 else "normal")
        _text(fig, 462, y, second, ha="center", va="center", weight="bold" if i == 0 else "normal")
    _header(fig, "Figure3", "b", 7, 427)
    rank_rho = float(spearmanr(rank.campaign_1_pp, rank.campaign_2_pp).statistic)
    _text(fig, 90, 412, f"Molecule ranks: ρ = {rank_rho:.2f}", color=S.PURPLE, weight="bold", fontsize=9)
    ax = _axes(fig, 89, 272, 126, 132)
    show = rank.sort_values(["absolute_rank_change", "molecule_id"], ascending=[False, True]).head(4)
    shown = set(show.molecule_id)
    rank_labels = {}
    for row in rank.itertuples():
        highlighted = row.molecule_id in shown
        ax.plot([0, 1], [row.campaign_1_rank, row.campaign_2_rank], color=S.PURPLE,
                alpha=1 if highlighted else .23, lw=1 if highlighted else .5)
        ax.plot([0, 1], [row.campaign_1_rank, row.campaign_2_rank], "o", ms=2.5,
                color=S.PURPLE, alpha=1 if highlighted else .3)
        # Each endpoint names its own campaign's rank.
        rank_labels[row.molecule_id, 0] = ax.text(-.055, row.campaign_1_rank, f"{row.campaign_1_rank:g}",
                                                ha="right", va="center", color=S.PURPLE)
        rank_labels[row.molecule_id, 1] = ax.text(1.055, row.campaign_2_rank, f"{row.campaign_2_rank:g}",
                                                ha="left", va="center", color=S.PURPLE)
    ax.set(xlim=(-.13, 1.13), ylim=(17.6, .4), xticks=[0, 1],
           xticklabels=["Campaign 1", "Campaign 2"], yticks=[])
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    slots = {1: (8, 359, 75, 44, 0), 14: (8, 273, 75, 48, 0),
             3: (231, 302, 69, 44, 1), 5: (231, 261, 69, 44, 1)}
    structures = {}
    for row in show.itertuples():
        left, bottom, sw, sh, side = slots[int(row.molecule_id)]
        native = _native(fig, left, bottom, sw, sh, row.SMILES,
                         orient=5 if row.molecule_id == 5 else None)
        native.set_gid(f"Figure3b-structure-{int(row.molecule_id)}")
        structures[int(row.molecule_id)] = native
        label = rank_labels[row.molecule_id, side]
        label.set_gid(f"r10-Figure3b-rank-{int(row.molecule_id)}")
    fig.canvas.draw()
    for molecule_id, native in structures.items():
        side = slots[molecule_id][-1]
        _rank_structure_leader(fig, native, rank_labels[molecule_id, side], molecule_id)

    shared = profiles[profiles.domain.eq("same shared molecules")].sort_values("feature_number")
    if len(shared) != 5:
        raise ValueError("Figure 3b requires the five shared-library contrasts")
    x0, cellw, rowh, y0 = 307, 36, 24, 340
    values = shared[["campaign_1_pp", "campaign_2_pp"]].to_numpy().T
    limit = float(np.max(np.abs(values)))
    norm, cmap = TwoSlopeNorm(0, -limit, limit), _cmap()
    profile_r = float(np.corrcoef(values)[0, 1])
    _text(fig, x0, 412, f"Feature profiles: r = {profile_r:.2f}", color=S.TEAL, fontsize=9, weight="bold")
    for j, row in enumerate(shared.itertuples()):
        _text(fig, x0 + (j + .5) * cellw, 390, PROFILE_LABELS[row.feature], ha="center", va="top")
        for i, value in enumerate((row.campaign_1_pp, row.campaign_2_pp)):
            art.add_patch(Rectangle((x0 + j * cellw, y0 - i * rowh), cellw, rowh,
                                    facecolor=cmap(norm(value)), edgecolor=S.WHITE, lw=.5))
            _text(fig, x0 + (j + .5) * cellw, y0 + rowh / 2 - i * rowh,
                  S.signed(value, 1), ha="center", va="center",
                  color=S.WHITE if abs(value) > .6 * limit else S.INK)
    _text(fig, x0, 367, "Campaign 1", color=S.MUTED)
    _text(fig, x0, 307, "Campaign 2", color=S.MUTED)
    bar = _axes(fig, x0, 291, 180, 5)
    bar.imshow(np.linspace(-limit, limit, 256)[None, :], extent=(-limit, limit, 0, 1), aspect="auto", cmap=cmap, norm=norm)
    bar.set(yticks=[], xticks=[-limit, 0, limit], xticklabels=[S.signed(-limit), "0", S.signed(limit)])
    bar.tick_params(length=0, pad=2)
    for spine in bar.spines.values():
        spine.set_visible(False)
    _text(fig, 496, 300, "pp", ha="left", va="center")
    _text(fig, x0, 269, "Same feature signs; campaign 2\ndifferences are about one-eighth as large", va="top", color=S.MUTED)

    _header(fig, "Figure3", "c", 7, 243)
    baselines = pd.read_csv(Path(p.results) / "sensitivity/structure_for_selection/campaign_selection_baselines.csv").set_index("rule")
    available = float(baselines.loc["structural profile", "gain_available_pp"])
    selected_gain = float(baselines.loc["structural profile", "top_ten_gain_pp"])
    similarity = baselines.loc[baselines.index.str.startswith("nearest aryl") | baselines.index.str.startswith("similarity-weighted"), "top_ten_gain_pp"].max()
    ls = float(baselines.loc["least squares on the same features", "top_ten_gain_pp"])
    ruler = _axes(fig, 295, 167, 190, 47)
    methods = [("Random", 0, S.GREY, 0), ("Similarity", similarity, S.GREY, 1),
               ("Least squares", ls, S.GREY, 0), ("Profile", selected_gain, S.TEAL, 1),
               ("Available", available, S.MUTED, 0)]
    ruler.axhline(0, color=S.GREY, lw=.6)
    for label, gain, colour, level in methods:
        yy = .7 if level else -.7
        ruler.plot(gain, 0, "D" if label == "Profile" else "o", color=colour, ms=4)
        ruler.plot([gain, gain], [0, yy * .75], color=colour, lw=.5)
        ruler.text(gain, yy, f"{label}\n{gain:.1f}", ha="center", va="bottom" if level else "top", color=colour,
                   weight="bold" if label == "Profile" else "normal")
    ruler.set(xlim=(-.9, 11.2), ylim=(-2, 2), xticks=[], yticks=[])
    for spine in ruler.spines.values():
        spine.set_visible(False)
    _text(fig, 390, 151, "Gain above random selection (pp)", ha="center")
    ax = _axes(fig, 50, 165, 210, 61)
    background = selection.top_ten_selection_weight.eq(0)
    ax.scatter(selection.loc[background, "profile_selection_rank"], selection.loc[background, "target_contribution_pp"],
               s=9, color=S.GREY, lw=0)
    for weight, fillstyle in ((1, "full"), (.8, "left")):
        rows = selection[selection.top_ten_selection_weight.eq(weight)]
        ax.plot(rows.profile_selection_rank, rows.target_contribution_pp, linestyle="none", marker="D",
                markersize=np.sqrt(15), fillstyle=fillstyle, markerfacecolor=S.TEAL,
                markerfacecoloralt=S.WHITE, markeredgecolor=S.TEAL, markeredgewidth=.8, zorder=3)
    ax.axhline(selection.target_contribution_pp.mean(), color=S.GREY, lw=.5, ls="--")
    ax.annotate("mean of 52", (52, selection.target_contribution_pp.mean()),
                xytext=(-3, 4), textcoords="offset points", ha="right", va="bottom",
                color=S.MUTED, gid="r12-Figure3c-mean-label")
    handles = [Line2D([], [], marker="D", ls="", ms=np.sqrt(15), fillstyle=fill,
                      mfc=S.TEAL, mfcalt=S.WHITE, mec=S.TEAL, mew=.8, label=label)
               for fill, label in (("full", "Weight 1"), ("left", "Weight 0.8"))]
    legend = ax.legend(handles=handles, loc="upper right", ncol=2, handlelength=.6,
                       handletextpad=.4, columnspacing=1, frameon=False, borderpad=0,
                       borderaxespad=.4, fontsize=8)
    legend.set_gid("r12-Figure3c-weight-legend")
    ax.set(xlim=(0, 53), xticks=[1, 10, 30, 52], yticks=[-10, 0, 20],
           xlabel="Campaign 1 profile rank", ylabel="Campaign 2 (pp)")
    ax.tick_params(length=2, pad=2)
    S.clean(ax)
    _text(fig, 258, 231, "52 new aryl bromides", ha="right", color=S.ORANGE)

    _header(fig, "Figure3", "d", 7, 129)
    features = pd.read_csv(Path(p.results) / "extended/aryl_halide_features.csv").set_index("SMILES")
    marker_labels = {"halogen next to ring N": "ring N", "fused ring at the halogen": "fused",
                     "electron-withdrawing group": "EWG", "electron-donating group": "EDG"}
    for rownum, weight in enumerate((1., .8)):
        rows = chosen[chosen.top_ten_selection_weight.eq(weight)].sort_values("profile_selection_rank")
        n, cell = len(rows), (width - 16) / 6
        offset = 8 if rownum == 0 else 8 + cell / 2
        bottom = 70 if rownum == 0 else 12
        if rownum == 1:
            art.add_patch(Rectangle((offset - 3, 2), cell * n + 6, 58, facecolor=S.LIGHT,
                                    edgecolor=S.TEAL, lw=.5))
        for j, row in enumerate(rows.itertuples()):
            x = offset + j * cell
            _native(fig, x, bottom + 11, cell - 4, 42, row.SMILES)
            _text(fig, x + (cell - 4) / 2, bottom + 2, f"{S.signed(row.target_contribution_pp, 1)} pp", ha="center",
                  color=S.TEAL, weight="bold" if row.SMILES == "COC(=O)c1ccnc(Br)c1" else "normal")
            labels = ["ring N (via F)" if row.SMILES == "Fc1ncc(Br)cn1" and label == "ring N" else label
                      for feature, label in marker_labels.items() if bool(features.loc[row.SMILES, feature])]
            label = _text(fig, x + (cell - 4) / 2, bottom - 7, " · ".join(labels), ha="center", color=S.ORANGE)
            if row.SMILES == "Fc1ncc(Br)cn1":
                label.set_gid("r7-Figure3d-via-F-label")
    _text(fig, width - 5, 121, "6 × 1 + 5 × 0.8 = 10", ha="right", color=S.TEAL)
    fig._r5_selection_weights = weights.tolist()
    fig._r5_profile_common_limit_pp = limit
    fig._r5_feature_markers = list(marker_labels)
    return fig


def figure4(p):
    cells = _read(p, "Figure4a_feature_matrix.csv")
    order = _read(p, "Figure4a_reaction_order.csv").sort_values("row_order")
    reactions, features = order.reaction.tolist(), [row[0] for row in FEATURES]
    if len(order) != 9 or len(cells) != 72:
        raise ValueError("Figure 4a must contain nine reactions and eight structural features")
    matrix = cells.pivot(index="reaction", columns="feature", values="difference_pp").reindex(index=reactions, columns=features).to_numpy()
    pvals = cells.pivot(index="reaction", columns="feature", values="mann_whitney_p").reindex(index=reactions, columns=features).to_numpy()
    fig, width = _figure(515), S.WIDTH_IN * 72
    art = _canvas(fig)
    _header(fig, "Figure4", "a", 7, 503)
    x0, colw, y0, matrix_h = 149, 42.5, 206, 180
    slots = []
    for j, (_, label, smiles) in enumerate(FEATURES):
        slots.append(_native(fig, x0 + j * colw, 420, colw, 68, smiles, -1))
        _text(fig, x0 + (j + .5) * colw, 412, label, ha="center", va="top")
    ax = _axes(fig, x0, y0, colw * 8, matrix_h)
    limit, cmap = float(np.nanmax(np.abs(matrix))), _cmap()
    norm = TwoSlopeNorm(0, -limit, limit)
    cmap.set_bad(S.WHITE, alpha=1)
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=norm,
                      interpolation="nearest")
    for i, reaction in enumerate(reactions):
        for j in range(8):
            value = matrix[i, j]
            ax.text(j, i, "·" if not np.isfinite(value) else S.signed(value), ha="center", va="center",
                    color=S.MUTED if not np.isfinite(value) else S.WHITE if abs(value) > .65 * limit else S.INK,
                    weight="bold" if pvals[i, j] < .05 else "normal")
        metal = str(order.metal.iloc[i]).replace("mixed Fe/Ni/Pd", "Fe/Ni/Pd")
        name = S.reaction_name(reaction)
        _text(fig, x0 - 5, y0 + matrix_h - (i + .5) * matrix_h / 9 + 3, name, ha="right", va="center")
        _text(fig, x0 - 5, y0 + matrix_h - (i + .5) * matrix_h / 9 - 6, f"{BONDS[reaction]} · {metal}",
              ha="right", va="center", color=S.MUTED)
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)
    highlighted_column = features.index("halogen next to ring N")
    ax.add_patch(Rectangle((highlighted_column - .5, -.5), 1, 9, fill=False, edgecolor=S.INK, lw=1.2, clip_on=False))
    for reaction in ("Heck", "Ullmann"):
        i = reactions.index(reaction)
        ax.add_patch(Rectangle((highlighted_column - .5, i - .5), 1, 1, fill=False, edgecolor=S.WHITE, lw=2.8))
        ax.add_patch(Rectangle((highlighted_column - .5, i - .5), 1, 1, fill=False, edgecolor=S.INK, lw=1.2))
    # A bracket connects the two Suzuki rows without changing row-label colours.
    sy = [y0 + matrix_h - (reactions.index(r) + .5) * matrix_h / 9 for r in ("Suzuki", "Suzuki Bpin")]
    art.plot([492, 497, 497, 492], [max(sy) + 8, max(sy) + 8, min(sy) - 8, min(sy) - 8], color=S.INK, lw=.6)
    _text(fig, 503, np.mean(sy), "Same bond, opposite reading", rotation=90, va="center", ha="center")
    cbar = fig.colorbar(image, cax=_axes(fig, x0, 197, 240, 5), orientation="horizontal", ticks=[-15, 0, 15])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, pad=2)
    _text(fig, 395, 195, "Yield difference (pp)", va="center")
    _text(fig, 95, 174, "Bold: P < 0.05", weight="bold", gid="r12-Figure4a-bold-legend")
    _text(fig, 212, 174, "·  class n < 3", gid="r12-Figure4a-small-class-legend")
    art.add_patch(Rectangle((322, 172), 7, 7, fill=False, edgecolor=S.INK, lw=.8,
                           gid="r12-Figure4a-frame-key"))
    _text(fig, 335, 174, "Ring N beside halogen", gid="r12-Figure4a-frame-legend")
    fig._matrix_alignment = ax, slots

    _header(fig, "Figure4", "b", 7, 160)
    paired = _read(p, "Figure4b_paired_feature_contrasts.csv")
    summary = _read(p, "Figure4b_paired_feature_summary.csv").iloc[0]
    if len(paired) != 67 or int(summary.paired_cells) != 67:
        raise ValueError("Figure 4b must compare 67 paired reaction–feature cells")
    ax = _axes(fig, 48, 34, 171, 106)
    xx, yy = paired.first_difference_pp, paired.second_difference_pp
    lo, hi = min(xx.min(), yy.min()) - 3, max(xx.max(), yy.max()) + 3
    ax.plot([lo, hi], [lo, hi], "--", color=S.GREY, lw=.5)
    ax.axhline(0, color=S.GREY, lw=.5)
    ax.axvline(0, color=S.GREY, lw=.5)
    ax.scatter(xx, yy, s=10, color=S.TEAL, lw=0)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="First 20 conditions (pp)", ylabel="Next 20 conditions (pp)")
    ax.text(.02, .98, f"67 cells; {summary.same_sign_fraction:.0%} same sign\nr = {summary.pearson_r:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, color=S.TEAL)
    S.clean(ax)

    _header(fig, "Figure4", "c", 247, 160)
    rows = _read(p, "Figure4c_methyl_2_bromoisonicotinate.csv")
    expected = [("Sonogashira", 7, 87), ("Heck", 56, 59), ("Suzuki", 98, 105), ("Suzuki Bpin", 36, 36)]
    observed = list(zip(rows.reaction, rows["rank"].astype(int), rows.total_molecules.astype(int)))
    if observed != expected:
        raise ValueError(f"Figure 4c ranks differ: {observed}")
    _native(fig, 254, 69, 77, 66, rows.SMILES.iloc[0], -1)
    _text(fig, 292, 50, "Methyl 2-bromo-\nisonicotinate", ha="center", va="top")
    ax = _axes(fig, 351, 42, 143, 96)
    for j, row in enumerate(rows.itertuples()):
        n = int(row.total_molecules)
        yy = np.linspace(1, 0, n)
        ax.plot(np.full(n, j), yy, "_", color=S.GREY, ms=7, mew=.5)
        target = 1 - (int(row.rank) - 1) / (n - 1)
        ax.plot(j, target, "D", color=S.PURPLE, ms=4, zorder=3)
        ax.text(j, 1.12, f"{int(row.rank)}/{n}", ha="center", color=S.PURPLE, fontsize=9, weight="bold")
    ax.set(xlim=(-.5, 3.5), ylim=(-.07, 1.08), yticks=[], xticks=range(4),
           xticklabels=["Sonoga-\nshira", S.reaction_name("Heck"),
                        S.reaction_name("Suzuki").replace("–", "–\n"),
                        S.reaction_name("Suzuki Bpin").replace("–", "–\n").replace(" (", "\n(")])
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def _rank_icon(ax, cx, cy, colour, parallel=False):
    left = (6, 2, -2, -6) if parallel else (10, 4, -3, -9)
    right = left if parallel else (-6, 9, -10, 3)
    for index, (a, b) in enumerate(zip(left, right), 1):
        ax.plot([cx - 16, cx + 16], [cy + a, cy + b], color=colour, lw=.7,
                gid=f"r8-borrow-rank-{index}" if parallel else None)
        ax.plot([cx - 16, cx + 16], [cy + a, cy + b], "o", color=colour, ms=2)


def _profile_icon(ax, cx, cy, double=False):
    palette = [S.BLUE, S.RED, S.RED, S.BLUE]
    for row in range(2 if double else 1):
        for j, colour in enumerate(palette):
            ax.add_patch(Rectangle((cx - 19 + j * 10, cy - row * 10), 8, 7,
                                   facecolor=colour, edgecolor=colour, lw=.5))


def _column_header(fig, letter, left, baseline, width):
    _, title = _header(fig, "Figure5", letter, left, baseline)
    title.set_verticalalignment("top")
    title.set_y((baseline + 6.5) / (fig.get_figheight() * 72))
    title.set_wrap(True)
    # Constrain the displayed line breaks while preserving the exact panel-title text.
    title._get_wrap_line_width = lambda: (width - 20) * fig.dpi / 72


def _column_reaction(name):
    return S.reaction_name(name).replace("–", "–\n").replace(" (", "\n(").replace(" ether", "\nether")


def figure5(p):
    record_rows = _read(p, "Figure5b_partner_selection.csv")
    unseen_rows = _read(p, "Figure5c_selection_without_record.csv")
    borrow = _read(p, "Figure5d_selection_vs_agreement.csv")
    better = int((record_rows.median_ranking_pp > record_rows.median_profile_pp).sum())
    bh_gain = float(unseen_rows.loc[unseen_rows.source.eq("Buchwald-Hartwig"), "top_ten_gain_pp"].item())
    other_gain = float(unseen_rows.loc[~unseen_rows.source.eq("Buchwald-Hartwig"), "top_ten_gain_pp"].max())
    heck = borrow[borrow.rxn_a.eq("Heck") & borrow.rxn_b.eq("Suzuki") & borrow.side.eq("elec")].iloc[0]
    amines = borrow[borrow.rxn_a.eq("Buchwald-Hartwig") & borrow.rxn_b.eq("SNAr") & borrow.side.eq("nuc")].iloc[0]
    fig, width = _figure(), S.WIDTH_IN * 72
    art = _canvas(fig)
    for letter, (left, bottom, right, top) in {
        "a": (5, 348, width, 538), "b": (5, 0, 171, 348),
        "c": (175, 0, 341, 348), "d": (345, 0, width, 348),
    }.items():
        _r7_boundary(fig, f"r7-Figure5-panel-{letter}", _rect(fig, left, bottom, right - left, top - bottom))
    _header(fig, "Figure5", "a", 7, 526)
    _text(fig, 254, 506, "Measured in this reaction?", ha="center", fontsize=9, weight="bold")
    centers = (85, 254, 425)
    for x in (centers[0], (centers[1] + centers[2]) / 2):
        art.plot([254, x], [498, 490], color=S.GREY, lw=.6)
    _text(fig, centers[0], 480, "Yes", ha="center", color=S.MUTED)
    no_center = (centers[1] + centers[2]) / 2
    _text(fig, no_center, 480, "No", ha="center", color=S.MUTED)
    art.plot([centers[1], centers[2]], [470, 470], color=S.GREY, lw=.6)
    art.plot([no_center, no_center], [477, 470], color=S.GREY, lw=.6)
    for x in centers[1:]:
        art.plot([x, x], [470, 465], color=S.GREY, lw=.6)
    _text(fig, centers[1], 459, "Profile available\nfor this reaction", ha="center", va="top", color=S.TEAL)
    _text(fig, centers[2], 459, "Another reaction ranks\nshared molecules alike", ha="center", va="top", color=S.MUTED,
          gid="r7-Figure5a-borrow-condition")
    _rank_icon(art, centers[0], 434, S.PURPLE)
    _profile_icon(art, centers[1] - 37, 430)
    _text(fig, centers[1] - 10, 433.5, "Structural profile", va="center", color=S.MUTED,
          gid="r13-Figure5a-profile-label")
    _rank_icon(art, centers[2], 429, S.PURPLE, parallel=True)
    branches = [
        ("Use the record", f"Better in {better}/{len(record_rows)} reactions", "", S.PURPLE),
        ("Use this reaction's profile", f"Buchwald–Hartwig: {bh_gain:.1f} pp",
         f"Other profiles ≤{other_gain:.1f} pp", S.TEAL),
        ("Borrow from that reaction", f"{S.signed(heck.source_gain_pp, 1)} of {heck.target_gain_pp:.1f} pp",
         f"Buchwald–Hartwig → SNAr:\n{S.signed(amines.source_gain_pp, 1)} pp", S.MUTED),
    ]
    for x, (label, key, qualifier, colour) in zip(centers, branches):
        _text(fig, x, 417, label, ha="center", va="top", color=colour)
        _text(fig, x, 400, key, ha="center", va="top", color=colour, fontsize=9)
        if qualifier:
            _text(fig, x, 384, qualifier, ha="center", va="top", color=colour)
        art.annotate("", (x, 354), (x, 365),
                     arrowprops=dict(arrowstyle="-|>", color=colour, lw=.6, mutation_scale=7))

    _column_header(fig, "b", 7, 340, 163)
    rows = record_rows.sort_values("median_ranking_pp", ascending=False)
    if len(rows) != 7 or int((rows.median_ranking_pp > rows.median_profile_pp).sum()) != 5:
        raise ValueError("Figure 5b requires records to exceed profiles in five of seven reactions")
    ax = _axes(fig, 83, 61, 77, 250)
    for i, row in enumerate(rows.itertuples()):
        ax.plot([row.median_profile_pp, row.median_ranking_pp], [i, i], color=S.GREY, lw=.7)
    ax.plot(rows.median_ranking_pp, range(len(rows)), "o", color=S.PURPLE, ms=4, label="Record")
    ax.plot(rows.median_profile_pp, range(len(rows)), "D", color=S.TEAL, ms=3.7, label="Profile")
    ax.set(yticks=range(len(rows)), yticklabels=[_column_reaction(r) for r in rows.reaction],
           xlabel="Gain with new\npartners (pp)", xticks=[0, 10, 20], xlim=(-1, 20))
    ax.invert_yaxis()
    ax.axvline(0, color=S.GREY, lw=.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.045), frameon=False, ncol=2, handlelength=.6, handletextpad=.5,
              borderpad=0, labelspacing=.3)
    S.clean(ax)

    _column_header(fig, "c", 177, 340, 163)
    rows = unseen_rows.sort_values("top_ten_gain_pp", ascending=False)
    ax = _axes(fig, 255, 61, 78, 250)
    if len(rows) != 9:
        raise ValueError("Figure 5c must compare nine source-reaction profiles")
    ax.barh(range(len(rows)), rows.top_ten_gain_pp, height=.6,
            color=[S.TEAL if r == "Buchwald-Hartwig" else S.GREY for r in rows.source])
    ax.set(yticks=range(len(rows)), yticklabels=[_column_reaction(r) for r in rows.source],
           xlabel="Gain for 52 new\naryl bromides (pp)", xlim=(-2, 11.3), xticks=[0, 5, 10])
    ax.invert_yaxis()
    best = float(rows.gain_available_pp.iloc[0])
    ax.axvline(best, color=S.MUTED, ls="--", lw=.7)
    ax.text(best, -.8, f"Available {best:.1f}", ha="right", color=S.MUTED)
    best_bh = rows[rows.source.eq("Buchwald-Hartwig")].iloc[0]
    ax.text(best_bh.top_ten_gain_pp + .2, list(rows.source).index("Buchwald-Hartwig"),
            f"{best_bh.top_ten_gain_pp:.1f}", va="center", color=S.TEAL, weight="bold")
    S.clean(ax)

    _column_header(fig, "d", 347, 340, 163)
    if len(borrow) != 20:
        raise ValueError("Figure 5d must show 20 cross-reaction selections")
    ax = _axes(fig, 389, 46, 110, 135)
    ax.set_gid("r7-Figure5d-scatter-axes")
    _r7_boundary(fig, "r7-Figure5d-plot", _rect(fig, 389, 46, 110, 135))
    for side, marker, label in (("elec", "o", "Electrophiles"), ("nuc", "^", "Nucleophiles")):
        subset = borrow[borrow.side.eq(side)]
        ax.scatter(subset.rho_obs, subset.source_gain_pp, color=S.PURPLE, marker=marker, s=15, lw=0, label=label)
    ax.axhline(0, color=S.GREY, lw=.5, gid="r7-Figure5d-zero-axis")
    ax.axvline(0, color=S.GREY, lw=.5, gid="r7-Figure5d-vertical-zero-axis")
    annotation = ax.annotate(f"Heck →\n{S.reaction_name('Suzuki')}\n{S.signed(heck.source_gain_pp, 1)} of {heck.target_gain_pp:.1f} pp",
                (heck.rho_obs, heck.source_gain_pp), xytext=(.995, 1.03), textcoords="axes fraction",
                color=S.PURPLE, ha="right", va="top", gid="r7-Figure5d-heck-annotation",
                arrowprops=dict(arrowstyle="-", color=S.GREY, lw=.5, shrinkB=2))
    annotation.arrow_patch.set_gid("r7-Figure5d-heck-leader")
    ax.set(xlabel="Rank agreement\nbetween reactions", ylabel="Gain above random\nten (pp)",
           xlim=(-.65, .95), ylim=(-4.4, 11), xticks=[-.5, 0, .5], yticks=[0, 5, 10])
    ax.legend(loc="lower left", bbox_to_anchor=(-.35, 1.08), frameon=False, ncol=2,
              handlelength=.7, handletextpad=.4, columnspacing=.7, borderpad=0, labelspacing=.3)
    S.clean(ax)
    examples = _read(p, "Figure5d_Heck_Suzuki_molecules.csv")
    shown = examples[examples.drawn.fillna(0).gt(0)].sort_values("drawn")
    if len(shown) != 4:
        raise ValueError("Figure 5d must illustrate four shared aryl bromides")
    import figures_2_3 as F23
    _text(fig, 425, 318, f"Heck / {S.reaction_name('Suzuki')} (pp)", ha="center", color=S.MUTED)
    for i, row in enumerate(shown.itertuples()):
        x, y = 347 + i % 2 * 80, 263 - i // 2 * 55
        _native(fig, x, y + 7, 77, 46, row.mol, F23.ORIENT.get(row.mol, 0))
        _text(fig, x + 38.5, y + 1, f"{S.signed(row.x_contribution_pp)} / {S.signed(row.y_contribution_pp)}", ha="center",
              gid=f"r10-Figure5d-value-{i + 1}")
    fig._r7_borrow_point = (float(heck.rho_obs), float(heck.source_gain_pp))
    fig._r7_borrow_annotation = annotation
    fig._r7_borrow_source = borrow[["rxn_a", "rxn_b", "side", "rho_obs", "source_gain_pp", "target_gain_pp"]].to_dict("records")
    return fig


def geometry_metadata(fig, svg_path=None):
    """Measure actual Figure 5 artists after finalization; export owns serialization."""
    import hashlib
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    objects = {artist.get_gid(): artist for artist in fig.findobj() if artist.get_gid()}
    panels = {letter: _r7_bounds(fig, objects[f"r7-Figure5-panel-{letter}"].get_window_extent(renderer))
              for letter in "abcd"}
    ax = objects["r7-Figure5d-scatter-axes"]
    scatter = _r7_bounds(fig, ax.get_window_extent(renderer))
    annotation = fig._r7_borrow_annotation
    # Annotation.get_window_extent also includes its arrow; Text measures the label alone.
    label = _r7_bounds(fig, Text.get_window_extent(annotation, renderer))
    vertices = annotation.arrow_patch.get_transform().transform(annotation.arrow_patch.get_path().vertices)
    vertices_pt = [[float(x * 72 / fig.dpi), float((fig.bbox.height - y) * 72 / fig.dpi)] for x, y in vertices]
    leader_length = float(np.linalg.norm(np.diff(vertices_pt, axis=0), axis=1).sum())
    zero = ax.transData.transform((0, 0))
    zero_x, zero_y = float(zero[0] * 72 / fig.dpi), float((fig.bbox.height - zero[1]) * 72 / fig.dpi)
    xclearance = max(label[0] - zero_x, zero_x - label[2])
    yclearance = max(label[1] - zero_y, zero_y - label[3])
    assert xclearance > 1, "Figure5 annotation crosses vertical zero line"
    assert yclearance > 1, "Figure5 annotation crosses horizontal zero line"
    assert _r7_contains(panels["d"], label), "Figure5 annotation outside panel d"
    assert leader_length <= 21.6, "Figure5 leader exceeds 0.3 in"
    assert all(_r7_contains(scatter, [x, y, x, y]) for x, y in vertices_pt), "Figure5 leader outside plot"
    segments = []
    for artist in fig.findobj(Line2D):
        if not artist.get_visible():
            continue
        points = artist.get_transform().transform(artist.get_path().vertices)
        points = [[float(x * 72 / fig.dpi), float((fig.bbox.height - y) * 72 / fig.dpi)] for x, y in points]
        for start, end in zip(points, points[1:]):
            if any(_r7_contains(scatter, [x, y, x, y]) for x, y in (start, end)):
                assert all(_r7_contains(scatter, [x, y, x, y]) for x, y in (start, end)), "Figure5 line extends outside plot"
                segments.append([start, end])
    centers = []
    for collection in ax.collections:
        for x, y in collection.get_offset_transform().transform(collection.get_offsets()):
            centers.append([float(x * 72 / fig.dpi), float((fig.bbox.height - y) * 72 / fig.dpi)])
    for center in centers:
        assert not (label[0] - 2 <= center[0] <= label[2] + 2 and label[1] - 2 <= center[1] <= label[3] + 2), "Figure5 annotation overlaps data"
    return {
        "coordinate_system": "PDF/SVG points, top-left origin, x right, y down",
        "page_size_pt": [float(fig.get_figwidth() * 72), float(fig.get_figheight() * 72)],
        "svg_sha256": hashlib.sha256(Path(svg_path).read_bytes()).hexdigest() if svg_path else None,
        "panels_pt": panels, "panel_d_pt": panels["d"], "scatter_d_pt": scatter,
        "zero_axis_x_pt": zero_x, "zero_axis_y_pt": zero_y,
        "annotation_bounds_pt": label, "annotation_vertical_zero_clearance_pt": xclearance,
        "annotation_horizontal_zero_clearance_pt": yclearance,
        "leader_vertices_pt": vertices_pt, "leader_length_pt": leader_length,
        "plot_line_segments_pt": segments, "data_markers_pt": centers,
        "source_anchors": {"rxn_a": "Heck", "rxn_b": "Suzuki", "side": "elec",
                           "rho_obs": fig._r7_borrow_point[0], "source_gain_pp": fig._r7_borrow_point[1]},
        "source_data_rows": fig._r7_borrow_source,
    }


def source_data_inventory(p):
    """Structures explicitly shown in evidence panels, for source-data provenance."""
    rank = _read(p, "Figure3_shared_molecules.csv")
    selection = _read(p, "Figure3_untested_molecules.csv")
    shared = rank.sort_values(["absolute_rank_change", "molecule_id"], ascending=[False, True]).head(4).copy()
    shared["panel"] = "Figure3b"
    selected = selection[selection.top_ten_selection_weight.gt(0)].copy()
    selected["panel"] = "Figure3d"
    return pd.concat([shared, selected], ignore_index=True)
