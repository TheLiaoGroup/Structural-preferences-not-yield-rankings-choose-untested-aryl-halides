"""Chemical context and molecular rankings, drawn from R5 Source Data only."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

import r5_style as S
from figures_2_3 import ORIENT

TITLES = [
    "Rankings disperse as the chemical context changes",
    "Changing partners weakens rankings across reactions",
    "The same amines reverse order",
    "Boronic acids retain a shared preference",
]


def _text(fig, x, y, label, key=False, **kwargs):
    kwargs.setdefault("gid", "key-number" if key else None)
    return fig.text(x, y, label, fontsize=9 if key else 8, **kwargs)


def _true(series):
    return series.astype(str).str.lower().eq("true")


def _r7_boundary(fig, gid, rect):
    artist = Rectangle(rect[:2], rect[2], rect[3], transform=fig.transFigure,
                       facecolor="none", edgecolor="none", gid=gid)
    fig.add_artist(artist)
    return artist


def _r7_bounds(fig, bbox):
    scale, height = 72 / fig.dpi, fig.bbox.height
    return [float(bbox.x0 * scale), float((height - bbox.y1) * scale),
            float(bbox.x1 * scale), float((height - bbox.y0) * scale)]


def _r7_contains(outer, inner, tolerance=.05):
    return (inner[0] >= outer[0] - tolerance and inner[1] >= outer[1] - tolerance
            and inner[2] <= outer[2] + tolerance and inner[3] <= outer[3] + tolerance)


def _r7_clearance(rect, center, radius=0):
    dx = max(rect[0] - center[0], center[0] - rect[2], 0)
    dy = max(rect[1] - center[1], center[1] - rect[3], 0)
    return float(np.hypot(dx, dy) - radius)


def _clouds(fig, p):
    points = p.tab("Figure2a_cloud_points.csv")
    summary = p.tab("Figure2a_cloud_summary.csv").set_index("category")
    aggregate = p.tab("Figure2a_aggregate_summary.csv").set_index("category")
    headings = ("New conditions", "New partners", "Another reaction")
    definitions = ("median of 30 groups", "median of 24 groups", "weighted mean; reliability\ncorrected; 58 comparisons")
    for i, (category, heading, definition) in enumerate(zip(("conditions", "partners", "reactions"), headings, definitions)):
        left = .085 + i * .315
        ax = fig.add_axes([left, .830, .255, .108])
        group = points.loc[points.category.eq(category)]
        ax.scatter(group.x_z, group.y_z, s=4.8, color=S.PURPLE, lw=0, alpha=.25, rasterized=False)
        ax.axhline(0, color=S.LIGHT, lw=.5, zorder=0)
        ax.axvline(0, color=S.LIGHT, lw=.5, zorder=0)
        ax.set(xlim=(-6, 6), ylim=(-6, 6), xticks=[-4, 0, 4], yticks=[-4, 0, 4])
        if i == 0:
            ax.set_ylabel("Contribution 2 (z)", labelpad=2)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Contribution 1 (z)", labelpad=1)
        S.clean(ax)
        _text(fig, left + .1275, .951, heading, ha="center", weight="bold")
        actual = summary.loc[category]
        ax.text(.04, .94, f"\u03c1 = {actual.rho_pooled:.2f}\nn = {int(actual.n_points):,}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8)
        _text(fig, left + .1275, .775, f"Across groups: \u03c1 = {aggregate.loc[category, 'estimate']:.2f}",
              ha="center", weight="bold")
        _text(fig, left + .1275, .760, definition, ha="center", va="top")


def _all_groups(fig, p):
    frame = p.tab("Figure2b_portability_by_set.csv")
    frame = frame.loc[frame.rho_P.notna()].sort_values("rho_P", ascending=False, kind="stable").reset_index(drop=True)
    if len(frame) != 24:
        raise ValueError("Figure 2b requires the 24 estimable reaction-side groups")
    ax = fig.add_axes([.255, .253, .215, .402])
    labels = []
    positions = np.arange(len(frame), dtype=float)
    for index in frame.index[frame.reaction.eq("SNAr")]:
        positions[index:] += .22
        positions[index + 1:] += .22
    partitions = p.tab("Figure2b_SNAr_partitions.csv")
    extra = partitions.loc[partitions.agreement.eq("partner halves") & ~partitions.partition.eq("fixed"), "rank_agreement"]
    for i, row in frame.iterrows():
        y = positions[i]
        label = f"{S.reaction_name(row.reaction)} · {'E' if row.side == 'elec' else 'N'}"
        if row.reaction == "SNAr" and row.side == "elec":
            label += f"  {row.rho_C:.2f} / {extra.median():.2f}"
            ax.axhspan(y - .47, y + .47, color=S.LIGHT, lw=0, zorder=0)
        labels.append(label)
        estimates = [value for value in (row.rho_C, row.rho_P, row.between_mean) if np.isfinite(value)]
        ax.plot([min(estimates), max(estimates)], [y, y], color=S.GREY, alpha=.5, lw=.5, zorder=1)
        for value, marker in ((row.rho_C, "D"), (row.rho_P, "o"), (row.between_mean, "^")):
            if np.isfinite(value):
                ax.plot(value, y, marker, ms=3.4, mfc=S.WHITE if marker == "o" else S.PURPLE,
                        mec=S.PURPLE, mew=.65, zorder=3)
    ax.axvline(0, color=S.GREY, lw=.5, zorder=0)
    ax.set(yticks=positions, yticklabels=labels, ylim=(positions[-1] + .5, -.5),
           xlim=(-.15, 1.05), xticks=[0, .5, 1], xlabel="Rank agreement, \u03c1")
    ax.tick_params(axis="y", length=0, pad=3)
    S.clean(ax)
    for tick, row in zip(ax.get_yticklabels(), frame.itertuples()):
        tick.set_verticalalignment("center")
        if row.reaction == "SNAr" and row.side == "elec":
            tick.set_weight("bold")
    handles = [Line2D([], [], marker=m, ls="", ms=3.5, mfc=S.WHITE if m == "o" else S.PURPLE,
                      mec=S.PURPLE, color=S.PURPLE, label=label)
               for m, label in (("D", "Conditions"), ("o", "Partners"), ("^", "Reaction"))]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.25, .670), ncol=3,
               handletextpad=.15, columnspacing=.6, frameon=False, borderpad=0, fontsize=8)


def _amines(fig, p):
    S.scheme(fig, [.51, .640, .475, .066], "Buchwald-Hartwig")
    S.scheme(fig, [.51, .572, .475, .066], "SNAr")
    _text(fig, .765, .690, "Pd, ligand, base", ha="center")
    _text(fig, .765, .622, "Base, heat", ha="center")
    frame = p.tab("Figure2c_BH_SNAr_amines.csv")
    aromatic = _true(frame.primary_arylamine)
    ax = fig.add_axes([.575, .424, .395, .130])
    for mask, color in ((~aromatic, S.GREY), (aromatic, S.PURPLE)):
        ax.scatter(frame.loc[mask, "x_contribution_pp"], frame.loc[mask, "y_contribution_pp"],
                   s=10, color=color, lw=0, alpha=.8, zorder=2)
    ax.axhline(0, color=S.LIGHT, lw=.5, zorder=0)
    ax.axvline(0, color=S.LIGHT, lw=.5, zorder=0)
    ax.set(xlim=(-30, 34), ylim=(-18, 23), xticks=[-20, 0, 20], yticks=[-10, 0, 10, 20],
           xlabel="Buchwald–Hartwig contribution (pp)", ylabel="SNAr contribution (pp)")
    ax.xaxis.label.set_bbox(dict(facecolor=S.WHITE, edgecolor="none", pad=1))
    S.clean(ax)
    _text(fig, .59, .545, f"{len(frame)} amines; \u03c1 = {S.signed(frame.rho_obs.iloc[0], 2)}", ha="left", va="top")
    handles = [Line2D([], [], marker="o", ls="", ms=3.2, color=color, label=label)
               for color, label in ((S.PURPLE, "Primary (hetero)arylamines"),
                                    (S.GREY, "Other N–H nucleophiles"))]
    legend = ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, 1.13),
                       handlelength=.6, handletextpad=.4, labelspacing=.5, frameon=False,
                       borderpad=0, borderaxespad=0, fontsize=8)
    legend.set_gid("r12-Figure2c-colour-legend")
    gap = p.tab("Figure2c_selection_summary.csv").target_top_ten_shortfall_pp.iloc[0]
    _text(fig, .52, .367, f"Buchwald–Hartwig's top ten trail by {gap:.1f} pp in SNAr", weight="bold",
          bbox=dict(facecolor=S.WHITE, edgecolor="none", pad=1))
    drawn = frame.loc[frame.drawn.fillna(0).gt(0)].sort_values("drawn")
    if len(drawn) != 4:
        raise ValueError("Figure 2c requires four rule-selected structures")
    for i, row in enumerate(drawn.itertuples()):
        left, bottom = .52 + (i % 2) * .235, .261 - (i // 2) * .110
        rect = [left, bottom, .22, .095]
        orient = ORIENT.get(row.mol, 0)
        # Matching numbers link structures to short callouts inside the data axes.
        S.native(fig, rect, row.mol, orient)
        offsets = {1: (3, -5), 2: (3, 2), 3: (-14, 8), 4: (-3, -20)}
        label = ax.annotate(str(int(row.drawn)), (row.x_contribution_pp, row.y_contribution_pp),
                            xytext=offsets[int(row.drawn)], textcoords="offset points", fontsize=8, weight="bold",
                            gid=f"r10-Figure2c-example-{int(row.drawn)}",
                            arrowprops=dict(arrowstyle="-", lw=.5, color=S.GREY, shrinkA=3, shrinkB=3.5))
        label.arrow_patch.set_gid(f"r10-Figure2c-leader-{int(row.drawn)}")
        _text(fig, left + .11, bottom - .005,
              f"{int(row.drawn)}  {S.signed(row.x_contribution_pp)} / {S.signed(row.y_contribution_pp)} pp",
              ha="center", va="top")


def _boronic(fig, p):
    frame = p.tab("Figure2d_boronic_molecules.csv")
    summary = p.tab("Figure2d_boronic_summary.csv").iloc[0]
    ax = fig.add_axes([.100, .060, .270, .097], gid="r7-Figure2d-scatter-axes")
    _r7_boundary(fig, "r7-Figure2d-plot", [.100, .060, .270, .097])
    flag = _true(frame.five_membered)
    for mask, color in ((~flag, S.GREY), (flag, S.PURPLE)):
        ax.scatter(frame.loc[mask, "suzuki_pp"], frame.loc[mask, "chanlam_pp"], s=9, color=color, lw=0, alpha=.8)
    ax.set(xlim=(-30, 50), ylim=(-10, 10), xticks=[-20, 20], yticks=[-5, 5],
           xlabel="Suzuki (pp)", ylabel="Chan–Lam (pp)")
    ax.xaxis.labelpad = 1
    ax.yaxis.labelpad = 1
    S.clean(ax)
    _text(fig, .100, .166, f"{len(frame)} shared; \u03c1 = {summary.rho_obs:.2f}", ha="left")
    handles = [Line2D([], [], marker="o", ls="", ms=3.5, color=S.PURPLE,
                      label="Five-membered heteroaryl boronic acids")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(.015, .99),
              handletextpad=.35, frameon=False, borderpad=0, borderaxespad=0, fontsize=8)
    native = S.native(fig, [.374, .071, .100, .070], "OB(O)c1ccsc1")
    native.set_gid("r7-Figure2d-boronic-native")
    first = _text(fig, .424, .060, f"{S.signed(summary.suzuki_five_minus_other_pp)} / ",
                  gid="r7-Figure2d-pp-suzuki")
    second = _text(fig, .424, .060, f"{S.signed(summary.chanlam_five_minus_other_pp, 1)} pp",
                   gid="r7-Figure2d-pp-chanlam")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    first_width = first.get_window_extent(renderer).width / fig.bbox.width
    second_width = second.get_window_extent(renderer).width / fig.bbox.width
    first.set_x(.424 - (first_width + second_width) / 2)
    second.set_x(first.get_position()[0] + first_width)
    fig._r7_boronic_source = frame[["SMILES", "suzuki_pp", "chanlam_pp", "five_membered"]].to_dict("records")
    fig._r7_boronic_contrasts = {
        "suzuki_five_minus_other_pp": float(summary.suzuki_five_minus_other_pp),
        "chanlam_five_minus_other_pp": float(summary.chanlam_five_minus_other_pp),
    }


def render(p):
    """Return the complete Figure 2; the caller owns export and QA."""
    S.style()
    fig = plt.figure(figsize=(S.WIDTH_IN, 200 / 25.4))
    _clouds(fig, p)
    _all_groups(fig, p)
    _amines(fig, p)
    _boronic(fig, p)
    for letter, rect in {
        "a": [.015, .745, .970, .250], "b": [.015, .215, .480, .515],
        "c": [.505, .130, .480, .600], "d": [.015, .000, .485, .205],
    }.items():
        _r7_boundary(fig, f"r7-Figure2-panel-{letter}", rect)
    for letter, x, y in (("a", .015, .98), ("b", .015, .698), ("c", .505, .715), ("d", .015, .190)):
        panel_letter, title = S.header(fig, "Figure2", letter, x, y, TITLES)
        if letter == "b":
            title.set_wrap(True)
            title._get_wrap_line_width = lambda: .37 * fig.bbox.width
            panel_letter.set_position((x, y + .017))
    return fig


def geometry_metadata(fig, svg_path=None):
    """Measure rendered artists; the final verifier independently reads SVG/PDF."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    objects = {artist.get_gid(): artist for artist in fig.findobj() if artist.get_gid()}
    panels = {letter: _r7_bounds(fig, objects[f"r7-Figure2-panel-{letter}"].get_window_extent(renderer))
              for letter in "abcd"}
    ax = objects["r7-Figure2d-scatter-axes"]
    scatter = _r7_bounds(fig, ax.get_window_extent(renderer))
    native = objects["r7-Figure2d-boronic-native"]
    pieces = [artist.get_window_extent(renderer) for artist in native.get_children()
              if artist.get_gid() == "chemdraw" and artist.get_visible()]
    native_bounds = _r7_bounds(fig, Bbox.union(pieces))
    contrasts = {key: _r7_bounds(fig, objects[f"r7-Figure2d-pp-{key}"].get_window_extent(renderer))
                 for key in ("suzuki", "chanlam")}
    legend = _r7_bounds(fig, ax.get_legend().get_window_extent(renderer))
    markers = []
    for collection in ax.collections:
        for center in collection.get_offset_transform().transform(collection.get_offsets()):
            markers.append([float(center[0] * 72 / fig.dpi),
                            float((fig.bbox.height - center[1]) * 72 / fig.dpi)])
    source = fig._r7_boronic_source
    source_order = [row for flag in (False, True) for row in source
                    if str(row["five_membered"]).lower() == str(flag).lower()]
    assert len(markers) == len(source_order) == 21, "Figure2d requires 21 rendered markers"
    measured = [{**row, "center_pt": center, "radius_pt": 1.5}
                for row, center in zip(source_order, markers)]
    purple = [row for row in measured if str(row["five_membered"]).lower() == "true"]
    nearest = min(_r7_clearance(legend, row["center_pt"], 1.5) for row in purple)
    assert _r7_contains(panels["d"], native_bounds), "Figure2 native structure outside panel d"
    assert all(_r7_contains(panels["d"], rect) for rect in contrasts.values()), "Figure2 contrast outside panel d"
    assert nearest <= 21.6, "Figure2 legend exceeds 0.3 in from purple data"
    for name, rect in {"native": native_bounds, "legend": legend, **contrasts}.items():
        assert all(_r7_clearance(rect, center, 1.5) > 0 for center in markers), f"Figure2 {name} overlaps data"
    assert min(_r7_clearance(native_bounds, center, 1.5) for center in markers) <= 21.6, "Figure2 native structure exceeds 0.3 in from data"
    return {
        "coordinate_system": "PDF/SVG points, top-left origin, x right, y down",
        "page_size_pt": [float(fig.get_figwidth() * 72), float(fig.get_figheight() * 72)],
        "svg_sha256": hashlib.sha256(Path(svg_path).read_bytes()).hexdigest() if svg_path else None,
        "panels_pt": panels, "panel_d_pt": panels["d"], "scatter_d_pt": scatter,
        "boronic_structure_pt": native_bounds, "contrast_text_bounds_pt": contrasts,
        "legend_pt": legend, "legend_nearest_purple_edge_pt": nearest,
        "native_nearest_marker_edge_pt": min(_r7_clearance(native_bounds, center, 1.5) for center in markers),
        "data_markers_pt": measured,
        "source_anchors": {"native_smiles": "OB(O)c1ccsc1", **fig._r7_boronic_contrasts},
    }


figure2 = render
