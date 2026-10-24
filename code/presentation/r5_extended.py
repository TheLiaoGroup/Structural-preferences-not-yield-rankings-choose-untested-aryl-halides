"""Final-size R5 Extended Data layouts, retaining the released data producers."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Text

import r5_style as S


def _full(text):
    return (text.replace("B–H", "Buchwald–Hartwig")
            .replace("B-H", "Buchwald–Hartwig")
            .replace("S$_\\mathregular{N}$Ar", "SNAr"))


def _finish(fig, p, name):
    short_names = {_full(value): S.reaction_name(key) for key, value in S.BF.SHORT.items()}
    titles = [ax for ax in fig.axes if ax.get_title(loc="left")]
    expected = p.titles[name]
    if len(titles) != len(expected):
        raise ValueError(f"{name}: expected {len(expected)} panel titles, found {len(titles)}")
    for ax, title in zip(titles, expected):
        ax._left_title.set_text(title)
    for ax in fig.axes:
        if not ax.axison:
            continue
        for dimension in ("x", "y"):
            labels = getattr(ax, f"get_{dimension}ticklabels")()
            old = [label.get_text() for label in labels]
            updated = [short_names.get(_full(value), S.reaction_text(_full(value))) for value in old]
            if updated != old:
                getattr(ax, f"set_{dimension}ticklabels")(updated)
    for text in fig.findobj(Text):
        text.set_text(S.reaction_text(_full(text.get_text())))
    S.finalize(fig)
    return fig


def _fill_legend(fig, anchor, labels, gid, colour=S.MUTED):
    handles = [Line2D([], [], marker="o", ls="", ms=4.5, mec=colour,
                      mfc=colour if filled else S.WHITE, label=label)
               for filled, label in zip((True, False), labels)]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=anchor,
                        ncol=2, handlelength=.8, handletextpad=.4, columnspacing=1,
                        frameon=False, borderpad=0, borderaxespad=0, fontsize=8)
    legend.set_gid(gid)
    return legend


def ed1(p):
    S.style()
    designs = p.tab("ExtendedDataFig1_reaction_designs.csv")
    composition = p.tab("ExtendedDataFig1_condition_composition.csv")
    if len(designs) != 15:
        raise ValueError("Extended Data Figure 1 requires fifteen reaction arrays")
    assets = {row["reaction"]: row for row in S.BF.CD.manifest()["schemes"]}
    heights = [max(35, float(assets[row.reaction]["height_pt"]) + 0.5) for row in designs.itertuples()]
    height, width = 30 + sum(heights), S.WIDTH_IN * 72
    fig = plt.figure(figsize=(S.WIDTH_IN, height / 72))
    S.header(fig, "ExtendedDataFig1", "a", .012, 1 - 10 / height, p.titles)
    fig.text(.012, 1 - 23 / height, "Distinct components: M, metal sources; L, ligands; B, base/acid; O, other reagents; S, solvents.", fontsize=8)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set(xlim=(0, width), ylim=(0, height))
    top = height - 29
    for row, rowheight in zip(designs.itertuples(), heights):
        asset = assets[row.reaction]
        if asset["width_pt"] > 295:
            raise ValueError(f"Native ED1 scheme exceeds its physical width: {row.reaction}")
        S.BF.CD.draw(ax, asset, 4, top - (rowheight - float(asset["height_pt"])) / 2)
        conditions = composition[composition.rxn_class.eq(row.reaction)]
        if len(conditions) != 20 or conditions.temperature_C.nunique() != 1 or conditions.time_h.nunique() != 1:
            raise ValueError(f"ED1 requires a complete twenty-condition composition: {row.reaction}")
        def count(column):
            return len({item.strip() for value in conditions[column].fillna("").astype(str)
                        for item in value.split(";") if item.strip()})
        counts = [count(column) for column in ("metal_sources", "ligands", "bases_acids", "other_reagents", "solvent")]
        title = _full(S.BF.NAMES[row.reaction])
        x = 310
        ax.text(x, top - 1, title, fontsize=9, weight="bold", va="top", gid="key-number")
        ax.text(x, top - 13, f"{int(row.pairs):,} pairs · 20 conditions · {conditions.temperature_C.iloc[0]:g} °C / {conditions.time_h.iloc[0]:g} h",
                fontsize=8, va="top")
        ax.text(x, top - 24, " · ".join(f"{label} {n}" for label, n in zip("MLBOS", counts)), fontsize=8, va="top")
        ax.axhline(top - rowheight, color=S.LIGHT, lw=.5)
        top -= rowheight
    return _finish(fig, p, "ExtendedDataFig1")


def ed2(p):
    S.style()
    fig = p.ed2()
    fig.set_size_inches(S.WIDTH_IN, 300 / 72)
    fig.axes[0].set_position([.13, .15, .69, .72])
    fig.axes[1].set_position([.855, .15, .022, .72])
    record_cmap = LinearSegmentedColormap.from_list("record-pair-yield", [S.GREY, S.WHITE, S.PURPLE])
    for image in fig.axes[0].images:
        image.set_cmap(record_cmap)
    for spine in fig.axes[0].spines.values():
        spine.set_color(S.GREY)
    return _finish(fig, p, "ExtendedDataFig2")


def ed3(p):
    S.style()
    fig = p.ed4()
    fig.set_size_inches(S.WIDTH_IN, 510 / 72)
    rows = p.tab("ExtendedDataFig4b_undefined_benchmark.csv").reset_index(drop=True)
    ax = [axis for axis in fig.axes if axis.axison][1]
    ax.set_position([.53, .12, .44, .30])
    ax.set_yticks(range(len(rows)), [f"{S.reaction_name(row.rxn_a)} / {S.reaction_name(row.rxn_b)}"
                                   f" \u00b7 {'E' if row.side == 'elec' else 'N'} ({int(row.molecules):,})"
                                   for row in rows.itertuples()])
    handles = [Line2D([], [], marker="o", ls="", ms=4, mfc=colour, mec=S.WHITE,
                      mew=.4, label=label)
               for colour, label in ((S.PURPLE, "electrophiles"), (S.GREY, "nucleophiles"))]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.16, .93),
                        ncol=2, handlelength=.8, handletextpad=.4, columnspacing=1,
                        frameon=False, borderpad=0, borderaxespad=0, fontsize=8)
    legend.set_gid("r12-ED3a-colour-legend")
    defined = p.tab("ExtendedDataFig4a_defined_benchmark.csv")
    counts = [int(defined.molecules.min()), int(defined.molecules.max())]
    handles = [Line2D([], [], marker="o", ls="", ms=(10 + n) ** .5, mfc=S.GREY,
                      mec=S.WHITE, mew=.4, label=f"{n:,} shared molecules") for n in counts]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.58, .93),
                        ncol=2, handlelength=1, handletextpad=.4, columnspacing=1,
                        frameon=False, borderpad=0, borderaxespad=0, fontsize=8)
    legend.set_gid("r12-ED3a-size-legend")
    return _finish(fig, p, "ExtendedDataFig3")


def ed4(p):
    S.style()
    fig = p.ed5()
    fig.set_size_inches(S.WIDTH_IN, 515 / 72)
    matrix, colourbar = fig.axes[:2]
    matrix.set_position([.23, .28, .74, .65])
    colourbar.set_position([.31, .058, .54, .012])
    record_cmap = LinearSegmentedColormap.from_list("record-rank-agreement", [S.GREY, S.WHITE, S.PURPLE])
    for image in matrix.images:
        image.set_cmap(record_cmap)
        image.cmap.set_bad(S.LIGHT)
    for patch in matrix.patches:
        patch.set_facecolor(S.GREY)
        patch.set_edgecolor(S.WHITE)
    return _finish(fig, p, "ExtendedDataFig4")


def ed5(p):
    S.style()
    fig = p.ed6()
    fig.set_size_inches(S.WIDTH_IN, 520 / 72)
    axes = [ax for ax in fig.axes if not getattr(ax, "_figure_header_carrier", False)]
    for ax, position in zip(axes, ([.25, .58, .29, .29], [.67, .58, .29, .29],
                                   [.25, .12, .29, .28], [.67, .12, .29, .28])):
        ax.set_position(position)
    for line in axes[1].lines:
        if line.get_marker() == "D":
            line.set_color(S.TEAL)
            line.set_markeredgecolor(S.TEAL)
            if line.get_markerfacecolor() != "white":
                line.set_markerfacecolor(S.TEAL)
    if len(axes[-1].collections) > 1:
        axes[-1].collections[-1].set_facecolor(S.TEAL)
    _fill_legend(fig, (.25, .515), ("≥10 shared molecules", "<10 shared molecules"),
                 "r12-ED5a-shared-count-legend", S.PURPLE)
    return _finish(fig, p, "ExtendedDataFig5")


def ed6(p):
    S.style()
    fig = p.ed7()
    fig.set_size_inches(S.WIDTH_IN, 555 / 72)
    axes = [ax for ax in fig.axes if not getattr(ax, "_figure_header_carrier", False)]
    axes[0].set_position([.25, .57, .70, .30])
    axes[1].set_position([.25, .10, .70, .30])
    for ax in axes:
        for line in ax.lines:
            if line.get_marker() == "D":
                line.set_color(S.TEAL)
                line.set_markeredgecolor(S.TEAL)
                if line.get_markerfacecolor() != "white":
                    line.set_markerfacecolor(S.TEAL)
    axes[0].get_legend().remove()
    handles = [Line2D([], [], marker=marker, ls="", ms=4.5, color=colour, label=label)
               for marker, colour, label in (("o", S.PURPLE, "molecule ranking"),
                                              ("D", S.TEAL, "structural profile"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.25, .925), ncol=2,
               handlelength=.8, handletextpad=.4, columnspacing=1, frameon=False,
               borderpad=0, borderaxespad=0, fontsize=8)
    _fill_legend(fig, (.25, .895), ("First condition set", "Second condition set"),
                 "r12-ED6a-condition-set-legend")
    for line in axes[1].lines:
        if line.get_linewidth() > 3:
            line.set_color(S.LIGHT)
    handles = [
        Line2D([], [], color=S.LIGHT, lw=3.5, label="Random groups: central 95%"),
        Line2D([], [], marker="|", ls="", color=S.MUTED, ms=8, label="Median"),
        Line2D([], [], marker="D", ls="", color=S.TEAL, ms=4.5,
               label=">97.5th percentile"),
    ]
    legend = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.25, .445),
                        ncol=3, handlelength=1.3, handletextpad=.5, columnspacing=1.3,
                        frameon=False, borderpad=0, borderaxespad=0, fontsize=8)
    legend.set_gid("r13-ED6b-null-legend")
    return _finish(fig, p, "ExtendedDataFig6")


def ed7(p):
    S.style()
    fig = p.ed8()
    fig.set_size_inches(S.WIDTH_IN, 590 / 72)
    axes = [ax for ax in fig.axes if not getattr(ax, "_figure_header_carrier", False)]
    axes[0].set_position([.24, .62, .52, .28])
    axes[1].set_position([.31, .10, .65, .39])
    axes[1].get_legend().remove()
    handles = [Line2D([], [], marker="o", ls="", ms=4, color=colour, label=label)
               for colour, label in ((S.PURPLE, "electrophile"), (S.GREY, "nucleophile"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.31, .535), ncol=2,
               handlelength=.8, handletextpad=.4, columnspacing=1, frameon=False,
               borderpad=0, borderaxespad=0, fontsize=8)
    _fill_legend(fig, (.31, .512), ("95% interval > 0", "95% interval not > 0"),
                 "r12-ED7b-interval-legend")
    return _finish(fig, p, "ExtendedDataFig7")


def si1(p):
    S.style()
    fig = p.si1()
    fig.set_size_inches(S.WIDTH_IN, 300 / 72)
    axes = [ax for ax in fig.axes if not getattr(ax, "_figure_header_carrier", False)]
    axes[0].set_position([.22, .22, .25, .65])
    axes[1].set_position([.72, .22, .25, .65])
    for ax in axes:
        ax.set_title("", loc="left")
    for line in axes[1].lines:
        line.set_color(S.PURPLE)
    for collection in axes[1].collections:
        collection.set_color(S.PURPLE)
    for text in fig.findobj(Text):
        if text.get_gid() in ("panel-letter", "panel-title"):
            text.set_visible(False)
    for letter, x in (("a", .025), ("b", .525)):
        S.header(fig, "SupplementaryFig1", letter, x, .95, p.titles)
    legend = axes[0].get_legend()
    if legend:
        legend.set_bbox_to_anchor((.50, -.20))
    return _finish(fig, p, "SupplementaryFig1")


MAKERS = {"ExtendedDataFig1": ed1, "ExtendedDataFig2": ed2, "ExtendedDataFig3": ed3,
          "ExtendedDataFig4": ed4, "ExtendedDataFig5": ed5, "ExtendedDataFig6": ed6,
          "ExtendedDataFig7": ed7, "SupplementaryFig1": si1}


def figures(p):
    return {name: make(p) for name, make in MAKERS.items()}
