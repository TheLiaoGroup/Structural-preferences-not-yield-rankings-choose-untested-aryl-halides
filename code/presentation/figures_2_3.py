"""Figures 2 and 3 of the Article, drawn from the Source Data (article_sources.py) only.
Style, names, colours and structure drawing come from plot_figures.py. Before saving, every plotted
row is checked against its rendered tick label, and visible text boxes are checked for overlaps."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import plot_figures as BF  # noqa: E402  shared style, names, colours and structure drawing

SRC = ROOT / "output/presentation/source_data"
OUT = ROOT / "output/presentation/figures"
NAMES, SHORT = BF.NAMES, BF.SHORT
BLUE, ORANGE, TEAL, RED, PURPLE = BF.BLUE, BF.ORANGE, BF.TEAL, BF.RED, BF.PURPLE
INK, GREY, LIGHT, MUTED = BF.INK, BF.GREY, BF.LIGHT, BF.MUTED
RULE = "#ABB3B9"
# Orientation of every drawn structure (BF.upright): rings with vertical side edges, then n x 60 degrees.
ORIENT = {"Brc1ccccn1": 0, "Brc1cccc2[nH]ncc12": 3,
          "CC(=O)c1cccc(N)c1": 3, "CCc1cccc(N)c1": 3, "CC1COCCN1": 0, "CC(C)Nc1ccccc1": 3,
          "Brc1cccc2cnccc12": 5, "Brc1ccc2cnccc2c1": 5, "COC(=O)c1ccnc(Br)c1": 3}
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


ROWS = []   # (axes carrying the row labels, y, expected label); read back after drawing


def locant(role):
    """A role as the figure shows it: the heteroatom locant of O-nucleophile in italic, as in the text."""
    return role.replace("O-nucleophile", "$\\mathit{O}$-nucleophile")


def register(ax, y, label):
    ROWS.append((ax, float(y), label))


def check_rows(fig):
    """Every plotted row must sit on a tick whose rendered label names what is drawn there."""
    mine = [(ax, y, lab) for ax, y, lab in ROWS if ax.figure is fig]
    if not mine:
        raise RuntimeError("no plotted rows registered")
    bad = []
    for ax, y, lab in mine:
        ticks = list(ax.get_yticks())
        texts = [x.get_text() for x in ax.get_yticklabels()]
        j = int(np.argmin([abs(v - y) for v in ticks]))
        if abs(ticks[j] - y) > 1e-9 or texts[j] != lab:
            bad.append((lab, y, texts[j]))
    return len(mine), bad


def tab(name):
    frame = pd.read_csv(SRC / name)
    if len(frame) == 0:
        raise RuntimeError(f"empty source table {name}")
    return frame


def header(fig, x, y, letter, title):
    """Panel letter and bold title on one baseline, in figure coordinates."""
    a, b = BF.figure_header(fig, x, y, letter, title)
    if "\n" in title:
        bx, by = b.get_position()
        b.set_position((bx, by - title.count("\n") * b.get_fontsize() * 1.2 / (fig.get_figheight() * 72)))
    return a, b


def save(fig, name):
    BF.export_figure(fig, OUT / name, creator="figures_2_3.py")


def overlaps(fig):
    """Pairs of visible text boxes that overlap by more than a pixel in the rendered figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    hidden = set()                      # tick labels whose tick lies outside the view are never drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update({id(tick.label1), id(tick.label2)})
    boxes = [(t.get_text(), t.get_window_extent(r), t.get_gid() == "scheme") for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    bad = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][2] and boxes[j][2]:
                continue  # glyph runs of a drawn reaction scheme sit where the drawing puts them
            a, b = boxes[i][1], boxes[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 1 and min(a.y1, b.y1) - max(a.y0, b.y0) > 1:
                bad.append((boxes[i][0], boxes[j][0]))
    return bad


def rows_axis(ax, labels, colors=None, size=6.8):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=size)
    if colors:
        for t, c in zip(ax.get_yticklabels(), colors):
            t.set_color(c)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


# --- Figure 2: what a reaction context changes ------------------------------------------------

def figure2():
    cells = tab("Figure2a_portability_by_set.csv")
    est = cells.loc[cells.estimable.astype(str).str.lower().eq("true")].sort_values("rho_P").reset_index(drop=True)
    assert len(est) > 0
    H, D = 7.15, 1.0
    Y = lambda v: (v * H + D) / (H + D)
    S = lambda v: v * H / (H + D)
    rect = lambda x, y, w, h: [x, Y(y), w, S(h)]
    fig = plt.figure(figsize=(7.0, H + D))

    # a: every set, three changes
    ax = fig.add_axes(rect(.205, .06, .27, .835))
    for i, r in est.iterrows():
        if r.reaction == "SNAr" and r.side == "elec":
            ax.axhspan(i - .5, i + .5, color=LIGHT, zorder=0, lw=0)
        values = [v for v in (r.rho_C, r.rho_P, r.between_mean) if np.isfinite(v)]
        ax.plot([min(values), max(values)], [i, i], color="#D3D8DC", lw=1.0, zorder=1)
        ax.plot(r.rho_C, i, "D", ms=3.6, color=TEAL, zorder=3)
        register(ax, i, f"{SHORT[r.reaction]} · {locant(r.role)}")
        ax.plot(r.rho_P, i, "o", ms=4.4, mfc=PURPLE if str(r.interval_above_zero).lower() == "true" else "white", mec=PURPLE, mew=1, zorder=3)
        if np.isfinite(r.between_mean):
            ax.plot(r.between_mean, i, "^", ms=4.4, color=RED, zorder=2)
    rows_axis(ax, [f"{SHORT[r.reaction]} · {locant(r.role)}" for _, r in est.iterrows()], size=6.6)
    ax.axvline(0, color=RULE, lw=.7, zorder=0)
    ax.set(xlim=(-.12, 1.03), ylim=(-.7, len(est) + 1.9), xticks=[0, .5, 1], xlabel="Rank agreement of a molecule's contribution")
    BF.clean(ax)
    ax.legend(handles=[Line2D([0], [0], marker="D", ls="", color=TEAL, ms=3.6, label="new conditions"),
                       Line2D([0], [0], marker="o", ls="", color=PURPLE, ms=4.4, label="new partners"),
                       Line2D([0], [0], marker="^", ls="", color=RED, ms=4.4, label="another reaction")],
              loc="upper left", ncol=1, frameon=False, fontsize=6.4, handletextpad=.1, borderaxespad=.2)
    header(fig, .005, Y(.925), "a", "Every reaction and partner")

    # b: SNAr, condition halves against partner partitions
    bt = tab("Figure2b_SNAr_partitions.csv")
    cond = bt.loc[bt.agreement.eq("condition halves"), "rank_agreement"].item()
    rel = bt.loc[bt.agreement.eq("partner halves") & bt.partition.eq("fixed"), "rank_agreement"].item()
    sal = bt.loc[bt.agreement.eq("partner halves") & ~bt.partition.eq("fixed"), "rank_agreement"].to_numpy()
    assert len(sal) == 20
    ax = fig.add_axes(rect(.68, .835, .305, .07))
    ax.plot(cond, 1, "D", ms=4.2, color=TEAL, zorder=3)
    register(ax, 1, "condition halves")
    jitter = np.where(np.arange(len(sal)) % 2 == 0, .13, -.13)
    order = np.argsort(sal)
    ax.scatter(sal[order], jitter, s=11, color=PURPLE, lw=0, alpha=.85, zorder=3)
    med = float(np.median(sal))
    ax.plot([med, med], [-.42, .42], color=INK, lw=1.3, zorder=4)
    ax.plot(rel, 0, "o", ms=5.2, mfc="white", mec=PURPLE, mew=1.2, zorder=5)
    register(ax, 0, "partner halves,\n20 partitions")
    ax.annotate(f"fixed split {rel:.2f}".replace("-", "−"), (rel, 0), xytext=(rel + .03, -.95), fontsize=6.2, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6), ha="left", va="center")
    ax.text(med, .55, f"median {med:.2f}", fontsize=6.2, ha="center", va="bottom", color=INK)
    ax.text(cond, 1.42, f"{cond:.2f}", fontsize=6.2, ha="center", va="bottom", color=TEAL)
    rows_axis(ax, ["partner halves,\n20 partitions", "condition halves"], size=6.4)
    ax.set(xlim=(-.12, 1.03), ylim=(-1.3, 1.9), xticks=[0, .5, 1])
    ax.axvline(0, color=RULE, lw=.7, zorder=0)
    BF.clean(ax)
    header(fig, .555, Y(.925), "b", "S$_\\mathregular{N}$Ar aryl halides: repeatable, partner-sensitive")

    # c: a ring N beside the halogen, with the N-H as the contrast
    st = tab("Figure2c_structure_effects.csv")
    order_rx = st.loc[st.feature.eq("halogen next to ring N") & st.data.eq("first condition set")].set_index("reaction").difference_pp
    reactions = sorted(set(st.reaction), key=lambda r: order_rx.get(r, np.inf))
    metal = st.drop_duplicates("reaction").set_index("reaction").metal
    heads = {"halogen next to ring N": "Adjacent ring N", "N–H": "N–H"}
    axes_c = []
    for k, (feature, (_, smiles)) in enumerate(BF.STRUCTURE_FEATURES.items()):
        ax = fig.add_axes(rect(.745 + k * .13, .430, .115, .190))
        axes_c.append(ax)
        ax.axvline(0, color=RULE, lw=.7, zorder=0)
        for rx, g in st.loc[st.feature.eq(feature)].groupby("reaction"):
            y, c = reactions.index(rx), BLUE if metal[rx] == "Pd" else ORANGE
            register(axes_c[0], y, NAMES[rx])
            ax.plot([g.difference_pp.min(), g.difference_pp.max()], [y, y], color="#D3D8DC", lw=1.0, zorder=1)
            for d, (mk, ms, filled) in BF.STRUCTURE_SETS.items():
                v = g.loc[g.data.eq(d), "difference_pp"]
                if len(v):
                    ax.plot(v.iloc[0], y, mk, ms=ms * .85, mfc=c if filled else "white", mec=c, mew=.9, zorder=3)
        ax.set(xlim=(-18, 19), ylim=(-.7, len(reactions) - .3), xticks=[-10, 0, 10])
        BF.clean(ax)
        if k == 0:
            rows_axis(ax, [NAMES[r] for r in reactions], [BLUE if metal[r] == "Pd" else ORANGE for r in reactions], size=6.4)
        else:
            rows_axis(ax, [""] * len(reactions))
        ax.set_title(heads[feature], loc="left", fontsize=6.8, fontweight="bold", pad=4)
        asset = BF.CD.molecule(smiles, ORIENT.get(smiles, 0))
        w_pt, h_pt = float(asset["width_pt"]), float(asset["height_pt"])
        figure_w, figure_h = fig.get_size_inches() * 72
        center_x, top_y = (.802 + k * .13) * figure_w, Y(.740) * figure_h
        icon = fig.add_axes([(center_x - w_pt / 2 - 1) / figure_w,
                             (top_y - h_pt - 1) / figure_h,
                             (w_pt + 2) / figure_w, (h_pt + 2) / figure_h])
        icon.set_axis_off()
        icon.set(xlim=(0, w_pt + 2), ylim=(0, h_pt + 2))
        anchor = asset.get("anchor_y_pt")
        BF.structure(icon, smiles, w_pt / 2 + 1,
                     h_pt + 1 - (h_pt / 2 if anchor is None else float(anchor)),
                     1.0, ORIENT.get(smiles, 0))
    fig.text(.86, Y(.386), "With the feature minus without (pp)", ha="center", fontsize=7)
    fig.legend(handles=[Line2D([0], [0], marker=mk, ls="", mfc=GREY if filled else "white", mec=GREY, ms=ms * .85,
                               label=lab) for (mk, ms, filled), lab in zip(BF.STRUCTURE_SETS.values(), ("1st", "2nd set"))] +
                       [Line2D([0], [0], marker="s", ls="", color=BLUE, ms=3.6, label="Pd"),
                        Line2D([0], [0], marker="s", ls="", color=ORANGE, ms=3.6, label="other")],
               loc="center", bbox_to_anchor=(.78, Y(.360)), ncol=4, frameon=False, fontsize=6.0, handletextpad=.15, columnspacing=.7)
    header(fig, .555, Y(.765), "c", "One motif, opposite effects")

    # d: Buchwald-Hartwig against SNAr amines
    neg = tab("Figure2d_BH_SNAr_amines.csv")
    paa = neg.primary_arylamine.astype(str).str.lower().eq("true")
    ax = fig.add_axes(rect(.665, .145, .32, .165))
    ax.axhline(0, color="#CED4D8", lw=.7); ax.axvline(0, color="#CED4D8", lw=.7)
    ax.scatter(neg.x_contribution_pp[~paa], neg.y_contribution_pp[~paa], s=12, color=GREY, ec="white", lw=.3)
    ax.scatter(neg.x_contribution_pp[paa], neg.y_contribution_pp[paa], s=15, color=RED, ec="white", lw=.3)
    for _, r in neg.loc[neg.drawn.gt(0)].iterrows():
        number = int(r.drawn)
        position = {3: (-24, 12), 4: (-19, 10)}.get(number, (3, 2))
        free_label = number in (3, 4)
        ax.annotate(str(number), (r.x_contribution_pp, r.y_contribution_pp), xytext=position,
                    textcoords="data" if free_label else "offset points",
                    ha="center" if free_label else "left", va="center" if free_label else "baseline",
                    arrowprops={"arrowstyle": "-", "lw": .5, "color": GREY} if free_label else None,
                    fontsize=6.5, fontweight="bold", color=RED if str(r.primary_arylamine).lower() == "true" else GREY)
    ax.set(xlabel="Buchwald–Hartwig (pp)", ylabel="S$_\\mathregular{N}$Ar (pp)")
    ax.margins(.1)
    BF.clean(ax)
    ax.text(.02, .98, f"{len(neg)} shared N–H nucleophiles; {int(paa.sum())} arylamines (red)\nrank agreement {neg.rho_obs.iloc[0]:.2f}".replace("-", "−"),
            transform=ax.transAxes, va="top", fontsize=6.0)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + .3 * (hi - lo))
    header(fig, .555, Y(.335), "d", "Arylamines invert between B–H and S$_\\mathregular{N}$Ar")
    strip = [{"smiles": r.mol, "orient": ORIENT.get(r.mol, 0), "number": int(r.drawn), "color": RED if str(r.primary_arylamine).lower() == "true" else GREY,
              "caption": f"{BF.signed(r.x_contribution_pp)} / {BF.signed(r.y_contribution_pp)}"}
             for _, r in neg.loc[neg.drawn.gt(0)].sort_values("drawn").iterrows()]
    # The original plots keep their physical sizes above a full-width native-size structure row.
    BF.structure_strip(fig, [.04, .01, .92, .85 / (H + D)], strip, centre=.55)
    fig.text(.04, .01 + (.85 + 1 / 72) / (H + D), "d  Numbered nucleophiles (B–H / SNAr, pp)",
             fontsize=7, color=MUTED)
    return fig


# --- Figure 3: what can persist -----------------------------------------------------------------

def figure3():
    q = tab("Figure3a_campaign_recognition.csv").sort_values("profile_similarity", ascending=False).reset_index(drop=True)
    assert len(q) == 9 and q.analysed_reaction.iloc[0] == "Buchwald-Hartwig"
    H, D = 7.8, 2.5   # inches: panels a-c keep their 5.3-inch layout above a 2.5-inch row for d
    Y = lambda v: (v * 5.3 + D) / H
    S = lambda v: v * 5.3 / H
    fig = plt.figure(figsize=(7.0, H))
    y = np.arange(len(q))[::-1]
    bh = int(np.flatnonzero(q.analysed_reaction.eq("Buchwald-Hartwig"))[0])
    elig = q.ranking_eligible.astype(str).str.lower().eq("true")
    for k, (col, title, color, mk) in enumerate((
            ("aryl_halide_rank_agreement", "Ranking of the shared aryl halides", PURPLE, "o"),
            ("profile_similarity", "Structural profile", BLUE, "D"))):
        ax = fig.add_axes([.205 + k * .43, Y(.575), .33, S(.3)])
        if k == 0:
            ax0 = ax
        ax.axhspan(y[bh] - .5, y[bh] + .5, color=LIGHT, zorder=0, lw=0)
        ax.axvline(0, color=RULE, lw=.7, zorder=0)
        for i, r in q.iterrows():
            filled = (k == 1) or bool(elig[i])
            ax.plot(r[col], y[i], mk, ms=5 if mk == "o" else 4.4, mfc=color if filled else "white", mec=color, mew=1.1, zorder=3)
            register(ax0, y[i], NAMES[r.analysed_reaction])
            if k == 0:
                ax.text(1.06, y[i], f"{int(r.shared_aryl_halides)}", fontsize=6.0, va="center", ha="left", color=MUTED,
                        transform=ax.get_yaxis_transform())
        ax.set(xlim=(-1.04, 1.04), ylim=(-1.5, len(q) - .4), xticks=[-1, -.5, 0, .5, 1])
        ax.spines["bottom"].set_bounds(-1, 1)
        BF.clean(ax)
        # row j of the axis holds the reaction plotted at y == j
        at = {int(y[i]): q.analysed_reaction[i] for i in range(len(q))}
        labels = [at[j] for j in range(len(q))]
        assert at[len(q) - 1] == "Buchwald-Hartwig" and all(at[int(y[i])] == q.analysed_reaction[i] for i in range(len(q)))
        rows_axis(ax, [NAMES[r] for r in labels] if k == 0 else [""] * len(q), size=6.8)
        ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=4)
        if k == 0:
            ax.set_xlabel("Rank agreement (Spearman)")
            rank = int(q.loc[bh, "ranking_rank_among_eligible"])
            ax.text(.02, .02, f"B–H: rank {rank} of {int(elig.sum())} (filled: ≥10 shared)", transform=ax.transAxes, fontsize=6.2, color=INK, va="bottom",
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=4)
            ax.text(1.06, len(q) - .45, "shared", fontsize=6.0, ha="left", va="bottom", color=MUTED, transform=ax.get_yaxis_transform())
        else:
            ax.set_xlabel("Profile similarity (Pearson)")
            ax.text(.02, .02, f"B–H: 1st of {len(q)}", transform=ax.transAxes, fontsize=6.2, color=INK, va="bottom")
    header(fig, .005, Y(.925), "a", "A second Buchwald–Hartwig campaign: closest to its own reaction by structure, not by ranking")

    # b and c share one reaction order
    b = tab("Figure3b_partner_change.csv")
    c = tab("Figure3c_random_grouping.csv")
    first = b[b.condition_set.eq("first")].set_index("reaction")
    order = list((first.r_structure - first.rho_molecule).sort_values().index)
    assert len(order) == 7
    yb = {r: i for i, r in enumerate(order)}
    ax = fig.add_axes([.205, Y(.085), .3, S(.34)])
    ax.axvline(0, color=RULE, lw=.7, zorder=0)
    for (rx, cs), g in b.groupby(["reaction", "condition_set"]):
        r = g.iloc[0]
        yy = yb[rx] + (.17 if cs == "first" else -.17)
        filled = cs == "first"
        ax.plot([r.rho_molecule, r.r_structure], [yy, yy], color="#D3D8DC", lw=1.0, zorder=1)
        ax.plot(r.rho_molecule, yy, "o", ms=4.4, mfc=PURPLE if filled else "white", mec=PURPLE, mew=1, zorder=3)
        register(ax, yb[rx], NAMES[rx])
        ax.plot(r.r_structure, yy, "D", ms=3.9, mfc=BLUE if filled else "white", mec=BLUE, mew=1, zorder=3)
    rows_axis(ax, [NAMES[r] for r in order], size=6.8)
    ax.set(xlim=(-1.0, 1.08), ylim=(-.6, len(order) + .9), xticks=[-1, -.5, 0, .5, 1], xlabel="Agreement between partner halves")
    ax.spines["bottom"].set_bounds(-1, 1)   # room for markers at the right end; the axis line stops at 1
    BF.clean(ax)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=PURPLE, ms=4.4, label="aryl-halide ranking"),
                       Line2D([0], [0], marker="D", ls="", color=BLUE, ms=3.9, label="structural profile"),
                       Line2D([0], [0], marker="o", ls="", mfc=GREY, mec=GREY, ms=4.2, label="1st condition set"),
                       Line2D([0], [0], marker="o", ls="", mfc="white", mec=GREY, ms=4.2, label="2nd")],
              loc="upper left", ncol=2, frameon=False, fontsize=6.0, handletextpad=.1, columnspacing=.8, borderaxespad=.1)
    header(fig, .005, Y(.465), "b", "New partners: the profile usually holds better")

    ax = fig.add_axes([.68, Y(.085), .305, S(.34)])
    ax.axvline(0, color=RULE, lw=.7, zorder=0)
    cf = c[c.condition_set.eq("first")].set_index("reaction")
    for rx in order:
        r, yy = cf.loc[rx], yb[rx]
        ax.plot([r["null_2.5"], r["null_97.5"]], [yy, yy], color="#C9CFD4", lw=3.2, solid_capstyle="butt", zorder=1)
        ax.plot([r.null_median] * 2, [yy - .22, yy + .22], color=MUTED, lw=1.1, zorder=2)
        above = bool(str(r["above_null_97.5"]).lower() == "true")
        ax.plot(r.r_structure, yy, "D", ms=4.4, mfc=BLUE if above else "white", mec=BLUE, mew=1.1, zorder=3)
        register(ax, yy, NAMES[rx])
    rows_axis(ax, [NAMES[r] for r in order], size=6.8)
    ax.set(xlim=(-1.0, 1.08), ylim=(-.6, len(order) + .9), xticks=[-1, -.5, 0, .5, 1], xlabel="Agreement of structural profiles")
    ax.spines["bottom"].set_bounds(-1, 1)
    BF.clean(ax)
    ax.legend(handles=[Line2D([0], [0], color="#C9CFD4", lw=3.2, label="random groups, 95%"),
                       Line2D([0], [0], color=MUTED, lw=1.1, label="median"),
                       Line2D([0], [0], marker="D", ls="", mfc=BLUE, mec=BLUE, ms=4.2, label="real classes, above 97.5%"),
                       Line2D([0], [0], marker="D", ls="", mfc="white", mec=BLUE, ms=4.2, label="within")],
              loc="upper left", ncol=2, frameon=False, fontsize=6.0, handlelength=1.6, handletextpad=.6, columnspacing=.8, borderaxespad=.1)
    header(fig, .555, Y(.465), "c", "Structure classes against\nrandom groups of the same size")

    # d: aryl halides that the first campaign never ran, chosen by each analysed reaction's structural profile
    dd = tab("Figure3d_selection_without_record.csv").sort_values("rho_unseen", ascending=False).reset_index(drop=True)
    assert len(dd) == 9 and dd.source.iloc[0] == "Buchwald-Hartwig" and dd.molecules.nunique() == 1
    avail, n_new = float(dd.gain_available_pp.iloc[0]), int(dd.molecules.iloc[0])
    assert np.allclose(dd.gain_available_pp, avail)
    yd = np.arange(len(dd))[::-1]
    at = {int(yd[i]): dd.source[i] for i in range(len(dd))}
    for k, (col, title, xlim, xticks, xlabel) in enumerate((
            ("rho_unseen", f"Ranking the {n_new} aryl halides", (-1.0, 1.0), [-1, -.5, 0, .5, 1], "Rank agreement (Spearman)"),
            ("top_ten_gain_pp", "The ten ranked highest", (-1.0, 12.0), [0, 2, 4, 6, 8, 10], f"Above the mean of the {n_new} (pp)"))):
        ax = fig.add_axes([.205 + k * .43, .07, .33, .195])
        if k == 0:
            axd = ax
        ax.axhspan(yd[0] - .5, yd[0] + .5, color=LIGHT, zorder=0, lw=0)
        ax.axvline(0, color=RULE, lw=.7, zorder=0)
        if k == 1:
            ax.axvline(avail, color=MUTED, lw=.8, ls=(0, (3, 2)), zorder=1)
            ax.text(avail - .3, 0, f"available {avail:.1f}", fontsize=6.0, color=MUTED, ha="right", va="center")
        for i, r in dd.iterrows():
            ax.plot(r[col], yd[i], "D", ms=4.4, mfc=BLUE, mec=BLUE, mew=1.1, zorder=3)
            register(axd, yd[i], NAMES[r.source])
        ax.set(xlim=xlim, ylim=(-.6, len(dd) - .4), xticks=xticks, xlabel=xlabel)
        BF.clean(ax)
        rows_axis(ax, [NAMES[at[j]] for j in range(len(dd))] if k == 0 else [""] * len(dd), size=6.8)
        ax.set_title(title, loc="left", fontsize=7.2, fontweight="bold", pad=4)
    header(fig, .005, .305, "d", "Choosing aryl halides that have no record")
    return fig




def main():
    BF.style()
    for name, make in (("Figure2", figure2), ("Figure3", figure3)):
        fig = make()
        BF.finalize(fig)   # text overlaps are checked on the finished layout
        bad = overlaps(fig)
        n_rows, wrong = check_rows(fig)
        if wrong:
            raise RuntimeError(f"{name}: row labels do not match the plotted rows: {wrong}")
        print(name, f"rows checked: {n_rows}, mismatches: 0")
        save(fig, name)
        plt.close(fig)
        print(name, "text overlaps:", bad if bad else "none")


if __name__ == "__main__":
    main()
