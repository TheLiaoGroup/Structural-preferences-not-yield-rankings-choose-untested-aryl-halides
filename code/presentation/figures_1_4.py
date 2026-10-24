"""Figures 1 and 4 and Extended Data Figs. 1 and 2, drawn from the Source Data
(article_sources.py) and condition metadata. Figure 1a draws fifteen native reaction schemes with design counts.
Style, names, colours and structure drawing come from plot_figures.py; panel headers and the
text-overlap check from figures_2_3.py."""
import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "output/presentation/source_data"
OUT = ROOT / "output/presentation/figures"
import plot_figures as BF  # noqa: E402  shared style, names, colours and structure drawing
import figures_2_3 as V1  # noqa: E402  panel headers and the text-overlap check

NAMES, ROLE = BF.NAMES, BF.ROLE
BLUE, ORANGE, TEAL, RED, PURPLE = BF.BLUE, BF.ORANGE, BF.TEAL, BF.RED, BF.PURPLE
INK, GREY, LIGHT, MUTED = BF.INK, BF.GREY, BF.LIGHT, BF.MUTED








def tab(name):
    return pd.read_csv(SRC / name)


def save(fig, name):
    BF.export_figure(fig, OUT / name, creator="figures_1_4.py")


CHANGE = {"Conditions": TEAL, "Partners": PURPLE, "Reaction": RED}   # one colour per change, as in Fig. 1c and Fig. 2a


SCHEMES = {s["reaction"]: s for s in json.loads((ROOT / "code" / "assets" / "Figure1a_schemes.json").read_text(encoding="utf-8"))["schemes"]}
SCHEME_SCALE = 1.0


def scheme_size(s):
    """Physical dimensions of the native ChemDraw scheme."""
    asset = BF.CD.scheme(s["reaction"])
    return float(asset["width_pt"]), float(asset["height_pt"])


def draw_scheme(ax, s, left, top, scale=SCHEME_SCALE):
    """Replay the native ChemDraw scheme at 1:1 points; legacy scale is ignored."""
    return BF.CD.draw(ax, BF.CD.scheme(s["reaction"]), left, top)


def figure1():
    t = tab("Figure1a_reactions.csv")
    assert set(t.reaction) == set(SCHEMES), set(t.reaction) ^ set(SCHEMES)
    W = 7.0
    LOW = 100.0
    rows = [list(t.reaction.iloc[i:i + 4]) for i in range(0, len(t), 4)]
    band = [max(scheme_size(SCHEMES[r])[1] for r in row) * SCHEME_SCALE for row in rows]
    NAME, GAP, COUNT, PAD = 7.0, 1.0, 13.0, 2.0
    tile = [PAD + NAME + GAP + b + GAP + COUNT + PAD for b in band]
    a_h = sum(tile) + 2.0 * (len(rows) - 1)
    H = (LOW + a_h + 17.0) / 72
    fig = plt.figure(figsize=(W, H))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def header_width(rx):
        name = renderer.get_text_width_height_descent(NAMES[rx], FontProperties(family="Arial", size=7), "$" in NAMES[rx])[0]
        shared = renderer.get_text_width_height_descent(f"{int(t.loc[t.reaction.eq(rx), 'shared_molecules'].iloc[0]):,} shared",
                                                      FontProperties(family="Arial", size=6), False)[0]
        return (name + shared) * 72 / fig.dpi + 14

    lower = LOW / (H * 72)
    ax = fig.add_axes([0, LOW / 72 / H, 1, a_h / 72 / H]); ax.set_axis_off(); ax.set(xlim=(0, W * 72), ylim=(0, a_h))
    y, rec = a_h, {r.reaction: r for r in t.itertuples()}
    for row, b, h in zip(rows, band, tile):
        widths = [max(110.0, scheme_size(SCHEMES[rx])[0] + 10.0, header_width(rx)) for rx in row]
        available = W * 72 - 5.0 - 3.0 * (len(row) - 1)
        if sum(widths) > available:
            raise ValueError(f"Native-size scheme row exceeds the available width: {row}")
        widths = [w + (available - sum(widths)) / len(row) for w in widths]
        x0 = 2.5
        for rx, w in zip(row, widths):
            r = rec[rx]
            ax.add_patch(FancyBboxPatch((x0, y - h), w, h, boxstyle="round,pad=0,rounding_size=3.5", fc=LIGHT, ec="none"))
            ax.text(x0 + 6, y - PAD, NAMES[rx], fontsize=7.2, fontweight="bold", va="top")
            ax.text(x0 + w - 6, y - PAD, f"{r.shared_molecules:,} shared", fontsize=6.2, va="top", ha="right", color=MUTED)
            sw, sh = scheme_size(SCHEMES[rx])
            assert sw * SCHEME_SCALE <= w - 8, (rx, sw * SCHEME_SCALE, w)
            draw_scheme(ax, SCHEMES[rx], x0 + (w - sw * SCHEME_SCALE) / 2, y - PAD - NAME - GAP - (b - sh * SCHEME_SCALE) / 2)
            ax.text(x0 + 6, y - h + 3, f"{r.electrophiles:,} × {r.nucleophiles:,} molecules\n{r.pairs:,} pairs", fontsize=6.2,
                    va="bottom", color=MUTED)
            x0 += w + 3.0
        y -= h + 2.0
    BF.figure_header(fig, .005, 1 - 10.5 / 72 / H, "a", "Fifteen arrays, crossed under twenty conditions each")

    rows = tab("Figure1b_three_changes.csv")
    ax = fig.add_axes([.02, .012, .42, lower - .037]); ax.set_axis_off(); ax.set(xlim=(0, 1), ylim=(0, 1))

    def tiles(x, y, n, fc, ec, hatch=None):
        for j in range(n):
            ax.add_patch(Rectangle((x + j * .048, y), .04, .06, fc=fc, ec=ec, lw=.7, hatch=hatch))
    for i, r in rows.iterrows():
        y, c = .78 - i * .31, CHANGE[r.change]
        ax.text(0, y + .10, r.change, color=c, fontweight="bold", fontsize=7.0, va="center")
        ax.text(0, y + .01, r.question, fontsize=6.0, va="center")
        ax.text(0, y - .08, "kept: " + r.kept, fontsize=6.0, color=MUTED, va="center")
        x0 = .67
        if i == 0:
            ax.text(x0 - .02, y + .07, "conditions A", fontsize=5.8, ha="right", va="center", color=MUTED)
            ax.text(x0 - .02, y - .03, "conditions B", fontsize=5.8, ha="right", va="center", color=MUTED)
            tiles(x0, y + .04, 6, c, c); tiles(x0, y - .06, 6, "white", c, "////")
        elif i == 1:
            ax.text(x0 - .02, y + .02, "partners", fontsize=5.8, ha="right", va="center", color=MUTED)
            tiles(x0, y - .01, 3, c, c); tiles(x0 + 3 * .048, y - .01, 3, "white", c, "////")
            ax.text(x0 + .07, y - .06, "A", fontsize=6, ha="center", color=c); ax.text(x0 + .215, y - .06, "B", fontsize=6, ha="center", color=c)
        else:
            ax.text(x0 - .02, y + .07, "reaction 1", fontsize=5.8, ha="right", va="center", color=MUTED)
            ax.text(x0 - .02, y - .03, "reaction 2", fontsize=5.8, ha="right", va="center", color=MUTED)
            tiles(x0, y + .04, 6, c, c); tiles(x0, y - .06, 6, GREY, GREY)
    BF.figure_header(fig, .005, lower - .018, "b", "Three ways to reuse a record")

    wb = tab("Figure1c_pooled_portability.csv")
    ax = fig.add_axes([.60, .052, .385, lower - .103])
    color = {"conditions": TEAL, "partners": PURPLE, "reactions": RED}
    x, ticks, labels, centers = 0.0, [], [], {}
    for kind in ("conditions", "partners", "reactions"):
        start = x
        for r in wb.loc[wb.kind.eq(kind)].itertuples():
            filled = r.interval == "95% CI"
            ax.plot([x, x], [r.ci_low, r.ci_high], color=color[kind], lw=1.1)
            ax.plot(x, r.estimate, "o" if filled else "s", ms=4.6, mfc=color[kind] if filled else "white", mec=color[kind], mew=1.1)
            ticks.append(x); labels.append(r.label); x += 1
        centers[kind] = (start + x - 1) / 2
        x += .7
    ax.axhline(0, color="#ABB3B9", lw=.7)
    compact_labels = ["Condition\nhalves", "1st /\n2nd set", "Partner\nhalves",
                      "1st\nset", "2nd\nset", "2nd B–H\ncampaign"]
    assert len(labels) == len(compact_labels)
    ax.set_xticks(ticks); ax.set_xticklabels(compact_labels, fontsize=6.0)
    ax.set(ylim=(-.25, 1.02), xlim=(-.6, x - .1), ylabel="Rank agreement")
    for kind, text in (("conditions", "new\nconditions"), ("partners", "new\npartners"), ("reactions", "another\nreaction")):
        ax.text(centers[kind], 1.01, text, ha="center", va="bottom", fontsize=6.4, color=color[kind], fontweight="bold")
    BF.clean(ax)
    BF.figure_header(fig, .5, lower - .018, "c", "How much of the record each change keeps")
    return fig


def figure4():
    pairs, metric = tab("Figure4a_heldout_amide_pairs.csv"), tab("Figure4a_metrics.csv").iloc[0]
    sel, pos = tab("Figure4b_selection_vs_agreement.csv"), tab("Figure4c_Heck_Suzuki_molecules.csv")
    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_axes([.085, .57, .38, .35])
    ax.scatter(pairs.prediction_pp, pairs.target_pp, s=4, c=BLUE, alpha=.3, edgecolors="none")
    values = pairs[["prediction_pp", "target_pp"]].to_numpy()
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    padding = .08 * (hi - lo)
    lim = (lo - padding, hi + padding)
    ax.plot(lim, lim, color=GREY, lw=.8, ls=(0, (4, 3)))
    ax.set(xlim=lim, ylim=lim, aspect="equal")
    ax.set_xlabel("Predicted (pp)"); ax.set_ylabel("Measured, condition-adjusted (pp)")
    ax.text(.04, .97, f"{int(metric.scored_pairs):,} held-out amide pairs\nerror {metric.prediction_rmse_pp:.1f} pp\n(mean: {metric.reaction_mean_rmse_pp:.1f} pp)",
            transform=ax.transAxes, va="top", fontsize=6.4)
    BF.clean(ax)
    BF.figure_header(fig, .005, .955, "a", "Held-out pairs")
    ax = fig.add_axes([.6, .57, .38, .35])
    ax.axhline(0, color=GREY, lw=.8)
    for side, color in (("elec", BLUE), ("nuc", ORANGE)):
        r = sel.loc[sel.side.eq(side)]
        ax.scatter(r.rho_obs, r.source_gain_pp, s=20, c=color, edgecolors="white", linewidths=.5, label="electrophiles" if side == "elec" else "nucleophiles")
    for ra, rb, side, label, dx, dy in (("Buchwald-Hartwig", "SNAr", "nuc", "B–H → S$_\\mathregular{N}$Ar", .025, 3.2),
                                        ("Heck", "Suzuki", "elec", "Heck → Suzuki", -.05, 2.2)):
        r = sel.loc[sel.rxn_a.eq(ra) & sel.rxn_b.eq(rb) & sel.side.eq(side)].iloc[0]
        ax.annotate(label, (r.rho_obs, r.source_gain_pp), xytext=(r.rho_obs + dx, r.source_gain_pp + dy), fontsize=6.3,
                    arrowprops={"arrowstyle": "-", "lw": .5, "color": GREY})
    ax.set(xlabel="Agreement of the two reactions", ylabel="Gain over a random ten (pp)")
    ax.margins(x=.10, y=.18)
    ax.legend(loc="lower right", frameon=False, fontsize=6.2, handletextpad=.1)
    BF.clean(ax)
    BF.figure_header(fig, .515, .955, "b", "Borrowing a record")
    ax = fig.add_axes([.085, .08, .38, .36])
    ax.axhline(0, color="#CED4D8", lw=.7); ax.axvline(0, color="#CED4D8", lw=.7)
    ax.scatter(pos.x_contribution_pp, pos.y_contribution_pp, s=18, color=BLUE, alpha=.9, ec="white", lw=.4)
    for _, r in pos.loc[pos.drawn.gt(0)].iterrows():
        ax.annotate(str(int(r.drawn)), (r.x_contribution_pp, r.y_contribution_pp), xytext=(4, 3), textcoords="offset points", fontsize=7, fontweight="bold", color=INK)
    ax.set(xlabel="Heck (pp)", ylabel="Suzuki–Miyaura (pp)"); ax.margins(.12); BF.clean(ax)
    lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + .24 * (hi - lo))
    ax.text(.03, .97, f"{len(pos)} shared aryl bromides\nrank agreement {pos.rho_obs.iloc[0]:.2f}".replace("-", "−"), transform=ax.transAxes, va="top", fontsize=6.6)
    BF.figure_header(fig, .005, .49, "c", "A pair of reactions whose records carry: Heck and Suzuki–Miyaura")
    strip = [{"smiles": r.mol, "orient": V1.ORIENT.get(r.mol, 0), "number": int(r.drawn), "caption": f"{BF.signed(r.x_contribution_pp)} / {BF.signed(r.y_contribution_pp)}"}
             for _, r in pos.loc[pos.drawn.gt(0)].sort_values("drawn").iterrows()]
    assert len(strip) == 4
    BF.structure_strip(fig, [.53, .25, .44, .2], strip[:2], centre=.53)
    BF.structure_strip(fig, [.53, .03, .44, .2], strip[2:], centre=.53)
    return fig


CONDITION_ROLES = ("metal_sources", "ligands", "bases_acids", "other_reagents")
LIGANDS = {
    "1,10-Phen", "1-(pyridin-2-yl)propan-2-one", "1H-imidazole-4-carboxylic acid",
    "2,2-bipyridine", "2-(2-pyridyl)benzimidazole", "8-hydroxyquinoline", "AdBippyPhos",
    "BrettPhos", "CyAPhos", "CyPF-tBu", "DETA", "DMEDA", "DavePhos", "IMes.HCl", "JohnPhos",
    "L-proline", "Me4Phen", "Me4tBuXPhos", "N,N-dimethylglycine", "N-methyl-L-proline",
    "O=PPh3", "P(Mes)3", "P(o-tolyl)3", "P(tBu)3", "PBu3", "PCy3", "PPh3",
    "RuPhos", "SIPr.HCl", "SPhos", "SymPhos", "TAEA", "TMEDA", "XPhos", "XantPhos",
    "dipivaloylmethane", "dppb", "dppf", "ethyl 2-oxocyclohexanecarboxylate", "lysine",
    "neocuproine", "tBuBrettPhos", "tBuXphos", "trans-1,2-Cy-(NHMe)2",
}
BASES_ACIDS = {
    "2,6-lutidine", "AcOH", "Cs2CO3", "Cy2NMe", "DBU", "DIPEA", "DMAP", "H3BO3",
    "HCOONa", "K2CO3", "K3PO4", "KHCO3", "KOAc", "KOH", "KOSiMe3", "KOPh", "KOtBu",
    "LiHMDS", "MTBD", "NMI", "NMM", "Na2CO3", "NaHCO3", "NaOAc", "NaOH", "NaOMe",
    "NaOtAm", "NaOtBu", "PTSA monohydrate", "Py", "TEA", "TFA", "decanoic acid",
    "iPr2NH", "triethanolamine",
}


def _condition_role(reagent):
    if reagent in LIGANDS:
        return "ligands"
    if reagent in BASES_ACIDS:
        return "bases_acids"
    if re.search(r"(?:^|[ (])(?:Pd|Cu|Ni|Fe|Cd|Zn|Mg|Ti|Sn|Au|Ag)(?:[A-Z0-9(\[ /]|$)", reagent) or reagent in {"MTO", "PEPPSI-iPr"}:
        return "metal_sources"
    return "other_reagents"


def condition_composition_data():
    """Exact twenty-condition records, with display roles derived from named components."""
    counts = tab("Figure1a_reactions.csv")
    conditions = pd.read_csv(ROOT / "data/metadata/base_condition_roles.csv").fillna("")
    conditions = conditions.loc[conditions["set"].eq("analysed")].copy()
    if set(conditions.rxn_class) != set(counts.reaction):
        raise ValueError("The fifteen arrays and condition metadata differ")
    records = []
    for array_order, reaction in enumerate(counts.reaction, 1):
        group = conditions.loc[conditions.rxn_class.eq(reaction)].sort_values("coverage_rank")
        if len(group) != 20 or group.condition_id.nunique() != 20:
            raise ValueError(f"Expected twenty current conditions for {reaction}")
        for record in group.to_dict("records"):
            by_role = {role: [] for role in CONDITION_ROLES}
            for reagent in record["reagents"].split(";"):
                reagent = reagent.strip()
                if reagent:
                    by_role[_condition_role(reagent)].append(reagent)
            records.append({"array_order": array_order, **record,
                            **{role: "; ".join(items) for role, items in by_role.items()}})
    return pd.DataFrame.from_records(records)


def condition_composition_summary():
    """Exact distinct component scope across each array's twenty conditions."""
    records = []
    for reaction, group in condition_composition_data().groupby("rxn_class", sort=False):
        if group.temperature_C.nunique() != 1 or group.time_h.nunique() != 1:
            raise ValueError(f"Temperature/time varies within {reaction}")
        row = {"array_order": int(group.array_order.iloc[0]), "reaction": reaction, "n_conditions": len(group),
               "temperature_C": float(group.temperature_C.iloc[0]), "time_h": float(group.time_h.iloc[0])}
        for role in CONDITION_ROLES:
            row[role] = "; ".join(sorted({item.strip() for value in group[role] for item in value.split(";") if item.strip()}, key=str.casefold))
        row["solvents"] = "; ".join(sorted(group.solvent.unique(), key=str.casefold))
        row["conditions_without_added_reagents"] = int(group.reagents.eq("").sum())
        records.append(row)
    return pd.DataFrame.from_records(records)


def _composition_display(value):
    # The metadata uses replacement characters at hydrate/solvate separators.
    return str(value).replace("\ufffd\ufffd", "·")


def _composition_wrap(text, width_pt, renderer, fontsize):
    props = FontProperties(family="Arial", size=fontsize)
    pixels_per_pt = renderer.points_to_pixels(1)
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if line and renderer.get_text_width_height_descent(trial, props, False)[0] > width_pt * pixels_per_pt:
            lines.append(line)
            line = word
        else:
            line = trial
        if renderer.get_text_width_height_descent(line, props, False)[0] > width_pt * pixels_per_pt:
            pieces = re.split(r"(?<=[-;])", line)
            line = ""
            for piece in pieces:
                trial = line + piece
                if line and renderer.get_text_width_height_descent(trial, props, False)[0] > width_pt * pixels_per_pt:
                    lines.append(line)
                    line = piece
                else:
                    line = trial
    if line:
        lines.append(line)
    return "\n".join(lines)


def ed_fig1():
    """Show the actual reagent and solvent scope of all fifteen twenty-condition arrays."""
    summary = condition_composition_summary()
    width, fontsize, leading = 504.0, 5.5, 6.4
    widths = [60.0, 140.0, 110.0, 75.0, 85.0, 26.0]
    headers = ["Array\n20 conditions each", "Metal sources / coupling\nand other reagents", "Ligands", "Bases / acids", "Solvents", "°C / h"]
    fig = plt.figure(figsize=(width / 72, 7.4))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rows = []
    for record in summary.to_dict("records"):
        other = record["other_reagents"]
        if record["conditions_without_added_reagents"]:
            other = (other + "; " if other else "") + "no added reagent"
        reaction_name = "SNAr" if record["reaction"] == "SNAr" else NAMES[record["reaction"]]
        metal_other = "; ".join(value for value in (record["metal_sources"], other) if value)
        values = [reaction_name, metal_other or "none", record["ligands"] or "none", record["bases_acids"] or "none",
                  record["solvents"], f"{record['temperature_C']:g}\n{record['time_h']:g}"]
        wrapped = [_composition_wrap(_composition_display(value), w - 6, renderer, fontsize)
                   if i != 5 else value for i, (value, w) in enumerate(zip(values, widths))]
        rows.append((wrapped, max(text.count("\n") + 1 for text in wrapped) * leading + 7))
    top_margin, header_height, bottom_margin = 18.0, 20.0, 5.0
    height = top_margin + header_height + sum(row_height for _, row_height in rows) + bottom_margin
    if height > 538.5:
        raise ValueError(f"Condition-composition table is too tall for a 190-mm page panel: {height:g} pt")
    fig.set_size_inches(width / 72, height / 72)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.set_title("Reagent and solvent composition of the fifteen condition arrays", loc="left", y=1, pad=-10,
                 fontsize=7.0, fontweight="bold", gid="condition-composition")
    x_edges = np.r_[4, 4 + np.cumsum(widths)]
    top = height - top_margin
    ax.add_patch(Rectangle((4, top - header_height), sum(widths), header_height, fc=INK, ec="none"))
    for left, text in zip(x_edges[:-1], headers):
        ax.text(left + 3, top - 3, text, fontsize=5.5, weight="bold", color="white", va="top", gid="condition-composition")
    top -= header_height
    for i, (texts, row_height) in enumerate(rows):
        ax.add_patch(Rectangle((4, top - row_height), sum(widths), row_height, fc=LIGHT if i % 2 == 0 else "white", ec="none"))
        for column, (left, text) in enumerate(zip(x_edges[:-1], texts)):
            ax.text(left + 3, top - 3.5, text, fontsize=fontsize, weight="bold" if column == 0 else "normal",
                    va="top", linespacing=leading / fontsize, gid="condition-composition")
        ax.plot([4, width - 4], [top - row_height, top - row_height], color="#C8CDD1", lw=.35)
        top -= row_height
    return fig


def ed_fig2():
    m = tab("ExtendedDataFig2_amide_matrix.csv")
    fig = plt.figure(figsize=(4.8, 3.7))
    ax = fig.add_axes([.12, .14, .70, .80])
    n_acid, n_amine = m.acid_rank.max(), m.amine_rank.max()
    grid = np.full((n_acid, n_amine), np.nan)
    grid[m.acid_rank - 1, m.amine_rank - 1] = m.pair_effect_pp
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("white")
    lim = float(np.nanpercentile(np.abs(grid), 98))
    im = ax.imshow(grid, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    ax.set(xticks=[], yticks=[])
    ax.set_xlabel(f"Amines and sulfonamides ({n_amine}), highest contribution at left")
    ax.set_ylabel(f"Carboxylic acids ({n_acid}), highest contribution at top")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#C8CDD1"); sp.set_linewidth(.6)
    cax = fig.add_axes([.85, .14, .025, .80])
    cb = fig.colorbar(im, cax=cax); cb.set_label("Pair yield relative to reaction mean (pp)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    return fig




def main():
    BF.style()
    for name, make in (("Figure1", figure1), ("Figure4", figure4), ("ExtendedDataFig1", ed_fig1), ("ExtendedDataFig2", ed_fig2)):
        fig = make()
        BF.finalize(fig)   # text overlaps are checked on the finished layout
        bad = V1.overlaps(fig)
        save(fig, name)
        plt.close(fig)
        print(name, "text overlaps:", bad if bad else "none")


if __name__ == "__main__":
    main()
