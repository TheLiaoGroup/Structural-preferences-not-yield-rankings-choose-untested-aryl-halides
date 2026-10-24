"""Extended Data Figs 3-8 and the shared figure style; plots the panel tables only."""

from __future__ import annotations

import argparse

import math

import os

from pathlib import Path

import re

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib import font_manager

from matplotlib.lines import Line2D

from matplotlib.patches import Rectangle, PathPatch

from matplotlib.path import Path as MPath

from matplotlib.text import Annotation, Text

from matplotlib.mathtext import MathTextParser

from rdkit import Chem, RDLogger

from rdkit.Chem import Draw, rdDepictor

from rdkit.Chem.Draw import rdMolDraw2D

from rdkit.Geometry import Point3D

import chemdraw_vector as CD

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "output/presentation/panel_data"
OUT = ROOT / "output/presentation/figures"

STRUCTURE_FEATURES = {"halogen next to ring N": ("Halogen beside a ring N", "Brc1ccccn1"),
                      "N–H": ("N–H in the aryl halide", "Brc1cccc2[nH]ncc12")}

STRUCTURE_SETS = {"first condition set": ("o", 4.6, True), "second condition set": ("D", 3.4, False)}

BLUE, ORANGE, TEAL, RED = "#28658A", "#B86C27", "#147A71", "#BD4939"

INK, GREY, LIGHT, MUTED = "#202932", "#929BA3", "#EEF1F3", "#59636C"

NAMES = {"Acylation": "Amide coupling", "Heck": "Heck", "Suzuki": "Suzuki–Miyaura",
         "Ullmann": "Ullmann ether", "Buchwald-Hartwig": "Buchwald–Hartwig",
         "Kumada": "Kumada", "Chan-Lam": "Chan–Lam", "SNAr": "S$_\\mathregular{N}$Ar",
         "Hiyama": "Hiyama", "Grignard": "Grignard addition", "C-O competitive reactions": "C–O coupling",
         "Reductive-amination": "Reductive amination", "Esterification": "Esterification",
         "Sonogashira": "Sonogashira", "Suzuki Bpin": "Suzuki–Miyaura (Bpin)"}

SHORT = {**NAMES, "Buchwald-Hartwig": "B–H", "Acylation": "Amide", "Suzuki": "Suzuki",
         "Grignard": "Grignard add.", "C-O competitive reactions": "C–O", "Reductive-amination": "Red. amin.",
         "Esterification": "Ester", "Sonogashira": "Sonog.", "Ullmann": "Ullmann", "Suzuki Bpin": "Suzuki (Bpin)"}

ROLE = {"Acylation": ("acid", "amine"), "Esterification": ("acid", "alcohol"),
        "Buchwald-Hartwig": ("aryl halide", "amine"), "SNAr": ("aryl halide", "amine"),
        "Chan-Lam": ("boronic acid", "amine"), "Reductive-amination": ("carbonyl", "amine"),
        "Suzuki": ("aryl halide", "boronic acid"), "Suzuki Bpin": ("aryl halide", "boronate"),
        "Heck": ("aryl bromide", "alkene"), "Sonogashira": ("aryl bromide", "alkyne"),
        "Hiyama": ("aryl halide", "silane"), "Kumada": ("aryl halide", "RMgX"),
        "Grignard": ("carbonyl", "RMgX"), "Ullmann": ("aryl bromide", "phenol"),
        "C-O competitive reactions": ("aryl bromide", "O-nucleophile")}

def dec(v):
    return f"{v:+.1f}".replace("-", "−")

def signed(v):
    return f"{v:+.0f}".replace("-", "−")

def table(name):
    return pd.read_csv(SOURCES / name)


def bool_series(series):
    return series.astype(str).str.lower().eq("true")

def style():
    # Duplicate font installations must not let cache discovery order change PDF embedding.
    system_fonts = Path(os.environ["WINDIR"]) / "Fonts" if os.name == "nt" and "WINDIR" in os.environ else None
    font_manager.fontManager.ttflist.sort(key=lambda entry: (
        system_fonts is not None and not Path(entry.fname).is_relative_to(system_fonts),
        Path(entry.fname).as_posix().casefold()))
    font_manager.fontManager._findfont_cached.cache_clear()
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.linewidth": .65,
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": SVG_TEXT,
                         "svg.hashsalt": "jacs-building-block-yield",
                         "axes.unicode_minus": True,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})

def panel(ax, letter, title=None, x=-.1):
    """Panel letter and title share one baseline, 6 pt above the top of the axes; finalize() puts the letter on the left
    edge of the panel and the title after it."""
    a = ax.annotate(letter, xy=(x, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold", va="baseline", ha="left", gid="panel-letter")
    a.panel_ax = ax
    if title:
        b = ax.set_title(title, loc="left", x=x, y=1.0, pad=6, fontsize=8.5,
                         fontweight="bold", va="baseline", ha="left", gid="panel-title")
        b.panel_letter = a


def figure_header(fig, x, y, letter, title):
    """A figure-coordinate header whose displayed title is an actual left axes title."""
    a = fig.text(x, y, letter, fontsize=10, fontweight="bold", va="baseline", ha="left", gid="panel-letter")
    carrier = fig.add_axes([0, 0, 1, 1], label=f"panel-header-{len(fig.axes)}")
    carrier.set_axis_off()
    carrier._figure_header_carrier = True
    b = carrier.set_title(title, loc="left", x=x + .026, y=y, pad=0, va="baseline", ha="left",
                          fontsize=8.5, fontweight="bold", gid="panel-title")
    b.panel_letter = a
    return a, b


# One type scale and one panel layout for every figure (applied by finalize() before saving).
LABEL_CASE = "lower"        # "upper" for journals that print capital panel letters
SVG_TEXT = "none"           # "path" writes SVG text as outlines, so any SVG reader draws it with the same spacing
GAP_PT = 13.0               # panel title starts this far after the left edge of its letter
CAP = .716                  # cap height of Arial as a fraction of the font size: titles are raised so capitals share one top
MAX_WIDTH_MM = 180.0
MIN_TEXT_PT = 5.0
_MATH_TEXT = MathTextParser("path")


def tier(size):
    """Every text other than a panel letter (10 pt) or panel title (8.5 pt): 7 pt, or 6 pt for small annotations."""
    return 7.0 if size >= 6.5 else 6.0


def _shift(fig, t, dx, dy=0.0):
    """Move a text dx display pixels to the right and dy up."""
    if isinstance(t, Annotation):
        ox, oy = t.xyann
        t.xyann = (ox + dx * 72 / fig.dpi, oy + dy * 72 / fig.dpi)
    else:
        X, Y = t.get_transform().transform(t.get_position())
        t.set_position(tuple(t.get_transform().inverted().transform((X + dx, Y + dy))))


def finalize(fig):
    texts = [t for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    for t in texts:
        gid = t.get_gid()
        if gid in ("scheme", "chemdraw", "condition-composition"):
            continue
        t.set_text(re.sub(r"(?<![A-Za-z0-9])-(?=(?:\d|\.\d))", "−", t.get_text()))
        if gid == "panel-letter":
            t.set_text(t.get_text().upper() if LABEL_CASE == "upper" else t.get_text().lower())
            t.set_fontsize(10.0)
        elif gid == "panel-title":
            t.set_fontsize(8.5)
        else:
            t.set_fontsize(tier(t.get_fontsize()))
        if t.get_parse_math() and "$" in t.get_text() and re.search(r"[_^]", t.get_text()):
            t.set_fontsize(max(t.get_fontsize(), 7.8))  # Keep 70% scripts above 5 pt after 7-to-6.5-inch placement.
    letters = [t for t in texts if t.get_gid() == "panel-letter"]
    titles = [t for t in texts if t.get_gid() == "panel-title"]
    if not letters:
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for t in letters:   # a letter drawn by panel() sits on the left edge of its panel, tick labels included
        ax = getattr(t, "panel_ax", None)
        if ax is not None:
            _shift(fig, t, ax.get_tightbbox(r).x0 - t.get_window_extent(r).x0)
    fig.canvas.draw()
    for t in letters + titles:
        t.set_visible(False)
    carriers = [ax for ax in fig.axes if getattr(ax, "_figure_header_carrier", False) and ax.get_visible()]
    for ax in carriers:
        ax.set_visible(False)
    left = fig.get_tightbbox(r).x0 * fig.dpi
    for ax in carriers:
        ax.set_visible(True)
    for t in letters + titles:
        t.set_visible(True)
    x0 = {t: t.get_window_extent(r).x0 for t in letters}
    first = min(x0.values())
    edge = min(left, first)
    for t in letters:   # the first column: every letter on the left edge of the drawing
        if x0[t] < first + 3 * fig.dpi / 72:
            _shift(fig, t, edge - x0[t])
    fig.canvas.draw()
    for t in titles:   # after its letter, on the letter's baseline raised so that the capitals of both share one top
        letter = t.panel_letter
        _shift(fig, t, letter.get_window_extent(r).x0 + GAP_PT * fig.dpi / 72 - t.get_window_extent(r).x0,
               CAP * (letter.get_fontsize() - t.get_fontsize()) * fig.dpi / 72 if not getattr(t, "raised", False) else 0.0)
        t.raised = True
    fig.canvas.draw()


def figure_qa(fig):
    """Check physical page size and rendered text, without changing the drawing scale."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update((id(tick.label1), id(tick.label2)))
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and id(t) not in hidden
             and (t.axes is None or t.axes.get_visible())]
    smallest = float("inf")
    undersized, outside = [], []
    width_px, height_px = fig.bbox.width, fig.bbox.height
    tolerance = .5 * fig.dpi / 72
    for t in texts:
        size = t.get_fontsize()
        if t.get_parse_math() and "$" in t.get_text():
            glyphs = [glyph for line in t.get_text().splitlines()
                      for glyph in _MATH_TEXT.parse(line, dpi=72, prop=t.get_fontproperties()).glyphs]
            size = min((glyph[1] for glyph in glyphs), default=size)
        smallest = min(smallest, size)
        if size < MIN_TEXT_PT - 1e-6:
            undersized.append((t.get_text(), float(size)))
        bbox = t.get_window_extent(renderer)
        if bbox.x0 < -tolerance or bbox.y0 < -tolerance or bbox.x1 > width_px + tolerance or bbox.y1 > height_px + tolerance:
            outside.append(t.get_text())
    return {"width_mm": float(fig.get_figwidth() * 25.4), "height_mm": float(fig.get_figheight() * 25.4),
            "min_text_pt": float(smallest), "undersized_text": undersized, "outside_canvas_text": outside,
            "left_titles": [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]}


def export_figure(fig, destination, creator="build_figures.py"):
    """Export the original physical canvas; no crop, expansion or structure scaling is applied."""
    finalize(fig)
    qa = figure_qa(fig)
    if qa["width_mm"] > MAX_WIDTH_MM + 1e-6:
        raise ValueError(f"Figure width {qa['width_mm']:.3f} mm exceeds {MAX_WIDTH_MM:g} mm")
    if qa["undersized_text"]:
        raise ValueError(f"Figure has text below {MIN_TEXT_PT:g} pt: {qa['undersized_text']}")
    if qa["outside_canvas_text"]:
        raise ValueError(f"Figure text extends outside its fixed canvas: {qa['outside_canvas_text']}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for ext in ("pdf", "svg", "png"):
            metadata = {"Creator": creator, "CreationDate": None, "ModDate": None} if ext == "pdf" else (
                {"Creator": creator, "Date": None} if ext == "svg" else {"Software": creator})
            fig.savefig(destination.parent / f"{destination.name}.{ext}", dpi=600 if ext == "png" else 300,
                        bbox_inches=None, pad_inches=0, metadata=metadata)
    return qa


def save_figure(fig, name):
    export_figure(fig, OUT / name)
    plt.close(fig)

def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=.65)

_NUM = r"-?\d+(?:\.\d+)?"

def _svg_path(d):
    tokens = re.findall(rf"[MLQCZ]|{_NUM}", d)
    verts, codes, i, cmd = [], [], 0, None
    while i < len(tokens):
        t = tokens[i]
        if t in "MLQCZ":
            cmd, i = t, i + 1
            if cmd == "Z":
                verts.append(verts[-1] if verts else (0, 0)); codes.append(MPath.CLOSEPOLY)
            continue
        n = {"M": 2, "L": 2, "Q": 4, "C": 6}[cmd]
        vals = [float(v) for v in tokens[i:i + n]]
        i += n
        if cmd in ("M", "L"):
            verts.append(tuple(vals)); codes.append(MPath.MOVETO if cmd == "M" else MPath.LINETO)
            cmd = "L" if cmd == "M" else cmd
        elif cmd == "Q":
            verts += [tuple(vals[:2]), tuple(vals[2:])]; codes += [MPath.CURVE3] * 2
        else:
            verts += [tuple(vals[:2]), tuple(vals[2:4]), tuple(vals[4:])]; codes += [MPath.CURVE4] * 3
    return MPath(np.array(verts), codes)

def upright(smiles, orient):
    """RDKit's 2D depiction turned rigidly so that its first six-membered ring stands with vertical side edges, then by
    orient x 60 degrees. Connectivity, bond lengths and atom labels are unchanged. Returns the molecule and that ring."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    ring = max(mol.GetRingInfo().AtomRings(), key=lambda r: (len(r) == 6, len(r)))
    p, q = conf.GetAtomPosition(ring[0]), conf.GetAtomPosition(ring[1])
    a = math.radians((30 - math.degrees(math.atan2(q.y - p.y, q.x - p.x))) % 60 + 60 * orient)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x * math.cos(a) - p.y * math.sin(a), p.x * math.sin(a) + p.y * math.cos(a), 0.0))
    return mol, ring

def depiction(smiles, orient=None):
    """ChemDraw SVG and physical point dimensions, with an optional vertical anchor."""
    asset = CD.molecule(smiles, orient)
    return CD.svg_text(asset), float(asset["width_pt"]), float(asset["height_pt"]), asset.get("anchor_y_pt")

def structure(ax, smiles, cx, cy, scale, orient=None):
    """Draw the editable ChemDraw master's vectors at 1:1 points; legacy scale is ignored."""
    asset = CD.molecule(smiles, orient)
    w, h = float(asset["width_pt"]), float(asset["height_pt"])
    anchor = asset.get("anchor_y_pt")
    return CD.draw(ax, asset, cx - w / 2, cy + (h / 2 if anchor is None else float(anchor)))

def structure_strip(fig, rect, rows, scale=.62, label_color=INK, centre=.58):
    """A strip of numbered structures under a scatter; rows carry smiles, number, a caption and optionally orient (see
    structure). Numbers and captions sit at fixed heights; the structures are drawn on the line at centre x height."""
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    width_pt, height_pt = rect[2] * fig.get_figwidth() * 72, rect[3] * fig.get_figheight() * 72
    ax.set(xlim=(0, width_pt), ylim=(0, height_pt))
    dimensions = [depiction(r["smiles"], r.get("orient"))[1:] for r in rows]
    gap = (width_pt - sum(w for w, _, _ in dimensions)) / (len(rows) + 1)
    if gap < 0:
        raise ValueError("ChemDraw structure row is narrower than its native-size drawings")
    line = height_pt * centre
    oriented = [depiction(r["smiles"], r["orient"])[1:] for r in rows if r.get("orient") is not None]
    oriented = [(w, h, h / 2 if ry is None else ry) for w, h, ry in oriented]
    if oriented:   # one ring line for the row, placed so that the row's drawings are centred at centre x height
        line -= (max(ry for _, _, ry in oriented) - max(h - ry for _, h, ry in oriented)) / 2
    left = gap
    for r, (w, _, _) in zip(rows, dimensions):
        cx = left + w / 2
        structure(ax, r["smiles"], cx, line, scale, r.get("orient"))
        ax.text(cx, height_pt * .1, r["caption"], ha="center", va="center", fontsize=6.3, color=MUTED)
        ax.text(left, height_pt * .95, str(r["number"]), ha="left", va="top", fontsize=7.5,
                fontweight="bold", color=r.get("color", label_color))
        left += w + gap
    return ax


PURPLE = "#766196"





def extended_data_fig2():
    defined = table("ExtendedDataFig2_defined_benchmark.csv"); pooled = table("ExtendedDataFig2_pooled_shortfall.csv")
    fig, ax = plt.subplots(figsize=(4.0, 3.7)); fig.subplots_adjust(left=.16, right=.97, top=.95, bottom=.14)
    ax.plot([0, 1], [0, 1], color=GREY, lw=.9, ls=(0, (4, 3)))
    ax.axhline(0, color="#CAD0D5", lw=.7)
    for side, color in (("elec", BLUE), ("nuc", ORANGE)):
        r = defined.loc[defined.side.eq(side)]
        ax.scatter(r.geometric_benchmark, r.rho_R_half, s=10 + 1.1 * r.molecules, c=color, alpha=.9, edgecolors="white", linewidths=.5,
                   label="Electrophiles" if side == "elec" else "Nucleophiles")
    ax.set(xlim=(0, 1), ylim=(-.6, 1), xlabel="Agreement within each reaction, new partners", ylabel="Agreement between the two reactions")
    for ra, rb, label, dx, dy in (("Heck", "Suzuki", "Heck / Suzuki", -.30, .22), ("Acylation", "Chan-Lam", "Amide / Chan–Lam", .10, .10)):
        r = defined.loc[defined.rxn_a.eq(ra) & defined.rxn_b.eq(rb)].iloc[0]
        ax.annotate(label, (r.geometric_benchmark, r.rho_R_half), xytext=(r.geometric_benchmark + dx, r.rho_R_half + dy), fontsize=6.5,
                    arrowprops={"arrowstyle": "-", "lw": .5, "color": GREY})
    p = pooled.loc[pooled.minimum_partners.eq(2)].iloc[0]
    ax.text(.97, .04, f"{int(p.positive_benchmark_comparisons)} comparisons\nmean shortfall {p.D_G:.2f} ({p.low:.2f}–{p.high:.2f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, handletextpad=.2)
    clean(ax)
    save_figure(fig, "ExtendedDataFig3")

def figure_s1():
    inc = table("ExtendedDataFig7_prediction_all_reactions.csv")
    order = inc.groupby("reaction").scored_pairs.first().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(6.5, 5.2)); fig.subplots_adjust(left=.27, right=.97, top=.93, bottom=.1)
    for y in range(len(order)):
        if y % 2 == 0:
            ax.axhspan(y - .48, y + .48, color=LIGHT, zorder=0)
    for i, rxn in enumerate(order):
        for side, shift, color in (("elec", -.15, BLUE), ("nuc", .15, ORANGE)):
            r = inc.loc[inc.reaction.eq(rxn) & inc.side.eq(side)].iloc[0]
            lo, hi = max(r.low, -2), min(r.high, 1)
            ax.plot([lo, hi], [i + shift] * 2, color=color, lw=1)
            ax.plot(np.clip(r.increment, -2, 1), i + shift, "o", ms=4, mfc=color if r.interval_above_zero else "white", mec=color)
    n = inc.groupby("reaction").scored_pairs.first()
    ax.set(yticks=range(len(order)), yticklabels=[f"{NAMES[r]} ({int(n[r])})" for r in order], ylim=(len(order) - .5, -.6),
           xlim=(-2.05, 1.05), xlabel="Share of prediction error removed by one partner's contributions")
    ax.axvline(0, color="#ABB3B9", lw=.8); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False); clean(ax)
    ax.legend(handles=[Line2D([0], [0], marker="o", color=BLUE, label="Electrophile"), Line2D([0], [0], marker="o", color=ORANGE, label="Nucleophile")],
              loc="upper left", frameon=False)
    save_figure(fig, "ExtendedDataFig8")

FEATURE_SHORT = {"halogen next to ring N": "ring N beside X", "N–H": "N–H", "substituent beside the halogen": "ortho substituent",
                 "chloride leaving group": "X = Cl", "iodide leaving group": "X = I", "sulfonate leaving group": "X = OSO$_2$R",
                 "heteroaromatic ring at the halogen": "heteroaryl at X", "five-membered ring at the halogen": "five-membered ring at X",
                 "fused ring at the halogen": "fused ring at X", "electron-withdrawing group": "EWG", "electron-donating group": "EDG"}

def extended_data_profiles():
    comp = table("ExtendedDataFig6bc_profile_comparisons.csv")
    fig = plt.figure(figsize=(6.5, 3.0))
    c = comp.loc[comp.comparison.eq("second Buchwald-Hartwig campaign against an analysed reaction") & comp.reaction_a.eq("Buchwald-Hartwig") & comp.reaction_b.eq("Buchwald-Hartwig(CMC)")].iloc[0]
    ax = fig.add_axes([.18, .18, .28, .67])
    within = comp.loc[comp.comparison.str.startswith("same reaction")]
    order = sorted(set(within.reaction_a), key=lambda r: -within.loc[within.reaction_a.eq(r)].pearson.min())
    for i, r in enumerate(order):
        for kind, filled in (("same reaction, second condition set", True),):
            v = within.loc[within.reaction_a.eq(r) & within.comparison.eq(kind), "pearson"]
            if len(v):
                ax.plot(v.iloc[0], i, "o", ms=4, mfc=TEAL if filled else "white", mec=TEAL, mew=1)
    ax.set(yticks=range(len(order)), yticklabels=[SHORT[r] for r in order], ylim=(len(order) - .4, -.6), xlim=(-.5, 1.05), xlabel="Structural profile, r")
    ax.axvline(0, color="#ABB3B9", lw=.7); ax.tick_params(axis="y", length=0, labelsize=6.4); ax.spines["left"].set_visible(False); clean(ax)
    panel(ax, "a", "New conditions")
    ax = fig.add_axes([.64, .18, .32, .67])
    two = comp.loc[comp.comparison.eq("two analysed reactions")]
    ax.axhline(0, color="#CED4D8", lw=.6); ax.axvline(0, color="#CED4D8", lw=.6)
    ax.scatter(two.record_rank_agreement, two.pearson, s=12, color=GREY, ec="white", lw=.3, label="two reactions")
    ax.scatter([c.record_rank_agreement], [c.pearson], s=26, color=RED, ec="white", lw=.4, zorder=3, label="two B–H campaigns")
    ax.set(xlim=(-.6, 1), ylim=(-1, 1.05), xlabel="Building blocks: rank agreement", ylabel="Structural profile, r"); clean(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=6.2, handletextpad=.1, borderaxespad=.1)
    panel(ax, "b", "Two reactions")
    save_figure(fig, "ExtendedDataFig7")

def figure_s3():
    und = table("ExtendedDataFig3_undefined_benchmark.csv").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); fig.subplots_adjust(left=.3, right=.97, top=.95, bottom=.24)
    for i, r in und.iterrows():
        if i % 2 == 0:
            ax.axhspan(i - .45, i + .45, color=LIGHT, zorder=0)
        ax.plot([r.rho_P_a, r.rho_P_b], [i, i], color="#BAC2C7", lw=.8)
        ax.plot(r.rho_P_a, i, marker="|", ms=8, mew=1.4, color=BLUE)
        ax.plot(r.rho_P_b, i, marker="|", ms=8, mew=1.4, color=ORANGE)
        ax.plot(r.rho_R_half, i, "o", color=RED, ms=4)
    labels = [f"{SHORT[r.rxn_a]} / {SHORT[r.rxn_b]} · {'E' if r.side == 'elec' else 'N'} ({int(r.molecules)})" for _, r in und.iterrows()]
    ax.axvline(0, color=GREY, lw=.7)
    ax.set(xlim=(-1, 1), ylim=(len(und) - .4, -.6), yticks=range(len(und)), yticklabels=labels, xlabel="Rank agreement on the same shared molecules")
    ax.tick_params(axis="y", length=0, labelsize=6.6); ax.spines["left"].set_visible(False)
    ax.legend(handles=[Line2D([0], [0], marker="|", ls="", ms=7, color=BLUE, label="First reaction, new partners"),
                       Line2D([0], [0], marker="|", ls="", ms=7, color=ORANGE, label="Second reaction, new partners"),
                       Line2D([0], [0], marker="o", ls="", ms=4, color=RED, label="Between reactions")],
              loc="upper center", bbox_to_anchor=(.4, -.17), ncol=3, frameon=False, fontsize=6.5)
    save_figure(fig, "ExtendedDataFig4")

def figure_s4():
    om = table("ExtendedDataFig4a_leave_one_reaction_out.csv"); sens = table("ExtendedDataFig4b_outcome_scale.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=.17, right=.97, top=.9, bottom=.2, wspace=.9)
    ax = axes[0]; panel(ax, "a", "Leaving one reaction out")
    reactions = list(om.reaction_left_out.unique())
    for degree, shift, color in ((2, -.12, BLUE), (3, .12, ORANGE)):
        for _, r in om.loc[om.minimum_partners.eq(degree)].iterrows():
            ax.plot(r.D_G, reactions.index(r.reaction_left_out) + shift, "o", ms=3.6, color=color)
    ax.set(yticks=range(len(reactions)), yticklabels=[SHORT[x] for x in reactions], ylim=(len(reactions) - .4, -.6), xlim=(0, .5), xlabel="Mean shortfall")
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=BLUE, label="≥2 partners"), Line2D([0], [0], marker="o", ls="", color=ORANGE, label="≥3 partners")],
              loc="upper center", bbox_to_anchor=(.4, -.13), frameon=False, ncol=2)
    ax = axes[1]; panel(ax, "b", "Other yield scales")
    for i, r in sens.reset_index(drop=True).iterrows():
        ax.errorbar(r.D_G, i, xerr=[[r.D_G - r.low], [r.high - r.D_G]], fmt="o", ms=4, color=TEAL, capsize=2, lw=1)
        ax.text(.01, i + .28, f"{int(r.positive_benchmark_comparisons)}/{int(r.eligible_comparisons)} comparisons", fontsize=6.2, va="top", color=MUTED)
    ax.axvline(0, color=GREY, lw=.7)
    ax.set(yticks=range(len(sens)), yticklabels=[str(x).replace("Without SNAr and Chan–Lam", "Without S$_\\mathregular{N}$Ar\nand Chan–Lam") for x in sens.label],
           ylim=(len(sens) - .4, -.65), xlim=(-.02, .8), xlabel="Mean shortfall")
    save_figure(fig, "ExtendedDataFig5")

def figure_s2():
    data = table("ExtendedDataFig5_reaction_map.csv")
    order = sorted(set(data.rxn_a) | set(data.rxn_b)); arr = np.full((len(order), len(order)), np.nan); counts = {}
    for _, r in data.iterrows():
        a, b = order.index(r.rxn_a), order.index(r.rxn_b)
        i, j = (max(a, b), min(a, b)) if r.side == "elec" else (min(a, b), max(a, b))
        arr[i, j] = r.rho_obs; counts[i, j] = int(r.n_shared)
    fig, ax = plt.subplots(figsize=(6.5, 6.3)); fig.subplots_adjust(left=.2, right=.97, top=.97, bottom=.24)
    cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("#EDF0F2")
    im = ax.imshow(arr, vmin=-1, vmax=1, cmap=cmap)
    for (i, j), n in counts.items():
        ax.text(j, i, str(n), ha="center", va="center", fontsize=6.5, color="white" if abs(arr[i, j]) > .65 else INK)
    for i in range(len(order)):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fc="#C8CDD1", ec="white", lw=.6))
    ax.set(xticks=range(len(order)), yticks=range(len(order)), xticklabels=[SHORT[x] for x in order], yticklabels=[SHORT[x] for x in order])
    plt.setp(ax.get_xticklabels(), rotation=90); ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(order), 1), minor=True); ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.6); ax.tick_params(which="minor", length=0)
    cax = fig.add_axes([.3, .08, .5, .022]); cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cb.set_label("Rank agreement of shared molecules")
    save_figure(fig, "ExtendedDataFig6")

FIGURES = ["ExtendedDataFig3", "ExtendedDataFig4", "ExtendedDataFig5", "ExtendedDataFig6", "ExtendedDataFig7", "ExtendedDataFig8"]

def plot_figures():
    style()
    extended_data_fig2(); figure_s1(); figure_s2(); figure_s3(); figure_s4(); extended_data_profiles()
    print("Exported Extended Data Figs 3-8.")

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_figures()
