"""Shared semantic colours and final-size typography for the R5 Article."""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FuncFormatter

import plot_figures as BF

WIDTH_IN = 180 / 25.4
COLORS = {
    "record": "#73519B", "reward": "#28658A", "penalty": "#BD4939",
    "profile": "#147A71", "untested": "#C47829", "background": "#929BA3",
    "light": "#EEF1F3", "muted": "#59636C", "ink": "#202932", "white": "#FFFFFF",
}
PURPLE, BLUE, RED, TEAL, ORANGE = (COLORS[k] for k in ("record", "reward", "penalty", "profile", "untested"))
GREY, LIGHT, MUTED, INK, WHITE = (COLORS[k] for k in ("background", "light", "muted", "ink", "white"))
CMAP = LinearSegmentedColormap.from_list("structural_response", [RED, WHITE, BLUE])
clean = BF.clean
REACTION_NAMES = {**BF.NAMES, "SNAr": "SNAr"}
REACTION_ALIASES = {
    **REACTION_NAMES,
    "C–O competition": REACTION_NAMES["C-O competitive reactions"],
    "Grignard add.": REACTION_NAMES["Grignard"],
    "Ullmann etherification": REACTION_NAMES["Ullmann"],
    "Ullmann ether synthesis": REACTION_NAMES["Ullmann"],
    "Suzuki (Bpin)": REACTION_NAMES["Suzuki Bpin"],
    "B–H": REACTION_NAMES["Buchwald-Hartwig"],
    "B-H": REACTION_NAMES["Buchwald-Hartwig"],
    "S$_\\mathregular{N}$Ar": "SNAr",
    "Red. amin.": REACTION_NAMES["Reductive-amination"],
}
REACTION_ALIASES.update({name: name for name in REACTION_NAMES.values()})
_REACTION_PATTERN = re.compile(r"(?<![\w])(?:" + "|".join(
    re.escape(name).replace(r"\ ", r"\s+").replace("\u2013", "\u2013\\s*")
    if name in REACTION_NAMES.values() else re.escape(name)
    for name in sorted(REACTION_ALIASES, key=len, reverse=True)) + r")(?![\w])")


def reaction_name(name):
    return REACTION_NAMES.get(name, REACTION_ALIASES.get(name, name))


def reaction_text(text):
    # Longest full names take precedence; already canonical wrapped labels stay wrapped.
    return _REACTION_PATTERN.sub(lambda match: REACTION_ALIASES.get(match.group(), match.group()), text)


def display_text(text):
    # Apply scripts only to drawn labels; data and workbook labels stay plain text.
    return reaction_text(text).replace("SNAr", "S$_\\mathregular{N}$Ar")


def style():
    BF.style()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.prop_cycle": plt.cycler(color=[PURPLE, TEAL, GREY]),
                         "svg.fonttype": "none", "svg.hashsalt": "p1-natchem-r5"})


def signed(value, decimals=0):
    value = round(float(value), decimals)
    return (f"{0:.{decimals}f}" if value == 0 else f"{value:+.{decimals}f}").replace("-", "\u2212")


def header(fig, name, letter, x, y, titles):
    title = titles[name][ord(letter) - ord("a")] if isinstance(titles, dict) else titles[ord(letter) - ord("a")]
    return BF.figure_header(fig, x, y, letter, title)


def native(fig, rect, smiles, orient=None):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    width, height = rect[2] * fig.get_figwidth() * 72, rect[3] * fig.get_figheight() * 72
    asset = BF.CD.molecule(smiles, orient)
    aw, ah = asset["width_pt"], asset["height_pt"]
    if aw > width + .01 or ah > height + .01:
        raise ValueError(f"Native drawing does not fit its fixed slot: {smiles}: {aw:.1f} x {ah:.1f} > {width:.1f} x {height:.1f}")
    ax.set(xlim=(0, width), ylim=(0, height))
    BF.CD.draw(ax, asset, (width - aw) / 2, (height + ah) / 2)
    return ax


def scheme(fig, rect, reaction):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    width, height = rect[2] * fig.get_figwidth() * 72, rect[3] * fig.get_figheight() * 72
    asset = BF.CD.scheme(reaction)
    aw, ah = asset["width_pt"], asset["height_pt"]
    if aw > width + .01 or ah > height + .01:
        raise ValueError(f"Native scheme does not fit: {reaction}: {aw:.1f} x {ah:.1f} > {width:.1f} x {height:.1f}")
    ax.set(xlim=(0, width), ylim=(0, height))
    BF.CD.draw(ax, asset, (width - aw) / 2, (height + ah) / 2)
    return ax


def finalize(fig):
    # Tick labels are regenerated on draw, so format their source as well.
    for axis in fig.axes:
        for coordinate in (axis.xaxis, axis.yaxis):
            for formatter in (coordinate.get_major_formatter(), coordinate.get_minor_formatter()):
                if isinstance(formatter, FixedFormatter):
                    formatter.seq = [display_text(value) for value in formatter.seq]
                elif isinstance(formatter, FuncFormatter) and not getattr(formatter, "_reaction_display", False):
                    original = formatter.func
                    formatter.func = lambda value, position, original=original: display_text(original(value, position))
                    formatter._reaction_display = True
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        text.set_text(display_text(re.sub(r"(?<![A-Za-z0-9])-(?=(?:\d|\.\d))", "\u2212", text.get_text())))
        if text.get_gid() == "chemdraw":
            continue
        text.set_fontfamily("Arial")
        text.set_fontsize(9 if text.get_gid() in ("panel-letter", "panel-title", "key-number") else 8)
    fig.canvas.draw()
