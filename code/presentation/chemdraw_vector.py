"""Replay ChemDraw SVG objects at their original physical size in Matplotlib."""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from rdkit import Chem

ASSETS = Path(__file__).resolve().parents[1] / "assets/ChemDraw"
NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"


def _numbers(value):
    return [float(v) for v in re.findall(NUMBER, value)]


def _transform(value):
    result = np.eye(3)
    for kind, content in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", value or ""):
        values = _numbers(content)
        matrix = np.eye(3)
        if kind == "matrix":
            a, b, c, d, e, f = values
            matrix = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
        elif kind == "translate":
            matrix[:2, 2] = values[0], values[1] if len(values) > 1 else 0
        elif kind == "scale":
            matrix[0, 0], matrix[1, 1] = values[0], values[-1]
        elif kind == "rotate":
            angle = math.radians(values[0])
            c, s = math.cos(angle), math.sin(angle)
            matrix[:2, :2] = [[c, -s], [s, c]]
            if len(values) == 3:
                center = np.array(values[1:])
                matrix[:2, 2] = center - matrix[:2, :2] @ center
        else:
            raise ValueError(f"Unsupported ChemDraw SVG transform: {kind}")
        result = result @ matrix
    return result


def _path(value):
    tokens = re.findall(rf"[A-Za-z]|{NUMBER}", value)
    vertices, codes = [], []
    index, command = 0, None
    current, start = np.zeros(2), np.zeros(2)
    while index < len(tokens):
        if tokens[index].isalpha():
            command, index = tokens[index], index + 1
            if command.upper() == "Z":
                vertices.append(start.copy())
                codes.append(MPath.CLOSEPOLY)
                current = start.copy()
                continue
        upper = command.upper()
        if upper not in ("M", "L", "H", "V", "C", "Q"):
            raise ValueError(f"Unsupported ChemDraw SVG path command: {command}")
        count = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "Q": 4}[upper]
        values = np.array([float(v) for v in tokens[index:index + count]])
        index += count
        relative = command.islower()
        if upper in ("H", "V"):
            point = current.copy()
            dimension = 0 if upper == "H" else 1
            point[dimension] = values[0] + (current[dimension] if relative else 0)
            vertices.append(point)
            codes.append(MPath.LINETO)
            current = point
        else:
            points = values.reshape(-1, 2) + (current if relative else 0)
            vertices.extend(points)
            codes.extend([{"M": MPath.MOVETO, "L": MPath.LINETO,
                           "C": MPath.CURVE4, "Q": MPath.CURVE3}[upper]] * len(points))
            current = points[-1]
            if upper == "M":
                start = current.copy()
                command = "l" if relative else "L"
    return MPath(np.array(vertices), codes)


def _style(element, inherited):
    style = dict(inherited)
    style.update(element.attrib)
    for pair in element.get("style", "").split(";"):
        if ":" in pair:
            key, value = pair.split(":", 1)
            style[key.strip()] = value.strip()
    return style


@lru_cache(maxsize=None)
def manifest():
    path = ASSETS / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"ChemDraw masters are required: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def molecule(smiles, orient=None):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
    orientation = None if orient is None else int(orient)
    matches = [row for row in manifest()["molecules"]
               if row["canonical_smiles"] == canonical and row.get("orient") == orientation]
    if len(matches) != 1:
        raise ValueError(f"Expected one ChemDraw master for {canonical}, orient={orientation}; found {len(matches)}")
    return matches[0]


def scheme(reaction):
    matches = [row for row in manifest()["schemes"] if row["reaction"] == reaction]
    if len(matches) != 1:
        raise ValueError(f"Expected one ChemDraw scheme for {reaction}; found {len(matches)}")
    return matches[0]


def svg_text(asset):
    return (ASSETS / asset["svg_path"]).read_text(encoding="utf-8")


def draw(ax, asset, left, top):
    """Place an asset by its upper-left corner; axes coordinates must be points."""
    root = ET.fromstring(svg_text(asset))
    x0, y0, width, height = _numbers(root.get("viewBox"))
    width_pt, height_pt = float(asset["width_pt"]), float(asset["height_pt"])
    sx, sy = width_pt / width, height_pt / height
    if not math.isclose(sx, sy, rel_tol=1e-5):
        raise ValueError(f"ChemDraw asset would be distorted: {asset['svg_path']}")
    physical = np.array([[sx, 0, left - sx * x0], [0, -sy, top + sy * y0], [0, 0, 1]])

    def walk(element, parent, inherited):
        tag = element.tag.rsplit("}", 1)[-1]
        style = _style(element, inherited)
        if style.get("display") == "none" or float(style.get("opacity", "1")) == 0:
            return
        matrix = parent @ _transform(element.get("transform"))
        final = physical @ matrix
        if tag == "path":
            path = _path(element.get("d", ""))
            vertices = np.column_stack((path.vertices, np.ones(len(path.vertices)))) @ final.T
            fill, stroke = style.get("fill", "black"), style.get("stroke", "none")
            stroke_scale = np.linalg.norm(final[:2, 0])
            ax.add_patch(PathPatch(MPath(vertices[:, :2], path.codes),
                                   facecolor=fill, edgecolor=stroke,
                                   lw=float(_numbers(style.get("stroke-width", "1"))[0]) * stroke_scale,
                                   capstyle="butt", joinstyle="miter", gid="chemdraw"))
        elif tag == "text":
            if list(element):
                raise ValueError(f"Nested ChemDraw text requires explicit handling: {asset['svg_path']}")
            label = (element.text or "").strip()
            if label:
                x, y = _numbers(element.get("x", "0"))[0], _numbers(element.get("y", "0"))[0]
                point = final @ np.array([x, y, 1])
                fontsize = _numbers(style["font-size"])[0] * np.linalg.norm(final[:2, 0])
                angle = math.degrees(math.atan2(final[1, 0], final[0, 0]))
                ax.text(point[0], point[1], label, fontsize=fontsize,
                        family=style.get("font-family", "Arial"),
                        weight=style.get("font-weight", "normal"),
                        style=style.get("font-style", "normal"),
                        color=style.get("fill", "black"), va="baseline",
                        ha={"start": "left", "middle": "center", "end": "right"}.get(style.get("text-anchor"), "left"),
                        rotation=angle, rotation_mode="anchor", gid="chemdraw")
        elif tag not in ("svg", "g", "title", "desc", "metadata", "defs"):
            raise ValueError(f"Unsupported ChemDraw SVG element: {tag}")
        if tag not in ("text", "defs", "metadata"):
            for child in element:
                walk(child, matrix, style)

    walk(root, np.eye(3), {})
    return width_pt, height_pt
