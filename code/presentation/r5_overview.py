"""Editable concept scene: a reaction archive supports two kinds of choice."""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import json
import hashlib
import math
import xml.etree.ElementTree as ET

import chemdraw_vector as CD
import r5_style as S

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
WIDTH, HEIGHT = 180 * 72 / 25.4, 380
TITLE = "Which aryl bromide for the next coupling?"


def draw(p):
    root = ET.Element(f"{{{NS}}}svg", width="180mm", height=f"{HEIGHT / 72 * 25.4:.6f}mm",
                      viewBox=f"0 0 {WIDTH:.6f} {HEIGHT}", version="1.1")
    text_rows, native_placements, rank_lines, icons, signs = [], [], [], [], []
    bounds = {"archive-channel": [4, 48, 92, 304], "record-channel": [109, 44, 407, 166],
              "profile-channel": [109, 176, 407, 374], "decision-channel": [423, 44, 507, 374]}

    def element(tag, parent=root, **attrs):
        return ET.SubElement(parent, f"{{{NS}}}{tag}",
                             {key.replace("_", "-"): str(value) for key, value in attrs.items()})

    def group(gid):
        return element("g", id=gid)

    def text(x, y, value, *, key=False, color=S.INK, anchor="start", gid=None, parent=root):
        attrs = dict(x=x, y=y, fill=color, font_family="Arial", font_size=9 if key else 8,
                     text_anchor=anchor, font_weight="bold" if key else "normal")
        if gid:
            attrs["id"] = gid
        node = element("text", parent, **attrs)
        for row, content in enumerate(value.split("\n")):
            span = ET.SubElement(node, f"{{{NS}}}tspan", x=str(x), dy="0" if row == 0 else "10")
            span.text = content
        text_rows.append(dict(x=x, y=y, text=value, fontsize=9 if key else 8, gid=gid))
        return node

    def line(x1, y1, x2, y2, *, color=S.GREY, width=.6, arrow=False, gid=None, parent=root):
        attrs = dict(x1=x1, y1=y1, x2=x2, y2=y2, stroke=color, stroke_width=width)
        if gid:
            attrs["id"] = gid
        node = element("line", parent, **attrs)
        if arrow:
            angle = math.atan2(y2-y1, x2-x1)
            a, b = angle+2.65, angle-2.65
            element("path", parent, d=f"M{x2+4*math.cos(a):.3f},{y2+4*math.sin(a):.3f} L{x2},{y2} L{x2+4*math.cos(b):.3f},{y2+4*math.sin(b):.3f}",
                    fill="none", stroke=color, stroke_width=width)
        return node

    def native(smiles, orient, cx, top, gid, parent):
        asset = CD.molecule(smiles, orient)
        svg = ET.fromstring(CD.svg_text(asset))
        x0, y0, width, _ = map(float, svg.get("viewBox").split())
        factor = asset["width_pt"] / width
        node = element("g", parent, id=gid,
                       transform=f"translate({cx-asset['width_pt']/2:.6f},{top:.6f}) scale({factor:.12g}) translate({-x0},{-y0})")
        for child in svg:
            node.append(deepcopy(child))
        native_placements.append(dict(id=asset["id"], gid=gid, x=cx-asset["width_pt"]/2, y=top,
                                      width_pt=asset["width_pt"], height_pt=asset["height_pt"]))

    def hexagon(cx, cy, radius, gid, *, selected=False, color=S.ORANGE, parent=root):
        points = " ".join(f"{cx+radius*math.cos(math.pi/3*k):.3f},{cy+radius*math.sin(math.pi/3*k):.3f}" for k in range(6))
        element("polygon", parent, id=gid, points=points, fill=S.TEAL if selected else S.WHITE,
                stroke=S.TEAL if selected else color, stroke_width=.8)

    def plate_icon(x, y, gid, parent):
        node = element("g", parent, id=gid)
        element("rect", node, x=x, y=y, width=12, height=9, rx=.8,
                fill="none", stroke=S.GREY, stroke_width=.5)
        for row in range(8):
            for column in range(12):
                element("circle", node, cx=x+1.05+.9*column, cy=y+1.1+.95*row,
                        r=.3, fill=S.GREY)

    def library_icon(x, y, gid, parent):
        node = element("g", parent, id=gid)
        for row in range(2):
            for column in range(2):
                cx, cy = x+2+5*column, y+2+5*row
                points = " ".join(f"{cx+1.65*math.cos(math.pi/3*k):.3f},{cy+1.65*math.sin(math.pi/3*k):.3f}" for k in range(6))
                element("polygon", node, points=points, fill="none", stroke=S.GREY, stroke_width=.5)

    def reactivity_icon(gid, parent):
        node = element("g", parent, id=gid)
        line(282, 205, 282, 221, parent=node)
        line(282, 221, 302, 221, parent=node)
        line(284, 219, 300, 207, color=S.MUTED, parent=node)
        text(277, 199, "log k", color=S.MUTED, parent=node)
        text(300, 232, "\u03c3", color=S.MUTED, parent=node)

    for gid, fill, stroke in (("archive-channel", "#F3F4F5", "none"),
                              ("record-channel", "#F4F0F8", "none"),
                              ("profile-channel", "#EEF7F5", S.TEAL),
                              ("decision-channel", S.WHITE, "#CBD0D5")):
        node = group(gid)
        x1, y1, x2, y2 = bounds[gid]
        element("rect", node, x=x1, y=y1, width=x2-x1, height=y2-y1, rx=4,
                fill=fill, stroke=stroke, stroke_width=1 if gid == "profile-channel" else .6)
    archive, record, profile, decision = (root.find(f"{{{NS}}}g[@id='{gid}']") for gid in bounds)

    text(4, 13, TITLE, key=True, gid="overview-question")
    text(4, 28, "A yield archive answers in two ways; only the reaction's reading of structure reaches molecules that were never run.",
         gid="overview-subtitle")
    counts = p.tab("Figure1_archive_counts.csv").iloc[0]
    text(48, 68, "Reaction archive", key=True, anchor="middle", parent=archive)
    plate = element("g", archive, id="archive-plate")
    for row in range(8):
        for col in range(6):
            element("circle", plate, cx=22+10*col, cy=98+10*row, r=2.8, fill="#CCD2D7")
    for y, value in zip((190, 203, 216), (f"{int(counts.experiments):,} experiments",
                        f"{int(counts.transformations)} transformations", f"{int(counts.substrates):,} substrates")):
        text(48, y, value, anchor="middle", parent=archive)
    plate_icon(14, 249, "archive-prior-work-icon", archive)
    text(30, 257, "HTE archives", parent=archive)
    text(14, 276, "King-Smith 2024\nAhlbrecht 2026", color=S.MUTED, parent=archive)
    line(93, 203, 100, 203)
    line(100, 110, 100, 267)
    line(100, 110, 108, 110, color=S.PURPLE, arrow=True)
    line(100, 267, 108, 267, color=S.TEAL, arrow=True)

    text(117, 59, "What each molecule did", key=True, color=S.PURPLE, gid="record-title", parent=record)
    library_icon(210, 72, "scope-prior-work-icon", record)
    text(225, 79, "Scope tables; informer libraries", parent=record)
    text(225, 90, "Kutchukian 2016; Rana 2024", color=S.MUTED, parent=record)
    # These five paths illustrate rank changes; none represents a measured molecule.
    order = (3, 0, 4, 1, 2)
    for i, target in enumerate(order):
        y1, y2 = 98+9*i, 98+9*target
        gid = f"schematic-rank-{i+1}"
        line(130, y1, 185, y2, color=S.PURPLE if i == 0 else "#B8A8CD",
             width=1 if i == 0 else .6, gid=gid, parent=record)
        hexagon(125, y1, 3.5, f"rank-start-{i+1}", color=S.PURPLE, parent=record)
        hexagon(190, y2, 3.5, f"rank-end-{i+1}", color=S.PURPLE, parent=record)
        rank_lines.append(dict(gid=gid, schematic=True, x1=130, y1=y1, x2=185, y2=y2))
    text(158, 149, "New partners /\nnew reaction", anchor="middle", color=S.MUTED, parent=record)
    ranks = p.tab("Figure3_shared_molecules.csv")
    example = ranks.loc[ranks.SMILES.eq("O=Cc1cccc(Br)n1")].iloc[0]
    native(example.SMILES, None, 239, 103, "true-rank-example", record)
    text(239, 148, f"rank {int(example.campaign_1_rank)} \u2192 {int(example.campaign_2_rank)}",
         color=S.PURPLE, key=True, anchor="middle", gid="true-rank-label", parent=record)
    text(294, 111, "Partners and\nreaction matter", parent=record)
    aggregate = p.tab("Figure2a_aggregate_summary.csv").set_index("category").estimate
    text(294, 142, " \u2192 ".join(f"{aggregate[k]:.2f}" for k in ("conditions", "partners", "reactions")),
         key=True, color=S.PURPLE, gid="rank-headlines", parent=record)
    text(294, 157, "Fig. 2", color=S.MUTED, parent=record)

    text(117, 191, "How the reaction reads structure", key=True, color=S.TEAL, gid="profile-title", parent=profile)
    text(117, 205, "this work", color=S.TEAL, parent=profile)
    reactivity_icon("reactivity-prior-work-icon", profile)
    text(309, 205, "Hammett \u03c3/\u03c1; Mayr N/E", parent=profile)
    text(309, 216, "1937; 1994", color=S.MUTED, parent=profile)
    text(117, 222, "Buchwald\u2013Hartwig profile", parent=profile)
    features = [("halogen next to ring N", "Brc1ccccn1", "Ring N\nbeside Br"),
                ("substituent beside the halogen", "Cc1ccccc1Br", "Ortho\nsubstituent"),
                ("five-membered ring at the halogen", "Brc1ccsc1", "Five-membered\nring")]
    full = p.tab("Figure3_structural_profiles.csv")
    full = full.loc[full.domain.eq("full libraries")].set_index("feature")
    for i, (feature, smiles, label) in enumerate(features):
        cx = 139+54*i
        value = float(full.loc[feature, "campaign_1_pp"])
        sign = "+" if value > 0 else "\u2212" if value < 0 else "0"
        native(smiles, -1, cx, 242, f"profile-motif-{i+1}", profile)
        text(cx, 236, sign, key=True, color=S.BLUE if value > 0 else S.RED,
             anchor="middle", gid=f"motif-sign-{i+1}", parent=profile)
        text(cx, 296, label, anchor="middle", parent=profile)
        signs.append(dict(feature=feature, campaign=1, value_pp=value, sign=sign))
    line(272, 262, 289, 262, color=S.TEAL, width=1, arrow=True, parent=profile)
    text(344, 239, "Never run", color=S.ORANGE, anchor="middle", parent=profile)
    for i in range(6):
        cx, cy = 310+18*(i%3), 252+17*(i//3)
        selected = i in (0, 2, 4)
        gid = f"unmeasured-icon-{i+1}"
        hexagon(cx, cy, 6, gid, selected=selected, parent=profile)
        icons.append(dict(gid=gid, selected=selected, schematic=True))
    line(354.5, 252, 361, 252, color=S.TEAL, parent=profile)
    text(365, 255, "chosen", color=S.TEAL, gid="r12-Figure1-chosen-legend", parent=profile)
    text(300, 291, f"{p.values['selected_gain_pp']:.1f} of {p.values['available_gain_pp']:.1f} pp",
         key=True, color=S.TEAL, gid="selection-headline", parent=profile)
    text(300, 305, "Fig. 3", color=S.MUTED, parent=profile)
    line(117, 314, 399, 314, color="#C5DED8", parent=profile)
    native("Brc1ccccn1", -1, 139, 325, "reaction-dependent-motif", profile)
    contrasts = p.tab("Figure4a_feature_matrix.csv")
    contrasts = contrasts.loc[contrasts.feature.eq("halogen next to ring N")].set_index("reaction")
    value_column = next(column for column in ("delta_pp", "mean_difference_pp", "effect_pp", "difference_pp") if column in contrasts)
    text(169, 332, "Ring N beside Br", parent=profile)
    text(169, 350, f"Heck {S.signed(contrasts.loc['Heck', value_column])} pp", key=True, color=S.RED,
         gid="heck-headline", parent=profile)
    text(169, 365, f"Ullmann ether {S.signed(contrasts.loc['Ullmann', value_column])} pp", key=True, color=S.BLUE,
         gid="ullmann-headline", parent=profile)
    text(371, 365, "Fig. 4", color=S.MUTED, parent=profile)

    text(465, 60, "Which history?", key=True, anchor="middle", parent=decision)
    rules = [("decision-measured", 104, "Measured here", "Its record", S.PURPLE),
             ("decision-unmeasured", 214, "Never run", "The reaction's\nprofile", S.TEAL),
             ("decision-borrow", 292, "Another reaction\nranks shared\nmolecules alike", "Borrow its\nrecord", S.PURPLE)]
    for gid, y, question, answer, color in rules:
        node = element("g", decision, id=gid)
        rows = question.count("\n")+1
        text(465, y, question, anchor="middle", parent=node)
        line(465, y+10*rows, 465, y+10*rows+9, color=color, arrow=True, parent=node)
        text(465, y+10*rows+25, answer, key=True, color=color, anchor="middle", parent=node)
    text(465, 367, "Fig. 5", color=S.MUTED, anchor="middle", parent=decision)
    line(408, 112, 421, 112, color=S.PURPLE, arrow=True)
    line(408, 257, 421, 257, color=S.TEAL, arrow=True)

    destination = p.out / "figures/Figure1.svg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    p.qa["Figure1"] = dict(width_mm=180., height_mm=HEIGHT/72*25.4, min_text_pt=8., panel_titles=[TITLE],
                           left_titles=[TITLE], undersized_text=[], outside_canvas_text=[], text_overlap_pairs=0,
                           word_placement_scale=1., word_placed_min_text_pt=8., editable_vector_svg=True)
    (p.out / "overview_text.json").write_text(json.dumps(text_rows, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    metadata = dict(coordinate_system="upper-left SVG/PDF points", panel_bounds=bounds,
                    svg_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), page_size_pt=[WIDTH, HEIGHT],
                    native_assets=native_placements, schematic_rank_lines=rank_lines,
                    schematic_unmeasured_icons=icons, motif_signs=signs, feature_colour_tiles=[],
                    true_rank_line_count=0, example=dict(smiles=example.SMILES,
                    campaign_1_rank=int(example.campaign_1_rank), campaign_2_rank=int(example.campaign_2_rank)))
    (p.out / "overview_geometry.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return destination


def render(svg):
    """Render editable SVG and inspect the actual PDF before raster export."""
    import pymupdf
    from matplotlib import font_manager
    from reportlab import rl_config
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.graphics import renderPDF
    from svglib.fonts import register_font
    from svglib.svglib import svg2rlg
    rl_config.invariant = 1
    for name, weight in (("Arial", "normal"), ("Arial-Bold", "bold")):
        path = font_manager.findfont(font_manager.FontProperties(family="Arial", weight=weight), fallback_to_default=False)
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, path))
        register_font("Arial", path, weight=weight, rlgFontName=name)
    pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")
    drawing = svg2rlg(str(svg))
    renderPDF.drawToFile(drawing, str(svg.with_suffix(".pdf")), title=TITLE)
    with pymupdf.open(svg.with_suffix(".pdf")) as document:
        spans = [span for block in document[0].get_text("dict")["blocks"] if block["type"] == 0
                 for line in block["lines"] for span in line["spans"]]
        if not any(span["text"] == TITLE and "Bold" in span["font"] for span in spans):
            raise ValueError("The SVG renderer lost the overview's Arial bold font mapping")
        overlaps = []
        for first, second in combinations(spans, 2):
            intersection = pymupdf.Rect(first["bbox"]) & pymupdf.Rect(second["bbox"])
            if intersection.width > .2 and intersection.height > .2:
                overlaps.append([first["text"], second["text"]])
        outside = [span["text"] for span in spans if not document[0].rect.contains(pymupdf.Rect(span["bbox"]))]
        if overlaps or outside:
            raise ValueError(f"Overview rendered text collision or overflow: {overlaps}; {outside}")
        document[0].get_pixmap(matrix=pymupdf.Matrix(600/72, 600/72), alpha=False).save(svg.with_suffix(".png"))
        return dict(text_overlap_pairs=len(overlaps), outside_canvas_text=outside,
                    actual_pdf_text_spans_checked=len(spans))
