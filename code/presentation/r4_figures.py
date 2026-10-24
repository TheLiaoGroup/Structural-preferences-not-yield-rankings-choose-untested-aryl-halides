"""R4 Article figures and Extended Data, using released numerical producers.

Legacy panel tables remain reproducible intermediates. This module gives their
final Article locations and draws native-size ChemDraw vectors, never bitmaps.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import plot_figures as BF
import figures_1_4 as F14
import figures_2_3 as F23

BLUE, ORANGE, TEAL, RED, PURPLE = BF.BLUE, BF.ORANGE, BF.TEAL, BF.RED, BF.PURPLE
INK, GREY, LIGHT, MUTED = BF.INK, BF.GREY, BF.LIGHT, BF.MUTED
NAMES, SHORT = BF.NAMES, BF.SHORT
ROOT = Path(__file__).resolve().parents[2]
TITLES = {
    "Figure2": ["The record repeats, then fades", "Few productive partners set an SNAr ranking", "The same amines, opposite orders"],
    "Figure3": ["What each reaction rewards and penalizes", "Twenty new conditions, same reading", "One 2-bromopyridine, four readings"],
    "Figure4": ["A molecule with a record: use the record", "No record: the same reaction's profile", "Borrowing works when the two reactions agree"],
    "ExtendedDataFig1": ["Fifteen reaction arrays and their twenty-condition compositions"],
    "ExtendedDataFig2": ["Amide pairs ordered by their partners' contributions"],
    "ExtendedDataFig3": ["Every reaction and partner"],
    "ExtendedDataFig4": ["Changing reaction costs more than changing partners", "Comparisons without a positive partner benchmark"],
    "ExtendedDataFig5": ["Pairwise rank agreement between reactions"],
    "ExtendedDataFig6": ["Recognizing a second Buchwald–Hartwig campaign", "Structural profiles under new conditions", "Structural similarity between reactions"],
    "ExtendedDataFig7": ["New partners: the profile usually holds better", "Structure classes against random groups of the same size"],
    "ExtendedDataFig8": ["Held-out amide pairs", "Prediction gains across fifteen reaction arrays"],
    "SupplementaryFig1": ["Leaving one reaction out", "Other yield scales"],
}
FEATURES = [
    ("halogen next to ring N", "ring N\nbeside X", "Brc1ccccn1"),
    ("N–H", "N–H\nanywhere", "Brc1ccc2[nH]ncc2c1"),
    ("five-membered ring at the halogen", "five-membered\nring", "Brc1ccsc1"),
    ("substituent beside the halogen", "ortho\nsubstituent", "Cc1ccccc1Br"),
    ("fused ring at the halogen", "fused\nring", "Brc1cccc2ccccc12"),
    ("heteroaromatic ring at the halogen", "heteroaryl\nat X", "Brc1cccnc1"),
    ("electron-withdrawing group", "EWG", "N#Cc1ccc(Br)cc1"),
    ("electron-donating group", "EDG", "COc1ccc(Br)cc1"),
]

# (producer, source CSV, final CSV, workbook sheet). Values are copied verbatim.
SOURCE_MAP = [
    ("source", "Figure1c_pooled_portability.csv", "Figure2a_pooled_portability.csv", "Fig2a"),
    ("source", "Figure2b_SNAr_partitions.csv", "Figure2b_SNAr_partitions.csv", "Fig2b"),
    ("source", "Figure2d_BH_SNAr_amines.csv", "Figure2c_BH_SNAr_amines.csv", "Fig2c_amines"),
    ("source", "Figure2d_amine_class_summary.csv", "Figure2c_amine_class_summary.csv", "Fig2c_classes"),
    ("readings", "aryl_feature_matrix.csv", "Figure3a_feature_matrix.csv", "Fig3a_cells"),
    ("readings", "aryl_reaction_order.csv", "Figure3a_reaction_order.csv", "Fig3a_order"),
    ("readings", "paired_feature_contrasts.csv", "Figure3b_paired_feature_contrasts.csv", "Fig3b_cells"),
    ("readings", "paired_feature_summary.csv", "Figure3b_paired_feature_summary.csv", "Fig3b_summary"),
    ("readings", "methyl_2_bromoisonicotinate.csv", "Figure3c_methyl_2_bromoisonicotinate.csv", "Fig3c"),
    ("selection", "partner_selection_by_profile.csv", "Figure4a_partner_selection.csv", "Fig4a"),
    ("source", "Figure3d_selection_without_record.csv", "Figure4b_selection_without_record.csv", "Fig4b"),
    ("source", "Figure4b_selection_vs_agreement.csv", "Figure4c_selection_vs_agreement.csv", "Fig4c_selection"),
    ("source", "Figure4c_Heck_Suzuki_molecules.csv", "Figure4c_Heck_Suzuki_molecules.csv", "Fig4c_molecules"),
    ("source", "Figure1a_reactions.csv", "ExtendedDataFig1_reaction_designs.csv", "EDFig1_designs"),
    ("source", "ExtendedDataFig1_condition_composition.csv", "ExtendedDataFig1_condition_composition.csv", "EDFig1_conditions"),
    ("source", "ExtendedDataFig2_amide_matrix.csv", "ExtendedDataFig2_amide_matrix.csv", "EDFig2"),
    ("source", "Figure2a_portability_by_set.csv", "ExtendedDataFig3_portability_by_set.csv", "EDFig3"),
    ("source", "ExtendedDataFig3_defined_benchmark.csv", "ExtendedDataFig4a_defined_benchmark.csv", "EDFig4a_benchmark"),
    ("source", "ExtendedDataFig3_pooled_shortfall.csv", "ExtendedDataFig4a_pooled_shortfall.csv", "EDFig4a_shortfall"),
    ("source", "ExtendedDataFig4_undefined_benchmark.csv", "ExtendedDataFig4b_undefined_benchmark.csv", "EDFig4b"),
    ("source", "ExtendedDataFig6_reaction_map.csv", "ExtendedDataFig5_reaction_map.csv", "EDFig5"),
    ("source", "Figure3a_campaign_recognition.csv", "ExtendedDataFig6a_campaign_recognition.csv", "EDFig6a"),
    ("source", "ExtendedDataFig7ab_profile_comparisons.csv", "ExtendedDataFig6bc_profile_comparisons.csv", "EDFig6bc"),
    ("source", "Figure3b_partner_change.csv", "ExtendedDataFig7a_partner_change.csv", "EDFig7a"),
    ("source", "Figure3c_random_grouping.csv", "ExtendedDataFig7b_random_grouping.csv", "EDFig7b"),
    ("source", "Figure4a_heldout_amide_pairs.csv", "ExtendedDataFig8a_heldout_amide_pairs.csv", "EDFig8a_pairs"),
    ("source", "Figure4a_metrics.csv", "ExtendedDataFig8a_metrics.csv", "EDFig8a_metrics"),
    ("source", "ExtendedDataFig8_prediction_all_reactions.csv", "ExtendedDataFig8b_prediction_all_reactions.csv", "EDFig8b"),
    ("source", "ExtendedDataFig5a_leave_one_reaction_out.csv", "SupplementaryFig1a_leave_one_reaction_out.csv", "SIFig1a"),
    ("source", "ExtendedDataFig5b_outcome_scale.csv", "SupplementaryFig1b_outcome_scale.csv", "SIFig1b"),
    ("tables", "ExtendedDataTable1.csv", "ExtendedDataTable1.csv", "EDTable1"),
]


def signed(value, decimals=0):
    rounded = round(float(value), decimals)
    return (f"{0:.{decimals}f}" if rounded == 0 else f"{rounded:+.{decimals}f}").replace("-", "−")


def header(fig, name, letter, x, y):
    return BF.figure_header(fig, x, y, letter, TITLES[name][ord(letter) - ord("a")])


def native(fig, rect, smiles, orient=None):
    ax = fig.add_axes(rect)
    ax.set_axis_off()
    w, h = rect[2] * fig.get_figwidth() * 72, rect[3] * fig.get_figheight() * 72
    asset = BF.CD.molecule(smiles, orient)
    aw, ah = float(asset["width_pt"]), float(asset["height_pt"])
    if aw > w or ah > h:
        raise ValueError(f"Native ChemDraw drawing exceeds its slot: {smiles}: {aw}x{ah} > {w}x{h}")
    ax.set(xlim=(0, w), ylim=(0, h))
    BF.CD.draw(ax, asset, (w - aw) / 2, (h + ah) / 2)
    return ax


class Presentation:
    def __init__(self, results, out, readings=None):
        self.results, self.out = Path(results), Path(out)
        self.readings = Path(readings) if readings else self.results / "reaction_readings"
        self.source = self.results / "presentation/source_data"
        self.panels = self.results / "presentation/panel_data"
        qa_path = self.out / "figure_qa.json"
        self.qa = json.loads(qa_path.read_text(encoding="utf-8")) if qa_path.exists() else {}

    def tab(self, name):
        return pd.read_csv(self.out / "source_data" / name)

    def sources(self):
        directory = self.out / "source_data"
        directory.mkdir(parents=True, exist_ok=True)
        bases = {"source": self.source, "readings": self.readings,
                 "selection": self.results / "sensitivity/structure_for_selection", "tables": self.results / "presentation/tables"}
        sheets = {}
        for producer, old, new, sheet in SOURCE_MAP:
            shutil.copyfile(bases[producer] / old, directory / new)
            sheets[sheet] = directory / new
        return sheets

    def export(self, name, fig):
        BF.finalize(fig)
        actual = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
        if actual != TITLES[name]:
            raise ValueError(f"{name}: panel titles differ: {actual!r}")
        overlaps = F23.overlaps(fig)
        if overlaps:
            raise ValueError(f"{name}: overlapping rendered text: {overlaps}")
        qa = BF.export_figure(fig, self.out / "figures" / name, creator="r4_figures.py")
        qa["text_overlap_pairs"] = len(overlaps)
        qa["panel_titles"] = actual
        if hasattr(fig, "_matrix_alignment"):
            matrix_ax, headers = fig._matrix_alignment
            errors = [abs(slot.get_window_extent().x0 + slot.get_window_extent().width / 2 - matrix_ax.transData.transform((j, 0))[0]) for j, slot in enumerate(headers)]
            if max(errors) > .5:
                raise ValueError(f"{name}: matrix and structure headers misaligned by {max(errors):g} pixels")
            qa["matrix_header_max_alignment_error_px"] = max(errors)
        placement_scale = min(1., 6.8 * 25.4 / qa["width_mm"], 7.6 * 25.4 / qa["height_mm"])
        qa["word_placement_scale"] = placement_scale
        qa["word_placed_min_text_pt"] = qa["min_text_pt"] * placement_scale
        if qa["word_placed_min_text_pt"] < 5:
            raise ValueError(f"{name}: text falls below 5 pt at final Word size: {qa['word_placed_min_text_pt']}")
        self.qa[name] = qa
        plt.close(fig)

    def figure2(self):
        name = "Figure2"
        fig = plt.figure(figsize=(7, 6.6))
        ladder = self.tab("Figure2a_pooled_portability.csv")
        ax = fig.add_axes([.105, .62, .385, .27])
        levels = [("conditions", "1st set, halves", "new\nconditions", TEAL),
                  ("conditions", "1st vs 2nd set", "second\ncondition set", TEAL),
                  ("partners", "1st set, halves", "new\npartners", PURPLE),
                  ("reactions", "1st set", "another\nreaction", RED)]
        for i, (kind, label, _, color) in enumerate(levels):
            row = ladder[ladder.kind.eq(kind) & ladder.label.eq(label)].iloc[0]
            ax.plot([i, i], [row.ci_low, row.ci_high], color=color, lw=1.2)
            ax.plot(i, row.estimate, "o", color=color, ms=5)
            ax.text(i + .13, row.estimate, f"{row.estimate:.2f}", color=color, fontsize=7, weight="bold", va="center")
        ax.axhline(0, color=GREY, lw=.6)
        ax.set(xticks=range(4), xticklabels=[x[2] for x in levels], ylim=(-.1, 1.03), xlim=(-.3, 3.6), ylabel="Rank agreement")
        BF.clean(ax)
        header(fig, name, "a", .025, .955)

        p = self.tab("Figure2b_SNAr_partitions.csv")
        cond = p.loc[p.agreement.eq("condition halves"), "rank_agreement"].item()
        partner = p.loc[p.agreement.eq("partner halves") & ~p.partition.eq("fixed"), "rank_agreement"]
        assert len(partner) == 20
        ax = fig.add_axes([.69, .67, .28, .18])
        jitter = np.where(np.arange(20) % 2 == 0, .065, -.065)
        ax.scatter(np.sort(partner), jitter, color=PURPLE, s=14, lw=0, alpha=.75)
        ax.plot([partner.median()] * 2, [-.25, .25], color=PURPLE, lw=1.3)
        ax.plot(cond, 1, "D", color=TEAL, ms=5)
        ax.text(partner.median(), -.36, f"median {partner.median():.2f}", ha="center", va="top", fontsize=6.3, color=PURPLE)
        ax.text(cond, 1.3, f"{cond:.2f}", ha="center", fontsize=6.3, color=TEAL)
        ax.set(yticks=[0, 1], yticklabels=["new partners\n(20 splits)", "new conditions"], xlim=(-.05, 1.05), ylim=(-.7, 1.55), xlabel="Rank agreement")
        BF.clean(ax)
        header(fig, name, "b", .535, .955)

        amines = self.tab("Figure2c_BH_SNAr_amines.csv")
        flag = BF.bool_series(amines.primary_arylamine)
        ax = fig.add_axes([.105, .10, .385, .335])
        ax.axhline(0, color=GREY, lw=.5); ax.axvline(0, color=GREY, lw=.5)
        for selection, color in ((~flag, GREY), (flag, RED)):
            ax.scatter(amines.x_contribution_pp[selection], amines.y_contribution_pp[selection], s=13, color=color, lw=0)
        for row in amines[amines.drawn.fillna(0).gt(0)].itertuples():
            ax.annotate(str(int(row.drawn)), (row.x_contribution_pp, row.y_contribution_pp), xytext=(3, 3), textcoords="offset points", fontsize=7, weight="bold")
        ax.set(xlabel="Buchwald–Hartwig (pp)", ylabel="SNAr (pp)")
        ax.margins(.1); BF.clean(ax)
        fig.text(.105, .47, f"{len(amines):,} shared amines; rank agreement {signed(amines.rho_obs.iloc[0], 2)}\n{int(flag.sum()):,} primary (hetero)arylamines (red)", fontsize=7)
        header(fig, name, "c", .025, .535)
        drawn = amines[amines.drawn.fillna(0).gt(0)].sort_values("drawn")
        fig.text(.56, .43, "Buchwald–Hartwig / SNAr contributions (pp)", fontsize=6.2, color=MUTED)
        for i, row in enumerate(drawn.itertuples()):
            x, y = .55 + (i % 2) * .22, .255 - (i // 2) * .19
            native(fig, [x, y, .21, .15], row.mol, F23.ORIENT.get(row.mol, 0))
            fig.text(x + .105, y - .01, f"{int(row.drawn)}  {signed(row.x_contribution_pp)} / {signed(row.y_contribution_pp)}", ha="center", fontsize=7)
        return fig

    def figure3(self):
        name = "Figure3"
        cells = self.tab("Figure3a_feature_matrix.csv")
        order = self.tab("Figure3a_reaction_order.csv").sort_values("row_order")
        assert len(order) == 9 and len(cells) == 72
        reactions = order.reaction.tolist()
        features = [x[0] for x in FEATURES]
        matrix = cells.pivot(index="reaction", columns="feature", values="difference_pp").reindex(index=reactions, columns=features).to_numpy()
        pvalues = cells.pivot(index="reaction", columns="feature", values="mann_whitney_p").reindex(index=reactions, columns=features).to_numpy()
        fig = plt.figure(figsize=(7, 7.5))
        header(fig, name, "a", .015, .973)
        # Each slot is measured in points; native asset dimensions are never scaled.
        slots = []
        for j, (_, label, smiles) in enumerate(FEATURES):
            x = .235 + j * .09
            slots.append(native(fig, [x, .79, .09, .155], smiles, -1))
            fig.text(x + .045, .785, label, ha="center", va="top", fontsize=6)
        ax = fig.add_axes([.235, .395, .72, .325])
        fig._matrix_alignment = ax, slots
        limit = float(np.nanmax(np.abs(matrix)))
        cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("white")
        ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit))
        for i in range(9):
            for j in range(8):
                ax.text(j, i, "·" if not np.isfinite(matrix[i, j]) else signed(matrix[i, j]), ha="center", va="center", fontsize=7,
                        weight="bold" if pvalues[i, j] < .05 else "normal",
                        color=MUTED if not np.isfinite(matrix[i, j]) else "white" if abs(matrix[i, j]) > .65 * limit else INK)
        labels = [NAMES[r] for r in reactions]
        ax.set(xticks=[], yticks=range(9), yticklabels=labels)
        for i, (label, metal) in enumerate(zip(ax.get_yticklabels(), order.metal)):
            label.set_color(BLUE if metal == "Pd" else ORANGE)
            if i and order.metal.iloc[i] != order.metal.iloc[i - 1]:
                ax.axhline(i - .5, color="white", lw=2.2)
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.text(.235, .365, "Yield difference (pp): blue, positive; red, negative. Bold, P < 0.05; ·, not testable.\nFirst condition set. Row labels: palladium (blue), other or no metal (ochre).", fontsize=6, color=MUTED)

        paired, summary = self.tab("Figure3b_paired_feature_contrasts.csv"), self.tab("Figure3b_paired_feature_summary.csv").iloc[0]
        ax = fig.add_axes([.105, .075, .30, .205])
        x, y = paired.first_difference_pp, paired.second_difference_pp
        lo, hi = float(min(x.min(), y.min()) - 2), float(max(x.max(), y.max()) + 2)
        ax.scatter(x, y, s=10, color=BLUE, lw=0)
        ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=.7)
        ax.axhline(0, color=GREY, lw=.4); ax.axvline(0, color=GREY, lw=.4)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="First condition set (pp)", ylabel="Second condition set (pp)")
        ax.text(.03, .97, f"{int(summary.paired_cells):,} reaction–feature cells\nsame sign: {summary.same_sign_fraction:.0%}; r = {summary.pearson_r:.2f}", transform=ax.transAxes, va="top", fontsize=6)
        BF.clean(ax); header(fig, name, "b", .015, .31)

        rows = self.tab("Figure3c_methyl_2_bromoisonicotinate.csv")
        header(fig, name, "c", .49, .31)
        native(fig, [.49, .09, .13, .17], rows.SMILES.iloc[0], -1)
        fig.text(.55, .065, "methyl 2-bromo-\nisonicotinate", ha="center", fontsize=6)
        ax = fig.add_axes([.80, .08, .12, .20])
        ax.barh(range(len(rows)), rows.contribution_pp, height=.55, color=[BLUE if v > 0 else RED for v in rows.contribution_pp])
        ax.set(yticks=range(len(rows)), yticklabels=[SHORT[r] for r in rows.reaction], xlabel="Contribution\n(pp)")
        ax.invert_yaxis(); ax.axvline(0, color=GREY, lw=.5); BF.clean(ax)
        for i, row in enumerate(rows.itertuples()):
            ax.text(1.05, i, f"{int(row.rank)}/{int(row.total_molecules)}", transform=ax.get_yaxis_transform(), va="center", fontsize=6)
        return fig

    def figure4(self):
        name = "Figure4"
        fig = plt.figure(figsize=(7, 7))
        rows = self.tab("Figure4a_partner_selection.csv").sort_values("median_ranking_pp", ascending=False)
        assert len(rows) == 7 and int((rows.median_ranking_pp > rows.median_profile_pp).sum()) == 5
        ax = fig.add_axes([.22, .61, .255, .28])
        for i, row in enumerate(rows.itertuples()):
            ax.plot([row.median_profile_pp, row.median_ranking_pp], [i, i], color=GREY, lw=1)
        ax.plot(rows.median_ranking_pp, range(len(rows)), "o", color=PURPLE, ms=4.5, label="yield record")
        ax.plot(rows.median_profile_pp, range(len(rows)), "D", color=TEAL, ms=4, label="structural profile")
        ax.set(yticks=range(len(rows)), yticklabels=[NAMES[r] for r in rows.reaction], xlabel="Gain with new partners (pp)")
        ax.invert_yaxis(); ax.axvline(0, color=GREY, lw=.5); BF.clean(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(.45, -.18), ncol=2, frameon=False, fontsize=6, columnspacing=.7)
        header(fig, name, "a", .015, .955)

        unseen = self.tab("Figure4b_selection_without_record.csv").sort_values("top_ten_gain_pp", ascending=False)
        ax = fig.add_axes([.735, .61, .24, .28])
        ax.barh(range(len(unseen)), unseen.top_ten_gain_pp, height=.58, color=[TEAL if r == "Buchwald-Hartwig" else GREY for r in unseen.source])
        ax.set(yticks=range(len(unseen)), yticklabels=[SHORT[r] for r in unseen.source], xlabel="Gain among 52 new\naryl bromides (pp)")
        ax.invert_yaxis(); BF.clean(ax)
        available = float(unseen.gain_available_pp.iloc[0])
        ax.axvline(available, color=MUTED, ls="--", lw=.7)
        ax.text(available, -.8, f"available {available:.1f}", ha="right", fontsize=6)
        header(fig, name, "b", .525, .955)

        borrow = self.tab("Figure4c_selection_vs_agreement.csv")
        ax = fig.add_axes([.105, .10, .38, .31])
        for side, color, label in (("elec", BLUE, "electrophiles"), ("nuc", ORANGE, "nucleophiles")):
            subset = borrow[borrow.side.eq(side)]
            ax.scatter(subset.rho_obs, subset.source_gain_pp, color=color, s=15, lw=0, label=label)
        ax.axhline(0, color=GREY, lw=.6)
        for ra, rb, side, label, xytext in (("Heck", "Suzuki", "elec", "Heck → Suzuki", (0.38, 8.4)),
                ("Buchwald-Hartwig", "SNAr", "nuc", "B–H → SNAr amines", (-.27, .5))):
            row = borrow[borrow.rxn_a.eq(ra) & borrow.rxn_b.eq(rb) & borrow.side.eq(side)].iloc[0]
            ax.annotate(f"{label}\n{signed(row.source_gain_pp, 1)} of {row.target_gain_pp:.1f} pp", (row.rho_obs, row.source_gain_pp), xytext=xytext,
                        fontsize=6, arrowprops=dict(arrowstyle="-", color=GREY, lw=.6))
        ax.set(xlabel="Rank agreement between the reactions", ylabel="Gain over a random ten (pp)", ylim=(-4.4, 10))
        ax.legend(loc="lower right", frameon=False, fontsize=6); BF.clean(ax)
        header(fig, name, "c", .015, .48)
        examples = self.tab("Figure4c_Heck_Suzuki_molecules.csv")
        fig.text(.55, .41, "Heck / Suzuki–Miyaura contributions (pp)", fontsize=6.2, color=MUTED)
        for i, row in enumerate(examples[examples.drawn.fillna(0).gt(0)].sort_values("drawn").itertuples()):
            x, y = .55 + i % 2 * .22, .235 - i // 2 * .18
            native(fig, [x, y, .21, .16], row.mol, F23.ORIENT.get(row.mol, 0))
            fig.text(x + .105, y - .01, f"{signed(row.x_contribution_pp)} / {signed(row.y_contribution_pp)}", ha="center", fontsize=7)
        return fig

    def ed1(self):
        """Fifteen native schemes with the scope of each twenty-condition array."""
        F14.SRC = self.source
        summary = F14.condition_composition_summary()
        designs = self.tab("ExtendedDataFig1_reaction_designs.csv").set_index("reaction")
        records = summary.to_dict("records")
        bands = [records[i:i + 3] for i in range(0, 15, 3)]
        scheme_heights = [max(float(BF.CD.scheme(r["reaction"])["height_pt"]) for r in band) for band in bands]
        row_heights = [h + 26 for h in scheme_heights]
        height = sum(row_heights) + 29
        fig = plt.figure(figsize=(7, height / 72))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.set(xlim=(0, 504), ylim=(0, height))
        top = height - 25
        for band, row_height, scheme_height in zip(bands, row_heights, scheme_heights):
            for j, record in enumerate(band):
                left, width, reaction = 4 + j * 166, 164, record["reaction"]
                asset, design = BF.CD.scheme(reaction), designs.loc[reaction]
                if float(asset["width_pt"]) > width - 8: raise ValueError(f"Native ED1 scheme too wide: {reaction}")
                ax.add_patch(Rectangle((left, top - row_height + 1), width, row_height - 2, fc=LIGHT, ec="none"))
                ax.text(left + 4, top - 1, "SNAr" if reaction == "SNAr" else NAMES[reaction], fontsize=7, weight="bold", va="top", gid="condition-composition")
                BF.CD.draw(ax, asset, left + (width - float(asset["width_pt"])) / 2, top - 10 - (scheme_height - float(asset["height_pt"])) / 2)
                def count(role):
                    return len([item for item in record[role].split(";") if item.strip()])
                scope = f"M {count('metal_sources')}; L {count('ligands')}; B/A {count('bases_acids')}; O {count('other_reagents')}; S {count('solvents')}"
                ax.text(left + 4, top - row_height + 8, f"{int(design.electrophiles):,} × {int(design.nucleophiles):,} molecules; {int(design.pairs):,} pairs", fontsize=5.5, va="bottom", gid="condition-composition")
                ax.text(left + 4, top - row_height + 1, scope, fontsize=5.5, va="bottom", gid="condition-composition")
                ax.text(left + width - 4, top - row_height + 1, f"{record['temperature_C']:g} °C, {record['time_h']:g} h", fontsize=5.5, ha="right", va="bottom", gid="condition-composition")
            top -= row_height
        header(fig, "ExtendedDataFig1", "a", .01, 1 - 11 / height)
        fig.text(.012, 1 - 23 / height, "20 conditions per array; distinct components: M, metal sources; L, ligands; B/A, bases/acids; O, other reagents; S, solvents.", fontsize=5.5, gid="condition-composition")
        return fig

    def ed2(self):
        F14.SRC = self.source
        fig = F14.ed_fig2()
        fig.set_size_inches(6, 4.2)
        fig.axes[0].set_position([.14, .15, .66, .72])
        fig.axes[1].set_position([.84, .15, .025, .72])
        header(fig, "ExtendedDataFig2", "a", .025, .96)
        return fig

    def ed3(self):
        rows = self.tab("ExtendedDataFig3_portability_by_set.csv")
        rows = rows[BF.bool_series(rows.estimable)].sort_values("rho_P").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(6.3, 7.1)); fig.subplots_adjust(left=.30, right=.97, bottom=.09, top=.94)
        for i, row in enumerate(rows.itertuples()):
            values = [v for v in (row.rho_C, row.rho_P, row.between_mean) if np.isfinite(v)]
            ax.plot([min(values), max(values)], [i, i], color="#D3D8DC", lw=1)
            ax.plot(row.rho_C, i, "D", color=TEAL, ms=4)
            ax.plot(row.rho_P, i, "o", mfc=PURPLE if str(row.interval_above_zero).lower() == "true" else "white", mec=PURPLE, ms=4.5)
            if np.isfinite(row.between_mean): ax.plot(row.between_mean, i, "^", color=RED, ms=4.5)
        ax.set(yticks=range(len(rows)), yticklabels=[f"{SHORT[r.reaction]} · {F23.locant(r.role)}" for r in rows.itertuples()],
               xlim=(-.12, 1.04), ylim=(-.7, len(rows) + 1.8), xlabel="Rank agreement of a molecule's contribution")
        ax.axvline(0, color=GREY, lw=.6); BF.clean(ax)
        ax.legend(handles=[Line2D([0], [0], marker=m, ls="", color=c, label=t) for m, c, t in (("D", TEAL, "new conditions"), ("o", PURPLE, "new partners"), ("^", RED, "another reaction"))], frameon=False, loc="upper left")
        header(fig, "ExtendedDataFig3", "a", .025, .975)
        return fig

    def ed4(self):
        name = "ExtendedDataFig4"
        defined = self.tab("ExtendedDataFig4a_defined_benchmark.csv")
        pooled = self.tab("ExtendedDataFig4a_pooled_shortfall.csv")
        undefined = self.tab("ExtendedDataFig4b_undefined_benchmark.csv").reset_index(drop=True)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([.16, .58, .72, .31])
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=.7)
        ax.axhline(0, color=GREY, lw=.5)
        for side, color in (("elec", BLUE), ("nuc", ORANGE)):
            subset = defined[defined.side.eq(side)]
            ax.scatter(subset.geometric_benchmark, subset.rho_R_half, s=10 + subset.molecules, color=color, lw=.4, ec="white")
        row = pooled[pooled.minimum_partners.eq(2)].iloc[0]
        ax.text(.98, .03, f"{int(row.positive_benchmark_comparisons):,} comparisons\nmean shortfall {row.D_G:.2f} ({row.low:.2f}–{row.high:.2f})", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.set(xlim=(0, 1), ylim=(-.6, 1), xlabel="Agreement within each reaction, new partners", ylabel="Agreement between reactions")
        BF.clean(ax); header(fig, name, "a", .025, .955)
        ax = fig.add_axes([.32, .12, .56, .30])
        for i, row in enumerate(undefined.itertuples()):
            ax.plot([row.rho_P_a, row.rho_P_b], [i, i], color=GREY, lw=.8)
            ax.plot(row.rho_P_a, i, "|", ms=8, color=BLUE); ax.plot(row.rho_P_b, i, "|", ms=8, color=ORANGE)
            ax.plot(row.rho_R_half, i, "o", ms=4, color=RED)
        ax.set(yticks=range(len(undefined)), yticklabels=[f"{SHORT[r.rxn_a]} / {SHORT[r.rxn_b]} · {'E' if r.side == 'elec' else 'N'} ({int(r.molecules):,})" for r in undefined.itertuples()],
               xlim=(-1, 1), ylim=(len(undefined) - .4, -.6), xlabel="Rank agreement on the same shared molecules")
        ax.axvline(0, color=GREY, lw=.6); BF.clean(ax)
        fig.legend(handles=[Line2D([0], [0], marker=m, ls="", color=c, label=t) for m, c, t in (("|", BLUE, "first reaction, new partners"), ("|", ORANGE, "second reaction, new partners"), ("o", RED, "between reactions"))], loc="lower center", bbox_to_anchor=(.54, .015), ncol=2, frameon=False, fontsize=6)
        header(fig, name, "b", .025, .485)
        return fig

    def capture(self, function):
        captured = []
        previous = BF.save_figure
        BF.save_figure = lambda fig, name: captured.append(fig)
        try: function()
        finally: BF.save_figure = previous
        if len(captured) != 1: raise ValueError("Expected one legacy panel figure")
        return captured[0]

    def ed5(self):
        BF.SOURCES = self.panels
        fig = self.capture(BF.figure_s2)
        fig.axes[0].set_position([.2, .25, .76, .69])
        header(fig, "ExtendedDataFig5", "a", .025, .98)
        return fig

    def ed6(self):
        name = "ExtendedDataFig6"
        rows = self.tab("ExtendedDataFig6a_campaign_recognition.csv").sort_values("profile_similarity", ascending=False).reset_index(drop=True)
        comp = self.tab("ExtendedDataFig6bc_profile_comparisons.csv")
        fig = plt.figure(figsize=(7, 7))
        for j, column in enumerate(("aryl_halide_rank_agreement", "profile_similarity")):
            ax = fig.add_axes([.235 + j * .415, .58, .285, .30])
            for i, row in enumerate(rows.itertuples()):
                filled = j == 1 or bool(row.ranking_eligible)
                ax.plot(getattr(row, column), i, "o" if j == 0 else "D", mfc=(PURPLE if j == 0 else BLUE) if filled else "white", mec=PURPLE if j == 0 else BLUE, ms=4.5)
            ax.set(yticks=range(len(rows)), yticklabels=[NAMES[r] for r in rows.analysed_reaction] if j == 0 else [""] * len(rows), xlim=(-1.05, 1.05), ylim=(len(rows) - .4, -.6), xlabel="Rank agreement" if j == 0 else "Profile similarity")
            ax.axvline(0, color=GREY, lw=.6); BF.clean(ax)
        fig.text(.235, .91, "Shared aryl-halide rankings", fontsize=7, weight="bold")
        fig.text(.65, .91, "Structural profiles", fontsize=7, weight="bold")
        header(fig, name, "a", .025, .96)
        within = comp[comp.comparison.eq("same reaction, second condition set")].sort_values("pearson")
        ax = fig.add_axes([.18, .11, .285, .29])
        ax.plot(within.pearson, range(len(within)), "o", color=TEAL, ms=4.5)
        ax.set(yticks=range(len(within)), yticklabels=[SHORT[r] for r in within.reaction_a], xlim=(-.5, 1.05), xlabel="Structural profile, r")
        ax.axvline(0, color=GREY, lw=.6); BF.clean(ax)
        header(fig, name, "b", .025, .46)
        two = comp[comp.comparison.eq("two analysed reactions")]
        campaign = comp[comp.comparison.eq("second Buchwald-Hartwig campaign against an analysed reaction") & comp.reaction_a.eq("Buchwald-Hartwig")].iloc[0]
        ax = fig.add_axes([.65, .11, .30, .29])
        ax.scatter(two.record_rank_agreement, two.pearson, s=13, color=GREY, lw=0)
        ax.scatter([campaign.record_rank_agreement], [campaign.pearson], s=25, color=RED, lw=0)
        ax.axhline(0, color=GREY, lw=.5); ax.axvline(0, color=GREY, lw=.5)
        ax.set(xlim=(-.6, 1), ylim=(-1, 1.05), xlabel="Molecule rank agreement", ylabel="Structural profile, r")
        BF.clean(ax); header(fig, name, "c", .51, .46)
        return fig

    def ed7(self):
        name = "ExtendedDataFig7"
        b, c = self.tab("ExtendedDataFig7a_partner_change.csv"), self.tab("ExtendedDataFig7b_random_grouping.csv")
        first = b[b.condition_set.eq("first")].set_index("reaction")
        order = (first.r_structure - first.rho_molecule).sort_values().index.tolist()
        fig = plt.figure(figsize=(7, 6.7))
        ax = fig.add_axes([.23, .57, .70, .30])
        for row in b.itertuples():
            y = order.index(row.reaction) + (.15 if row.condition_set == "first" else -.15)
            fill = row.condition_set == "first"
            ax.plot([row.rho_molecule, row.r_structure], [y, y], color=GREY, lw=.8)
            ax.plot(row.rho_molecule, y, "o", ms=4.5, mec=PURPLE, mfc=PURPLE if fill else "white")
            ax.plot(row.r_structure, y, "D", ms=4, mec=BLUE, mfc=BLUE if fill else "white")
        ax.set(yticks=range(len(order)), yticklabels=[NAMES[r] for r in order], xlim=(-1, 1.08), ylim=(-.6, len(order) + .6), xlabel="Agreement between partner halves")
        ax.axvline(0, color=GREY, lw=.6); BF.clean(ax)
        ax.legend(handles=[Line2D([0], [0], marker=m, ls="", color=color, label=label) for m, color, label in (("o", PURPLE, "molecule ranking"), ("D", BLUE, "structural profile"))], frameon=False, loc="upper left", ncol=2, fontsize=6)
        header(fig, name, "a", .025, .95)
        ax = fig.add_axes([.23, .10, .70, .30])
        cf = c[c.condition_set.eq("first")].set_index("reaction")
        for i, reaction in enumerate(order):
            row = cf.loc[reaction]
            ax.plot([row["null_2.5"], row["null_97.5"]], [i, i], color="#C9CFD4", lw=3.5)
            ax.plot([row.null_median] * 2, [i - .2, i + .2], color=MUTED, lw=1)
            ax.plot(row.r_structure, i, "D", mec=BLUE, mfc=BLUE if row.r_structure > row["null_97.5"] else "white", ms=4.5)
        ax.set(yticks=range(len(order)), yticklabels=[NAMES[r] for r in order], xlim=(-1, 1.08), ylim=(-.6, len(order) + .6), xlabel="Agreement of structural profiles")
        ax.axvline(0, color=GREY, lw=.6); BF.clean(ax)
        header(fig, name, "b", .025, .48)
        return fig

    def ed8(self):
        name = "ExtendedDataFig8"
        pairs, metric = self.tab("ExtendedDataFig8a_heldout_amide_pairs.csv"), self.tab("ExtendedDataFig8a_metrics.csv").iloc[0]
        fig = plt.figure(figsize=(7, 8.2))
        ax = fig.add_axes([.24, .62, .52, .28])
        ax.scatter(pairs.prediction_pp, pairs.target_pp, s=3.5, color=BLUE, alpha=.25, lw=0)
        values = pairs[["prediction_pp", "target_pp"]].to_numpy()
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values)); pad = .035 * (hi - lo)
        limits = (lo - pad, hi + pad)
        ax.plot(limits, limits, "--", color=GREY, lw=.7)
        ax.set(xlim=limits, ylim=limits, xlabel="Predicted (pp)", ylabel="Measured, condition-adjusted (pp)")
        ax.text(.03, .97, f"{int(metric.scored_pairs):,} held-out pairs\nerror {metric.prediction_rmse_pp:.1f} pp\nreaction mean: {metric.reaction_mean_rmse_pp:.1f} pp", transform=ax.transAxes, va="top", fontsize=7)
        BF.clean(ax); header(fig, name, "a", .025, .96)
        inc = self.tab("ExtendedDataFig8b_prediction_all_reactions.csv")
        order = inc.groupby("reaction").scored_pairs.first().sort_values(ascending=False).index.tolist()
        ax = fig.add_axes([.30, .10, .66, .39])
        for i, reaction in enumerate(order):
            for side, shift, color in (("elec", -.15, BLUE), ("nuc", .15, ORANGE)):
                row = inc[inc.reaction.eq(reaction) & inc.side.eq(side)].iloc[0]
                ax.plot([max(row.low, -2), min(row.high, 1)], [i + shift] * 2, color=color, lw=1)
                ax.plot(np.clip(row.increment, -2, 1), i + shift, "o", ms=4, mec=color, mfc=color if row.interval_above_zero else "white")
        n = inc.groupby("reaction").scored_pairs.first()
        ax.set(yticks=range(len(order)), yticklabels=[f"{NAMES[r]} ({int(n[r]):,})" for r in order], ylim=(len(order) - .5, -.6),
               xlim=(-2.05, 1.05), xlabel="Share of prediction error removed by one partner's contributions")
        ax.axvline(0, color=GREY, lw=.6); BF.clean(ax)
        ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=color, label=label) for color, label in ((BLUE, "electrophile"), (ORANGE, "nucleophile"))], frameon=False, loc="upper left", ncol=2, fontsize=6)
        header(fig, name, "b", .025, .55)
        return fig

    def si1(self):
        BF.SOURCES = self.panels
        fig = self.capture(BF.figure_s4)
        # Existing figure's two titles are already the final SI titles.
        return fig

    def render(self, only=None):
        BF.style()
        self.out.joinpath("figures").mkdir(parents=True, exist_ok=True)
        for name, make in (("Figure2", self.figure2), ("Figure3", self.figure3), ("Figure4", self.figure4),
                           ("ExtendedDataFig1", self.ed1), ("ExtendedDataFig2", self.ed2), ("ExtendedDataFig3", self.ed3),
                           ("ExtendedDataFig4", self.ed4), ("ExtendedDataFig5", self.ed5), ("ExtendedDataFig6", self.ed6),
                           ("ExtendedDataFig7", self.ed7), ("ExtendedDataFig8", self.ed8), ("SupplementaryFig1", self.si1)):
            if only and name not in only:
                continue
            self.export(name, make())
            print(f"{name}: native vectors, titles, physical bounds and text overlap checked")
        (self.out / "figure_qa.json").write_text(json.dumps(self.qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

