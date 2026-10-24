"""Extended Data Table 1."""
from definitions import NAMES
from presentation_io import SOURCES, read_csv, read_upstream

def figure_names():
    return {k: v.replace('S$_\\mathregular{N}$Ar', 'SNAr') for k, v in NAMES.items()}


def table1():
    """Table 1, built from the plotted source data: for every reaction and partner, the rank agreement
    across partner halves and the share of held-out prediction error removed."""
    src = SOURCES
    cells = read_csv(src / 'Figure2a_portability_by_set.csv').set_index(['reaction', 'side'])
    inc = read_csv(src / 'ExtendedDataFig7_prediction_all_reactions.csv').set_index(['reaction', 'side'])
    labels = read_upstream('reaction_labels.csv')[0].set_index('rxn_class').metal
    if len(inc) != 30 or len(cells) != 30:
        raise RuntimeError('Table 1 expects fifteen reactions with two partners each')
    names = figure_names()
    order = inc.xs('elec', level='side').scored_pairs.sort_values(ascending=False, kind='stable').index

    def agreement(rc, side):
        r = cells.loc[(rc, side)]
        if not bool(r.estimable):
            return '—'
        text = f'{r.rho_P:.2f}'.replace('-', '−')
        return f'**{text}**' if bool(r.interval_above_zero) else text

    def removed(rc, side):
        r = inc.loc[(rc, side)]
        text = '<0' if r.increment < 0 else f'{100 * r.increment:.0f}%'
        return f'**{text}**' if bool(r.interval_above_zero) else text

    metal = lambda m: 'mixed' if m.startswith('mixed') else m
    rows = ['| Reaction | Metal | Electrophile | New partners | Error removed | Nucleophile | New partners | Error removed | Pairs |',
            '|---|---|---|---|---|---|---|---|---|']
    for rc in order:
        rows.append(f'| {names[rc]} | {metal(labels[rc])} | {inc.loc[(rc, "elec")].role} | {agreement(rc, "elec")} | {removed(rc, "elec")} '
                    f'| {inc.loc[(rc, "nuc")].role} | {agreement(rc, "nuc")} | {removed(rc, "nuc")} | {int(inc.loc[(rc, "elec")].scored_pairs)} |')
    caption = ("**Extended Data Table 1 | Where a building block's record carries within each reaction.** For each partner, the rank agreement of contributions "
               "estimated from two disjoint halves of each molecule's partners (new partners) and the share of held-out prediction error removed "
               "by that partner's contributions; bold, 95% interval above zero; —, too few partners per molecule to split; <0, prediction worse "
               "than without. Metal, the metal named in at least 15 of the 20 conditions (mixed, metals present but none in 15). Pairs, held-out pairs scored.")
    return caption + '\n\n' + '\n'.join(rows)
