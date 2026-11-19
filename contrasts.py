"""Tree-grass comparison between new and mature vegetation under a common sample.

A temperature baseline that overlaps establishment absorbs some early cooling, and the amount can differ between
vegetation types. The reassessment therefore holds events and benchmarks fixed, summarizes events by city medians,
keeps cities with at least five tree and five grass events, and gives every city equal weight.
"""
import pandas as pd


def tree_grass_city_equal(events, new_col='cooling', mature_col='benchmark', type_col='veg_type', city_col='city',
                          min_per_type=5):
    """City-equal tree-minus-grass differences for new and matched mature vegetation.

    Returns the new advantage, the mature advantage and their paired difference (mature minus new).
    """
    ev = events[events[type_col].isin(['tree', 'grass'])].dropna(subset=[new_col, mature_col])
    counts = ev.groupby([city_col, type_col]).size().unstack(fill_value=0)
    keep = counts.index[(counts.get('tree', 0) >= min_per_type) & (counts.get('grass', 0) >= min_per_type)]
    med = ev[ev[city_col].isin(keep)].groupby([city_col, type_col])[[new_col, mature_col]].median().unstack(type_col)
    new_adv = (med[(new_col, 'tree')] - med[(new_col, 'grass')]).mean()
    mature_adv = (med[(mature_col, 'tree')] - med[(mature_col, 'grass')]).mean()
    return pd.Series({'new_tree_minus_grass': new_adv, 'mature_tree_minus_grass': mature_adv,
                      'mature_minus_new': mature_adv - new_adv, 'n_cities': len(keep)})
