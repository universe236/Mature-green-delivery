"""Annual cooling trajectories (Equations 3 and 4, Equation S1).

Each event is aligned to its first sustained NDVI rise above the 2013-2014 level (year 0). Annual delivery is the
patch-to-surrounding contrast in a year relative to the 2013-2014 mean, which removes the site's initial thermal
difference from its surroundings. A fixed cohort keeps the same events in every year, so changes in sample
membership cannot create an apparent growth trajectory.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE_YEARS = (2013, 2014)


def detect_onset(ndvi, base_years=BASE_YEARS, rise=0.05, sustain=0.025):
    """First year in which NDVI exceeds its baseline mean by `rise` and stays above it by `sustain` afterwards.

    ndvi: Series indexed by calendar year (missing years as NaN). Returns the onset year or None.
    """
    s = ndvi.dropna()
    if not all(y in s.index for y in base_years):
        return None
    excess = s - s.loc[list(base_years)].mean()
    for year in s.index[s.index > max(base_years)]:
        later = excess.loc[excess.index > year]
        if excess.loc[year] >= rise and (later >= sustain).all():
            return int(year)
    return None


def annual_delivery(contrast_by_year, onset, k_range=range(-3, 7), base_years=BASE_YEARS):
    """Delivery d_{i,k} = c_{i,g_i+k} - mean(c_{i,2013}, c_{i,2014}) for each event i and relative year k.

    contrast_by_year: DataFrame, one row per event, one column per calendar year.
    onset: Series of onset years g_i aligned with the rows.
    """
    base = contrast_by_year[list(base_years)].mean(axis=1, skipna=False)
    out = pd.DataFrame(np.nan, index=contrast_by_year.index, columns=list(k_range))
    for k in k_range:
        year = onset + k
        vals = [contrast_by_year.at[i, y] if pd.notna(y) and y in contrast_by_year.columns else np.nan
                for i, y in zip(contrast_by_year.index, year)]
        out[k] = np.asarray(vals, dtype=float) - base.to_numpy()
    return out


def fixed_cohort(delivery, onset, onset_years=(2016, 2017), required=range(0, 7)):
    """Events with onset in 2016-2017 and complete baseline and post-onset observations (years 0-6)."""
    ok = onset.isin(onset_years) & delivery[list(required)].notna().all(axis=1)
    return delivery.index[ok]


def year1_share(mean_delivery):
    """Share of a group's later cooling already present at year 1: S = 2 d_1 / (d_5 + d_6).

    The denominator is the group's own years 5-6 mean, so speed is separated from cooling amplitude.
    """
    return 2 * mean_delivery[1] / (mean_delivery[5] + mean_delivery[6])


def _saturating(k, a, tau):
    return a * (1 - np.exp(-(k + 1) / tau))


def saturating_fit(mean_delivery, years=range(0, 7)):
    """Fit f(k) = A[1 - exp(-(k+1)/tau)] to group means for years 0-6 (Equation S1).

    The k + 1 offset allows positive cooling at onset. t90 = tau ln(10) - 1 is the relative year at which the fitted
    curve reaches 90% of its asymptote; it does not use the mature-green benchmark.
    """
    k = np.asarray(list(years), dtype=float)
    d = np.asarray([mean_delivery[int(x)] for x in k], dtype=float)
    (a, tau), _ = curve_fit(_saturating, k, d, p0=(max(d.max(), 0.1), 2.0), bounds=([0, 0.05], [np.inf, 100]))
    return {'asymptote': a, 'tau': tau, 't90': tau * np.log(10) - 1}
