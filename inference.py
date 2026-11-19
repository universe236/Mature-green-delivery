"""Uncertainty and the within-city regression.

Events in the same city share climate, data quality and planting practice, so uncertainty is estimated by
resampling whole cities with all their events. Differences and ratios are computed jointly in each draw. The
regression uses city fixed effects and city-clustered standard errors.
"""
import numpy as np
import pandas as pd
from scipy import stats


def city_bootstrap(events, statistic, city_col='city', n_draws=5000, level=0.95, seed=0):
    """Point estimates and percentile intervals from resampling whole cities.

    statistic: function mapping an events DataFrame to a dict of named estimates.
    Returns a DataFrame with columns estimate, low and high.
    """
    rng = np.random.default_rng(seed)
    point = statistic(events)
    groups = {c: g for c, g in events.groupby(city_col)}
    cities = np.array(list(groups))
    draws = []
    for _ in range(n_draws):
        pick = rng.choice(cities, size=len(cities), replace=True)
        sample = pd.concat([groups[c] for c in pick], ignore_index=True)
        draws.append(statistic(sample))
    draws = pd.DataFrame(draws)
    a = (1 - level) / 2
    return pd.DataFrame({'estimate': pd.Series(point), 'low': draws.quantile(a), 'high': draws.quantile(1 - a)})


def measurement_weights(se):
    """Weights 1 / (SE^2 + median(SE)^2), which limit the influence of very small nominal errors."""
    se = np.asarray(se, dtype=float)
    return 1.0 / (se ** 2 + np.median(se) ** 2)


def fixed_effects(y, X, city, weights=None, level=0.95):
    """Weighted regression with city fixed effects and city-clustered (CR1) standard errors.

    y: (n,) outcome; X: DataFrame of predictors (standardize continuous predictors beforehand); city: group labels.
    """
    names = list(X.columns)
    Xv = X.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    w = np.ones(len(yv)) if weights is None else np.asarray(weights, dtype=float)
    g = pd.Series(np.asarray(city))
    # weighted within-city demeaning absorbs the fixed effects
    Xd, yd = Xv.copy(), yv.copy()
    for _, idx in g.groupby(g).groups.items():
        idx = np.asarray(idx)
        ww = w[idx] / w[idx].sum()
        Xd[idx] -= ww @ Xv[idx]
        yd[idx] -= ww @ yv[idx]
    XtW = Xd.T * w
    bread = np.linalg.inv(XtW @ Xd)
    beta = bread @ XtW @ yd
    resid = yd - Xd @ beta
    meat = np.zeros((len(names), len(names)))
    for _, idx in g.groupby(g).groups.items():
        idx = np.asarray(idx)
        s = (Xd[idx].T * w[idx]) @ resid[idx]
        meat += np.outer(s, s)
    n, p, G = len(yv), len(names), g.nunique()
    vcov = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - p))
    se = np.sqrt(np.diag(vcov))
    t = stats.t.ppf(1 - (1 - level) / 2, G - 1)
    return pd.DataFrame({'coef': beta, 'se': se, 'low': beta - t * se, 'high': beta + t * se}, index=names)
