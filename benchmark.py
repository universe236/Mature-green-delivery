"""Mature-green benchmark (Equation 2).

Each new site is compared with established green space in the same city and in a similar urban setting. The five
nearest established-green patches of the same vegetation type, and five non-vegetated patches, are selected on
standardized log area, surrounding built-up cover and building cover. The benchmark is the late-period contrast of
the green matches minus that of the non-vegetated matches, which removes patch-to-surrounding contrasts that exist
without green cover. Greenness is left out of the main match because vegetation development is part of the
transition under study; it can be added as a second benchmark.
"""
import numpy as np
import pandas as pd

SETTING = ('area', 'builtup', 'building')


def _design(df, covariates, log_cols):
    X = df[list(covariates)].astype(float).copy()
    for c in log_cols:
        X[c] = np.log(X[c])
    return X


def match_benchmark(events, green, nonveg, k=5, covariates=SETTING, log_cols=('area',), add_greenness=False,
                    greenness_col='ndvi', type_col='veg_type', city_col='city', contrast_col='contrast_late'):
    """Return the mature-green benchmark B for each event (NaN when fewer than k matches are available).

    events: one row per new site with city, vegetation type and the matching covariates.
    green: established-green patches with city, vegetation type, covariates and late-period contrast.
    nonveg: non-vegetated patches with city, covariates and late-period contrast.
    """
    cov = list(covariates) + ([greenness_col] if add_greenness else [])
    out = pd.Series(np.nan, index=events.index, name='benchmark')
    for city, ev in events.groupby(city_col):
        g = green[green[city_col] == city]
        n = nonveg[nonveg[city_col] == city]
        if len(n) < k:
            continue
        pool = pd.concat([_design(d, cov, log_cols) for d in (ev, g, n)])
        mu, sd = pool.mean(), pool.std(ddof=0).replace(0, 1)
        z = lambda d: ((_design(d, cov, log_cols) - mu) / sd).to_numpy()
        ze, zg, zn = z(ev), z(g), z(n)
        cn = n[contrast_col].to_numpy()
        cg = g[contrast_col].to_numpy()
        gtype = g[type_col].to_numpy()
        for row, (idx, vt) in enumerate(zip(ev.index, ev[type_col])):
            same = np.flatnonzero(gtype == vt)
            if len(same) < k:
                continue
            dg = np.linalg.norm(zg[same] - ze[row], axis=1)
            dn = np.linalg.norm(zn - ze[row], axis=1)
            best_g = same[np.argsort(dg)[:k]]
            best_n = np.argsort(dn)[:k]
            out[idx] = cg[best_g].mean() - cn[best_n].mean()
    return out
