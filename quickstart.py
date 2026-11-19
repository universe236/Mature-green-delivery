"""Run every modelling step on a small synthetic data set.

The synthetic events only demonstrate the inputs and outputs of each function; their values carry no meaning.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import greendelivery as gd  # noqa: E402

rng = np.random.default_rng(0)
YEARS = list(range(2013, 2024))
KS = list(range(-3, 7))

# ------------------------------------------------------------------ synthetic inputs
SPEED = {'planted': 0.7, 'spontaneous': 4.0}            # time scale of cooling development (years)
AMPLITUDE = {('planted', 'tree'): 2.0, ('planted', 'grass'): 1.6, ('spontaneous', 'tree'): 1.7,
             ('spontaneous', 'grass'): 1.4}
events, ndvi, t_patch, t_ref = [], [], [], []
for city in range(20):
    city_heat = rng.normal(0, 1)
    for _ in range(rng.integers(20, 35)):
        mode = rng.choice(['planted', 'spontaneous'], p=[0.6, 0.4])
        veg = rng.choice(['tree', 'grass'], p=[0.75, 0.25] if mode == 'planted' else [0.55, 0.45])
        onset = int(rng.choice([2015, 2016, 2017, 2018]))
        k = np.array(YEARS) - onset
        rise = np.where(k >= 0, 1 - np.exp(-(k + 1) / SPEED[mode]), 0.0)
        ndvi.append(0.2 + 0.4 * rise + rng.normal(0, 0.01, len(YEARS)))
        c = rng.normal(0.2, 0.2) + AMPLITUDE[(mode, veg)] * rise + rng.normal(0, 0.2, len(YEARS))
        ref = 32 + city_heat + rng.normal(0, 0.5, len(YEARS))
        t_ref.append(ref)
        t_patch.append(ref - c)
        events.append({'city': f'C{city:02d}', 'mode': mode, 'veg_type': veg, 'true_onset': onset,
                       'area': rng.uniform(0.45, 5), 'builtup': rng.uniform(0.3, 0.9), 'building': rng.uniform(0, 0.1),
                       'heat_2013': city_heat + rng.normal(0, 1), 'nominal_se': rng.uniform(0.05, 0.4)})
events = pd.DataFrame(events)
ndvi = pd.DataFrame(ndvi, columns=YEARS)
t_patch = pd.DataFrame(t_patch, columns=YEARS)
t_ref = pd.DataFrame(t_ref, columns=YEARS)
cities = events['city'].unique()
green = pd.DataFrame({'city': np.repeat(cities, 40), 'veg_type': rng.choice(['tree', 'grass'], 40 * len(cities)),
                      'area': rng.uniform(0.45, 8, 40 * len(cities)), 'builtup': rng.uniform(0.3, 0.9, 40 * len(cities)),
                      'building': rng.uniform(0, 0.1, 40 * len(cities))})
green['contrast_late'] = np.where(green['veg_type'] == 'tree', 2.2, 1.6) + rng.normal(0, 0.4, len(green))
nonveg = pd.DataFrame({'city': np.repeat(cities, 30), 'area': rng.uniform(0.45, 8, 30 * len(cities)),
                       'builtup': rng.uniform(0.3, 0.9, 30 * len(cities)),
                       'building': rng.uniform(0, 0.1, 30 * len(cities))})
nonveg['contrast_late'] = rng.normal(0.1, 0.3, len(nonveg))
print('Synthetic data:', len(events), 'events in', len(cities), 'cities\n')

# ------------------------------------------------------------------ 1. local cooling and placebo
pre, post = [2014, 2015, 2016], [2021, 2022, 2023]
events['cooling'] = gd.local_cooling(t_patch[pre].mean(axis=1), t_patch[post].mean(axis=1),
                                     t_ref[pre].mean(axis=1), t_ref[post].mean(axis=1))
placebo = gd.local_cooling(*(30 + rng.normal(0, 0.3, (4, 500))))
print(f'1. Median two-period cooling {events.cooling.median():.2f} degC; placebo median {np.median(placebo):.2f}')

# ------------------------------------------------------------------ 2. formation mode and vegetation type
features = np.column_stack([
    (events['mode'] == 'planted') * 0.6 + rng.normal(0, 0.5, len(events)),      # patch shape regularity
    ndvi.diff(axis=1).max(axis=1) + rng.normal(0, 0.12, len(events)),           # abruptness of greening
    events['builtup'] + rng.normal(0, 0.1, len(events)),                        # urban context
])
interpreted = rng.choice(len(events), 200, replace=False)
model, metrics = gd.fit_formation_mode(features[interpreted], (events['mode'] == 'planted').to_numpy()[interpreted],
                                       events['city'].to_numpy()[interpreted], n_estimators=200)
events['mode_pred'] = gd.assign_mode(model.predict_proba(features)[:, 1])
print('2. Formation-mode classifier (city-grouped CV):', {k: round(float(v), 2) for k, v in metrics.items()})
print('   vegetation type from tree share 0.6 / 0.3 / 0.1:', gd.vegetation_type([0.6, 0.3, 0.1]))

# ------------------------------------------------------------------ 3. onset, annual delivery and fixed cohort
events['onset'] = [gd.detect_onset(ndvi.loc[i]) for i in events.index]
print(f"3. Onset recovered for {(events['onset'] == events['true_onset']).mean():.0%} of events")
contrast_by_year = pd.DataFrame(gd.contrast(t_ref, t_patch), columns=YEARS)
delivery = gd.annual_delivery(contrast_by_year, events['onset'], k_range=KS)
cohort = gd.fixed_cohort(delivery, events['onset'])
print(f'   fixed cohort: {len(cohort)} events with onset in 2016-2017 and complete years 0-6')

# ------------------------------------------------------------------ 4. mature-green benchmark
events['benchmark'] = gd.match_benchmark(events, green, nonveg)
print(f"4. Benchmark available for {events['benchmark'].notna().mean():.0%} of events; "
      f"mean {events['benchmark'].mean():.2f} degC")

# ------------------------------------------------------------------ 5. speed, level and cumulative delivery
data = pd.concat([events, delivery], axis=1).loc[cohort]
data['group'] = np.where(data['mode'] == 'spontaneous', 'spontaneous', 'planted ' + data['veg_type'])


def statistic(df):
    out = {}
    for g, sub in df.groupby('group'):
        out[f'{g}: year-1 share'] = gd.year1_share(sub[KS].mean())
        summary = gd.group_delivery(sub[KS], sub['benchmark'], windows=((-1, 3), (0, 4), (1, 5)))
        out[f'{g}: late / benchmark'] = summary['late']
        out[f'{g}: cumulative 0-4'] = summary['cumulative_0_4']
        out[f'{g}: overestimate 0-4 (%)'] = summary['overestimate_0_4']
    sh = {g: gd.year1_share(s[KS].mean()) for g, s in df.groupby(['mode', 'veg_type'])}
    out['planted minus spontaneous, trees (share)'] = sh[('planted', 'tree')] - sh[('spontaneous', 'tree')]
    out['planted grass minus planted trees (share)'] = sh[('planted', 'grass')] - sh[('planted', 'tree')]
    return out


print('5. Group results with city-bootstrap 95% intervals (200 draws here; 5,000 in the paper)')
print(gd.city_bootstrap(data, statistic, n_draws=200).round(2).to_string())
for g, sub in data.groupby('group'):
    fit = gd.saturating_fit(sub[KS].mean())
    print(f'   {g}: time to 90% of fitted level {fit["t90"]:.1f} years')

# ------------------------------------------------------------------ 6. within-city regression
X = pd.DataFrame({'ndvi_gain': ndvi[post].mean(axis=1) - ndvi[pre].mean(axis=1), 'heat_2013': events['heat_2013']})
X = (X - X.mean()) / X.std()
X['tree_share'] = np.where(events['veg_type'] == 'tree', 0.8, 0.1)
fe = gd.fixed_effects(events['cooling'], X, events['city'], gd.measurement_weights(events['nominal_se']))
print('6. City fixed-effects model (city-clustered 95% intervals)\n' + fe.round(2).to_string())

# ------------------------------------------------------------------ 7. tree-grass reassessment
events['late_delivery'] = delivery[[5, 6]].mean(axis=1)
print('7. City-equal tree-minus-grass comparison\n' +
      gd.tree_grass_city_equal(events, new_col='late_delivery').round(2).to_string())
