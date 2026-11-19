"""Formation mode (planted or spontaneous) and vegetation type.

Interpreted events train a random forest on patch geometry, vegetation characteristics, temporal development and
urban context. Combinations such as a regular patch shape with an abrupt rise in greenness can identify designed
planting. Cross-validation holds out whole cities, so performance reflects transfer to unseen cities.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def vegetation_type(tree_share, tree_min=0.5, grass_max=0.2):
    """Tree-dominated above 0.5 tree share, grass-dominated below 0.2, mixed otherwise."""
    s = np.asarray(tree_share, dtype=float)
    return np.where(s > tree_min, 'tree', np.where(s < grass_max, 'grass', 'mixed'))


def fit_formation_mode(features, planted, city, n_splits=5, threshold=0.7, seed=0, **rf_kwargs):
    """Fit the formation-mode classifier and evaluate it with city-grouped cross-validation.

    features: (n, p) array; planted: 1 for planted, 0 for spontaneous; city: group label per event.
    Returns the model refitted on all interpreted events and the cross-validated metrics.
    """
    X = np.asarray(features, dtype=float)
    y = np.asarray(planted, dtype=int)
    groups = np.asarray(city)
    params = dict(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    params.update(rf_kwargs)
    prob = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = RandomForestClassifier(**params).fit(X[train], y[train])
        prob[test] = model.predict_proba(X[test])[:, 1]
    high = prob > threshold
    metrics = {
        'auc': roc_auc_score(y, prob),
        'accuracy_at_0.5': accuracy_score(y, prob > 0.5),
        'planted_precision_above_threshold': y[high].mean() if high.any() else np.nan,
    }
    return RandomForestClassifier(**params).fit(X, y), metrics


def assign_mode(prob_planted, planted_min=0.7, spontaneous_max=0.4):
    """Planted above 0.7, spontaneous below 0.4; intermediate predictions are left unassigned (None)."""
    p = np.asarray(prob_planted, dtype=float)
    out = np.full(p.shape, None, dtype=object)
    out[p > planted_min] = 'planted'
    out[p < spontaneous_max] = 'spontaneous'
    return out
