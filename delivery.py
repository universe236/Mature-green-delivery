"""Late and cumulative delivery relative to the mature-green benchmark (Equation 5).

Cumulative delivery sums annual cooling over the first five annual observations from greening onset (years 0-4)
and compares it with five times the benchmark, giving each hot season equal weight (units: degC year). The
overestimate is the extra cooling implied by crediting mature performance throughout the window, expressed as a
percentage of observed delivery. Group means of delivery and benchmark always use the same matched events.
"""
import numpy as np


def cumulative_ratio(mean_delivery, benchmark, window=(0, 4)):
    """R = sum_{k in window} d_k / (n_years * B)."""
    ks = range(window[0], window[1] + 1)
    return sum(mean_delivery[k] for k in ks) / (len(ks) * benchmark)


def overestimate(ratio):
    """O = (1/R - 1) x 100%; differs from 1 - R, which is the shortfall as a share of the benchmark."""
    return (1.0 / ratio - 1.0) * 100.0


def late_ratio(mean_delivery, benchmark, years=(5, 6)):
    """Mean delivery in years 5-6 relative to the benchmark."""
    return np.mean([mean_delivery[k] for k in years]) / benchmark


def group_delivery(delivery, benchmark, windows=((0, 4),), late=(5, 6)):
    """Summaries for one group of matched events.

    delivery: DataFrame (events x relative years); benchmark: Series of B for the same events.
    Events without a benchmark, or without observations in a window, are dropped so that numerator and denominator
    use exactly the same events.
    """
    out = {}
    for w in windows:
        ks = list(range(w[0], w[1] + 1))
        keep = benchmark.notna() & delivery[ks].notna().all(axis=1)
        d = delivery.loc[keep].mean()
        b = benchmark[keep].mean()
        r = cumulative_ratio(d, b, w)
        out[f'cumulative_{w[0]}_{w[1]}'] = r
        out[f'overestimate_{w[0]}_{w[1]}'] = overestimate(r)
        out[f'delivered_{w[0]}_{w[1]}'] = d[ks].sum()
        out[f'benchmark_total_{w[0]}_{w[1]}'] = len(ks) * b
    keep = benchmark.notna() & delivery[list(late)].notna().all(axis=1)
    out['late'] = late_ratio(delivery.loc[keep].mean(), benchmark[keep].mean(), late)
    out['benchmark'] = benchmark[keep].mean()
    out['n_events'] = int(keep.sum())
    return out
