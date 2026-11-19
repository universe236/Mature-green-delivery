"""Model code for "Mature green overpromises how fast new urban greening cools"."""
from .benchmark import match_benchmark
from .contrasts import tree_grass_city_equal
from .cooling import contrast, local_cooling
from .delivery import cumulative_ratio, group_delivery, late_ratio, overestimate
from .formation import assign_mode, fit_formation_mode, vegetation_type
from .inference import city_bootstrap, fixed_effects, measurement_weights
from .trajectory import annual_delivery, detect_onset, fixed_cohort, saturating_fit, year1_share

__all__ = [
    'contrast', 'local_cooling', 'vegetation_type', 'fit_formation_mode', 'assign_mode', 'match_benchmark',
    'detect_onset', 'annual_delivery', 'fixed_cohort', 'year1_share', 'saturating_fit', 'cumulative_ratio',
    'late_ratio', 'overestimate', 'group_delivery', 'city_bootstrap', 'fixed_effects', 'measurement_weights',
    'tree_grass_city_equal',
]
