"""Local cooling of a greening event (Equation 1).

Cooling is the change in the patch temperature relative to stable land 60-300 m from the patch edge, between a
period before and a period after greening. Repeated observations remove persistent site differences, and the
reference removes changes shared with the neighbourhood (difference-in-differences logic).
"""
import numpy as np


def contrast(t_reference, t_patch):
    """Surrounding-minus-patch temperature contrast; positive when the patch is cooler than its surroundings."""
    return np.asarray(t_reference, dtype=float) - np.asarray(t_patch, dtype=float)


def local_cooling(t_patch_pre, t_patch_post, t_ref_pre, t_ref_post):
    """Two-period cooling C = (T_post^R - T_post^P) - (T_pre^R - T_pre^P).

    The same function applied to patches that stayed non-vegetated gives the placebo distribution.
    """
    return contrast(t_ref_post, t_patch_post) - contrast(t_ref_pre, t_patch_pre)
