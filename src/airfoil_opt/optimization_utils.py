"""Optimization helper functions."""
import numpy as np
from typing import Tuple, Callable
from scipy.stats import norm

from .geometry_utils import normalize_unit_chord, thickness_ratio, has_self_intersection

def calculate_expected_improvement(mu: float, sigma: float, J_best: float, xi: float = 0.01) -> float:
    """
    Computes the Expected Improvement per unit of cost.
    
    Args:
        mu: Predicted mean of the objective function (lower is better).
        sigma: Predicted standard deviation of the objective function.
        J_best: The best objective value found so far (minimum).
        xi: Exploration-exploitation trade-off parameter.
    
    Returns:
        The Expected Improvement value.
    """
    if sigma <= 1e-9:
        return 0.0

    # Since we want to MINIMIZE J, improvement is J_best - mu
    improvement = J_best - mu - xi
    Z = improvement / sigma
    
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei


def pack_dofs(mask: np.ndarray, control_deltas: np.ndarray) -> np.ndarray:
    """Extract active DOFs into 1D vector."""
    return control_deltas[mask]


def unpack_dofs(mask: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Reconstruct full control delta array from DOF vector."""
    control_deltas = np.zeros(mask.shape, dtype=float)
    control_deltas[mask] = vec
    return control_deltas


def vec_to_coords_factory(ffd_refined, best_coords, mask):
    """Create coordinate conversion function."""
    def vec_to_coords(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        control_deltas = unpack_dofs(mask, vec)
        coords = ffd_refined.deform(best_coords, control_deltas=control_deltas)
        return normalize_unit_chord(coords), control_deltas
    return vec_to_coords


def thickness_constraint_min(coords: np.ndarray, t_min: float = 0.12) -> float:
    """Minimum thickness constraint."""
    t = thickness_ratio(coords)
    return (t if np.isfinite(t) else -1.0) - t_min


def thickness_constraint_max(coords: np.ndarray, t_max: float = 0.35) -> float:
    """Maximum thickness constraint."""
    t = thickness_ratio(coords)
    return t_max - (t if np.isfinite(t) else 0.40)


def no_intersection_constraint(coords: np.ndarray) -> float:
    """Self-intersection constraint."""
    return 1.0 if not has_self_intersection(coords) else -1.0


def create_constraints(vec_to_coords: Callable):
    """Build SLSQP constraint dictionary."""
    def constraint_t_min(vec):
        coords, _ = vec_to_coords(vec)
        return thickness_constraint_min(coords)
    
    def constraint_t_max(vec):
        coords, _ = vec_to_coords(vec)
        return thickness_constraint_max(coords)
    
    def constraint_intersect(vec):
        coords, _ = vec_to_coords(vec)
        return no_intersection_constraint(coords)
    
    return (
        {'type': 'ineq', 'fun': constraint_t_min},
        {'type': 'ineq', 'fun': constraint_t_max},
        {'type': 'ineq', 'fun': constraint_intersect}
    )

def _setup_bounds(mask, x_bound, y_bound):
    """Create bounds for SLSQP."""
    limit = np.zeros(mask.shape, dtype=float)
    limit[:, :, 0] = x_bound
    limit[:, :, 1] = y_bound
    selected_limits = limit[mask]
    return [(-float(val), float(val)) for val in selected_limits]