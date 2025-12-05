"""Geometric utility functions for airfoil analysis."""
import numpy as np
from typing import Tuple


def normalize_unit_chord(coords: np.ndarray) -> np.ndarray:
    """Scale airfoil to unit chord length."""
    x = coords[:, 0]
    chord = float(np.ptp(x))
    if chord <= 1e-6:
        return coords.copy()
    
    result = coords.copy()
    result[:, 0] = (result[:, 0] - np.min(x)) / chord
    result[:, 1] = result[:, 1] / chord
    return result


def split_upper_lower(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split airfoil into upper and lower surfaces (both LE->TE)."""
    i_le = int(np.argmin(coords[:, 0]))
    upper = coords[:i_le + 1][::-1]
    lower = coords[i_le:]
    return upper, lower


def thickness_ratio(coords: np.ndarray) -> float:
    """Compute max thickness-to-chord ratio."""
    upper, lower = split_upper_lower(coords)
    x_min = max(np.min(upper[:, 0]), np.min(lower[:, 0]))
    x_max = min(np.max(upper[:, 0]), np.max(lower[:, 0]))
    
    if x_max <= x_min:
        return np.nan
    
    x_sample = np.linspace(x_min, x_max, 200)
    y_upper = np.interp(x_sample, upper[:, 0], upper[:, 1])
    y_lower = np.interp(x_sample, lower[:, 0], lower[:, 1])
    
    return float(np.max(y_upper - y_lower))


def extract_camberline(coords: np.ndarray, n_points: int = 200) -> np.ndarray:
    """Extract camber line from airfoil coordinates."""
    i_le = int(np.argmin(coords[:, 0]))
    upper = coords[:i_le + 1][::-1]
    lower = coords[i_le:]
    
    x_common = np.linspace(0, 1, n_points)
    y_upper = np.interp(x_common, upper[:, 0], upper[:, 1], left=np.nan, right=np.nan)
    y_lower = np.interp(x_common, lower[:, 0], lower[:, 1], left=np.nan, right=np.nan)
    
    valid = np.isfinite(y_upper) & np.isfinite(y_lower)
    camber = np.column_stack((x_common[valid], 0.5 * (y_upper[valid] + y_lower[valid])))
    
    return camber


def has_self_intersection(coords: np.ndarray) -> bool:
    """Check for self-intersecting segments."""
    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    
    def on_segment(a, b, c):
        return (min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))
    
    def segments_intersect(p1, p2, p3, p4):
        o1, o2 = ccw(p1, p2, p3), ccw(p1, p2, p4)
        o3, o4 = ccw(p3, p4, p1), ccw(p3, p4, p2)
        
        if o1 == 0 and on_segment(p1, p2, p3): return True
        if o2 == 0 and on_segment(p1, p2, p4): return True
        if o3 == 0 and on_segment(p3, p4, p1): return True
        if o4 == 0 and on_segment(p3, p4, p2): return True
        
        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)
    
    n = coords.shape[0]
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if i == 0 and j + 1 == n - 1:
                continue
            if segments_intersect(coords[i], coords[i+1], coords[j], coords[j+1]):
                return True
    return False


def _cosine_spacing(n: int) -> np.ndarray:
    """Return cosine-spaced samples from 0→1."""
    if n <= 1:
        return np.zeros(1)
    k = np.linspace(0.0, np.pi, n)
    return 0.5 * (1 - np.cos(k))


def resample_airfoil(coords: np.ndarray, panel_points: int) -> np.ndarray:
    """Resample airfoil using cosine spacing (dense near LE/TE)."""
    panel_points = max(int(panel_points), 4)

    upper, lower = split_upper_lower(coords)

    n_upper = max(2, (panel_points + 2) // 2)
    n_lower = max(2, panel_points - n_upper + 2)

    x_upper = _cosine_spacing(n_upper)
    x_lower = _cosine_spacing(n_lower)

    y_upper = np.interp(x_upper, upper[:, 0], upper[:, 1],
                        left=upper[0, 1], right=upper[-1, 1])
    y_lower = np.interp(x_lower, lower[:, 0], lower[:, 1],
                        left=lower[0, 1], right=lower[-1, 1])

    upper_resampled = np.column_stack((x_upper, y_upper))
    lower_resampled = np.column_stack((x_lower, y_lower))

    upper_TE2LE = upper_resampled[::-1]
    lower_LE2TE = lower_resampled[1:]

    return np.vstack((upper_TE2LE, lower_LE2TE))
