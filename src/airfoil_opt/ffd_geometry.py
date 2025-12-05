import numpy as np
from scipy.special import comb
from dataclasses import dataclass
from typing import Tuple, Optional

def bernstein(n: int, i: int, u: np.ndarray) -> np.ndarray:
    """Bernstein polynomial B(n,i) evaluated at points u."""
    return comb(n, i) * (u ** i) * ((1 - u) ** (n - i))

def batch_bernstein(n: int, u: np.ndarray) -> np.ndarray:
    """Compute all Bernstein basis functions for degree n at points u."""
    basis = np.empty((len(u), n + 1))
    for i in range(n + 1):
        basis[:, i] = bernstein(n, i, u)
    return basis

@dataclass
class FFDBox2D:
    """2D Free-Form Deformation box with Bezier control lattice."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    control_points: np.ndarray

    @staticmethod
    def from_airfoil(coords: np.ndarray, 
                     pad: Tuple[float, float] = (0.05, 0.2), 
                     grid: Tuple[int, int] = (5, 5)) -> "FFDBox2D":
        """Create FFD box around airfoil coordinates."""
        x_min = -pad[0]
        x_max = 1.0 + pad[0]
        y_min = float(np.min(coords[:, 1]) - pad[1])
        y_max = float(np.max(coords[:, 1]) + pad[1])

        nx, ny = grid
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)

        control_points = np.zeros((nx, ny, 2))
        for i, x in enumerate(x_grid):
            for j, y in enumerate(y_grid):
                control_points[i, j] = [x, y]

        return FFDBox2D(x_min, x_max, y_min, y_max, nx, ny, control_points)
    
    def deform(self, coords: np.ndarray, control_deltas: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply FFD deformation to input coordinates using control point displacements."""
        active_points = self.control_points if control_deltas is None else self.control_points + control_deltas
        active_points = active_points.copy()
        active_points[[0, -1], :, :] = self.control_points[[0, -1], :, :]


        s = np.clip((coords[:, 0] - self.x_min) / (self.x_max - self.x_min), 0.0, 1.0)
        t = np.clip((coords[:, 1] - self.y_min) / (self.y_max - self.y_min), 0.0, 1.0)

        basis_s = batch_bernstein(self.nx - 1, s)
        basis_t = batch_bernstein(self.ny - 1, t)

        weights = np.einsum('ni,nj->nij', basis_s, basis_t)  # (N, nx, ny)
        deformed = np.einsum('nij,ijk->nk', weights, active_points)  # (N, 2)
        
        return deformed