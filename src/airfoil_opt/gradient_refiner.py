"""
Gradient-based refinement using SLSQP to maximize Expected Improvement.
"""
import numpy as np
from scipy.optimize import minimize
import pickle
from typing import Dict, Any, Tuple
from . import config
from .ffd_geometry import FFDBox2D
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params
from .geometry_utils import extract_camberline, normalize_unit_chord
from .objective_function import score_design, Weights
from .optimization_utils import (
    pack_dofs, unpack_dofs, create_constraints, _setup_bounds,
    calculate_expected_improvement
)

def run_gradient_refinement(best_lhs: Dict, surrogate, params: Analysis_Params, seed_airfoil: np.ndarray, result_name: str) -> Dict:
    """Refine design using SLSQP to maximize Expected Improvement."""
    optimizer = SLSQPOptimizer(best_lhs, surrogate, params, seed_airfoil, result_name)
    return optimizer.optimize()

class SLSQPOptimizer:
    """
    SLSQP-based optimizer for the EGO acquisition function.
    This class finds the next best point to sample by maximizing Expected
    Improvement based on the surrogate model's predictions.
    """
    
    def __init__(self, best_lhs, surrogate, params, seed_airfoil, result_name):
        self.best_lhs = best_lhs
        self.surrogate = surrogate
        self.params = params
        self.weights = Weights()
        self.seed_airfoil = seed_airfoil
        self.result_name = result_name
        self.J_best = best_lhs['J']
        self.eval_count = 0
        self.best_vec = None
        self.max_ei = -float('inf')
        self.opt_history = []
    
    def optimize(self) -> Dict:
        """Execute the optimization."""
        print(f"Starting EGO refinement. Current best J = {self.J_best:.4f}")
        
        ffd_box = FFDBox2D.from_airfoil(self.seed_airfoil, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
        base_deltas = self.best_lhs.get('dP', np.zeros_like(ffd_box.control_points)).copy()
        mask = _control_dof_mask(ffd_box)
        
        # This factory function creates a closure that remembers the state
        # of the FFD box, seed airfoil, base deltas, and mask.
        self.vec_to_coords = self.vec_to_coords_factory(ffd_box, self.seed_airfoil, base_deltas, mask)
        
        vec0 = np.zeros(mask.sum())
        bounds = _setup_bounds(mask, config.OPT_X_BOUND, config.OPT_Y_BOUND)
        constraints = create_constraints(self.vec_to_coords)
        self.best_vec = vec0.copy()
        
        print("Running SLSQP to maximize Expected Improvement...")
        self.best_opt = vec0.copy()
        vec_opt, _ = self._run_slsqp(vec0, bounds, constraints)
        
        # Use the best vector found during the search for final verification
        final_vec = self.best_vec if self.best_vec is not None else vec_opt
        coords_opt, total_deltas = self.vec_to_coords(final_vec)
        
        xfoil_out, J_final = self._verify_xfoil(coords_opt)
        
        with config.OPT_HISTORY_PATH.open('wb') as f: pickle.dump(self.opt_history, f)
        
        return {
            'name': self.result_name, 'xy': coords_opt, 'dP': total_deltas,
            'xfoil': xfoil_out, 'J': float(J_final), 'camberline': extract_camberline(coords_opt),
            'params': self.best_lhs.get('params', {}),
            'max_ei': self.max_ei
        }

    @staticmethod
    def vec_to_coords_factory(ffd_box, base_coords, base_deltas, mask):
        """
        Creates a function that correctly combines the baseline displacements
        with new displacements from the optimizer vector.
        """
        def vec_to_coords(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            additional_deltas = unpack_dofs(mask, vec)
            # THIS IS THE CRITICAL FIX: Sum the new change with the previous best
            total_deltas = base_deltas + additional_deltas
            coords = ffd_box.deform(base_coords, control_deltas=total_deltas)
            return normalize_unit_chord(coords), total_deltas
        return vec_to_coords
    
    def _objective(self, vec: np.ndarray) -> float:
        """EGO objective: computes negative Expected Improvement to be minimized."""
        self.eval_count += 1
        if self.eval_count > config.MAX_SLSQP_ITERS: raise StopIteration("Max iterations reached")
            
        coords, _ = self.vec_to_coords(vec)
        aero_pred = self.surrogate.predict(coords, return_std=True)
        J_pred = score_design(aero_pred, coords, self.weights)
        
        cl_std = aero_pred.get('CL_max_std', 0.0)
        cd_std = aero_pred.get('CD_std', 0.0)
        # Composite uncertainty, weighted by importance in objective function
        sigma_pred = max(self.weights.w_cl * cl_std + self.weights.w_cd * cd_std, 1e-6)
        
        # Scale uncertainty by exploration factor to encourage creativity
        creative_sigma = sigma_pred * config.EXPLORATION_FACTOR
        ei = calculate_expected_improvement(J_pred, creative_sigma, self.J_best)
        
        self.opt_history.append({'iteration': self.eval_count, 'J_pred': J_pred, 'sigma_pred': creative_sigma, 'EI': ei})
        
        if ei > self.max_ei:
            self.max_ei = ei
            self.best_vec = vec.copy()
        
        if self.eval_count % 10 == 1: print(f"  [{self.eval_count:02d}] Pred J={J_pred:.3f}, CreativeSigma={creative_sigma:.4f}, EI={ei:.4e}")
        
        return -ei  # Minimize negative EI -> Maximize EI
    
    def _run_slsqp(self, vec0, bounds, constraints):
        try:
            result = minimize(self._objective, vec0, method='SLSQP', bounds=bounds, constraints=constraints,
                              options={'maxiter': config.MAX_SLSQP_ITERS, 'ftol': config.FTOL, 'eps': config.EPSILON})
            return result.x, result.success
        except StopIteration:
            print(f"Stopped at iteration {config.MAX_SLSQP_ITERS}")
            return self.best_vec, False

    def _verify_xfoil(self, coords):
        """Run a final XFOIL analysis on the candidate design."""
        print("\nRunning XFOIL verification on the new candidate design...")
        runner = Xfoil_Analysis(self.result_name, {"xy": coords}, self.params)
        runner._write_airfoil_dat()
        runner._write_xfoil_input()
        
        xfoil_out = runner._parse_polar() if runner._run_xfoil() else {"converged": False}
        J_final = score_design(xfoil_out, coords, self.weights)
        
        print(f"XFOIL Verified: J_actual={J_final:.3f}")
        return xfoil_out, J_final

def _control_dof_mask(ffd):
    nx, ny, _ = ffd.control_points.shape
    mask = np.zeros((nx, ny, 2), dtype=bool)
    mask[1:-1, :, 1] = True # y-displacements are active for all internal points
    if config.ENABLE_X_DOFS: 
        mask[1:-1, :, 0] = True # x-displacements if enabled
    return mask
