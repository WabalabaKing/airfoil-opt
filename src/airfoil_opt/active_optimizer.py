"""
Main optimization pipeline with active-learning loop.
"""
import pickle
import time
import shutil
from pathlib import Path
from scipy.stats import norm
from copy import deepcopy
from . import config
from .seed_generation import generate_seed_airfoils
from .surrogate_model import train_surrogate_models
from .gradient_refiner import run_gradient_refinement
from .visualization import create_all_plots
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params
from .objective_function import score_design
from .geometry_utils import thickness_ratio, extract_camberline
from .lhs_engine import run_lhs_sampling

def load_pickle_required(path: Path):
    if not path.exists(): raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("rb") as f: return pickle.load(f)

class ActiveAirfoilOptimizer:
    """Orchestrates the EGO optimization pipeline."""
    def __init__(self):
        self.params = Analysis_Params()
        self.seed_airfoils = []
        self.lhs_results = []
        self.best_lhs = {}
        self.initial_best_lhs = {}
        self.final_design = {}

    def run(self):
        """Execute the full optimization pipeline."""
        self._setup_directories()
        self._pipeline_start_time = time.perf_counter()
        print("\n" + "=" * 64 + "\nACTIVE AIRFOIL OPTIMIZATION PIPELINE\n" + "=" * 64)

        self._step_1_load_seeds()
        self._step_2_run_lhs()
        self._step_3_active_loop()
        self._step_4_visualize()

        total_time = time.perf_counter() - self._pipeline_start_time
        print(f"\nPipeline completed in {total_time/60:.1f} min")

    def _setup_directories(self):
        config.IMAGES_DIR.mkdir(exist_ok=True)
        config.DATA_DIR.mkdir(exist_ok=True)
        if config.OPTIMAL_DIR.exists(): shutil.rmtree(config.OPTIMAL_DIR)
        (config.OPTIMAL_DIR / "dat").mkdir(parents=True)
        (config.OPTIMAL_DIR / "input").mkdir(parents=True)
        (config.OPTIMAL_DIR / "polar").mkdir(parents=True)

    def _step_1_load_seeds(self):
        print("\n=== STEP 1: Loading Seed Airfoils ===")
        self.seed_airfoils = generate_seed_airfoils(config.SEED_AIRFOILS_DIR)
        print(f"Loaded {len(self.seed_airfoils)} seed airfoil(s).")

    def _step_2_run_lhs(self):
        print("\n=== STEP 2: Latin Hypercube Sampling ===")
        if config.REUSE_LHS_RESULTS and config.LHS_RESULTS_PATH.exists():
            print(f"Loading cached LHS from {config.LHS_RESULTS_PATH}")
            self.lhs_results = load_pickle_required(config.LHS_RESULTS_PATH)
        else:
            Xfoil_Analysis.init_dirs()
            from .lhs_engine import run_lhs_sampling
            self.lhs_results = run_lhs_sampling(self.seed_airfoils, self.params)
            with config.LHS_RESULTS_PATH.open("wb") as f: pickle.dump(self.lhs_results, f)
        
        self.best_lhs = self._get_best_record()
        self.initial_best_lhs = deepcopy(self.best_lhs)
        if not self.best_lhs: raise RuntimeError("No converged LHS samples found.")
        print(f"LHS phase complete. Best initial design: {self.best_lhs.get('name')} (J={self.best_lhs.get('J'):.4f})")

    def _step_3_active_loop(self):
        print("\n=== STEP 3: Active Surrogate Refinement Loop ===")
        Xfoil_Analysis.DAT_DIR = config.OPTIMAL_DIR / "dat"
        Xfoil_Analysis.INPUTS_DIR = config.OPTIMAL_DIR / "input"
        Xfoil_Analysis.POLAR_DIR = config.OPTIMAL_DIR / "polar"

        best_J_history = [self.best_lhs['J']]
        for k in range(config.ACTIVE_MAX_ITERS):
            print(f"\n--- Active iteration {k + 1} / {config.ACTIVE_MAX_ITERS} ---")
            best_before_iter = self._get_best_record()
            surrogate = train_surrogate_models(self.lhs_results)
            
            seed_idx = best_before_iter.get("params", {}).get("seed_idx", 0)
            
            new_design = run_gradient_refinement(
                best_lhs=best_before_iter,
                surrogate=surrogate,
                params=self.params,
                seed_airfoil=self.seed_airfoils[seed_idx],
                result_name=f"opt_{k+1:02d}"
            )
            
            # ... (handling of failed XFOIL run is the same) ...
            
            self.lhs_results.append(new_design)
            with config.LHS_RESULTS_PATH.open("wb") as f: pickle.dump(self.lhs_results, f)
            best_J_history.append(self._get_best_record()['J'])

            # --- UPDATED CONVERGENCE CHECK ---
            if config.USE_ADVANCED_CONVERGENCE:
                if self._check_advanced_convergence(new_design, surrogate):
                    break
            else: # Fallback to the simple heuristic
                if self._check_simple_convergence(best_J_history, surrogate, new_design):
                    break
        
        self.final_design = self._get_best_record()
        self._copy_final_outputs()
        self._print_final_results()
    
    def _step_4_visualize(self):
        print("\n=== STEP 4: Generating Visualizations ===")

        seeds_with_names = [
            {"name": f.stem, "xy": coords}
            for f, coords in zip(sorted(config.SEED_DAT_DIR.glob("*.dat")), self.seed_airfoils)
        ]

        create_all_plots(
            seed_airfoils=seeds_with_names,
            best_lhs=self.initial_best_lhs,
            final_design=self.final_design,
            all_lhs_results=self.lhs_results
        )

    def _check_convergence(self, J_history, surrogate, new_design) -> bool:
        # don't check convergence if we failed xfoil
        if not new_design.get("xfoil", {}).get("converged"):
            return False
        
        pred = surrogate.predict(new_design["xy"], return_std=True)
        cl_std = pred.get("CL_max_std")
        delta_J = J_history[-2] - J_history[-1]
        
        uncertainty_str = f"{cl_std:.4f}" if cl_std is not None else "N/A"
        print(f"dJ from this iteration = {delta_J:.4e} | CL_max uncertainty = {uncertainty_str}")

        if abs(delta_J) < config.ACTIVE_J_TOL and (cl_std is None or cl_std < config.ACTIVE_UNCERTAINTY_TOL):
            print("Converged: Negligible improvement and low uncertainty.")
            return True
        
        patience = 2
        if len(J_history) > patience and abs(J_history[-1 - patience] - J_history[-1]) < config.ACTIVE_J_TOL:
            print(f"Converged: Best score has stagnated for {patience} iterations.")
            return True
        return False

    def _get_best_record(self):
        return min(self.lhs_results, key=lambda r: r["J"]) if self.lhs_results else {}

    def _check_simple_convergence(self, J_history, surrogate, new_design) -> bool:
        # This is the old method, kept for comparison
        if not new_design.get("xfoil", {}).get("converged"):
            return False
    
    def _check_advanced_convergence(self, new_design, surrogate) -> bool:
        """
        Checks for convergence based on regret (max_ei) and sample complexity (LCB).
        """
        # 1. Regret-Based Check
        # Stop if the maximum possible expected improvement from the next sample is negligible.
        max_ei = new_design.get('max_ei', float('inf'))
        print(f"Max Expected Improvement (Regret) from this iteration: {max_ei:.4e}")
        if max_ei < config.EI_REGRET_TOLERANCE:
            print(f"Converged: Maximum potential regret is below tolerance ({config.EI_REGRET_TOLERANCE}).")
            return True
        
        # LCB = mu(x) - beta * sigma(x). Minimizing this is a common acquisition function.
        all_coords = [r['xy'] for r in self.lhs_results]
        predictions = [surrogate.predict(xy, return_std=True) for xy in all_coords]

        z_score = norm.ppf(config.CONFIDENCE_LEVEL)
        
        lcb_values = [
            p.get('CL_max', 0) - z_score * p.get('CL_max_std', 0)
            for p in predictions if 'CL_max' in p and 'CL_max_std' in p
        ]
        
        if not lcb_values: return False

        optimistic_best_J = min(lcb_values)
        current_best_J = self._get_best_record()['J']

        optimality_gap = current_best_J - optimistic_best_J
        print(f"Optimality Gap (Current Best J - Optimistic Best J): {optimality_gap:.4f}")

        if optimality_gap < config.ACTIVE_J_TOL:
            print(f"Converged: Optimality gap is below tolerance ({config.ACTIVE_J_TOL}).")
            return True
            
        return False

    
    def _copy_final_outputs(self):
        name = self.final_design.get("name")
        if not name: return
        dat_src = Xfoil_Analysis.DAT_DIR / f"{name}.dat"
        polar_src = Xfoil_Analysis.POLAR_DIR / f"{name}_polar.txt"
        if dat_src.exists(): shutil.copyfile(dat_src, config.OPTIMAL_DIR / "opt_final.dat")
        if polar_src.exists(): shutil.copyfile(polar_src, config.OPTIMAL_DIR / "opt_final_polar.txt")

    def _print_final_results(self):
        xf = self.final_design.get("xfoil", {})
        if not xf or not xf.get("converged"):
            print("\nOptimization did not converge to a valid design.")
            return
        
        print("\n" + "=" * 25 + " Optimization Complete " + "=" * 25)
        print(f"Final Design: {self.final_design.get('name')} | J={self.final_design.get('J'):.4f}")
        print(f"  CL_max = {xf.get('CL_max'):.4f} at alpha = {xf.get('alpha_CL_max'):.2f} deg")
        print(f"  CD at CL_max = {xf.get('CD'):.5f} | (L/D)_max = {xf.get('LD_max'):.2f}")
        print(f"  CM at CL_max = {xf.get('CM'):.4f} | t/c = {thickness_ratio(self.final_design.get('xy')):.4f}")

def main():
    optimizer = ActiveAirfoilOptimizer()
    optimizer.run()

if __name__ == "__main__":
    main()
