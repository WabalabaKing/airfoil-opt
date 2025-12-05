"""
LHS engine to generate initial design space samples.
"""
import numpy as np
from scipy.stats import qmc
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Union
from . import config
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params
from .ffd_geometry import FFDBox2D
from .geometry_utils import normalize_unit_chord, extract_camberline, thickness_ratio, resample_airfoil
from .objective_function import score_design, Weights

SeedType = Union[np.ndarray, Dict[str, np.ndarray]]

def run_lhs_sampling(seed_airfoils: List[SeedType], params: Analysis_Params) -> List[Dict]:
    """Orchestrates the LHS sampling and evaluation."""
    n_dofs_per_coord = 2 if config.ENABLE_X_DOFS else 1
    n_dofs = (config.FFD_NX - 2) * config.FFD_NY * n_dofs_per_coord
    print(f"FFD Grid: ({config.FFD_NX}, {config.FFD_NY}), DOFs: {n_dofs}")

    seed_arrays = []
    for seed in seed_airfoils:
        if isinstance(seed, dict):
            xy = seed.get("xy")
            if xy is None:
                raise ValueError("Seed dictionary is missing 'xy' coordinates.")
        else:
            xy = seed
        seed_arrays.append(np.asarray(xy))

    designs = _generate_designs(seed_arrays, n_dofs, params.PANEL_POINTS)
    print(f"Generated {len(designs)} valid initial geometries.")

    prepared_args = []
    for name, info in designs:
        runner = Xfoil_Analysis(name, info, params)
        runner._write_airfoil_dat()
        runner._write_xfoil_input()
        prepared_args.append((name, info['xy'], info.get('dP'), runner.dat_file, runner.input_file, runner.polar_file, params))

    n_proc = min(cpu_count(), len(prepared_args))
    print(f"Running XFOIL on {len(prepared_args)} samples using {n_proc} cores...")
    with Pool(processes=n_proc) as pool:
        raw_results = pool.map(_worker, prepared_args)

    weights = Weights()
    records = []

    for i, (name, coords, dP, camber, aero_res) in enumerate(raw_results):
        records.append({
            "name": name, "xy": coords, "dP": dP, "camberline": camber,
            "xfoil": aero_res, "J": float(score_design(aero_res, coords, weights)),
            "params": designs[i][1]["params"] 
        })

    print("[OK] All XFOIL tasks completed.")
    return records

def _generate_designs(seed_airfoils: List[np.ndarray], n_dofs: int, panel_points: int) -> List:
    """Create a list of design dictionaries from LHS samples."""
    designs, idx = [], 0
    samples_per_seed = config.LHS_SAMPLES // len(seed_airfoils)
    sampler = qmc.LatinHypercube(d=n_dofs, seed=42)
    
    for s_idx, seed_xy in enumerate(seed_airfoils):
        ffd = FFDBox2D.from_airfoil(seed_xy, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
        lhs_samples = sampler.random(n=samples_per_seed)
        
        for j in range(samples_per_seed):
            coords, deltas = _apply_ffd(ffd, seed_xy, lhs_samples[j], panel_points)
            if thickness_ratio(coords) > 0.06: # Simple validity check
                designs.append((f"lhs_{idx:03d}", {
                    "params": {"seed_idx": s_idx},
                    "xy": coords, "dP": deltas, "camberline": extract_camberline(coords)
                }))
                idx += 1
    return designs

def _apply_ffd(ffd: FFDBox2D, seed: np.ndarray, lhs_sample: np.ndarray, panel_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Applies a single LHS sample to an FFD box."""
    dP = np.zeros_like(ffd.control_points)
    k = 0
    
    for i in range(1, config.FFD_NX - 1):
        for j in range(config.FFD_NY):
            if config.ENABLE_X_DOFS:
                dP[i, j, 0] = (lhs_sample[k] - 0.5) * 2 * config.LHS_X_BOUND
                k += 1
            dP[i, j, 1] = (lhs_sample[k] - 0.5) * 2 * config.LHS_Y_BOUND
            k += 1
    
    xy = ffd.deform(seed, control_deltas=dP)
    xy = normalize_unit_chord(xy)
    xy = resample_airfoil(xy, panel_points)
    return xy, dP

def _worker(args: Tuple) -> Tuple:
    """A single parallel worker that runs XFOIL."""
    name, coords, dP, dat_path, input_path, polar_path, params = args
    runner = Xfoil_Analysis(name, {"xy": coords}, params)
    runner.dat_file, runner.input_file, runner.polar_file = dat_path, input_path, polar_path
    
    aero_res = runner._parse_polar() if runner._run_xfoil() else {"converged": False}
    if aero_res["converged"]:
        print(f"[XFOIL DONE] {name} -> polar written (OK)")
    else:
        print(f"[XFOIL FAIL] {name} -> no convergence or timeout")
    
    return name, coords, dP, extract_camberline(coords), aero_res
