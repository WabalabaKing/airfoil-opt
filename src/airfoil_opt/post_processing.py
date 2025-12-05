"""
Post-processing analysis and plotting script.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import shutil
from . import config
from .surrogate_model import MultiOutputGP
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.get_cmap("Dark2").colors)

def _read_dat(dat_path: Path) -> np.ndarray:
    return np.loadtxt(dat_path, skiprows=1)

def _read_cp_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", skiprows=3, header=None,
        names=["x", "y", "cp"]
    )

def _run_single_cp_analysis(design_coords: np.ndarray, alpha: float, cp_file: Path) -> pd.DataFrame:
    """Isolates the running of a single XFOIL Cp analysis in a clean environment."""
    temp_dir = Path("./temp_cp_run")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    orig_dirs = (Xfoil_Analysis.DAT_DIR, Xfoil_Analysis.INPUTS_DIR, Xfoil_Analysis.POLAR_DIR)
    Xfoil_Analysis.DAT_DIR = Xfoil_Analysis.INPUTS_DIR = Xfoil_Analysis.POLAR_DIR = temp_dir

    try:
        runner = Xfoil_Analysis("cp_run", {"xy": design_coords}, Analysis_Params())
        runner._write_airfoil_dat()
        runner._write_cp_input(alpha, cp_file)
        if runner._run_xfoil(timeout=30) and cp_file.exists():
            return _read_cp_file(cp_file)
    finally:
        Xfoil_Analysis.DAT_DIR, Xfoil_Analysis.INPUTS_DIR, Xfoil_Analysis.POLAR_DIR = orig_dirs
        shutil.rmtree(temp_dir)
    return pd.DataFrame()

def generate_cp_plots():
    """Generates and plots Cp distributions for the final optimal airfoil."""
    print("\n--- Generating Cp Distribution Plots ---")
    final_dat_path = config.OPTIMAL_DIR / "opt_final.dat"
    if not final_dat_path.exists():
        print(f"Warning: Final optimal airfoil file not found at {final_dat_path}. Skipping Cp plots.")
        return

    final_coords = _read_dat(final_dat_path)
    angles = [0.0, 5.0, 10.0, 15.0]
    
    aero_output_dir = config.IMAGES_DIR / "aero_data"
    aero_output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, alpha in zip(axes.flatten(), angles):
        cp_file_output = aero_output_dir / f"cp_alpha_{alpha:.1f}.txt"
        df_cp = _run_single_cp_analysis(final_coords, alpha, cp_file_output)
        
        if not df_cp.empty:
            ax.plot(df_cp["x"], df_cp["cp"], lw=2.0, color="k")
        
        ax.set_title(f"Cp at aoa = {alpha:.1f}")
        ax.set_xlabel("x/c"); ax.invert_yaxis(); ax.grid(True, alpha=0.3, linestyle="--")

    axes[0,0].set_ylabel("Cp (Pressure Coefficient)"); axes[1,0].set_ylabel("Cp (Pressure Coefficient)")
    fig.suptitle("Final Airfoil Pressure Coefficient Profiles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.IMAGES_DIR / "cp_distributions.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'cp_distributions.png'}")

def generate_surrogate_parity_plots():
    """Generates surrogate vs. XFOIL parity plots to validate model accuracy."""
    print("\n--- Generating Surrogate Parity Plots ---")
    if not config.LHS_RESULTS_PATH.exists() or not config.SURROGATE_MODELS_DIR.exists():
        print("Warning: LHS results or surrogate models not found. Skipping parity plots.")
        return

    with config.LHS_RESULTS_PATH.open("rb") as f:
        lhs_results = pickle.load(f)
    gp = MultiOutputGP.load(config.SURROGATE_MODELS_DIR)
    targets = ["CL_max", "CD", "CM"]
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    palette = plt.get_cmap("Dark2").colors

    for ax, key in zip(axes, targets):
        pts_train, pts_val = [], []
        for i, rec in enumerate(lhs_results):
            if not rec.get("xfoil", {}).get("converged"):
                continue
            
            pred = gp.predict(rec["xy"], return_std=False).get(key)
            true = rec["xfoil"].get(key)
            
            # Add this point to either training or validation set
            (pts_val if i % 4 == 0 else pts_train).append((true, pred))

        ax.scatter(*zip(*pts_train), color=palette[0], alpha=0.7, s=30, label="Train", edgecolors="white", linewidths=0.3)
        ax.scatter(*zip(*pts_val), color=palette[1], alpha=0.8, s=40, label="Validation", marker="D", edgecolors="white", linewidths=0.3)
        
        all_vals = [t for t, _ in pts_train + pts_val]
        lo, hi = min(all_vals), max(all_vals)
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="k", lw=1.2, ls="--", label="Ideal")
        
        ax.set_title(f"{key} Parity")
        ax.set_xlabel("XFOIL True Value")
        ax.set_ylabel("Surrogate Predicted Value")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(frameon=False)

    fig.suptitle("Surrogate vs. XFOIL Parity", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.IMAGES_DIR / "surrogate_parity.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'surrogate_parity.png'}")

def run_all_post_processing():
    generate_cp_plots()
    generate_surrogate_parity_plots()
