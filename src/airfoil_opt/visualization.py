"""
Unified visualization and diagnostic plotting utilities.
"""
import numpy as np
import pandas as pd
import matplotlib
import pickle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from matplotlib.ticker import MaxNLocator
from . import config
from .surrogate_model import MultiOutputGP
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params
from .ffd_geometry import FFDBox2D
from .seed_generation import generate_seed_airfoils

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.get_cmap("Dark2").colors)
palette = plt.get_cmap("Dark2").colors

def create_all_plots(
    seed_airfoils: List[Dict],
    best_lhs: Dict,
    final_design: Dict,
    all_lhs_results: List[Dict]
):    
    """Generate all final and diagnostic plots."""
    plot_geometry_comparison(seed_airfoils, best_lhs, final_design)
    plot_polar_data(best_lhs["name"], final_design["name"])
    plot_optimization_convergence(all_lhs_results)
    plot_lhs_performance_distribution(all_lhs_results, best_lhs, final_design)
    plot_seed_airfoils(seed_airfoils)

def plot_geometry_comparison(baseline, best_lhs, final):
    """Compare best LHS vs final optimal geometries with their deformed FFD grids."""
    seeds = _load_seed_library()
    seed_idx_best = best_lhs["params"]["seed_idx"]
    seed_idx_final = final.get("params", {}).get("seed_idx", seed_idx_best)
    base_seed_best = seeds[seed_idx_best]["xy"]
    base_seed_final = seeds[seed_idx_final]["xy"]

    best_coords = best_lhs["xy"]
    final_coords = final["xy"]

    ffd_best = FFDBox2D.from_airfoil(base_seed_best, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
    ffd_final = FFDBox2D.from_airfoil(base_seed_final, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
    cp_best = ffd_best.control_points + best_lhs.get("dP", np.zeros_like(ffd_best.control_points))
    cp_final = ffd_final.control_points + final.get("dP", np.zeros_like(ffd_final.control_points))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    _plot_airfoil_ffd(ax1, best_coords, cp_best, f"Best LHS • {best_lhs.get('name', 'unknown')}")
    _plot_airfoil_ffd(ax2, final_coords, cp_final, "Final Optimal Geometry")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.IMAGES_DIR / "final_geometry_comparison.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'final_geometry_comparison.png'}")

def plot_polar_data(best_lhs_name, final_name):
    """Generate polar plots comparing best LHS vs final optimal."""
    df_best = _read_polar(config.POLAR_DIR / f"{best_lhs_name}_polar.txt")
    df_final = _read_polar(config.OPTIMAL_DIR / "opt_final_polar.txt")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    axs = axes.flatten()
    keys = [("alpha", "CL"), ("alpha", "CD"), ("alpha", "CM"), ("CD", "CL")]
    titles = ["Lift vs. Alpha", "Drag vs. Alpha", "Moment vs. Alpha", "Drag Polar"]

    for ax, (x, y), title in zip(axs, keys, titles):
        ax.plot(df_best[x], df_best[y], lw=1.8, label="Best LHS", color=palette[0])
        ax.plot(df_final[x], df_final[y], lw=2.0, label="Final Optimal", color=palette[1])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, prune=None))
        ax.legend(frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.IMAGES_DIR / "final_polar_comparison.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'final_polar_comparison.png'}")

def plot_optimization_convergence(all_results: List[Dict]):
    with config.OPT_HISTORY_PATH.open("rb") as f: history = pickle.load(f)
    df = pd.DataFrame(history)

    running = []
    best_so_far = np.inf
    for idx, res in enumerate(all_results):
        J_val = res.get("J")
        if J_val is None: 
            continue
        best_so_far = min(best_so_far, J_val)
        phase = "LHS" if str(res.get("name", "")).startswith("lhs") else "Active"
        running.append({"eval": idx + 1, "J": J_val, "best": best_so_far, "phase": phase})
    run_df = pd.DataFrame(running)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    axes[0, 0].plot(run_df["eval"], run_df["best"], lw=2.0, label="Running Best J")
    axes[0, 0].scatter(run_df["eval"], run_df["J"], s=25, alpha=0.7, label="Sampled J")
    axes[0, 0].set_title("Running Best Objective (All Samples)")
    axes[0, 0].set_xlabel("Evaluation #")
    axes[0, 0].set_ylabel("Objective J")
    axes[0, 0].grid(True, alpha=0.3, linestyle="--")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(df["iteration"], df["J_pred"], marker="o", lw=2.0, label="Predicted J")
    axes[0, 1].set_title("EGO Predicted Objective per SLSQP Iteration")
    axes[0, 1].set_xlabel("Optimizer Iteration")
    axes[0, 1].set_ylabel("Predicted J")
    axes[0, 1].grid(True, alpha=0.3, linestyle="--")
    axes[0, 1].legend(frameon=False)

    if "sigma_pred" in df.columns:
        axes[1, 0].plot(df["iteration"], df["sigma_pred"], marker="s", lw=2.0, label="Exploration σ")
        axes[1, 0].set_ylabel("Scaled Predictive σ")
    if "EI" in df.columns:
        axes[1, 0].plot(df["iteration"], df["EI"], marker="^", lw=2.0, label="Expected Improvement")
    axes[1, 0].set_title("Exploration vs. Improvement Signals")
    axes[1, 0].set_xlabel("Optimizer Iteration")
    axes[1, 0].grid(True, alpha=0.3, linestyle="--")
    axes[1, 0].legend(frameon=False)

    if "EI" in df.columns:
        axes[1, 1].bar(df["iteration"], df["EI"], width=0.6)
        axes[1, 1].set_title("Expected Improvement (Bar View)")
        axes[1, 1].set_xlabel("Optimizer Iteration")
        axes[1, 1].set_ylabel("EI")
        axes[1, 1].grid(True, alpha=0.25, linestyle="--", axis="y")
    else:
        axes[1, 1].axis("off")

    fig.suptitle("Objective & Acquisition Convergence (EGO)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.IMAGES_DIR / "optimization_convergence.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'optimization_convergence.png'}")

def plot_lhs_performance_distribution(lhs_results, best_lhs: Dict = None, final_design: Dict = None):
    df = pd.DataFrame([r['xfoil'] for r in lhs_results])
    best_xfoil = best_lhs.get("xfoil", {}) if best_lhs else {}
    final_xfoil = final_design.get("xfoil", {}) if final_design else {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(df['CL_max'], bins=15, alpha=0.8)
    axes[0, 0].set_title("CL_max Distribution")
    axes[0, 0].set_xlabel("CL_max")
    axes[0, 0].set_ylabel("Count")
    _mark_hist_lines(axes[0, 0], best_xfoil.get("CL_max"), final_xfoil.get("CL_max"), palette)

    axes[0, 1].hist(df['CD'], bins=15, alpha=0.8)
    axes[0, 1].set_title("CD at CL_max Distribution")
    axes[0, 1].set_xlabel("CD")
    _mark_hist_lines(axes[0, 1], best_xfoil.get("CD"), final_xfoil.get("CD"), palette)

    axes[1, 0].hist(df['CM'], bins=15, alpha=0.8)
    axes[1, 0].set_title("CM at CL_max Distribution")
    axes[1, 0].set_xlabel("CM")
    _mark_hist_lines(axes[1, 0], best_xfoil.get("CM"), final_xfoil.get("CM"), palette)

    scatter = axes[1, 1].scatter(df['CD'], df['CL_max'], c=df['CM'], cmap="Dark2", s=45, edgecolors="white", linewidths=0.4)
    axes[1, 1].set_title("Drag Polar Colored by CM")
    axes[1, 1].set_xlabel("CD")
    axes[1, 1].set_ylabel("CL_max")

    if best_xfoil:
        axes[1, 1].scatter(best_xfoil.get("CD"), best_xfoil.get("CL_max"), color=palette[0], s=70, marker="D", label="Best LHS")
    if final_xfoil:
        axes[1, 1].scatter(final_xfoil.get("CD"), final_xfoil.get("CL_max"), color=palette[1], s=80, marker="*", label="Final Optimal", zorder=4)
    if best_xfoil or final_xfoil:
        axes[1, 1].legend(frameon=False)

    fig.colorbar(scatter, ax=axes[1, 1], label="CM at CL_max")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("LHS Performance Distribution (Converged Samples)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.IMAGES_DIR / "lhs_performance_distribution.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'lhs_performance_distribution.png'}")

def plot_seed_airfoils(seed_airfoils: List[Dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(seed_airfoils)))
    for i, seed_dict in enumerate(seed_airfoils):
        ax.plot(seed_dict["xy"][:, 0], seed_dict["xy"][:, 1], color=colors[i], lw=2.0, label=seed_dict["name"])

    fig.suptitle("Seed Airfoils", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(config.IMAGES_DIR / "seed_airfoils.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'seed_airfoils.png'}")

def plot_cp_distributions(best_lhs: Dict, final_design: Dict):
    """Plot Cp vs x/c for best LHS and final optimal at key angles of attack."""
    cp_dir = Path("aero_outputs")
    cp_dir.mkdir(exist_ok=True)

    # Clean only files we are about to regenerate
    for pattern in ("cp_best_*", "cp_final_*", "cp_dat", "cp_input", "cp_polar"):
        for path in cp_dir.glob(pattern):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path, ignore_errors=True)

    base_angles = [0.0, 5.0, 10.0]
    stall_best = best_lhs.get("xfoil", {}).get("alpha_CL_max")
    stall_final = final_design.get("xfoil", {}).get("alpha_CL_max")

    best_curves = _generate_cp_curves(best_lhs, base_angles + ([stall_best] if stall_best is not None else []), "cp_best", cp_dir)
    final_curves = _generate_cp_curves(final_design, base_angles + ([stall_final] if stall_final is not None else []), "cp_final", cp_dir)

    fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
    axes = axes.flatten()

    subplot_angles = base_angles + [None]  # last one reserved for stall plot
    for ax, target_angle in zip(axes, subplot_angles):
        if target_angle is None:
            ax.set_title("Stall Angle Cp")
            best_df, best_label = _pick_cp_curve(best_curves, stall_best, fallback=None, label_prefix="Best LHS")
            final_df, final_label = _pick_cp_curve(final_curves, stall_final, fallback=None, label_prefix="Final Optimal")
        else:
            ax.set_title(f"Cp at α = {target_angle:.0f}°")
            best_df, best_label = _pick_cp_curve(best_curves, target_angle, fallback=target_angle, label_prefix="Best LHS")
            final_df, final_label = _pick_cp_curve(final_curves, target_angle, fallback=target_angle, label_prefix="Final Optimal")

        if best_df is not None and not best_df.empty:
            ax.plot(best_df["x"], best_df["cp"], lw=2.0, color=palette[0], label=best_label)
        if final_df is not None and not final_df.empty:
            ax.plot(final_df["x"], final_df["cp"], lw=2.0, color=palette[1], label=final_label)

        ax.set_xlabel("x/c")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(frameon=False)

    axes[0].set_ylabel("Cp")
    axes[2].set_ylabel("Cp")
    fig.suptitle("Pressure Coefficient Profiles (Best LHS vs Final Optimal)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.IMAGES_DIR / "cp_distributions.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'cp_distributions.png'} (Cp data in {cp_dir})")

def _plot_airfoil_ffd(ax, coords, control_points, title):
    ax.plot(coords[:, 0], coords[:, 1], color="k", lw=2.0, zorder=4)
    nx, ny, _ = control_points.shape
    flat_cp = control_points.reshape(-1, 2)
    ax.scatter(flat_cp[:, 0], flat_cp[:, 1], color="red", s=16, zorder=3)
    for i in range(nx):
        ax.plot(control_points[i, :, 0], control_points[i, :, 1], color="red", lw=1.0, alpha=0.85, ls="--")
    for j in range(ny):
        ax.plot(control_points[:, j, 0], control_points[:, j, 1], color="red", lw=1.0, alpha=0.85, ls="--")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlabel("x/c")
    ax.set_ylabel("y/c")

def _read_polar(polar_path: Path) -> pd.DataFrame:
    if not polar_path.exists(): return pd.DataFrame()
    return pd.read_csv(
        polar_path, sep=r"\s+", skiprows=12, header=None,
        names=['alpha', 'CL', 'CD', 'CDp', 'CM', 'Top_Xtr', 'Bot_Xtr'],
        on_bad_lines='skip'
    ).sort_values("alpha").reset_index(drop=True)

def _generate_cp_curves(design: Dict, angles: List[float], prefix: str, cp_dir: Path):
    """Run XFOIL Cp dumps for given angles and return (alpha, df) pairs."""
    curves = []
    for alpha in angles:
        cp_file = cp_dir / f"{prefix}_alpha_{alpha:+.1f}.txt"
        df_cp = _run_cp_analysis(design, alpha, cp_file, cp_dir)
        if df_cp is not None:
            curves.append((alpha, df_cp))
    return curves

def _run_cp_analysis(design: Dict, alpha: float, cp_file: Path, cp_dir: Path) -> pd.DataFrame:
    """Write Cp input, run XFOIL, and parse Cp output."""
    # Setup directories
    work_dat = cp_dir / "cp_dat"
    work_inp = cp_dir / "cp_input"
    work_pol = cp_dir / "cp_polar"
    for d in (work_dat, work_inp, work_pol):
        d.mkdir(parents=True, exist_ok=True)

    # Preserve class-level directories
    orig_dirs = (Xfoil_Analysis.DAT_DIR, Xfoil_Analysis.INPUTS_DIR, Xfoil_Analysis.POLAR_DIR)
    
    try:
        Xfoil_Analysis.DAT_DIR = work_dat
        Xfoil_Analysis.INPUTS_DIR = work_inp
        Xfoil_Analysis.POLAR_DIR = work_pol

        coords = design["xy"]
        name = design.get("name", "cp_case")
        runner = Xfoil_Analysis(name, {"xy": coords}, Analysis_Params())
        
        cp_file.unlink(missing_ok=True)
        runner._write_airfoil_dat()
        runner._write_cp_input(alpha, cp_file)
        runner._run_xfoil(timeout=25)
        return _read_cp_file(cp_file)
        
    finally:
        # Always restore global state
        Xfoil_Analysis.DAT_DIR, Xfoil_Analysis.INPUTS_DIR, Xfoil_Analysis.POLAR_DIR = orig_dirs

def _read_cp_file(path: Path) -> pd.DataFrame:
    """Read XFOIL CPWR output: columns x, y, cp."""
    data = []
    with path.open("r") as f:
        for line in f:
            # Skip headers/empty lines based on XFOIL format
            clean_line = line.strip()
            if not clean_line or clean_line.startswith(("#", "Alfa", "opt")):
                continue
            
            parts = clean_line.split()
            data.append([float(x) for x in parts[:3]])

    return pd.DataFrame(data, columns=["x", "y", "cp"])

def _pick_cp_curve(curves: List, target_angle: float, label_prefix: str):
    """Select Cp curve matching target_angle."""
    for alpha, df in curves:
        if abs(alpha - target_angle) < 1e-6:
            return df, f"{label_prefix} (alpha={alpha:.1f}°)"
            
    # If code reaches here, data wasn't "good," but we aren't handling errors.
    return pd.DataFrame(), label_prefix

def plot_surrogate_parity():
    """Plot surrogate vs XFOIL parity for CL_max, CD, and CM."""
    with config.LHS_RESULTS_PATH.open("rb") as f:
        lhs_results = pickle.load(f)
    
    gp = MultiOutputGP.load(config.SURROGATE_MODELS_DIR)
    
    targets = ["CL_max", "CD", "CM"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, key in zip(axes, targets):
        pts_train, pts_val = [], []
        for i, rec in enumerate(lhs_results):
            xf = rec.get("xfoil", {})
            if not xf.get("converged"): 
                continue
            coords = rec.get("xy")
            pred = gp.predict(coords, return_std=False).get(key)
            true = xf.get(key)
            if pred is None or true is None or not (np.isfinite(pred) and np.isfinite(true)):
                continue
            (pts_val if i % 5 == 0 else pts_train).append((true, pred))
        if not pts_train and not pts_val:
            ax.axis("off"); continue
        if pts_train:
            ax.scatter(*zip(*pts_train), color=palette[0], alpha=0.75, s=30, label="Train", edgecolors="white", linewidths=0.3)
        if pts_val:
            ax.scatter(*zip(*pts_val), color=palette[1], alpha=0.85, s=35, label="Validation", marker="s", edgecolors="white", linewidths=0.3)
        all_vals = [t for t,_ in pts_train+pts_val]
        lo, hi = min(all_vals), max(all_vals)
        pad = 0.05 * (hi - lo + 1e-6)
        ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="k", lw=1.0, ls="--", label="Ideal")
        ax.set_title(f"{key} Parity")
        ax.set_xlabel("XFOIL True")
        ax.set_ylabel("Surrogate Pred")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(frameon=False)

    fig.suptitle("Surrogate vs XFOIL Parity (CL_max, CD, CM)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.IMAGES_DIR / "surrogate_parity.png", dpi=250)
    plt.close(fig)
    print(f"Saved: {config.IMAGES_DIR / 'surrogate_parity.png'}")

def _load_seed_library() -> List[Dict[str, Any]]:
    """Load seed airfoils along with their names for plotting."""
    dat_files = sorted(config.SEED_DAT_DIR.glob("*.dat"))
    coords_list = generate_seed_airfoils(config.SEED_AIRFOILS_DIR)
    seeds = []
    for dat_path, coords in zip(dat_files, coords_list):
        seeds.append({"name": dat_path.stem, "xy": coords})
    return seeds

def _mark_hist_lines(ax, best_val, final_val, palette):
    """Annotate histogram with vertical markers for best and final designs."""
    handles = []
    labels = []
    h_best = ax.axvline(best_val, color=palette[0], lw=2.0, ls="--")
    handles.append(h_best); labels.append("Best LHS")
    h_final = ax.axvline(final_val, color=palette[1], lw=2.2, ls="-.")
    handles.append(h_final); labels.append("Final Optimal")
    ax.legend(handles, labels, frameon=False)
