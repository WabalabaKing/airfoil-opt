"""
unified visualization and diagnostic plotting utilities.
"""
import numpy as np
import pandas as pd
import matplotlib
import pickle

matplotlib.use("agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
from matplotlib.ticker import MaxNLocator
from . import config
from .ffd_geometry import FFDBox2D
from .seed_generation import generate_seed_airfoils

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.get_cmap("Dark2").colors)
palette = plt.get_cmap("Dark2").colors

def create_all_plots(
    seed_airfoils: List[Dict],
    best_lhs: Dict,
    final_design: Dict,
    all_lhs_results: List[Dict],
    convergence_diagnostics: List[Dict]
):    
    """generate all standard diagnostic plots from an optimization run."""
    print("\n--- generating plots ---")
    plot_geometry_comparison(seed_airfoils, best_lhs, final_design)
    plot_polar_data(best_lhs.get("name"), final_design.get("name"))
    plot_optimization_convergence(all_lhs_results)
    plot_lhs_performance_distribution(all_lhs_results, best_lhs, final_design)
    plot_seed_airfoils(seed_airfoils)
    plot_convergence_diagnostics(convergence_diagnostics)
    print("all standard plots saved successfully.")

def plot_convergence_diagnostics(diagnostics_data: List[Dict]):
    """
    plots the optimality gap and max expected improvement over iterations.
    """
    if not diagnostics_data:
        print("no convergence diagnostic data to plot.")
        return

    df = pd.DataFrame(diagnostics_data).dropna()
    if df.empty:
        print("diagnostic data is empty after removing nans; cannot plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("active learning convergence diagnostics", fontsize=14, fontweight="bold")

    # plot optimality gap on the left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('active learning iteration')
    ax1.set_ylabel('optimality gap', color=color1)
    ax1.plot(df['iteration'], df['optimality_gap'], marker='o', linestyle='-', color=color1, label='optimality gap')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
    ax1.axhline(y=config.ACTIVE_J_TOL, color=color1, linestyle=':', linewidth=2, label=f'gap tolerance ({config.ACTIVE_J_TOL})')
    
    # instantiate a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()  
    color2 = 'tab:red'
    ax2.set_ylabel('max expected improvement (regret)', color=color2)
    ax2.plot(df['iteration'], df['max_expected_improvement'], marker='s', linestyle='--', color=color2, label='max ei (regret)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_yscale('log')
    ax2.axhline(y=config.EI_REGRET_TOLERANCE, color=color2, linestyle=':', linewidth=2, label=f'ei tolerance ({config.EI_REGRET_TOLERANCE})')

    # combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = config.IMAGES_DIR / "convergence_diagnostics.png"
    fig.savefig(save_path, dpi=250)
    plt.close(fig)
    print(f"saved: {save_path}")

def plot_geometry_comparison(seed_airfoils, best_lhs, final):
    """compare best lhs vs final optimal geometries with their deformed ffd grids."""
    if not best_lhs or not final:
        print("skipping geometry plot; missing best_lhs or final design.")
        return
        
    seed_idx_best = best_lhs["params"]["seed_idx"]
    seed_idx_final = final.get("params", {}).get("seed_idx", seed_idx_best)
    base_seed_best = seed_airfoils[seed_idx_best]["xy"]
    base_seed_final = seed_airfoils[seed_idx_final]["xy"]

    best_coords = best_lhs["xy"]
    final_coords = final["xy"]

    ffd_best = FFDBox2D.from_airfoil(base_seed_best, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
    ffd_final = FFDBox2D.from_airfoil(base_seed_final, pad=(config.FFD_PAD_X, config.FFD_PAD_Y), grid=(config.FFD_NX, config.FFD_NY))
    cp_best = ffd_best.control_points + best_lhs.get("dp", np.zeros_like(ffd_best.control_points))
    cp_final = ffd_final.control_points + final.get("dp", np.zeros_like(ffd_final.control_points))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    _plot_airfoil_ffd(ax1, best_coords, cp_best, f"best lhs • {best_lhs.get('name', 'unknown')}")
    _plot_airfoil_ffd(ax2, final_coords, cp_final, "final optimal geometry")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(config.IMAGES_DIR / "final_geometry_comparison.png", dpi=250)
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'final_geometry_comparison.png'}")

def plot_polar_data(best_lhs_name, final_name):
    """generate polar plots comparing best lhs vs final optimal."""
    if not best_lhs_name or not final_name:
        print("skipping polar plot; design names are missing.")
        return

    df_best = _read_polar(config.POLAR_DIR / f"{best_lhs_name}_polar.txt")
    df_final = _read_polar(config.OPTIMAL_DIR / "opt_final_polar.txt")

    if df_best.empty or df_final.empty:
        print("skipping polar plot; one or both polar files could not be read.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    axs = axes.flatten()
    keys = [("alpha", "cl"), ("alpha", "cd"), ("alpha", "cm"), ("cd", "cl")]
    titles = ["lift vs. alpha", "drag vs. alpha", "moment vs. alpha", "drag polar"]

    for ax, (x, y), title in zip(axs, keys, titles):
        ax.plot(df_best[x], df_best[y], lw=1.8, label="best lhs", color=palette[0])
        ax.plot(df_final[x], df_final[y], lw=2.0, label="final optimal", color=palette[1])
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
    print(f"saved: {config.IMAGES_DIR / 'final_polar_comparison.png'}")

def plot_optimization_convergence(all_results: List[Dict]):
    if not config.OPT_HISTORY_PATH.exists():
        print(f"skipping ego convergence plot; {config.OPT_HISTORY_PATH} not found.")
        return
    with config.OPT_HISTORY_PATH.open("rb") as f: history = pickle.load(f)
    df = pd.DataFrame(history)

    running = []
    best_so_far = np.inf
    for idx, res in enumerate(all_results):
        J_val = res.get("J")
        if J_val is None: 
            continue
        best_so_far = min(best_so_far, J_val)
        phase = "lhs" if str(res.get("name", "")).startswith("lhs") else "active"
        running.append({"eval": idx + 1, "J": J_val, "best": best_so_far, "phase": phase})
    run_df = pd.DataFrame(running)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    axes = axes.flatten()

    axes[0].plot(run_df["eval"], run_df["best"], lw=2.0, label="running best J")
    axes[0].scatter(run_df["eval"], run_df["J"], s=25, alpha=0.7, label="sampled J")
    axes[0].set_title("running best objective (all samples)")
    axes[0].set_xlabel("evaluation #")
    axes[0].set_ylabel("objective J")
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(frameon=False)

    axes[1].plot(df["iteration"], df["J_pred"], marker="o", lw=2.0, label="predicted J")
    axes[1].set_title("ego predicted objective per slsqp iteration")
    axes[1].set_xlabel("optimizer iteration")
    axes[1].set_ylabel("predicted J")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(frameon=False)

    if "sigma_pred" in df.columns:
        axes[2].plot(df["iteration"], df["sigma_pred"], marker="s", lw=2.0, label="exploration σ")
        axes[2].set_ylabel("scaled predictive σ")
    if "EI" in df.columns:
        axes[2].plot(df["iteration"], df["EI"], marker="^", lw=2.0, label="expected improvement")
    axes[2].set_title("exploration vs. improvement signals")
    axes[2].set_xlabel("optimizer iteration")
    axes[2].grid(True, alpha=0.3, linestyle="--")
    axes[2].legend(frameon=False)

    if "EI" in df.columns:
        axes[3].bar(df["iteration"], df["EI"], width=0.6)
        axes[3].set_title("expected improvement (bar view)")
        axes[3].set_xlabel("optimizer iteration")
        axes[3].set_ylabel("EI")
        axes[3].grid(True, alpha=0.25, linestyle="--", axis="y")
    else:
        axes[3].axis("off")

    fig.suptitle("objective & acquisition convergence (ego)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.IMAGES_DIR / "optimization_convergence.png", dpi=250)
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'optimization_convergence.png'}")

def plot_lhs_performance_distribution(lhs_results, best_lhs: Dict = None, final_design: Dict = None):
    converged_xfoil = [r['xfoil'] for r in lhs_results if r.get("xfoil", {}).get("converged")]
    if not converged_xfoil:
        print("skipping lhs distribution plot; no converged xfoil results found.")
        return
    df = pd.DataFrame(converged_xfoil)
    
    best_xfoil = best_lhs.get("xfoil", {}) if best_lhs else {}
    final_xfoil = final_design.get("xfoil", {}) if final_design else {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    axes[0].hist(df['CL_max'], bins=15, alpha=0.8)
    axes[0].set_title("CL_max distribution")
    axes[0].set_xlabel("CL_max")
    axes[0].set_ylabel("count")
    _mark_hist_lines(axes[0], best_xfoil.get("CL_max"), final_xfoil.get("CL_max"), palette)

    axes[1].hist(df['CD'], bins=15, alpha=0.8)
    axes[1].set_title("CD at CL_max distribution")
    axes[1].set_xlabel("CD")
    _mark_hist_lines(axes[1], best_xfoil.get("CD"), final_xfoil.get("CD"), palette)

    axes[2].hist(df['CM'], bins=15, alpha=0.8)
    axes[2].set_title("CM at CL_max distribution")
    axes[2].set_xlabel("CM")
    _mark_hist_lines(axes[2], best_xfoil.get("CM"), final_xfoil.get("CM"), palette)

    scatter = axes[3].scatter(df['CD'], df['CL_max'], c=df['CM'], cmap="viridis", s=45, edgecolors="white", linewidths=0.4, alpha=0.9)
    axes[3].set_title("drag polar colored by CM")
    axes[3].set_xlabel("CD")
    axes[3].set_ylabel("CL_max")

    if best_xfoil:
        axes[3].scatter(best_xfoil.get("CD"), best_xfoil.get("CL_max"), color=palette[0], s=80, marker="d", label="best lhs", edgecolors="k", zorder=5)
    if final_xfoil:
        axes[3].scatter(final_xfoil.get("CD"), final_xfoil.get("CL_max"), color=palette[1], s=120, marker="*", label="final optimal", edgecolors="k", zorder=5)
    if best_xfoil or final_xfoil:
        axes[3].legend(frameon=False)

    fig.colorbar(scatter, ax=axes[3], label="CM at CL_max")

    for ax in axes: ax.grid(True, alpha=0.3, linestyle="--")
    fig.suptitle("lhs performance distribution (converged samples)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(config.IMAGES_DIR / "lhs_performance_distribution.png", dpi=250)
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'lhs_performance_distribution.png'}")

def plot_seed_airfoils(seed_airfoils: List[Dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(seed_airfoils)))
    for i, seed_dict in enumerate(seed_airfoils):
        ax.plot(seed_dict["xy"][:, 0], seed_dict["xy"][:, 1], color=colors[i], lw=2.0, label=seed_dict["name"])

    ax.legend()
    ax.set_title("seed airfoils", fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(config.IMAGES_DIR / "seed_airfoils.png", dpi=250)
    plt.close(fig)
    print(f"saved: {config.IMAGES_DIR / 'seed_airfoils.png'}")

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
    try:
        df = pd.read_csv(
            polar_path, sep=r'\s+', skiprows=12, header=None,
            names=['alpha', 'cl', 'CD', 'CDp', 'cm', 'top_xtr', 'bot_xtr'],
            on_bad_lines='skip', dtype=float,
        )
        # column names in pandas are case-sensitive, ensure lowercase
        df.columns = [c.lower() for c in df.columns]
        return df.sort_values("alpha").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _mark_hist_lines(ax, best_val, final_val, palette):
    """annotate histogram with vertical markers for best and final designs."""
    if best_val is None and final_val is None: return
    handles = []
    labels = []
    if best_val is not None:
        h_best = ax.axvline(best_val, color=palette[0], lw=2.0, ls="--")
        handles.append(h_best); labels.append("best lhs")
    if final_val is not None:
        h_final = ax.axvline(final_val, color=palette[1], lw=2.2, ls="-.")
        handles.append(h_final); labels.append("final optimal")
    ax.legend(handles, labels, frameon=False)
