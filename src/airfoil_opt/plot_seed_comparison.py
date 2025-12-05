"""
This script runs XFOIL on the seed airfoils and plots their aerodynamic
performance for comparison on a single figure with four subplots.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

# Import necessary components from the existing codebase
from .ffd_xfoil_analysis import Xfoil_Analysis, Analysis_Params

SEED_AIRFOILS_DIR = Path("./seed_airfoils")
SEED_DAT_DIR = SEED_AIRFOILS_DIR / "dat_files"
SEED_INPUT_DIR = SEED_AIRFOILS_DIR / "input_files"
SEED_POLAR_DIR = SEED_AIRFOILS_DIR / "polar_files"

def plot_seed_comparison():
    """
    Generates and plots the aerodynamic performance of the 5 seed airfoils.
    """
    print("--- Running XFOIL for Seed Airfoils ---")

    # get the seed names from the seed airfoils directory
    seed_names = [f.stem for f in SEED_DAT_DIR.glob("*.dat")]

    # Standard analysis parameters for the XFOIL run
    params = Analysis_Params(
        alpha_start=-5.0,
        alpha_end=20.0,
        alpha_step=0.5,
        panel_points=120
    )
    # Set the Reynolds and Mach numbers as attributes, which is the expected pattern.
    params.RE_VAL = 5e6
    params.MACH = 0.2

    # check if input and polar directories exist, if not, create them
    # if they already exist, clear them
    if SEED_INPUT_DIR.exists():
        shutil.rmtree(SEED_INPUT_DIR)
    SEED_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if SEED_POLAR_DIR.exists():
        shutil.rmtree(SEED_POLAR_DIR)
    SEED_POLAR_DIR.mkdir(parents=True, exist_ok=True)
    
    polar_files = {}

    # Run XFOIL for each seed airfoil
    for name in seed_names:
        print(f"Running analysis for {name}...")
        # The design 'info' dictionary is minimal for this task
        info = {'name': name}

        xa = Xfoil_Analysis(name, info, params)
        xa.dat_file = SEED_DAT_DIR / f"{name}.dat"
        xa.input_file = SEED_INPUT_DIR / f"{name}_input.txt"
        xa.polar_file = SEED_POLAR_DIR / f"{name}_polar.txt"
        
        xa._write_xfoil_input()
        ok = xa._run_xfoil(timeout=20)
        if not ok:
            print(f"  [Error] Failed to write input for {name}")
            continue

        if ok and xa.polar_file.exists():
            polar_files[name] = xa.polar_file
            print(f"  Polar file generated at {xa.polar_file}")
        else:
            print(f"  [Warning] Polar file not generated for {name}")

    print("\n--- Generating Plots ---")
    
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Seed Airfoil Performance Comparison', fontsize=18)

    # use qualitative colormap
    colors = plt.get_cmap('Dark2').colors

    for i, name in enumerate(seed_names):
        if name not in polar_files:
            continue
            
        # Load the polar data from the text file
        polar_data = np.loadtxt(polar_files[name], skiprows=12)
        
        alpha = polar_data[:, 0]
        cl = polar_data[:, 1]
        cd = polar_data[:, 2]
        cm = polar_data[:, 4]

        # 1. Lift Coefficient vs. Angle of Attack
        axs[0, 0].plot(alpha, cl, label=name, color=colors[i])
        
        # 2. Drag Coefficient vs. Angle of Attack
        axs[0, 1].plot(alpha, cd, label=name, color=colors[i])

        # 3. Pitching Moment vs. Angle of Attack
        axs[1, 0].plot(alpha, cm, label=name, color=colors[i])

        # 4. Drag Polar (CL vs. CD)
        axs[1, 1].plot(cd, cl, label=name, color=colors[i])

    # --- Formatting for all subplots ---
    plot_info = {
        'Lift Coefficient': ('alpha [deg]', 'CL'),
        'Drag Coefficient': ('alpha [deg]', 'CD'),
        'Pitching Moment': ('alpha [deg]', 'CM'),
        'Drag Polar': ('CD', 'CL')
    }
    
    for i, ax in enumerate(axs.flat):
        title = list(plot_info.keys())[i]
        xlabel, ylabel = list(plot_info.values())[i]
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # Set common y-limits for drag for better comparison
    axs[0, 1].set_ylim(bottom=0, top=0.08)
    
    # Set limits for drag polar
    axs[1, 1].set_xlim(left=0, right=0.08)
    axs[1, 1].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the final figure
    output_filename = "seed_airfoil_comparison.png"
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")


if __name__ == '__main__':
    plot_seed_comparison()