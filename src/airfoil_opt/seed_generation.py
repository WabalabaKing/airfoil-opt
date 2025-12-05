"""generate seed airfoil(s) for optimization."""
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from .geometry_utils import normalize_unit_chord
from . import config

def generate_seed_airfoils():
    """load all available seed airfoils from the dat directory."""
    if not config.SEED_DAT_DIR.exists():
        raise FileNotFoundError(f"seed .dat files directory not found: {config.SEED_DAT_DIR}")
    
    dat_files = sorted(config.SEED_DAT_DIR.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"no .dat files found in {config.SEED_DAT_DIR}")
    
    seeds = []
    for dat_file in dat_files:
        coords = _read_airfoil_dat(dat_file)
        if coords.size == 0: continue
        
        normalized_coords = normalize_unit_chord(coords)
        seeds.append({"name": dat_file.stem, "xy": normalized_coords})
        print(f"loaded seed: {dat_file.name} ({len(normalized_coords)} points)")
    
    return seeds

def _read_airfoil_dat(filepath: Path) -> np.ndarray:
    """read airfoil coordinates from .dat file, skipping headers."""
    try:
        # numpy is much faster and more robust for this task
        return np.loadtxt(filepath, skiprows=1)
    except (ValueError, IndexError):
        # fallback for unusually formatted files
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # find the first line that contains two numbers
        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) == 2:
                try:
                    float(parts[0]), float(parts[1])
                    # read from this line onwards
                    return np.loadtxt(lines[i:])
                except ValueError:
                    continue
        return np.array([])