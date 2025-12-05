"""Generate seed airfoil(s) for optimization."""
import numpy as np
from pathlib import Path
from typing import List
from .geometry_utils import normalize_unit_chord

def generate_seed_airfoils(seed_parent: Path) -> List[np.ndarray]:
    """Load all available seed airfoils from the specified .dat directory."""
    if not seed_parent.exists():
        raise FileNotFoundError(f"Seed airfoil directory not found: {seed_parent}")

    seed_dat_dir = seed_parent / "dat_files"
    if not seed_dat_dir.exists():
        raise FileNotFoundError(f"Seed .dat files directory not found: {seed_dat_dir}")
    seed_dat_dir.mkdir(parents=True, exist_ok=True)    
    
    dat_files = sorted(seed_dat_dir.glob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {seed_dat_dir}")
    
    seeds = []
    for dat_file in dat_files:
        coords = _read_airfoil_dat(dat_file)
        coords = normalize_unit_chord(coords)
        seeds.append(coords)
        print(f"Loaded seed: {dat_file.name} ({len(coords)} points)")
    
    return seeds

def _read_airfoil_dat(filepath: Path) -> np.ndarray:
    """Read airfoil coordinates from .dat file."""
    with open(filepath, 'r') as f:
        # Skip header line(s) by finding the first line with two float values
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    # This is the first valid coordinate line
                    first_coord_line = line
                    break
                except ValueError:
                    continue
        else: # No valid coordinate lines found
            return np.array([])
        
        # Read the rest of the file from this point
        rest_of_file = f.readlines()
        all_lines = [first_coord_line] + rest_of_file

    coords = [list(map(float, line.split())) for line in all_lines if len(line.split()) == 2]
    return np.array(coords)