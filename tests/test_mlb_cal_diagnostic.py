"""Test all MLB beta cals to see if they crush predictions on synthetic inputs."""
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, '/Users/bpj520/sports-betting-ai')
from src.models.calibrator import BetaCalibrator

# All beta cals that exist
beta_files = {
    'so':   'models/mlb/so_beta_cal.json',
    'hr':   'models/mlb/hr_beta_cal.json',
    'tb':   'models/mlb/tb_beta_cal.json',
    'h':    'models/mlb/h_beta_cal.json',
    'r':    'models/mlb/r_beta_cal.json',
    'rbi':  'models/mlb/rbi_beta_cal.json',
    'sb':   'models/mlb/sb_beta_cal.json',
    'bb':   'models/mlb/bb_beta_cal.json',
    '1b':   'models/mlb/1b_beta_cal.json',
    '2b':   'models/mlb/2b_beta_cal.json',
    '3b':   'models/mlb/3b_beta_cal.json',
}

print(f"{'Stat':6s} {'a':>7s} {'b':>7s} {'c':>7s}  | {'p=0.5':>8s} {'p=0.7':>8s} {'p=0.9':>8s} {'p=0.95':>8s} {'p=0.99':>8s}  {'Verdict':>10s}")
print('-' * 110)
crushing_cals = []
for stat, path in beta_files.items():
    if not Path(path).exists():
        print(f"{stat:6s}  (missing)")
        continue
    bc = BetaCalibrator.load(Path(path))
    results = [float(bc(p)) for p in [0.5, 0.7, 0.9, 0.95, 0.99]]

    # Crushing diagnosis: p_raw=0.9 should give p_cal > 0.5
    # p_raw=0.7 should give p_cal > 0.4
    if results[2] < 0.5:
        verdict = "CRUSHING"
        crushing_cals.append(stat)
    elif results[2] < 0.7:
        verdict = "compressing"
    else:
        verdict = "OK"
    print(f"{stat:6s} {bc.a:7.3f} {bc.b:7.3f} {bc.c:7.3f}  | "
          f"{results[0]:8.4f} {results[1]:8.4f} {results[2]:8.4f} {results[3]:8.4f} {results[4]:8.4f}  {verdict:>10s}")

print()
print(f"CRUSHING CALS: {crushing_cals}")
print()
print("CRUSHING DIAGNOSIS:")
print("- p_raw=0.9 -> p_cal < 0.5 means SEVERE compression (crushing)")
print("- p_raw=0.9 -> p_cal < 0.7 means moderate compression")
print("- Normal: p_cal should be close to p_raw for high p_raw values")
