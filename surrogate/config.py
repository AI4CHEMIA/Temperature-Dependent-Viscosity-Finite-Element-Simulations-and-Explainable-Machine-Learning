"""Central configuration: paths, variables, base case and split rules.

Every script imports from here so that a single edit changes the whole study.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "data_ML_paper1.xlsx"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
FIG = RESULTS / "figures"

SEED = 2026

# ---------------------------------------------------------------- variables
ALL_INPUTS = ["Ra", "Da", "R", "A", "C", "Pr", "N"]
# Output columns as named in the spreadsheet -> safe file names
OUTPUTS = {"avgNu": "avgNu", "|psi|_max": "psi_max"}
TARGETS = list(OUTPUTS.values())                 # ["avgNu", "psi_max"]
TARGET_LABEL = {"avgNu": r"$\overline{Nu}$", "psi_max": r"$|\psi|_{max}$"}

# Inputs spanning several decades are modelled in log10 space.
LOG_INPUTS = ["Ra", "Da"]
# Both outputs are strictly positive and multiplicative in nature
# (power-law-like scaling), so they are modelled as log10(y).
LOG_TARGETS = True

# --------------------------------------------------------------- split rules
# Fractions are of the number of points in each one-at-a-time sweep.
EXT_TEST_FRAC = 0.07   # outermost 7 % at EACH end of every sweep -> test (extrapolation)
EXT_VAL_FRAC = 0.05    # next 5 % inward at each end               -> validation (extrapolation)
INTERP_PERIOD = 9      # interior points: index % 9 == 4 -> test, index % 9 == 8 -> validation
INTERP_TEST_PHASE = 4
INTERP_VAL_PHASE = 8

REL_TOL = 1e-6         # tolerance used to compare inputs with the base case
