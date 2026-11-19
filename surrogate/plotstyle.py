"""Shared matplotlib style (validated categorical palette, recessive axes)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_COLOR = {"MLP": "#2a78d6", "RF": "#eb6834", "GP": "#1baf7a",
               "PolyRidge": "#6250d6", "GBM": "#eda100", "ExtraTrees": "#e87ba4"}
MODEL_LABEL = {"MLP": "MLP (deep ensemble)", "RF": "Random forest", "GP": "Gaussian process",
               "PolyRidge": "Polynomial ridge", "GBM": "Gradient boosting",
               "ExtraTrees": "Extra trees"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATA_GREY = "#9b9a94"
FEAT_LABEL = {"Ra": "Ra", "Da": "Da", "R": "R", "A": "A", "C": "C",
              "log10(Ra*Da)": r"log$_{10}$(Ra·Da)"}


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
        "font.family": "DejaVu Sans", "mathtext.default": "regular",
    })
