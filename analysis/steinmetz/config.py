"""Paths, analysis constants and the numbers the paper reports."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "analysis"
AUTHORS_REPO = ROOT / "steinmetz-et-al-2019"
ATLAS_DIR = ANALYSIS / "resources" / "atlas"
PAPER_FIGS = ROOT / "origin"


@dataclass(frozen=True)
class Config:
    """Where one run reads from and writes to."""

    data: Path = ROOT / "data" / "allData"
    results: Path = ANALYSIS / "results"
    output: Path = ANALYSIS / "output"
    neuron_lists: Path | None = AUTHORS_REPO / "behavior" / "inclusionData" / "neurons"
    synthetic: bool = False
    min_subjects: int = 2  # regions need at least this many mice
    seed: int = 20191204


# Time
DT = 0.005

# Trial selection
WIN_COUNT_MOVE = (0.125, 0.4)
WIN_NO_MOVE = (-0.05, 0.4)
CONTRASTS = (0.25, 0.5, 1.0)
MIN_MATCHED_TRIALS = 10
MIN_INCLUDABLE_TRIALS = 10

# Neuron selection
MIN_RATE = 0.1
MIN_NEURONS_REGION = 13
ALPHA_RESPONSIVE = 0.05 / 6
SIGNIFICANT_Z = 4.0

# Kernel regression
STIM_WIN = (-0.05, 0.4)
MOVE_WIN = (-0.25, 0.025)
LC = 90
LD = 55
NPRED = 6 * LC + 2 * LD
BASIS_DIMS = 200
FULL_RANKS = 50
FULL_FOLDS = 5
RESID_COMPS = 75
NESTED_RANKS = 18
NESTED_FOLDS = 10
ENET_LAMBDA = 0.5
ENET_ALPHA = 0.5
VE_THRESHOLD = 0.02
NESTED_EVAL = ((0.05, 0.4), (-0.2, 0.0))

# Shuffles and engagement
CCCP_SHUFFLES = 2000
BONFERRONI_REGION = 0.05 / 42
ANOVA_WIN = (-0.25, -0.05)
ENGAGEMENT_WIN = (-0.2, 0.0)

# Wheel: one encoder unit at the wheel surface (Extended Data Fig 1 legend)
WHEEL_MM_PER_UNIT = 0.135

ANATOMY_ORDER = (
    "VISp", "VISl", "VISpm", "VISam", "VISrl", "VISa", "RSP", "ACA", "MOs", "PL", "ILA", "ORB", "MOp", "SSp",
    "DG", "CA3", "CA1", "POST", "SUB", "OLF", "BLA", "CP", "GPe", "SNr", "ACB", "LS",
    "LGd", "LP", "LD", "POL", "MD", "VPL", "PO", "VPM", "RT", "MG", "SCs", "SCm", "MRN", "APN", "PAG", "ZI",
)
LATENCY_ORDER = (
    "SCs", "VISp", "LP", "VISl", "VISpm", "VISam", "VISrl", "VISa", "LD", "CP", "ACA", "MOs", "PL", "GPe", "SCm", "MRN",
    "SNr", "ZI", "APN", "ORB", "ILA", "ACB", "MOp", "RSP", "SSp", "VPL", "LGd", "MD", "PO", "POL", "MG", "RT", "VPM",
    "DG", "CA3", "CA1", "POST", "SUB", "LS", "PAG", "OLF", "BLA",
)
TABLE1_TOTAL = {
    "ACA": 608, "ACB": 255, "APN": 235, "BLA": 261, "CA1": 1129, "CA3": 438, "CP": 914, "DG": 711, "GPe": 274, "ILA": 338,
    "LD": 308, "LGd": 811, "LP": 732, "LS": 844, "MD": 381, "MG": 276, "MOp": 682, "MOs": 1534, "MRN": 857, "OLF": 684,
    "ORB": 770, "PAG": 130, "PL": 728, "PO": 620, "POL": 190, "POST": 272, "RSP": 598, "RT": 160, "SCm": 997, "SCs": 317,
    "SNr": 274, "SSp": 461, "SUB": 669, "VISa": 285, "VISam": 805, "VISl": 403, "VISp": 923, "VISpm": 516, "VISrl": 236,
    "VPL": 255, "VPM": 357, "ZI": 220,
}
TABLE1_RESPONSIVE = {
    "ACA": 395, "ACB": 152, "APN": 177, "BLA": 107, "CA1": 494, "CA3": 243, "CP": 524, "DG": 336, "GPe": 146, "ILA": 192,
    "LD": 209, "LGd": 397, "LP": 393, "LS": 508, "MD": 244, "MG": 180, "MOp": 553, "MOs": 993, "MRN": 722, "OLF": 210,
    "ORB": 281, "PAG": 72, "PL": 438, "PO": 342, "POL": 132, "POST": 163, "RSP": 308, "RT": 111, "SCm": 738, "SCs": 192,
    "SNr": 201, "SSp": 296, "SUB": 494, "VISa": 207, "VISam": 501, "VISl": 248, "VISp": 649, "VISpm": 230, "VISrl": 85,
    "VPL": 202, "VPM": 235, "ZI": 167,
}
TOTAL_NEURONS = 29134
LOCALIZED_NEURONS = 22458
INCLUDED_TRIALS = 9538

# Region groups used by several analyses
FRONTAL = ("MOs", "PL", "MOp")
MIDBRAIN = ("MRN", "SCm", "ZI", "SNr")
STRIATUM = ("CP",)
VIS = ("VISp", "VISpm", "VISl", "VISrl", "VISa", "VISam")
FOREBRAIN = ("MOs", "PL", "MOp", "CP")

ASSIGNMENT_REGIONS = ("VISp", "LGd", "LP")
