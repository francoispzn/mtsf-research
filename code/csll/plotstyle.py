"""Shared publication-quality matplotlib style (IEEE-like: Times/STIX, clean spines).

Import and call ``apply()`` before plotting so every figure in the paper is visually consistent.
Kept deliberately conservative -- serif STIX fonts to match the IEEEtran body text, hairline
spines, subtle grid, vector PDF output at 300 dpi.
"""
from __future__ import annotations

import matplotlib as mpl

# A restrained, colour-blind-safe palette (Okabe-Ito derived) used across figures.
PALETTE = {
    "csll":   "#c1272d",   # CSLL highlight (deep crimson)
    "blue":   "#2e6fb7",
    "green":  "#2a9d54",
    "orange": "#e08a1e",
    "purple": "#7562a6",
    "grey":   "#5a5a5a",
    "teal":   "#159090",
    "ink":    "#1a1a1a",
}

# Semantic, FIXED per-entity styling so a model looks identical in every figure.
# (colour, marker, linestyle, zorder, display label)
MODEL_STYLE = {
    "CSLL2H":         dict(c="#c1272d", m="*",  ls="-",  z=6, lw=2.0, ms=10, label="CSLL2H (hybrid)"),
    "CSLL2":          dict(c="#e0703a", m="o",  ls="-",  z=5, lw=1.8, ms=6,  label="CSLL2 (phase)"),
    "CSLL2-phaseoff": dict(c="#159090", m="D",  ls="--", z=4, lw=1.5, ms=5,  label="mixing"),
    "CSLL2-static":   dict(c="#c9a227", m="v",  ls=":",  z=3, lw=1.3, ms=5,  label="static phase"),
    "CSLL2-backbone": dict(c="#8a8a8a", m="s",  ls=":",  z=2, lw=1.3, ms=5,  label="CI backbone"),
    "iTransformer":   dict(c="#2e6fb7", m="^",  ls="--", z=4, lw=1.5, ms=6,  label="iTransformer"),
    "PatchTST":       dict(c="#7562a6", m="P",  ls="--", z=3, lw=1.4, ms=6,  label="PatchTST"),
    "DLinear":        dict(c="#5a5a5a", m="X",  ls=":",  z=2, lw=1.2, ms=5,  label="DLinear"),
    "NLinear":        dict(c="#9c9c9c", m="x",  ls=":",  z=2, lw=1.2, ms=5,  label="NLinear"),
    "SeasonalNaive":  dict(c="#c2c2c2", m=".",  ls=":",  z=1, lw=1.0, ms=5,  label="Seasonal naive"),
    "VAR":            dict(c="#b0a08a", m="1",  ls=":",  z=1, lw=1.0, ms=5,  label="VAR"),
}

# Warm hues for influenza (phase-essential), cool for traffic (mixing) -- reinforces the dichotomy.
DATASET_COLOR = {
    "ili_us_state": "#c1272d", "ili_japan": "#e0703a", "ili_us_hhs": "#e0a53a",
    "metr_la": "#2e6fb7", "pems_bay": "#2a9d9d", "pems04": "#5a7fb7", "pems08": "#7b8fa8",
}

# IEEE column geometry (inches): single column ~3.45in, full text width ~7.16in.
COL_W = 3.45
DBL_W = 7.16


def style(name):
    return MODEL_STYLE.get(name, dict(c="#777", m="o", ls="-", z=2, lw=1.3, ms=5, label=name))


def apply():
    mpl.rcParams.update({
        # output
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,            # embed TrueType (editable/searchable), no Type-3
        "ps.fonttype": 42,
        # fonts -- STIX is Times-like and ships with matplotlib (matches IEEE body text)
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        # axes / spines
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#2b2b2b",
        "axes.labelcolor": "#111111",
        "axes.titlepad": 5.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#c9c9c9",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.color": "#2b2b2b",
        "ytick.color": "#2b2b2b",
        # lines / legend
        "lines.linewidth": 1.7,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.6,
        "lines.markeredgecolor": "#1a1a1a",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#d5d5d5",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.5,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    return PALETTE
