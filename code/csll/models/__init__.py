"""Model registry. Every model maps (B, seq_len, N) -> (B, pred_len, N)."""
from __future__ import annotations

from typing import Any, Dict

from .csll import CSLL
from .linear import DLinear, NLinear
from .naive import SeasonalNaive
from .lstm import LSTMForecaster
from .transformer import VanillaTransformer
from .patchtst import PatchTST
from .itransformer import ITransformer

TORCH_MODELS = {
    "CSLL": CSLL,
    "DLinear": DLinear,
    "NLinear": NLinear,
    "LSTM": LSTMForecaster,
    "Transformer": VanillaTransformer,
    "PatchTST": PatchTST,
    "iTransformer": ITransformer,
    # SeasonalNaive is a parameter-free nn.Module; it flows through the normal path
    # (the training loop detects no trainable params and only evaluates).
    "SeasonalNaive": SeasonalNaive,
    # CSLLX = CSLL with a strong iTransformer backbone (rework to beat SOTA): the complex
    # spectral lead-lag module is grafted onto iTransformer as a gated additive correction.
    "CSLLX": CSLL,
    # CSLL2/CSLL2X = v2 estimator (strict delay bound, zero-padded linear-shift DFT, warm
    # gate, cross-correlation delay initialisation). Separate names keep results/raw and
    # aggregation for the v1 paper artifacts untouched.
    "CSLL2": CSLL,
    "CSLL2X": CSLL,
    "CSLL2H": CSLL,     # hybrid: gated direct-phase + gated real-mixing branches
}

# VAR lives in a separate (non-torch) statsmodels path; see var.py and run_experiment.py.
NON_TORCH_MODELS = ["VAR"]


def build_model(name: str, cfg: Dict[str, Any]):
    """Instantiate a torch model by name from a config dict."""
    if name not in TORCH_MODELS:
        raise KeyError(f"unknown torch model '{name}'. Known: {list(TORCH_MODELS)}")
    return TORCH_MODELS[name](**cfg)
