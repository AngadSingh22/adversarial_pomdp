""""""
from __future__ import annotations
import numpy as np

def compute_action_mask(env) -> np.ndarray:
    """"""
    return env.get_action_mask()