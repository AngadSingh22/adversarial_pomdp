""""""
from __future__ import annotations
from typing import Optional
import numpy as np

class RandomTester:
    """"""

    def __init__(self, rng: Optional[np.random.Generator]=None) -> None:
        self.rng = rng or np.random.default_rng()
        self._belief: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._belief = None

    def act(self, obs: np.ndarray, info: dict, env) -> int:
        n_tests = env.n_tests
        n_comp = env.n_components
        untested = [i for i in range(n_tests) if obs[i] == -1.0]
        valid_actions = untested + list(range(n_tests, n_tests + n_comp))
        return int(self.rng.choice(valid_actions))

class GreedySplitTester:
    """"""

    def __init__(self, rng: Optional[np.random.Generator]=None, confidence_threshold: float=0.92) -> None:
        self.rng = rng or np.random.default_rng()
        self.confidence_threshold = confidence_threshold
        self._belief: Optional[np.ndarray] = None
        self._n_components: int = 0
        self._n_tests: int = 0
        self._channels: Optional[np.ndarray] = None
        self._fp_rate: float = 0.05
        self._fn_rate: float = 0.1

    def reset(self) -> None:
        self._belief = None

    def _init_from_env(self, env) -> None:
        self._n_components = env.n_components
        self._n_tests = env.n_tests
        self._channels = env.channels
        self._fp_rate = env.fp_rate
        self._fn_rate = env.fn_rate
        self._belief = np.ones(self._n_components, dtype=np.float64)

    def _update_belief(self, test_idx: int, result: int) -> None:
        """"""
        channels = self._channels
        fp, fn = (self._fp_rate, self._fn_rate)
        in_channel = channels[test_idx].astype(np.float64)
        if result == 1:
            likelihood = in_channel * (1 - fn) + (1 - in_channel) * fp
        else:
            likelihood = in_channel * fn + (1 - in_channel) * (1 - fp)
        self._belief *= likelihood
        total = self._belief.sum()
        if total > 0:
            self._belief /= total
        else:
            self._belief[:] = 1.0 / self._n_components

    def _choose_best_test(self, tested: set[int]) -> int:
        """"""
        channels = self._channels
        belief = self._belief
        best_score = -1.0
        best_t = -1
        for t in range(self._n_tests):
            if t in tested:
                continue
            in_mass = float(np.dot(channels[t], belief))
            out_mass = 1.0 - in_mass
            score = min(in_mass, out_mass)
            if score > best_score:
                best_score = score
                best_t = t
        return best_t if best_t >= 0 else next((t for t in range(self._n_tests) if t not in tested))

    def act(self, obs: np.ndarray, info: dict, env) -> int:
        if self._belief is None:
            self._init_from_env(env)
        n_tests = env.n_tests
        tested = {i for i in range(n_tests) if obs[i] != -1.0}
        self._belief = np.ones(self._n_components, dtype=np.float64)
        for i in tested:
            self._update_belief(i, int(obs[i]))
        if len(tested) > 0 and self._belief.max() >= self.confidence_threshold:
            best_comp = int(np.argmax(self._belief))
            return n_tests + best_comp
        if len(tested) < n_tests:
            return self._choose_best_test(tested)
        best_comp = int(np.argmax(self._belief))
        return n_tests + best_comp