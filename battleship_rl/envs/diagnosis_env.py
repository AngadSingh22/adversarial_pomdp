""""""
from __future__ import annotations
from typing import Any, Dict, Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
_DEFAULT_CHANNELS = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1]], dtype=np.float32)

class DiagnosisEnv(gym.Env):
    """"""
    metadata = {'render_modes': ['ansi']}

    def __init__(self, n_components: int=8, n_tests: int=6, channels: Optional[np.ndarray]=None, fp_rate: float=0.05, fn_rate: float=0.1, step_penalty: float=-0.05, max_steps: int=20, fault_distribution: str='uniform') -> None:
        super().__init__()
        self.n_components = n_components
        self.n_tests = n_tests
        self.channels = channels if channels is not None else _DEFAULT_CHANNELS[:n_tests, :n_components]
        assert self.channels.shape == (n_tests, n_components), f'channels must be ({n_tests}, {n_components}), got {self.channels.shape}'
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        assert fault_distribution in ('uniform', 'clustered', 'rare_hard'), f'Unknown fault_distribution: {fault_distribution!r}'
        self.fault_distribution = fault_distribution
        coverage = self.channels.sum(axis=0)
        self._rare_hard_subset = list(np.argsort(coverage)[:2])
        self._clustered_subset = list(range(min(3, n_components)))
        self.action_space = spaces.Discrete(n_tests + n_components)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(n_tests,), dtype=np.float32)
        self._faulty: int = 0
        self._obs: np.ndarray = np.full(n_tests, -1.0, dtype=np.float32)
        self._steps: int = 0

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None) -> tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if self.fault_distribution == 'uniform':
            self._faulty = int(self.np_random.integers(0, self.n_components))
        elif self.fault_distribution == 'clustered':
            self._faulty = int(self.np_random.choice(self._clustered_subset))
        elif self.fault_distribution == 'rare_hard':
            self._faulty = int(self.np_random.choice(self._rare_hard_subset))
        else:
            self._faulty = int(self.np_random.integers(0, self.n_components))
        self._obs = np.full(self.n_tests, -1.0, dtype=np.float32)
        self._steps = 0
        return (self._obs.copy(), {'faulty_component': self._faulty})

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)
        if action >= self.n_tests:
            declared = action - self.n_tests
            correct = declared == self._faulty
            reward = 1.0 if correct else -1.0
            info = {'outcome': 'correct' if correct else 'wrong', 'declared': declared, 'faulty': self._faulty}
            return (self._obs.copy(), reward, True, False, info)
        test_idx = action
        self._steps += 1
        component_in_channel = bool(self.channels[test_idx, self._faulty] > 0.5)
        if component_in_channel:
            result = 0 if self.np_random.random() < self.fn_rate else 1
        else:
            result = 1 if self.np_random.random() < self.fp_rate else 0
        self._obs[test_idx] = float(result)
        truncated = self._steps >= self.max_steps
        info = {'test_idx': test_idx, 'result': result, 'true_in_channel': component_in_channel}
        return (self._obs.copy(), self.step_penalty, False, truncated, info)

    def render(self) -> str:
        lines = [f'Faulty component: ??? (hidden)   Steps: {self._steps}/{self.max_steps}']
        for i, v in enumerate(self._obs):
            status = {-1.0: 'untested', 0.0: 'negative', 1.0: 'positive'}.get(v, '?')
            lines.append(f'  Test {i}: {status}')
        return '\n'.join(lines)

    def get_action_mask(self) -> np.ndarray:
        """"""
        mask = np.ones(self.n_tests + self.n_components, dtype=bool)
        for i in range(self.n_tests):
            if self._obs[i] != -1.0:
                mask[i] = False
        return mask