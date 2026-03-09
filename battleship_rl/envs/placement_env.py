from __future__ import annotations
from typing import Optional, Sequence
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from battleship_rl.envs.placement import decode_placement_action, _normalize_board_size
from battleship_rl.envs.battleship_env import BattleshipEnv

class BattleshipPlacementEnv(gym.Env):
    """"""
    metadata = {'render_modes': ['ansi']}

    def __init__(self, board_size: int | Sequence[int]=10, ships: Optional[Sequence[int]]=None, attacker_model: Optional[str]=None):
        super().__init__()
        self.height, self.width = _normalize_board_size(board_size)
        if ships is None:
            ships = [5, 4, 3, 3, 2]
        self.ships = list(ships)
        self.attacker_model_path = attacker_model
        self.attacker_agent = None
        self.attacker_needs_legacy_obs = False
        if attacker_model:
            from sb3_contrib import MaskablePPO
            try:
                self.attacker_agent = MaskablePPO.load(attacker_model)
                attacker_obs_shape = self.attacker_agent.observation_space.shape
                self.attacker_needs_legacy_obs = attacker_obs_shape[0] == 3
                if self.attacker_needs_legacy_obs:
                    print(f'[COMPAT] Attacker expects 3-channel obs, will slice 4-channel to 3')
            except Exception as e:
                print(f'Warning: Could not load attacker {attacker_model}: {e}')
        self.current_ship_idx = 0
        self.board = np.zeros((self.height, self.width), dtype=np.int32) - 1
        self.action_space = spaces.Discrete(2 * self.height * self.width)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, self.height, self.width), dtype=np.float32)

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        self.current_ship_idx = 0
        self.board.fill(-1)
        return (self._get_obs(), self._get_info())

    def _get_obs(self):
        obs = np.zeros((3, self.height, self.width), dtype=np.float32)
        obs[0] = (self.board != -1).astype(np.float32)
        if self.current_ship_idx < len(self.ships):
            length = self.ships[self.current_ship_idx]
            obs[1] = float(length) / max(self.ships)
            obs[2] = float(self.current_ship_idx) / len(self.ships)
        return obs

    def _get_info(self):
        return {'action_mask': self.action_masks()}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.current_ship_idx >= len(self.ships):
            return mask
        length = self.ships[self.current_ship_idx]
        for r in range(self.height):
            for c in range(self.width):
                idx_h = r * self.width + c
                if c + length <= self.width:
                    if np.all(self.board[r, c:c + length] == -1):
                        mask[idx_h] = True
                idx_v = self.height * self.width + idx_h
                if r + length <= self.height:
                    if np.all(self.board[r:r + length, c] == -1):
                        mask[idx_v] = True
        return mask

    def step(self, action):
        mask = self.action_masks()
        if not mask[action]:
            return (self._get_obs(), -50.0, True, False, self._get_info())
        action = int(action)
        length = self.ships[self.current_ship_idx]
        r, c, orientation = decode_placement_action(action, self.height, self.width)
        if orientation == 0:
            self.board[r, c:c + length] = self.current_ship_idx
        else:
            self.board[r:r + length, c] = self.current_ship_idx
        self.current_ship_idx += 1
        if self.current_ship_idx >= len(self.ships):
            reward = self._evaluate_layout()
            return (self._get_obs(), reward, True, False, self._get_info())
        return (self._get_obs(), 0.0, False, False, self._get_info())

    def _evaluate_layout(self) -> float:
        """"""

        class OneOffDefender:

            def __init__(self, arr):
                self.arr = arr

            def sample_layout(self, *args):
                return self.arr
        env = BattleshipEnv(board_size=(self.height, self.width), ships=self.ships, defender=OneOffDefender(self.board.copy()), debug=False)
        obs, _ = env.reset()
        max_steps = self.height * self.width
        shots = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and shots < max_steps:
            if self.attacker_agent:
                mask = env.get_action_mask()
                obs_for_attacker = obs[:3] if self.attacker_needs_legacy_obs else obs
                action, _ = self.attacker_agent.predict(obs_for_attacker, action_masks=mask, deterministic=True)
            else:
                mask = env.get_action_mask()
                candidates = np.flatnonzero(mask)
                action = np.random.choice(candidates)
            obs, _, terminated, truncated, _ = env.step(action)
            shots += 1
        return float(shots)