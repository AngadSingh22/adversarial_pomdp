""""""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from battleship_rl.agents.defender import UniformRandomDefender
from battleship_rl.envs.battleship_env import BattleshipEnv

def build_layout_pool(pool_size: int, board_size: int=10, ships: Optional[List[int]]=None, seed: int=0) -> np.ndarray:
    """"""
    if ships is None:
        ships = [5, 4, 3, 3, 2]
    rng = np.random.default_rng(seed)
    defender = UniformRandomDefender()
    H = board_size
    layouts = np.zeros((pool_size, H, H), dtype=np.int32)
    for i in range(pool_size):
        layout = defender.sample_layout((H, H), ships, rng)
        layouts[i] = layout.astype(np.int32)
    return layouts

def evaluate_attacker_on_layout(layout: np.ndarray, attacker_policy, k_episodes: int=1, n_parallel: int=1, board_size: int=10, ships: Optional[List[int]]=None, seed: int=0) -> Tuple[float, List[int]]:
    """"""
    if ships is None:
        ships = [5, 4, 3, 3, 2]
    if n_parallel <= 1:
        shot_counts: List[int] = []
        env = BattleshipEnv(board_size=board_size, ships=ships, debug=False)
        max_steps = board_size * board_size
        try:
            for ep in range(k_episodes):
                obs, _ = env.reset(seed=seed + ep)
                env.ship_id_grid = layout.astype(np.int32)
                env.backend.set_board(env.ship_id_grid)
                env.hits_grid = env.backend.hits
                env.miss_grid = env.backend.misses
                steps = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    mask = env.get_action_mask()
                    action, _ = attacker_policy.predict(obs[np.newaxis], action_masks=mask[np.newaxis], deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(int(action[0]))
                    steps += 1
                    if steps >= max_steps:
                        break
                shot_counts.append(steps)
        finally:
            env.close()
        return (float(np.mean(shot_counts)), shot_counts)
    max_steps = board_size * board_size
    n_slots = n_parallel
    shot_counts: List[int] = []
    completed = 0
    total_need = k_episodes
    envs = [BattleshipEnv(board_size=board_size, ships=ships, debug=False) for _ in range(n_slots)]
    obs_list = [None] * n_slots
    steps_per = [0] * n_slots
    active = list(range(n_slots))

    def _reset_slot(i: int, ep_offset: int):
        """"""
        obs, _ = envs[i].reset(seed=seed + ep_offset)
        envs[i].ship_id_grid = layout.astype(np.int32)
        envs[i].backend.set_board(envs[i].ship_id_grid)
        envs[i].hits_grid = envs[i].backend.hits
        envs[i].miss_grid = envs[i].backend.misses
        return obs
    for i in range(n_slots):
        obs_list[i] = _reset_slot(i, i)
    try:
        while completed < total_need:
            idxs = list(range(n_slots))
            batch_obs = np.stack([obs_list[i] for i in idxs])
            batch_masks = np.stack([envs[i].get_action_mask() for i in idxs])
            actions, _ = attacker_policy.predict(batch_obs, action_masks=batch_masks, deterministic=True)
            next_ep_seed = seed + n_slots + completed
            for rank, i in enumerate(idxs):
                obs, _, terminated, truncated, _ = envs[i].step(int(actions[rank]))
                steps_per[i] += 1
                obs_list[i] = obs
                if terminated or truncated or steps_per[i] >= max_steps:
                    shot_counts.append(steps_per[i])
                    completed += 1
                    if completed < total_need:
                        obs_list[i] = _reset_slot(i, next_ep_seed)
                        steps_per[i] = 0
                        next_ep_seed += 1
                    else:
                        obs_list[i] = np.zeros_like(obs_list[i])
    finally:
        for e in envs:
            e.close()
    return (float(np.mean(shot_counts[:k_episodes])), shot_counts[:k_episodes])

class DefenderEnv(gym.Env):
    """"""
    metadata = {'render_modes': []}

    def __init__(self, layout_pool: np.ndarray, attacker_policy=None, k_eval_episodes: int=2, n_eval_parallel: int=1, history_len: int=8, generation: int=0, max_generations: int=10, board_size: int=10, ships: Optional[List[int]]=None, seed: int=0) -> None:
        super().__init__()
        self.layout_pool = layout_pool
        self.attacker_policy = attacker_policy
        self.k_eval_episodes = k_eval_episodes
        self.n_eval_parallel = n_eval_parallel
        self.history_len = history_len
        self.generation = generation
        self.max_generations = max_generations
        self.board_size = board_size
        self.ships = ships or [5, 4, 3, 3, 2]
        self._base_seed = seed
        pool_size = layout_pool.shape[0]
        self.action_space = spaces.Discrete(pool_size)
        obs_dim = history_len + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._history: List[float] = [0.0] * history_len
        self._episode_seed = seed

    def set_attacker(self, attacker_policy) -> None:
        """"""
        self.attacker_policy = attacker_policy

    def _get_obs(self) -> np.ndarray:
        obs = np.array([v / 100.0 for v in self._history] + [self.generation / max(self.max_generations, 1)], dtype=np.float32)
        return obs

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        obs = self._get_obs()
        return (obs, {})

    def step(self, action: int):
        assert self.attacker_policy is not None, 'Set attacker_policy via env.set_attacker() before stepping DefenderEnv'
        action = int(action)
        layout = self.layout_pool[action]
        mean_shots, _ = evaluate_attacker_on_layout(layout=layout, attacker_policy=self.attacker_policy, k_episodes=self.k_eval_episodes, n_parallel=self.n_eval_parallel, board_size=self.board_size, ships=self.ships, seed=self._episode_seed)
        self._episode_seed += self.k_eval_episodes
        reward = float(mean_shots)
        self._history.pop(0)
        self._history.append(mean_shots)
        obs = self._get_obs()
        return (obs, reward, True, False, {'mean_shots': mean_shots, 'layout_idx': action})

    def close(self):
        pass