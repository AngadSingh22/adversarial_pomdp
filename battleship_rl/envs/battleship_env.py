from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from battleship_rl.agents.defender import UniformRandomDefender
from battleship_rl.envs.rewards import StepPenaltyReward
from battleship_rl.bindings.c_api import CBattleshipFactory

class BattleshipEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, config: Optional[dict]=None, board_size: int | Sequence[int]=10, ships: Optional[Sequence[int] | dict]=None, defender: Optional[object]=None, defenders: Optional[list]=None, defender_weights: Optional[list]=None, reward_fn: Optional[object]=None, debug: bool=False) -> None:
        super().__init__()
        cfg = config or {}
        board_size = cfg.get('board_size', board_size)
        ships = cfg.get('ship_config', ships)
        ships = cfg.get('ships', ships)
        if ships is None:
            ships = [5, 4, 3, 3, 2]
        if isinstance(board_size, int):
            height, width = (board_size, board_size)
        else:
            if len(board_size) != 2:
                raise ValueError('board_size must be int or length-2 sequence')
            height, width = (int(board_size[0]), int(board_size[1]))
        self.height = height
        self.width = width
        self.ship_lengths = [int(length) for length in (list(ships.values()) if isinstance(ships, dict) else list(ships))]
        self.ships = self.ship_lengths
        self.num_ships = len(self.ship_lengths)
        if defenders is not None:
            self._defenders = defenders
            w = defender_weights or [1.0] * len(defenders)
            total = sum(w)
            self._defender_weights = [x / total for x in w]
        else:
            self._defenders = [defender or UniformRandomDefender()]
            self._defender_weights = [1.0]
        self.defender = self._defenders[0]
        self._current_defender_name: str = type(self.defender).__name__
        self._debug_asserted: bool = False
        if reward_fn is not None:
            self.reward_fn = reward_fn
        elif 'reward_scheme' in cfg:
            rs = cfg['reward_scheme']
            step_penalty = float(rs.get('miss', -1.0))
            target_hit = float(rs['hit'])
            target_sink = float(rs['sink'])
            alpha = target_hit - step_penalty
            beta = target_sink - step_penalty
            from battleship_rl.envs.rewards import ShapedReward
            self.reward_fn = ShapedReward(alpha=alpha, beta=beta, step_penalty=step_penalty)
        else:
            self.reward_fn = StepPenaltyReward()
        self.debug = bool(debug)
        self.invalid_action_penalty = -100.0
        self.action_space = spaces.Discrete(self.height * self.width)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, self.height, self.width), dtype=np.float32)
        self.backend = CBattleshipFactory(self.height, self.width, self.ship_lengths)
        self.ship_id_grid: Optional[np.ndarray] = None
        self.hits_grid: Optional[np.ndarray] = None
        self.miss_grid: Optional[np.ndarray] = None
        self.sunk_ships: set[int] = set()

    def _build_info(self, outcome_type: Optional[str], outcome_ship_id: Optional[int]) -> Dict[str, Any]:
        return {'action_mask': self.get_action_mask(), 'outcome_type': outcome_type, 'outcome_ship_id': outcome_ship_id, 'last_outcome': (outcome_type, outcome_ship_id), 'defender_mode': self._current_defender_name}

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        super().reset(seed=seed)
        if len(self._defenders) > 1:
            idx = self.np_random.choice(len(self._defenders), p=self._defender_weights)
            self.defender = self._defenders[int(idx)]
        else:
            self.defender = self._defenders[0]
        self._current_defender_name = type(self.defender).__name__
        self.ship_id_grid = self.defender.sample_layout((self.height, self.width), self.ships, self.np_random)
        backend_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.backend.reset(backend_seed)
        self.backend.set_board(self.ship_id_grid)
        self.hits_grid = self.backend.hits
        self.miss_grid = self.backend.misses
        self.sunk_ships = set()
        obs = self.backend.get_obs()
        info = self._build_info(None, None)
        return (obs, info)

    def step(self, action):
        action = int(action)
        mask = self.get_action_mask()
        if action < 0 or action >= mask.size or (not mask[action]):
            if self.debug:
                raise ValueError('Invalid action: %s' % action)
            obs = self.backend.get_obs()
            return (obs, self.invalid_action_penalty, False, True, self._build_info('INVALID', None))
        res = self.backend.step(action)
        outcome_ship_id = None
        outcome_type = 'MISS'
        if res == 1:
            outcome_type = 'HIT'
            r, c = divmod(action, self.width)
            outcome_ship_id = int(self.ship_id_grid[r, c])
        elif res == 2:
            outcome_type = 'SUNK'
            r, c = divmod(action, self.width)
            outcome_ship_id = int(self.ship_id_grid[r, c])
            self.sunk_ships.add(outcome_ship_id)
        terminated = bool(np.all(self.backend.ship_sunk))
        reward = self.reward_fn(outcome_type, terminated)
        obs = self.backend.get_obs()
        info = self._build_info(outcome_type, outcome_ship_id)
        return (obs, reward, terminated, False, info)

    def set_defender_weights(self, weights: list) -> None:
        """"""
        total = sum(weights)
        self._defender_weights = [w / total for w in weights]

    def get_action_mask(self) -> np.ndarray:
        mask = (~(self.hits_grid | self.miss_grid)).reshape(-1)
        if self.debug and (not self._debug_asserted):
            assert mask.dtype == np.bool_, f'Mask dtype must be bool, got {mask.dtype}'
            assert mask.shape == (self.height * self.width,), f'Mask shape mismatch: {mask.shape}'
            self._debug_asserted = True
        return mask

    def render(self):
        if self.hits_grid is None or self.miss_grid is None:
            return ''
        symbols = np.full((self.height, self.width), '.', dtype='<U1')
        symbols[self.miss_grid] = 'o'
        symbols[self.hits_grid] = 'X'
        lines = [' '.join(row.tolist()) for row in symbols]
        board_str = '\n'.join(lines)
        return board_str