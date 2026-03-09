import pytest
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_action_mapping():
    """"""
    env = BattleshipEnv(board_size=(10, 10))
    env.reset()
    layout = np.zeros((10, 10), dtype=int) - 1
    layout[3, 4] = 0
    env.ship_id_grid = layout
    env.backend.set_board(layout)
    action = 34
    obs, reward, terminated, truncated, info = env.step(action)
    assert info['outcome_ship_id'] == 0, f'Expected to hit ship 0 at r=3, c=4. Got {info['outcome_ship_id']}'
    r, c = divmod(action, env.width)
    assert r == 3
    assert c == 4