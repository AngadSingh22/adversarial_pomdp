import pytest
import numpy as np
from battleship_rl.envs.battleship_env import BattleshipEnv

def test_invalid_action_behavior_debug():
    """"""
    env = BattleshipEnv(board_size=10, debug=True)
    obs, info = env.reset()
    env.step(0)
    with pytest.raises(ValueError, match='Invalid action'):
        env.step(0)

def test_invalid_action_behavior_non_debug():
    """"""
    env = BattleshipEnv(board_size=10, debug=False)
    obs1, info1 = env.reset()
    obs2, reward2, term2, trunc2, info2 = env.step(0)
    obs3, reward3, term3, trunc3, info3 = env.step(0)
    assert trunc3 is True, 'Invalid action must truncate episode'
    assert reward3 == -100.0, 'Invalid action reward must be -100'
    assert info3['outcome_type'] == 'INVALID'
    assert info3['outcome_ship_id'] is None
    np.testing.assert_array_equal(obs2, obs3, err_msg='State MUST NOT mutate on invalid action')