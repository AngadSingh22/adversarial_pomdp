import pytest
from battleship_rl.envs.battleship_env import BattleshipEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

def mask_fn(env: BattleshipEnv):
    return env.get_action_mask()

def test_sb3_integration():
    """"""

    def env_creator():
        e = BattleshipEnv(board_size=10)
        return ActionMasker(e, mask_fn)
    vec_env = make_vec_env(env_creator, n_envs=2)
    model = MaskablePPO('MlpPolicy', vec_env, n_steps=64, batch_size=32, n_epochs=1)
    model.learn(total_timesteps=128)
    assert True