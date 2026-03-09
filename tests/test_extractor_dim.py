import pytest
import torch as th
from gymnasium import spaces
import numpy as np
from battleship_rl.agents.policies import BattleshipFeatureExtractor

def test_feature_extractor_dimension():
    """"""
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 10, 10), dtype=np.float32)
    features_dim = 512
    extractor = BattleshipFeatureExtractor(observation_space, features_dim=features_dim)
    batch_size = 4
    dummy_obs = th.zeros((batch_size, 3, 10, 10), dtype=th.float32)
    with th.no_grad():
        output = extractor(dummy_obs)
    assert output.shape == (batch_size, features_dim), f'Extractor must return exactly shape (batch_size, 512). Got {output.shape}'