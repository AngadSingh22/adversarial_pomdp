from __future__ import annotations
import torch as th
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BattleshipFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim: int=512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        with th.no_grad():
            sample = th.zeros((1, n_input_channels, *observation_space.shape[1:]))
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class BattleshipCnnPolicy(MaskableActorCriticPolicy):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, features_extractor_class=BattleshipFeatureExtractor)

class BattleshipRecurrentFeatureExtractor(BaseFeaturesExtractor):
    """"""

    def __init__(self, observation_space, features_dim: int=512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        with th.no_grad():
            sample = th.zeros((1, n_input_channels, *observation_space.shape[1:]))
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def make_recurrent_policy_kwargs(features_dim: int=128) -> dict:
    """"""
    return {'features_extractor_class': BattleshipRecurrentFeatureExtractor, 'features_extractor_kwargs': {'features_dim': features_dim}, 'lstm_hidden_size': 256, 'n_lstm_layers': 1, 'shared_lstm': True}