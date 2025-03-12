import numpy as np
import gymnasium as gym

from gymnasium import ActionWrapper
from gymnasium.spaces import Box
from collections import OrderedDict


class ContinuousActions(ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        spaces = Box(low=-180.0, high=1e20, shape=(6,), dtype=np.float64)
        self._action_space = spaces

    def action(self, action: Box) -> OrderedDict:
        actions = OrderedDict(
            {
                "heading": np.array([action[0]], dtype=np.float32),
                "load_factor": np.array([action[1]], dtype=np.float32),
                "altitude": np.array([action[2]], dtype=np.float32),
                "base_altitude": np.array([action[3]], dtype=np.float32),
                "pitch": np.array([action[4]], dtype=np.float32),
                "airspeed": np.array([action[5]], dtype=np.float32),
            }
        )
        return actions
