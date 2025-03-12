from typing import Any, SupportsFloat
import numpy as np
import gymnasium
from gymnasium import ActionWrapper
from gymnasium.spaces import Discrete
from collections import OrderedDict


SOFTLEFT = 0
SOFTRIGHT = 1
SHARPLEFT = 2
SHARPRIGHT = 3
STRAIGHT = 4


class DiscreteActions(ActionWrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._action_space = Discrete(n=5, start=0)

    def action(self, action: OrderedDict):
        last_state = self.env.unwrapped._last_state

        last_heading = last_state.owner.player_state.heading
        last_altitude = last_state.owner.player_state.altitude

        discrete_actions = OrderedDict(
            {
                SOFTLEFT: OrderedDict(
                    {
                        "airspeed": np.array([257.222], dtype=np.int32),
                        "altitude": np.array([last_altitude], dtype=np.float32),
                        "base_altitude": np.array([last_altitude], dtype=np.float32),
                        "heading": np.array([last_heading - 30], dtype=np.float32),
                        "load_factor": np.array([3], dtype=np.float32),
                        "pitch": np.array([0], dtype=np.float32),
                    }
                ),
                SOFTRIGHT: OrderedDict(
                    {
                        "airspeed": np.array([257.222], dtype=np.int32),
                        "altitude": np.array([last_altitude], dtype=np.float32),
                        "base_altitude": np.array([last_altitude], dtype=np.float32),
                        "heading": np.array([last_heading + 30], dtype=np.float32),
                        "load_factor": np.array([3], dtype=np.float32),
                        "pitch": np.array([0], dtype=np.float32),
                    }
                ),
                SHARPLEFT: OrderedDict(
                    {
                        "airspeed": np.array([257.222], dtype=np.int32),
                        "altitude": np.array([last_altitude], dtype=np.float32),
                        "base_altitude": np.array([last_altitude], dtype=np.float32),
                        "heading": np.array([last_heading - 60], dtype=np.float32),
                        "load_factor": np.array([6], dtype=np.float32),
                        "pitch": np.array([0], dtype=np.float32),
                    }
                ),
                SHARPRIGHT: OrderedDict(
                    {
                        "airspeed": np.array([257.222], dtype=np.int32),
                        "altitude": np.array([last_altitude], dtype=np.float32),
                        "base_altitude": np.array([last_altitude], dtype=np.float32),
                        "heading": np.array([last_heading + 60], dtype=np.float32),
                        "load_factor": np.array([6], dtype=np.float32),
                        "pitch": np.array([0], dtype=np.float32),
                    }
                ),
                STRAIGHT: OrderedDict(
                    {
                        "airspeed": np.array([257.222], dtype=np.int32),
                        "altitude": np.array([last_altitude], dtype=np.float32),
                        "base_altitude": np.array([last_altitude], dtype=np.float32),
                        "heading": np.array([last_heading], dtype=np.float32),
                        "load_factor": np.array([1], dtype=np.float32),
                        "pitch": np.array([0], dtype=np.float32),
                    }
                ),
            }
        )
        return discrete_actions[action]

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        return super().step(action)
