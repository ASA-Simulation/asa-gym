"""Wrapper for flattening actions of an environment."""

from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenActions(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Action wrapper that flattens the actions."""

    def __init__(self, env: gym.Env):
        """Flattens the action space of an environment.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=spaces.flatten_space(env.action_space).shape,
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> spaces.Tuple:
        """Flattens an action.

        Args:
            action: The flattened action

        Returns:
            The dictionary action
        """
        actions = ()
        for base in range(0, action.shape[0], 6):
            heading = action[base + 0] * 180.0  # [-180.0, +180.0] deg
            load_factor = (action[base + 1] + 1) * 5.0  # [0.0, +10.0] g
            altitude = (action[base + 2] + 1) * 50_000.0  # [0.0, +100_000.0] feet
            base_altitude = (action[base + 3] + 1) * 50_000.0  # [0.0, +100_000.0] feet
            pitch = action[base + 4] * 180.0  # [-90.0, +90.0] deg
            airspeed = 100.0 + (action[base + 5] + 1) * 250.0  # [+100.0, +600.0] m/s
            actions += (
                OrderedDict(
                    {
                        "heading": np.array([heading], dtype=np.float32),
                        "load_factor": np.array([load_factor], dtype=np.float32),
                        "altitude": np.array([altitude], dtype=np.float32),
                        "base_altitude": np.array([base_altitude], dtype=np.float32),
                        "pitch": np.array([pitch], dtype=np.float32),
                        "airspeed": np.array([airspeed], dtype=np.int32),
                    }
                ),
            )
        return actions
