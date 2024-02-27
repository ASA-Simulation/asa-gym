from typing import Any
from copy import deepcopy

import numpy as np

from gymnasium import ObservationWrapper

from collections import OrderedDict
from gymnasium.spaces import Dict, Box


class RelativePosition(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = deepcopy(env.observation_space)
        self._observation_space["relative"] = Dict(
            {
                "player_state": Dict(
                    {
                        "latitude": Box(low=-90.0, high=90.0, dtype=np.float64),
                        "longitude": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "heading": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "airspeed": Box(low=0.0, high=np.inf, dtype=np.float64),
                    }
                )
            }
        )

    def observation(self, observation: Dict) -> Dict:
        fighter_state: Dict = observation["owner"]["player_state"]
        foe_state: Dict = observation["foe"]["player_state"]

        relative_player_state = OrderedDict(
            {
                "player_state": OrderedDict(
                    {
                        "latitude": np.squeeze(
                            np.array(
                                [foe_state["latitude"] - fighter_state["latitude"]]
                            )
                        ),
                        "longitude": np.squeeze(
                            np.array(
                                [foe_state["longitude"] - fighter_state["longitude"]]
                            )
                        ),
                        "altitude": np.squeeze(
                            np.array(
                                [foe_state["altitude"] - fighter_state["altitude"]]
                            )
                        ),
                        "heading": np.squeeze(np.array([0.0])),
                        "airspeed": np.squeeze(np.array([0.0])),
                    }
                )
            }
        )
        observation["relative"] = relative_player_state
        return observation
