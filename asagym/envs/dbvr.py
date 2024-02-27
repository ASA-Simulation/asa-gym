import logging
import pathlib
from typing import Optional, Callable

import numpy as np
from gymnasium.spaces import Box, Dict

import asagym.proto.simulator_pb2 as pb
from asagym.envs.asa import BaseAsaEnv

logger = logging.getLogger(__name__)


class DeepBeyondVisualRangeEnv(BaseAsaEnv):
    """
    """

    def __init__(
        self,
        reward: Callable[[pb.State],float],
        **kwargs
    ):
        self._reward_func = reward

        observation_space = Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        action_space = Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        BaseAsaEnv.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )

    def _reset_init(self) -> Optional[Dict]:
        init_lat = -16.9486272 - 0.1
        init_long = -48.2672957 - 0.1
        init_heading = 0.0

        data = {
            "players": {
                "B01@Jambock_1": {
                    "attributes": {
                        "initLatitude": 10.0,
                        "initLongitude": -10.0,
                    }
                },
                "Blue_HQ": {
                    "subcomponents": {
                        "taskOrders": {
                            "Blue_ATO_1": {
                                "attributes": {
                                    "initLat": init_lat,
                                    "initLon": init_long,
                                    "initHeading": init_heading,
                                }
                            }
                        }
                    }
                },
            }
        }
        return data

    def _get_action(self, action: np.ndarray) -> pb.Action:
        sim_action = pb.Action()

        sim_action.id = self.own_id

        sim_action.heading = action[0] * 180.0
        sim_action.load_factor = (action[1] + 1.0) * 10.0
        sim_action.altitude = (action[2] + 1.0) * 50_000.0  # max = 100_000 feet
        sim_action.base_altitude = (action[3] + 1.0) * 50_000.0  # max = 100_000 feet
        sim_action.pitch = action[4] * 90.0
        sim_action.airspeed = (action[5] + 1.0) * 5.0  # max = 10 G

        return sim_action

    def _get_info(self, simulation_state: pb.State) -> Optional[Dict]:
        return {"exec_time": simulation_state.exec_time}

    def _get_obs(self, simulation_state: pb.State) -> np.ndarray:
        logger.debug(simulation_state)

        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        obs[0] = simulation_state.owner.player_state.latitude
        obs[1] = simulation_state.owner.player_state.longitude
        obs[2] = simulation_state.owner.player_state.altitude
        obs[3] = simulation_state.owner.player_state.heading
        obs[4] = simulation_state.owner.player_state.airspeed

        return obs

    def _get_termination(self, simulation_state: pb.State) -> bool:
        if len(simulation_state.end_of_episode) > 0:
            return True
        return False

    def _get_reward(self, simulation_state: pb.State) -> float:
        return self._reward_func(simulation_state)
