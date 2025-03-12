import numpy
import logging
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np
from google.protobuf.json_format import MessageToDict
from gymnasium.spaces import Box, Dict, Space

import asagym.proto.simulator_pb2 as pb
from asagym.envs.asa import BaseAsaEnv
from asagym.utils.logger import fork_logger

from asagym.utils import mixr


class BeyondVisualRange2rlx1Env(BaseAsaEnv):
    """ """

    def __init__(self, reward: Callable[[pb.State], float], **kwargs):
        self._reward_func = reward

        observation_space = Dict(
            {
                "owner": Dict(
                    {
                        "player_state": Dict(
                            {
                                "latitude": Box(low=-90.0, high=90.0, dtype=np.float64),
                                "longitude": Box(
                                    low=-180.0, high=180.0, dtype=np.float64
                                ),
                                "altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                                "heading": Box(
                                    low=-180.0, high=180.0, dtype=np.float64
                                ),
                                "airspeed": Box(low=0.0, high=np.inf, dtype=np.float64),
                            }
                        ),
                        "base_altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "fuel_amount": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "num_msl": Box(low=-1, high=np.inf, dtype=np.int64),
                    }
                ),
                "wingman": Dict(
                    {
                        "player_state": Dict(
                            {
                                "latitude": Box(low=-90.0, high=90.0, dtype=np.float64),
                                "longitude": Box(
                                    low=-180.0, high=180.0, dtype=np.float64
                                ),
                                "altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                                "heading": Box(
                                    low=-180.0, high=180.0, dtype=np.float64
                                ),
                                "airspeed": Box(low=0.0, high=np.inf, dtype=np.float64),
                            }
                        ),
                    }
                ),
                "foe": Dict(
                    {
                        "player_state": Dict(
                            {
                                "latitude": Box(low=-90.0, high=90.0, dtype=np.float64),
                                "longitude": Box(
                                    low=-180.0, high=180.0, dtype=np.float64
                                ),
                                "altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                                "heading": Box(
                                    low=-180.0, high=180.0, dtype=np.float64
                                ),
                                "airspeed": Box(low=0.0, high=np.inf, dtype=np.float64),
                            }
                        ),
                        "true_azmth": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "rel_azmth": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "range": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "wez_own2foe_max": Box(low=-1.0, high=np.inf, dtype=np.float64),
                        "wez_own2foe_nez": Box(low=-1.0, high=np.inf, dtype=np.float64),
                        "wez_foe2own_max": Box(low=-1.0, high=np.inf, dtype=np.float64),
                        "wez_foe2own_nez": Box(low=-1.0, high=np.inf, dtype=np.float64),
                    }
                ),
            }
        )

        action_space = Dict(
            {
                "heading": Box(low=-180.0, high=180.0, dtype=np.float32),
                "load_factor": Box(low=0.0, high=20.0, dtype=np.float32),
                "altitude": Box(low=0.0, high=np.inf, dtype=np.float32),
                "base_altitude": Box(low=0.0, high=np.inf, dtype=np.float32),
                "pitch": Box(low=-90.0, high=90.0, dtype=np.float32),
                "airspeed": Box(low=0.0, high=np.inf, dtype=np.float32),
            }
        )

        BaseAsaEnv.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )

        self._logger = fork_logger("bvr", super().logger)

        self.last_obs = None

    # Especifica a posição inicial do agente blue1 em torno do target.
    # Especifica o heading incial do agente blue1 apontando para o target.
    def set_initial_experiment_setup(self):
        # agent: fighter with RL
        agent_init_lat = 0.0
        agent_init_long = 0.0
        agent_init_heading = 0.0

        # Blue_CAP_2 --> lat = -16.6081612 lon = -48.2530568

        # Target
        target_init_lat = -15.6994023
        target_init_long = -48.2373121
        target_init_heading = 180.0

        # Defining the agent_init_lat and agent_init_long
        delta_angle = 180.0 * numpy.random.rand()
        s = 100.0 * numpy.random.rand()
        signal = 1.0

        if s <= 50.0:
            signal = -1.0

        bearingDeg = mixr.aepcdDeg(target_init_heading + signal * delta_angle)

        minDistanceNM = 20.0
        maxDistanceNM = 90.0
        delta_distanceNM = maxDistanceNM - minDistanceNM

        # ground distance
        distanceNM = minDistanceNM + delta_distanceNM * numpy.random.rand()

        agent_init_lat, agent_init_long = mixr.gbd2ll(
            target_init_lat, target_init_long, bearingDeg, distanceNM, 0
        )

        # Defining the agent_init_heading
        agent_init_heading = mixr.aepcdDeg(bearingDeg + 180.0)

        data = {
            "players": {
                "Blue_HQ": {
                    "subcomponents": {
                        "taskOrders": {
                            "Blue_ATO_1": {
                                "attributes": {
                                    "initLat": agent_init_lat,
                                    "initLon": agent_init_long,
                                    "initHeading": agent_init_heading,
                                }
                            }
                        }
                    }
                },
                "Red_HQ": {
                    "subcomponents": {
                        "taskOrders": {
                            "ato_user_001": {
                                "attributes": {
                                    "initLat": target_init_lat,
                                    "initLon": target_init_long,
                                    "initHeading": target_init_heading,
                                }
                            }
                        }
                    }
                },
            }
        }
        return data

    def reset_init(self) -> Optional[dict]:
        data = self.set_initial_experiment_setup()
        return data

    def get_action(self, action: Dict) -> pb.Action:
        action = pb.Action(
            id=self.own_id,
            heading=action["heading"][0],
            load_factor=action["load_factor"][0],
            altitude=action["altitude"][0],
            base_altitude=action["base_altitude"][0],
            pitch=action["pitch"][0],
            airspeed=action["airspeed"][0],
        )
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(f"Action: {MessageToDict(action)}")
        return action

    def get_info(self, simulation_state: pb.State) -> Optional[Dict]:
        info = {
            "exec_time": simulation_state.exec_time,
            "step_count": self.step_counter,
            "end_of_episode": simulation_state.end_of_episode,
        }
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(f"Information: {info}")
        return info

    def get_obs(self, simulation_state: pb.State) -> Space:
        if len(simulation_state.foes) == 0:
            simulation_state["foes"] = [dict(self.last_obs["foe"])]

        simulation_state.owner.player_state.heading

        obs = OrderedDict(
            {
                "owner": OrderedDict(
                    {
                        "player_state": OrderedDict(
                            {
                                "latitude": np.squeeze(
                                    np.array(
                                        [simulation_state.owner.player_state.latitude]
                                    )
                                ),
                                "longitude": np.squeeze(
                                    np.array(
                                        [simulation_state.owner.player_state.longitude]
                                    )
                                ),
                                "altitude": np.squeeze(
                                    np.array(
                                        [simulation_state.owner.player_state.altitude]
                                    )
                                ),
                                "heading": np.squeeze(
                                    np.array(
                                        [simulation_state.owner.player_state.heading]
                                    )
                                ),
                                "airspeed": np.squeeze(
                                    np.array(
                                        [simulation_state.owner.player_state.airspeed]
                                    )
                                ),
                            }
                        ),
                        "base_altitude": np.squeeze(
                            np.array([simulation_state.owner.base_altitude])
                        ),
                        "fuel_amount": np.squeeze(
                            np.array([simulation_state.owner.fuel_amount])
                        ),
                        "num_msl": np.squeeze(
                            np.array([simulation_state.owner.num_msl])
                        ),
                    }
                ),
                "wingman": OrderedDict(
                    {
                        "player_state": OrderedDict(
                            {
                                "latitude": np.squeeze(
                                    np.array(
                                        [simulation_state.wing.player_state.latitude]
                                    )
                                ),
                                "longitude": np.squeeze(
                                    np.array(
                                        [simulation_state.wing.player_state.longitude]
                                    )
                                ),
                                "altitude": np.squeeze(
                                    np.array(
                                        [simulation_state.wing.player_state.altitude]
                                    )
                                ),
                                "heading": np.squeeze(
                                    np.array(
                                        [simulation_state.wing.player_state.heading]
                                    )
                                ),
                                "airspeed": np.squeeze(
                                    np.array(
                                        [simulation_state.wing.player_state.airspeed]
                                    )
                                ),
                            }
                        ),
                    }
                ),
                "foe": OrderedDict(
                    {
                        "player_state": OrderedDict(
                            {
                                "latitude": np.squeeze(
                                    np.array(
                                        [simulation_state.foes[0].player_state.latitude]
                                    )
                                ),
                                "longitude": np.squeeze(
                                    np.array(
                                        [
                                            simulation_state.foes[
                                                0
                                            ].player_state.longitude
                                        ]
                                    )
                                ),
                                "altitude": np.squeeze(
                                    np.array(
                                        [simulation_state.foes[0].player_state.altitude]
                                    )
                                ),
                                "heading": np.squeeze(
                                    np.array(
                                        [simulation_state.foes[0].player_state.heading]
                                    )
                                ),
                                "airspeed": np.squeeze(
                                    np.array(
                                        [simulation_state.foes[0].player_state.airspeed]
                                    )
                                ),
                            }
                        ),
                        "true_azmth": np.squeeze(
                            np.array([simulation_state.foes[0].true_azmth])
                        ),
                        "rel_azmth": np.squeeze(
                            np.array([simulation_state.foes[0].rel_azmth])
                        ),
                        "range": np.squeeze(np.array([simulation_state.foes[0].range])),
                        "wez_own2foe_max": np.squeeze(
                            np.array([simulation_state.foes[0].wez_own2foe_max])
                        ),
                        "wez_own2foe_nez": np.squeeze(
                            np.array([simulation_state.foes[0].wez_own2foe_nez])
                        ),
                        "wez_foe2own_max": np.squeeze(
                            np.array([simulation_state.foes[0].wez_foe2own_max])
                        ),
                        "wez_foe2own_nez": np.squeeze(
                            np.array([simulation_state.foes[0].wez_foe2own_nez])
                        ),
                    }
                ),
            }
        )
        self.last_obs = obs
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(f"Observation: {obs}")
        return obs

    def get_termination(self, simulation_state: pb.State) -> bool:
        termination = len(simulation_state.end_of_episode) > 0
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(
                f"Termination: '{simulation_state.end_of_episode}' => {termination}"
            )
        return termination

    def get_reward(self, simulation_state: pb.State) -> float:
        reward = self._reward_func(simulation_state)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(f"Reward: {reward}")
        return reward
