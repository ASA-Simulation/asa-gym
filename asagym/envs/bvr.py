import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import numpy as np
from google.protobuf.json_format import MessageToDict
from gymnasium.spaces import Box, Dict, Space

import asagym.proto.simulator_pb2 as pb
from asagym.envs.asa import BaseAsaEnv
from asagym.utils.logger import fork_logger


class BeyondVisualRangeEnv(BaseAsaEnv):
    """Scenario: 1 RL x 1 BT"""

    def __init__(
        self,
        reward: Callable[[pb.State], float],
        initialization: Callable[[], Optional[dict]],
        **kwargs,
    ):
        self._reward_func = reward
        self._initialization_func = initialization

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

    def reset_init(self) -> Tuple[int, Optional[dict]]:
        return 1, self._initialization_func()

    def get_action(self, action: Dict) -> List[pb.Action]:
        action = pb.Action(
            id=self.own_id,
            heading=action["heading"][0],
            load_factor=action["load_factor"][0],
            altitude=action["altitude"][0],
            base_altitude=action["base_altitude"][0],
            pitch=action["pitch"][0],
            airspeed=action["airspeed"][0],
        )
        self._logger.debug(f"Action: {MessageToDict(action)}")
        return [action]

    def get_info(self, states: List[pb.State]) -> Optional[Dict]:
        assert len(states) == 1
        state = states[0]

        info = {
            "exec_time": state.exec_time,
            "step_count": self.step_counter,
            "end_of_episode": state.end_of_episode,
        }
        self._logger.debug(f"Information: {info}")
        return info

    def get_obs(self, states: List[pb.State]) -> Space:
        assert len(states) == 1
        state = states[0]

        # foes field may be empty
        if len(state.foes) == 0:
            state["foes"] = [dict(self.last_obs["foe"])]

        obs = OrderedDict(
            {
                "owner": OrderedDict(
                    {
                        "player_state": OrderedDict(
                            {
                                "latitude": np.squeeze(
                                    np.array([state.owner.player_state.latitude])
                                ),
                                "longitude": np.squeeze(
                                    np.array([state.owner.player_state.longitude])
                                ),
                                "altitude": np.squeeze(
                                    np.array([state.owner.player_state.altitude])
                                ),
                                "heading": np.squeeze(
                                    np.array([state.owner.player_state.heading])
                                ),
                                "airspeed": np.squeeze(
                                    np.array([state.owner.player_state.airspeed])
                                ),
                            }
                        ),
                        "base_altitude": np.squeeze(
                            np.array([state.owner.base_altitude])
                        ),
                        "fuel_amount": np.squeeze(np.array([state.owner.fuel_amount])),
                        "num_msl": np.squeeze(np.array([state.owner.num_msl])),
                    }
                ),
                "foe": OrderedDict(
                    {
                        "player_state": OrderedDict(
                            {
                                "latitude": np.squeeze(
                                    np.array([state.foes[0].player_state.latitude])
                                ),
                                "longitude": np.squeeze(
                                    np.array([state.foes[0].player_state.longitude])
                                ),
                                "altitude": np.squeeze(
                                    np.array([state.foes[0].player_state.altitude])
                                ),
                                "heading": np.squeeze(
                                    np.array([state.foes[0].player_state.heading])
                                ),
                                "airspeed": np.squeeze(
                                    np.array([state.foes[0].player_state.airspeed])
                                ),
                            }
                        ),
                        "true_azmth": np.squeeze(np.array([state.foes[0].true_azmth])),
                        "rel_azmth": np.squeeze(np.array([state.foes[0].rel_azmth])),
                        "range": np.squeeze(np.array([state.foes[0].range])),
                        "wez_own2foe_max": np.squeeze(
                            np.array([state.foes[0].wez_own2foe_max])
                        ),
                        "wez_own2foe_nez": np.squeeze(
                            np.array([state.foes[0].wez_own2foe_nez])
                        ),
                        "wez_foe2own_max": np.squeeze(
                            np.array([state.foes[0].wez_foe2own_max])
                        ),
                        "wez_foe2own_nez": np.squeeze(
                            np.array([state.foes[0].wez_foe2own_nez])
                        ),
                    }
                ),
            }
        )
        self.last_obs = obs
        self._logger.debug(f"Observation: {obs}")
        return obs

    def get_termination(self, states: List[pb.State]) -> bool:
        eoe = False
        for state in states:
            termination = len(state.end_of_episode) > 0
            self._logger.debug(
                f"Termination: '{state.end_of_episode}' => {termination}"
            )
            eoe |= termination
        return eoe

    def get_reward(self, states: List[pb.State]) -> float:
        assert len(states) == 1
        state = states[0]

        reward = self._reward_func(state)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(f"Reward: {reward}")
        return reward
