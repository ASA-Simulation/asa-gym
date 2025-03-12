import logging
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Mapping

import numpy as np
from google.protobuf.json_format import MessageToDict, ParseDict
from gymnasium.spaces import Box, Dict, Discrete, Space, Sequence, Tuple as SpaceTuple

import asagym.proto.simulator_pb2 as pb
from asagym.envs.asa import BaseAsaEnv
from asagym.utils.logger import fork_logger


class NMBeyondVisualRangeEnv(BaseAsaEnv):
    """Scenario: N RL x M Opponents"""

    def __init__(
        self,
        num_players: int,
        num_opponents: int,
        reward: Callable[[BaseAsaEnv, List[pb.State], bool], float],
        initialization: Callable[[], Optional[dict]],
        **kwargs,
    ):
        self._num_players = num_players
        self._num_opponents = num_opponents
        self._reward_func = reward
        self._initialization_func = initialization

        self._own_ids = []

        allies = ()
        for _ in range(self._num_players):
            allies += (
                Dict(
                    {
                        "latitude": Box(low=-90.0, high=90.0, dtype=np.float64),
                        "longitude": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "heading": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "airspeed": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "base_altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "fuel_amount": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "num_msl": Box(low=-1, high=np.inf, dtype=np.int64),
                        "active": Discrete(2, seed=21),
                    }
                ),
            )

        foes = ()
        for _ in range(self._num_opponents):
            foes += (
                Dict(
                    {
                        "latitude": Box(low=-90.0, high=90.0, dtype=np.float64),
                        "longitude": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "altitude": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "heading": Box(low=-180.0, high=180.0, dtype=np.float64),
                        "airspeed": Box(low=0.0, high=np.inf, dtype=np.float64),
                        "active": Discrete(2, seed=21),
                    }
                ),
            )

        observation_space = Dict(
            {
                "allies": SpaceTuple(allies),
                "foes": SpaceTuple(foes),
            }
        )

        actions = ()
        for _ in range(self._num_players):
            actions += (
                Dict(
                    {
                        "heading": Box(low=-180.0, high=180.0, dtype=np.float32),
                        "load_factor": Box(low=0.0, high=20.0, dtype=np.float32),
                        "altitude": Box(low=0.0, high=np.inf, dtype=np.float32),
                        "base_altitude": Box(low=0.0, high=np.inf, dtype=np.float32),
                        "pitch": Box(low=-90.0, high=90.0, dtype=np.float32),
                        "airspeed": Box(low=0.0, high=np.inf, dtype=np.float32),
                    }
                ),
            )

        action_space = SpaceTuple(actions)

        BaseAsaEnv.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            **kwargs,
        )

        self._logger = fork_logger("nmbvr", super().logger)

        self.last_obs = None

    def reset_init(self) -> Tuple[int, Optional[dict]]:
        return self._num_players, self._initialization_func()

    def reset_callback(self, _: List[pb.State]) -> None:
        # saving own player's ids in ascending order to fill the action messages
        self._own_ids = sorted(self.summary.own_team.keys())

    def get_action(self, action: Tuple) -> List[pb.Action]:
        # allocate buffer with actions
        actions = [pb.Action() for _ in range(self._num_players)]
        for idx, dict_action in enumerate(action):
            ParseDict(dict_action, actions[idx])
            actions[idx].id = self._own_ids[idx]
            self._logger.debug(f"Action: {MessageToDict(actions[idx])}")
        return actions

    def get_info(self, states: List[pb.State]) -> Optional[Dict]:
        assert len(states) >= 1
        state = states[0]
        info = {
            "exec_time": state.exec_time,
            # "step_count": self.step_counter, <= TODO: is this really necessary?
            "end_of_episode": state.end_of_episode,
        }
        self._logger.debug(f"Information: {info}")
        return info

    def get_obs(self, states: List[pb.State]) -> Space:
        assert len(states) >= 1
        obs = {
            "allies": (),
            "foes": (),
        }

        for id in sorted(self.summary.own_team.keys()):
            player_state = self.summary.own_team.get(id)
            dict_state = MessageToDict(player_state)
            dict_state.pop("id")
            # the following fields do not belong to PlayerState
            for s in states:
                if s.id == id:
                    dict_state["base_altitude"] = s.owner.base_altitude
                    dict_state["fuel_amount"] = s.owner.fuel_amount
                    dict_state["num_msl"] = s.owner.num_msl
                    dict_state["active"] = s.active
                    break
            # append to tuple (sequence space)
            obs["allies"] += (dict_state,)

        for id in sorted(self.summary.ene_team.keys()):
            player_state = self.summary.ene_team.get(id)
            dict_state = MessageToDict(player_state)
            dict_state.pop("id")
            for s in states:
                if s.id == id:
                    dict_state["active"] = s.active
            # append to tuple (sequence space)
            obs["foes"] += (dict_state,)

        # observation
        self.last_obs = obs
        self._logger.debug(f"Observation: {obs}")
        return obs

    def get_termination(self, states: List[pb.State]) -> bool:
        assert len(states) >= 1
        # checking if there is any player on both teams team
        blues = [
            state for state in states if state.side == pb.Side.BLUE and state.active
        ]
        if len(blues) == 0:
            self._logger.debug(f"Termination: no blue fighter left")
            return True
        reds = [state for state in states if state.side == pb.Side.RED and state.active]
        if len(reds) == 0:
            self._logger.debug(f"Termination: no red fighter left")
            return True
        # checking players' actions
        eoe = True
        for state in blues:
            termination = len(state.end_of_episode) > 0
            self._logger.debug(
                f"Termination: '{state.end_of_episode}' => {termination}"
            )
            eoe &= termination
        return eoe

    def get_reward(self, states: List[pb.State], done: bool) -> float:
        assert len(states) >= 1

        reward = self._reward_func(self, states, done)
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(f"Reward: {reward}")
        return reward
