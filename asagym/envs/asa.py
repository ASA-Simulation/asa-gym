import json
import os
import os.path
import pathlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from subprocess import Popen, DEVNULL
from typing import Dict, List, Optional, Tuple
from logging import Logger

import gymnasium as gym
import numpy as np
import pygame
import zmq
from gymnasium.spaces import Space

import asagym.proto.simulator_pb2 as pb
from asagym.utils.communication import (
    recv_message_from_simulation,
    send_message_to_simulation,
)
from asagym.utils.drawing import SCREEN_HEIGHT, SCREEN_WIDTH
from asagym.utils.logger import new_logger
from asagym.utils.preprocessing import merge_observations
from asagym.utils.simulation import Simulation


class BaseAsaEnv(gym.Env, ABC):
    """Superclass for all ASA environments."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 160}

    def __init__(
        self,
        simu_path: pathlib.Path,
        base_path: pathlib.Path,
        observation_space: Space,
        action_space: Space,
        render_mode: Optional[str] = None,
        rank: int = 0,
        use_docker: bool = False,
        log_level: str = "INFO",
    ):
        if not base_path.exists():
            raise OSError(f"File {base_path.absolute()} does not exist")

        if not base_path.is_dir():
            raise OSError(f"File {base_path.absolute()} is not a directory")

        # save scenario edl file
        self.simu_path = simu_path

        # setting class logger
        log_path = base_path.joinpath(f"./var/log/AsaGym/{rank}")
        os.makedirs(log_path, exist_ok=True)
        self._logger = new_logger(
            file=log_path.joinpath(f"./{datetime.now()}.log"),
            name=__name__,
            level=log_level,
        )

        # execution modes
        self.use_docker = use_docker
        self.base_path = base_path

        # render settings
        self.render_mode = render_mode
        if self.render_mode is not None:
            self._graphics = Simulation()

        self._logger.info(
            f"Instantiating an environment using base path: {self.base_path}"
        )

        self.rank = rank  # the id of the execution instance (many in parallel)
        self.own_id = 0  # the id of the player being controlled
        self.episode_counter = 0  # how many episodes have been run
        self.step_counter = 0  # how many steps have been run in a episode

        self.node = None

        self._summary = pb.Summary()

        # gymnasium environment variables
        self._logger.debug(f"ASA env with Obervation Space: {observation_space}")
        self._logger.debug(f"ASA env with Action Space: {action_space}")
        self.observation_space = observation_space
        self.action_space = action_space

        # stablishing communication with the underlying simulator
        self.context = zmq.Context()
        url = "127.0.0.1:" + str(8000 + self.rank)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://" + url)

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def summary(self):
        return self._summary

    def reset(self, *, seed: int = None, options: Optional[dict] = None) -> tuple:
        super().reset(seed=seed, options=options)

        if self.node is not None:
            # attempt clean shutdown underlying simulator
            # the reset method should be idempotent
            self._close_simulation()

        # starts the underlying simulator
        num_players, init_data = self.reset_init()

        self._initialize_simulation(num_players)

        self._summary = pb.Summary()
        states = self._reset_simulation(init_data)
        self._summary = merge_observations(states, self._summary)

        if self.render_mode is not None:
            self._graphics.reset(self._summary)

        self._save_recording()

        # a new episode has ended
        self.step_counter = 0
        self.episode_counter += 1

        self.own_id = states[0].owner.player_state.id

        # a callback to be used to reset/initialize subclasses
        self.reset_callback(states)

        observation = self.get_obs(states)
        info = self.get_info(states)

        return observation, info

    def _initialize_simulation(self, num_players: int):
        # loading scenario edl file
        self._logger.info(f"Using scenario: {self.simu_path.absolute()}")
        with open(self.simu_path, "r") as file:
            scenario = file.read()

        exec_uuid = uuid.uuid4()

        # Store execution uuid
        self.uuid = exec_uuid

        scenario = scenario.replace("!EXEC_UUID!", str(exec_uuid))

        # running the underlying simulator process
        if self.use_docker:
            # inside docker
            self.node = Popen(
                f"docker run -it -v {self.base_path}:/home/asa/workspace -p {50051 + self.rank}:50051 hub.asa.dcta.mil.br/asa/gym:latest ./AsaGym --id={self.rank}",
                shell=True,
            )
        else:
            # or as a regular process
            cwd_path = self.base_path.joinpath("./bin")
            exe_path = self.base_path.joinpath("./bin/AsaWrapper.sh")

            data_path = self.base_path.joinpath(
                f"./var/data/executions/{exec_uuid}/gym.log"
            )

            env = os.environ.copy()
            self.node = Popen(
                [
                    "bash",
                    exe_path,
                    "./AsaGym",
                    f"--id={self.rank}",
                    f"--uuid={exec_uuid}",
                ],
                stdout=DEVNULL,  # TODO: pipe this to somewhere
                stderr=DEVNULL,  # TODO: pipe this to somewhere
                cwd=cwd_path,
                env=env,
                shell=False,
            )

        self._logger.info(f"Spawning simulation instance #{self.rank}")

        # sending the Init request
        request = pb.InitRequest()
        request.edl = scenario
        request.num_players = num_players
        send_message_to_simulation(self.socket, request)
        # receive the Init reply
        _ = recv_message_from_simulation(self.socket, pb.INIT)

    def _reset_simulation(self, options: Optional[dict]) -> List[pb.State]:
        # sending the Reset request
        request = pb.ResetRequest()

        if options is not None:
            request.data = json.dumps(options)

        send_message_to_simulation(self.socket, request)
        # receive the Reset reply
        reply = recv_message_from_simulation(self.socket, pb.RESET)
        return reply.states

    def step(self, action) -> tuple:
        # incrementing step counter
        self.step_counter += 1

        # high level action => low level action
        sim_action = self.get_action(action)

        # forwards the step to the simulator
        sim_state = self._step_simulation(sim_action)
        self._summary = merge_observations(sim_state, self._summary)

        if self.render_mode is not None:
            self._graphics.update(self._summary)

        # low level state => high level observations
        observation = self.get_obs(sim_state)
        terminated = self.get_termination(sim_state)
        reward = self.get_reward(sim_state, terminated)
        info = self.get_info(sim_state)

        self._last_state = sim_state

        return observation, reward, terminated, False, info

    def _step_simulation(self, actions: List[pb.Action]) -> List[pb.State]:
        # sending the Step request
        request = pb.StepRequest(actions=actions)
        send_message_to_simulation(self.socket, request)

        # receive the Step reply
        reply = recv_message_from_simulation(self.socket, pb.STEP)
        return reply.states

    def _close_simulation(self) -> None:
        # stop simulation process
        self.node.kill()
        self.node.wait(timeout=10.0)
        self.node = None

        # save recording
        self._save_recording()

    def close(self) -> None:
        if self.node is not None:
            # closing siumulation
            self._close_simulation()

        # closing connection
        self.socket.close()
        self.context.term()

    def _save_recording(self) -> None:
        cwd = os.getcwd()
        input_path = f"{cwd}/../bin/execution.acmi"
        output_path = f"{cwd}/../var/data/AsaGym/recordings/{self.rank}"

        os.makedirs(output_path, exist_ok=True)

        if self.episode_counter != 0 and os.path.exists(input_path):
            # move the recording (execution.acmi) to avoid overwriting it
            os.rename(input_path, f"{output_path}/{self.episode_counter}.acmi")

    def render(self) -> np.ndarray:
        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill((255, 255, 255))

        self._graphics.draw(canvas)
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    # -------------------
    # methods to override
    # -------------------

    def reset_callback(self, simulation_state: List[pb.State]) -> None:
        pass

    @abstractmethod
    def reset_init(self) -> Tuple[int, Optional[Dict]]:
        raise NotImplementedError

    @abstractmethod
    def get_action(self, action) -> List[pb.Action]:
        return NotImplementedError

    @abstractmethod
    def get_info(self, simulation_state: List[pb.State]) -> Optional[Dict]:
        return NotImplementedError

    @abstractmethod
    def get_obs(self, simulation_state: List[pb.State]) -> Space:
        raise NotImplementedError

    @abstractmethod
    def get_termination(self, simulation_state: List[pb.State]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_reward(
        self, simulation_state: pb.State | List[pb.State], done: bool
    ) -> float:
        raise NotImplementedError
