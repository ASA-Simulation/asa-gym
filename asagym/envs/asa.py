import json
import os
import os.path
import pathlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from subprocess import Popen
from typing import Dict, Optional

import gymnasium as gym
import numpy as np
import pygame
import zmq
from gymnasium.spaces import Space

import asagym.proto.simulator_pb2 as pb
from asagym.utils.communication import (recv_message_from_simulation,
                                        send_message_to_simulation)
from asagym.utils.drawing import SCREEN_HEIGHT, SCREEN_WIDTH
from asagym.utils.logger import new_logger
from asagym.utils.simulation import Simulation


class BaseAsaEnv(gym.Env, ABC):
    """Superclass for all ASA environments."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 160}

    def __init__(
        self,
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

        self._logger.info(f"Instantiating an environment using base path: {self.base_path}")

        self.rank = rank  # the id of the execution instance (many in parallel)
        self.own_id = 0  # the id of the player being controlled
        self.episode_counter = 0  # how many episodes have been run
        self.step_counter = 0  # how many steps have been run in a episode

        self.node = None

        self._last_state = pb.State()

        # gymnasium environment variables
        self._logger.debug(f"ASA env with Obervation Space: {observation_space}")
        self._logger.debug(f"ASA env with Action Space: {action_space}")
        self.observation_space = observation_space
        self.action_space = action_space

        # stablishing communication with the underlying simulator
        self.context = zmq.Context()
        url = "asa-gym:" + str(self.rank)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("ipc://" + url)

    @property
    def logger(self):
        return self._logger

    def reset(self, *, seed: int = None, options: Optional[dict] = None) -> tuple:
        super().reset(seed=seed)

        if self.node is not None:
            # attempt clean shutdown underlying simulator
            # the reset method should be idempotent
            self._close_simulation()

        # starts the underlying simulator
        self._initialize_simulation()

        init_data = self._reset_init()
        simulation_state = self._reset_simulation(init_data)

        if self.render_mode is not None:
            self._graphics.reset(simulation_state)

        self._save_recording()

        # a new episode has ended
        self.step_counter = 0
        self.episode_counter += 1
        self._last_state = pb.State()

        self.own_id = simulation_state.id

        observation = self._get_obs(simulation_state)
        info = self._get_info(simulation_state)

        return observation, info

    def _initialize_simulation(self):
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

            env = os.environ.copy()
            self.node = Popen(
                ["bash", exe_path, "./AsaGym", f"--id={self.rank}", f"--uuid={uuid.uuid4()}"],
                cwd=cwd_path,
                env=env,
                shell=False,
            )

        self._logger.info(f"Spawning simulation instance #{self.rank}")

        # sending the Init request
        request = pb.InitRequest()
        send_message_to_simulation(self.socket, request)
        # receive the Init reply
        _ = recv_message_from_simulation(self.socket, pb.INIT)

    def _reset_simulation(self, options: Optional[dict]) -> pb.State:
        # sending the Reset request
        request = pb.ResetRequest()

        if options is not None:
            request.data = json.dumps(options)

        send_message_to_simulation(self.socket, request)
        # receive the Reset reply
        reply = recv_message_from_simulation(self.socket, pb.RESET)
        return reply.state

    def _merge_obs(self, curr_obs: pb.State, new_obs: pb.State) -> pb.State:
        merged_obs = pb.State()
        merged_obs.MergeFrom(new_obs)
        if len(merged_obs.foes) == 0:
            if len(curr_obs.foes) == 0:
                self._logger.warning("curr_obs.foes is empty: this REALLY should not be happening!")
                merged_obs.foes.append(pb.FoeState())
            else:
                merged_obs.foes.append(curr_obs.foes[0])
        return merged_obs

    def step(self, action) -> tuple:
        # incrementing step counter
        self.step_counter += 1

        # high level action => low level action
        sim_action = self._get_action(action)

        # forwards the step to the simulator
        new_state = self._step_simulation(sim_action)
        sim_state = self._merge_obs(self._last_state, new_state)

        if self.render_mode is not None:
            self._graphics.update(sim_state)

        # low level state => high level observations
        observation = self._get_obs(sim_state)
        terminated = self._get_termination(sim_state)
        reward = self._get_reward(sim_state)
        info = self._get_info(sim_state)

        self._last_state = sim_state

        return observation, reward, terminated, False, info

    def _step_simulation(self, action: pb.Action) -> pb.State:
        # sending the Step request
        request = pb.StepRequest(action=action)
        send_message_to_simulation(self.socket, request)

        # receive the Step reply
        reply = recv_message_from_simulation(self.socket, pb.STEP)
        return reply.state

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
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )


    # -------------------
    # methods to override
    # -------------------

    @abstractmethod
    def _reset_init(self) -> Optional[Dict]:
        raise NotImplementedError

    @abstractmethod
    def _get_action(self, action) -> pb.Action:
        return NotImplementedError

    @abstractmethod
    def _get_info(self, simulation_state: pb.State) -> Optional[Dict]:
        return NotImplementedError

    @abstractmethod
    def _get_obs(self, simulation_state: pb.State) -> Space:
        raise NotImplementedError

    @abstractmethod
    def _get_termination(self, simulation_state: pb.State) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self, simulation_state: pb.State) -> float:
        raise NotImplementedError
