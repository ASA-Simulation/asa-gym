import os
import pathlib

import gymnasium
from gymnasium.wrappers import FlattenObservation, HumanRendering

import asagym  # noqa: F401
from asagym.envs import random_reward_func
from asagym.wrappers.discrete_actions import DiscreteActions
from asagym.wrappers.skip_frame import SkipFrameWrapper
from asagym.wrappers.profiling import Profiler

curr_path = pathlib.Path(__file__).parent.absolute()
base_path = curr_path.joinpath("../dist/")
data_path = base_path.joinpath("./var/data/AsaGym")
simu_path = curr_path.joinpath("../asa-ai/experiments/2x1_rlfighter_rlfighter.edl")

os.makedirs(data_path, exist_ok=True)

with gymnasium.make(
    "asagym:NMBeyondVisualRangeEnv-v0",
    initialization=lambda: None,
    reward=random_reward_func,
    simu_path=simu_path,
    base_path=base_path,
    num_players=2,
    num_opponents=1,
    rank=0,
    use_docker=False,
    render_mode="rgb_array",
    log_level="DEBUG",
) as env:
    # env = HumanRendering(env)
    env = FlattenObservation(env)

    env = SkipFrameWrapper(env, skip_count=49)
    env = Profiler(env)

    observation, info = env.reset(seed=21)

    for counter in range(1, 1_000_000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=21)
