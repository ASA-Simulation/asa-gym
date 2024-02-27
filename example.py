import os
import pathlib

import gymnasium
from gymnasium.wrappers import FlattenObservation, HumanRendering

import asagym  # noqa: F401
from asagym.envs import random_reward_func
from asagym.wrappers.discrete_actions import DiscreteActions
from asagym.wrappers.relative_position import RelativePosition
from asagym.wrappers.skip_frame import SkipFrameWrapper

base_path = pathlib.Path(os.getcwd()).joinpath("../dist/")

with gymnasium.make(
    "asagym:BeyondVisualRangeEnv-v0",
    reward=random_reward_func,
    base_path=base_path,
    rank=0,
    use_docker=False,
    render_mode="rgb_array",
    log_level="DEBUG",
) as env:
    env = HumanRendering(env)
    env = RelativePosition(env)
    env = FlattenObservation(env)
    env = DiscreteActions(env)

    env = SkipFrameWrapper(env, skip_count=49)

    observation, info = env.reset(seed=21)

    for counter in range(1, 1_000_000):
        action = env.action_space.sample()
        action = 4
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset(seed=21)
