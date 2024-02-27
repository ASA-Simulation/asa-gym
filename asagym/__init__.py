from gymnasium.envs.registration import register

from asagym.envs.bvr import BeyondVisualRangeEnv  # noqa: F401
from asagym.envs.dbvr import DeepBeyondVisualRangeEnv  # noqa: F401

register(
    id="BeyondVisualRangeEnv-v0",
    entry_point="asagym.envs.bvr:BeyondVisualRangeEnv",
)

register(
    id="DeepBeyondVisualRangeEnv-v0",
    entry_point="asagym.envs.dbvr:DeepBeyondVisualRangeEnv",
)

__version__ = "1.0.0"
