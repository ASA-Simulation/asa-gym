from gymnasium.envs.registration import register

from asagym.envs.bvr import BeyondVisualRangeEnv  # noqa: F401
from asagym.envs.nmbvr import NMBeyondVisualRangeEnv  # noqa: F401
from asagym.envs.bvr2x1 import BeyondVisualRange2x1Env  # noqa: F401
from asagym.envs.bvr2rlx1 import BeyondVisualRange2rlx1Env  # noqa: F401

register(
    id="BeyondVisualRangeEnv-v0",
    entry_point="asagym.envs.bvr:BeyondVisualRangeEnv",
)

register(
    id="NMBeyondVisualRangeEnv-v0",
    entry_point="asagym.envs.nmbvr:NMBeyondVisualRangeEnv",
)

register(
    id="BeyondVisualRange2x1Env-v0",
    entry_point="asagym.envs.bvr2x1:BeyondVisualRange2x1Env",
)

register(
    id="BeyondVisualRange2rlx1Env-v0",
    entry_point="asagym.envs.bvr2rlx1:BeyondVisualRange2rlx1Env",
)

__version__ = "1.1.0"
