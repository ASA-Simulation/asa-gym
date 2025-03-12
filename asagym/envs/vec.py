from typing import Any
from logging import getLogger

from stable_baselines3.common.vec_env import SubprocVecEnv


class AsaVecEnv(SubprocVecEnv):
    """Wrap SubprocVecEnv to support with-statement. This ensures that underlying simulators are properly closed on"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def __enter__(self) -> SubprocVecEnv:
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any) -> bool:
        """Support with-statement for the environment and closes the environment."""
        self.close()
        getLogger(__name__).debug("Vectorized env closed: AsaVecEnv.__exit__")
        return False
