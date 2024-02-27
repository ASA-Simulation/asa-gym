from typing import Any, SupportsFloat
from gymnasium import Wrapper


class SkipFrameWrapper(Wrapper):
    def __init__(self, env, skip_count: int):
        super().__init__(env)
        self.skip_count = skip_count

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0
        for _ in range(1, 1 + self.skip_count):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
