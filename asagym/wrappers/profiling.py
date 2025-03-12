import time
import logging

from typing import Any, SupportsFloat
from gymnasium import Wrapper

from gymnasium.core import ActType, ObsType, WrapperObsType, RenderFrame

logger = logging.getLogger(__name__)


class Profiler(Wrapper):
    def __init__(self, env, stdout: bool = True, log: bool = False, csv: bool = False):
        super().__init__(env)
        self.stdout = stdout
        self.log = log
        self.csv = csv
        self._write("Using profiler")

    def _write(self, msg: str):
        msg = f"[PRF] {msg}"
        if self.stdout:
            print(msg)
        if self.log:
            logger.debug(msg)
        if self.csv:
            pass

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        start = time.time()
        obs, reward, terminated, truncated, info = self.env.step(action)
        delta = time.time() - start
        self._write(f"STEP : {delta:7.3f} s => {5.0/delta:7.3f} X")
        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        start = time.time()
        obs, info = self.env.reset(seed=seed, options=options)
        delta = time.time() - start
        self._write(f"RESET: {delta:7.3f} s")
        return obs, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        start = time.time()
        frames = self.env.render()
        delta = time.time() - start
        self._write(f"RENDER: {delta:7.3f} s")
        return frames

    def close(self):
        start = time.time()
        self.env.close()
        delta = time.time() - start
        self._write(f"CLOSE: {delta:7.3f} s")
