import numpy as np

from asagym.proto.simulator_pb2 import State

def random_reward_func(_: State) -> float:
    return np.random.rand()
