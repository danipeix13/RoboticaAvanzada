from collections import OrderedDict
from gym import spaces
import numpy as np

def set_action_space():
    limits = np.ones(5)
    action_space = spaces.Box(low=-1*limits, high=limits, dtype=np.int32)
    return action_space

def set_observation_space(observation):
    print(type(observation))
    observation_space = __convert_observation_to_space(observation)
    return observation_space

def __convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, __convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float("inf"), dtype=np.float32)
        high = np.full(observation.shape, float("inf"), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space