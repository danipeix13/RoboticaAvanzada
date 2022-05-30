from collections import OrderedDict
from gym import spaces
import numpy as np
import time

n_shapes = 0

def set_action_space():
    limits = np.ones(2)
    action_space = spaces.Box(low=-1*limits, high=limits, dtype=np.int32)
    action_space.n = np.prod(action_space.shape)
    return action_space

def set_observation_space(observation):
    # print(type(observation))
    global n_shapes 
    n_shapes = 0
    observation_space, n_actions = __convert_observation_to_space(observation)
    print(len(observation_space))
    n = 100 ^ n_shapes
    return observation_space, n

def __convert_observation_to_space(observation):
    print(observation)
    global n_shapes
    
    time.sleep(0.5)
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
        low = np.full(observation.shape, -1.0, dtype=np.float32)
        high = np.full(observation.shape, 1.0, dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    elif isinstance(observation, float):
        observation = np.array([observation*1000])
        n_shapes += 1
        low = np.full(observation.shape, -50, dtype=np.int32)
        high = np.full(observation.shape, 50, dtype=np.int32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

def state2index(observ):
    x = (observ["distX"] + 2) * 25 * 100   # *1000 para pasar a mm
    y = (observ["distY"] + 2) * 25         # +50 para tranformar en positivo
    
    print(observ, x, y)
    return int(x + y)

def action2index(action):
    ACTION_TABLE = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8] ]
    return ACTION_TABLE[action[0]+1][action[1]+1]
    # return 3*(action[0]+1) + (action[0] + 1)
