import random

import numpy as np


def get_action(obs) -> int:
    distances = obs[:4]
    action = np.where(distances == distances.min())[0]
    action = action[random.randint(0, len(action) - 1)]
    return action + 1


def shortest_distance(obs: dict | list[int]) -> int | dict:
    """
    Aec envs use individual lists while parallel use dicts
    """
    if type(obs) is dict:
        actions = {}
        for agent, o in obs.items():
            actions[agent] = get_action(o)
        return actions
    else:
        return get_action(obs)
