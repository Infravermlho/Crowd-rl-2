"""
File for testing the env
Run pip install .\crowd-rl\ before running
"""

import random
from copy import deepcopy

import numpy as np
from crowd_rl import crowd_rl_v0 as crowd
from test_configs import mall

agent_data = {}
attendant_data = {}
queues_data = {}

# Mall, Shop, Health
if __name__ == "__main__":
    env = crowd.env(config=mall.env_config, render_mode="human", render_fps=12)
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            obs = observation

            distances = obs[:4]
            action = np.where(distances == distances.min())[0]
            action = action[random.randint(0, len(action) - 1)]
            action += 1


        agent_data[env.frames] = deepcopy(env.agents_data)
        attendant_data[env.frames] = deepcopy(env.attendants_data)
        queues_data[env.frames] = deepcopy(env.queues_data)


        env.step(action)

    env.close()