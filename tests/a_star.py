"""
File for testing the env
Run pip install .\crowd-rl\ before running
"""

import random

import numpy as np
from crowd_rl import crowd_rl_v0 as crowd
from test_configs import mall

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

        env.step(action)

    env.close()
