"""
File for testing the env
Run pip install .\crowd-rl\ before running
"""

from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.policies import shortest_distance
from crowd_rl.sample_configs import mall

if __name__ == "__main__":
    env = crowd.parallel_env(config=mall, render_mode="human", render_fps=12)

    observations, infos = env.reset(seed=42)
    while env.agents:
        actions = shortest_distance(observations)
        observations, rewards, terminations, truncations, infos = env.step(actions)

    env.close()
