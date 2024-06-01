"""
File for testing the env
Run pip install .\crowd-rl\ before running
"""

from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.policies import shortest_distance
from crowd_rl.sample_configs import mall

if __name__ == "__main__":
    env = crowd.env(config=mall, collect_data=True, render_mode="human", render_fps=12)
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # Politica de menor distancia
            action = shortest_distance(observation)

        env.step(action)

    env.close()
