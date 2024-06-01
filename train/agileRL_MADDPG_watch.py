"""
Lembre de usar a mesma
configuração entre o _train e _watch
"""

import os

import imageio
import numpy as np
import supersuit as ss
import torch
from agilerl.algorithms.maddpg import MADDPG
from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.sample_configs import mall
from PIL import Image, ImageDraw


def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = crowd.parallel_env(config=mall)
env.reset()

state_dim = list(env.observation_space(agent).shape for agent in env.possible_agents)
action_dim = list(env.action_space(agent).n for agent in env.possible_agents)

n_agents = env.max_num_agents
agent_ids = list(agent_id for agent_id in env.possible_agents)

agent = MADDPG(
    state_dims=state_dim,
    action_dims=action_dim,
    n_agents=n_agents,
    agent_ids=agent_ids,
    discrete_actions=True,
    max_action=None,
    min_action=None,
    one_hot=False,
    device=device,
)

# Load the saved algorithm into the MADDPG object
path = "./models/MADDPG/MADDPG_trained_agent.pt"
maddpg.loadCheckpoint(path)

# Define test loop parameters
episodes = 10  # Number of episodes to test agent on
max_steps = 500  # Max number of steps to take in the environment in each episode

rewards = []  # List to collect total episodic reward
frames = []  # List to collect frames
indi_agent_rewards = {
    agent_id: [] for agent_id in agent_ids
}  # Dictionary to collect inidivdual agent rewards

# Test loop for inference
for ep in range(episodes):
    state, info = env.reset()
    agent_reward = {agent_id: 0 for agent_id in agent_ids}
    score = 0
    for _ in range(max_steps):
        if channels_last:
            state = {
                agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                for agent_id, s in state.items()
            }

        agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
        env_defined_actions = (
            info["env_defined_actions"]
            if "env_defined_actions" in info.keys()
            else None
        )

        # Get next action from agent
        cont_actions, discrete_action = maddpg.getAction(
            state,
            epsilon=0,
            agent_mask=agent_mask,
            env_defined_actions=env_defined_actions,
        )
        if maddpg.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        # Save the frame for this step and append to frames list
        frame = env.render()
        frames.append(_label_with_episode_number(frame, episode_num=ep))

        # Take action in environment
        state, reward, termination, truncation, info = env.step(action)

        # Save agent's reward for this step in this episode
        for agent_id, r in reward.items():
            agent_reward[agent_id] += r

        # Determine total score for the episode and then append to rewards list
        score = sum(agent_reward.values())

        # Stop episode if any agents have terminated
        if any(truncation.values()) or any(termination.values()):
            break

    rewards.append(score)

    # Record agent specific episodic reward for each agent
    for agent_id in agent_ids:
        indi_agent_rewards[agent_id].append(agent_reward[agent_id])

    print("-" * 15, f"Episode: {ep}", "-" * 15)
    print("Episodic Reward: ", rewards[-1])
    for agent_id, reward_list in indi_agent_rewards.items():
        print(f"{agent_id} reward: {reward_list[-1]}")
env.close()

# Save the gif to specified path
gif_path = "./videos/"
os.makedirs(gif_path, exist_ok=True)
imageio.mimwrite(os.path.join("./videos/", "space_invaders.gif"), frames, duration=10)
