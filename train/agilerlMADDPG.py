from copy import deepcopy

import numpy as np
import torch
from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from crowd_rl import crowd_rl_v0 as crowd
from crowd_rl.sample_configs import mall

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mall["groups"][0]["amount"] = 1
mall["groups"][1]["amount"] = 0

env = crowd.parallel_env(config=mall)
env.reset()

# State Space configs
state_dim = [env.observation_space(agent).shape for agent in env.possible_agents]
one_hot = False

# Action Space configs
action_dim = [env.action_space(agent).n for agent in env.possible_agents]
discrete_actions = True
max_action = None
min_action = None

channels_last = False
n_agents = env.max_num_agents
agent_ids = [agent_id for agent_id in env.possible_agents]
field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    memory_size=1_000_000, field_names=field_names, agent_ids=agent_ids, device=device
)

agent = MADDPG(
    state_dims=state_dim,
    action_dims=action_dim,
    one_hot=one_hot,
    n_agents=n_agents,
    agent_ids=agent_ids,
    max_action=max_action,
    min_action=min_action,
    discrete_actions=discrete_actions,
    device=device,
)

episodes = 1000
max_steps = 25
epsilon = 1.0
eps_end = 0.1
eps_decay = 0.995

dead_state = {
    agent_id: np.zeros(state_dim[i], dtype=np.float32)
    for i, agent_id in enumerate(env.possible_agents)
}
dead_reward = {agent_id: 0 for agent_id in env.possible_agents}

for ep in range(episodes):
    state, info = env.reset()  # Reset environment at start of episode
    state = dead_state | state

    agent_reward = {agent_id: 0 for agent_id in env.possible_agents}
    for _ in range(max_steps):
        cont_actions, discrete_action = agent.getAction(state, epsilon)
        action = discrete_action
        print(action)

        next_state, reward, termination, truncation, info = env.step(
            action
        )  # Act in environment
        reward = dead_reward | reward
        next_state = dead_state | next_state

        # Save experiences to replay buffer
        memory.save2memory(state, action, reward, next_state, termination)

        for agent_id, r in reward.items():
            agent_reward[agent_id] += r

        # Learn according to learning frequency
        if (memory.counter % agent.learn_step == 0) and (
            len(memory) >= agent.batch_size
        ):
            experiences = memory.sample(agent.batch_size)  # Sample replay buffer
            agent.learn(experiences)  # Learn according to agent's RL algorithm

        state = next_state

        # Stop episode if any agents have terminated
        if any(truncation.values()) or any(termination.values()):
            break

    # Save the total episode reward
    score = sum(agent_reward.values())
    agent.scores.append(score)

    # Update epsilon for exploration
    epsilon = max(eps_end, epsilon * eps_decay)
