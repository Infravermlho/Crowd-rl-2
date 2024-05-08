import functools
import os
import random
from collections import deque
from copy import copy, deepcopy
from typing import Tuple

import distinctipy
import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .pygame_utils import fill, get_image
from .schema import Agent, Attendant, Config, Entrance, Exit, Queue


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "crowd_rl_v0",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(
        self,
        config: Config,
        render_mode: str | None = None,
        render_fps: int = 15,
        screen_scaling: int = 12,
        max_cycles: int = 900,
    ):
        EzPickle.__init__(self, config, render_mode, screen_scaling)
        super().__init__()

        self.config = config
        self.max_cycles = max_cycles

        self.frames = 0
        self.run = True

        # Constants derived from config
        self._seed(config.seed)
        self.map = np.array(self.config.worldmap, dtype=np.int8)
        self.colision = self.map[:]
        self.height = self.config.height
        self.width = self.config.width
        self.max_progress = max(list(x.order for x in self.config.queues)) + 1

        self.init_agent_data = {}
        self.init_attendants_data = {}
        self.init_queue_data = [[] for _ in range(self.max_progress)]
        # --

        # Setting init_agent_data
        for group in self.config.groups:
            for _ in range(group.amount):
                i = len(self.init_agent_data)
                self.init_agent_data[f"agent_{i}"] = Agent(
                    id=f"agent_{i}",
                    group=group.name,
                    starting_products=group.starting_products,
                )
        # --

        # Setting init_queue_data
        for queue in self.config.queues:
            self.init_queue_data[queue.order].append(queue)
        # --

        # Setting init_attendants_data
        for attendant in self.config.attendants:
            self.init_attendants_data[attendant.id] = attendant
        # --

        # World vars
        self.possible_agents = list(self.init_agent_data)
        self.agents = []

        self.agents_data: dict[str, Agent] = deepcopy(self.init_agent_data)
        self.attendants_data: dict[str, Attendant] = deepcopy(self.init_attendants_data)
        self.queues_data: list[list[Queue]] = deepcopy(self.init_queue_data)
        self.entrances_data: list[Entrance] = deepcopy(self.config.entrances)
        self.exits_data: list[Exit] = deepcopy(self.config.exits)

        self.agent_dist_map = {i: None for i in self.possible_agents}
        # --

        self.max_targets = max(list(len(x) for x in self.queues_data))
        # Action and Observation spaces
        self.unflattened_observation_spaces = {
            i: spaces.Dict(
                {
                    "own_position": gymnasium.spaces.Box(
                        low=-1024, high=1024, shape=(2,), dtype=np.float32
                    ),
                    "target_position": gymnasium.spaces.Box(
                        low=-1024,
                        high=1024,
                        shape=(self.max_targets, 2),
                        dtype=np.float32,
                    ),
                    "distance": spaces.Box(
                        low=0,
                        high=255,
                        shape=(4,),
                        dtype=np.float32,
                    ),
                }
            )
            for i in self.possible_agents
        }

        self._observation_spaces = {
            i: spaces.flatten_space(self.unflattened_observation_spaces[i])
            for i in self.unflattened_observation_spaces
        }

        self._action_spaces = {i: spaces.Discrete(6) for i in self.possible_agents}

        self.observation_space = lambda agent: self._observation_spaces[agent]
        self.action_space = lambda agent: self._action_spaces[agent]
        # --

        self.metadata["render_fps"] = render_fps
        self.render_mode = render_mode
        self.group_map = {}
        self.tile_map = self._load_images() if render_mode == "human" else {}

        self.screen = None
        self.font = None
        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self._agent_selector = agent_selector(self.agents)

    def reset(self, seed=None, options=None):
        self.frames = 0

        self.agents_data: dict[str, Agent] = deepcopy(self.init_agent_data)
        self.attendants_data: dict[str, Attendant] = deepcopy(self.init_attendants_data)
        self.queues_data: list[list[Queue]] = deepcopy(self.init_queue_data)
        self.entrances_data: list[Entrance] = deepcopy(self.config.entrances)
        self.exits_data: list[Exit] = deepcopy(self.config.exits)
        self.colision = copy(self.map)

        self.agent_dist_map = {i: None for i in self.possible_agents}

        self.rewards = {i: 0 for i in self.possible_agents}
        self._cumulative_rewards = {i: 0 for i in self.possible_agents}

        self.terminations = {i: False for i in self.possible_agents}
        self.truncations = {i: False for i in self.possible_agents}
        self.infos = {i: {} for i in self.possible_agents}

        self.agents = []
        self._first_deployments()

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        self._agent_act(agent, action)

        if self._agent_selector.is_last():
            self._update_attendants()
            self._update_queues()
            self._allocate_rewards()
            self._update_exit()
            self._update_door()

            self._agent_selector.reinit(self.agents)

            if self.render_mode == "human":
                self.render()

            self.frames += 1

        terminate = not self.run
        truncate = self.frames >= self.max_cycles
        self.terminations = {a: terminate for a in self.possible_agents}
        self.truncations = {a: truncate for a in self.possible_agents}

        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        self._clear_rewards()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        self._deads_step_first()

    def observe(self, agent):
        agent = self.agents_data[agent]

        own_position_obs = agent.pos.tuple
        distance_obs = self._get_agent_dist_obs(agent.id)
        target_postion = self._get_target_pos(agent.id)

        return spaces.flatten(
            self.unflattened_observation_spaces[agent.id],
            {
                "own_position": own_position_obs,
                "target_position": target_postion,
                "distance": distance_obs,
            },
        ).astype("float32")

    def render(self):
        tile_size = 40

        screen_width = tile_size * self.config.width
        screen_height = tile_size * self.config.height

        if self.screen is None:
            pygame.init()
            self.font = pygame.font.SysFont("arial", tile_size)

            if self.render_mode == "human":
                pygame.display.set_caption("Crowd-RL Prototype")
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            elif self.render_mode == "rgb_array":
                self.screen = pygame.Surface((screen_width, screen_height))
            else:
                gymnasium.logger.warn(
                    "You are calling render method without specifying any render mode."
                )
                return

        self.screen.fill([255, 255, 255])
        full_feature_map = copy(self.map)
        text_list = []

        assert self.font

        for queue_order in self.queues_data:
            for queue in queue_order:
                for spot in queue.wait_spots:
                    full_feature_map[spot.y][spot.x] = 4

                if not queue.full:
                    prio_wait_spot = queue.wait_spots[queue.busy]
                    full_feature_map[prio_wait_spot.y][prio_wait_spot.x] = 2

        for _, attendant in self.attendants_data.items():
            full_feature_map[attendant.pos.y][attendant.pos.x] = 3

        for entrance in self.entrances_data:
            full_feature_map[entrance.pos.y][entrance.pos.x] = 6

        for _exit in self.exits_data:
            full_feature_map[_exit.pos.y][_exit.pos.x] = 5

        for _, agent_char in self.agents_data.items():
            if agent_char.deployed:
                text_list.append(
                    (
                        agent_char.pos.tuple,
                        self.font.render(
                            str(agent_char.products), True, (255, 255, 255)
                        ),
                    )
                )

                full_feature_map[agent_char.pos.y][agent_char.pos.x] = self.group_map[
                    agent_char.group
                ]

        for index, tile in enumerate(full_feature_map.flat):
            x = index % self.width
            y = int(index / self.width)

            if tile != 0:
                self.screen.blit(
                    self.tile_map[tile],
                    ((x * tile_size, y * tile_size)),
                )

        for text in text_list:
            self.screen.blit(
                text[1],
                (
                    (tile_size * 0.25) + (tile_size * text[0][0]),
                    (tile_size * 0.10) + (tile_size * text[0][1]),
                ),
            )

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
        self.screen = None

    def _get_agent_dist_map(self, agent_id):
        self.agent_dist_map[agent_id] = self._distance_map_merge_targets(
            self._get_target_pos(agent_id)
        )

        return self.agent_dist_map[agent_id]

    def _get_agent_dist_obs(self, agent_id):
        directions = {
            0: [-1, 0],
            1: [1, 0],
            2: [0, 1],
            3: [0, -1],
        }

        r = np.full((4,), 255, "int16")
        dist_map = self._get_agent_dist_map(agent_id)

        current_pos = self.agents_data[agent_id].pos

        for key, di in directions.items():
            pos = [current_pos.x + di[0], current_pos.y + di[1]]
            if 0 <= pos[1] < self.height and 0 <= pos[0] < self.width:
                r[key] = dist_map[pos[1], pos[0]]

        return r

    def _get_target_pos(self, agent_id: str):
        target_array = np.full((self.max_targets, 2), -1, "int16")
        agent = self.agents_data[agent_id]

        if agent.progress >= self.max_progress:
            targets = [x.pos.tuple for x in self.exits_data if agent.group in x.accepts]
            target_array[: len(targets)] = np.array(targets, "int16")

        elif agent.cleared_for:
            targets = [self.attendants_data[agent.cleared_for].pos.tuple]
            target_array[: len(targets)] = np.array(targets, "int16")

        else:
            busy = list(
                queue.busy
                for queue in self.queues_data[agent.progress]
                if agent.group in queue.accepts and not queue.full
            )

            if busy:
                min_busy = min(busy)

                free_queues = list(
                    queue.wait_spots[queue.busy].tuple
                    for queue in self.queues_data[agent.progress]
                    if not queue.full
                    and queue.busy < min_busy + 2
                    and agent.group in queue.accepts
                )
            else:
                free_queues = []

            if len(free_queues) > 0:
                target_array[: len(free_queues)] = np.array(free_queues, "int16")

        return target_array

    def _update_queues(self):
        for queue_order in self.queues_data:
            for queue in queue_order:
                changed = True
                while not queue.full and changed:
                    changed = False

                    prio_wait_spot = queue.wait_spots[queue.busy].tuple
                    for _, agent in self.agents_data.items():
                        if (
                            agent.progress == queue.order
                            and agent.group in queue.accepts
                            and not agent.cleared_for
                            and not agent.in_queue
                            and agent.pos.tuple == prio_wait_spot
                        ):
                            queue.members.append(agent)
                            agent.in_queue = True

                            changed = True
                            break

        for queue_order in self.queues_data:
            for queue in queue_order:
                if queue.free_spots > 0:
                    for q_pos, agent in enumerate(queue.members):
                        self._move_agent(
                            agent.id,
                            queue.wait_spots[q_pos].x,
                            queue.wait_spots[q_pos].y,
                        )
                    queue.free_spots -= 1

                if queue.busy:
                    for attendant_id in queue.attendants:
                        _attendant_obj = self.attendants_data[attendant_id]
                        if not _attendant_obj.assigned:
                            agent = queue.members.popleft()
                            _attendant_obj.assigned = agent.id

                            agent.cleared_for = attendant_id
                            agent.in_queue = False
                            queue.free_spots += 1

                            return

    def _get_interval(self, value: tuple[int, int] | int) -> int:
        if isinstance(value, tuple):
            return self.np_random.integers(value[0], value[1])
        return value

    def _update_attendants(self):
        for _, attendant in self.attendants_data.items():
            if attendant.busy:
                attendant.cooldown -= 1
                if attendant.cooldown <= 0:
                    assert attendant.client_id
                    agent = self.agents_data[attendant.client_id]

                    agent.products -= 1
                    if agent.products <= 0:
                        attendant.busy = False
                        attendant.assigned = None
                        attendant.client_id = None

                        agent.in_attendance = False
                        agent.progress += 1

                        if agent.progress < self.max_progress:
                            # agent.products = agent.starting_products
                            while not [
                                i
                                for i in self.queues_data[agent.progress]
                                if agent.group in i.accepts
                            ]:
                                agent.progress += 1
                                if agent.progress >= self.max_progress:
                                    break
                    else:
                        attendant.cooldown = self._get_interval(
                            attendant.processing_time
                        )

            else:
                for key, agent in self.agents_data.items():
                    if agent.id == attendant.assigned and agent.pos == attendant.pos:
                        attendant.busy = True
                        attendant.cooldown = self._get_interval(
                            attendant.processing_time
                        )

                        attendant.client_id = key

                        agent.cleared_for = None
                        agent.in_attendance = True
                        break

    def _get_next_agent(self, accepts: list[str]) -> Agent | None:
        accepted_agents = [
            agent
            for agent in self.agents_data.values()
            if (not agent.deployed and agent.group in accepts)
        ]

        if accepted_agents:
            rand = self.np_random.integers(0, len(accepted_agents))
            return accepted_agents[rand]

        return None

    def _first_deployments(self):
        fastest_door: Entrance | None = None
        deployee: Agent | None = None

        for entrance in self.entrances_data:
            deployee = self._get_next_agent(entrance.accepts)

            if not deployee:
                continue

            if not fastest_door:
                fastest_door = entrance
            else:
                if entrance.rate < fastest_door.rate:
                    fastest_door = entrance

        assert fastest_door
        assert deployee

        self._deploy(deployee, fastest_door)

    def _deploy(self, agent: Agent, entrance: Entrance):
        agent.pos.x, agent.pos.y = (
            entrance.pos.x,
            entrance.pos.y,
        )

        agent.deployed = True
        agent.products = self._get_interval(agent.starting_products)
        self.agents.append(agent.id)
        entrance.cooldown = entrance.rate

    def _update_door(self):
        for entrance in self.entrances_data:
            if entrance.cooldown <= 0:
                entrance.cooldown = entrance.rate
                deployee = self._get_next_agent(entrance.accepts)

                if deployee:
                    self._deploy(deployee, entrance)

            else:
                entrance.cooldown -= 1

    def _update_exit(self):
        for _exit in self.exits_data:
            for key, agent in self.agents_data.items():
                if agent.pos.tuple == _exit.pos.tuple:
                    if agent.progress >= self.max_progress:
                        self.colision[agent.pos.y][agent.pos.x] = 0
                        self.agents.remove(key)
                        self.agents_data[key] = deepcopy(self.init_agent_data[key])

    def _agent_act(self, agent_id: str, action: int):
        # [no_action, move_left, move_right, move_down, move_up]
        directions = {
            1: [-1, 0],
            2: [1, 0],
            3: [0, 1],
            4: [0, -1],
        }

        if action in directions:
            agent = self.agents_data[agent_id]
            if agent.in_queue or agent.in_attendance:
                return

            current_pos = agent.pos
            direction = directions[action]

            new_x = current_pos.x + direction[0]
            new_y = current_pos.y + direction[1]

            if 0 <= new_y < self.height and 0 <= new_x < self.width:
                if self.colision[new_y, new_x] == 0:
                    self._move_agent(agent_id, new_x, new_y)
                    return

            agent.stuck = True

    def _move_agent(self, agent_id: str, new_x: int, new_y: int):
        agent = self.agents_data[agent_id]

        current_pos = agent.pos
        agent.stuck = False

        self.colision[current_pos.y][current_pos.x] = 0
        self.colision[new_y][new_x] = 1
        current_pos.x = new_x
        current_pos.y = new_y

    def _distance_map_merge_targets(self, targets):
        dist_map = np.full((self.config.height, self.config.width), 255, "int16")
        for x in targets:
            dist_map = np.minimum(dist_map, self._distance_map_target(tuple(x)))

        return dist_map

    def _distance_map_target(self, target):
        directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        q = deque()
        q.append(target)

        dist_map = np.full((self.config.height, self.config.width), 255, "int16")
        dist_map[target[1]][target[0]] = 0

        full_feature_map = copy(self.map)
        for agent in self.agents_data.values():
            if agent.in_attendance or agent.in_queue or agent.stuck:
                full_feature_map[agent.pos.y][agent.pos.x] = 1

        while q:
            node = q.popleft()
            dist = dist_map[node[1]][node[0]]

            for di in directions:
                next_node = [node[0] + di[0], node[1] + di[1]]

                if 0 <= next_node[1] < self.height and 0 <= next_node[0] < self.width:
                    if (
                        dist_map[next_node[1]][next_node[0]] == 255
                        and full_feature_map[next_node[1]][next_node[0]] == 0
                    ):
                        dist_map[next_node[1]][next_node[0]] = dist + 1
                        q.append(next_node)

        return dist_map

    def _allocate_rewards(self):
        self.rewards = {
            i: -1 if self.agents_data[i].deployed else 0 for i in self.possible_agents
        }
        for key in self.agents:
            agent = self.agents_data[key]
            if not agent.in_queue and not agent.in_attendance:
                dist_map = self.agent_dist_map[key]

                current_pos = agent.pos
                current_dist = dist_map[current_pos.y][current_pos.x]

                if agent.proximity is None:
                    agent.proximity = current_dist
                elif agent.proximity > current_dist:
                    self.rewards[key] = 1
                    agent.proximity = current_dist

    def _load_images(self):
        tile_size = 40

        agent = get_image(os.path.join("img", "agent.png"))
        agent = pygame.transform.scale(agent, (tile_size, tile_size))

        wall = get_image(os.path.join("img", "wall.png"))
        wall = pygame.transform.scale(wall, (tile_size, tile_size))

        target = get_image(os.path.join("img", "target.png"))
        target = pygame.transform.scale(target, (tile_size, tile_size))

        target_inactive = get_image(os.path.join("img", "target_inactive.png"))
        target_inactive = pygame.transform.scale(
            target_inactive, (tile_size, tile_size)
        )

        target_primary = get_image(os.path.join("img", "target_primary.png"))
        target_primary = pygame.transform.scale(target_primary, (tile_size, tile_size))

        entrance_sprite = get_image(os.path.join("img", "entrance.png"))
        entrance_sprite = pygame.transform.scale(
            entrance_sprite, (tile_size, tile_size)
        )

        exit_sprite = get_image(os.path.join("img", "exit.png"))
        exit_sprite = pygame.transform.scale(exit_sprite, (tile_size, tile_size))

        tiles = {
            1: wall,
            2: target,
            3: target_primary,
            4: target_inactive,
            5: exit_sprite,
            6: entrance_sprite,
        }

        colors = distinctipy.get_colors(len(self.config.groups), pastel_factor=0.7)

        colors = [(int(v * 255) for v in color + (0,)) for color in colors]

        for i, group in enumerate(self.config.groups):
            single_agent = copy(agent)
            fill(
                single_agent,
                (colors[i]),
            )
            tiles[9 + i] = single_agent
            self.group_map[group.name] = 9 + i

        return tiles

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
