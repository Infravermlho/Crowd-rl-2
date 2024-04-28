from crowd_rl.crowd_rl_v0 import Config, Agent, Coords, Queue, Attendant, Entrance, Exit

queues = [
    Queue(
        order=0,
        accepts=[0],
        attendants=[
            Attendant(pos=Coords(x=2, y=1)),
        ],
        wait_spots=[
            Coords(x=6, y=4),
            Coords(x=6, y=5),
            Coords(x=6, y=6),
            Coords(x=6, y=7),
        ],
    ),
]

agents = [
    Agent(type=0, pos=Coords(x=3, y=8)),
]

worldmap = [
    [1, 0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
]

exits = [Exit(pos=Coords(x=1, y=0), accepts=[0])]

env_config = Config(
    worldmap=worldmap,
    queues=queues,
    agents=agents,
    exits=exits,
)
