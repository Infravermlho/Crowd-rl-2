from crowd_rl.crowd_rl_v0 import Agent, Attendant, Config, Coords, Entrance, Exit, Queue
from crowd_rl.environment.schema import Group

attendants = [
    Attendant(
        id="c1",
        pos=Coords(x=2, y=1),
        processing_time=30,
    ),
    Attendant(id="c2", pos=Coords(x=4, y=1), processing_time=(20, 50)),
    Attendant(id="c3", pos=Coords(x=6, y=1), processing_time=(10, 20)),
    Attendant(id="p1", pos=Coords(x=9, y=1), processing_time=50),
]

queues = [
    Queue(
        order=0,
        accepts=["common", "pref"],
        attendants=["c1", "c2", "c3"],
        wait_spots=[
            Coords(x=1, y=4),
            Coords(x=1, y=5),
            Coords(x=2, y=5),
            Coords(x=3, y=5),
            Coords(x=4, y=5),
            Coords(x=5, y=5),
            Coords(x=6, y=5),
            Coords(x=7, y=5),
            Coords(x=7, y=6),
            Coords(x=7, y=7),
            Coords(x=6, y=7),
            Coords(x=5, y=7),
            Coords(x=4, y=7),
            Coords(x=3, y=7),
            Coords(x=2, y=7),
            Coords(x=1, y=7),
            Coords(x=1, y=8),
        ],
    ),
    Queue(
        order=0,
        accepts=["pref"],
        attendants=["p1"],
        wait_spots=[
            Coords(x=9, y=4),
            Coords(x=9, y=5),
            Coords(x=9, y=6),
            Coords(x=9, y=7),
            Coords(x=9, y=8),
        ],
    ),
]

entrances = [Entrance(pos=Coords(x=0, y=11), rate=5, accepts=["common", "pref"])]

exits = [
    Exit(pos=Coords(x=14, y=12), accepts=["common", "pref"]),
]

groups = [
    Group(name="common", amount=15, starting_products=(2, 5)),
    Group(name="pref", amount=5, starting_products=(5, 7)),
]

worldmap = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
]

env_config = Config(
    attendants=attendants,
    worldmap=worldmap,
    groups=groups,
    queues=queues,
    entrances=entrances,
    exits=exits,
)
