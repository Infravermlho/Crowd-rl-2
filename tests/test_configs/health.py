from crowd_rl.crowd_rl_v0 import Attendant, Config, Coords, Entrance, Exit, Queue
from crowd_rl.environment.schema import Group

attendants = [
    Attendant(id="at1", pos=Coords(x=14, y=11), rate=20),
    Attendant(id="at2", pos=Coords(x=13, y=11), rate=20),
    Attendant(id="at3", pos=Coords(x=12, y=11), rate=20),
    Attendant(id="med1", pos=Coords(x=1, y=5), rate=50),
    Attendant(id="med2", pos=Coords(x=1, y=8), rate=50),
    Attendant(id="med3", pos=Coords(x=1, y=11), rate=50),
    Attendant(id="med4", pos=Coords(x=7, y=5), rate=50),
    Attendant(id="med5", pos=Coords(x=7, y=8), rate=50),
    Attendant(id="med6", pos=Coords(x=7, y=11), rate=50),
]

queues = [
    Queue(
        order=0,
        accepts=["oftalmo", "checkup"],
        attendants=["at1", "at2"],
        wait_spots=[
            Coords(x=14, y=9),
            Coords(x=14, y=8),
            Coords(x=14, y=7),
            Coords(x=14, y=6),
            Coords(x=14, y=5),
        ],
    ),
    Queue(
        order=1,
        accepts=["checkup"],
        attendants=["med1", "med2", "med3"],
        wait_spots=[
            Coords(x=10, y=4),
            Coords(x=10, y=5),
            Coords(x=10, y=6),
            Coords(x=10, y=7),
            Coords(x=10, y=8),
        ],
    ),
    Queue(
        order=1,
        accepts=["oftalmo"],
        attendants=["med4", "med5", "med6"],
        wait_spots=[
            Coords(x=11, y=4),
            Coords(x=11, y=5),
            Coords(x=11, y=6),
            Coords(x=11, y=7),
            Coords(x=11, y=8),
        ],
    ),
    Queue(
        order=2,
        accepts=["checkup"],
        attendants=["at1", "at2", "at3"],
        wait_spots=[
            Coords(x=12, y=4),
            Coords(x=12, y=5),
            Coords(x=12, y=6),
            Coords(x=12, y=7),
            Coords(x=12, y=8),
        ],
    ),
]

entrances = [
    Entrance(pos=Coords(x=14, y=0), rate=12, accepts=["oftalmo", "checkup"]),
]

exits = [
    Exit(pos=Coords(x=13, y=0), accepts=["oftalmo", "checkup"]),
]

groups = [Group(name="oftalmo", amount=10), Group(name="checkup", amount=5)]

worldmap = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

env_config = Config(
    attendants=attendants,
    worldmap=worldmap,
    groups=groups,
    queues=queues,
    entrances=entrances,
    exits=exits,
)
