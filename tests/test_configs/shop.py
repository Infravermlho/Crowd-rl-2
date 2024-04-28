from crowd_rl.crowd_rl_v0 import Attendant, Config, Coords, Entrance, Exit, Queue
from crowd_rl.environment.schema import Group

# Atendentes Prio
# Variabilidade atendimento por Attendente, Agente (Tonalidade)
attendants = [
    Attendant(id="p1", pos=Coords(x=3, y=1), rate=50),
    Attendant(id="c1", pos=Coords(x=5, y=1), rate=20),
    Attendant(id="c2", pos=Coords(x=7, y=1), rate=20),
    Attendant(id="c3", pos=Coords(x=9, y=1), rate=30),
]

queues = [
    Queue(
        order=0,
        accepts=["pref"],
        attendants=["p1"],
        wait_spots=[
            Coords(x=3, y=4),
            Coords(x=3, y=5),
            Coords(x=3, y=6),
            Coords(x=3, y=7),
            Coords(x=3, y=8),
        ],
    ),
    Queue(
        order=0,
        accepts=["common", "pref"],
        attendants=["c1"],
        wait_spots=[
            Coords(x=5, y=4),
            Coords(x=5, y=5),
            Coords(x=5, y=6),
            Coords(x=5, y=7),
            Coords(x=5, y=8),
        ],
    ),
    Queue(
        order=0,
        accepts=["common", "pref"],
        attendants=["c2"],
        wait_spots=[
            Coords(x=7, y=4),
            Coords(x=7, y=5),
            Coords(x=7, y=6),
            Coords(x=7, y=7),
            Coords(x=7, y=8),
        ],
    ),
    Queue(
        order=0,
        accepts=["common", "pref"],
        attendants=["c3"],
        wait_spots=[
            Coords(x=9, y=4),
            Coords(x=9, y=5),
            Coords(x=9, y=6),
            Coords(x=9, y=7),
            Coords(x=9, y=8),
        ],
    ),
]

entrances = [
    Entrance(pos=Coords(x=0, y=11), rate=5, accepts=["common", "pref"]),
    Entrance(pos=Coords(x=9, y=12), rate=5, accepts=["common", "pref"]),
]

exits = [
    Exit(pos=Coords(x=14, y=0), accepts=["common", "pref"]),
]

groups = [
    Group(name="common", amount=15),
    Group(name="pref", amount=5),
]

worldmap = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
]

env_config = Config(
    attendants=attendants,
    worldmap=worldmap,
    groups=groups,
    queues=queues,
    entrances=entrances,
    exits=exits,
)
