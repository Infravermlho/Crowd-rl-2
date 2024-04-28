"""
Config file that will be passed to world.py
"""

from collections import deque
from typing import Deque, List, Optional, Tuple

from pydantic import BaseModel, computed_field


class Coords(BaseModel):
    x: int
    y: int

    @computed_field
    @property
    def tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


class Agent(BaseModel):
    id: str
    group: str
    starting_products: Tuple[int, int] | int

    products: int = 0
    pos: Coords = Coords(x=-1, y=-1)
    proximity: Optional[int] = None
    progress: int = 0
    deployed: bool = False
    in_queue: bool = False
    in_attendance: bool = False
    stuck: bool = False
    cleared_for: Optional[str] = None


class Group(BaseModel):
    name: str
    amount: int
    starting_products: Tuple[int, int] | int


class Entrance(BaseModel):
    pos: Coords
    rate: int
    accepts: List[str]

    cooldown: int = 0


class Exit(BaseModel):
    pos: Coords
    accepts: List[str]


class Attendant(BaseModel):
    id: str
    pos: Coords
    processing_time: Tuple[int, int] | int

    client_id: Optional[str] = None
    assigned: Optional[str] = None
    busy: bool = False
    cooldown: int = 0


class Queue(BaseModel):
    order: int
    accepts: List[str]
    attendants: List[str]
    wait_spots: List[Coords]

    free_spots: int = 0
    members: Deque[Agent] = deque()

    @computed_field
    @property
    def full(self) -> bool:
        return self.busy >= len(self.wait_spots)

    @computed_field
    @property
    def busy(self) -> int:
        return len(self.members) + self.free_spots


class Config(BaseModel):
    worldmap: List
    queues: List[Queue]
    exits: List[Exit]
    entrances: List[Entrance]
    groups: List[Group]
    attendants: List[Attendant]

    seed: Optional[int] = None

    @computed_field
    @property
    def height(self) -> int:
        return len(self.worldmap)

    @computed_field
    @property
    def width(self) -> int:
        return len(self.worldmap[0])


def json_to_config(json_world):
    return Config(**json_world)
