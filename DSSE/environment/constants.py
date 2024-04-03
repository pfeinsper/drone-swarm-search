from enum import Enum

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Actions
class Actions(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    SEARCH = 8

# Rewards
class Rewards(Enum):
    DEFAULT = 1
    LEAVE_GRID = -100_000
    EXCEED_TIMESTEP = -100_000
    DRONES_COLLISION = -100_000
    SEARCH_CELL = 1
    SEARCH_AND_FIND = 100_000
