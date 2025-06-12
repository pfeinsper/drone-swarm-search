from enum import Enum
from collections import namedtuple

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
<<<<<<< HEAD
=======
BLUE = (0, 51, 153)
>>>>>>> 17c652b (Finished everything)


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

<<<<<<< HEAD
=======
class AdvancedActions(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    STAY = 8

>>>>>>> 17c652b (Finished everything)

Reward = namedtuple(
    "Reward",
    [
        "default",
        "leave_grid",
        "exceed_timestep",
        "drones_collision",
        "search_cell",
        "search_and_find",
    ],
)
