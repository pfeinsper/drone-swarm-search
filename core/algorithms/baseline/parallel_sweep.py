from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import math


class PossibleActions(Enum):
    left = 0
    right = 1
    up = 2
    down = 3
    search = 4


@dataclass
class DroneInfo:
    drone_id: int
    grid_size: int
    initial_position: Tuple[int, int]
    last_vertice: Tuple[int, int]


class SingleParallelSweep:
    def __init__(self, drone_info: DroneInfo):
        """
        Parallel Sweep algorithm.

        The agent is the drone and starts at the bottom left corner of the grid. The goal is to
        search the person in the grid going all the way to the right, going down, and go all the
        way to the left, going down, and so on, until all the grid is searched.

        :param grid_size: The size of the grid
        """
        self.drone_id = drone_info.drone_id
        self.grid_size = drone_info.grid_size
        self.drone_x = drone_info.initial_position[0]
        self.drone_y = drone_info.initial_position[1]
        self.last_vertice_x = drone_info.last_vertice[0]
        self.last_vertice_y = drone_info.last_vertice[1]
        self.end_position_x, self.end_position_y = self.get_end_position()

    def get_end_position(self):
        """
        Get the end position of the drone.

        :return: The end position of the drone
        """
        return (
            (self.last_vertice_x, self.last_vertice_y - self.grid_size + 1)
            if self.grid_size % 2 == 0
            else (self.last_vertice_x, self.last_vertice_y)
        )

    def check_if_done(self):
        """
        Check if the drone is at the end position.

        :return: True if the drone is at the end position, False otherwise
        """
        return (
            self.drone_x == self.end_position_x and self.drone_y == self.end_position_y
        )

    def generate_next_movement(self):
        """
        Generate the next movement of the drone.

        :yield: The next action of the drone
        """
        if self.check_if_done():
            yield PossibleActions.search
            return

        is_going_right = True
        done = False

        while not done:
            if is_going_right:
                yield PossibleActions.search
                yield PossibleActions.right
                self.drone_y += 1

                if self.drone_y == self.grid_size - 1:
                    is_going_right = False
                    done = self.check_if_done()
                    if not done:
                        yield PossibleActions.down
                        self.drone_x += 1
            else:
                yield PossibleActions.search
                yield PossibleActions.left
                self.drone_y -= 1

                if self.drone_y == 0:
                    is_going_right = True
                    done = self.check_if_done()
                    if not done:
                        yield PossibleActions.down
                        self.drone_x += 1

        yield PossibleActions.search

    def genarate_next_action(self):
        """
        Generate the next action of the drone.

        :yield: The next action of the drone
        """
        for action in self.generate_next_movement():
            yield {f"drone{self.drone_id}": action.value}


class MultipleParallelSweep:
    def __init__(self, env) -> None:
        self.grid_size = env.grid_size
        self.n_drones = len(env.possible_agents)

    def get_all_drone_informations(self):
        """
        Get all the drone cell boundaries.

        :return: All the drone cell boundaries
        """
        # divide the grid in a multiple of 4
        if self.n_drones not in {1, 2} and self.n_drones % 4 != 0:
            raise ValueError("The number of agents must be 1 or 2 or a multiple of 4")

        griz_size_each_drone = self.grid_size / math.sqrt(self.n_drones)

        if griz_size_each_drone % 1 != 0:
            raise ValueError("The grid size must be a multiple of the number of agents")

        griz_size_each_drone = int(griz_size_each_drone)

        drone_cell_boundaries = []

        drone_id = 0
        for i in range(0, self.grid_size, griz_size_each_drone):
            for j in range(0, self.grid_size, griz_size_each_drone):
                drone_cell_boundaries.append(
                    DroneInfo(
                        drone_id=drone_id,
                        grid_size=griz_size_each_drone,
                        initial_position=(i, j),
                        last_vertice=(
                            i + griz_size_each_drone - 1,
                            j + griz_size_each_drone - 1,
                        ),
                    )
                )

                drone_id += 1

        return drone_cell_boundaries

    def generate_next_action(self):
        """
        Generate the next action of all the drones.

        :yield: The next action of all the drones
        """
        drone_cell_boundaries = self.get_all_drone_informations()

        parallel_sweep = SingleParallelSweep(drone_cell_boundaries[0])
        done = False

        for action in parallel_sweep.genarate_next_action():
            actions = []

            for drone in drone_cell_boundaries:
                actions.append(action)

            yield actions
