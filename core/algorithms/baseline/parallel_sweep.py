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
            yield action.value


class MultipleParallelSweep:
    def __init__(self, env) -> None:
        self.env = env
        self.grid_size = env.grid_size
        self.n_drones = len(env.possible_agents)
        self.grid_size_each_drone = self.get_each_drone_grid_size()

    def get_each_drone_grid_size(self):
        """
        Get the size of the grid that each drone will search.

        :return: The size of the grid that each drone will search
        """
        if self.n_drones not in {1, 2} and self.n_drones % 4 != 0:
            raise ValueError("The number of agents must be 1 or 2 or a multiple of 4")

        grid_size_each_drone = self.grid_size / math.sqrt(self.n_drones)

        if grid_size_each_drone % 1 != 0:
            raise ValueError("The grid size must be a multiple of the number of agents")

        return int(grid_size_each_drone)

    def get_first_drone_info(self):
        """
        Get the information of the first drone.

        :return: The information of the first drone
        """
        return DroneInfo(
            grid_size=self.grid_size_each_drone,
            initial_position=(0, 0),
            last_vertice=(self.grid_size_each_drone - 1, self.grid_size_each_drone - 1),
        )

    def get_drones_initial_positions(self):
        """
        Get all the drone cell boundaries.

        :return: All the drone cell boundaries
        """
        drones_initial_positions = []
        for i in range(0, self.grid_size, self.grid_size_each_drone):
            for j in range(0, self.grid_size, self.grid_size_each_drone):
                drones_initial_positions.append((i, j))

        return drones_initial_positions

    def generate_next_action(self):
        """
        Generate the next action of all the drones.

        :yield: The next action of all the drones
        """

        parallel_sweep = SingleParallelSweep(self.get_first_drone_info())

        for action in parallel_sweep.genarate_next_action():
            actions = {}

            for i in range(self.n_drones):
                actions[f"drone{i}"] = action

            yield actions

    def run(self):
        """
        Run the algorithm.

        :param env: The environment
        """
        drones_positions = self.get_drones_initial_positions()
        self.env.reset(drones_positions=drones_positions)

        for actions in self.generate_next_action():
            self.env.step(actions)
