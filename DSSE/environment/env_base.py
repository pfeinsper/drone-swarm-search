import functools
import numpy as np
from abc import ABC, abstractmethod
from pettingzoo import ParallelEnv
from .entities.drone import DroneData
from .pygame_interface import PygameInterface
from .simulation.dynamic_probability import ProbabilityMatrix
from .constants import Actions
from gymnasium.spaces import MultiDiscrete, Discrete, Tuple, Box
from copy import copy


class DroneSwarmSearchBase(ABC, ParallelEnv):
    def __init__(
        self,
        grid_size=7,
        render_mode="ansi",
        render_grid=False,
        render_gradient=True,
        timestep_limit=100,
        drone_amount=1,
        drone_speed=10,
        probability_of_detection=1,
    ) -> None:
        self.cell_size = 130  # in meters
        self.grid_size = grid_size
        self._was_reset = False
        if not isinstance(drone_amount, int):
            raise ValueError("Drone amount must be an integer")
    
        self.drone = DroneData(
            amount=drone_amount,
            speed=drone_speed,
            pod=probability_of_detection,
        )
        self.probability_of_detection = probability_of_detection

        # Error Checking
        if self.drone.amount > self.grid_size * self.grid_size:
            raise ValueError(
                "There are more drones than grid spots. Reduce number of drones or increase grid size."
            )

        if render_mode != "ansi" and render_mode != "human":
            raise ValueError("Render mode not recognized")

        self.timestep = None
        self.timestep_limit = timestep_limit
        self.time_step_relation = self.calculate_simulation_time_step(
            self.drone.speed, self.cell_size
        )

        self.possible_agents = []
        self.agents_positions = [(None, None)] * self.drone.amount
        for i in range(self.drone.amount):
            agent_name = "drone" + str(i)
            self.possible_agents.append(agent_name)

        self.render_mode = render_mode

        # Initializing render
        self.pygame_renderer = PygameInterface(
            self.grid_size, render_gradient, render_grid
        )

    def calculate_simulation_time_step(
        self, drone_max_speed: float, cell_size: float, wind_resistance: float = 0.0
    ) -> float:
        """
        Calculate the time step for the simulation based on the maximum speed of the drones and the cell size

        Args:
        max_speed: float
            Maximum speed of the drones in m/s
        cell_size: float
            Size of the cells in meters
        wind_resistance: float
            Wind resistance in m/s
        """
        return cell_size / (drone_max_speed - wind_resistance)  # in seconds

    def render(self):
        self.pygame_renderer.render_map()
        self.pygame_renderer.render_entities(self.agents_positions)
        self.pygame_renderer.refresh_screen()

    @abstractmethod
    def reset(
        self,
        seed=None,
        options=None,
    ):
        self._was_reset = True
        drones_positions = options.get("drones_positions") if options else None

        if drones_positions is not None:
            if not self.is_valid_position_drones(drones_positions):
                raise ValueError(
                    "You are trying to place the drone in a invalid position"
                )

        self.agents = copy(self.possible_agents)
        self.timestep = 0

        if drones_positions is None:
            self.default_drones_positions()
        else:
            self.required_drone_positions(drones_positions)

        if self.render_mode == "human":
            self.pygame_renderer.probability_matrix = self.probability_matrix
            self.pygame_renderer.enable_render()
            self.render()

        self.pre_search_simulate()
        observations = self.create_observations()
        infos = {drone: {"Found": False} for drone in self.agents}
        return observations, infos

    def is_valid_position_drones(self, positions: list[tuple[int, int]]) -> bool:
        seen = set()
        for position in positions:
            if not self.is_valid_position(position) or position in seen:
                return False
            seen.add(position)
        return True

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        valid_x = position[0] >= 0 and position[0] < self.grid_size
        valid_y = position[1] >= 0 and position[1] < self.grid_size
        return valid_x and valid_y

    def default_drones_positions(self):
        counter_x = 0
        counter_y = 0
        for agent_index in range(len(self.agents)):
            if counter_x >= self.grid_size:
                counter_x = 0
                counter_y += 1
            self.agents_positions[agent_index] = (counter_x, counter_y)
            counter_x += 1

    def required_drone_positions(self, drones_positions: list):
        if len(drones_positions) != len(self.possible_agents):
            raise ValueError(
                "There are more or less initial positions than drones,"
                "please make sure there are the same number of initial possitions "
                "and number of drones."
            )
        for i in range(len(drones_positions)):
            x, y = drones_positions[i]
            self.agents_positions[i] = (x, y)

    @abstractmethod
    def pre_search_simulate(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def create_observations(self):
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def step(self, actions):
        raise NotImplementedError("Method not implemented")

    def compute_drone_collision(self, terminations, rewards):
        """
        Check for drone collision and compute terminations, rewards and truncations.
        """
        for drone_1_id in range(len(self.agents)):
            for drone_2_id in range(drone_1_id + 1, len(self.agents)):
                drone_1_name = self.agents[drone_1_id]
                drone_2_name = self.agents[drone_2_id]
                if self.agents_positions[drone_1_id] == self.agents_positions[drone_2_id]:
                    terminations[drone_1_name] = True
                    terminations[drone_2_name] = True
                    rewards[drone_1_name] = self.reward_scheme.drones_collision
                    rewards[drone_2_name] = self.reward_scheme.drones_collision


    def move_drone(self, position, action):
        """
        Returns a tuple with (is_terminal, new_position, reward)
        """
        match action:
            case Actions.LEFT.value:  # LEFT
                new_position = (position[0] - 1, position[1])
            case Actions.RIGHT.value:  # RIGHT
                new_position = (position[0] + 1, position[1])
            case Actions.UP.value:  # UP
                new_position = (position[0], position[1] - 1)
            case Actions.DOWN.value:  # DOWN
                new_position = (position[0], position[1] + 1)
            case Actions.UP_LEFT.value:  # UP_LEFT
                new_position = (position[0] - 1, position[1] - 1)
            case Actions.UP_RIGHT.value:  # UP_RIGHT
                new_position = (position[0] + 1, position[1] - 1)
            case Actions.DOWN_LEFT.value:  # DOWN_LEFT
                new_position = (position[0] - 1, position[1] + 1)
            case Actions.DOWN_RIGHT.value:  # DOWN_RIGHT
                new_position = (position[0] + 1, position[1] + 1)

        return new_position

    def get_agents(self):
        return self.possible_agents

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Observation space for each agent:
        # - MultiDiscrete: (x, y) position of the agent
        # - Box: Probability matrix
        return Tuple(
            (
                MultiDiscrete([self.grid_size, self.grid_size]),
                Box(
                    low=0,
                    high=1,
                    shape=(self.grid_size, self.grid_size),
                    dtype=np.float32,
                ),
            )
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(self.possible_actions))
