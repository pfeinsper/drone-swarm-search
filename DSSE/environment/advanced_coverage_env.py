from random import random, randint
import functools
import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete, Box, Tuple
from .constants import AdvancedActions, BLUE, Reward
from .entities.person import Person
from .env_base import DroneSwarmSearchBase
from .simulation.dynamic_probability import ProbabilityMatrix
import os
from datetime import datetime, timedelta
import h5py
from .simulation.particle_interface import ParticleInterface
from math import sqrt, tan, pi
from copy import deepcopy

class AdvancedCoverageDroneSwarmSearch(DroneSwarmSearchBase):
    """
    PettingZoo based environment for SAR missions using drones. Has more realistic simulation than normal 
    coverage environment and includes interactions with a dynamic probability distribution. 
    """

    metadata = {
        "name": "AdvancedCoverageEnvironment",
    }

    possible_actions = {action for action in AdvancedActions}

    reward_scheme = Reward(
        default=0,
        leave_grid=0,
        exceed_timestep=0,
        drones_collision=0,
        search_cell=0,
        search_and_find=0
    )

    def __init__(self, dataset_pth:str,
                 drone_amount:int, 
                 drone_speed: float, 
                 drone_height:float, 
                 survival_time:timedelta,
                 drone_pod:float=0.8,
                 drone_fov=90,
                 grid_cell_size:float=None, 
                 dataset_example:str=None, 
                 time_step:timedelta=None,
                 pre_render_time:timedelta=timedelta(milliseconds=0),
                 render_gradient:bool=True,
                 render_grid=False,
                 render_mode="ansi",
                 render_fps=5,
                 square_matrix:bool=True):
        """Initializes the environment. To start the environment, .reset() must be called with 
        drone_positions.

        :param dataset_pth: The path to the h5 dataset
        :type dataset_pth: str
        :param drone_amount: the number of drones 
        :type drone_amount: int
        :param drone_speed: How fast each drone is going in m/s
        :type drone_speed: float
        :param drone_height: The height the drone flies at. Is used in determining drone feild of view
        :type drone_height: float
        :param survival_time: How long the victim of the disaster is expected to survive. Is the termination requirement
        :type survival_time: timedelta
        :param drone_pod: the probabillity the drone finds a person if they are in the same grid cell, defaults to 0.8
        :type drone_pod: float, optional
        :param drone_fov: the feild of view of the drones camera when pointed towards the ground, defaults to 90
        :type drone_fov: int, optional
        :param grid_cell_size: The side length of the grid cell in m. This parameter and time_step are very closely related. 
            If both are left as None, the default interpretation is that the grid cell size is determined by the area the 
            drone can 'see' at any given time.If the grid cell size is given, it is asumed that the environment has been 
            broken down into sectors that each drone is responsible for searching in a parallel track search. For whatever 
            reason, if the grid cell size is not given, but the time step is the environment will let you chose the 
            time_step but will still asumes grid cell size is determined by the area the drone can 'see' at any given time. 
            Defaults to None
        :type grid_cell_size: float, optional
        :param dataset_example: The name of the dataset example in the dataset that you chose. Default is random choice.
          Defaults to None
        :type dataset_example: str, optional
        :param time_step: The length of time it takes for a drone to move between cells. It is reccomended that 
            this parameter is left unchanged.

            This parameter and time_step are very closely related. 
            If both are left as None, the default interpretation is that the grid cell size is determined by the area the 
            drone can 'see' at any given time.If the grid cell size is given, it is asumed that the environment has been 
            broken down into sectors that each drone is responsible for searching in a parallel track search. For whatever 
            reason, if the grid cell size is not given, but the time step is the environment will let you chose the 
            time_step but will still asumes grid cell size is determined by the area the drone can 'see' at any given time. 
            Defaults to None, defaults to None
        :type time_step: timedelta, optional
        :param pre_render_time: The time since last recorded position that the search states, defaults to timedelta(milliseconds=0)
        :type pre_render_time: timedelta, optional
        :param render_gradient: whether or not to render the probability gradient, defaults to True
        :type render_gradient: bool, optional
        :param render_grid: reccomended to be false for large grid sizes, defaults to False
        :type render_grid: bool, optional
        :param render_mode: either "ansi" or "human". human means that the pygame display is rendered.
            "ansi" is used for testing, training, and all development related things. defaults to "ansi"
        :type render_mode: str, optional
        :param render_fps: The goal fps for the pygame simultion. , defaults to 5
        :type render_fps: int, optional
        :param square_matrix: Whether or not the observations from the environment should come from a square matrix. 
        This also affects whether the movement is confined to a rectangulaar or square matrix. If square matrix is true,
        zero values will be used to make it square, defaults to True
        :type square_matrix: bool, optional
        :raises Warning: _description_
        """
        
        self.dataset_pth=dataset_pth
        self.drone_speed=drone_speed
        self.drone_height=drone_height
        self.grid_cell_size=grid_cell_size
        self.survival_time = survival_time
        self.pre_render_time = pre_render_time
        self.render_grid=render_grid
        self.time_step = time_step
        self.drone_pod = drone_pod
        self.square_matrix = square_matrix
        self.dataset_example = dataset_example

        if dataset_example == None:
            with h5py.File(self.dataset_pth, "r") as ds:
                print(np.array(ds.keys()))
                self.dataset_example = np.random.choice([i for i in ds.keys() if i.startswith("example")])
                print(f"Dataset Example: {self.dataset_example}")

        assert drone_fov < 180
        if self.grid_cell_size == None and self.time_step == None:
            # If both are left as None, the default interpretation is that the grid cell size is determined 
            # by the area the drone can 'see' at any given time
            self.grid_cell_size = sqrt(2) * drone_height * tan(drone_fov * pi / 360)
            self.time_step = timedelta(seconds=self.grid_cell_size / self.drone_speed)
        elif self.grid_cell_size != None and self.time_step == None:
            # If the grid cell size is given, it is asumed that the environment has been broken down
            # into sectors that each drone is responsible for searching in a parallel track search
            s = sqrt(2) * drone_height * tan(drone_fov * pi / 360)
            time_step = self.grid_cell_size ** 2 / (drone_speed * s) 
            self.time_step = timedelta(seconds=time_step)
        elif self.grid_cell_size == None and self.time_step != None:
            # For whatever reason, if the grid cell size is not given, but the time step is
            # The environment will let you chose the time_step but will still asumes
            # grid cell size is determined by the area the drone can 'see' at any given time
            self.grid_cell_size = sqrt(2) * drone_height * tan(drone_fov * pi / 360)
            raise Warning("Grid cell size is not given, but time step is. The grid cell size will be determined by the area the drone can 'see' at any given time.")

        # The following four variables are all connected to self.particle_interface and need to be updated evertime 
        # self.particle_interface.step() is called
        self.particle_interface = ParticleInterface(self.dataset_pth, self.dataset_example, self.time_step, 
            cell_size=self.grid_cell_size, pre_render_time=self.pre_render_time, square_matrix=self.square_matrix)
        
        self.probability_matrix = self.particle_interface # This is just for the super class

        # This matrix tracks how many times each cell is visited
        # It doesn't matter if the matrix is square or not, since any additional cells will be zero, and this is used for counting non-zero cells
        self.coverage_info_matrix = np.zeros_like(self.particle_interface.get_raw_matrix()) 

        # This is the cumulative probability of success
        self.cumm_p_success = self.particle_interface.get_cummulative_p_success()
        
        super().__init__(
            grid_size=max(self.particle_interface.prob_matrix_shape),
            render_mode=render_mode,
            render_grid=render_grid,
            render_gradient=render_gradient,
            timestep_limit=0,
            drone_amount=drone_amount,
            drone_speed=drone_speed,
            probability_of_detection=drone_pod,
            grid_cell_size=self.grid_cell_size, # in m
            render_fps=render_fps
        )

        # After init, there is a connection to the dataset file (created with the particle interface), the number of
        # agents has been set, but their positions have not been set.

    def reset(
        self,
        options=None,
    ) -> tuple[dict[str, tuple[tuple[int, int], np.ndarray]], dict[str, dict]]:
        """_summary_

        :param seed: _description_, defaults to None
        :type seed: _type_, optional
        :param options: _description_, defaults to None
        :type options: _type_, optional
        :return: _description_
        :rtype: tuple[dict[str, tuple[tuple[int, int], np.ndarray]], dict[str, dict]]
        """
        observations, _ = super().reset(seed=0, options=options)
        infos = self.compute_infos(False)
        return observations, infos

    def pre_search_simulate(self):
        pass

    def create_observations(self) -> dict[str, tuple[tuple[int, int], np.ndarray]]:
        """Creates observations given the current state of the environment

        :return: Returns a dictionary, where the keys are the name of the agents, like drone0, drone1, ... droneN.
        The value is a tuple. This first element of the tuple is the position of the agent in row column format. For example,
        (row, col), (0, 5), (4, 5). The second element of the tuple is the probability matrix
        :rtype: dict
        """
        obs = {}
        prob_matrix = self.particle_interface.get_raw_matrix()
        for idx, agent in enumerate(self.agents):
            observation = (
                self.agents_positions[idx],
                prob_matrix,
            )
            obs[agent] = observation
        return obs

    def step(self, actions: dict[str, int]) -> tuple[dict[str, tuple[tuple[int, int], np.ndarray]], 
                                                    dict[str, float], 
                                                    dict[str, bool], 
                                                    dict[str, bool], 
                                                    dict[str, dict]]:
        """
        Returns a tuple with (observations, rewards, terminations, truncations, infos)
        """
        if not self._was_reset:
            raise ValueError("Please reset the env before interacting with it")
        
        # Search the grid, step the particle interface, and update coverage info matrix
        # When tranisitioning to the next step, the agents search the grid they were in
        old_prob_matrix = deepcopy(self.particle_interface.get_raw_matrix())
        _, self.cumm_p_success, sim_ended = self.particle_interface.step(self.agents_positions, self.drone_pod)
        self.coverage_info_matrix[tuple(zip(*self.agents_positions))] += 1

        terminations = {a: False for a in self.agents}
        rewards = {a: self.reward_scheme.default for a in self.agents}
        truncations = {a: False for a in self.agents}
        # Iterate over the agents and update their positions
        for idx, agent in enumerate(self.agents):
            if agent not in actions:
                raise ValueError("Missing action for " + agent)

            drone_action = actions[agent]
            if drone_action not in self.action_space(agent):
                raise ValueError("Invalid action for " + agent)

            # Check truncation conditions (overwrites termination conditions)
            # Truncations are used to indicate that the simulation only goes up to a certain point
            if sim_ended:
                truncations[agent] = True
                terminations[agent] = True
                continue

            # Check termination conditions
            # Terminations are used to indicate that the person of interest has deceased
            if self.survival_time < self.particle_interface.get_elapsed_time():
                terminations[agent] = True
                rewards[agent] = self.reward_scheme.exceed_timestep
                continue
            
            drone_r, drone_c = self.agents_positions[idx]

            new_position = self.move_drone((drone_r, drone_c), drone_action)
            if not self.is_valid_position(new_position):
                rewards[agent] = self.reward_scheme.leave_grid
            else:
                self.agents_positions[idx] = new_position

            # If the agent is searching, it will find the person with a probability of detection (POD)
            # The POD is multiplied by the probability of finding the person in the cell
            # The reward is the cumulative probability of success
            rewards[agent] += old_prob_matrix[self.agents_positions[idx][0], self.agents_positions[idx][1]] * self.drone_pod

        observations = self.create_observations()

        # Render the screen
        self.render_step(any(terminations.values()))

        # # Update the timestep
        # self.timestep += 1

        # If any agent has terminated or truncated, reset the agents
        # This is a requirement for the PettingZoo parallel environment
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, self.compute_infos(any(terminations.values()))
    
    def is_valid_position(self, position: tuple[int, int]) -> bool:
        valid_r = position[0] >= 0 and position[0] < self.particle_interface.prob_matrix_shape[0]
        valid_c = position[1] >= 0 and position[1] < self.particle_interface.prob_matrix_shape[1]
        return valid_r and valid_c
    
    def render_step(self, terminal):
        if self.render_mode == "human":
            if terminal:
                self.pygame_renderer.render_episode_end_screen(
                    f"The Probabillity of Finding the Lost Person in {self.particle_interface.get_elapsed_time()} is {round(self.cumm_p_success, 3)}", BLUE
                )
            else:
                self.render()
        
    def compute_infos(self, is_completed: bool) -> dict[str, dict]:
        # This is the percent of non-zero cells that have been visited
        non_zero_cells = self.particle_interface.get_raw_matrix() > 0 # All non-zero cells
        covered_cells = self.coverage_info_matrix > 0
        covered_and_non_zero_cells = np.logical_and(covered_cells, non_zero_cells)
        coverage_rate = np.sum(covered_and_non_zero_cells) / np.sum(non_zero_cells) if np.sum(non_zero_cells) > 0 else 0

        infos = {
            "is_completed": is_completed,
            "coverage_rate": coverage_rate, # This is the percent of non-zero cells that have been visited
            "repeated_cells": np.sum(self.coverage_info_matrix > 1),
            "visited_cells": np.sum(self.coverage_info_matrix > 0), # This is the number of cells that have been visited
            "accumulated_pos": self.cumm_p_success,
        }
        return {drone: infos for drone in self.agents}
    
    def move_drone(self, position, action):
        """
        Returns a new position in (row, column) format based on the action taken.
        The actions are defined in the AdvancedActions enum. The position is a tuple of (row, column).
        """
        match action:
            case AdvancedActions.LEFT.value:  # LEFT
                new_position = (position[0], position[1] - 1)
            case AdvancedActions.RIGHT.value:  # RIGHT
                new_position = (position[0], position[1] + 1)
            case AdvancedActions.UP.value:  # UP
                new_position = (position[0] - 1, position[1])
            case AdvancedActions.DOWN.value:  # DOWN
                new_position = (position[0] + 1, position[1])
            case AdvancedActions.UP_LEFT.value:  # UP_LEFT
                new_position = (position[0] - 1, position[1] - 1)
            case AdvancedActions.UP_RIGHT.value:  # UP_RIGHT
                new_position = (position[0] - 1, position[1] + 1)
            case AdvancedActions.DOWN_LEFT.value:  # DOWN_LEFT
                new_position = (position[0] + 1, position[1] - 1)
            case AdvancedActions.DOWN_RIGHT.value:  # DOWN_RIGHT
                new_position = (position[0] + 1, position[1] + 1)
            case _:
                new_position = position

        return new_position
    
    def render(self):
        # Need to edit this method because the agen positions are in (row, column) format
        # and the pygame renderer expects them in (x, y) format
        self.pygame_renderer.render_map()
        self.pygame_renderer.render_entities([tuple(reversed(i)) for i in self.agents_positions])
        self.pygame_renderer.refresh_screen()