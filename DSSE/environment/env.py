from random import random
import functools
from copy import copy
import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo.utils.env import ParallelEnv
from .generator.dynamic_probability import ProbabilityMatrix
from .constants import RED, GREEN, Actions
from .pygame_interface import PygameInterface
from .drone import DroneData
from .person import Person


class DroneSwarmSearch(ParallelEnv):
    """
    PettingZoo based environment for SAR missions using drones.
    """

    def __init__(
            self,
            grid_size=7,
            render_mode="ansi",
            render_grid=False,
            render_gradient=True,
            vector=(-0.5, -0.5),
            disperse_constant=10,
            timestep_limit=100,
            person_amount=1,
            person_initial_position=(0, 0),
            drone_amount=1,
            drone_speed=10,
            drone_probability_of_detection=0.9,
    ):
        self.cell_size = 130  # in meters
        self.grid_size = grid_size
        self._was_reset = False

        self.person = Person(
                amount=person_amount,
                initial_position=person_initial_position,
            )
        self.drone = DroneData(
                amount=drone_amount,
                speed=drone_speed,
                probability_of_detection=drone_probability_of_detection,
            )

        # Error Checking
        if self.drone.amount > grid_size * grid_size:
            raise ValueError(
                "There are more drones than grid spots. Reduce number of drones or increase grid size."
            )
        if not self.is_valid_position(self.person.initial_position):
            raise ValueError("Person initial position is out of the matrix")

        if render_mode != "ansi" and render_mode != "human":
            raise ValueError("Render mode not recognized")

        self.timestep = None
        self.timestep_limit = timestep_limit
        self.disperse_constant = disperse_constant
        self.vector = vector
        self.possible_agents = []
        self.agents_positions = {}
        self.render_mode_matrix = None

        for i in range(self.drone.amount):
            self.possible_agents.append("drone" + str(i))
            self.agents_positions["drone" + str(i)] = (None, None)

        self.render_mode = render_mode
        self.probability_matrix = None

        # Initializing render
        self.pygame_renderer = PygameInterface(self.grid_size, render_gradient, render_grid)
        self.rewards_sum = {a: 0 for a in self.possible_agents}
        self.rewards_sum["total"] = 0

        self.pod = drone_probability_of_detection

        print(f"{self.pod=}")
        # Reward Function
        self.reward_scheme = {
            "default": 1,
            "leave_grid": -100000,
            "exceed_timestep": -100000,
            "drones_collision": -100000,
            "search_cell": 1,
            "search_and_find": 100000,
        }

    def is_valid_position(self, position: tuple[int, int]) -> bool:
        valid_x = position[0] >= 0 and position[0] < self.grid_size
        valid_y = position[1] >= 0 and position[1] < self.grid_size
        return valid_x and valid_y
    

    def default_drones_positions(self):
        counter_x = 0
        counter_y = 0
        for agent in self.agents:
            if counter_x >= self.grid_size:
                counter_x = 0
                counter_y += 1
            self.agents_positions[agent] = (counter_x, counter_y)
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
            self.agents_positions[self.possible_agents[i]] = (x, y)

    def render(self):
        self.pygame_renderer.render_map()
        self.pygame_renderer.render_drones(self.agents_positions.values())
        self.pygame_renderer.render_person(self.person.x, self.person.y)
        self.pygame_renderer.refresh_screen()


    def reset(
            self,
            seed=None,
            return_info=False,
            options=None,
            drones_positions=None,
            vector=None,
    ):
        self._was_reset = True
        
        if drones_positions is not None:
            if not self.is_valid_position_drones(drones_positions):
                raise ValueError("You are trying to place the drone in a invalid position")

        # reset target position
        self.person.reset_position()

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.person.reset_time_step_counter()
        self.vector = vector if vector else self.vector
        self.rewards_sum = {a: 0 for a in self.agents}
        self.rewards_sum["total"] = 0
        self.probability_matrix = ProbabilityMatrix(
            40,
            self.disperse_constant,
            self.disperse_constant,
            self.vector,
            [self.person.initial_position[1], self.person.initial_position[0]],
            self.grid_size,
        )

        if drones_positions is None:
            self.default_drones_positions()
        else:
            self.required_drone_positions(drones_positions)

        if self.render_mode == "human":
            self.pygame_renderer.probability_matrix = self.probability_matrix
            self.pygame_renderer.enable_render()
            self.render()

        observations = self.create_observations()
        return observations
    
    def is_valid_position_drones(self, positions: list[tuple[int, int]]) -> bool:
        seen = set()
        for position in positions:
            if not self.is_valid_position(position) or position in seen:
                return False
            seen.add(position)
        return True

    
    def create_observations(self):
        observations = {}

        drone_timestep = self.calculate_simulation_time_step(
            self.drone.speed,
            self.cell_size
        )

        person_speed = self.person.calculate_speed(
            water_speed=self.vector,
        )

        person_time_step = self.person.calculate_time_step(
            drone_timestep,
            person_speed,
            self.cell_size
        )
        self.person.update_time_step_relation(person_time_step)

        if self.person.reached_time_step():
            self.probability_matrix.step()

            movement_map = self.build_movement_matrix()

            movement = self.person.update_shipwrecked_position(movement_map)
            actual_movement = self.person.noise_person_movement(movement, self.vector, epsilon=0.0)

            self.person.update_position(
                x=self.safe_1d_position_update(self.person.x, actual_movement[0]),
                y=self.safe_1d_position_update(self.person.y, actual_movement[1])
            )

        probability_matrix = self.probability_matrix.get_matrix()
        for agent in self.possible_agents:
            observation = (
                (self.agents_positions[agent][0], self.agents_positions[agent][1]),
                probability_matrix,
            )
            observations[agent] = {"observation": observation}

        return observations

    def build_movement_matrix(self) -> np.array:
        """
        Builds and outputs a 3x3 matrix from the probabality matrix to use in the person movement function.
        """

        # Boundaries for the 3x3 movement matrix.
        left_x = max(self.person.x - 1, 0)
        right_x = min(self.person.x + 2, self.grid_size)
        left_y = max(self.person.y - 1, 0)
        right_y = min(self.person.y + 2, self.grid_size)

        probability_matrix = self.probability_matrix.get_matrix()
        movement_map = probability_matrix[left_y:right_y, left_x:right_x]

        # Pad the matrix
        if self.person.x == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=1)
        elif self.person.x == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=1)
        
        if self.person.y == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=0)
        elif self.person.y == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=0)
        
        return movement_map



    def safe_1d_position_update(self, previous: int, movement: int) -> int:
        """
        Updates the shipwrecked person position on a given axis, checking for edge cases first.

        Output:
            new position: int
        """
        new_position_on_axis = previous + movement
        if new_position_on_axis >= 0 and new_position_on_axis < self.grid_size:
            return new_position_on_axis
        return previous
 

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

        if not self.is_valid_position(new_position):
            return True, new_position, self.reward_scheme["leave_grid"]

        return False, new_position, self.reward_scheme["default"]

    def step(self, actions):
        """
        Returns a tuple with (observations, rewards, terminations, truncations, infos)
        """
        if not self._was_reset:
            raise ValueError("Please reset the env before interacting with it")
        
        terminations = {a: False for a in self.agents}
        rewards = {a: self.reward_scheme["default"] for a in self.agents}
        truncations = {a: False for a in self.agents}
        person_found = False
        
        for agent in self.agents:
            if agent not in actions:
                raise ValueError("Missing action for " + agent)

            drone_action = actions[agent]
            if drone_action not in self.action_space(agent):
                raise ValueError("Invalid action for " + agent)

            drone_x = self.agents_positions[agent][0]
            drone_y = self.agents_positions[agent][1]
            is_searching = drone_action == Actions.SEARCH.value

            if drone_action != Actions.SEARCH.value:
                is_terminal, new_position, reward = self.move_drone(
                    (drone_x, drone_y), drone_action
                )
                self.agents_positions[agent] = new_position
                rewards[agent] = reward
                terminations[agent] = is_terminal
                truncations[agent] = is_terminal

            
            drone_found_person = drone_x == self.person.x and drone_y == self.person.y and is_searching
            random_value = random()
            if drone_found_person and random_value < self.pod:
                time_reward_corrected = self.reward_scheme["search_and_find"] * (1 - self.timestep / self.timestep_limit)
                rewards[agent] = self.reward_scheme["search_and_find"] + time_reward_corrected
                terminations = {a: True for a in self.agents}
                truncations = {a: True for a in self.agents}
                person_found = True
            elif is_searching:
                prob_matrix = self.probability_matrix.get_matrix()
                rewards[agent] = prob_matrix[drone_y][drone_x] * 10000 if prob_matrix[drone_y][drone_x] * 100 > 1 else -100

            # Check truncation conditions (overwrites termination conditions)
            # TODO: Think, should this be >= ??
            if self.timestep > self.timestep_limit:
                rewards[agent] = self.rewards_sum[agent] * -1 + self.reward_scheme["exceed_timestep"]
                truncations[agent] = True
                terminations[agent] = True

            self.rewards_sum[agent] += rewards[agent]

        self.timestep += 1
        # Get observations
        observations = self.create_observations()
        # Get dummy infos
        infos = {"Found": person_found}

        # CHECK COLISION - Drone
        self.compute_drone_collision(terminations, rewards, truncations)
        rewards["total_reward"] = sum(rewards.values())
        self.rewards_sum["total"] += rewards["total_reward"]

        self.render_step(any(terminations.values()), person_found)        

        return observations, rewards, terminations, truncations, infos
    
    def compute_drone_collision(self, terminations, rewards, truncations):
        """
        Check for drone collision and compute terminations, rewards and truncations.
        """
        
        for drone_1_id, drone_1_position in self.agents_positions.items():
            for drone_2_id, drone_2_position in self.agents_positions.items():
                if drone_1_id == drone_2_id:
                    continue
            
                if drone_1_position[0] == drone_2_position[0] and drone_1_position[1] == drone_2_position[1]:
                    truncations[drone_1_id] = True
                    terminations[drone_1_id] = True
                    rewards[drone_1_id] = self.reward_scheme["drones_collision"]


    def render_step(self, terminal, person_found):
        if self.render_mode == "human":
            if terminal:
                if person_found:
                    self.pygame_renderer.render_episode_end_screen(f"The target was found in {self.timestep} moves", GREEN)
                else:
                    self.pygame_renderer.render_episode_end_screen("The target was not found.", RED)
            else:
                self.render()


    def get_agents(self):
        return self.possible_agents

    def calculate_simulation_time_step(self, drone_max_speed: float, cell_size: float, wind_resistance: float = 0.0) -> float:
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
        return cell_size / (drone_max_speed - wind_resistance) # in seconds

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # TODO: If x and y are the observation, then this should the observation space
        return MultiDiscrete([self.grid_size] * 2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return [moviment.value for moviment in Actions]
