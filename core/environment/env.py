import functools
from copy import copy
import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo.utils.env import ParallelEnv
from core.environment.generator.map import update_shipwrecked_position, noise_person_movement
from core.environment.generator.dynamic_probability import ProbabilityMatrix
from core.environment.constants import RED, GREEN
from core.environment.pygame_interface import PygameInterface


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
            n_drones=1,
            vector=(-0.5, -0.5),
            person_initial_position=[0, 0],
            disperse_constant=10,
            timestep_limit=100,
    ):
        self.grid_size = grid_size

        # Error Checking
        if n_drones > grid_size * grid_size:
            raise ValueError(
                "There are more drones than grid spots. Reduce number of drones or increase grid size."
            )
        if not self.is_valid_position(person_initial_position):
            raise ValueError("Person initial position is out of the matrix")

        if render_mode != "ansi" and render_mode != "human":
            raise ValueError("Render mode not recognized")

        self.person_initial_position = person_initial_position
        self.person_y = person_initial_position[1]
        self.person_x = person_initial_position[0]
        self.timestep = None
        self.timestep_limit = timestep_limit
        self.disperse_constant = disperse_constant
        self.vector = vector
        self.possible_agents = []
        self.agents_positions = {}
        self.render_mode_matrix = None
        for i in range(n_drones):
            self.possible_agents.append("drone" + str(i))
            self.agents_positions["drone" + str(i)] = (None, None)

        self.render_mode = render_mode
        self.probability_matrix = ProbabilityMatrix(
            40,
            disperse_constant,
            disperse_constant,
            self.vector,
            [person_initial_position[1], person_initial_position[0]],
            self.grid_size,
        )

        # Initializing render
        self.pygame_renderer = PygameInterface(self.grid_size, render_gradient, render_grid)
        self.rewards_sum = {a: 0 for a in self.possible_agents}
        self.rewards_sum["total"] = 0

        # Reward Function
        self.reward_scheme = {
            "default": 1,
            "leave_grid": -100000,
            "exceed_timestep": -100000,
            "drones_collision": -100000,
            "search_cell": 1,
            "search_and_find": 100000,
        }
    
    def is_valid_position(self, position: tuple[int]) -> bool:
        valid_x = position[0] >= 0 and position[0] < self.grid_size
        valid_y = position[1] >= 0 and position[1] < self.grid_size
        return valid_x and valid_y

    def default_drones_positions(self):
        counter_x = 0
        counter_y = 0
        for i in self.agents:
            if counter_x >= self.grid_size:
                counter_x = 0
                counter_y += 1
            self.agents_positions[i] = (counter_x, counter_y)
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
        if self.render_mode == "human":
            self.pygame_renderer.render(
                self.agents_positions.values(), 
                (self.person_x, self.person_y), 
                self.probability_matrix.get_matrix()
            )

    def reset(
            self,
            seed=None,
            return_info=False,
            options=None,
            drones_positions=None,
            vector=None,
    ):
        if drones_positions is not None:
            for position in drones_positions:
                if max(position) > self.grid_size:
                    raise ValueError("You are trying to place the drone outside the grid")

        # reset target position
        self.person_y = self.person_initial_position[1]
        self.person_x = self.person_initial_position[0]

        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.vector = vector if vector else self.vector
        self.rewards_sum = {a: 0 for a in self.agents}
        self.rewards_sum["total"] = 0
        self.probability_matrix = ProbabilityMatrix(
            40,
            self.disperse_constant,
            self.disperse_constant,
            self.vector,
            [self.person_initial_position[1], self.person_initial_position[0]],
            self.grid_size,
        )
        
        if drones_positions is None:
            self.default_drones_positions()
        else:
            self.required_drone_positions(drones_positions)
        
        if self.render_mode == "human":
            self.pygame_renderer.enable_render()
            self.pygame_renderer.render(
                self.agents_positions.values(),
                (self.person_x, self.person_y),
                self.probability_matrix.get_matrix()
            )

        observations = self.create_observations()
        return observations

    
    def create_observations(self):
        observations = {}
        self.probability_matrix.step()

        movement_map = self.build_movement_matrix()

        movement = update_shipwrecked_position(movement_map)
        actual_movement = noise_person_movement(movement, self.vector, epsilon=0.0)
        
        self.person_x = self.safe_1d_position_update(self.person_x, actual_movement[0])
        self.person_y = self.safe_1d_position_update(self.person_y, actual_movement[1])

        for i in self.possible_agents:
            observation = (
                (self.agents_positions[i][0], self.agents_positions[i][1]),
                self.probability_matrix.get_matrix(),
            )
            observations[i] = {"observation": observation}

        self.render_probability_matrix(self.render_mode_matrix)
        return observations

    def build_movement_matrix(self) -> np.array:
        """
        Builds and outputs a 3x3 matrix from the probabality matrix to use in the person movement function.
        """
        # Boundaries for the 3x3 movement matrix.
        left_x = max(self.person_x - 1, 0)
        right_x = min(self.person_x + 2, self.grid_size)
        left_y = max(self.person_y - 1, 0)
        right_y = min(self.person_y + 2, self.grid_size)

        probability_matrix = self.probability_matrix.get_matrix()
        movement_map = probability_matrix[left_y:right_y, left_x:right_x]

        # Pad the matrix
        if self.person_x == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=1)
        elif self.person_x == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=1)
        if self.person_y == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=0)
        elif self.person_y == self.grid_size - 1:
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
            case 0:  # LEFT
                new_position = (position[0] - 1, position[1])
            case 1:  # RIGHT
                new_position = (position[0] + 1, position[1])
            case 2:  # UP
                new_position = (position[0], position[1] - 1)
            case 3:  # DOWN
                new_position = (position[0], position[1] + 1)

        if not self.is_valid_position(new_position):
            return True, new_position, self.reward_scheme["leave_grid"]

        return False, new_position, self.reward_scheme["default"]

    def step(self, actions):
        """Returns a tuple with (observations, rewards, terminations, truncations, infos)"""
        terminations = {a: False for a in self.agents}
        rewards = {a: self.reward_scheme["default"] for a in self.agents}
        truncations = {a: False for a in self.agents}
        person_found = False
        for i in self.agents:
            if i not in actions:
                raise ValueError("Missing action for " + i)

            drone_action = actions[i]
            drone_x = self.agents_positions[i][0]
            drone_y = self.agents_positions[i][1]
            is_searching = drone_action == 4

            if drone_action in {0, 1, 2, 3}:
                is_terminal, new_position, reward = self.move_drone(
                    (drone_x, drone_y), drone_action
                )
                self.agents_positions[i] = new_position
                rewards[i] = reward
                terminations[i] = is_terminal
                truncations[i] = is_terminal

            if drone_x == self.person_x and drone_y == self.person_y and is_searching:
                rewards[i] = self.reward_scheme["search_and_find"] + self.reward_scheme["search_and_find"] * (1 - self.timestep / self.timestep_limit)
                terminations = {a: True for a in self.agents}
                truncations = {a: True for a in self.agents}
                person_found = True
            elif is_searching:
                prob_matrix = self.probability_matrix.get_matrix()
                rewards[i] = prob_matrix[drone_y][drone_x] * 10000 if prob_matrix[drone_y][drone_x] * 100 > 1 else -100

            # Check truncation conditions (overwrites termination conditions)
            if self.timestep > self.timestep_limit:
                rewards[i] = self.rewards_sum[i] * -1 + self.reward_scheme["exceed_timestep"]
                truncations[i] = True
                terminations[i] = True

            self.rewards_sum[i] += rewards[i]

        self.timestep += 1
        # Get observations
        observations = self.create_observations()
        # Get dummy infos
        infos = {"Found": person_found}

        # CHECK COLISION - Drone
        self.compute_drone_collision(terminations, rewards, truncations)
        rewards["total_reward"] = sum(rewards.values())
        self.rewards_sum["total"] += rewards["total_reward"]

        if self.render_mode == "human":
            if any(terminations.values()):
                if person_found:
                    self.pygame_renderer.render_episode_end_screen(f"The target was found in {self.timestep} moves", GREEN)
                else:
                    self.pygame_renderer.render_episode_end_screen("The target was not found.", RED)
            else:
                self.pygame_renderer.render(self.agents_positions.values(), (self.person_x, self.person_y), self.probability_matrix.get_matrix())

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

    def render_probability_matrix(self, mode="human-terminal"):
        if mode == "human-terminal":
            grid = self.probability_matrix.get_matrix()
            print("PROBABILITY MATRIX:")

            print("----" * self.grid_size)
            for i in grid:
                string = "| "
                for e in i:
                    if e >= 10:
                        string += "{0} | ".format(e)
                    else:
                        string += "{0}  | ".format(e)
                print(string)
            print("----" * self.grid_size)
        elif mode == "human":
            self.probability_matrix.render()

    def get_agents(self):
        return self.possible_agents

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # TODO: If x and y are the observation, then this should the observation space
        return MultiDiscrete([self.grid_size] * 2)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return [0, 1, 2, 3, 4]
