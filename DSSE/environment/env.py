from random import random, randint
import functools
from copy import copy
import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete, Box, Tuple
from pettingzoo.utils.env import ParallelEnv
from .generator.dynamic_probability import ProbabilityMatrix
from .constants import RED, GREEN, Actions, Reward
from .pygame_interface import PygameInterface
from .drone import DroneData
from .person import Person




class DroneSwarmSearch(ParallelEnv):
    """
    PettingZoo based environment for SAR missions using drones.
    """
    possible_actions = {action for action in Actions}
    metadata = {
        "name": "DroneSwarmSearchV0",
    }

    reward_scheme = Reward(
            default=1, 
            leave_grid=-100_000, 
            exceed_timestep=-100_000, 
            drones_collision=-100_000, 
            search_cell=1,
            search_and_find=100_000
        )

    def __init__(
        self,
        grid_size=7,
        render_mode="ansi",
        render_grid=False,
        render_gradient=True,
        vector=(-0.5, -0.5),
        dispersion_inc=0.1,
        dispersion_start=0.5,
        timestep_limit=100,
        person_amount=1,
        person_initial_position=(0, 0),
        drone_amount=1,
        drone_speed=10,
        probability_of_detection=0.9,
        pre_render_time = 0,
    ):
        self.cell_size = 130  # in meters
        self.grid_size = grid_size
        self._was_reset = False
        self.pre_render_steps = round((pre_render_time * 60) / (self.calculate_simulation_time_step(drone_speed, self.cell_size)))
        self.person_amount = person_amount
        self.person_initial_position = person_initial_position
        self.probability_of_detection = probability_of_detection

        print(f"Pre render time: {pre_render_time} minutes")
        print(f"Pre render steps: {self.pre_render_steps}")
        
        if self.probability_of_detection < 0 or self.probability_of_detection > 1:
            return ValueError("Probability of detection must be between 0 and 1")

        self.drone = DroneData(
            amount=drone_amount,
            speed=drone_speed,
        )

        # Error Checking
        if self.drone.amount > self.grid_size * self.grid_size:
            raise ValueError(
                "There are more drones than grid spots. Reduce number of drones or increase grid size."
            )

        if render_mode != "ansi" and render_mode != "human":
            raise ValueError("Render mode not recognized")

        if person_amount <= 0:
            raise ValueError("The number of persons must be greater than 0.")

        self.timestep = None
        self.timestep_limit = timestep_limit
        self.time_step_relation = self.calculate_simulation_time_step(
            self.drone.speed,
            self.cell_size
        )
        self.dispersion_inc = dispersion_inc
        self.dispersion_start = dispersion_start
        self.vector = vector
        self.possible_agents = []
        self.agents_positions = {}
        self.render_mode_matrix = None

        for i in range(self.drone.amount):
            self.possible_agents.append("drone" + str(i))
            self.agents_positions["drone" + str(i)] = (None, None)

        self.render_mode = render_mode
        self.probability_matrix = None

        # Person initialization
        self.persons_list = self.create_list_person()

        # Initializing render
        self.pygame_renderer = PygameInterface(self.grid_size, render_gradient, render_grid)
        self.rewards_sum = {a: 0 for a in self.possible_agents}
        self.rewards_sum["total"] = 0

        # Define the action and observation spaces for compatibility with RL libraries 
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}

    def create_random_positions_person(self, central_position: tuple[int, int], amount: int, max_distance: int = 2) -> list[tuple[int, int]]:
        if not self.is_valid_position(central_position):
            raise ValueError("Central position is out of the matrix")

        max_distance_range = (max_distance * 2 + 1)**2

        if amount > max_distance_range:
            raise ValueError("There are more persons than grid spots. Reduce number of persons or increase grid size.")

        unique_random_positions = {central_position}
        while len(unique_random_positions) < amount:
            dx = randint(-max_distance, max_distance)
            dy = randint(-max_distance, max_distance)

            # Checking to avoid including the central position or duplicate positions.
            if (dx, dy) != (0, 0):
                new_position = (central_position[0] + dx, central_position[1] + dy)
                if self.is_valid_position(new_position):
                    unique_random_positions.add(new_position)

        return list(unique_random_positions)

    def create_list_person(self) -> list[Person]:
        list_person = []
        self.position = self.create_random_positions_person(
            self.person_initial_position, self.person_amount
        )

        for i in range(self.person_amount):
            list_person.append(
                Person(
                    initial_position=self.position[i],
                    grid_size=self.grid_size,
                    probability_of_detection=self.probability_of_detection,
                )
            )
            list_person[i].calculate_movement_vector(self.vector)
            list_person[i].update_time_step_relation(
                self.time_step_relation, self.cell_size
            )

            if not self.is_valid_position(list_person[i].initial_position):
                raise ValueError("Person initial position is out of the matrix")

        return list_person

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
        self.pygame_renderer.render_entities(self.persons_list)
        self.pygame_renderer.render_entities(self.agents_positions.values())
        self.pygame_renderer.refresh_screen()

    def reset(
            self,
            seed=None,
            return_info=False,
            options=None,
    ):
        vector = options.get("vector") if options else None
        drones_positions = options.get("drones_positions") if options else None
        individual_pods = options.get("individual_pods") if options else None
        self._was_reset = True

        if drones_positions is not None:
            if not self.is_valid_position_drones(drones_positions):
                raise ValueError("You are trying to place the drone in a invalid position")
        
        if individual_pods is not None:
            if not self.is_valid_pod(individual_pods):
                raise ValueError("The pod scale is invalid. It must be between 0 and 1. and must be a number.")

        
        # Person initialization
        self.persons_list = self.create_list_person()

        # reset target position
        for person in self.persons_list:
            person.reset_position()
            person.reset_time_step_counter()
        
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.vector = vector if vector else self.vector
        self.rewards_sum = {a: 0 for a in self.agents}
        self.rewards_sum["total"] = 0
        self.probability_matrix = ProbabilityMatrix(
            40,
            self.dispersion_start,
            self.dispersion_inc,
            self.vector,
            [self.persons_list[0].initial_position[1], self.persons_list[0].initial_position[0]],
            self.grid_size,
        )
        
        if drones_positions is None:
            self.default_drones_positions()
        else:
            self.required_drone_positions(drones_positions)
        
        if individual_pods is not None:
            for i, pod in enumerate(individual_pods):
                self.persons_list[i].set_pod(pod)

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

    def is_valid_pod(self, individual_pods: list[int]) -> bool:
        if len(individual_pods) != len(self.persons_list):
            return False
        for pod in individual_pods:
            if type(pod) == str:
                return False
            if pod < 0 or pod > 1:
                raise False
        return True
    
    def pre_search_simulate(self):
        for _ in range(self.pre_render_steps):
            self.create_observations()
            if self.render_mode == "human":
                self.render()

    def create_observations(self):
        observations = {}

        self.probability_matrix.step(self.drone.speed)

        if len(self.persons_list) > 0:
            for person in self.persons_list:
                if person.reached_time_step():
                    movement_map = self.build_movement_matrix()
                    person.update_position(self.drone.speed, movement_map)

        probability_matrix = self.probability_matrix.get_matrix()
        for agent in self.agents:
            observation = (
                (self.agents_positions[agent][0], self.agents_positions[agent][1]),
                probability_matrix,
            )
            observations[agent] = observation

        return observations

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

    def step(self, actions):
        """
        Returns a tuple with (observations, rewards, terminations, truncations, infos)
        """
        if not self._was_reset:
            raise ValueError("Please reset the env before interacting with it")

        terminations = {a: False for a in self.agents}
        rewards = {a: self.reward_scheme.default for a in self.agents}
        truncations = {a: False for a in self.agents}
        person_found = False

        for agent in self.agents:
            if agent not in actions:
                raise ValueError("Missing action for " + agent)

            drone_action = actions[agent]
            if drone_action not in self.action_space(agent):
                raise ValueError("Invalid action for " + agent)

            drone_x, drone_y = self.agents_positions[agent]
            is_searching = drone_action == Actions.SEARCH.value

            if drone_action != Actions.SEARCH.value:
                new_position = self.move_drone((drone_x, drone_y), drone_action)
                if not self.is_valid_position(new_position):
                    rewards[agent] = self.reward_scheme.leave_grid
                else:
                    self.agents_positions[agent] = new_position
                    rewards[agent] = self.reward_scheme.default

            human_id = 0
            for i, human in enumerate(self.persons_list):
                drone_found_person = human.x == drone_x and human.y == drone_y and is_searching
                if drone_found_person:
                    human_id = i
                    break

            random_value = random()
            if drone_found_person and random_value <= self.persons_list[human_id].get_pod():
                del self.persons_list[human_id]
                time_reward_corrected = self.reward_scheme.search_and_find * (1 - self.timestep / self.timestep_limit)
                rewards[agent] = self.reward_scheme.search_and_find + time_reward_corrected

                if len(self.persons_list) == 0:
                    person_found = True
                    for agent in self.agents:
                        terminations[agent] = True
                        truncations[agent] = True
            elif is_searching:
                prob_matrix = self.probability_matrix.get_matrix()
                rewards[agent] = prob_matrix[drone_y][drone_x] * 10000 if prob_matrix[drone_y][drone_x] * 100 > 1 else -100

            # Check truncation conditions (overwrites termination conditions)
            # TODO: Think, should this be >= ??
            if self.timestep > self.timestep_limit:
                rewards[agent] = self.rewards_sum[agent] * -1 + self.reward_scheme.exceed_timestep
                truncations[agent] = True
                terminations[agent] = True

            self.rewards_sum[agent] += rewards[agent]

        self.timestep += 1
        # Get dummy infos
        infos = {drone: {"Found": person_found} for drone in self.agents}

        # CHECK COLISION - Drone
        self.compute_drone_collision(terminations, rewards, truncations)
        # TODO: Check real usage of this, gives error when using w/ RL libs
        # rewards["total_reward"] = sum(rewards.values())
        # self.rewards_sum["total"] += rewards["total_reward"]

        self.render_step(any(terminations.values()), person_found)        

        # Get observations
        observations = self.create_observations()
        # If terminted, reset the agents (pettingzoo parallel env requirement)
        if any(terminations.values()) or any(truncations.values()):
            self.agents = []
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
                    rewards[drone_1_id] = self.reward_scheme.drones_collision

    def render_step(self, terminal, person_found):
        if self.render_mode == "human":
            if terminal:
                if person_found:
                    self.pygame_renderer.render_episode_end_screen(f"The target was found in {self.timestep} moves", GREEN)
                else:
                    self.pygame_renderer.render_episode_end_screen("The target was not found.", RED)
            else:
                self.render()


    def build_movement_matrix(self) -> np.array:
        """
        Builds and outputs a 3x3 matrix from the probabality matrix to use in the person movement function.
        """

        # Boundaries for the 3x3 movement matrix.
        left_x = max(self.persons_list[0].x - 1, 0)
        right_x = min(self.persons_list[0].x + 2, self.grid_size)
        left_y = max(self.persons_list[0].y - 1, 0)
        right_y = min(self.persons_list[0].y + 2, self.grid_size)

        probability_matrix = self.probability_matrix.get_matrix()
        movement_map = probability_matrix[left_y:right_y, left_x:right_x]

        # Pad the matrix
        if self.persons_list[0].x == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=1)
        elif self.persons_list[0].x == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=1)

        if self.persons_list[0].y == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=0)
        elif self.persons_list[0].y == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=0)

        return movement_map

    def get_agents(self):
        return self.possible_agents

    def get_persons(self):
        return self.persons_list
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Observation space for each agent:
        # - MultiDiscrete: (x, y) position of the agent
        # - Box: Probability matrix
        return Tuple(
            (
                MultiDiscrete([self.grid_size, self.grid_size]),
                Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.float32),
            )
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(self.possible_actions))
