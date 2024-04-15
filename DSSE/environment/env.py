from random import random, randint
import functools
from copy import copy
import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete, Box, Tuple
from pettingzoo.utils.env import ParallelEnv
from .simulation.dynamic_probability import ProbabilityMatrix
from .constants import RED, GREEN, Actions, Reward
from .pygame_interface import PygameInterface
from .drone import DroneData
from .person import Person
from .env_base import DroneSwarmSearchBase
from DSSE.environment.time_step import calculate_simulation_time_step


class DroneSwarmSearch(DroneSwarmSearchBase):
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
        search_and_find=100_000,
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
        pre_render_time=0,
    ):
        if person_amount <= 0:
            raise ValueError("The number of persons must be greater than 0.")
        self.person_amount = person_amount
        
        super().__init__(
            grid_size,
            render_mode,
            render_grid,
            render_gradient,
            vector,
            dispersion_inc,
            dispersion_start,
            timestep_limit,
            drone_amount,
            drone_speed,
            probability_of_detection,
            pre_render_time,
        )

        # Person initialization
        self.person_initial_position = person_initial_position
        self.persons_list = self.create_list_person()

        # Initializing render
        self.rewards_sum = {a: 0 for a in self.possible_agents}
        self.rewards_sum["total"] = 0

        # Define the action and observation spaces for compatibility with RL libraries
        self.action_spaces = {
            agent: self.action_space(agent) for agent in self.possible_agents
        }
        self.observation_spaces = {
            agent: self.observation_space(agent) for agent in self.possible_agents
        }

    def create_list_person(self) -> list[Person]:
        list_person = []
        position = self.create_random_positions_person(
            self.person_initial_position, self.person_amount
        )

        for i in range(self.person_amount):
            list_person.append(
                Person(
                    initial_position=position[i],
                    grid_size=self.grid_size,
                    probability_of_detection=self.probability_of_detection,
                )
            )
            list_person[i].calculate_movement_vector(self.vector)

            if not self.is_valid_position(list_person[i].initial_position):
                raise ValueError("Person initial position is out of the matrix")

        return list_person

    def create_random_positions_person(
        self, central_position: tuple[int, int], amount: int, max_distance: int = 2
    ) -> list[tuple[int, int]]:
        if not self.is_valid_position(central_position):
            raise ValueError("Central position is out of the matrix")

        max_distance_range = (max_distance * 2 + 1) ** 2

        if amount > max_distance_range:
            raise ValueError(
                "There are more persons than grid spots. Reduce number of persons or increase grid size."
            )

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
                raise ValueError(
                    "You are trying to place the drone in a invalid position"
                )

        if individual_pods is not None:
            if not self.is_valid_pod(individual_pods):
                raise ValueError(
                    "The pod scale is invalid. It must be between 0 and 1. and must be a number."
                )

        # Person initialization
        self.persons_list = self.create_list_person()
        # reset target position
        for person in self.persons_list:
            person.reset_position()
            person.update_time_step_relation(self.time_step_relation, self.cell_size)

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
            [
                self.persons_list[0].initial_position[1],
                self.persons_list[0].initial_position[0],
            ],
            self.grid_size,
        )

        self.probability_matrix.update_time_step_relation(
            self.time_step_relation, self.cell_size
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

    def is_valid_pod(self, individual_pods: list[int]) -> bool:
        if len(individual_pods) != len(self.persons_list):
            return False
        for pod in individual_pods:
            if not isinstance(pod, (int, float)) or pod < 0 or pod > 1:
                return False
        return True

    def pre_search_simulate(self):
        for _ in range(self.pre_render_steps):
            self.create_observations()
            if self.render_mode == "human":
                self.render()

    def create_observations(self):
        observations = {}

        for person in self.persons_list:
            movement_map = self.build_movement_matrix(person)
            person.step(movement_map)

        self.probability_matrix.step()

        probability_matrix = self.probability_matrix.get_matrix()
        for agent in self.agents:
            observation = (
                (self.agents_positions[agent][0], self.agents_positions[agent][1]),
                probability_matrix,
            )
            observations[agent] = observation

        return observations

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
                drone_found_person = (
                    human.x == drone_x and human.y == drone_y and is_searching
                )
                if drone_found_person:
                    human_id = i
                    break

            random_value = random()
            if (
                drone_found_person
                and random_value <= self.persons_list[human_id].get_pod()
            ):
                del self.persons_list[human_id]
                time_reward_corrected = self.reward_scheme.search_and_find * (
                    1 - self.timestep / self.timestep_limit
                )
                rewards[agent] = (
                    self.reward_scheme.search_and_find + time_reward_corrected
                )

                if len(self.persons_list) == 0:
                    person_found = True
                    for agent in self.agents:
                        terminations[agent] = True
                        truncations[agent] = True
            elif is_searching:
                prob_matrix = self.probability_matrix.get_matrix()
                rewards[agent] = (
                    prob_matrix[drone_y][drone_x] * 10000
                    if prob_matrix[drone_y][drone_x] * 100 > 1
                    else -100
                )

            # Check truncation conditions (overwrites termination conditions)
            if self.timestep >= self.timestep_limit:
                # TODO: Check if is really necessary to add rewards_sum into this reward
                rewards[agent] = self.reward_scheme.exceed_timestep 
                if self.rewards_sum[agent] > 0:
                    rewards[agent] += self.rewards_sum[agent] // 2
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

    def render_step(self, terminal, person_found):
        if self.render_mode == "human":
            if terminal:
                if person_found:
                    self.pygame_renderer.render_episode_end_screen(
                        f"The target was found in {self.timestep} moves", GREEN
                    )
                else:
                    self.pygame_renderer.render_episode_end_screen(
                        "The target was not found.", RED
                    )
            else:
                self.render()

    def build_movement_matrix(self, person: Person) -> np.ndarray:
        """
        Builds and outputs a 3x3 matrix from the probabality matrix to use in the person movement function.
        """
        # Boundaries for the 3x3 movement matrix.
        left_x = max(person.x - 1, 0)
        right_x = min(person.x + 2, self.grid_size)
        left_y = max(person.y - 1, 0)
        right_y = min(person.y + 2, self.grid_size)

        probability_matrix = self.probability_matrix.get_matrix()
        movement_map = probability_matrix[left_y:right_y, left_x:right_x].copy()

        # Pad the matrix
        if person.x == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=1)
        elif person.x == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=1)

        if person.y == 0:
            movement_map = np.insert(movement_map, 0, 0, axis=0)
        elif person.y == self.grid_size - 1:
            movement_map = np.insert(movement_map, 2, 0, axis=0)

        return movement_map

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
