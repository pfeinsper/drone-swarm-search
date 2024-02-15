import functools
from copy import copy
import time
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame
from pettingzoo.utils.env import ParallelEnv
from core.environment.generator.map import update_shipwrecked_position, noise_person_movement
from core.environment.generator.dynamic_probability import ProbabilityMatrix
from core.environment.constants import RED, GREEN


class DroneSwarmSearch(ParallelEnv):
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
        # Error Checking
        if n_drones > grid_size * grid_size:
            raise Exception(
                "There are more drones than grid spots. Reduce number of drones or increase grid size."
            )

        if render_mode != "ansi" and render_mode != "human":
            raise Exception("Render mode not recognized")

        self.grid_size = grid_size
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
            self.agents_positions["drone" + str(i)] = [None, None]

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
        pygame.init()
        self.window_size = 700
        self.screen = pygame.Surface([self.window_size + 20, self.window_size + 20])
        self.renderOn = False

        self.block_size = self.window_size / self.grid_size
        self.drone_img = None
        self.person_img = None

        self.render_gradient = render_gradient
        self.render_grid = render_grid
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

    def default_drones_positions(self):
        counter_x = 0
        counter_y = 0
        for i in self.agents:
            if counter_x >= self.grid_size:
                counter_x = 0
                counter_y += 1
            self.agents_positions[i] = [counter_x, counter_y]
            counter_x += 1

    def required_drone_positions(self, drones_positions: list):
        if len(drones_positions) != len(self.possible_agents):
            raise Exception(
                "There are more or less initial positions than drones,"
                "please make sure there are the same number of initial possitions "
                "and number of drones."
            )
        for i in range(len(drones_positions)):
            x, y = drones_positions[i]
            self.agents_positions[self.possible_agents[i]] = [x, y]

    def reset(
            self,
            seed=None,
            return_info=False,
            options=None,
            drones_positions=None,
            vector=None,
    ):
        if drones_positions is not None:
            for i in drones_positions:
                if max(i) > self.grid_size:
                    raise Exception("You are trying to place the drone outside the grid")

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
        self.default_drones_positions() if drones_positions is None else self.required_drone_positions(
            drones_positions
        )
        if self.render_mode == "human":
            self.render()

        observations = self.create_observations()
        return observations

    
    def get_new_person_position(self, previous_position: int, updated_position: int):
        match updated_position:
            case 0:
                return previous_position - 1
            case 1:
                return previous_position
            case _:
                return previous_position + 1
    
       

    def create_observations(self):
        observations = {}
        self.probability_matrix.step()

        probability_matrix = self.probability_matrix.get_matrix()

        # WARNING: There is a bug in this section if the grid size is small
        leftX = self.person_x - 1 if self.person_x > 0 else 0
        rightX = (
            self.person_x + 2 if self.person_x < self.grid_size - 1 else self.grid_size
        )
        leftY = self.person_y - 1 if self.person_y > 0 else 0
        rightY = (
            self.person_y + 2 if self.person_y < self.grid_size - 1 else self.grid_size
        )

        temp_map = [line[leftX:rightX] for line in probability_matrix[leftY:rightY]]

        if self.person_x == 0:
            for i in range(rightY - leftY):
                temp_map[i] = np.insert(temp_map[i], 0, 0)
        elif self.person_x == self.grid_size - 1:
            for i in range(rightY - leftY):
                temp_map[i] = np.insert(temp_map[i], 2, 0)
        if self.person_y == 0:
            temp_map = np.insert(temp_map, 0, np.array([0, 0, 0]), axis=0)
        elif self.person_y == self.grid_size - 1:
            temp_map = np.insert(temp_map, 2, np.array([0, 0, 0]), axis=0)

        prev_person_x, prev_person_y = self.person_x, self.person_y

        # self.map, self.person_x, self.person_y = generate_map(np.array(temp_map))
        # self.person_x = self.get_new_person_position(prev_person_x, self.person_x)
        # self.person_y = self.get_new_person_position(prev_person_y, self.person_y)
        movement = update_shipwrecked_position(np.array(temp_map))
        actual_movement = noise_person_movement(movement, self.vector, epsilon=0.8)
        
        self.person_x = self.safe_1d_position_update(prev_person_x, actual_movement[0])
        self.person_y = self.safe_1d_position_update(prev_person_y, actual_movement[1])

        for i in self.possible_agents:
            observation = (
                (self.agents_positions[i][0], self.agents_positions[i][1]),
                self.probability_matrix.get_matrix(),
            )
            observations[i] = {"observation": observation}

        self.render_probability_matrix(self.render_mode_matrix)
        return observations


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
 

    def move(self, position, action):
        """Returns a tuple with (is_terminal, new_position, reward)"""
        match action:
            case 0:  # LEFT
                new_position = (position[0] - 1, position[1])
            case 1:  # RIGHT
                new_position = (position[0] + 1, position[1])
            case 2:  # UP
                new_position = (position[0], position[1] - 1)
            case 3:  # DOWN
                new_position = (position[0], position[1] + 1)

        if (
                new_position[0] < 0
                or new_position[0] >= self.grid_size
                or new_position[1] < 0
                or new_position[1] >= self.grid_size
        ):
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
                raise Exception("Missing action for " + i)

            drone_action = actions[i]
            drone_x = self.agents_positions[i][0]
            drone_y = self.agents_positions[i][1]
            isSearching = False

            if drone_action in {0, 1, 2, 3}:
                is_terminal, new_position, reward = self.move(
                    (drone_x, drone_y), drone_action
                )
                self.agents_positions[i] = new_position
                rewards[i] = reward
                terminations[i] = is_terminal
                truncations[i] = is_terminal
            elif drone_action == 4:  # search
                isSearching = True

            if drone_x == self.person_x and drone_y == self.person_y and isSearching:
                rewards[i] = self.reward_scheme["search_and_find"] + self.reward_scheme["search_and_find"] * (1 - self.timestep / self.timestep_limit)
                terminations = {a: True for a in self.agents}
                truncations = {a: True for a in self.agents}
                person_found = True
            elif isSearching:
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
        rewards["total_reward"] = sum([e for e in rewards.values()])
        self.rewards_sum["total"] += rewards["total_reward"]

        if self.render_mode == "human":
            if True in terminations.values():
                if person_found:
                    self.render_episode_end_screen(f"The target was found in {self.timestep} moves", GREEN)
                else:
                    self.render_episode_end_screen("The target was not found.", RED)
            else:
                self.render()

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

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())

            self.drone_img = pygame.image.load(
                "core/environment/imgs/drone.png"
            ).convert()
            self.drone_img = pygame.transform.scale(
                self.drone_img, (self.block_size, self.block_size)
            )

            self.person_img = pygame.image.load(
                "core/environment/imgs/person-swimming.png"
            ).convert()
            self.person_img = pygame.transform.scale(
                self.person_img, (self.block_size, self.block_size)
            )

            self.renderOn = True

    def render(self):
        """Renders the environment."""
        self.enable_render(self.render_mode)
        self.draw()
        if self.render_mode == "human":
            pygame.display.flip()
            return

    def draw(self):
        time.sleep(0.2)
        self.screen.fill((0, 0, 0))
        drone_positions = [[x, y] for x, y in self.agents_positions.values()]
        person_position = [self.person_x, self.person_y]
        matrix = self.probability_matrix.get_matrix()

        max_matrix = matrix.max()
        counter_x = 0
        for x in np.arange(10, self.window_size + 10, self.block_size):
            counter_y = 0
            for y in np.arange(10, self.window_size + 10, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                prob = matrix[counter_y][counter_x]
                
                #TODO: Arrumar esse warning
                normalizedProb = prob / max_matrix

                if self.render_gradient:
                    if prob == 0:
                        r, g = 255, 0
                    elif prob > 0.99:
                        r, g = 0, 255
                    else:
                        g = normalizedProb
                        r = 1 - normalizedProb
                        g = g * 255
                        r = r * 255
                        max_color = max(r, g)
                        g = (g) * (255) / (max_color)
                        r = (r) * (255) / (max_color)
                else:
                    r, g = (
                        (0, 255)
                        if normalizedProb >= 0.75
                        else (255, 255)
                        if normalizedProb >= 0.25
                        else (255, 0)
                    )

                pygame.draw.rect(self.screen, (r, g, 0), rect)
                if self.render_grid:
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)

                if [counter_x, counter_y] in drone_positions:
                    self.screen.blit(self.drone_img, rect)
                elif [counter_x, counter_y] == person_position:
                    self.screen.blit(self.person_img, rect)
                counter_y += 1
            counter_x += 1

    def render_episode_end_screen(self, message: str, color: tuple):
        font = pygame.font.SysFont(None, 50)
        text = font.render(message, True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2))
        self.screen.fill(color)
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        time.sleep(1)

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False

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
