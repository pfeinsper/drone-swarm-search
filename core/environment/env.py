import functools
import random
from copy import copy, deepcopy
import os
import numpy as np
from gymnasium.spaces import MultiDiscrete, Discrete
import pygame
from pettingzoo.utils.env import ParallelEnv
import sys
import time
from core.environment.generator.map import generate_map, generate_matrix
from core.environment.generator.dynamic_probability import probability_matrix


class DroneSwarmSearch(ParallelEnv):
    def __init__(self, grid_size=7, 
                 render_mode="ansi", 
                 render_grid = False,
                 render_gradient = True,
                 n_drones=1, 
                 vector = (-0.5, -0.5), 
                 person_initial_position = [0, 0],
                 disperse_constant = 10):
        
        #Error Checking 
        if n_drones > grid_size*grid_size:
            raise Exception("There are more drones than grid spots. Reduce number of drones or increase grid size.")
        if render_mode != "ansi" and render_mode != "human":
            raise Exception("Render mode not recognized")
        
        self.grid_size = grid_size
        self.person_y = None
        self.person_x = None
        self.timestep = None
        self.vector = vector
        self.possible_agents = []
        self.agents_positions = {}
        self.render_mode_matrix = None
        for i in range(n_drones):
            self.possible_agents.append("drone" + str(i))
            self.agents_positions["drone" + str(i)] = [None, None]

        self.render_mode = render_mode
        self.probability_matrix = probability_matrix(
            40, disperse_constant, disperse_constant, self.vector, person_initial_position, self.grid_size
        )
        self.map, self.person_x, self.person_y = generate_map(
            self.probability_matrix.get_matrix()
        )

        #Initializing render
        pygame.init()
        self.window_size = 700
        self.screen = pygame.Surface([self.window_size + 20, self.window_size + 20])
        self.renderOn = False

        self.block_size = self.window_size / self.grid_size
        self.drone_img = None
        self.person_img = None

        self.render_gradient = render_gradient
        self.render_grid = render_grid


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
        for i in range(len(drones_positions)):
            x, y = drones_positions[i]
            self.agents_positions[self.possible_agents[i]] = [x, y]

    def reset(self, seed=None, return_info=False, options=None, drones_positions=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.default_drones_positions() if drones_positions is None else self.required_drone_positions(
            drones_positions
        )
        if self.render_mode == "human":
            self.render()

        observations = self.create_observations()
        return observations

    def create_observations(self):
        observations = {}
        self.probability_matrix.step()
        self.map, self.person_x, self.person_y = generate_map(
            self.probability_matrix.get_matrix()
        )

        for i in self.possible_agents:
            observation = (
                (self.agents_positions[i][0], self.agents_positions[i][1]),
                self.probability_matrix.get_matrix(),
            )

            observations[i] = {"observation": observation}
        self.render_probability_matrix(self.render_mode_matrix)
        return observations

    def step(self, actions):
        terminations = {a: False for a in self.agents}
        rewards = {a: -1 for a in self.agents}
        truncations = {a: False for a in self.agents}

        for i in self.agents:
            if not i in actions:
                raise Exception("Missing action for " + i)
            
            drone_action = actions[i]
            
            drone_x = self.agents_positions[i][0]
            drone_y = self.agents_positions[i][1]

            isSearching = False

            if drone_action == 0:  # LEFT
                if drone_x > 0:
                    self.agents_positions[i][0] -= 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True

            elif drone_action == 1: # RIGHT
                if drone_x < self.grid_size - 1:
                    self.agents_positions[i][0] += 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True
            elif drone_action == 2:  # UP
                if drone_y > 0:
                    self.agents_positions[i][1] -= 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True
            elif drone_action == 3:  # DOWN
                if drone_y < self.grid_size - 1:
                    self.agents_positions[i][1] += 1
                else:
                    rewards[i] = -1000
                    truncations[i] = True
                    terminations[i] = True

            elif drone_action == 4:  # search
                isSearching = True
            elif drone_action == 5:  # idle
                pass

            if drone_x == self.person_x and drone_y == self.person_y and isSearching:
                rewards = {a: 0 for a in self.agents}
                terminations = {a: True for a in self.agents}
                truncations = {a: True for a in self.agents}
                self.agents = []

            elif isSearching:
                prob_matrix = self.probability_matrix.get_matrix()
                rewards[i] = prob_matrix[drone_y][drone_x] - 100

            # Check truncation conditions (overwrites termination conditions)
            if self.timestep > 500:
                rewards[i] = -1000
                truncations[i] = True
                terminations[i] = True
                self.agents = []
        self.timestep += 1

        # Get observations
        observations = self.create_observations()

        # Get dummy infos 
        infos = {e: {} for e in self.agents}

        # CHECK COLISION
        for ki, i in self.agents_positions.items():
            for ke, e in self.agents_positions.items():
                if ki is not ke:
                    if i[0] == e[0] and i[1] == e[1]:
                        truncations[ki] = True
                        terminations[ki] = True
                        rewards[ki] = -2000
        rewards["total_reward"] = sum([e for e in rewards.values()])

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

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
        for x in np.arange(10, self.window_size+10, self.block_size):
            counter_y = 0
            for y in np.arange(10, self.window_size+10, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
                prob = matrix[counter_y][counter_x]
                normalizedProb = prob/max_matrix
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
                        g = (g) * (255)/( max_color)
                        r = (r) * (255)/( max_color)
                else:
                    r, g = (0, 255) if normalizedProb >= 0.75 else  (255, 255) if normalizedProb >= 0.25 else (255, 0)

                pygame.draw.rect(self.screen, (r,g,0) , rect)
                if self.render_grid:
                    pygame.draw.rect(self.screen, (0,0,0) , rect, 2)

                if [counter_x, counter_y] in drone_positions:
                    self.screen.blit(self.drone_img, rect)
                elif [counter_x, counter_y] == person_position:
                    self.screen.blit(self.person_img, rect)
                counter_y += 1
            counter_x += 1

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
        return [0, 1, 2, 3, 4, 5]


from pettingzoo.test import parallel_api_test  # noqa: E402

if __name__ == "__main__":
    parallel_api_test(
        DroneSwarmSearch(7, "human", n_drones=5),
        num_cycles=1000,
    )
