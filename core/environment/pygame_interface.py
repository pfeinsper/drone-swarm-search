import time
import pygame
import os
import numpy as np
from .constants import BLACK


class PygameInterface:
    """
    Class for rendering the grafical interface of the simulation
    """

    FPS = 3

    def __init__(
        self, grid_size: int, render_gradient: bool, render_grid: bool
    ) -> None:
        pygame.init()
        self.grid_size = grid_size
        self.render_gradient = render_gradient
        self.render_grid = render_grid
        self.window_size = 700
        self.screen = pygame.Surface([self.window_size + 20, self.window_size + 20])
        self.render_on = False
        self.probability_matrix = None

        self.block_size = self.window_size / self.grid_size
        self.drone_img = None
        self.person_img = None
        self.clock = None

    def render_map(self):
        self.draw()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        pygame.event.pump()

    def refresh_screen(self):
        pygame.display.flip()

    def enable_render(self):
        if self.render_on:
            return

        self.screen = pygame.display.set_mode(self.screen.get_size())

        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.drone_img = self.load_and_scale_image(f"{current_directory}/imgs/drone.png")
        self.person_img = self.load_and_scale_image(f"{current_directory}/imgs/person-swimming.png")

        self.clock = pygame.time.Clock()
        self.render_on = True

    def load_and_scale_image(self, image_path: str):
        loaded_img = pygame.image.load(image_path).convert()
        scaled_img = pygame.transform.scale(
            loaded_img, (self.block_size, self.block_size)
        )
        return scaled_img

    def render_drones(self, drone_positions) -> None:
        for position in drone_positions:
            rectangle = self.get_position_rectangle(position)
            self.screen.blit(self.drone_img, rectangle)

    def render_person(self, person_x: int, person_y: int) -> None:
        self.screen.blit(
            self.person_img, self.get_position_rectangle((person_x, person_y))
        )

    def get_position_rectangle(self, position: tuple[int]) -> pygame.Rect:
        x = 10 + self.block_size * position[0]
        y = 10 + self.block_size * position[1]
        return pygame.Rect(x, y, self.block_size, self.block_size)

    def draw(self):
        self.clock.tick(self.FPS)
        self.screen.fill(BLACK)

        matrix = self.probability_matrix.get_matrix()
        max_matrix = matrix.max()
        if max_matrix == 0.0:
            max_matrix = 1.0

        for counter_x, x in enumerate(
            np.arange(10, self.window_size + 10, self.block_size)
        ):
            for counter_y, y in enumerate(
                np.arange(10, self.window_size + 10, self.block_size)
            ):
                rectangle = pygame.Rect(x, y, self.block_size, self.block_size)
                prob = matrix[counter_y][counter_x]
                normalized_prob = prob / max_matrix

                computed_prob_color = self.compute_cell_color(normalized_prob)
                pygame.draw.rect(self.screen, computed_prob_color, rectangle)

                if self.render_grid:
                    pygame.draw.rect(self.screen, BLACK, rectangle, 2)

    def compute_cell_color(self, normalized_prob):
        if self.render_gradient:
            green = normalized_prob * 255
            red = (1 - normalized_prob) * 255
            max_color = max(red, green)
            green = (green * 255) / (max_color)
            red = (red * 255) / (max_color)
        else:
            red = 255
            green = 0
            if normalized_prob >= 0.75:
                red = 0
                green = 255
            elif normalized_prob >= 0.25:
                red = 255
                green = 255
        return (red, green, 0)

    def render_episode_end_screen(self, message: str, color: tuple):
        font = pygame.font.SysFont(None, 50)
        text = font.render(message, True, BLACK)
        text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2))
        self.screen.fill(color)
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        time.sleep(1)

    def close(self):
        if self.render_on:
            pygame.event.pump()
            pygame.display.quit()
            self.render_on = False
