from numpy import zeros
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math


class probability_matrix:
    def __init__(
        self, amplitude, spacement_x, spacement_y, vector, initial_position, size
    ):
        # amplitude of gaussian curve
        self.amplitude = amplitude
        # spacement of gaussian curve along the x axis
        self.spacement_x = spacement_x
        # spacement of gaussian curve along the y axis
        self.spacement_y = spacement_y
        # vector that determines the movement of the target ("water current")
        self.vector = vector
        # position of target, in form of list: [row, column]
        self.initial_position = initial_position
        # Gaussian map
        self.map = zeros((size, size), dtype=float)
        # Gaussian map with probabilities
        self.map_prob = zeros((size, size), dtype=float)
        # These determine the movement of the target as well
        self.x = 0
        self.y = 0
        # Parameters of the ellipse
        self.params = [1, 1, initial_position[1], initial_position[0]]
        # Increase of the area, begins at 0
        self.increase_area = 0

    def calc_prob(self, point):
        x = ((point[0] - self.initial_position[1]) ** 2) / (2 * (self.spacement_x**2))
        y = ((point[1] - self.initial_position[0]) ** 2) / (2 * (self.spacement_y**2))

        probability = self.amplitude * np.exp(-(x + y))

        return probability

    def diffuse_probability(self, map):
        entire_cells = []
        h = self.params[2]
        k = self.params[3]
        a = self.params[0]
        b = self.params[1]
        for row in range(len(map)):
            for column in range(0, len(map[row])):
                p1 = (math.pow((column - (h - 0.5)), 2) / math.pow(a, 2)) + (
                    math.pow((row - k), 2) / math.pow(b, 2)
                )
                p2 = (math.pow((column - (h + 0.5)), 2) / math.pow(a, 2)) + (
                    math.pow((row - k), 2) / math.pow(b, 2)
                )
                p3 = (math.pow((column - h), 2) / math.pow(a, 2)) + (
                    math.pow((row - (k - 0.5)), 2) / math.pow(b, 2)
                )
                p4 = (math.pow((column - h), 2) / math.pow(a, 2)) + (
                    math.pow((row - (k + 0.5)), 2) / math.pow(b, 2)
                )
                if p1 <= 1 and p2 <= 1 and p3 <= 1 and p4 <= 1:
                    entire_cells.append((column, row))
        map_copy = zeros((len(map), len(map[0])), dtype=float)
        for cell in entire_cells:
            map_copy[cell[1]][cell[0]] = self.calc_prob(cell)
        self.map_prob = map_copy / map_copy.sum()
        self.map = copy.deepcopy(map_copy)

    def dynamic_probability(self):
        if abs(self.x) >= 1:
            self.x = 0
        if abs(self.y) >= 1:
            self.y = 0
        self.x += self.vector[0]
        self.y += self.vector[1]
        map_copy = zeros((len(self.map), len(self.map[0])), dtype=float)
        if self.initial_position[0] + int(self.y) < len(
            self.map
        ) and self.initial_position[1] + int(self.x) < len(self.map[0]):
            if (
                self.initial_position[0] + int(self.y) >= 0
                and self.initial_position[1] + int(self.x) >= 0
            ):
                map_copy[self.initial_position[0] + int(self.y)][
                    self.initial_position[1] + int(self.x)
                ] += self.map[self.initial_position[0]][self.initial_position[1]]
                self.initial_position[0] += int(self.y)
                self.initial_position[1] += int(self.x)
        self.diffuse_probability(map_copy)

    def step(self):
        self.dynamic_probability()
        self.params = [
            1 + self.increase_area,
            1 + self.increase_area,
            self.initial_position[1],
            self.initial_position[0],
        ]
        self.increase_area += 0.5
        return self.map_prob

    def render(self):
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25, left=0.25)
        angle = np.linspace(0, 2 * np.pi, 150)

        radius_x = self.params[0]
        radius_y = self.params[1]

        x = radius_x * np.cos(angle) + self.params[2]
        y = radius_y * np.sin(angle) + self.params[3]

        ax.plot(x, y)
        matshow = ax.matshow(self.map_prob)
        plt.show()

    def render_episode(self, animation_matrix, cirumference_params):
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25, left=0.25)
        angle = np.linspace(0, 2 * np.pi, 150)

        radius_x = cirumference_params[0][0]
        radius_y = cirumference_params[0][1]

        x = radius_x * np.cos(angle) + cirumference_params[0][2]
        y = radius_y * np.sin(angle) + cirumference_params[0][3]

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider = Slider(
            ax_slider,
            label="Frame",
            valmin=0,
            valmax=len(animation_matrix) - 1,
            valinit=0,
            valstep=1,
        )
        ax.plot(x, y)
        matshow = ax.matshow(animation_matrix[0])

        def update(_):
            # print(animation_matrix[slider.val].max())
            radius_x = cirumference_params[slider.val][0]
            radius_y = cirumference_params[slider.val][1]

            x = radius_x * np.cos(angle) + cirumference_params[slider.val][2]
            y = radius_y * np.sin(angle) + cirumference_params[slider.val][3]
            ax.cla()
            ax.plot(x, y)
            matshow = ax.matshow(animation_matrix[slider.val])
            fig.canvas.draw_idle()

        slider.on_changed(update)
        # matshow.set_data(animation_matrix[0])
        plt.show()

    def get_matrix(self):
        return self.map_prob

    def get_params(self):
        return self.params


# prob_matrix = probability_matrix(40, 3, 3, (-0.5, 0.5), [0, 19], 20)
# list_matrix = []
# list_params = []
# for i in range(40):
#     prob_matrix.step()
#     list_matrix.append(prob_matrix.get_matrix())
#     list_params.append(prob_matrix.get_params())

# prob_matrix.render_episode(list_matrix, list_params)
