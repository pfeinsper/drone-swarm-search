import numpy as np
from numba import njit


class ProbabilityMatrix:
    def __init__(
        self, amplitude, spacement_start, spacement_inc, vector, initial_position, size
    ):
        """
        Probability matrix that represents the probability of finding a target in a certain position
            :param amplitude: Amplitude of the gaussian function
            :param spacement_start: Initial spacement of the gaussian function
            :param spacement_inc: Increment of the spacement of the gaussian function
            :param vector: Vector that determines the movement of the matrix
            :param initial_position: Initial position of the matrix
            :param size: Size of the matrix
        """
        self.amplitude = amplitude
        self.supposed_position = initial_position
        self.map = np.zeros((size, size), dtype=np.float32)
        self.map_prob = np.zeros((size, size), dtype=np.float32)
        self.vector = vector

        # These determine the movement of the target as well
        self.inc_x = 0
        self.inc_y = 0
        # These determine the shape of the gaussian function
        self.spacement_inc = spacement_inc
        self.spacement = spacement_start

    def step(self, drone_speed):
        self.update_position(drone_speed)
        self.diffuse_probability()

    def update_position(self, drone_speed):
        if abs(self.inc_x) >= 1:
            self.inc_x = 0
        if abs(self.inc_y) >= 1:
            self.inc_y = 0

        self.inc_x += self.vector[0] / drone_speed
        self.inc_y += self.vector[1] / drone_speed

        # On row, column notation (y, x)
        new_position = (
            self.supposed_position[0] + int(self.inc_y),
            self.supposed_position[1] + int(self.inc_x),
        )

        if self.is_valid_position(new_position):
            self.supposed_position = new_position

    def is_valid_position(self, position: tuple[int]) -> bool:
        rows, columns = self.map.shape
        is_valid_y = position[0] >= 0 and position[0] < rows
        is_valid_x = position[1] >= 0 and position[1] < columns
        return is_valid_x and is_valid_y

    def diffuse_probability(self):
        map_copy = self.calc_all_probs(
            self.amplitude,
            self.supposed_position,
            self.spacement,
            self.map.shape,
        )

        map_copy_sum = map_copy.sum()
        if map_copy_sum == 0:
            self.map_prob = map_copy
        else:
            self.map_prob = map_copy / map_copy_sum

        self.map = map_copy
        self.spacement += self.spacement_inc

    def calc_all_probs_numpy(self):
        rows, columns = self.map.shape
        x = np.arange(0, columns)
        y = np.arange(0, rows)
        x, y = np.meshgrid(x, y)
        x_prob = ((x - self.supposed_position[1]) ** 2) / (2 * (self.spacement_x**2))
        y_prob = ((y - self.supposed_position[0]) ** 2) / (2 * (self.spacement_y**2))
        probabilities = self.amplitude * np.exp(-(x_prob + y_prob))
        return probabilities

    @staticmethod
    @njit(cache=True, fastmath=True)
    def calc_all_probs(
        amplitude: int, supposed_position: tuple, spacement: float, shape: tuple
    ) -> np.array:
        probabilities = np.zeros(shape, dtype=np.float32)
        # On row, column notation (y, x)
        x0 = supposed_position[1]
        y0 = supposed_position[0]
        for row in range(shape[0]):
            for column in range(shape[1]):
                x = ((column - x0) ** 2) / (2 * (spacement**2))
                y = ((row - y0) ** 2) / (2 * (spacement**2))
                probabilities[row][column] = amplitude * np.exp(-(x + y))
        return probabilities

    def get_matrix(self):
        return self.map_prob
