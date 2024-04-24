import numpy as np
from numba import njit
from .time_step import calculate_time_step


class ProbabilityMatrix:
    """
    Probability matrix that represents the probability of finding a target in a certain position.

    Attributes
    ----------
    amplitude : int
        Amplitude of the gaussian function
    supposed_position : tuple
        Center of the gaussian function
    map, map_prob : np.array
        Matrix that represents the probability of finding a target in a certain position
    movement_vector : tuple
        Vector that determines the movement of the matrix
    inc_x : float
        Increment of the x position of the matrix
    inc_y : float
        Increment of the y position of the matrix
    spacement_inc : float
        Increment of the spacement of the gaussian function
    spacement : float
        Spacement of the gaussian function
    time_step_relation : int
        Relation between the matrix time step and the drone time step
    """

    def __init__(
        self,
        amplitude: int,
        spacement_start: float,
        spacement_inc: float,
        vector: tuple[float, float],
        initial_position: tuple[int, int],
        size: int,
    ):
        """
        Probability matrix that represents the probability of finding a target in a certain position

        Parameters
        ----------
        amplitude : int
            Amplitude of the gaussian function
        spacement_start : float
            Initial spacement of the gaussian function
        spacement_inc : float
            Increment of the spacement of the gaussian function
        vector : tuple
            Vector that determines the movement of the matrix
        initial_position : tuple
            Initial position of the matrix
        size : int
            Size of the matrix (size x size)
        """

        self.amplitude = amplitude
        self.supposed_position = initial_position
        self.map = np.zeros((size, size), dtype=np.float32)
        self.map_prob = np.zeros((size, size), dtype=np.float32)
        self.movement_vector = vector

        # These determine the movement of the target as well
        self.inc_x = 0
        self.inc_y = 0

        # These determine the shape of the gaussian function
        self.spacement_inc = spacement_inc
        self.spacement = spacement_start

        # Time step control
        self.time_step_relation = 1

    def step(self) -> None:
        self.update_position()
        self.diffuse_probability()

    def update_position(self) -> None:
        self.increment_movement()

        # On row, column notation (y, x)
        new_position = (
            self.supposed_position[0] + int(self.inc_y),
            self.supposed_position[1] + int(self.inc_x),
        )

        if self.is_valid_position(new_position):
            self.supposed_position = new_position

        if abs(self.inc_x) >= 1:
            self.inc_x -= np.sign(self.inc_x)
        if abs(self.inc_y) >= 1:
            self.inc_y -= np.sign(self.inc_y)

    def is_valid_position(self, position: tuple[int]) -> bool:
        rows, columns = self.map.shape
        is_valid_y = position[0] >= 0 and position[0] < rows
        is_valid_x = position[1] >= 0 and position[1] < columns
        return is_valid_x and is_valid_y

    def diffuse_probability(self) -> None:
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

    def increment_movement(self) -> None:
        self.inc_x += self.movement_vector[0] / self.time_step_relation
        self.inc_y += self.movement_vector[1] / self.time_step_relation

    @staticmethod
    @njit(cache=True, fastmath=True)
    def calc_all_probs(
        amplitude: int,
        supposed_position: tuple,
        spacement: float,
        shape: tuple,
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

    def update_time_step_relation(self, time_step: float, cell_size: float) -> None:
        self.time_step_relation = calculate_time_step(
            time_step, self.movement_vector, cell_size
        )

    def get_matrix(self):
        return self.map_prob
