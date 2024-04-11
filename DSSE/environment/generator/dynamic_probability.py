import numpy as np
from numpy.linalg import norm
from numba import njit


class ProbabilityMatrix:
    """
    Probability matrix that represents the probability of finding a target in a certain position.

    Attributes
    ----------
    amplitude : int
        Amplitude of the gaussian function
    supposed_position : tuple
        Center of the gaussian function
    map : np.array
        Matrix that represents the probability of finding a target in a certain position
    vector : tuple
        Vector that determines the movement of the matrix
    inc_x : float
        Increment of the x position of the matrix
    inc_y : float
        Increment of the y position of the matrix
    spacement_inc : float
        Increment of the spacement of the gaussian function
    spacement : float
        Spacement of the gaussian function
    time_step_counter : int
        Counter to keep track of the time steps
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
        self.vector = vector

        # These determine the movement of the target as well
        self.inc_x = 0
        self.inc_y = 0

        # These determine the shape of the gaussian function
        self.spacement_inc = spacement_inc
        self.spacement = spacement_start

        # Time step control
        self.time_step_counter = 0
        self.time_step_relation = 1

    def step(self) -> None:
        self.update_position()
        self.diffuse_probability()

    def update_position(self) -> None:
        if abs(self.inc_x) >= 1:
            self.inc_x -= 1
        if abs(self.inc_y) >= 1:
            self.inc_y -= 1

        self.inc_x += self.vector[0] / norm(self.vector)
        self.inc_y += self.vector[1] / norm(self.vector)

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

    def reached_time_step(self) -> bool:
        if self.time_step_counter >= self.time_step_relation:
            self.reset_time_step_counter()
            return True
        self.increment_time_step_counter()
        return False

    def reset_time_step_counter(self) -> None:
        self.time_step_counter = 0

    def increment_time_step_counter(self) -> None:
        self.time_step_counter += 1

    def will_move(self) -> bool:
        return self.inc_x >= 1 or self.inc_y >= 1

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
        self.time_step_relation = self.calculate_time_step(time_step, self.vector, cell_size)

    def calculate_time_step(
            self,
            time_step: float,
            speed: tuple[float],
            cell_size: float
        ) -> int:
        """
        Parameters:
        ----------
        time_step: float
            Time step in seconds
        person_speed: tuple[float]
            Speed of the person in the water in m/s (x and y components)
        cell_size: float
            Size of the cells in meters

        Returns:
        -------
        int
            Time step realtion in number of iterations
        """
        speed_magnitude, _ = self.calculate_vector_magnitude_and_direction(speed)
        return int(cell_size / speed_magnitude / time_step)

    def calculate_vector_magnitude_and_direction(self, vector: tuple[float]) -> tuple[float, tuple[int]]:
        """
        Args:
        vector: tuple[float]
            Vector with x and y components

        Returns:
        tuple[float]
            Magnitude and direction of the vector
        Magnitude is in m/s
        Direction is in x and y components, a unit vector
        """
        magnitude = np.linalg.norm(vector)
        angle = np.arctan2(vector[1], vector[0])

        # Calculate cosine and sine values
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)

        # Determine direction based on the sign of cosine and sine
        x_direction = np.sign(cos_val) if abs(cos_val) > 0.0001 else 0
        y_direction = np.sign(sin_val) if abs(sin_val) > 0.0001 else 0
        return (magnitude, (x_direction, y_direction))

    def get_matrix(self):
        return self.map_prob
