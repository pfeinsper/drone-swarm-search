import numpy as np
from random import uniform, choice
from numpy.linalg import norm
from math import cos, sin, radians, pi, exp
from DSSE.environment.simulation.time_step import calculate_time_step


class Person:
    """
    Class that represents a shipwrecked person in the environment.
    """

    def __init__(
        self,
        index: int,
        initial_position: tuple[int, int],
        grid_size: int,
        mult: float = 1,
    ):
        """
        Class that represents a shipwrecked person in the environment.

        Parameters:
        -----------
        initial_position: tuple
            The initial position of the shipwrecked person in the environment.
        grid_size: int
            The size of the grid that represents the environment (grid_size x grid_size).
        mult: float, optional
            The multiplication factor that adjusts the probability of the person's finding.
        """
        self.index = index
        self.initial_position = initial_position
        self.x, self.y = self.initial_position
        self.inc_x, self.inc_y = 0, 0
        self.grid_size = grid_size
        self.time_step_relation = 1
        self.mult = mult
        self.movement_vector = (0.0, 0.0)
        self.angle_range = choice([(0, 45), (20, 55), (300, 360), (315, 360)])

    def step(self, movement_map: np.array) -> None:
        if self.will_move():
            movement = self.update_position(
                movement_map, dimension=movement_map.shape[0], prob_walk_weight=0.05
            )
            self.safe_position_update(movement)

    def update_position(
        self,
        movement_map: np.array,
        dimension: int = 3,
        prob_walk_weight: float = 1.0,
    ) -> tuple[int]:
        """
        Function that takes a cut of the DynamicProbability matrix,
        along with the dimension of the matrix and the probability of walking,
        that will be used to wheight the decision of walking towards the person's movement vector.

        Output:
            (movement_x, movement_y): tuple[int]
        """
        self.increment_movement()

        # On row, column notation (y, x) -> (row, column)
        movement = (
            round(self.inc_x),
            round(self.inc_y),
        )

        row_mov = np.sign(movement[1]) + 1
        col_mov = np.sign(movement[0]) + 1
        movement_map[row_mov][col_mov] += prob_walk_weight
        movement_map /= np.sum(movement_map)
        movement_index = np.random.choice(9, size=1, p=movement_map.flatten())[0]

        if abs(self.inc_x) >= 1:
            self.inc_x -= np.sign(self.inc_x)
        if abs(self.inc_y) >= 1:
            self.inc_y -= np.sign(self.inc_y)

        movement_cartesian = self.movement_to_cartesian(
            movement_index, dimension=dimension
        )
        return movement_cartesian

    def movement_to_cartesian(
        self, movement_index: int, dimension: int = 3
    ) -> tuple[int]:
        """
        The movement of the shipwrecked person on the input follows the scheme
        (for the value of line and column):
            - if 0 -> Move to the left (x - 1) or to the top (y - 1).
            - if 1 -> No movement.
            - if 2 -> Move to the right (x + 1) or to the bottom (y + 1).

        So this function converts from this matrix movement notation to cartesian,
        as the matrix that creates this indexes is only 3x3,
        just removing 1 converts it back to cartesian movement.
        """
        x_component = (movement_index % dimension) - (dimension // 2)
        y_component = (movement_index // dimension) - (dimension // 2)

        return x_component, y_component

    def will_move(self) -> bool:
        if abs(self.inc_x) >= 1 or abs(self.inc_y) >= 1:
            return True
        self.increment_movement()
        return False

    def increment_movement(self) -> None:
        self.inc_x += self.movement_vector[0] / self.time_step_relation
        self.inc_y += self.movement_vector[1] / self.time_step_relation

    def calculate_movement_vector(self, primary_movement_vector: tuple[float]) -> None:
        """
        Function that calculates the person's movement vector
        based on the primary movement vector that is being applied
        by the environment, that is the water drift vector.

        The resulting movement vector is the average of the primary movement vector
        and a noised version of it, simulating more natural or unpredictable motion.
        """
        noised_vector = self.noise_vector(primary_movement_vector)
        self.movement_vector = (
            (primary_movement_vector[0] + noised_vector[0]) / 2,
            (primary_movement_vector[1] + noised_vector[1]) / 2,
        )

    def noise_vector(
        self,
        primary_movement_vector: tuple[float],
        speed_factor_range: tuple[float, float] = (0.5, 1.5),
    ) -> tuple[float]:
        """
        Generates a 'noised' version of a given primary movement vector,
        simulating more natural or unpredictable motion.

        The function introduces variability to the vector's direction and magnitude by
        applying a randomly chosen angle and an inverse exponential adjustment based on
        the angle's deviation from a base angle (90 degrees).
        A speed factor, influenced by the angle's normalized deviation, scales the
        vector's magnitude within a specified range.

        Parameters:
        ----------
        primary_movement_vector : tuple[float]
            The original movement vector to be noised.
        speed_factor_range : tuple[float, float], optional
            A tuple specifying the range within which the speed factor will be scaled.
            Default values allows for both reduction and amplification of the vector's magnitude.

        Returns:
        -------
        tuple[float]: A noised version of the input vector, represented as a tuple of floats,
        incorporating random variations in both direction and magnitude.

        Notes:
        ------
        If the input vector is a zero vector, the function returns a zero vector since no direction
        or magnitude can be meaningfully adjusted.
        """

        if norm(primary_movement_vector) == 0:
            return (0.0, 0.0)

        angle = radians(uniform(*self.angle_range))
        angle_base = pi / 2
        normalized_angle_diff = abs(angle - angle_base) / pi

        speed_factor = speed_factor_range[0] + (
            exp(-normalized_angle_diff)
            * (speed_factor_range[1] - speed_factor_range[0])
        )

        primary_direction = np.array(primary_movement_vector) / norm(
            primary_movement_vector
        )
        rotation_matrix = np.array(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        )
        noised_direction = np.dot(rotation_matrix, primary_direction)
        noised_vector = noised_direction * norm(primary_movement_vector) * speed_factor

        return tuple(noised_vector)

    def safe_position_update(self, movement: tuple[int]) -> None:
        """
        Updates the shipwrecked person position on given movement, checking for edge cases first.
        """
        new_x = self.x + movement[0]
        new_y = self.y + movement[1]
        if self.is_safe_position(new_x):
            self.x = new_x
        if self.is_safe_position(new_y):
            self.y = new_y

    def is_safe_position(self, new_position: int) -> bool:
        return 0 <= new_position < self.grid_size

    def update_time_step_relation(self, time_step: float, cell_size: float) -> None:
        self.time_step_relation = calculate_time_step(
            time_step, self.movement_vector, cell_size
        )

    def reset_position(self):
        self.x, self.y = self.initial_position

    def get_position(self) -> tuple[int]:
        return (self.x, self.y)

    def set_mult(self, mult: int) -> None:
        self.mult = mult

    def get_mult(self) -> int:
        return self.mult

    def __hash__(self) -> int:
        return hash(self.index)
