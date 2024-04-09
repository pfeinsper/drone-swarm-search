from random import randint, random, uniform, choice
from numpy.linalg import norm
from math import cos, sin, radians, pi, exp
import numpy as np

class Person():
    """
    Attributes:
    -----------
    amount: int
        The number of shipwrecked people in the environment.
    person_initial_position: tuple
        The initial position of the shipwrecked people in the environment.
    time_step_counter: int
        The time step counter of the person in the environment.
    time_step_relation: int
        The number of time steps in relation
        to the environment's time step. 
        It defines the amount of environment's time steps that must 
        occur in order to the person's time step to occur.
    x: int
        The x coordinate of the person in the environment.
    y: int
        The y coordinate of the person in the environment.
    movement_vector: tuple
        The vector that determines the movement of the person in the environment.
    """
    angle_ranges_list = [(0, 45), (20, 55), (300, 360), (315, 360)]
    
    # Selecting a random range from the list and storing it as a class variable.
    angle_range = choice(angle_ranges_list)
    
    def __init__(
            self,
            initial_position: tuple[int, int],
            grid_size: int,
            probability_of_detection: float = 0.9
        ):

        self.initial_position = initial_position
        self.x, self.y = self.initial_position
        self.inc_x, self.inc_y = 0, 0
        self.grid_size = grid_size
        self.time_step_counter = 0
        self.time_step_relation = 1
        self.pod = probability_of_detection
        self.movement_vector = (0.0, 0.0)

    def calculate_movement_vector(self, primary_movement_vector: tuple[float]) -> None:
        """
        Function that calculates the person's movement vector 
        based on the primary movement vector that is being applied 
        by the environment, that is the water drift vector.
        """
        noised_vector = self.noise_vector(primary_movement_vector)
        self.movement_vector = (
            (primary_movement_vector[0] + noised_vector[0]),
            (primary_movement_vector[1] + noised_vector[1])
        )

    def noise_vector(self, primary_movement_vector: tuple[float], speed_factor_range: tuple[float, float] = (0.5, 1.5)) -> tuple[float]:
        """
        Generates a 'noised' version of a given primary movement vector, simulating more natural or unpredictable motion.
        
        The function introduces variability to the vector's direction and magnitude by applying a randomly chosen angle 
        and an inverse exponential adjustment based on the angle's deviation from a base angle (90 degrees).
        A speed factor, influenced by the angle's normalized deviation, scales the vector's magnitude within a specified range.

        Parameters:
        - primary_movement_vector (tuple[float]): The original movement vector to be noised.
        - speed_factor_range (tuple[float, float], optional): A tuple specifying the range within which the speed factor will be scaled. 
        Defaults to (0.5, 1.5), allowing for both reduction and amplification of the vector's magnitude.

        Returns:
        - tuple[float]: A noised version of the input vector, represented as a tuple of floats, incorporating random variations in both direction and magnitude.

        Note:
        - If the input vector is a zero vector, the function returns a zero vector since no direction or magnitude can be meaningfully adjusted.
        """
        if norm(primary_movement_vector) == 0:
            return (0.0, 0.0)
        
        angle = radians(uniform(*Person.angle_range))
        angle_base = pi / 2
        normalized_angle_diff = abs(angle - angle_base) / pi
        
        speed_factor = speed_factor_range[0] + (exp(-normalized_angle_diff) * (speed_factor_range[1] - speed_factor_range[0]))
        
        primary_direction = np.array(primary_movement_vector) / norm(primary_movement_vector)
        rotation_matrix = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        noised_direction = np.dot(rotation_matrix, primary_direction)
        noised_vector = noised_direction * norm(primary_movement_vector) * speed_factor

        return tuple(noised_vector)

    def angle_between(self, movement: np.array, drift_vector: list[float]) -> float:
        direction_movement = self.get_unit_vector(movement)
        direction_vector = self.get_unit_vector(np.array(drift_vector))
        dot_product = np.dot(direction_movement, direction_vector)
        return np.degrees(np.arccos(dot_product))

    def get_unit_vector(self, original_vector: np.array) -> float:
        vector_norm = np.linalg.norm(original_vector)
        if vector_norm == 0.0:
            vector_norm = 1.0
        return original_vector / vector_norm

    def update_position(self, drone_speed: float, movement_map: np.array = None) -> None:
        movement = self.update_shipwrecked_position(drone_speed, movement_map)

        self.x = self.safe_1d_position_update(self.x, movement[0])
        self.y = self.safe_1d_position_update(self.y, movement[1])
    
    def update_shipwrecked_position(self, drone_speed: float, probability_matrix: np.array = None, dimension: int = 3) -> tuple[int]:
        """
        Function that takes a 3x3 cut of the DynamicProbability matrix, multiplies it by a random numbers matrix [0, 1],
        and returns the column and line of the highest probability on the resulting matrix.

        Output:
            (movement_x, movement_y): tuple[int]
        """

        # OLD CODE
        # random_numbers_matrix = np.random.rand(*probability_matrix.shape)
        # probabilities_mult_random_factor = random_numbers_matrix * probability_matrix

        # # Using a numpy function to find the line and column of the greatest probability in the random factor multiplied matrix.
        # max_probabilities = np.unravel_index(
        #     np.argmax(probabilities_mult_random_factor, axis=None), probability_matrix.shape
        # )
        # max_line = max_probabilities[0]
        # max_column = max_probabilities[1]

        # print(f"Max line: {max_line}, Max column: {max_column}")

        # return self.movement_to_cartesian(max_column, max_line, dimension)

        # NEW CODE
        if abs(self.inc_x) >= 1:
            self.inc_x = 0
        if abs(self.inc_y) >= 1:
            self.inc_y = 0

        self.inc_x += self.movement_vector[0] / drone_speed
        self.inc_y += self.movement_vector[1] / drone_speed

        # On row, column notation (y, x) -> (row, column)
        new_position = (
            int(self.inc_x),
            int(self.inc_y),
        )

        return self.movement_to_cartesian(new_position[0], new_position[1], dimension=0)

    def movement_to_cartesian(self, mov_x: int, mov_y: int, dimension: int) -> tuple[int]:
        """
        The movement of the shipwrecked person on the input follows the scheme (for the value of line and column):
            - if 0 -> Move to the left (x - 1) or to the top (y - 1).
            - if 1 -> No movement.
            - if 2 -> Move to the right (x + 1) or to the bottom (y + 1).

        So this function converts from this matrix movement notation to cartesian, as the matrix that creates this indexes is only 3x3,
        just removing 1 converts it back to cartesian movement.
        """
        x_component = mov_x - int(dimension / 2)
        y_component = mov_y - int(dimension / 2)

        return x_component, y_component
    
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

    def reset_position(self):
        self.x, self.y = self.initial_position

    def update_time_step_relation(self, time_step: float, cell_size: float) -> None:
        self.time_step_relation = self.calculate_time_step(time_step, self.movement_vector, cell_size)

    def calculate_time_step(
            self,
            time_step: float,
            person_speed: tuple[float],
            cell_size: float
        ) -> int:
        """
        Args:
        time_step: float
            Time step in seconds
        person_speed: tuple[float]
            Speed of the person in the water in m/s (x and y components)
        cell_size: float
            Size of the cells in meters
        """
        speed_magnitude, _ = self.calculate_vector_magnitude_and_direction(person_speed)
        return int(cell_size / speed_magnitude / time_step)

    def reached_time_step(self):
        reached = self.time_step_counter >= self.time_step_relation
        if reached:
            self.reset_time_step_counter()
        else:
            self.increment_time_step_counter()
        return reached

    def reset_time_step_counter(self) -> None:
        self.time_step_counter = 0

    def increment_time_step_counter(self) -> None:
        self.time_step_counter += 1

    def calculate_speed(
            self,
            water_speed: tuple[float] = (0, 0),
            wind_speed: tuple[float] = (0, 0),
            swimming_speed: tuple[float] = (0, 0)
        ) -> tuple[float]:
        """
        Calculate the speed of a person in the water
        This speed is calculated based on the sea surface current velocity, wind-induced drift velocity, and the swimming speed of the person
        v(x, t) = V_current(x, t) + V_leeway(x, t) + V_swim(x, t)

        Args:
        water_speed: tuple[float] (components in x and y directions)
            Sea surface current velocity in m/s
        wind_speed: tuple[float] (components in x and y directions)
            Wind speed in m/s
        swimming_speed: tuple[float] (components in x and y directions)
            Swimming speed of the person in m/s
        """
        return (
            water_speed[0] + wind_speed[0] + swimming_speed[0],
            water_speed[1] + wind_speed[1] + swimming_speed[1]
        ) # in m/s

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
    
    def get_position(self) -> tuple[int]:
        return (self.x, self.y)
    
    def set_pod(self, pod: int) -> None:
        self.pod = pod
    
    def get_pod(self) -> int:
        return self.pod
