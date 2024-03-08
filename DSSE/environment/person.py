from random import randint, random
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
    """

    def __init__(self, amount: int, initial_position: tuple[int, int]):
        if amount <= 0:
            raise ValueError("The number of persons must be greater than 0.")
        self.amount = amount
        self.initial_position = initial_position
        self.x, self.y = self.initial_position
        self.time_step_counter = 0
        self.time_step_relation = 1
        
    def noise_person_movement(
        self, current_movement: tuple[int], drift_vector: list[int], epsilon=1.0
    ) -> tuple[int]:
        chance = random()
        if chance < epsilon:
            randomized_movement = np.array([randint(-1, 1), randint(-1, 1)])
            angle = self.angle_between(randomized_movement, drift_vector)
            # Only noises the movement if the new movement isnt against the vector.
            if angle < 120 or angle > 240:
                return randomized_movement
        return current_movement

    def angle_between(self, movement: np.array, drift_vector: list[int]) -> float:
        direction_movement = self.get_unit_vector(movement)
        direction_vector = self.get_unit_vector(np.array(drift_vector))
        dot_product = np.dot(direction_movement, direction_vector)
        return np.degrees(np.arccos(dot_product))

    def get_unit_vector(self, original_vector: np.array) -> float:
        vector_norm = np.linalg.norm(original_vector)
        if vector_norm == 0.0:
            vector_norm = 1.0
        return original_vector / vector_norm


    def update_shipwrecked_position(self, probability_matrix: np.array) -> tuple[int]:
        """
        Function that takes a 3x3 cut of the DynamicProbability matrix, multiplies it by a random numbers matrix [0, 1],
        and returns the column and line of the highest probability on the resulting matrix.

        Output:
            (movement_x, movement_y): tuple[int]
        """
        random_numbers_matrix = np.random.rand(*probability_matrix.shape)
        probabilities_mult_random_factor = random_numbers_matrix * probability_matrix

        # Using a numpy function to find the line and column of the greatest probability in the random factor multiplied matrix.
        max_probabilities = np.unravel_index(
            np.argmax(probabilities_mult_random_factor, axis=None), probability_matrix.shape
        )
        max_line = max_probabilities[0]
        max_column = max_probabilities[1]

        return self.movement_to_cartesian(max_column, max_line)


    def movement_to_cartesian(self, mov_x: int, mov_y: int) -> tuple[int]:
        """
        The movement of the shipwrecked person on the input follows the scheme (for the value of line and column):
            - if 0 -> Move to the left (x - 1) or to the top (y - 1).
            - if 1 -> No movement.
            - if 2 -> Move to the right (x + 1) or to the bottom (y + 1).

        So this function converts from this matrix movement notation to cartesian, as the matrix that creates this indexes is only 3x3,
        just removing 1 converts it back to cartesian movement.
        """
        x_component = mov_x - 1
        y_component = mov_y - 1
        return x_component, y_component


    def update_position(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def reset_position(self):
        self.x, self.y = self.initial_position

    def update_time_step_relation(self, time_step_relation: int):
        self.time_step_relation = time_step_relation

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
