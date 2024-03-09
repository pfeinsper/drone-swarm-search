import numpy as np
from numba import njit
from .circle import Circle


class ProbabilityMatrix:
    def __init__(
        self, amplitude, spacement_x, spacement_y, vector, initial_position, size
    ):
        """
        Dynamic Probability Matrix

        Parameters
        ----------
        amplitude : float
            amplitude of gaussian curve
        spacement_x : float
            spacement of gaussian curve along the x axis
        spacement_y : float
            spacement of gaussian curve along the y axis
        vector : list
            vector that determines the movement of the target ("water current + wind")
        initial_position : list
            position of target, in form of list: [row, column]
        size : int
            size of the map
        """
        self.amplitude = amplitude
        self.spacement_x = spacement_x
        self.spacement_y = spacement_y
        self.supposed_position = initial_position
        self.map = np.zeros((size, size), dtype=float)
        self.map_prob = np.zeros((size, size), dtype=float)
        self.vector = vector

        # These determine the movement of the target as well
        self.x = 0
        self.y = 0
        # Circle that cointains the probabilities
        self.circle = Circle(1, initial_position[1], initial_position[0])

    def step(self):
        self.update_position()
        self.diffuse_probability()

        self.circle.update_center(self.supposed_position)
        self.circle.increase_area()

    def update_position(self):
        if abs(self.x) >= 1:
            self.x = 0
        if abs(self.y) >= 1:
            self.y = 0
        self.x += self.vector[0]
        self.y += self.vector[1]
        new_position = (
            self.supposed_position[0] + int(self.y),
            self.supposed_position[1] + int(self.x),
        )

        if self.is_valid_position(new_position):
            self.supposed_position = new_position

    def is_valid_position(self, position: tuple[int]) -> bool:
        is_valid_y = position[0] >= 0 and position[0] < len(self.map)
        is_valid_x = position[1] >= 0 and position[1] < len(self.map[0])
        return is_valid_x and is_valid_y

    def diffuse_probability(self):
        # NUMBA version of the diffuse_probability function
        all_cells = self.all_cells_inside_circle(
            self.circle.x0, self.circle.y0, self.circle.radius, *self.map.shape
        )
        map_copy = self.calc_probs(
            all_cells,
            self.amplitude,
            self.supposed_position,
            self.spacement_x,
            self.spacement_y,
            self.map.shape,
        )

        # Numpy version of the diffuse_probability function
        # entire_cells = self.all_points_inside_circle_numpy()
        # map_copy = self.calc_probs_numpy(entire_cells)

        map_copy_sum = map_copy.sum()
        if map_copy_sum == 0:
            self.map_prob = map_copy
        else:
            self.map_prob = map_copy / map_copy_sum

        self.map = map_copy

    # Uses classmethod because numba does not support instance methods.
    @staticmethod
    @njit(cache=True, fastmath=True)
    def all_cells_inside_circle(x0, y0, radius, rows, columns) -> np.array:
        radius_sq = radius**2

        # Optimization: Cut the search area to the minimum necessary
        max_x = min(np.ceil(x0 + radius) + 1, columns)
        min_x = max(np.floor(x0 - radius) - 1, 0)
        max_y = min(np.ceil(y0 + radius) + 1, rows)
        min_y = max(np.floor(y0 - radius) - 1, 0)

        res = np.zeros((rows, columns), dtype=np.int8)
        for row in range(min_y, max_y):
            for column in range(min_x, max_x):
                distance = (column - x0) ** 2 + (row - y0) ** 2
                if distance <= radius_sq:
                    res[row, column] = 1

        return np.argwhere(res == 1)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def calc_probs(
        cells: np.array, amplitude, supposed_position, spacement_x, spacement_y, shape
    ) -> np.array:
        probabilities = np.zeros(shape, dtype=np.float64)
        for row, column in cells:
            x = ((column - supposed_position[1]) ** 2) / (2 * (spacement_x**2))
            y = ((row - supposed_position[0]) ** 2) / (2 * (spacement_y**2))

            probability = amplitude * np.exp(-(x + y))
            probabilities[row, column] = probability
        return probabilities

    def all_points_inside_circle_numpy(self):
        x0, y0, radius = self.circle.x0, self.circle.y0, self.circle.radius
        rows, columns = self.map.shape
        x = np.arange(0, columns)
        y = np.arange(0, rows)
        x, y = np.meshgrid(x, y)
        res = (x - x0) ** 2 + (y - y0) ** 2 <= radius**2
        return np.argwhere(res)

    def calc_probs_numpy(self, cells):
        probabilities = np.zeros(self.map.shape, dtype=np.float64)
        # Numpy version of the calc_prob function
        x = ((cells[:, 1] - self.supposed_position[1]) ** 2) / (
            2 * (self.spacement_x**2)
        )
        y = ((cells[:, 0] - self.supposed_position[0]) ** 2) / (
            2 * (self.spacement_y**2)
        )
        probabilities[cells[:, 0], cells[:, 1]] = self.amplitude * np.exp(-(x + y))
        return probabilities

    def get_matrix(self):
        return self.map_prob

    def get_params(self):
        return self.circle
