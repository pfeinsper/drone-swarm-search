import h5py
from datetime import timedelta, datetime
import os
import numpy as np
from math import pi, ceil
import matplotlib.pyplot as plt
from copy import deepcopy

EARTH_MEAN_RADIUS = 6.3781 * 10**6  # in meters


class ParticleInterface:
    """
    A class for interfacing with the .h5 dataset and turning it into a matrix of probabillities of containment
    """

    def __init__(
        self,
        dataset_path: str,
        group_key: str,
        time_step: timedelta,
        cell_size: float = 45,
        pre_render_time: timedelta = timedelta(microseconds=0),
        square_matrix: bool = True,
    ):
        """
        A class for interfacing with the .h5 dataset and turning it into a matrix of probabillities of containment.

        :param dataset_path: The absolute path to the .h5 dataset
        :type dataset_path: str
        :param group_key: the key for the group in the h5 dataset that is being used. Typically in the form example_i
        :type group_key: str
        :param time_step: The between the calling of the .step() method. Typically the time it takes for a drone to traverse its swept area
        :type time_step: timedelta
        :param cell_size: the size of the cells in m. Typically the area a drone can view at any given time ("swept area")
        :type cell_size: float
        :param pre_render_time: Time to simulate before the first time step
        :type pre_render_time: timedelta
        """

        self.dataset_path = dataset_path
        self.group_key = group_key
        self.env_time_step = time_step
        self.cell_size = cell_size
        self.env_time_steps = 0  # how many times the .step() method has been called
        self.pre_render_time = pre_render_time
        self.elapsed_time = (
            pre_render_time  # how much time has passed in the simulation
        )
        self.cumm_p_success = 0
        self.square_matrix = square_matrix

        try:
            with h5py.File(self.dataset_path, "r") as f:
                # loading the h5py data into memory
                # the only missing data is the start time, which should be unecessary

                grp = f[self.group_key]
                lat = grp["lat"][()]
                long = grp["long"][()]
                self.particle_probabillities = grp["prob"][()]
                self.particle_probabillities_no_bayes = deepcopy(
                    self.particle_probabillities
                )
                self.particle_sim_time_delta = str(grp["time delta"][()])

                if (
                    self.particle_sim_time_delta.split(" ")[1].strip("'")
                    != "nanoseconds"
                ):  # checks for the correct units of timedelta
                    print("The unit for timedelta must be nanoseconds")
                    print(
                        "Found " + self.particle_sim_time_delta.split(" ")[1].strip("'")
                    )
                    raise AssertionError

                self.particle_sim_time_delta = timedelta(
                    microseconds=int(self.particle_sim_time_delta.split(" ")[0][2:])
                    / 1000
                )
                self.particle_sim_time_steps = int(
                    round((pre_render_time) / self.particle_sim_time_delta)
                )  # how many time steps have passed for the particle simulation. Essentially, which frame of the particle trajectory are we on

                print("Successfully loaded h5 dataset")

        except FileNotFoundError:
            print("Invalid path to h5 dataset")
            raise FileNotFoundError
        except KeyError:
            print("Invalid group key for h5 dataset")
            raise KeyError

        # Converting the lat long coordinates into absolute coordinates with units of meters based on the reference rectangle
        # The reference rectangle is the rectangle that holds all particles over all time periods (it isn't moving or anything)
        # If it crosses the line that is -180 = 180 longitude (weird edge case)
        edge_case = False
        if (long > 0).any() and (long < 0).any() and np.max(long) - np.min(long) > 180:
            south_west_ref = (np.min(lat), np.max(long))  # (lat, long)
            north_west_ref = (np.max(lat), np.max(long[long < 0]))
            edge_case = True
        else:
            south_west_ref = (np.min(lat), np.min(long))
            north_west_ref = (np.max(lat), np.max(long))

        if not edge_case:
            lat_relative = lat - south_west_ref[0]
            long -= south_west_ref[1]
        else:
            lat_relative = lat - south_west_ref[0]
            for row in long:
                for elem in row:
                    if elem > 0:
                        elem = elem - south_west_ref[1]
                    else:
                        elem = elem - south_west_ref[0] + 360

        # This grid will be built such that the south west reference is (0,0) and east is increasing x and north is increasing y
        self.y_vals = EARTH_MEAN_RADIUS * lat_relative / 180 * pi
        self.x_vals = (
            EARTH_MEAN_RADIUS * long / 180 * pi * np.cos(south_west_ref[0] / 180 * pi)
        )

        r = ceil(np.max(self.y_vals) / self.cell_size)
        c = ceil(np.max(self.x_vals) / self.cell_size)
        self.probability_matrix = np.zeros(shape=(r, c))
        if self.square_matrix:
            self.probability_matrix = make_array_square(self.probability_matrix)
        self.prob_matrix_shape = self.probability_matrix.shape

        print(f"Grid Size: {self.prob_matrix_shape}")

        # First dimention = time. Then it is just a grid. Each grid contains a list of particle indices that it contains
        self.particle_matrix = [
            [
                [[] for i in range(self.probability_matrix.shape[1])]
                for j in range(self.probability_matrix.shape[0])
            ]
            for k in range(self.x_vals.shape[1])
        ]

        self.y_len = np.max(self.y_vals)
        self.x_len = np.max(self.x_vals)

        for time_step in range(self.x_vals.shape[1]):
            for particle_num in range(self.x_vals.shape[0]):
                row = int(
                    (self.y_len - self.y_vals[particle_num][time_step]) / self.y_len * r
                )
                col = int(self.x_vals[particle_num][time_step] / self.x_len * c)
                if row == r:
                    row -= 1
                if col == c:
                    col -= 1
                try:
                    self.particle_matrix[time_step][row][col].append(particle_num)
                except IndexError:
                    raise IndexError

        self.cur_particle_matrix = self.particle_matrix_with_interpolation()
        self.update_probabillity_matrix_from_particle_matrix()

    def particle_matrix_with_interpolation(self):
        # The proportions is what fraction of the way the sim is between successive updates
        proportion = (
            self.elapsed_time
            - self.particle_sim_time_steps * self.particle_sim_time_delta
        ) / self.particle_sim_time_delta
        # Interpolates between the two particle postitions at the current time step and the next one
        x_vals_interpol = (
            self.x_vals[:, self.particle_sim_time_steps] * (1 - proportion)
            + self.x_vals[:, self.particle_sim_time_steps + 1] * proportion
        )
        # Does the same for the y values
        y_vals_interpol = (
            self.y_vals[:, self.particle_sim_time_steps] * (1 - proportion)
            + self.y_vals[:, self.particle_sim_time_steps + 1] * proportion
        )
        # Creates a new matrix with the interpolated values
        new_particle_matrix = [
            [[] for _ in range(self.probability_matrix.shape[1])]
            for _ in range(self.probability_matrix.shape[0])
        ]
        r = ceil(np.max(self.y_vals) / self.cell_size)
        c = ceil(np.max(self.x_vals) / self.cell_size)
        for particle_num in range(self.x_vals.shape[0]):
            row = int((self.y_len - y_vals_interpol[particle_num]) / self.y_len * r)
            col = int(x_vals_interpol[particle_num] / self.x_len * c)
            if row == r:
                row -= 1
            if col == c:
                col -= 1
            try:
                new_particle_matrix[row][col].append(particle_num)
            except IndexError:
                raise IndexError
        return new_particle_matrix

    def draw_plot(self):
        colors = [(0, 0, (20 * i) / 255) for i in range(11)]
        for i in range(self.x_vals.shape[1]):
            plt.scatter(x=self.x_vals[:, i], y=self.y_vals[:, i], c=colors[i])
        plt.show()

    def update_probabillity_matrix_from_particle_matrix(self):
        # O(n^3)
        for row in range(len(self.cur_particle_matrix)):
            for col in range(len(self.cur_particle_matrix[row])):
                p = 0
                for particle_id in self.cur_particle_matrix[row][col]:
                    p += self.particle_probabillities[particle_id]
                self.probability_matrix[row][col] = p

    def step(self, indices=[], pod=0.75) -> tuple[np.ndarray, float, bool]:
        """
        Should be called everytime step is called in the main loop, or whenever a unit of time equivalent to the environment's
        step time (drone movement time between two cells) has passed

        :param indices: List of indices where drones are searching in the form [(row, col), (row, col), ...]
        :type indices:
        :param pod: Probabillity the drone find the survivor if the survivor is in the drone's cell
        :type pod: float
        :return: Probabillity matrix, The total cumulative proabibillity the victim has been found until now, and if the simulation has reached the end of the particle simulation
        :rtype: tuple[np.ndarray, float, bool]
        """
        sim_over = False
        self.cumm_p_success = self.search_grid(pod, indices)

        self.env_time_steps += 1
        self.elapsed_time = min(
            self.pre_render_time + self.env_time_steps * self.env_time_step,
            (self.x_vals.shape[1] - 1) * self.particle_sim_time_delta,
        )
        if (
            self.elapsed_time
            - self.particle_sim_time_steps * self.particle_sim_time_delta
            >= self.particle_sim_time_delta
        ):
            if self.particle_sim_time_steps < len(self.particle_matrix) - 1:
                self.particle_sim_time_steps += int(
                    round(
                        (
                            self.pre_render_time
                            + self.env_time_steps * self.env_time_step
                            - self.particle_sim_time_steps
                            * self.particle_sim_time_delta
                        )
                        / self.particle_sim_time_delta
                    )
                )

        if (
            self.elapsed_time
            == (self.x_vals.shape[1] - 1) * self.particle_sim_time_delta
        ):
            sim_over = True
            self.particle_sim_time_steps = len(self.particle_matrix) - 1

        if self.particle_sim_time_steps < len(self.particle_matrix) - 1:
            self.cur_particle_matrix = self.particle_matrix_with_interpolation()
        else:
            self.cur_particle_matrix = self.particle_matrix[
                self.particle_sim_time_steps
            ]
            print("Reached the end of the particle simulation")

        self.update_probabillity_matrix_from_particle_matrix()

        return self.probability_matrix, self.cumm_p_success, sim_over

    def get_elapsed_time(self) -> timedelta:
        """Returns the elapsed time since the start of the simulation, including the pre-render time

        :return: elapsed time
        :rtype: timedelta
        """
        return self.elapsed_time

    def get_raw_matrix(self):
        return self.probability_matrix

    def get_matrix(self) -> np.ndarray:
        """Returns square probabillity matrix where extra zeros are added to make it square
        The original, smaller matrix exists in the top left.

        :return: square probabillity matrix
        :rtype: np.ndarray
        """
        return make_array_square(self.probability_matrix)

    def get_cummulative_p_success(self):
        return self.cumm_p_success

    def draw_heatmap(self, search_random=0, search_greedy=0, pod=0.75):
        print(f"Total Probabillity: {sum(self.particle_probabillities)}")

        fig, ax = plt.subplots()
        heatmap = ax.imshow(
            self.probability_matrix, cmap="viridis", interpolation="nearest"
        )
        col_bar = plt.colorbar(heatmap, ax=ax, label="Probability")
        ax.set_title("Press Space to Rerender")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        def on_key(event):
            if event.key == " ":  # Spacebar
                selected_search_cells = []

                if search_random > 0:
                    rows = np.random.randint(
                        0,
                        len(self.particle_matrix[self.particle_sim_time_steps]),
                        size=(search_random),
                    )
                    cols = np.random.randint(
                        0,
                        len(self.particle_matrix[self.particle_sim_time_steps][0]),
                        size=(search_random),
                    )
                    cells = list(zip(rows, cols))
                    selected_search_cells += [cells]

                if search_greedy > 0:
                    selected_search_cells += top_n_indices(
                        self.probability_matrix, search_greedy
                    )

                self.step(indices=selected_search_cells, pod=pod)

                ax.imshow(
                    self.probability_matrix, cmap="viridis", interpolation="nearest"
                )

                plt.draw()

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    def search_grid(self, pod: float, searched_cells: list[tuple[int]]) -> float:
        """
        Simulates the effect of searching a single square by changing the probabillity of the particle with a bayesian update
        Assumes that you do not find the victim.
        Asumes: The probabillity of finding each particle in a cell is independent, there is only one victim

        :param pod: The probabillity of detection, or the probabillity that the drone finds the person given they are there
        :type pod: float
        :param searched_cells: a list of tuples in the form (row, col)
        :type pod: list[tuple[int, int]]
        :return: the probabillity the victim was found during the ENTIRE search operation, not just during this search
        :rtype: float from 0 to 1
        """

        # Change to format [row, col, number of drones]
        numbered_searched_cells = []  # in format [row, col, number of drones]
        for i in range(len(searched_cells)):
            if searched_cells[i] in numbered_searched_cells:
                numbered_searched_cells.index(searched_cells[i])[2] += 1
            else:
                numbered_searched_cells.append(list(searched_cells[i]) + [1])

        # Calculate stuff for searched particles
        for cell in numbered_searched_cells:
            row, col, num_drones = cell
            for particle_id in self.cur_particle_matrix[row][col]:
                self.particle_probabillities[particle_id] *= (
                    1 - pod
                ) ** num_drones  # pod is probabillity of detection given the victim is there
                self.particle_probabillities_no_bayes[particle_id] *= (
                    1 - pod
                ) ** num_drones

        self.particle_probabillities /= sum(self.particle_probabillities)
        if sum(self.particle_probabillities) == 0:
            print(
                "All cells have been searched with complete certainity, so bayesian updates cannot be made"
            )
            raise AssertionError

        print(
            f"Prob Found During Operation: {1 - sum(self.particle_probabillities_no_bayes)}"
        )

        return 1 - sum(
            self.particle_probabillities_no_bayes
        )  # probabillity that the victim has been found


def make_array_square(arr: np.ndarray) -> np.ndarray:
    """Returns a square array so that both dimensions take the value of the current largest dimension and the
        new rows/columns that are added have a value of zero.

    :param arr: input rectangular array
    :type arr: np.ndarray
    :return: output square array
    :rtype: np.ndarray
    """
    assert len(arr.shape) == 2

    # Get current shape
    rows, cols = arr.shape
    max_dim = max(rows, cols)

    # Create a square zero-padded array
    square_arr = np.zeros((max_dim, max_dim), dtype=arr.dtype)

    # Copy original array into the top-left corner
    square_arr[:rows, :cols] = arr

    return square_arr


def top_n_indices(arr, n):
    # Flatten the array and get the indices of the top n values
    flat_indices = np.argpartition(arr.flatten(), -n)[-n:]
    # Convert flat indices to 2D indices
    return list(zip(*np.unravel_index(flat_indices, arr.shape)))
