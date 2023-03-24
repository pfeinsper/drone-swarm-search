"""Compare the results of the two methods."""
import matplotlib.pyplot as plt
from typing import Tuple
import time

from config import Config
from greedy_search import greedy_search
from qlearning import qlearning
from qlearning_train import train_qlearning


def calculate_funtion_performance(function, *args) -> Tuple[float, int]:
    """
    Calculate the performance of a function.

    :param function: Function to be tested.
    :param args: Arguments to be passed to the function.
    :return: Tuple with the total time and the number of movements.
    """
    start_time = time.time()

    if function.__name__ == "greedy_search":
        _, movement_matrix = function(*args)
    else:
        movement_matrix = function(*args)

    end_time = time.time()
    total_time = end_time - start_time

    return total_time, len(movement_matrix)


def simulation():
    """
    Run greedy_search 30 times and plot the results.
    Run create_movement_matrix a single time and plot the in the same graph.
    """

    matrix_size = list(range(5, Config.grid_size + 1, 1))

    greedy_search_times = []
    greedy_search_movements = []
    qlearning_search_times = []
    qlearning_search_movements = []

    for size in matrix_size:
        greedy_search_time, greedy_search_movement = calculate_funtion_performance(
            greedy_search, size
        )
        greedy_search_times.append(greedy_search_time)
        greedy_search_movements.append(greedy_search_movement)

        train_qlearning(grid_size=size)
        qlearning_time, qlearning_movements = calculate_funtion_performance(
            qlearning, size
        )
        qlearning_search_times.append(qlearning_time)
        qlearning_search_movements.append(qlearning_movements)

    # Plot Greedy Search Movements vs Time
    plt.plot(matrix_size, greedy_search_times, label="Greedy Search")
    plt.plot(matrix_size, qlearning_search_times, label="Q-Learning")
    plt.xlabel("Matrix Size (n x n)")
    plt.ylabel("Time (s)")
    plt.title("Time Comparasion")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot Q-Learning Movements vs Time
    plt.plot(matrix_size, greedy_search_movements, label="Greedy Search")
    plt.plot(matrix_size, qlearning_search_movements, label="Q-Learning")
    plt.xlabel("Matrix Size (n x n)")
    plt.ylabel("Movements Taken (n)")
    plt.title("Movements Taken Comparasion")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    simulation()
