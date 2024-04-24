from numpy.linalg import norm


def calculate_simulation_time_step(
    drone_max_speed: float, cell_size: float, wind_resistance: float = 0.0
) -> float:
    """
    Calculate the time step for the simulation based on the maximum speed of
    the drones and the cell size.

    Arguments:
    ----------
    drone_max_speed: float
        Maximum speed of the drones in m/s
    cell_size: float
        Size of the cells in meters
    wind_resistance: float
        Wind resistance in m/s

    Returns:
    --------
    float: The time in seconds that elapsed in real life for each simulation time step.
    """
    return cell_size / (drone_max_speed - wind_resistance)


def calculate_time_step(
    time_step: float, person_speed: tuple[float], cell_size: float
) -> float:
    """
    Parameters:
    ----------
    time_step: float
        Time step in seconds
    person_speed: tuple[float]
        Speed of the person-in-water in m/s (x and y components)
    cell_size: float
        Size of the cells in meters

    Returns:
    -------
    int
        Time step realtion in number of iterations
    """
    speed_magnitude = norm(person_speed)
    return cell_size / speed_magnitude / time_step
