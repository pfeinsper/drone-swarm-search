from numpy.linalg import norm

def calculate_time_step(
        time_step: float,
        speed: tuple[float],
        cell_size: float
    ) -> float:
    """
    Parameters:
    ----------
    time_step: float
        Time step in seconds
    speed: tuple[float]
        Speed of the object which the time step is being calculated in m/s (x and y components) 
    cell_size: float
        Size of the cells in meters

    Returns:
    -------
    int
        Time step realtion in number of iterations
    """
    speed_magnitude = norm(speed)
    return cell_size / speed_magnitude / time_step
