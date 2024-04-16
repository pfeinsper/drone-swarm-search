from dataclasses import dataclass


@dataclass
class DroneData:
    """
    Class to wrap the drone data.

    Attributes:
    -----------
    amount: int
        The number of drones in the environment.
    speed: float
        The speed of the drone in m/s.
    """

    amount: int
    speed: float
    pod: float = 1

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("The number of drones must be greater than 0.")
        if self.speed <= 0:
            raise ValueError("The drone's speed must be greater than 0.")
        if self.pod < 0 or self.pod > 1:
            raise ValueError("The probability of detection must be between 0 and 1.")
