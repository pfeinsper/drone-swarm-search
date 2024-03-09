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
    probability_of_detection: float
        The probability of detection is the likelihood that
        the drone will detect a target in the search area.
    """
    amount: int
    speed: float
    probability_of_detection: float

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("The number of drones must be greater than 0.")
        if self.speed <= 0:
            raise ValueError("The drone's speed must be greater than 0.")
        if self.probability_of_detection <= 0:
            raise ValueError("The probability of detection must be greater than 0.")
