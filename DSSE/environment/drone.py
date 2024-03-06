"""
Module to implement the Drone Dataclass.

This class defines the dromne data in order to parameterize the drone's behavior and
characteristics, such as the battery life, the speed, the maximum payload, sweep
width and track spacing, therefore enhancing the simulation of the drone's 
behavior and the accuracy of the environment according to the real world.

"""

from dataclasses import dataclass
from numpy import exp

@dataclass
class DroneData:
    """
    
    Class to wrap the drone data.

    Attributes:
    -----------
    speed: float
        The speed of the drone in m/s.
    sweep_width: float
        The sweep width of the drone in m.
    track_spacing: float
        The track spacing of the drone in m.
    coverage_factor: float
        The coverage factor is a measure for the degree of the search area that is being covered.
    probability_of_detection: float
        The probability of detection is the likelihood that
        the drone will detect a target in the search area.

    """
    speed: float
    sweep_width: float
    track_spacing: float

    def __post_init__(self):
        """
        Method to check if the attributes are valid and
        calculate the remaining attributes.
        """
        if self.speed <= 0:
            raise ValueError("The drone's speed must be greater than 0.")
        if self.sweep_width <= 0:
            raise ValueError("The drone's sweep width must be greater than 0.")
        if self.track_spacing <= 0:
            raise ValueError("The drone's track spacing must be greater than 0.")

        self.coverage_factor = self.sweep_width / self.track_spacing
        self.probability_of_detection = 1 - exp(-self.coverage_factor)
