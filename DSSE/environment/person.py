"""
Module for the Person Dataclass.

This class defines the person data in order to parameterize the person's behavior and
characteristics, such as the speed, the initial position and time step, therefore
enhancing the simulation of the person's behavior and the accuracy of the environment
according to the real world.

"""

from dataclasses import dataclass

@dataclass
class PersonData:
    """
    
    Class to wrap the person data.

    Attributes:
    -----------
    number_of_persons: int
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

    number_of_persons: int = 1
    initial_position: tuple[int, int] = (0, 0)

    def __post_init__(self):
        self.x, self.y = self.initial_position
        self.time_step_counter = 0
        self.time_step_relation = 1
        if self.number_of_persons <= 0:
            raise ValueError("The number of persons must be greater than 0.")

    def update_position(self, x: int, y: int):
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

    def reset_time_step_counter(self):
        self.time_step_counter = 0

    def increment_time_step_counter(self):
        self.time_step_counter += 1
