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
    pod: float
        The probability of detection for a target (between 0 and 1).
    battery_capacity: int
        The full battery capacity for each drone (default is 100).
    energy_per_move: int
        The energy consumption per move for each drone (default is 1).
    """

    amount: int
    speed: float
    pod: float = 1
    battery_capacity: int = 100
    energy_per_move: int = 1

    def __post_init__(self):
        if self.amount <= 0:
            raise ValueError("The number of drones must be greater than 0.")
        if self.speed <= 0:
            raise ValueError("The drone's speed must be greater than 0.")
        if self.pod < 0 or self.pod > 1:
            raise ValueError("The probability of detection must be between 0 and 1.")
        if self.battery_capacity <= 0:
            raise ValueError("Battery capacity must be greater than 0.")
        if self.battery_capacity > 100:
            raise ValueError("Battery capacity must be no more than 100.")
        if self.energy_per_move <= 0:
            raise ValueError("Energy consumed per move must be greater than 0.")
        if self.energy_per_move > self.battery_capacity:
            raise ValueError("Energy consumed per move must not be greater than battery capacity.")

        # Initialize battery for all drones to max
        self.batteries = [self.battery_capacity for _ in range(self.amount)]

    def consume_energy(self, drone_idx):
        # Decrease battery energy for one move
        if self.batteries[drone_idx] > 0:
            self.batteries[drone_idx] -= self.energy_per_move
            print(f"[DEBUG] Drone {drone_idx} battery after move: {self.batteries[drone_idx]}")
            return self.batteries[drone_idx]
        print(f"[DEBUG] Drone {drone_idx} has run out of battery.")
        return 0

    def recharge(self, drone_idx):
        # Recharge the specified drone to full battery
        self.batteries[drone_idx] = self.battery_capacity
        print(f"[DEBUG] Drone {drone_idx} recharged to full battery: {self.batteries[drone_idx]}")

    def get_battery(self, drone_idx):
        # Returns the current battery level for the specified drone
        return self.batteries[drone_idx]
