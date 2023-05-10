from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for environment variables"""

    grid_size: int
    n_drones: int
    vector: list[float]
    drones_initial_positions: list[list[float]]
    person_initial_position: list[float]
    disperse_constant: float
    time_limit: int


def get_config(config_number: int) -> EnvConfig:
    """Configuration for environment variables"""

    match config_number:
        case 1:
            return EnvConfig(
                grid_size=10,
                n_drones=1,
                vector=[0.5, 0.5],
                drones_initial_positions=[[0, 0]],
                person_initial_position=[5, 5],
                disperse_constant=3,
                time_limit=100,
            )
        case 2:
            return EnvConfig(
                grid_size=20,
                n_drones=1,
                vector=[0.5, 0.5],
                drones_initial_positions=[[0, 0]],
                person_initial_position=[10, 10],
                disperse_constant=5,
                time_limit=100,
            )

        case 3:
            return EnvConfig(
                grid_size=20,
                n_drones=2,
                vector=[0.5, 0.5],
                drones_initial_positions=[[0, 0], [0, 1]],
                person_initial_position=[10, 10],
                disperse_constant=5,
                time_limit=100,
            )
