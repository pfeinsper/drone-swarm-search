from core.algorithms.baseline.parallel_sweep import MultipleParallelSweep
from core.environment.env import DroneSwarmSearch


env = DroneSwarmSearch(
    grid_size=6,
    render_mode="human",
    render_grid=True,
    render_gradient=True,
    n_drones=2,
    vector=[0.5, 0.5],
    person_initial_position=[1, 1],
    disperse_constant=5,
)
algorithm = MultipleParallelSweep(env)
algorithm.run()
