from core.algorithms.baseline.parallel_sweep import MultipleParallelSweep
from core.environment.env import CustomEnvironment


env = CustomEnvironment(grid_size=8, n_drones=2, render_mode="human")
algorithm = MultipleParallelSweep(env)
algorithm.run()
