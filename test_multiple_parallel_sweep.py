from core.algorithms.baseline.parallel_sweep import MultipleParallelSweep
from core.environment.env import CustomEnvironment


env = CustomEnvironment(grid_size=8, n_drones=16)
algorithm = MultipleParallelSweep(env)
for action in algorithm.generate_next_action():
    print(action)
