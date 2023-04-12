from core.algorithms.baseline.parallel_sweep_single import ParallelSweepSingle
from core.environment.env import CustomEnvironment


def parallel_sweep_single():
    matrix_size = 5
    parallel_sweep = ParallelSweepSingle(matrix_size)
    env = CustomEnvironment(matrix_size, render_mode="human")

    env.reset()
    done = False

    while not done:
        for action in parallel_sweep.genarate_next_action():
            _, _, _, done, _ = env.step(action)
            done = done["drone"]

            if done:
                break


if __name__ == "__main__":
    parallel_sweep_single()
