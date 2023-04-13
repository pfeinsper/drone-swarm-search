from core.algorithms.baseline.parallel_sweep import SingleParallelSweep, DroneInfo
from core.environment.env import CustomEnvironment


def parallel_sweep_single():
    matrix_size = 1
    env = CustomEnvironment(matrix_size, render_mode="human")
    drone_info = DroneInfo(
        drone_id=0,
        grid_size=env.grid_size,
        initial_position=(0, 0),
        last_vertice=(env.grid_size - 1, env.grid_size - 1),
    )
    parallel_sweep = SingleParallelSweep(drone_info=drone_info)

    env.reset()
    done = False

    while not done:
        for action in parallel_sweep.genarate_next_action():
            _, _, _, done, _ = env.step(action)
            done = done["drone0"]

            if done:
                break


if __name__ == "__main__":
    parallel_sweep_single()
