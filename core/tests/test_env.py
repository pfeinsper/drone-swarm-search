import pytest
from core.environment.env import DroneSwarmSearch

def test_wrong_drone_number():
    with pytest.raises(Exception):
        # More drones than spaces!
        DroneSwarmSearch(
            grid_size=5,
            render_mode="ansi",
            render_grid=True,
            render_gradient=True,
            n_drones=26,
            vector=[-0.2, 0],
            person_initial_position=[19, 19],
            disperse_constant=1,
        )


def test_drone_collision_termination():
    env = DroneSwarmSearch(
            grid_size=20,
            render_mode="ansi",
            render_grid=True,
            render_gradient=True,
            n_drones=2,
            vector=[-0.2, 0],
            person_initial_position=[19, 19],
            disperse_constant=1,
        )
    _ = env.reset()
    # observations = env.reset()

    rewards = 0
    done = False
    while not done:
        actions = {"drone0": 4, "drone1": 0}
        _, reward, terminations, done, _ = env.step(actions)
        rewards += reward["total_reward"]
        done = any([e for e in done.values()])

        assert reward["total_reward"] < 0
        assert done is True
        assert any(terminations.values())

def test_timeout_termination():
    timestep_limit = 50
    env = DroneSwarmSearch(
            grid_size=20,
            render_mode="ansi",
            n_drones=1,
            vector=[-0.2, 0],
            person_initial_position=[19, 19],
            disperse_constant=1,
            timestep_limit=timestep_limit
        )
    _ = env.reset()
    # observations = env.reset()

    done = False
    timestep_counter = 0
    while not done:
        actions = {"drone0": 4}
        _, reward, terminations, done, _ = env.step(actions)
        done = any([e for e in done.values()])

        if timestep_counter > timestep_limit:
            assert reward["total_reward"] < 0
            assert done is True
            assert any(terminations.values())
        timestep_counter += 1
    
def test_leave_grid_termination():
    env = DroneSwarmSearch(
        grid_size=15,
        render_mode="ansi",
        n_drones=1,
        person_initial_position=[4, 4],
        disperse_constant=1
    )
    _ = env.reset()

    done = False
    step_counter = 0
    while not done:
        actions = {"drone0": 1}
        _, reward, terminations, done, _ = env.step(actions)
        step_counter += 1
        done = any([e for e in done.values()])

        if step_counter >= 15:
            assert done is True
            assert reward["total_reward"] < 0
            assert any(terminations.values())
            



# TODO: Test error for drone positions in reset.
