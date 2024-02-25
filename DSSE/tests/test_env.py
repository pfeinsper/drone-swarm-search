import pytest
from DSSE import DroneSwarmSearch
from DSSE import Actions

def test_wrong_drone_number():
    with pytest.raises(ValueError):
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
        actions = {"drone0": Actions.SEARCH.value, "drone1": Actions.LEFT.value}
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
        actions = {"drone0": Actions.SEARCH.value}
        _, reward, terminations, done, _ = env.step(actions)
        done = any([e for e in done.values()])

        if timestep_counter > timestep_limit:
            assert reward["total_reward"] < 0
            assert done is True
            assert any(terminations.values())
        timestep_counter += 1
    assert timestep_counter > timestep_limit
    
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
        actions = {"drone0": Actions.RIGHT.value}
        _, reward, terminations, done, _ = env.step(actions)
        step_counter += 1
        done = any([e for e in done.values()])

        if step_counter >= 15:
            assert done is True
            assert reward["total_reward"] < 0
            assert any(terminations.values())
    assert step_counter >= 15


def test_should_raise_invalid_person_position():
    with pytest.raises(ValueError):
        DroneSwarmSearch(
            grid_size=5,
            render_mode="ansi",
            render_grid=True,
            render_gradient=True,
            n_drones=1,
            vector=[-0.2, 0],
            person_initial_position=[19, 19],
            disperse_constant=1,
        )

            



# TODO: Test error for drone positions in reset.
