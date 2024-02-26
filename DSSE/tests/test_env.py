import pytest
from DSSE import DroneSwarmSearch, Actions

@pytest.mark.parametrize("grid_size, n_drones", [
    (5, 26),
    (10, 101),
    (20, 401),
    (50, 2501),
])
def test_wrong_drone_number(grid_size, n_drones):
    with pytest.raises(ValueError):
        DroneSwarmSearch(
            grid_size=grid_size,
            render_mode="ansi",
            render_grid=True,
            render_gradient=True,
            n_drones=n_drones,
            vector=[-0.2, 0],
            person_initial_position=[0, 0],
            disperse_constant=1,
        )

@pytest.mark.parametrize("grid_size, max_drones", [
    (5, 25),
    (10, 100),
    (20, 400),
])
def test_maximum_drones_allowed(grid_size, max_drones):
    try:
        env = DroneSwarmSearch(
            grid_size=grid_size,
            render_mode="ansi",
            render_grid=True,
            render_gradient=True,
            n_drones=max_drones,
            vector=[-0.2, 0],
            person_initial_position=[0, 0],
            disperse_constant=1,
        )
        env.reset()
    except ValueError as e:
        pytest.fail(f"The system should not fail with the maximum allowed number of drones. Error: {str(e)}")
    
    assert len(env.get_agents()) == max_drones, f"There should be {max_drones} drones, but found {len(env.get_agents())}."



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

    rewards = 0
    done = False
    while not done:
        actions = {"drone0": Actions.SEARCH.value, "drone1": Actions.LEFT.value}
        _, reward, terminations, done, _ = env.step(actions)
        rewards += reward["total_reward"]
        done = any(done.values())
        
        assert done, "The simulation should terminate upon drone collision."
        assert any(terminations.values()), "There should be a termination flag set due to the collision."
        assert reward["total_reward"] < 0, "The total reward should be negative after a collision."

@pytest.mark.parametrize("timestep_limit", [
    10,
    20,
    30,
    40,
    50,
])
def test_timeout_termination(timestep_limit):
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

    done = False
    timestep_counter = 0
    while not done:
        actions = {"drone0": Actions.SEARCH.value}
        _, reward, terminations, done, _ = env.step(actions)
        done = any([e for e in done.values()])

        if timestep_counter > timestep_limit:
            assert reward["total_reward"] < 0, f"The total reward should be negative, indicating a penalty. But the total reward was: {reward['total_reward']}."
            assert done, "The simulation should flag as done after exceeding the timestep limit."
            assert any(terminations.values()), "A termination flag should be set due to exceeding the timestep limit."
        timestep_counter += 1
    assert timestep_counter > timestep_limit, "The simulation should run beyond the timestep limit before terminating."



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
    while not done and step_counter < 15:
        actions = {"drone0": Actions.RIGHT.value}
        _, reward, terminations, done, _ = env.step(actions)
        step_counter += 1
        done = any(done.values())

    assert step_counter >= 15, "The simulation should run for at least 15 steps before terminating."
    assert done, "The simulation should end, indicating the drone left the grid or another termination condition was met."
    assert reward["total_reward"] < 0, "The total reward should be negative, indicating a penalty was applied."
    assert any(terminations.values()), "There should be at least one termination condition met."


@pytest.mark.parametrize("person_position", [
    (19, 19),
    (-1, 0),
    (0, -1),
    (5, 5),
])
def test_should_raise_invalid_person_position(person_position):
    with pytest.raises(ValueError):
        DroneSwarmSearch(
            grid_size=5,
            render_mode="ansi",
            render_grid=True,
            render_gradient=True,
            n_drones=1,
            vector=[-0.2, 0],
            person_initial_position=person_position,
            disperse_constant=1,
        )



@pytest.mark.parametrize("n_drones", [
    5,
    10,
    15,
    20,
])
def test_if_all_drones_are_created(n_drones):
    env = DroneSwarmSearch(
        grid_size=20,
        render_mode="ansi",
        render_grid=True,
        render_gradient=True,
        n_drones=n_drones,
        vector=[-0.2, 0],
        person_initial_position=[4, 4],
        disperse_constant=1,
    )
    _ = env.reset()
    
    assert len(env.get_agents()) == n_drones, f"Should have {n_drones} drones, but found {len(env.get_agents())}."


@pytest.mark.parametrize("n_drones, drones_positions", [
    (1, [(3, 3)]),
    (2, [(12, 0), (0, 13)]),
    (3, [(0, 0), (19, 19), (15, 10)]),
    (4, [(5, 0), (0, 0), (1, 1), (10, 10)]),
])
def test_position_drone_is_correct_after_reset(n_drones, drones_positions):
    env = DroneSwarmSearch(
        grid_size=20,
        render_mode="ansi",
        render_grid=True,
        render_gradient=True,
        n_drones=n_drones,
        vector=[-0.2, 0],
        person_initial_position=[4, 4],
        disperse_constant=1,
    )
    
    observations = env.reset(drones_positions=drones_positions)
    
    for i, position in enumerate(drones_positions):
        drone_id = f"drone{i}"
        real_position_drone = observations[drone_id]["observation"][0]
        
        assert real_position_drone == position, f"Expected {drone_id}'s position to be {position} after reset, but was {real_position_drone}."


@pytest.mark.parametrize("n_drones, drones_positions", [
    (1, [(-1, 3)]),
    (2, [(12, 0), (25, 13)]),
    (3, [(0, 0), (19, 19), (25, -10)]),
    (4, [(5, 0), (0, 0), (10, 10), (10, 10)]),
])
def test_invalid_drone_position_raises_error(n_drones, drones_positions):
    with pytest.raises(ValueError):
        env = DroneSwarmSearch(
            grid_size=20,
            render_mode="ansi",
            render_grid=True,
            render_gradient=True,
            n_drones=n_drones,
            vector=[-0.2, 0],
            person_initial_position=[4, 4],
            disperse_constant=1,
        )
        _ = env.reset(drones_positions=drones_positions)


@pytest.mark.parametrize("n_drones", [
    1,
    20,
    35,
    48,
])
def test_if_all_drones_are_created_with_default_positions(n_drones):
    env = DroneSwarmSearch(
        grid_size=20,
        render_mode="ansi",
        render_grid=True,
        render_gradient=True,
        n_drones=n_drones,
        vector=[-0.2, 0],
        person_initial_position=[4, 4],
        disperse_constant=1,
    )
    _ = env.reset()
    
    assert len(env.get_agents()) == n_drones, f"Should have {n_drones} drones, but found {len(env.get_agents())}."


@pytest.mark.parametrize("n_drones, grid_size", [
    (1, 10),
    (2, 15),
    (5, 20),
    (15, 25),
])
def test_with_the_observation_size_is_correct_for_all_drones(n_drones, grid_size):
    env = DroneSwarmSearch(
        grid_size=grid_size,
        render_mode="ansi",
        render_grid=True,
        render_gradient=True,
        n_drones=n_drones,
        vector=[-0.2, 0],
        person_initial_position=[4, 4],
        disperse_constant=1,
    )
    observations = env.reset()
    
    for drone in range(n_drones):
        drone_id = f"drone{drone}"
        observation_matriz = observations[drone_id]["observation"][1]
        
        assert observation_matriz.shape == (grid_size, grid_size), f"The observation matrix for {drone_id} should have a shape of ({grid_size}, {grid_size}), but was {observation_matriz.shape}."
