import pytest
from DSSE import DroneSwarmSearch
from DSSE import Actions

@pytest.mark.parametrize("grid_size, n_drones", [
    (5, 26),   # Testing with more drones than available spaces in a 5x5 grid
    (10, 101), # Testing with more drones than available spaces in a 10x10 grid
    (20, 401), # Testing with more drones than available spaces in a 20x20 grid
    (50, 2501),# Testing with more drones than available spaces in a 50x50 grid
])
def test_wrong_drone_number(grid_size, n_drones):
    """
    Tests that a ValueError is raised when attempting to create more drones than there are spaces in the grid.
    """
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
    (5, 25),   # 5x5 grid can accommodate 25 drones
    (10, 100), # 10x10 grid can accommodate 100 drones
    (20, 400), # 20x20 grid can accommodate 400 drones
])
def test_maximum_drones_allowed(grid_size, max_drones):
    """
    Tests if the system accepts the maximum number of drones that fit in the specified grid.
    """
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
    """
    This test verifies that the DroneSwarmSearch simulation correctly terminates with a penalty when two drones collide, ensuring collision detection and response mechanisms function as intended.
    """
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
    """
    Tests if the DroneSwarmSearch simulation properly terminates after reaching a predefined timestep limit,
    ensuring the simulation enforces time constraints and penalizes for exceeding the allowed duration.
    """
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
    """
    This test verifies that the DroneSwarmSearch simulation terminates, applying penalties as expected, when a drone leaves a 15x15 grid.
    """
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
    (19, 19),  # Outside the grid boundaries.
    (-1, 0),   # Negative position value, invalid.
    (0, -1),   # Negative position value, invalid.
    (5, 5),    # Exactly on the boundary (assuming 0-indexed and grid_size is exclusive).
])
def test_should_raise_invalid_person_position(person_position):
    """
    Tests DroneSwarmSearch raises ValueError for invalid or boundary person positions.
    
    The function initializes a DroneSwarmSearch environment with specified parameters,
    including the person's position. It expects a ValueError to be raised for positions
    that are outside the grid or exactly on its boundary, indicating the position is invalid.
    """
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
    5,  # Testing with 5 drones.
    10, # Testing with 10 drones.
    15, # Testing with 15 drones to see if the system scales.
    20, # Testing with 20 drones for upper limit in a fixed grid size.
])
def test_if_all_drones_are_created(n_drones):
    """
    Tests whether the specified number of drones (n_drones) are correctly created in the DroneSwarmSearch environment.
    """
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
    (1, [(3, 3)]),                      # Testing with 1 drone.
    (2, [(12, 0), (0, 13)]),            # Testing with 2 drones.
    (3, [(0, 0), (19, 19), (15, 10)]),  # Testing with 3 drones.
    (4, [(5, 0), (0, 0), (1, 1), (10, 10)]),  # Testing with 4 drones.
])
def test_position_drone_is_correct_after_reset(n_drones, drones_positions):
    """
    Verifies that each drone is correctly positioned as per the specified initial positions
    after the environment is reset, with positions expected to be tuples.
    """
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
    (1, [(-1, 3)]),                      # Testing with 1 drone.
    (2, [(12, 0), (25, 13)]),            # Testing with 2 drones.
    (3, [(0, 0), (19, 19), (25, -10)]),  # Testing with 3 drones.
    (4, [(5, 0), (0, 0), (10, 10), (10, 10)]),  # Testing with 4 drones.
])
def test_invalid_drone_position_raises_error(n_drones, drones_positions):
    """
    Verifies that the DroneSwarmSearch environment raises a ValueError when trying to place a drone in an invalid position.
    """
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
    """
    Tests whether the specified number of drones are correctly created in the DroneSwarmSearch environment with default positions.
    """
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
    """
    Tests if the size of the observation for each drone is correct.
    """
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


