import pytest
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.constants import Actions
from pettingzoo.test import parallel_api_test


def init_drone_swarm_search(
    drone_amount=1,
    render_mode="ansi",
):
    
    return CoverageDroneSwarmSearch(
        drone_amount=drone_amount,
        render_mode=render_mode,
        pre_render_time=12,
    )


@pytest.mark.parametrize(
    "grid_size, drone_amount",
    [
        (5, 26),
        (10, 101),
        (20, 401),
        (50, 2501),
    ],
)
def test_wrong_drone_number(grid_size, drone_amount):
    with pytest.raises(ValueError):
        init_drone_swarm_search(grid_size=grid_size, drone_amount=drone_amount)


@pytest.mark.parametrize(
    "grid_size, drone_amount",
    [
        (5, 25),
        (10, 100),
        (20, 400),
    ],
)
def test_maximum_drones_allowed(grid_size, drone_amount):
    try:
        env = init_drone_swarm_search(grid_size=grid_size, drone_amount=drone_amount)
        _ = env.reset()
    except ValueError as e:
        pytest.fail(
            f"The system should not fail with the maximum allowed number of drones. Error: {str(e)}"
        )

    assert (
        len(env.get_agents()) == drone_amount
    ), f"There should be {drone_amount} drones, but found {len(env.get_agents())}."


@pytest.mark.parametrize(
    "drone_amount, drones_positions",
    [
        (2, [(0, 0), (2, 0),]),
    ],
)
def test_drone_collision_termination(drone_amount, drones_positions):

    env = init_drone_swarm_search(drone_amount=drone_amount)
    opt = {
    "drones_positions": drones_positions,
    }
    _ = env.reset(options=opt)

    done = False
    while not done:
        actions = {"drone0": Actions.RIGHT.value, "drone1": Actions.LEFT.value}
        _, reward, terminations, truncations, _ = env.step(actions)
        done = any(terminations.values()) or any(truncations.values())

        assert done, "The simulation should terminate upon drone collision."
        assert any(
            terminations.values()
        ), "There should be a termination flag set due to the collision."
        assert (
            sum(reward.values()) < 0
        ), "The total reward should be negative after a collision."

@pytest.mark.parametrize(
    "grid_size",
    [
        15,
        20,
        25,
        30,
    ],
)
def test_leave_grid_get_negative_reward(grid_size):
    env = init_drone_swarm_search(
        grid_size=grid_size
    )
    opt = {"drones_positions": [(0, 0)]}
    _ = env.reset(options=opt)

    done = False
    reward_sum = 0
    while not done and reward_sum >=  (env.reward_scheme.leave_grid * (env.timestep_limit-1)) +1:
        actions = {"drone0": Actions.UP.value}
        _, reward, terminations, done, _ = env.step(actions)
        done = any(done.values())
        reward_sum += sum(reward.values())

    assert (
        not done
    ), "The simulation should not end, indicating the drone left the grid or another termination condition was met."
    assert (
        sum(reward.values()) < 0
    ), "The total reward should be negative, indicating a penalty was applied."
    assert not any(
        terminations.values()
    ), "There not should be at least one termination condition met."

@pytest.mark.parametrize(
    "drone_amount",
    [
        5,
        10,
        15,
        20,
    ],
)
def test_if_all_drones_are_created(drone_amount):
    env = init_drone_swarm_search(drone_amount=drone_amount)
    _ = env.reset()

    assert (
        len(env.get_agents()) == drone_amount
    ), f"Should have {drone_amount} drones, but found {len(env.get_agents())}."


@pytest.mark.parametrize(
    "drone_amount, drones_positions",
    [
        (1, [(3, 3)]),
        (2, [(12, 0), (0, 13)]),
        (3, [(0, 0), (19, 19), (15, 10)]),
        (4, [(5, 0), (0, 0), (1, 1), (10, 10)]),
    ],
)
def test_position_drone_is_correct_after_reset(drone_amount, drones_positions):
    env = init_drone_swarm_search(drone_amount=drone_amount)

    opt = {"drones_positions": drones_positions}
    observations, _ = env.reset(options=opt)

    for i, position in enumerate(drones_positions):
        drone_id = f"drone{i}"
        real_position_drone = observations[drone_id][0]

        assert (
            real_position_drone == position
        ), f"Expected {drone_id}'s position to be {position} after reset, but was {real_position_drone}."


@pytest.mark.parametrize(
    "drone_amount, drones_positions",
    [
        (1, [(-1, 3)]),
        (2, [(12, 0), (25, 13)]),
        (3, [(0, 0), (19, 19), (25, -10)]),
        (4, [(5, 0), (0, 0), (10, 10), (10, 10)]),
    ],
)
def test_invalid_drone_position_raises_error(drone_amount, drones_positions):
    with pytest.raises(ValueError):
        env = init_drone_swarm_search(drone_amount=drone_amount)
        opt = {"drones_positions": drones_positions}
        _ = env.reset(options=opt)


@pytest.mark.parametrize(
    "drone_amount",
    [
        1,
        20,
        35,
        48,
    ],
)
def test_if_all_drones_are_created_with_default_positions(drone_amount):
    env = init_drone_swarm_search(drone_amount=drone_amount)

    _ = env.reset()

    assert (
        len(env.get_agents()) == drone_amount
    ), f"Should have {drone_amount} drones, but found {len(env.get_agents())}."


@pytest.mark.parametrize(
    "drone_amount, grid_size",
    [
        (1, 10),
        (2, 15),
        (5, 20),
        (15, 25),
    ],
)
def test_with_the_observation_size_is_correct_for_all_drones(drone_amount, grid_size):
    env = init_drone_swarm_search(grid_size=grid_size, drone_amount=drone_amount)

    observations, _ = env.reset()

    for drone in range(drone_amount):
        drone_id = f"drone{drone}"
        observation_matriz = observations[drone_id][1]

        assert observation_matriz.shape == (
            grid_size,
            grid_size,
        ), f"The observation matrix for {drone_id} should have a shape of ({grid_size}, {grid_size}), but was {observation_matriz.shape}."


def test_petting_zoo_interface_works():
    env = init_drone_swarm_search()
    parallel_api_test(env)
    env.close()
