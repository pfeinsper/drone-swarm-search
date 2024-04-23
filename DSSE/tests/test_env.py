import pytest
from DSSE import DroneSwarmSearch
from DSSE.environment.constants import Actions
from DSSE.tests.drone_policy import policy
from pettingzoo.test import parallel_api_test


def init_drone_swarm_search(
    grid_size=20,
    render_mode="ansi",
    render_grid=True,
    render_gradient=True,
    vector=(3.5, -0.5),
    disperse_inc=0.1,
    timestep_limit=200,
    person_amount=1,
    person_initial_position=None,
    drone_amount=1,
    drone_speed=10,
    probability_of_detection=0.9,
    pre_render_time=0,
):

    if person_initial_position is None:
        person_initial_position = (
            grid_size - round(grid_size / 2),
            grid_size - round(grid_size / 2),
        )

    return DroneSwarmSearch(
        grid_size=grid_size,
        render_mode=render_mode,
        render_grid=render_grid,
        render_gradient=render_gradient,
        vector=vector,
        dispersion_inc=disperse_inc,
        timestep_limit=timestep_limit,
        person_amount=person_amount,
        person_initial_position=person_initial_position,
        drone_amount=drone_amount,
        drone_speed=drone_speed,
        probability_of_detection=probability_of_detection,
        pre_render_time=pre_render_time,
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
    "drone_amount",
    [
        2,
    ],
)
def test_drone_collision_termination(drone_amount):

    env = init_drone_swarm_search(drone_amount=drone_amount)
    _ = env.reset()

    done = False
    while not done:
        actions = {"drone0": Actions.SEARCH.value, "drone1": Actions.LEFT.value}
        _, reward, terminations, truncations, _ = env.step(actions)
        done = any(truncations.values()) or any(terminations.values())

        assert done, "The simulation should terminate upon drone collision."
        assert any(
            terminations.values()
        ), "There should be a termination flag set due to the collision."
        assert (
            sum(reward.values()) < 0
        ), "The total reward should be negative after a collision."


@pytest.mark.parametrize(
    "timestep_limit",
    [
        10,
        20,
        30,
        40,
        50,
    ],
)
def test_timeout_termination(timestep_limit):
    env = init_drone_swarm_search(timestep_limit=timestep_limit)
    _ = env.reset()

    done = False
    timestep_counter = 0
    while not done:
        actions = {"drone0": Actions.SEARCH.value}
        _, reward, terminations, done, _ = env.step(actions)
        done = any([e for e in done.values()])

        if timestep_counter > timestep_limit:
            assert (
                sum(reward.values()) < 0
            ), f"The total reward should be negative, indicating a penalty. But the total reward was: {reward['total_reward']}."
            assert (
                done
            ), "The simulation should flag as done after exceeding the timestep limit."
            assert any(
                terminations.values()
            ), "A termination flag should be set due to exceeding the timestep limit."
        timestep_counter += 1
    assert (
        timestep_counter > timestep_limit
    ), "The simulation should run beyond the timestep limit before terminating."


@pytest.mark.parametrize(
    "grid_size, person_initial_position",
    [
        (15, (4, 4)),
        (20, (10, 10)),
        (25, (15, 15)),
        (30, (20, 20)),
    ],
)
def test_leave_grid_get_negative_reward(grid_size, person_initial_position):
    env = init_drone_swarm_search(
        grid_size=grid_size, person_initial_position=person_initial_position
    )
    opt = {"drones_positions": [(0, 0)]}
    _ = env.reset(options=opt)

    done = False
    reward_sum = 0
    while not done and reward_sum >= env.reward_scheme.leave_grid * (env.timestep_limit - 1):
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
    "grid_size, person_position",
    [
        (5, (19, 19)),
        (5, (-1, 0)),
        (5, (0, -1)),
        (5, (5, 5)),
    ],
)
def test_should_raise_invalid_person_position(grid_size, person_position):
    with pytest.raises(ValueError):
        init_drone_swarm_search(
            grid_size=grid_size, person_initial_position=person_position
        )


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


@pytest.mark.parametrize(
    "person_initial_position, person_amount",
    [
        ((10, 10), 10),
        ((10, 10), 15),
        ((10, 10), 20),
        ((10, 10), 25),
    ],
)
def test_castaway_count_after_reset(person_initial_position, person_amount):
    env = init_drone_swarm_search(
        person_amount=person_amount, person_initial_position=person_initial_position
    )
    _ = env.reset()

    assert (
        len(env.get_persons()) == person_amount
    ), f"Should have {person_amount} castaways, but found {len(env.get_persons())}."


@pytest.mark.parametrize(
    "person_initial_position, person_amount, drone_amount",
    [
        ((10, 10), 1, 1),
        ((1, 10), 5, 1),
        ((19, 5), 10, 1),
        ((5, 16), 15, 1),
    ],
)
def test_castaway_count_after_reset(
    person_initial_position, person_amount, drone_amount
):
    env = init_drone_swarm_search(
        person_amount=person_amount,
        person_initial_position=person_initial_position,
        drone_amount=drone_amount,
    )
    observations = env.reset()

    rewards = 0
    done = False
    while not done:
        actions = policy(observations, env.get_agents(), env)
        observations, reward, _, done, info = env.step(actions)
        rewards += sum(reward.values())
        done = any(done.values())

    _ = env.reset()

    assert (
        rewards >= DroneSwarmSearch.reward_scheme.search_and_find * person_amount
    ), f"The total reward should be positive after finding all castaways. But the total reward was: {rewards}."
    assert done, "The simulation should end after finding all castaways."
    assert (
        len(env.get_persons()) == person_amount
    ), f"Should have {person_amount} castaways, but found {len(env.get_persons())}."
    assert (
        len(env.get_agents()) == drone_amount
    ), f"Should have {drone_amount} drones, but found {len(env.get_agents())}."


@pytest.mark.parametrize(
    "pre_render_time, cell_size, drone_max_speed, wind_resistance",
    [
        (1, 130, 10, 0.0),
        (5, 130, 20, 0.0),
        (10, 130, 30, 0.0),
        (15, 130, 40, 0.0),
        (20, 130, 50, 0.0),
    ],
)
def test_pre_render_work_after_reset(
    pre_render_time, cell_size, drone_max_speed, wind_resistance
):
    env = init_drone_swarm_search(
        pre_render_time=pre_render_time, drone_speed=drone_max_speed
    )
    _ = env.reset()
    pre_render_steps = round(
        (pre_render_time * 60) / (cell_size / (drone_max_speed - wind_resistance))
    )

    assert (
        env.pre_render_steps == pre_render_steps
    ), f"The pre-render time should be {pre_render_steps}, but was {env.pre_render_time}."

    _ = env.reset()

    assert (
        env.pre_render_steps == pre_render_steps
    ), f"The pre-render time should be {pre_render_steps}, but was {env.pre_render_time}."


@pytest.mark.parametrize(
    "person_amount, mult",
    [
        (1, ["1"]),
        (2, [1, "0.8"]),
        (3, [1, "0.8", 0.7]),
        (4, ["1", 0.8, "0.7", 0.6]),
        (5, ["1", "0.8", "0.7", "0.6", "0.5"]),
    ],
)
def test_get_wrong_if_scale_pod_is_not_a_number(person_amount, mult):
    with pytest.raises(Exception):
        env = init_drone_swarm_search(person_amount=person_amount)
        opt = {"person_pod_multipliers": mult}
        _ = env.reset(options=opt)


@pytest.mark.parametrize(
    "person_amount, mult",
    [
        (1, [-1.2]),
        (2, [1, -0.8]),
        (3, [1, -0.8, 1.7]),
        (4, [1, 0.8, -0.7, 0.6]),
        (5, [1, 0.8, -0, 0.6, -3.5]),
    ],
)
def test_get_wrong_if_scale_mult_is_not_greater_than_0(person_amount, mult):
    with pytest.raises(Exception):
        env = init_drone_swarm_search(person_amount=person_amount)
        opt = {"person_pod_multipliers": mult}
        _ = env.reset(options=opt)


@pytest.mark.parametrize(
    "person_amount, mult",
    [
        (1, [1, 0.1]),
        (2, [1]),
        (3, [1, 0.8, 0.7, 1]),
        (4, [1, 0.8]),
        (5, [1, 0.8, 0.7, 0.6, 0.5, 0.6, 0.5]),
    ],
)
def test_get_wrong_if_number_of_mults_is_not_equal_to_person_amount(
    person_amount, mult
):
    with pytest.raises(Exception):
        env = init_drone_swarm_search(person_amount=person_amount)
        opt = {"person_pod_multipliers": mult}
        _ = env.reset(options=opt)
