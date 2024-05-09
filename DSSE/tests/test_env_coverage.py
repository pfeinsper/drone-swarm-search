import pytest
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.constants import Actions
from pettingzoo.test import parallel_api_test
import numpy as np


def init_Coverage_drone_swarm_search(
    render_mode="ansi",
    render_grid=True,
    render_gradient=True,
    timestep_limit=100,
    drone_amount=1,
    pre_render_time=10,
    prob_matrix_path="DSSE/tests/matrix.npy",
    disaster_position=(-24.04, -46.17),
    drone_speed=10,
    drone_probability_of_detection=1,
):
    
    return CoverageDroneSwarmSearch(
        render_mode=render_mode,
        render_grid=render_grid,
        render_gradient=render_gradient,
        timestep_limit=timestep_limit,
        drone_amount=drone_amount,
        pre_render_time=pre_render_time,
        prob_matrix_path=prob_matrix_path,
        disaster_position=disaster_position,
        drone_speed=drone_speed,
        drone_probability_of_detection=drone_probability_of_detection,
    )


@pytest.mark.parametrize(
    "drone_amount",
    [
        -26,
        101.2,
        "401",
        4901,
    ],
)
def test_wrong_drone_number(drone_amount):
    with pytest.raises(ValueError):
        init_Coverage_drone_swarm_search(drone_amount=drone_amount)


@pytest.mark.parametrize(
    "drone_amount, drones_positions",
    [
        (2, [(0, 0), (2, 0),]),
    ],
)
def test_drone_collision_termination(drone_amount, drones_positions):

    env = init_Coverage_drone_swarm_search(drone_amount=drone_amount)
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
    "drone_amount, drones_positions",
    [
        (1, [(0, 0)]),
    ],
)
def test_leave_grid_get_negative_reward(drone_amount, drones_positions):
    env = init_Coverage_drone_swarm_search(drone_amount=drone_amount)
    opt = {"drones_positions": drones_positions}
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
    env = init_Coverage_drone_swarm_search(drone_amount=drone_amount)
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
    env = init_Coverage_drone_swarm_search(drone_amount=drone_amount)

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
        (2, [(1200, 0), (25, 13)]),
        (3, [(0, 0), (19, 19), (25, -10)]),
        (4, [(5, 0), (0, 0), (10, 10), (10, 10)]),
    ],
)
def test_invalid_drone_position_raises_error(drone_amount, drones_positions):
    with pytest.raises(ValueError):
        env = init_Coverage_drone_swarm_search(drone_amount=drone_amount)
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
    env = init_Coverage_drone_swarm_search(drone_amount=drone_amount)

    _ = env.reset()

    assert (
        len(env.get_agents()) == drone_amount
    ), f"Should have {drone_amount} drones, but found {len(env.get_agents())}."


@pytest.mark.parametrize(
    "drone_amount",
    [
        2,
        5,
        15,
    ],
)
def test_with_the_observation_size_is_correct_for_all_drones(drone_amount):
    env = init_Coverage_drone_swarm_search(drone_amount=drone_amount)

    observations, _ = env.reset()

    # Obtém o tamanho da primeira observação
    first_observation_shape = observations[f"drone0"][1].shape

    for drone in range(1, drone_amount):
        drone_id = f"drone{drone}"
        observation_matriz = observations[drone_id][1]
        
        # Verifica se o tamanho da observação é igual ao da primeira observação
        assert np.array_equal(observation_matriz.shape, first_observation_shape), f"Observation size mismatch for drone {drone_id}. Expected size: {first_observation_shape}, actual size: {observation_matriz.shape}"

@pytest.mark.parametrize(
    "timestep_limit",
    [
        100,
        550,
        150,
    ],
)
def test_if_the_timestep_limit_is_correct(timestep_limit):
    # Cria o ambiente com o limite de passos de tempo especificado
    env = init_Coverage_drone_swarm_search(timestep_limit=timestep_limit)
    _, _ = env.reset()

    # Contador de passos de tempo
    steps = 0

    # Loop até o ambiente atingir o limite de passos de tempo
    while env.agents:
        steps += 1
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        _, _, _, _, _ = env.step(actions)
    
    assert steps == timestep_limit, f"Number of steps does not match timestep limit. Expected: {timestep_limit}, Actual: {steps}"
    

def test_petting_zoo_interface_works():
    env = init_Coverage_drone_swarm_search()
    parallel_api_test(env)
    env.close()
