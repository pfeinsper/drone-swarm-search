from DSSE import Actions
import math


def calculate_action(drone_position, target_position):
    drone_x, drone_y = drone_position
    target_x, target_y = target_position

    if drone_x == target_x and drone_y == target_y:
        return Actions.SEARCH.value

    diff_x = target_x - drone_x
    diff_y = target_y - drone_y

    if diff_x < 0 and diff_y < 0:
        return Actions.UP_LEFT.value
    elif diff_x > 0 and diff_y < 0:
        return Actions.UP_RIGHT.value
    elif diff_x < 0 and diff_y > 0:
        return Actions.DOWN_LEFT.value
    elif diff_x > 0 and diff_y > 0:
        return Actions.DOWN_RIGHT.value
    elif diff_x < 0:
        return Actions.LEFT.value
    elif diff_x > 0:
        return Actions.RIGHT.value
    elif diff_y < 0:
        return Actions.UP.value
    else:
        return Actions.DOWN.value


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    diff_x = x2 - x1
    diff_y = y2 - y1

    distance = math.sqrt(diff_x**2 + diff_y**2)
    return distance


def policy(obs, agents, env):
    actions = {}
    for idx, agent in enumerate(agents):
        castaways_positions = [person.get_position() for person in env.get_persons()]

        drone_position = env.agents_positions[idx]

        nearest_castaway = min(
            castaways_positions, key=lambda pos: distance(drone_position, pos)
        )

        action = calculate_action(drone_position, nearest_castaway)

        actions[agent] = action

    return actions
