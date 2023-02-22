from numpy import array
from operator import itemgetter


def calculate_distance(point_a: tuple, point_b: tuple) -> float:
    return (
        ((point_a[0] - point_b[0]) ** 2) + ((point_a[1] - point_b[1]) ** 2)
    ) ** (1 / 2)


def calculate_distances(
    point: tuple,
    possible_points: list[tuple],
    total_distance: float,
    order: list,
    final_point: tuple,
) -> tuple[float, list]:

    if len(order) != 0:
        total_distance += calculate_distance(point, order[-1])

    new_order: list = order.copy()
    new_order.append(point)

    if len(possible_points) == 0:
        total_distance += calculate_distance(point, final_point)
        new_order.append(final_point)

        return (total_distance, new_order)

    results: list = []

    for possible_point in possible_points:

        new_possible_points: list = possible_points.copy()
        new_possible_points.remove(possible_point)

        results.append(
            calculate_distances(
                possible_point,
                new_possible_points,
                total_distance,
                new_order,
                final_point,
            )
        )

    return min(results, key=lambda t: t[0])


def order_same_probabilities(i_j_probalities: list[tuple]) -> list[tuple]:

    probabilities_dict: dict = {}

    for index, probability_triple in enumerate(i_j_probalities):

        new_dict_value: tuple = (
            index,
            probability_triple[0],
            probability_triple[1],
        )
        probability_dict_value: tuple = probabilities_dict.get(
            probability_triple[2], None
        )

        if probability_dict_value:
            probability_dict_value.append(new_dict_value)
        else:
            probabilities_dict[probability_triple[2]] = [new_dict_value]

    repeated_probability_index_i_j: list = [
        value for value in probabilities_dict.values() if len(value) > 1
    ]

    initial_indexes: list = []

    for index, list_triple_index_i_j in enumerate(
        repeated_probability_index_i_j
    ):

        initial_index: int = (
            list_triple_index_i_j[0][0] - 1
            if list_triple_index_i_j[0][0] - 1 >= 0
            else None
        )

        final_index: int = (
            list_triple_index_i_j[len(list_triple_index_i_j) - 1][0] + 1
            if list_triple_index_i_j[len(list_triple_index_i_j) - 1][0] + 1
            < len(i_j_probalities)
            else None
        )

        first_element: list[tuple] = [(0, 0)]
        if initial_index:
            first_element = [i_j_probalities[initial_index][:2]]

        last_element: list[tuple] = [(0, 0)]
        if final_index:
            last_element = [i_j_probalities[final_index][:2]]

        repeated_probability_index_i_j[index] = (
            first_element
            + [
                (triple_index_i_j[1], triple_index_i_j[2])
                for triple_index_i_j in list_triple_index_i_j
            ]
            + last_element
        )

        initial_indexes.append(initial_index)

    group_points_ordered: list = []

    for group_of_probabilities in repeated_probability_index_i_j:
        initial_point: tuple = group_of_probabilities[0]
        final_point: tuple = group_of_probabilities[-1]

        del group_of_probabilities[0]
        del group_of_probabilities[-1]

        group_points_ordered.append(
            calculate_distances(
                initial_point, group_of_probabilities, 0, [], final_point
            )[1]
        )

    for index, element in enumerate(group_points_ordered):
        base_index: int = initial_indexes[index] + 1

        iterator: int = 0
        while iterator < len(element[1:-1]):
            i_j_probalities[base_index + iterator] = element[iterator + 1]
            iterator += 1

    return [
        (i_j_probality[0], i_j_probality[1])
        for i_j_probality in i_j_probalities
    ]


def generate_path(matrix: array) -> list[tuple]:

    probabilities: list = [
        probability for line in matrix for probability in line
    ]
    index_probabilities: list = [
        (index, probability) for index, probability in enumerate(probabilities)
    ]
    almost_sorted_index_probabilities: list = sorted(
        index_probabilities, key=itemgetter(1), reverse=True
    )
    almost_sorted_i_j_probabilities: list = [
        (
            element[0] // matrix.shape[1],
            element[0] % matrix.shape[1],
            element[1],
        )
        for element in almost_sorted_index_probabilities
    ]

    print(almost_sorted_i_j_probabilities)
    sorted_i_j: list = order_same_probabilities(
        almost_sorted_i_j_probabilities
    )

    return sorted_i_j
