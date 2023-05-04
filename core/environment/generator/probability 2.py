from numpy import array
from random import randint, shuffle


def divide_number(size: int) -> list:
    parts = []
    number_rest = size * 10
    parts_number = size

    # Defining the divisor which will be used to calculate probabilities
    divisor = number_rest // 100 if number_rest > 100 else parts_number / 10

    for i in range(1, parts_number + 1):
        if i == parts_number:
            new_number = randint(0, 50)
            parts.append(new_number / divisor)
            break
        else:
            # max_range = number_rest - parts_number
            new_number = randint(0, 50)

        # number_rest -= new_number
        parts.append(new_number / divisor)

    shuffle(parts)
    return parts


def generate_probability_matrix(size_x: int, size_y: int) -> array:
    temporary_matrix = []
    probability_list = divide_number(size_x * size_y)
    counter = 0

    for _ in range(size_x):
        temporary_list = []

        for _ in range(size_y):
            temporary_list.append(int(probability_list[counter]))
            counter += 1

        temporary_matrix.append(temporary_list)

    final_matrix = array(temporary_matrix)
    return final_matrix
