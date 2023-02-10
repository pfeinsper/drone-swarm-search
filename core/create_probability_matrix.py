import numpy as np
from random import randint, shuffle


def divide_number(number: int, parts_number: int) -> list:
    parts = []
    number_rest = number

    if number > 100:
        divisor = number // 100
    else:
        divisor = parts_number / 10

    for i in range(1, parts_number + 1):
        if i == parts_number:
            parts.append(number_rest / divisor)
            break
        else:
            new_number = randint(1, (number_rest - (parts_number - i)) // 2)

        number_rest -= new_number
        parts.append(new_number / divisor)

    shuffle(parts)
    return parts


def create_probability_matrix(size_x: int, size_y: int) -> np.array:
    temporary_matrix = []
    probability_list = divide_number(size_x * size_y * 10, size_x * size_y)
    counter = 0

    for _ in range(size_x):
        temporary_list = []

        for _ in range(size_y):
            temporary_list.append(probability_list[counter])
            counter += 1

        temporary_matrix.append(temporary_list)

    final_matrix = np.array(temporary_matrix)
    return final_matrix
