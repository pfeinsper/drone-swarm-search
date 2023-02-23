from core.environment.generator.probability import generate_probability_matrix
from core.environment.generator.map import generate_map
from numpy import array


def check_if_filled(matrix: array) -> bool:
    values: list = [element for line in matrix for element in line]

    if ("P" in values and "X" in values) or ("PX" in values):
        return True

    return False


def test_matrix_dimensions():
    assert generate_map(generate_probability_matrix(10, 2)).shape == (10, 2)
    assert generate_map(generate_probability_matrix(10, 10)).shape == (10, 10)
    assert generate_map(generate_probability_matrix(30, 100)).shape == (30, 100)
    assert generate_map(generate_probability_matrix(35, 20)).shape == (35, 20)
    assert generate_map(generate_probability_matrix(53, 61)).shape == (53, 61)
    assert generate_map(generate_probability_matrix(13, 77)).shape == (13, 77)
    assert generate_map(generate_probability_matrix(32, 2346)).shape == (32, 2346)
    assert generate_map(generate_probability_matrix(1, 2)).shape == (1, 2)
    assert generate_map(generate_probability_matrix(2, 4)).shape == (2, 4)


def test_matrix_filled():
    assert check_if_filled(generate_map(generate_probability_matrix(44, 99))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(33, 17))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(1, 100))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(95, 29))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(10, 10))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(81, 99))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(44, 92))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(11, 2))) == True
    assert check_if_filled(generate_map(generate_probability_matrix(29, 42))) == True
