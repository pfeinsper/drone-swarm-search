from DSSE.environment.generator.dynamic_probability import ProbabilityMatrix


def create_prob_matrix(
    person_position: tuple = (10, 10),
    vector: list = [0.1, 0.1],
    grid_size: int = 40,
    disperse_constant: int = 3,
):
    prob_matrix = ProbabilityMatrix(
        40,
        disperse_constant,
        disperse_constant,
        vector,
        person_position,
        grid_size,
    )
    return prob_matrix

def test_matrix_dimensions():
    grid_size = 40
    prob_matrix = create_prob_matrix(grid_size=grid_size)
    assert prob_matrix.map.shape == (grid_size, grid_size)


