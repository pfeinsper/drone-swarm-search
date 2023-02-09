from turtle import Turtle, Screen
import time

MATRIX_LENGTH = 5
MATRIX_WIDTH = 2


def config_screen() -> Screen:
    """Set screen size based on matrix size"""
    screen = Screen()
    screen.setup(
        width=50 * MATRIX_LENGTH + 100,
        height=50 * MATRIX_WIDTH + 100,
        startx=0,
        starty=0,
    )

    screen.title("Drone Path")
    screen.bgcolor("blue")
    screen.tracer(False)
    return screen


def config_turtle_environment() -> Turtle:
    """Set turtle environment"""
    turtle = Turtle()
    turtle.speed(0)
    turtle.shape("circle")
    turtle.pensize(3)
    turtle.penup()
    turtle.goto(-50 * MATRIX_LENGTH / 2, 50 * MATRIX_WIDTH / 2)
    return turtle


screen = config_screen()

turtle = config_turtle_environment()


# Function to draw a matrix
def draw_matrix(matrix: list[list]):
    # Draw the matrix
    for row in matrix:
        for value in row:
            turtle.color("black") if value == "X" else turtle.color("white")
            turtle.write(value, font=("Arial", 20, "normal"))
            turtle.forward(50)

        turtle.backward(50 * len(row))
        turtle.right(90)
        turtle.forward(50)
        turtle.left(90)

    """ time.sleep(0.5)
    turtle.clear() """
    screen.exitonclick()


def draw_drone_path(path: list[list]) -> None:
    for matrix in path:
        draw_matrix(matrix)


matrixs = [
    [
        ["X", "", "", "", ""],
        ["", "", "D", "", ""],
    ],  # 0
    [
        ["", "X", "", "", ""],
        ["", "", "D", "", ""],
    ],  # 1
]

#  draw_drone_path(matrixs)
draw_matrix(matrixs[0])
